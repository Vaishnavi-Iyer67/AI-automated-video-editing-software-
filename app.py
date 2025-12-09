# app.py — Syntheta (VN/CapCut-like editor) — Final single-file
# Workflow: Upload -> Trim/Filters/Captions/Audio (store settings) -> Export (one final render & download)
#
# Usage:
#   pip install -U streamlit moviepy pillow numpy openai-whisper
#   python -m streamlit run app.py
#
# Ensure ffmpeg is installed and on PATH.

import streamlit as st
import tempfile, os, time, shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import whisper
from moviepy.editor import (
    VideoFileClip, AudioFileClip, CompositeVideoClip, VideoClip,
    CompositeAudioClip, concatenate_audioclips, ImageClip, vfx
)
from moviepy.video.fx.all import colorx, lum_contrast


st.set_page_config(page_title="Syntheta — Video Editor", layout="wide")
st.title("🎬 Syntheta — Video Editor (Preview Top, Tools Bottom)")


def clamp_int(v, a, b):
    try:
        iv = int(v)
    except:
        return a
    return max(a, min(b, iv))

def parse_rgb(value):
    s = str(value).strip()
    if s.startswith("#"):
        s = s[1:]
    if ',' in s:
        parts = [int(x) for x in s.split(',')]
        return tuple(parts)
    if len(s) == 6:
        return (int(s[0:2],16), int(s[2:4],16), int(s[4:6],16))
    return (255,255,255)

def wrap_text_to_lines(text, font, max_width, draw):
    words = text.split()
    if not words:
        return [""]
    lines = []
    cur = words[0]
    for w in words[1:]:
        trial = cur + " " + w
        bbox = draw.textbbox((0,0), trial, font=font)
        if bbox[2] - bbox[0] <= max_width:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines

def render_caption_rgb_and_mask(text, W, H, reveal_chars, font, color_rgb, padding_x=30):
    try:
        W = max(1, int(W))
    except:
        W = 320
    try:
        H = max(1, int(H))
    except:
        H = 120

    img = Image.new("RGBA", (W, H), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    max_w = max(50, W - padding_x*2)

    try:
        lines = wrap_text_to_lines(text, font, max_w, draw)
    except Exception:
        lines = [text[:100]] if text else [""]

    heights = []
    for ln in lines:
        try:
            bbox = draw.textbbox((0,0), ln, font=font)
            h = max(1, bbox[3] - bbox[1])
        except:
            h = 20
        heights.append(h)

    total_h = sum(heights) + max(0, (len(heights)-1))*6
    y = max(0,(H - total_h)//2)

    highlight = (color_rgb[0], color_rgb[1], color_rgb[2], 255)
    dim = (max(0,int(color_rgb[0]*0.6)), max(0,int(color_rgb[1]*0.6)), max(0,int(color_rgb[2]*0.6)), 255)

    joined = "".join([ln + (" " if i < len(lines)-1 else "") for i,ln in enumerate(lines)])
    total_chars = len(joined)
    reveal = clamp_int(reveal_chars, 0, total_chars)

    remaining = reveal
    for i,ln in enumerate(lines):
        try:
            draw.text((padding_x, y), ln, font=font, fill=dim)
        except:
            ImageDraw.Draw(img).text((padding_x, y), ln, fill=dim)
        ln_len = len(ln)
        to_high = min(ln_len, remaining)
        if to_high > 0:
            overlay = Image.new("RGBA", (W, H), (0,0,0,0))
            d2 = ImageDraw.Draw(overlay)
            try:
                d2.text((padding_x, y), ln[:to_high], font=font, fill=highlight)
                img = Image.alpha_composite(img, overlay)
            except:
                pass
            remaining -= to_high
        y += heights[i] + 6

    rgba = np.array(img)
    if rgba.size == 0 or rgba.shape[0] == 0 or rgba.shape[1] == 0:
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        mask = np.zeros((H, W), dtype=np.float32)
        return rgb, mask

    rgb = rgba[:, :, :3].astype(np.uint8)
    mask = (rgba[:, :, 3] / 255.0).astype(np.float32)
    if mask.size == 0:
        mask = np.zeros((H, W), dtype=np.float32)
    if rgb.shape[0] != mask.shape[0] or rgb.shape[1] != mask.shape[1]:
        mask = np.resize(mask, (rgb.shape[0], rgb.shape[1]))
    return rgb, mask

def build_caption_clip(subs, video_w, video_h, strip_h=None, font_path='arial.ttf', color_hex="#FFFFFF", style='1 - Bottom Center', font_size=36):
    try:
        video_w = max(1, int(video_w))
    except:
        video_w = 320
    try:
        video_h = max(1, int(video_h))
    except:
        video_h = 240

    if strip_h is None:
        strip_h = max(80, int(video_h * 0.18))
    else:
        strip_h = max(80, int(strip_h))

    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

    color_rgb = parse_rgb(color_hex)

    valid_subs = []
    for item in subs:
        try:
            if isinstance(item, dict):
                s = float(item.get('start',0.0)); e = float(item.get('end', s + 0.01)); t = str(item.get('text',''))
            else:
                s,e,t = item; s=float(s); e=float(e); t=str(t)
            if t.strip() == "": continue
            if e <= s: e = s + 0.01
            valid_subs.append((s,e,t))
        except:
            continue

    if not valid_subs:
        return None

    duration = max(e for s,e,t in valid_subs)
    if duration <= 0:
        duration = 0.01

    def frame_fn(t):
        try:
            for s,e,text in valid_subs:
                if s <= t <= e:
                    dur = max(0.001, e - s)
                    progress = (t - s) / dur
                    reveal = int(progress * max(1, len(text)))
                    rgb, _ = render_caption_rgb_and_mask(text, video_w, strip_h, reveal, font, color_rgb)
                    if rgb is None or rgb.size == 0:
                        return np.zeros((strip_h, video_w, 3), dtype=np.uint8)
                    return rgb
        except:
            pass
        return np.zeros((strip_h, video_w, 3), dtype=np.uint8)

    def mask_fn(t):
        try:
            for s,e,text in valid_subs:
                if s <= t <= e:
                    dur = max(0.001, e - s)
                    progress = (t - s) / dur
                    reveal = int(progress * max(1, len(text)))
                    _, mask = render_caption_rgb_and_mask(text, video_w, strip_h, reveal, font, color_rgb)
                    if mask is None or mask.size == 0:
                        return np.zeros((strip_h, video_w), dtype=np.float32)
                    mask = mask.astype(np.float32)
                    if mask.shape[0] != strip_h or mask.shape[1] != video_w:
                        mask = np.resize(mask, (strip_h, video_w))
                    return mask
        except:
            pass
        return np.zeros((strip_h, video_w), dtype=np.float32)

    cap = VideoClip(make_frame=frame_fn, duration=duration)
    mask = VideoClip(make_frame=mask_fn, ismask=True, duration=duration)
    cap = cap.set_mask(mask)

    if style.startswith('1'):
        cap = cap.set_position(("center", "bottom"))
    elif style.startswith('2'):
        cap = cap.set_position(("center", video_h - strip_h - 20))
    else:
        cap = cap.set_position(("center", video_h - strip_h - 120))

    try:
        if cap.duration is None or cap.duration <= 0:
            cap = cap.set_duration(duration if duration > 0 else 0.01)
    except:
        cap = cap.set_duration(duration if duration > 0 else 0.01)

    return cap

def mix_bgm_with_clip(video_clip, bgm_path, bgm_vol=0.18, reduce_orig=False, reduce_factor=0.8):
    if not bgm_path or not os.path.exists(bgm_path):
        return video_clip
    bgm = AudioFileClip(bgm_path).volumex(bgm_vol)
    if bgm.duration < video_clip.duration:
        loops = int(video_clip.duration // bgm.duration) + 1
        bgm = concatenate_audioclips([bgm] * loops)
    bgm = bgm.subclip(0, video_clip.duration)
    orig = video_clip.audio if video_clip.audio else None
    if orig:
        left = orig.volumex(reduce_factor) if reduce_orig else orig
        mixed = CompositeAudioClip([left, bgm])
    else:
        mixed = bgm
    return video_clip.set_audio(mixed)

def render_font_sample(font_path, sample_text="Sample 48px", size=48, color=(255,255,255)):
    try:
        font = ImageFont.truetype(font_path, size)
    except:
        font = ImageFont.load_default()
    W, H = 520, 120
    img = Image.new("RGB", (W, H), (30,30,30))
    d = ImageDraw.Draw(img)
    bbox = d.textbbox((0,0), sample_text, font=font)
    txw = bbox[2] - bbox[0]; txh = bbox[3] - bbox[1]
    x = (W - txw)//2; y = (H - txh)//2
    d.text((x,y), sample_text, font=font, fill=color)
    out = os.path.join(tempfile.gettempdir(), f"font_sample_{int(time.time()*1000)}.png")
    img.save(out)
    return out


preview_placeholder = st.empty()
status = st.empty()


if 'temp_files' not in st.session_state: st.session_state.temp_files = []
if 'captions' not in st.session_state: st.session_state.captions = []
if 'applied_filter' not in st.session_state: st.session_state.applied_filter = {'name':'None','intensity':1.0}
if 'trim' not in st.session_state: st.session_state.trim = (0.0,0.0)
if 'caption_settings' not in st.session_state: st.session_state.caption_settings = {'font_path':'arial.ttf','color':'#FFFFFF','style':'1 - Bottom Center','size':36}


uploaded_video = st.file_uploader("Upload video (mp4/mov/mkv)", type=['mp4','mov','mkv'])
video_path = None
default_duration = 0.0
video_w, video_h = 848, 478
if uploaded_video:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_video.name)[1])
    tmp.write(uploaded_video.read()); tmp.close()
    st.session_state.temp_files.append(tmp.name)
    video_path = tmp.name
    try:
        probe = VideoFileClip(video_path)
        default_duration = probe.duration
        video_w, video_h = probe.size[0], probe.size[1]
        probe.close()
        preview_placeholder.video(video_path)
    except Exception as e:
        status.error(f"Cannot load video: {e}")

st.markdown("---")
tool = st.radio("Tools", ['Trim','Filters','Captions','Audio','Export'], horizontal=True, label_visibility="collapsed")


if tool == 'Trim':
    st.sidebar.header("Trim")
    if not video_path: st.sidebar.warning("Upload a video first")
    start = st.sidebar.number_input("Start (s)", min_value=0.0, value=float(st.session_state.trim[0] or 0.0), step=0.5, format="%.2f")
    end_default = default_duration if default_duration > 0 else (st.session_state.trim[1] or 0.0)
    end = st.sidebar.number_input("End (s)", min_value=0.0, value=float(end_default), step=0.5, format="%.2f")
    if st.sidebar.button("Store Trim"):
        st.session_state.trim = (start, min(end, default_duration))
        status.success(f"Trim stored: {st.session_state.trim}")


elif tool == 'Filters':
    st.sidebar.header("Filters")
    filter_group = st.sidebar.selectbox("Filter group", ["Basic","Cinematic","Both"])
    basic = ['None','Brightness','Contrast','Saturation','Warm','Cool','Black & White','Sharpen']
    cinematic = ['Teal & Orange','Moody Blue','Cyberpunk Purple','Film Fade','Soft Glow','Warm Brown','High Contrast Grit']
    if filter_group == "Basic": filters = basic
    elif filter_group == "Cinematic": filters = cinematic
    else: filters = basic + cinematic

    selected_filter = st.sidebar.selectbox("Filter", filters)
    intensity = st.sidebar.slider("Intensity", 0.0, 2.0, 1.0)
    if st.sidebar.button("Store Filter"):
        st.session_state.applied_filter = {'name': selected_filter, 'intensity': intensity}
        status.success(f"Filter stored: {selected_filter} (intensity {intensity})")


    if video_path:
        try:
            p = VideoFileClip(video_path)
            frame = p.get_frame(0.0 if p.duration>0 else 0.0)
            p.close()
            before = Image.fromarray(frame).convert("RGB")
            
            def pil_filter(img, name, intensity):
                arr = np.array(img).astype(np.float32)
                if name == 'None': return img
                if name == 'Brightness': arr = np.clip(arr*(1.0+0.2*intensity),0,255)
                elif name == 'Contrast': arr = np.clip((arr/255.0-0.5)*(1+0.5*intensity)+0.5,0,1)*255
                elif name == 'Saturation':
                    gray = arr.mean(axis=2, keepdims=True)
                    arr = np.clip(arr*(1+0.4*intensity) - gray*(0.4*intensity),0,255)
                elif name == 'Warm': arr[:,:,0] = np.clip(arr[:,:,0]*(1.0+0.07*intensity),0,255)
                elif name == 'Cool': arr[:,:,2] = np.clip(arr[:,:,2]*(1.0+0.07*intensity),0,255)
                elif name == 'Black & White': return img.convert('L').convert('RGB')
                elif name == 'Sharpen': return img.filter(ImageFilter.UnsharpMask(radius=2, percent=150*intensity))
                elif name == 'Teal & Orange': arr[:,:,0] = np.clip(arr[:,:,0]*0.9,0,255); arr[:,:,2] = np.clip(arr[:,:,2]*1.05,0,255)
                elif name == 'Moody Blue': arr[:,:,2] = np.clip(arr[:,:,2]*(1.1+0.2*intensity),0,255); arr[:,:,0]*=0.9
                elif name == 'Cyberpunk Purple': arr[:,:,0] = np.clip(arr[:,:,0]*0.8,0,255); arr[:,:,2] = np.clip(arr[:,:,2]*1.2,0,255)
                elif name == 'Film Fade': arr = np.clip(arr*0.95 + 8, 0, 255)
                elif name == 'Soft Glow': return img.filter(ImageFilter.GaussianBlur(radius=max(1,int(2*intensity))))
                elif name == 'Warm Brown': arr = np.clip(arr*0.9 + [15,5,-10], 0, 255)
                elif name == 'High Contrast Grit': arr = np.clip((arr/255.0-0.5)*(1+0.8*intensity)+0.5,0,1)*255
                return Image.fromarray(arr.astype(np.uint8))
            try:
                after = pil_filter(before, selected_filter, intensity)
                col1, col2 = st.columns(2)
                col1.image(before, caption="Before")
                col2.image(after, caption=f"After — {selected_filter}")
            except:
                st.image(before, caption="Preview frame")
        except:
            pass


elif tool == 'Captions':
    st.sidebar.header("Captions")
    enable_whisper = st.sidebar.checkbox("Enable Whisper auto captions", value=True)
    font_upload = st.sidebar.file_uploader("Upload TTF font (optional)", type=['ttf'])
    font_options = ['arial.ttf','DejaVuSans.ttf','Poppins-Regular.ttf','LiberationSans-Regular.ttf']
    fallback_font = st.sidebar.selectbox("Fallback font", font_options)
    caption_font_size = st.sidebar.slider("Text size", 18, 72, 36)
    caption_color = st.sidebar.color_picker("Caption color", "#FFFFFF")
    caption_style = st.sidebar.selectbox("Style", ['1 - Bottom Center','2 - Full-width bottom wrap','3 - Floating above bottom'])
    if st.sidebar.button("Run Whisper"):
        if not video_path:
            status.error("Upload a video first")
        else:
            try:
                status.info("Running Whisper (small) — may take time")
                model = whisper.load_model("small")
                res = model.transcribe(video_path)
                segs = []
                for seg in res.get('segments', []):
                    s = float(seg['start']); e = float(seg['end']); txt = seg['text'].strip()
                    if txt: segs.append((s,e,txt))
                st.session_state.captions = segs
                status.success(f"Whisper generated {len(segs)} segments")
            except Exception as e:
                status.error(f"Whisper failed: {e}")

    st.write("Captions editor")
    caps = st.session_state.get('captions', [])
    rows = []
    for s,e,t in caps:
        rows.append({"start":float(s),"end":float(e),"text":t})
    if not rows:
        rows = [{"start":0.0,"end":1.0,"text":"Example caption"}]

    try:
        edited = st.experimental_data_editor(rows, num_rows="dynamic", use_container_width=True)
        d = edited.to_dict(orient='records') if hasattr(edited, "to_dict") else edited
        parsed = []
        for r in d:
            try:
                s = float(r.get('start',0.0)); e = float(r.get('end', s+0.01)); tx = str(r.get('text',''))
                if tx.strip(): parsed.append((s,e,tx))
            except:
                continue
        st.session_state.captions = parsed
        if st.button("Save captions"):
            status.success("Captions saved")
    except:
        editor_text = "\n".join([f"{r['start']:.3f}|{r['end']:.3f}|{r['text']}" for r in rows])
        edited_text = st.text_area("Edit captions (start|end|text)", value=editor_text, height=240)
        if st.button("Save captions (textarea)"):
            parsed=[]
            for line in edited_text.splitlines():
                if not line.strip(): continue
                parts = line.split("|",2)
                if len(parts) < 3: continue
                try:
                    ss = float(parts[0]); ee = float(parts[1]); tt = parts[2].strip()
                    if tt.strip(): parsed.append((ss,ee,tt))
                except:
                    continue
            st.session_state.captions = parsed
            status.success("Captions saved")

    
    if font_upload:
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".ttf")
        tf.write(font_upload.read()); tf.close()
        st.session_state['uploaded_font_path'] = tf.name
        st.session_state.caption_settings['font_path'] = tf.name
    else:
        st.session_state.caption_settings['font_path'] = st.session_state.caption_settings.get('font_path', fallback_font)
    st.session_state.caption_settings['color'] = caption_color
    st.session_state.caption_settings['style'] = caption_style
    st.session_state.caption_settings['size'] = caption_font_size

    
    st.write("Font preview")
    colL, colR = st.columns(2)
    sample = "Sample — 48px"
    if font_upload:
        colL.image(render_font_sample(st.session_state.get('uploaded_font_path'), sample, size=48))
    try:
        colR.image(render_font_sample(fallback_font, sample, size=48), caption=fallback_font)
    except:
        colR.write(fallback_font)


elif tool == 'Audio':
    st.sidebar.header("Audio / BGM")
    uploaded_bgm = st.sidebar.file_uploader("Upload BGM (mp3/wav) — optional", type=['mp3','wav'])
    bgm_volume = st.sidebar.slider("BGM volume", 0.0, 1.0, 0.18)
    reduce_orig = st.sidebar.checkbox("Reduce original audio when mixing BGM", value=True)
    if uploaded_bgm:
        tb = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_bgm.name)[1])
        tb.write(uploaded_bgm.read()); tb.close()
        st.session_state.temp_files.append(tb.name)
        st.session_state['uploaded_bgm_tmp'] = tb.name
        st.session_state['bgm_vol'] = bgm_volume
        st.session_state['reduce_orig'] = reduce_orig
        st.audio(tb.name)
        status.success("Uploaded BGM ready (will be mixed at Export)")


elif tool == 'Export':
    st.sidebar.header("Export (final render)")
    out_filename = st.sidebar.text_input("Output filename", value="syntheta_output.mp4")
    save_to_downloads = st.sidebar.checkbox("Save to Downloads folder", value=True)
    if st.sidebar.button("Export Video (apply all settings)"):
        if not video_path:
            status.error("Upload a video before exporting.")
        else:
            status.info("Rendering final video — this may take time.")
            try:
                clip = VideoFileClip(video_path)
                tstart, tend = st.session_state.get('trim', (0.0, clip.duration))
                tstart = max(0.0, float(tstart))
                tend = min(clip.duration, float(tend) if float(tend) > 0 else clip.duration)
                if tend <= tstart: tend = clip.duration
                clip = clip.subclip(tstart, tend)

                
                af = st.session_state.get('applied_filter', {'name':'None','intensity':1.0})
                fname = af.get('name','None'); fint = af.get('intensity',1.0)
                if fname == 'Black & White':
                    clip = clip.fx(vfx.blackwhite)
                elif fname == 'Brightness':
                    clip = clip.fx(colorx, 1.0 + 0.2*fint)
                elif fname == 'Contrast':
                    clip = clip.fx(lum_contrast, int(10*fint))
                elif fname == 'Saturation':
                    clip = clip.fx(colorx, 1.0 + 0.15*fint)
                elif fname == 'Warm':
                    clip = clip.fx(colorx, 1.0 + 0.07*fint)
                elif fname == 'Cool':
                    clip = clip.fx(colorx, 1.0 - 0.05*fint)
                elif fname == 'Teal & Orange':
                    clip = clip.fx(colorx, 0.95).fx(lum_contrast, 5)
                elif fname == 'High Contrast Grit':
                    clip = clip.fx(lum_contrast, int(15*fint))
                

                
                bgm_tmp = st.session_state.get('uploaded_bgm_tmp', None)
                if bgm_tmp:
                    clip = mix_bgm_with_clip(clip, bgm_tmp, bgm_vol=st.session_state.get('bgm_vol',0.18), reduce_orig=st.session_state.get('reduce_orig', True))

                
                captions = st.session_state.get('captions', [])
                caption_clip = None
                if captions:
                    font_path = st.session_state.get('uploaded_font_path', st.session_state.caption_settings.get('font_path','arial.ttf'))
                    color_hex = st.session_state.caption_settings.get('color', '#FFFFFF')
                    style = st.session_state.caption_settings.get('style', '1 - Bottom Center')
                    size = st.session_state.caption_settings.get('size', 36)
                    parsed = []
                    for item in captions:
                        if isinstance(item, dict):
                            s = float(item.get('start',0.0)); e = float(item.get('end', s+0.01)); t = str(item.get('text',''))
                            if t.strip(): parsed.append((s,e,t))
                        else:
                            if len(item) >= 3 and str(item[2]).strip():
                                parsed.append(item)
                    if parsed:
                        caption_clip = build_caption_clip(parsed, clip.w, clip.h, None, font_path, color_hex, style, font_size=size)
                        if caption_clip: caption_clip = caption_clip.set_duration(clip.duration)

                layers = [clip]
                if caption_clip: layers.append(caption_clip)
                final = CompositeVideoClip(layers, size=clip.size).set_duration(clip.duration)

                out_path = os.path.join(tempfile.gettempdir(), out_filename)
                final.write_videofile(out_path, codec='libx264', audio_codec='aac')
                status.success("Export finished — rendering complete")
                preview_placeholder.video(out_path)

                with open(out_path, 'rb') as f:
                    st.download_button("⬇ Download final video", f.read(), file_name=out_filename, mime='video/mp4')

                if save_to_downloads:
                    try:
                        downloads = os.path.join(os.path.expanduser("~"), "Downloads")
                        os.makedirs(downloads, exist_ok=True)
                        dest = os.path.join(downloads, out_filename)
                        if os.path.exists(dest):
                            base, ext = os.path.splitext(out_filename)
                            dest = os.path.join(downloads, f"{base}_{int(time.time())}{ext}")
                        shutil.copy(out_path, dest)
                        st.info(f"Saved to Downloads: {dest}")
                    except Exception as e:
                        st.warning(f"Could not save to Downloads: {e}")

            except Exception as e:
                debug = ""
                try:
                    debug += f" clip.size={getattr(clip,'size',None)} clip.duration={getattr(clip,'duration',None)}"
                except:
                    pass
                status.error(f"Export failed: {e} | debug: {debug}")


st.sidebar.markdown("---")
st.sidebar.caption("Syntheta — Streamlit video editor. Edits are applied only at Export.")

