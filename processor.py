# backend/processor.py
import os
import math
import numpy as np
from moviepy.editor import (
    VideoFileClip, AudioFileClip, CompositeVideoClip, VideoClip,
    concatenate_audioclips, CompositeAudioClip, ImageClip
)
from PIL import Image, ImageDraw, ImageFont
import whisper

# ---------- Helper functions (same style as earlier) ----------
def safe_truetype(font_path_or_name, size):
    try:
        return ImageFont.truetype(font_path_or_name, size)
    except Exception:
        try:
            return ImageFont.truetype("arial.ttf", size)
        except Exception:
            return ImageFont.load_default()

def wrap_text(text, font, max_width, draw):
    words = text.split()
    if not words:
        return [""]
    lines = []
    cur = words[0]
    for w in words[1:]:
        test = cur + " " + w
        bbox = draw.textbbox((0,0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            cur = test
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines

def render_karaoke_frame(text, w, caption_h, reveal_chars, font, color_rgb):
    img = Image.new("RGBA", (w, caption_h), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    padding_x = 40
    max_text_w = max(50, w - padding_x*2)
    lines = wrap_text(text, font, max_text_w, draw)
    # build string pieces
    line_strings = []
    for i, ln in enumerate(lines):
        if i < len(lines)-1:
            line_strings.append(ln + " ")
        else:
            line_strings.append(ln)
    joined = "".join(line_strings)
    total_chars = len(joined)
    reveal = int(max(0, min(reveal_chars, total_chars)))
    # colors
    highlight = (color_rgb[0], color_rgb[1], color_rgb[2], 255)
    dim = (max(0, int(color_rgb[0]*0.5)), max(0, int(color_rgb[1]*0.5)), max(0, int(color_rgb[2]*0.5)), 255)
    # layout
    heights = []
    for ln in line_strings:
        bbox = draw.textbbox((0,0), ln, font=font)
        heights.append(bbox[3] - bbox[1])
    total_h = sum(heights) + max(0, (len(heights)-1))*6
    y = (caption_h - total_h)//2
    remaining = reveal
    for ln in line_strings:
        draw.text((padding_x, y), ln, font=font, fill=dim)
        ln_len = len(ln)
        to_high = min(ln_len, remaining)
        if to_high > 0:
            overlay = Image.new("RGBA", (w, caption_h), (0,0,0,0))
            d2 = ImageDraw.Draw(overlay)
            d2.text((padding_x, y), ln[:to_high], font=font, fill=highlight)
            img = Image.alpha_composite(img, overlay)
            remaining -= to_high
        y += heights.pop(0) + 6
    return np.array(img.convert("RGB"))

def generate_karaoke_clip(subtitles, video_w, video_h, caption_h, font_path, color_rgb):
    font = safe_truetype(font_path, 36)
    subs = []
    for (s,e,t) in subtitles:
        dur = max(0.001, e - s)
        subs.append((s,e,t,dur))
    duration = max([e for (_,e,_,) in subs]) if subs else 0.01
    def make_frame(t):
        for (s,e,txt,dur) in subs:
            if s <= t <= e:
                progress = (t - s) / dur
                reveal = int(progress * max(1, len(txt)))
                return render_karaoke_frame(txt, video_w, caption_h, reveal, font, color_rgb)
        return np.zeros((caption_h, video_w, 3), dtype=np.uint8)
    clip = VideoClip(make_frame=make_frame, duration=duration)
    clip = clip.set_position(("center","bottom"))
    return clip

def mix_bgm(video_clip, bgm_path, bgm_vol=0.18):
    bgm = AudioFileClip(bgm_path).volumex(bgm_vol)
    if bgm.duration < video_clip.duration:
        loops = int(video_clip.duration // bgm.duration) + 1
        bgm = concatenate_audioclips([bgm]*loops)
    bgm = bgm.subclip(0, video_clip.duration)
    orig = video_clip.audio if video_clip.audio else None
    if orig:
        mixed = CompositeAudioClip([orig.volumex(1.0), bgm])
    else:
        mixed = bgm
    return video_clip.set_audio(mixed)

# ---------- Main process function ----------
def process_video(video_path: str, options: dict, out_dir: str = "outputs", job_id: str = None) -> str:
    """
    video_path: path to input video
    options: dict with keys:
        - intro_text
        - auto_captions (bool)
        - bgm_path (str or None)
        - font_choice (str)
        - text_color (e.g. "#FFFFFF" or "255,255,255")
        - caption_style (int)
        - logo_path (str or None)
    out_dir: where to save final output
    returns: output filename (basename)
    """
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    base_name = f"final_{job_id or 'job'}.mp4"
    out_path = os.path.join(out_dir, base_name)

    # load video
    video = VideoFileClip(video_path)
    video_w, video_h = video.size

    # mix bgm if present
    if options.get("bgm_path"):
        video = mix_bgm(video, options["bgm_path"], bgm_vol=0.18)

    # generate captions
    subtitles = []
    if options.get("auto_captions", True):
        model = whisper.load_model("small")
        result = model.transcribe(video_path)
        for seg in result.get("segments", []):
            s = float(seg["start"]); e = float(seg["end"]); txt = seg["text"].strip()
            if txt:
                subtitles.append((s, e, txt))

    # parse color to rgb tuple
    col = options.get("text_color", "#FFFFFF")
    if isinstance(col, str) and col.startswith("#"):
        c = col.lstrip("#")
        color_rgb = (int(c[0:2],16), int(c[2:4],16), int(c[4:6],16))
    elif isinstance(col, str) and "," in col:
        parts = [int(x) for x in col.split(",")]
        color_rgb = tuple(parts[:3])
    else:
        color_rgb = (255,255,255)

    # build caption clip if exist
    caption_clip = None
    CAP_H = 140
    if subtitles:
        caption_clip = generate_karaoke_clip(subtitles, video_w, video_h, CAP_H, options.get("font_choice","arial.ttf"), color_rgb)
        caption_clip = caption_clip.set_duration(video.duration)

    # logo overlay
    overlays = []
    if options.get("logo_path"):
        if os.path.exists(options["logo_path"]):
            logo_ic = ImageClip(options["logo_path"]).set_opacity(0.85).set_duration(video.duration).set_position(("right","bottom"))
            overlays.append(logo_ic)

    layers = [video] + overlays
    if caption_clip:
        layers.append(caption_clip)

    final = CompositeVideoClip(layers).set_duration(video.duration)
    if video.audio:
        final = final.set_audio(video.audio)

    # write file (synchronous)
    final.write_videofile(out_path, codec="libx264", audio_codec="aac", fps=video.fps)

    return base_name
