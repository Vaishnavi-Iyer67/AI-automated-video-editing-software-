"""
auto_trim_bot.py
Features:
- Whisper auto captions (karaoke char-reveal)
- No background behind captions
- Choose text color, font, size
- Caption style options (1 bottom center, 2 full-width bottom wrap, 3 floating above bottom)
- Edit captions interactively
- BGM mixing under original audio (low vol)
- Minimal overlays/logo/SFX prompts (optional)
- Pillow-based rendering (no ImageMagick)
- Friendly console UI/UX with clear prompts & defaults
"""

import os, sys, math, time
import whisper
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import (
    VideoFileClip, AudioFileClip, CompositeVideoClip, VideoClip,
    CompositeAudioClip, concatenate_audioclips, ImageClip
)

# -------------------------
# Small UI helpers
# -------------------------
def prompt(msg, default=None):
    if default is None:
        r = input(msg + " ").strip()
    else:
        r = input(f"{msg} (default: {default}) ").strip()
        if r == "":
            r = str(default)
    return r

def yesno(msg, default="no"):
    r = prompt(msg + f" (yes/no)", default)
    return r.lower() in ("y", "yes", "true", "1")

def choose_from(msg, choices, default=None):
    print(msg)
    for i, c in enumerate(choices, start=1):
        print(f"  {i}) {c}")
    sel = prompt("Choose number", default)
    try:
        n = int(sel)
        if 1 <= n <= len(choices):
            return n
    except:
        pass
    return default or 1

def safe_float(x, fallback=0.0):
    try:
        return float(x)
    except:
        return fallback

# -------------------------
# Text wrapping utility
# -------------------------
def wrap_text_to_lines(text, font, max_width, draw):
    """Return list of wrapped lines that fit in max_width using draw.textbbox"""
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

# -------------------------
# Caption rendering (PIL) — karaoke char reveal
# -------------------------
def render_karaoke_strip(text, video_w, strip_h, reveal_chars, font, style):
    """
    Render a strip image (RGB numpy array) of width video_w and height strip_h.
    text: caption text (string)
    reveal_chars: number of characters to highlight (int)
    font: PIL ImageFont
    style: 1-bottom-center,2-fullwidth-wrap,3-floating (handled by caller for y placement)
    """
    img = Image.new("RGBA", (video_w, strip_h), (0,0,0,0))
    draw = ImageDraw.Draw(img)

    padding_x = 40
    max_text_w = max(50, video_w - padding_x*2)
    lines = wrap_text_to_lines(text, font, max_text_w, draw)
    # Build line strings (keep trailing space except last) to count chars
    line_strings = []
    for i, ln in enumerate(lines):
        if i < len(lines) - 1:
            line_strings.append(ln + " ")
        else:
            line_strings.append(ln)
    joined = "".join(line_strings)
    total_chars = len(joined)
    reveal = clamp_int(reveal_chars, 0, total_chars)

    dim_color = (180,180,180,255)   # base grey
    highlight_color = (255,255,255,255)  # white

    # compute vertical layout
    line_heights = []
    for ln in line_strings:
        bbox = draw.textbbox((0,0), ln, font=font)
        line_heights.append(bbox[3] - bbox[1])
    total_h = sum(line_heights) + max(0, (len(line_heights)-1)) * 6
    y = (strip_h - total_h)//2

    remaining = reveal
    for ln in line_strings:
        # base draw
        draw.text((padding_x, y), ln, font=font, fill=dim_color)
        ln_len = len(ln)
        to_high = min(ln_len, remaining)
        if to_high > 0:
            highlight_text = ln[:to_high]
            # draw highlight on overlay to preserve antialias
            overlay = Image.new("RGBA", (video_w, strip_h), (0,0,0,0))
            d2 = ImageDraw.Draw(overlay)
            d2.text((padding_x, y), highlight_text, font=font, fill=highlight_color)
            img = Image.alpha_composite(img, overlay)
            remaining -= to_high
        y += line_heights.pop(0) + 6

    # convert to RGB (remove alpha)
    return np.array(img.convert("RGB"))

def clamp_int(v, a, b):
    try:
        iv = int(v)
    except:
        iv = 0
    return max(a, min(b, iv))

# -------------------------
# Build karaoke caption VideoClip
# -------------------------
def build_karaoke_clip(subtitles, video_w, video_h, strip_h, font_path, text_color, style):
    # prepare PIL font; attempt to load by path/name, then fall back
    font_size = 36
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

    # Color is applied by drawing white highlight and grey base; to allow custom color,
    # we'll tint the highlight after rendering by blending. Simpler: pass text_color as tuple.
    # But for performance we will draw highlight in white then multiply. Instead, draw in chosen color.
    # We'll keep highlight color = text_color; dim color = scaled toward grey.
    # Modify render function slightly by passing color - easier to re-implement inline here.

    subs_data = []
    for (s,e,t) in subtitles:
        dur = max(0.001, e - s)
        subs_data.append((s,e,t,dur,len(t)))

    duration = max([e for (_,e,_,_,) in [(s,e,t,_) for (s,e,t) in subtitles]] ) if subtitles else 0.01

    def make_frame(t):
        # determine active subtitle
        for s,e,txt,dur,count in subs_data:
            if s <= t <= e:
                progress = (t - s) / dur
                reveal = int(progress * max(1, len(txt)))
                frame = render_karaoke_strip_custom_color(txt, video_w, strip_h, reveal, font, text_color)
                return frame
        # else, transparent strip => return black-ish empty strip (but without occluding video)
        return np.zeros((strip_h, video_w, 3), dtype=np.uint8)

    clip = VideoClip(make_frame=make_frame, duration=duration)
    # position at bottom; caller will set position
    clip = clip.set_position(("center", "bottom"))
    return clip

# We'll implement a variant to directly accept color (tuple) rather than fixed white
def render_karaoke_strip_custom_color(text, video_w, strip_h, reveal_chars, font, color_rgb):
    img = Image.new("RGBA", (video_w, strip_h), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    padding_x = 40
    max_text_w = max(50, video_w - padding_x*2)
    lines = wrap_text_to_lines(text, font, max_text_w, draw)
    line_strings = []
    for i, ln in enumerate(lines):
        if i < len(lines)-1:
            line_strings.append(ln + " ")
        else:
            line_strings.append(ln)
    joined = "".join(line_strings)
    total = len(joined)
    reveal = clamp_int(reveal_chars, 0, total)
    # compute dim color as darker variant of chosen color
    highlight_color = (color_rgb[0], color_rgb[1], color_rgb[2], 255)
    dim_color = (max(0, int(color_rgb[0]*0.6)), max(0, int(color_rgb[1]*0.6)), max(0, int(color_rgb[2]*0.6)), 255)

    # layout heights
    heights = []
    for ln in line_strings:
        bbox = draw.textbbox((0,0), ln, font=font)
        heights.append(bbox[3]-bbox[1])
    total_h = sum(heights) + max(0, (len(heights)-1))*6
    y = (strip_h - total_h)//2
    remaining = reveal
    for ln in line_strings:
        draw.text((padding_x, y), ln, font=font, fill=dim_color)
        ln_len = len(ln)
        to_high = min(ln_len, remaining)
        if to_high > 0:
            subtxt = ln[:to_high]
            overlay = Image.new("RGBA", (video_w, strip_h), (0,0,0,0))
            d2 = ImageDraw.Draw(overlay)
            d2.text((padding_x, y), subtxt, font=font, fill=highlight_color)
            img = Image.alpha_composite(img, overlay)
            remaining -= to_high
        y += heights.pop(0) + 6
    return np.array(img.convert("RGB"))

# -------------------------
# Audio mixing helper
# -------------------------
def mix_bgm_under_original(video_clip, bgm_path, bgm_vol=0.18):
    if not os.path.exists(bgm_path):
        print("BGM not found:", bgm_path)
        return video_clip
    bgm = AudioFileClip(bgm_path).volumex(bgm_vol)
    if bgm.duration < video_clip.duration:
        loops = int(video_clip.duration // bgm.duration) + 1
        bgm = concatenate_audioclips([bgm] * loops)
    bgm = bgm.subclip(0, video_clip.duration)
    orig = video_clip.audio if video_clip.audio else None
    if orig:
        mixed = CompositeAudioClip([orig.volumex(1.0), bgm])
    else:
        mixed = bgm
    return video_clip.set_audio(mixed)

# -------------------------
# Small clamp util
# -------------------------
def to_rgb_tuple(s):
    s = s.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 6:
        r = int(s[0:2],16); g = int(s[2:4],16); b = int(s[4:6],16)
        return (r,g,b)
    parts = s.split(",")
    if len(parts) == 3:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    # fallback
    return (255,255,255)

# -------------------------
# Main UI flow
# -------------------------
def main():
    print("🎬 AutoTrim Bot — Karaoke captions + BGM + UI")
    print("------------------------------------------------")

    video_file = prompt("Enter video filename (e.g., video.mp4):")
    if not os.path.exists(video_file):
        print("Video not found. Exiting.")
        return

    # basic options
    auto_choice = yesno("Generate AUTO captions with Whisper? (yes/no):", "yes")
    edit_after_gen = yesno("Allow editing captions after generation? (yes/no):", "yes")
    # caption style
    style = choose_from("Choose caption style (1-bottom center, 2-full bottom wrapped, 3-floating above bottom):",
                       ["Bottom center","Full-width bottom wrap","Floating above bottom"], default=1)
    # font & color
    font_choice = prompt("Enter font file path or font name (leave blank for Arial):", "arial.ttf")
    color_input = prompt("Enter caption color hex (e.g. #FFFFFF) or r,g,b (default white):", "#FFFFFF")
    text_color = to_rgb_tuple(color_input)

    # bgm option
    want_bgm = yesno("Do you want background music mixed under original audio? (yes/no):", "no")
    bgm_path = None
    if want_bgm:
        bgm_path = prompt("Enter BGM path (e.g., bgm.mp3):")

    # minimal overlays / logo / sfx question
    want_logo = yesno("Add a logo watermark? (yes/no):", "no")
    logo_path = None
    logo_pos = ("right","bottom")
    logo_opacity = 0.8
    if want_logo:
        logo_path = prompt("Enter logo PNG path:")
        pos_choice = choose_from("Logo position:", ["top-left","top-right","bottom-left","bottom-right"], default=4)
        if pos_choice == 1: logo_pos=("left","top")
        elif pos_choice == 2: logo_pos=("right","top")
        elif pos_choice == 3: logo_pos=("left","bottom")
        else: logo_pos=("right","bottom")
        logo_opacity = safe_float(prompt("Logo opacity 0.0-1.0:", "0.8"), 0.8)

    # load video
    print("\n🔄 Loading video...")
    video = VideoFileClip(video_file)
    video_w, video_h = video.size

    # mix bgm if requested
    if want_bgm and bgm_path:
        print("🎧 Mixing BGM...")
        video = mix_bgm_under_original(video, bgm_path, bgm_vol=0.18)

    # generate captions via whisper (segments)
    subtitles = []
    if auto_choice:
        print("\n📝 Running Whisper transcription (may take time)...")
        model = whisper.load_model("small")
        res = model.transcribe(video_file)
        for seg in res.get("segments", []):
            s = float(seg["start"]); e = float(seg["end"]); txt = seg["text"].strip()
            if txt:
                subtitles.append((s, e, txt))
        print(f"Generated {len(subtitles)} segments.")

    # allow edit
    if subtitles and edit_after_gen:
        print("\nGenerated captions preview:")
        for i,(s,e,t) in enumerate(subtitles, start=1):
            print(f"{i}) {s:.2f} -> {e:.2f} : {t}")
        if yesno("Do you want to edit any caption lines? (yes/no):", "no"):
            print("Enter edits like: index|new text  — type __done__ when finished.")
            while True:
                line = prompt("edit>")
                if line.strip() == "__done__":
                    break
                if "|" not in line:
                    print("Bad format.")
                    continue
                idx_s, newtxt = line.split("|",1)
                try:
                    idx = int(idx_s.strip())-1
                    if 0 <= idx < len(subtitles):
                        s,e,_ = subtitles[idx]
                        subtitles[idx] = (s,e,newtxt.strip())
                        print("Updated.")
                    else:
                        print("Index out of range.")
                except:
                    print("Invalid index.")

    # ask final confirm
    print("\nSummary:")
    print(f"Video: {video_file} | Captions: {len(subtitles)} segments | Style: {style} | Font: {font_choice} | Color: {text_color}")
    if want_bgm and bgm_path:
        print("BGM:", bgm_path)
    if want_logo and logo_path:
        print("Logo:", logo_path, "pos", logo_pos, "opacity", logo_opacity)
    if not yesno("Proceed to render final video? (yes/no):", "yes"):
        print("Aborted.")
        return

    # build caption clip
    caption_clip = None
    strip_h = 140  # px height for caption strip
    if subtitles:
        print("🎛 Building karaoke caption clip...")
        caption_clip = build_karaoke_clip(subtitles, video_w, video_h, strip_h, font_choice, text_color, style)
        caption_clip = caption_clip.set_duration(video.duration)
        # If style 2 (full width), make strip cover full width (already does). For style 3 adjust vertical pos:
        if style == 3:
            caption_clip = caption_clip.set_position(("center", video_h - strip_h - 120))
        elif style == 1:
            caption_clip = caption_clip.set_position(("center", video_h - strip_h - 30))
        else:
            caption_clip = caption_clip.set_position(("center", "bottom"))

    # logo overlay
    overlays = []
    if want_logo and logo_path and os.path.exists(logo_path):
        logo_ic = ImageClip(logo_path).set_opacity(logo_opacity).set_duration(video.duration).set_position(logo_pos)
        overlays.append(logo_ic)

    # final compose
    layers = [video]
    layers.extend(overlays)
    if caption_clip:
        layers.append(caption_clip)
    final = CompositeVideoClip(layers).set_duration(video.duration)
    final = final.set_audio(video.audio)  # preserve mixed audio

    out_name = "final_output.mp4"
    print(f"\n💾 Rendering to {out_name} — this can take time. Progress will show...")
    final.write_videofile(out_name, codec="libx264", audio_codec="aac", fps=video.fps)

    print("✅ Done — saved as", out_name)

if __name__ == "__main__":
    main()
