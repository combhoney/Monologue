import random
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from src.config import FONTS_DIR, MIRROR_BG_VIDEO, THUMBNAIL_BLUR_RADIUS, THUMBNAIL_BRIGHTNESS

THUMB_W = 1280
THUMB_H = 720
POPPINS_EXTRABOLD_URL = "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-ExtraBold.ttf"
MONTSERRAT_BOLD_URL = "https://github.com/googlefonts/montserrat/raw/master/fonts/ttf/Montserrat-Bold.ttf"

MODERN_LIGHT_COLORS = [
    ("Snow White", (248, 250, 252)),
    ("Champagne Gold", (253, 230, 138)),
    ("Ice Sky Blue", (186, 230, 253)),
    ("Soft Mint", (167, 243, 208)),
    ("Warm Peach", (253, 186, 116)),
    ("Pastel Lavender", (221, 214, 254)),
    ("Rose Blush", (254, 205, 211)),
    ("Warm Cream", (254, 243, 199)),
]


def _load_bold_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if FONTS_DIR.exists():
        custom_fonts = sorted(
            [p for p in FONTS_DIR.iterdir() if p.suffix.lower() in (".ttf", ".otf")]
        )
        if custom_fonts:
            return ImageFont.truetype(str(custom_fonts[0]), size=size)

    candidate_paths = [
        FONTS_DIR / "Poppins-ExtraBold.ttf",
        FONTS_DIR / "Montserrat-Bold.ttf",
        Path("/usr/share/fonts/truetype/montserrat/Montserrat-ExtraBold.ttf"),
        Path("/usr/share/fonts/truetype/montserrat/Montserrat-Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    ]
    for fp in candidate_paths:
        if fp.exists():
            return ImageFont.truetype(str(fp), size=size)

    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    for url, fname in (
        (POPPINS_EXTRABOLD_URL, "Poppins-ExtraBold.ttf"),
        (MONTSERRAT_BOLD_URL, "Montserrat-Bold.ttf"),
    ):
        dl_path = FONTS_DIR / fname
        try:
            urllib.request.urlretrieve(url, dl_path)
            if dl_path.exists() and dl_path.stat().st_size > 10_000:
                return ImageFont.truetype(str(dl_path), size=size)
        except Exception as err:
            print(f"[Thumbnail] ফন্ট ডাউনলোড ব্যর্থ ({fname}): {err}", flush=True)

    return ImageFont.load_default(size=size)


def _extract_video_cover_frame(video_path: Path, seek_sec: float = 10.0) -> Image.Image:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"থাম্বনেইলের জন্য গেমপ্লে ভিডিও খোলা যায়নি: {video_path}")

    chosen_bgr = None
    for candidate_sec in (seek_sec, max(2.0, seek_sec * 1.5), 4.0, 1.0):
        cap.set(cv2.CAP_PROP_POS_MSEC, candidate_sec * 1000.0)
        ok, frame = cap.read()
        if ok and frame is not None:
            chosen_bgr = frame
            if float(np.mean(frame)) > 25.0:
                break

    cap.release()
    if chosen_bgr is None:
        raise RuntimeError(f"গেমপ্লে ভিডিও থেকে ফ্রেম বের করা যায়নি: {video_path}")

    rgb = cv2.cvtColor(chosen_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = max(THUMB_W / w, THUMB_H / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    x0 = max(0, (new_w - THUMB_W) // 2)
    y0 = max(0, (new_h - THUMB_H) // 2)
    cropped = resized[y0 : y0 + THUMB_H, x0 : x0 + THUMB_W]
    img_out = Image.fromarray(cropped)
    if MIRROR_BG_VIDEO:
        img_out = ImageOps.mirror(img_out)
    return img_out


def _prepare_dark_blurred_bg(bg: Image.Image) -> Image.Image:
    blurred = bg.filter(ImageFilter.GaussianBlur(radius=THUMBNAIL_BLUR_RADIUS))
    darkened = ImageEnhance.Brightness(blurred).enhance(THUMBNAIL_BRIGHTNESS)
    darkened = ImageEnhance.Contrast(darkened).enhance(1.12)

    arr = np.asarray(darkened, dtype=np.float32)
    yy, xx = np.mgrid[0:THUMB_H, 0:THUMB_W]
    dist = np.sqrt(((xx - THUMB_W / 2) / (THUMB_W * 0.62)) ** 2 + ((yy - THUMB_H / 2) / (THUMB_H * 0.62)) ** 2)
    vignette = np.clip(1.0 - 0.35 * dist, 0.50, 1.0)
    out = np.clip(arr * vignette[:, :, None], 0, 255).astype(np.uint8)
    return Image.fromarray(out).convert("RGBA")


def _format_balanced_slogan(slogan: str) -> str:
    words = [w.strip() for w in slogan.upper().split() if w.strip()]
    if not words:
        return "THE QUIET\nTRUTH"
    if len(words) <= 2:
        return " ".join(words)

    best_idx = 1
    best_diff = 10**9
    for i in range(1, len(words)):
        l1 = " ".join(words[:i])
        l2 = " ".join(words[i:])
        diff = abs(len(l1) - len(l2))
        if diff < best_diff:
            best_diff = diff
            best_idx = i
    return " ".join(words[:best_idx]) + "\n" + " ".join(words[best_idx:])


def create_modern_thumbnail(
    video_path: Path,
    slogan: str,
    output_jpg: Path,
    seek_sec: float = 10.0,
) -> Path:
    output_jpg.parent.mkdir(parents=True, exist_ok=True)

    bg_rgb = _extract_video_cover_frame(video_path, seek_sec=seek_sec)
    base = _prepare_dark_blurred_bg(bg_rgb)
    text = _format_balanced_slogan(slogan)

    color_name, text_rgb = random.choice(MODERN_LIGHT_COLORS)

    probe_draw = ImageDraw.Draw(Image.new("L", (THUMB_W, THUMB_H), 0))
    chosen_font = _load_bold_font(120)

    for font_size in range(138, 48, -4):
        font = _load_bold_font(font_size)
        bbox = probe_draw.multiline_textbbox(
            (THUMB_W // 2, THUMB_H // 2),
            text,
            font=font,
            anchor="mm",
            align="center",
            spacing=22,
        )
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        if tw <= THUMB_W * 0.82 and th <= THUMB_H * 0.64:
            chosen_font = font
            break

    shadow = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).multiline_text(
        (THUMB_W // 2 + 5, THUMB_H // 2 + 9),
        text,
        font=chosen_font,
        fill=(0, 0, 0, 225),
        anchor="mm",
        align="center",
        spacing=22,
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(14))
    composed = Image.alpha_composite(base, shadow)

    ImageDraw.Draw(composed).multiline_text(
        (THUMB_W // 2, THUMB_H // 2),
        text,
        font=chosen_font,
        fill=(*text_rgb, 255),
        stroke_width=5,
        stroke_fill=(12, 12, 16, 230),
        anchor="mm",
        align="center",
        spacing=22,
    )

    final_rgb = composed.convert("RGB")
    final_rgb.save(str(output_jpg), format="JPEG", quality=92, optimize=True)

    if output_jpg.stat().st_size > 1_900_000:
        final_rgb.save(str(output_jpg), format="JPEG", quality=82, optimize=True)

    size_kb = output_jpg.stat().st_size / 1024
    print(
        f"[Thumbnail] মডার্ন ডার্ক-ব্লার থাম্বনেইল তৈরি হয়েছে: '{slogan}' "
        f"| কালার: {color_name} -> {output_jpg.name} ({size_kb:.1f} KB)",
        flush=True,
    )
    return output_jpg
