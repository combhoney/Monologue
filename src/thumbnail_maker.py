import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from src.config import FONTS_DIR

THUMB_W = 1280
THUMB_H = 720
MONTSERRAT_BOLD_URL = "https://github.com/googlefonts/montserrat/raw/master/fonts/ttf/Montserrat-Bold.ttf"


def _load_bold_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if FONTS_DIR.exists():
        custom_fonts = sorted(
            [p for p in FONTS_DIR.iterdir() if p.suffix.lower() in (".ttf", ".otf")]
        )
        if custom_fonts:
            return ImageFont.truetype(str(custom_fonts[0]), size=size)

    candidate_paths = [
        FONTS_DIR / "Montserrat-Bold.ttf",
        Path("/usr/share/fonts/truetype/montserrat/Montserrat-Bold.ttf"),
        Path("/usr/share/fonts/truetype/montserrat/Montserrat-ExtraBold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    ]
    for fp in candidate_paths:
        if fp.exists():
            return ImageFont.truetype(str(fp), size=size)

    try:
        FONTS_DIR.mkdir(parents=True, exist_ok=True)
        dl_path = FONTS_DIR / "Montserrat-Bold.ttf"
        urllib.request.urlretrieve(MONTSERRAT_BOLD_URL, dl_path)
        if dl_path.exists() and dl_path.stat().st_size > 10_000:
            return ImageFont.truetype(str(dl_path), size=size)
    except Exception as err:
        print(f"[Thumbnail] ফন্ট ডাউনলোড করা যায়নি, ডিফল্ট ফন্ট ব্যবহার হবে: {err}")

    return ImageFont.load_default(size=size)


def _extract_video_cover_frame(video_path: Path, seek_sec: float = 10.0) -> Image.Image:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"থাম্বনেইলের জন্য গেমপ্লে ভিডিও খোলা যায়নি: {video_path}")

    chosen_bgr = None
    for candidate_sec in (seek_sec, max(2.0, seek_sec * 1.5), 2.0, 0.5):
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
    return Image.fromarray(cropped)


def _apply_cinematic_vignette(bg: Image.Image) -> Image.Image:
    yy, xx = np.mgrid[0:THUMB_H, 0:THUMB_W]
    dist = np.sqrt(((xx - THUMB_W / 2) / (THUMB_W * 0.58)) ** 2 + ((yy - THUMB_H / 2) / (THUMB_H * 0.58)) ** 2)
    vignette = np.clip(0.68 - 0.34 * dist, 0.22, 0.72)

    arr = np.asarray(bg, dtype=np.float32)
    avg_lum = float(np.mean(0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]))
    if avg_lum > 140:
        vignette *= 0.65

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


def _build_metallic_gold_texture(y0: float, y1: float) -> Image.Image:
    box_h = max(1.0, y1 - y0)
    stops = [
        (0.00, (255, 249, 204)),
        (0.28, (255, 215, 0)),
        (0.56, (212, 175, 55)),
        (0.82, (142, 95, 16)),
        (1.00, (246, 208, 92)),
    ]

    grad_col = np.zeros((THUMB_H, 3), dtype=np.uint8)
    for y in range(THUMB_H):
        p = float(np.clip((y - y0) / box_h, 0.0, 1.0))
        for (p0, c0), (p1, c1) in zip(stops[:-1], stops[1:]):
            if p0 <= p <= p1:
                t = (p - p0) / max(1e-6, p1 - p0)
                grad_col[y] = [int(c0[k] + t * (c1[k] - c0[k])) for k in range(3)]
                break

    full = np.broadcast_to(grad_col[:, None, :], (THUMB_H, THUMB_W, 3)).copy()
    return Image.fromarray(full).convert("RGBA")


def create_gold_thumbnail(
    video_path: Path,
    slogan: str,
    output_jpg: Path,
    seek_sec: float = 10.0,
) -> Path:
    output_jpg.parent.mkdir(parents=True, exist_ok=True)

    bg_rgb = _extract_video_cover_frame(video_path, seek_sec=seek_sec)
    base = _apply_cinematic_vignette(bg_rgb)
    text = _format_balanced_slogan(slogan)

    probe_draw = ImageDraw.Draw(Image.new("L", (THUMB_W, THUMB_H), 0))
    chosen_font = _load_bold_font(120)
    bbox = (0, 0, THUMB_W, THUMB_H)

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
        if tw <= THUMB_W * 0.82 and th <= THUMB_H * 0.62:
            chosen_font = font
            break

    shadow = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.multiline_text(
        (THUMB_W // 2 + 6, THUMB_H // 2 + 10),
        text,
        font=chosen_font,
        fill=(0, 0, 0, 240),
        anchor="mm",
        align="center",
        spacing=22,
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(14))

    outline = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
    od = ImageDraw.Draw(outline)
    od.multiline_text(
        (THUMB_W // 2, THUMB_H // 2),
        text,
        font=chosen_font,
        fill=(18, 12, 4, 255),
        stroke_width=7,
        stroke_fill=(18, 12, 4, 255),
        anchor="mm",
        align="center",
        spacing=22,
    )

    mask = Image.new("L", (THUMB_W, THUMB_H), 0)
    md = ImageDraw.Draw(mask)
    md.multiline_text(
        (THUMB_W // 2, THUMB_H // 2),
        text,
        font=chosen_font,
        fill=255,
        anchor="mm",
        align="center",
        spacing=22,
    )

    gold_tex = _build_metallic_gold_texture(bbox[1], bbox[3])
    gold_text = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
    gold_text.paste(gold_tex, (0, 0), mask)

    composed = Image.alpha_composite(base, shadow)
    composed = Image.alpha_composite(composed, outline)
    composed = Image.alpha_composite(composed, gold_text)

    final_rgb = composed.convert("RGB")
    final_rgb.save(str(output_jpg), format="JPEG", quality=92, optimize=True)

    if output_jpg.stat().st_size > 1_900_000:
        final_rgb.save(str(output_jpg), format="JPEG", quality=82, optimize=True)

    size_kb = output_jpg.stat().st_size / 1024
    print(f"[Thumbnail] গেমপ্লে থাম্বনেইল তৈরি হয়েছে: '{slogan}' -> {output_jpg.name} ({size_kb:.1f} KB)")
    return output_jpg