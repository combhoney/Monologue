import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from src.config import FONTS_DIR

THUMB_W = 1280
THUMB_H = 720
ANTON_URL = "https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf"
MONTSERRAT_BOLD_URL = "https://github.com/googlefonts/montserrat/raw/master/fonts/ttf/Montserrat-Bold.ttf"


def _download_font_if_needed(url: str, filename: str) -> Path | None:
    dest = FONTS_DIR / filename
    if dest.exists() and dest.stat().st_size > 10_000:
        return dest
    try:
        FONTS_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
        if dest.exists() and dest.stat().st_size > 10_000:
            return dest
    except Exception as err:
        print(f"[Thumbnail] ফন্ট ডাউনলোড ব্যর্থ ({filename}): {err}", flush=True)
    return None


def _load_hero_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # ব্যবহারকারী নিজে fonts/ ফোল্ডারে কাস্টম ফন্ট দিলে সেটিকে অগ্রাধিকার দেওয়া
    if FONTS_DIR.exists():
        custom = sorted(
            [
                p
                for p in FONTS_DIR.iterdir()
                if p.suffix.lower() in (".ttf", ".otf") and p.name not in ("Montserrat-Bold.ttf",)
            ]
        )
        if custom:
            return ImageFont.truetype(str(custom[0]), size=size)

    anton_path = _download_font_if_needed(ANTON_URL, "Anton-Regular.ttf")
    if anton_path:
        return ImageFont.truetype(str(anton_path), size=size)

    return _load_badge_font(size)


def _load_badge_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        FONTS_DIR / "Montserrat-Bold.ttf",
        Path("/usr/share/fonts/truetype/montserrat/Montserrat-Bold.ttf"),
        Path("/usr/share/fonts/truetype/montserrat/Montserrat-ExtraBold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    ]
    for fp in candidates:
        if fp.exists():
            return ImageFont.truetype(str(fp), size=size)

    dl = _download_font_if_needed(MONTSERRAT_BOLD_URL, "Montserrat-Bold.ttf")
    if dl:
        return ImageFont.truetype(str(dl), size=size)

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
            if float(np.mean(frame)) > 28.0:
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


def _apply_cinematic_grade_and_glow(bg: Image.Image) -> Image.Image:
    # কিনেম্যাটিক কালার গ্রেডিং: ব্যাকগ্রাউন্ডের বাড়তি রঙ কমিয়ে কনট্রাস্ট ও শার্পনেস বাড়ানো
    bg = ImageEnhance.Color(bg).enhance(0.72)
    bg = ImageEnhance.Contrast(bg).enhance(1.24)
    bg = ImageEnhance.Sharpness(bg).enhance(1.28)

    yy, xx = np.mgrid[0:THUMB_H, 0:THUMB_W]
    dist = np.sqrt(((xx - THUMB_W / 2) / (THUMB_W * 0.50)) ** 2 + ((yy - THUMB_H / 2) / (THUMB_H * 0.50)) ** 2)
    vignette = np.clip(0.64 - 0.44 * dist, 0.12, 0.66)

    arr = np.asarray(bg, dtype=np.float32)
    avg_lum = float(np.mean(0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]))
    if avg_lum > 140:
        vignette *= 0.68

    # হালকা কুল ডার্ক শ্যাডো যাতে গোল্ডেন টেক্সট আরও বেশি উজ্জ্বল দেখায়
    graded = arr * vignette[:, :, None]
    graded[:, :, 0] *= 0.94
    graded[:, :, 2] *= 1.06
    base = Image.fromarray(np.clip(graded, 0, 255).astype(np.uint8)).convert("RGBA")

    # মাঝখানে ওয়ার্ম অ্যাম্বার-গোল্ডেন স্পটলাইট গ্লো
    glow_dist = np.sqrt(((xx - THUMB_W / 2) / (THUMB_W * 0.38)) ** 2 + ((yy - THUMB_H * 0.53) / (THUMB_H * 0.30)) ** 2)
    glow_alpha = np.clip(1.0 - glow_dist, 0.0, 1.0) ** 2.2
    glow = np.zeros((THUMB_H, THUMB_W, 4), dtype=np.uint8)
    glow[:, :, 0] = 255
    glow[:, :, 1] = 185
    glow[:, :, 2] = 40
    glow[:, :, 3] = (glow_alpha * 78).astype(np.uint8)
    base = Image.alpha_composite(base, Image.fromarray(glow))

    # বাইরের চারপাশে হালকা গোল্ডেন বোকেহ পার্টিকেল
    rng = np.random.default_rng(108)
    bokeh = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
    bd = ImageDraw.Draw(bokeh)
    for _ in range(30):
        bx = int(rng.integers(35, THUMB_W - 35))
        by = int(rng.integers(35, THUMB_H - 35))
        if abs(bx - THUMB_W // 2) < 330 and abs(by - THUMB_H // 2) < 170:
            continue
        r = int(rng.integers(3, 12))
        alpha = int(rng.integers(40, 120))
        bd.ellipse((bx - r, by - r, bx + r, by + r), fill=(255, 212, 85, alpha))
    bokeh = bokeh.filter(ImageFilter.GaussianBlur(3))
    base = Image.alpha_composite(base, bokeh)

    return base


def _split_hero_lines(slogan: str) -> tuple[str, str]:
    words = [w.strip() for w in slogan.upper().split() if w.strip()]
    if not words:
        return "SILENCE IS", "POWER"
    if len(words) == 1:
        return "", words[0]
    if len(words) == 2:
        return words[0], words[1]
    if len(words) == 3:
        return f"{words[0]} {words[1]}", words[2]
    return " ".join(words[:2]), " ".join(words[2:])


def _fit_line_font(text: str, max_w: float, max_h: float, start_size: int, min_size: int = 44):
    probe = ImageDraw.Draw(Image.new("L", (THUMB_W, THUMB_H), 0))
    for sz in range(start_size, min_size - 1, -4):
        font = _load_hero_font(sz)
        bb = probe.textbbox((THUMB_W // 2, THUMB_H // 2), text, font=font, anchor="mm")
        if (bb[2] - bb[0]) <= max_w and (bb[3] - bb[1]) <= max_h:
            return font
    return _load_hero_font(min_size)


def _build_metallic_gold_texture(y0: float, y1: float) -> Image.Image:
    box_h = max(1.0, y1 - y0)
    stops = [
        (0.00, (255, 252, 218)),  # টপ স্পেকুলার হাইলাইট
        (0.26, (255, 220, 40)),   # উজ্জ্বল পিওর গোল্ড
        (0.54, (220, 168, 38)),   # রিচ মেটালিক গোল্ড
        (0.80, (148, 92, 14)),    # ডিপ ব্রোঞ্জ বেভেল
        (1.00, (255, 218, 96)),   # বটম শাইন
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


def _draw_badge_pill(canvas: Image.Image, badge_text: str, cy: int) -> None:
    clean_badge = "  ".join(badge_text.upper().split()) or "THE HARD TRUTH"
    font = _load_badge_font(28)
    draw = ImageDraw.Draw(canvas)
    bb = draw.textbbox((0, 0), clean_badge, font=font)
    bw, bh = bb[2] - bb[0], bb[3] - bb[1]
    pad_x, pad_y = 32, 14
    rect = [
        THUMB_W // 2 - bw // 2 - pad_x,
        cy - bh // 2 - pad_y,
        THUMB_W // 2 + bw // 2 + pad_x,
        cy + bh // 2 + pad_y,
    ]
    draw.rounded_rectangle(
        rect,
        radius=26,
        fill=(14, 11, 7, 215),
        outline=(245, 200, 68, 235),
        width=3,
    )
    draw.text((THUMB_W // 2, cy), clean_badge, font=font, fill=(255, 228, 125, 255), anchor="mm")


def _draw_diamond_divider(canvas: Image.Image, cy: int) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.line((THUMB_W // 2 - 270, cy, THUMB_W // 2 - 26, cy), fill=(245, 200, 65, 215), width=4)
    draw.line((THUMB_W // 2 + 26, cy, THUMB_W // 2 + 270, cy), fill=(245, 200, 65, 215), width=4)
    diamond = [
        (THUMB_W // 2, cy - 11),
        (THUMB_W // 2 + 11, cy),
        (THUMB_W // 2, cy + 11),
        (THUMB_W // 2 - 11, cy),
    ]
    draw.polygon(diamond, fill=(255, 228, 120, 255))


def create_gold_thumbnail(
    video_path: Path,
    slogan: str,
    output_jpg: Path,
    seek_sec: float = 10.0,
    badge: str = "THE HARD TRUTH",
) -> Path:
    output_jpg.parent.mkdir(parents=True, exist_ok=True)

    bg_rgb = _extract_video_cover_frame(video_path, seek_sec=seek_sec)
    canvas = _apply_cinematic_grade_and_glow(bg_rgb)

    top_line, hero_line = _split_hero_lines(slogan)

    if top_line:
        badge_y = 128
        top_y = 266
        hero_y = 438
        div_y = 592

        _draw_badge_pill(canvas, badge, badge_y)

        # ১ম লাইন: ক্রিস্প উজ্জ্বল সাদা + ডার্ক ড্রপ শ্যাডো ও আউটলাইন
        font_top = _fit_line_font(top_line, max_w=THUMB_W * 0.80, max_h=120, start_size=104)
        top_shadow = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
        ImageDraw.Draw(top_shadow).text(
            (THUMB_W // 2 + 4, top_y + 7), top_line, font=font_top, fill=(0, 0, 0, 230), anchor="mm"
        )
        top_shadow = top_shadow.filter(ImageFilter.GaussianBlur(10))
        canvas = Image.alpha_composite(canvas, top_shadow)

        ImageDraw.Draw(canvas).text(
            (THUMB_W // 2, top_y),
            top_line,
            font=font_top,
            fill=(248, 250, 252, 255),
            stroke_width=6,
            stroke_fill=(10, 10, 14, 255),
            anchor="mm",
        )
        font_hero = _fit_line_font(hero_line, max_w=THUMB_W * 0.86, max_h=190, start_size=168)
    else:
        badge_y = 155
        hero_y = 365
        div_y = 560
        _draw_badge_pill(canvas, badge, badge_y)
        font_hero = _fit_line_font(hero_line, max_w=THUMB_W * 0.86, max_h=240, start_size=195)

    # ২য়/হিরো লাইন: আউটার গোল্ডেন ব্লুম + 3D এক্সট্রুশন ডেপথ + মেটালিক গোল্ড গ্রেডিয়েন্ট
    bloom = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
    ImageDraw.Draw(bloom).text(
        (THUMB_W // 2, hero_y), hero_line, font=font_hero, fill=(255, 195, 40, 170), anchor="mm"
    )
    bloom = bloom.filter(ImageFilter.GaussianBlur(26))
    canvas = Image.alpha_composite(canvas, bloom)

    # 3D এক্সট্রুশন শ্যাডো স্ট্যাক
    ext = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
    ed = ImageDraw.Draw(ext)
    for d in range(14, 0, -2):
        ed.text(
            (THUMB_W // 2 + d // 2, hero_y + d),
            hero_line,
            font=font_hero,
            fill=(38, 22, 4, 238),
            stroke_width=8,
            stroke_fill=(14, 9, 2, 245),
            anchor="mm",
        )
    canvas = Image.alpha_composite(canvas, ext)

    # হিরো লাইনের এক্স্যাক্ট উচ্চতা অনুযায়ী মেটালিক গোল্ড গ্রেডিয়েন্ট
    probe = ImageDraw.Draw(Image.new("L", (THUMB_W, THUMB_H), 0))
    hb = probe.textbbox((THUMB_W // 2, hero_y), hero_line, font=font_hero, anchor="mm")
    gold_tex = _build_metallic_gold_texture(hb[1], hb[3])

    mask = Image.new("L", (THUMB_W, THUMB_H), 0)
    ImageDraw.Draw(mask).text((THUMB_W // 2, hero_y), hero_line, font=font_hero, fill=255, anchor="mm")
    gold_layer = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
    gold_layer.paste(gold_tex, (0, 0), mask)
    canvas = Image.alpha_composite(canvas, gold_layer)

    _draw_diamond_divider(canvas, div_y)

    final_rgb = canvas.convert("RGB")
    final_rgb.save(str(output_jpg), format="JPEG", quality=93, optimize=True)

    if output_jpg.stat().st_size > 1_900_000:
        final_rgb.save(str(output_jpg), format="JPEG", quality=84, optimize=True)

    size_kb = output_jpg.stat().st_size / 1024
    print(f"[Thumbnail] হাই-CTR গোল্ডেন থাম্বনেইল তৈরি হয়েছে: '{badge}' | '{slogan}' -> {output_jpg.name} ({size_kb:.1f} KB)", flush=True)
    return output_jpg
