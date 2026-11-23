import bisect
import random
import subprocess
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
from PIL import Image, ImageSequence

from src.config import (
    BRIGHTNESS_THRESHOLD,
    FFMPEG_PRESET,
    GIF_DIR,
    SPECTRUM_AREA_RATIO,
    SPECTRUM_NUM_BARS,
    SPECTRUM_STYLE,
    VIDEO_FPS,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
)


def pick_random_gif() -> Path:
    gifs = [
        p
        for p in GIF_DIR.iterdir()
        if p.is_file() and p.suffix.lower() == ".gif"
    ]
    if not gifs:
        raise FileNotFoundError(f"'{GIF_DIR}' ফোল্ডারে কোনো .gif ফাইল পাওয়া যায়নি!")
    chosen = random.choice(gifs)
    print(f"[Video] রেনডম GIF সিলেক্ট হয়েছে: {chosen.name}")
    return chosen


def _cover_resize(rgb_frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = rgb_frame.shape[:2]
    scale = max(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(rgb_frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    x0 = max(0, (new_w - target_w) // 2)
    y0 = max(0, (new_h - target_h) // 2)
    return resized[y0 : y0 + target_h, x0 : x0 + target_w].copy()


def _load_gif_frames(gif_path: Path, target_w: int, target_h: int) -> tuple[list[np.ndarray], list[float], float]:
    img = Image.open(gif_path)
    frames: list[np.ndarray] = []
    cum_times: list[float] = []
    total_time = 0.0

    # অত্যাধিক ফ্রেম হলে RAM বাঁচাতে ডাউনস্কেলড ইন্টারমিডিয়েট বা ডিরেক্ট রিসাইজ
    n_est = getattr(img, "n_frames", 1)
    pre_resize = n_est <= 300

    for frame in ImageSequence.Iterator(img):
        dur_ms = frame.info.get("duration", 100)
        if dur_ms <= 0:
            dur_ms = 100
        dur_sec = max(dur_ms, 20) / 1000.0

        rgba = frame.convert("RGBA")
        black_bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
        rgb = Image.alpha_composite(black_bg, rgba).convert("RGB")
        arr = np.asarray(rgb, dtype=np.uint8)

        if pre_resize:
            arr = _cover_resize(arr, target_w, target_h)

        frames.append(arr)
        total_time += dur_sec
        cum_times.append(total_time)

    if not frames:
        raise RuntimeError(f"GIF ফাইলটি পড়া যায়নি: {gif_path}")

    return frames, cum_times, total_time


def _detect_is_dark_background(frames: list[np.ndarray], target_w: int, target_h: int) -> tuple[bool, float]:
    step = max(1, len(frames) // 20)
    sampled = frames[::step]
    luminances = []

    for fr in sampled:
        if fr.shape[1] != target_w or fr.shape[0] != target_h:
            fr = _cover_resize(fr, target_w, target_h)
        y0 = int(target_h * 0.10)
        y1 = int(target_h * 0.90)
        x0 = int(target_w * 0.10)
        x1 = int(target_w * 0.90)
        roi = fr[y0:y1, x0:x1].astype(np.float32)
        # Perceived luminance formula (RGB)
        lum = 0.299 * roi[:, :, 0] + 0.587 * roi[:, :, 1] + 0.114 * roi[:, :, 2]
        luminances.append(float(np.mean(lum)))

    avg_lum = float(np.mean(luminances))
    is_dark = avg_lum < BRIGHTNESS_THRESHOLD
    print(
        f"[Video] ব্যাকগ্রাউন্ড গড় উজ্জ্বলতা: {avg_lum:.1f}/255 -> "
        f"{'ডার্ক (সাদা স্পেকট্রাম)' if is_dark else 'লাইট (কালো স্পেকট্রাম)'}"
    )
    return is_dark, avg_lum


def _compute_audio_spectrum(wav_path: Path, fps: int, num_bars: int) -> tuple[np.ndarray, float]:
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    duration = len(audio) / sr
    n_frames = max(1, int(np.ceil(duration * fps)))

    win_size = 2048
    hann = np.hanning(win_size).astype(np.float32)
    freqs = np.fft.rfftfreq(win_size, 1.0 / sr)

    # মাঝখানে ভয়েস ব্যালেন্সড দেখাতে অর্ধেক বিন থেকে মিরর-সিমেট্রিক বা ফুল-স্প্যান তৈরি
    half_bars = num_bars // 2
    band_edges = np.geomspace(85.0, min(4500.0, sr * 0.45), half_bars + 1)
    band_masks = []
    band_tilts = []

    for b in range(half_bars):
        mask = (freqs >= band_edges[b]) & (freqs < band_edges[b + 1])
        if not np.any(mask):
            idx = int(np.argmin(np.abs(freqs - 0.5 * (band_edges[b] + band_edges[b + 1]))))
            mask = np.zeros_like(freqs, dtype=bool)
            mask[idx] = True
        band_masks.append(mask)
        center_f = 0.5 * (band_edges[b] + band_edges[b + 1])
        band_tilts.append((center_f / 220.0) ** 0.35)

    raw_half = np.zeros((n_frames, half_bars), dtype=np.float32)

    for i in range(n_frames):
        center = int((i / fps) * sr)
        start = center - win_size // 2
        end = start + win_size

        if start < 0 or end > len(audio):
            seg = np.zeros(win_size, dtype=np.float32)
            s0 = max(0, start)
            e0 = min(len(audio), end)
            if e0 > s0:
                seg[(s0 - start) : (e0 - start)] = audio[s0:e0]
        else:
            seg = audio[start:end]

        mag = np.abs(np.fft.rfft(seg * hann))
        for b in range(half_bars):
            raw_half[i, b] = float(np.mean(mag[band_masks[b]])) * band_tilts[b]

    # স্প্যাশিয়াল স্মুদিং যাতে পাশাপাশি বারগুলো ঢেউয়ের মতো সুন্দর নড়ে
    spatial_kernel = np.array([0.15, 0.25, 0.60, 0.25, 0.15], dtype=np.float32)
    spatial_kernel /= spatial_kernel.sum()
    for i in range(n_frames):
        padded = np.pad(raw_half[i], 2, mode="edge")
        raw_half[i] = np.convolve(padded, spatial_kernel, mode="valid")

    # লগ-কম্প্রেশন + ৯৭তম পার্সেন্টাইল নরমালাইজেশন
    ref = max(float(np.percentile(raw_half, 97)), 1e-6)
    norm_half = np.log1p((raw_half / ref) * 10.0) / np.log1p(10.0)
    norm_half = np.clip(norm_half, 0.0, 1.0)

    # টেম্পোরাল Attack/Decay স্মুদিং (ভয়েসে দ্রুত উঠবে, ধীরে নামবে)
    smoothed_half = np.zeros_like(norm_half)
    prev = np.zeros(half_bars, dtype=np.float32)
    attack = 0.58
    decay = 0.24
    for i in range(n_frames):
        curr = norm_half[i]
        alpha = np.where(curr > prev, attack, decay)
        prev = prev + alpha * (curr - prev)
        smoothed_half[i] = prev

    # সেন্টার-আউট মিররড অ্যারেন্জমেন্ট (মাঝখানে হাই-এনার্জি বেস/ফান্ডামেন্টাল, দুই পাশে মিড/হাই)
    left_side = smoothed_half[:, ::-1]
    right_side = smoothed_half
    full_spec = np.concatenate([left_side, right_side], axis=1)

    if full_spec.shape[1] < num_bars:
        pad_col = full_spec[:, -1:]
        full_spec = np.concatenate([full_spec, pad_col], axis=1)

    return full_spec, duration


def _draw_bars_spectrum(
    canvas: np.ndarray,
    amps: np.ndarray,
    fg_color: tuple[int, int, int],
    shadow_color: tuple[int, int, int],
) -> None:
    h, w = canvas.shape[:2]
    spec_w = int(w * SPECTRUM_AREA_RATIO)
    spec_h = int(h * SPECTRUM_AREA_RATIO)
    start_x = (w - spec_w) // 2
    cy = h // 2

    n = len(amps)
    slot_w = spec_w / n
    bar_thick = max(4, int(slot_w * 0.52))

    # ROI-তে হালকা কনট্রাস্ট শ্যাডো যাতে উজ্জ্বল স্পট থাকলেও সাদা/কালো বার স্পষ্ট থাকে
    pad = bar_thick + 10
    rx0 = max(0, start_x - pad)
    rx1 = min(w, start_x + spec_w + pad)
    ry0 = max(0, cy - spec_h // 2 - pad)
    ry1 = min(h, cy + spec_h // 2 + pad)

    roi = canvas[ry0:ry1, rx0:rx1]
    shadow_roi = roi.copy()

    xs = [int(start_x + (i + 0.5) * slot_w) - rx0 for i in range(n)]
    cy_roi = cy - ry0
    half_heights = [max(3, int(float(a) * (spec_h * 0.48))) for a in amps]

    for x, hh in zip(xs, half_heights):
        cv2.line(
            shadow_roi,
            (x, cy_roi - hh),
            (x, cy_roi + hh),
            shadow_color,
            bar_thick + 6,
            lineType=cv2.LINE_AA,
        )

    cv2.addWeighted(shadow_roi, 0.25, roi, 0.75, 0, dst=roi)

    # মাঝের সরু বেসলাইন ও মূল সাদা/কালো পিল-শেপড বার
    cv2.line(roi, (start_x - rx0, cy_roi), (start_x + spec_w - rx0, cy_roi), fg_color, 2, lineType=cv2.LINE_AA)
    for x, hh in zip(xs, half_heights):
        cv2.line(
            roi,
            (x, cy_roi - hh),
            (x, cy_roi + hh),
            fg_color,
            bar_thick,
            lineType=cv2.LINE_AA,
        )


def _draw_circular_spectrum(
    canvas: np.ndarray,
    amps: np.ndarray,
    fg_color: tuple[int, int, int],
    shadow_color: tuple[int, int, int],
) -> None:
    h, w = canvas.shape[:2]
    cx, cy = w // 2, h // 2
    max_diam = int(min(w, h) * SPECTRUM_AREA_RATIO)
    base_r = int(max_diam * 0.23)
    max_ext = int(max_diam * 0.27)

    r_box = base_r + max_ext + 20
    rx0, rx1 = max(0, cx - r_box), min(w, cx + r_box)
    ry0, ry1 = max(0, cy - r_box), min(h, cy + r_box)
    roi = canvas[ry0:ry1, rx0:rx1]
    shadow_roi = roi.copy()

    cx_r, cy_r = cx - rx0, cy - ry0
    n = len(amps)
    angles = np.linspace(-np.pi / 2, 1.5 * np.pi, n, endpoint=False)
    thick = max(4, int((2 * np.pi * base_r / n) * 0.50))

    pts = []
    for ang, a in zip(angles, amps):
        r_in = max(8, int(base_r - float(a) * max_ext * 0.22))
        r_out = int(base_r + float(a) * max_ext)
        p1 = (int(cx_r + r_in * np.cos(ang)), int(cy_r + r_in * np.sin(ang)))
        p2 = (int(cx_r + r_out * np.cos(ang)), int(cy_r + r_out * np.sin(ang)))
        pts.append((p1, p2))
        cv2.line(shadow_roi, p1, p2, shadow_color, thick + 6, lineType=cv2.LINE_AA)

    cv2.circle(shadow_roi, (cx_r, cy_r), base_r, shadow_color, 8, lineType=cv2.LINE_AA)
    cv2.addWeighted(shadow_roi, 0.25, roi, 0.75, 0, dst=roi)

    cv2.circle(roi, (cx_r, cy_r), base_r, fg_color, 3, lineType=cv2.LINE_AA)
    for p1, p2 in pts:
        cv2.line(roi, p1, p2, fg_color, thick, lineType=cv2.LINE_AA)


def _draw_wave_spectrum(
    canvas: np.ndarray,
    amps: np.ndarray,
    fg_color: tuple[int, int, int],
    shadow_color: tuple[int, int, int],
) -> None:
    h, w = canvas.shape[:2]
    spec_w = int(w * SPECTRUM_AREA_RATIO)
    spec_h = int(h * SPECTRUM_AREA_RATIO)
    start_x = (w - spec_w) // 2
    cy = h // 2

    xs = np.linspace(start_x, start_x + spec_w, len(amps)).astype(np.int32)
    half_hs = np.maximum(3, (amps * (spec_h * 0.48)).astype(np.int32))
    top_pts = np.column_stack([xs, cy - half_hs])
    bot_pts = np.column_stack([xs[::-1], cy + half_hs[::-1]])
    poly = np.vstack([top_pts, bot_pts])

    overlay = canvas.copy()
    cv2.fillPoly(overlay, [poly], fg_color, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, dst=canvas)
    cv2.polylines(canvas, [top_pts], False, shadow_color, 2, lineType=cv2.LINE_AA)
    cv2.polylines(canvas, [bot_pts], False, shadow_color, 2, lineType=cv2.LINE_AA)


def create_spectrum_video(audio_wav: Path, output_mp4: Path) -> Path:
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    gif_path = pick_random_gif()

    w, h, fps = VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS
    gif_frames, cum_times, gif_total_dur = _load_gif_frames(gif_path, w, h)
    is_dark, _ = _detect_is_dark_background(gif_frames, w, h)

    fg_color = (255, 255, 255) if is_dark else (0, 0, 0)
    shadow_color = (0, 0, 0) if is_dark else (255, 255, 255)

    spec_matrix, audio_dur = _compute_audio_spectrum(audio_wav, fps, SPECTRUM_NUM_BARS)
    n_frames = spec_matrix.shape[0]
    print(
        f"[Video] রেন্ডারিং শুরু: {w}x{h} @ {fps}fps | মোট ফ্রেম: {n_frames} "
        f"| দৈর্ঘ্য: {audio_dur:.2f}s | স্টাইল: {SPECTRUM_STYLE}"
    )

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-i",
        str(audio_wav),
        "-c:v",
        "libx264",
        "-preset",
        FFMPEG_PRESET,
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output_mp4),
    ]

    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    style = SPECTRUM_STYLE.lower()
    drawer = _draw_bars_spectrum
    if style == "circular":
        drawer = _draw_circular_spectrum
    elif style == "wave":
        drawer = _draw_wave_spectrum

    try:
        assert proc.stdin is not None
        for i in range(n_frames):
            t = i / fps
            gif_t = t % gif_total_dur
            g_idx = bisect.bisect_right(cum_times, gif_t)
            if g_idx >= len(gif_frames):
                g_idx = 0

            bg = gif_frames[g_idx]
            if bg.shape[1] != w or bg.shape[0] != h:
                bg = _cover_resize(bg, w, h)
            else:
                bg = bg.copy()

            drawer(bg, spec_matrix[i], fg_color, shadow_color)
            proc.stdin.write(bg.tobytes())

        proc.stdin.close()
        stderr_out = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"FFmpeg ভিডিও এনকোডিং ব্যর্থ (code={ret}):\n{stderr_out[-1500:]}")
    except BrokenPipeError:
        stderr_out = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
        proc.wait()
        raise RuntimeError(f"FFmpeg পাইপ ভেঙে গেছে:\n{stderr_out[-1500:]}")

    if not output_mp4.exists() or output_mp4.stat().st_size == 0:
        raise RuntimeError("আউটপুট ভিডিও ফাইল তৈরি হয়নি।")

    size_mb = output_mp4.stat().st_size / (1024 * 1024)
    print(f"[Video] ভিডিও তৈরি সম্পন্ন: {output_mp4.name} ({size_mb:.2f} MB)")
    return gif_path