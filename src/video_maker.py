import json
import random
import subprocess
from pathlib import Path

import soundfile as sf

from src.config import (
    ALLOW_BG_LOOP,
    BG_DELETE_THRESHOLD_SEC,
    FFMPEG_CRF,
    FFMPEG_PRESET,
    VIDEO_FPS,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
)
from src.gdrive_uploader import download_bg_video, list_bg_videos


def _get_video_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe দিয়ে ভিডিওর ডিউরেশন মাপা যায়নি ({video_path.name}): {proc.stderr}")
    data = json.loads(proc.stdout or "{}")
    dur = float((data.get("format") or {}).get("duration", 0.0))
    if dur <= 0:
        raise RuntimeError(f"ভিডিওটির ডিউরেশন সঠিক নয়: {video_path.name}")
    return dur


def _select_and_download_gameplay_clips(audio_duration: float, bg_tmp_dir: Path) -> dict:
    remote_videos = list_bg_videos()
    if not remote_videos:
        raise RuntimeError("Google Drive-এর গেমপ্লে ফোল্ডারে কোনো ভিডিও পাওয়া যায়নি!")

    random.shuffle(remote_videos)
    bg_tmp_dir.mkdir(parents=True, exist_ok=True)

    remaining = audio_duration
    segments: list[dict] = []
    downloaded_cache: dict[str, tuple[Path, float]] = {}
    usage_totals: dict[str, float] = {}

    for idx, item in enumerate(remote_videos):
        if remaining <= 0.05:
            break
        remote_name = item["Name"]
        safe_local_name = f"bg_{idx:03d}_{Path(remote_name).name}"
        local_path = bg_tmp_dir / safe_local_name

        download_bg_video(remote_name, local_path)
        full_dur = _get_video_duration(local_path)
        if full_dur <= 0.5:
            print(f"[Video] খুব ছোট/নষ্ট ভিডিও স্কিপ করা হলো: {remote_name} ({full_dur:.2f}s)", flush=True)
            continue

        use_dur = min(full_dur, remaining)
        segments.append(
            {
                "remote_name": remote_name,
                "local_path": local_path,
                "used_duration": use_dur,
                "full_duration": full_dur,
            }
        )
        downloaded_cache[remote_name] = (local_path, full_dur)
        usage_totals[remote_name] = usage_totals.get(remote_name, 0.0) + use_dur
        remaining -= use_dur

        print(
            f"[Video] ক্লিপ যুক্ত হলো: '{remote_name}' | "
            f"মূল দৈর্ঘ্য: {full_dur:.1f}s | ব্যবহৃত: {use_dur:.1f}s | বাকি অডিও: {max(0.0, remaining):.1f}s",
            flush=True,
        )

    if remaining > 0.5:
        if not ALLOW_BG_LOOP or not segments:
            raise RuntimeError(
                f"অডিওর দৈর্ঘ্য ({audio_duration:.1f}s) অনুযায়ী পর্যাপ্ত গেমপ্লে ভিডিও নেই! "
                f"আরও {remaining:.1f}s ভিডিও Google Drive ফোল্ডারে যোগ করুন।"
            )
        print(
            f"[Video] সতর্কতা: ফোল্ডারের সব ভিডিও মিলিয়েও {remaining:.1f}s কম পড়েছে। "
            "ALLOW_BG_LOOP=true থাকায় বাছাইকৃত ক্লিপ থেকে বাকি অংশ পূরণ করা হচ্ছে...",
            flush=True,
        )
        base_segments = list(segments)
        while remaining > 0.5:
            for seg in base_segments:
                if remaining <= 0.05:
                    break
                use_dur = min(seg["full_duration"], remaining)
                segments.append(
                    {
                        "remote_name": seg["remote_name"],
                        "local_path": seg["local_path"],
                        "used_duration": use_dur,
                        "full_duration": seg["full_duration"],
                    }
                )
                usage_totals[seg["remote_name"]] += use_dur
                remaining -= use_dur

    to_delete: list[dict] = []
    to_keep: list[dict] = []

    for r_name, used_sec in usage_totals.items():
        _, full_sec = downloaded_cache[r_name]
        is_fully_used = used_sec >= (full_sec - 0.5)
        is_over_one_min = used_sec > BG_DELETE_THRESHOLD_SEC

        entry = {"remote_name": r_name, "used_sec": used_sec, "full_sec": full_sec}
        if is_fully_used or is_over_one_min:
            to_delete.append(entry)
        else:
            to_keep.append(entry)

    print(
        f"[Video] ডিলিট প্ল্যান (শুধু লাইভ আপলোডে প্রযোজ্য) -> ডিলিটযোগ্য: {[d['remote_name'] for d in to_delete]} | "
        f"অক্ষত (<=1m ব্যবহৃত): {[k['remote_name'] for k in to_keep]}",
        flush=True,
    )

    return {
        "segments": segments,
        "to_delete": to_delete,
        "to_keep": to_keep,
        "thumb_source_video": segments[0]["local_path"],
        "thumb_seek_sec": min(segments[0]["used_duration"] * 0.25, 15.0),
    }


def create_gameplay_video(audio_wav: Path, output_mp4: Path, bg_tmp_dir: Path) -> dict:
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    w, h, fps = VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS

    audio_dur = float(sf.info(str(audio_wav)).duration)
    plan = _select_and_download_gameplay_clips(audio_dur, bg_tmp_dir)
    segments = plan["segments"]

    print(
        f"[Video] FFmpeg দিয়ে স্টুডিও-মাস্টারড ভিডিও তৈরি হচ্ছে: {w}x{h} @ {fps}fps "
        f"(preset={FFMPEG_PRESET}, crf={FFMPEG_CRF}) | অডিও দৈর্ঘ্য: {audio_dur:.2f}s | ক্লিপ: {len(segments)}টি",
        flush=True,
    )

    cmd = ["ffmpeg", "-y", "-nostdin"]
    filter_parts = []
    concat_inputs = []

    for idx, seg in enumerate(segments):
        cmd.extend(["-i", str(seg["local_path"])])
        extra = 0.15 if idx == len(segments) - 1 else 0.0
        dur = f"{seg['used_duration'] + extra:.3f}"
        filter_parts.append(
            f"[{idx}:v]setpts=PTS-STARTPTS,trim=duration={dur},setpts=PTS-STARTPTS,"
            f"fps={fps},scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h},setsar=1[v{idx}]"
        )
        concat_inputs.append(f"[v{idx}]")

    audio_idx = len(segments)
    cmd.extend(["-i", str(audio_wav)])

    video_chain = (
        ";".join(filter_parts)
        + f";{''.join(concat_inputs)}concat=n={len(segments)}:v=1:a=0,format=yuv420p[outv]"
    )
    # স্টুডিও ভয়েস মাস্টারিং: লো-নয়েজ কাট + ওয়ার্ম EQ + YouTube -14 LUFS + 48kHz
    audio_chain = (
        f";[{audio_idx}:a]highpass=f=70,"
        "equalizer=f=180:t=q:w=1.2:g=2.0,"
        "equalizer=f=3500:t=q:w=1.5:g=1.5,"
        "loudnorm=I=-14:TP=-1.5:LRA=9,"
        "aresample=48000[outa]"
    )
    filter_complex = video_chain + audio_chain

    cmd.extend([
        "-filter_complex",
        filter_complex,
        "-map",
        "[outv]",
        "-map",
        "[outa]",
        "-c:v",
        "libx264",
        "-preset",
        FFMPEG_PRESET,
        "-crf",
        str(FFMPEG_CRF),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "320k",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output_mp4),
    ])

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg ভিডিও এনকোডিং ব্যর্থ:\n{proc.stderr[-2000:]}")

    if not output_mp4.exists() or output_mp4.stat().st_size == 0:
        raise RuntimeError("আউটপুট ভিডিও ফাইল তৈরি হয়নি।")

    size_mb = output_mp4.stat().st_size / (1024 * 1024)
    print(f"[Video] ভিডিও তৈরি সম্পন্ন: {output_mp4.name} ({size_mb:.2f} MB)", flush=True)
    return plan
