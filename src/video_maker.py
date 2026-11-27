import json
import random
import subprocess
from pathlib import Path

import soundfile as sf

from src.config import (
    ALLOW_BG_LOOP,
    AUDIO_SPEED,
    BG_DELETE_THRESHOLD_SEC,
    BG_LONG_VIDEO_MIN_SEC,
    FFMPEG_CRF,
    FFMPEG_PRESET,
    MIRROR_BG_VIDEO,
    SUBTITLE_ENABLED,
    VIDEO_FPS,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
)
from src.gdrive_uploader import (
    download_bg_video,
    list_bg_videos,
    serve_bg_folder_http,
)
from src.subtitle_maker import generate_animated_ass_subtitles


def _get_media_duration(target: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        target,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe দিয়ে ভিডিওর ডিউরেশন মাপা যায়নি: {proc.stderr}")
    data = json.loads(proc.stdout or "{}")
    dur = float((data.get("format") or {}).get("duration", 0.0))
    if dur <= 0:
        raise RuntimeError("ভিডিওটির ডিউরেশন সঠিক নয়।")
    return dur


def _prepare_mastered_audio(raw_wav: Path, mastered_wav: Path) -> float:
    speed = max(0.5, min(2.0, float(AUDIO_SPEED)))
    filters = []
    if abs(speed - 1.0) > 0.005:
        filters.append(f"atempo={speed:.4f}")
    filters.extend([
        "highpass=f=70",
        "equalizer=f=180:t=q:w=1.2:g=2.0",
        "equalizer=f=3500:t=q:w=1.5:g=1.5",
        "loudnorm=I=-14:TP=-1.5:LRA=9",
        "aresample=48000",
    ])
    af_chain = ",".join(filters)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(raw_wav),
        "-af",
        af_chain,
        "-ar",
        "48000",
        "-ac",
        "2",
        str(mastered_wav),
    ]
    print(f"[Audio] স্টুডিও মাস্টারিং ও স্পিড ({speed:.2f}x) প্রয়োগ হচ্ছে...", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg অডিও মাস্টারিং ব্যর্থ:\n{proc.stderr[-1500:]}")

    dur = float(sf.info(str(mastered_wav)).duration)
    print(f"[Audio] ফাইনাল অডিও প্রস্তুত: {dur:.2f} সেকেন্ড (48kHz Stereo)", flush=True)
    return dur


def _choose_random_start_offset(full_dur: float, use_dur: float) -> float:
    if full_dur >= BG_LONG_VIDEO_MIN_SEC and (full_dur - use_dur) > 5.0:
        slack = full_dur - use_dur
        min_start = min(60.0, slack * 0.10)
        max_start = max(min_start, slack - min(30.0, slack * 0.05))
        return round(random.uniform(min_start, max_start), 2)
    return 0.0


def _slice_remote_video_segment(stream_url: str, start_sec: float, use_dur: float, dest_path: Path) -> bool:
    cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-ss",
        f"{start_sec:.3f}",
        "-i",
        stream_url,
        "-t",
        f"{use_dur + 4.0:.3f}",
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        str(dest_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dest_path.exists() or dest_path.stat().st_size < 10_000:
        return False

    try:
        sliced_dur = _get_media_duration(str(dest_path))
        return sliced_dur >= (use_dur * 0.90)
    except Exception:
        return False


def _select_and_download_gameplay_clips(audio_duration: float, bg_tmp_dir: Path) -> dict:
    remote_videos = list_bg_videos()
    if not remote_videos:
        raise RuntimeError("Google Drive-এর গেমপ্লে ফোল্ডারে কোনো ভিডিও পাওয়া যায়নি!")

    random.shuffle(remote_videos)
    bg_tmp_dir.mkdir(parents=True, exist_ok=True)

    remaining = audio_duration
    segments: list[dict] = []
    full_duration_map: dict[str, float] = {}
    usage_totals: dict[str, float] = {}

    with serve_bg_folder_http() as get_stream_url:
        for idx, item in enumerate(remote_videos):
            if remaining <= 0.05:
                break
            remote_name = item["Name"]
            stream_url = get_stream_url(remote_name)
            safe_local_name = f"bg_{idx:03d}_{Path(remote_name).name}"
            local_path = bg_tmp_dir / safe_local_name

            full_dur = 0.0
            try:
                full_dur = _get_media_duration(stream_url)
            except Exception as err:
                print(f"[Video] স্ট্রিম থেকে ডিউরেশন পড়া যায়নি ({remote_name}): {err}", flush=True)

            if full_dur > 0.5:
                use_dur = min(full_dur, remaining)
                start_sec = _choose_random_start_offset(full_dur, use_dur)
                print(
                    f"[Video] নির্বাচিত ক্লিপ '{remote_name}' | পূর্ণ দৈর্ঘ্য: {full_dur/60:.1f}m | "
                    f"রেনডম অংশ: {start_sec/60:.2f}m -> {(start_sec + use_dur)/60:.2f}m ({use_dur:.1f}s)...",
                    flush=True,
                )

                sliced_ok = False
                if full_dur >= BG_LONG_VIDEO_MIN_SEC or use_dur < (full_dur - 15.0):
                    sliced_ok = _slice_remote_video_segment(stream_url, start_sec, use_dur, local_path)

                if sliced_ok:
                    local_start = 0.0
                else:
                    if local_path.exists():
                        local_path.unlink()
                    download_bg_video(remote_name, local_path)
                    local_start = start_sec
            else:
                download_bg_video(remote_name, local_path)
                full_dur = _get_media_duration(str(local_path))
                if full_dur <= 0.5:
                    print(f"[Video] খুব ছোট/নষ্ট ভিডিও স্কিপ করা হলো: {remote_name} ({full_dur:.2f}s)", flush=True)
                    continue
                use_dur = min(full_dur, remaining)
                local_start = _choose_random_start_offset(full_dur, use_dur)

            segments.append(
                {
                    "remote_name": remote_name,
                    "local_path": local_path,
                    "start_sec": local_start,
                    "used_duration": use_dur,
                    "full_duration": full_dur,
                }
            )
            full_duration_map[remote_name] = full_dur
            usage_totals[remote_name] = usage_totals.get(remote_name, 0.0) + use_dur
            remaining -= use_dur

            print(
                f"[Video] ক্লিপ প্রস্তুত: '{remote_name}' | ব্যবহৃত: {use_dur:.1f}s | "
                f"বাকি অডিও: {max(0.0, remaining):.1f}s",
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
                avail_local = _get_media_duration(str(seg["local_path"]))
                use_dur = min(avail_local, remaining)
                segments.append(
                    {
                        "remote_name": seg["remote_name"],
                        "local_path": seg["local_path"],
                        "start_sec": 0.0,
                        "used_duration": use_dur,
                        "full_duration": seg["full_duration"],
                    }
                )
                usage_totals[seg["remote_name"]] += use_dur
                remaining -= use_dur

    to_delete: list[dict] = []
    to_keep: list[dict] = []

    for r_name, used_sec in usage_totals.items():
        full_sec = full_duration_map[r_name]
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

    first_seg = segments[0]
    return {
        "segments": segments,
        "to_delete": to_delete,
        "to_keep": to_keep,
        "thumb_source_video": first_seg["local_path"],
        "thumb_seek_sec": first_seg["start_sec"] + min(first_seg["used_duration"] * 0.25, 15.0),
    }


def _escape_ffmpeg_path(p: Path) -> str:
    return p.as_posix().replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def create_gameplay_video(
    audio_wav: Path,
    output_mp4: Path,
    bg_tmp_dir: Path,
    script_text: str = "",
) -> dict:
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    bg_tmp_dir.mkdir(parents=True, exist_ok=True)
    w, h, fps = VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS

    mastered_wav = bg_tmp_dir / "mastered_voiceover.wav"
    audio_dur = _prepare_mastered_audio(audio_wav, mastered_wav)

    ass_file: Path | None = None
    sub_fonts_dir = bg_tmp_dir / "sub_fonts"
    if SUBTITLE_ENABLED and script_text.strip():
        ass_file = generate_animated_ass_subtitles(
            audio_wav=mastered_wav,
            script_text=script_text,
            out_ass=bg_tmp_dir / "captions.ass",
            fonts_dir_out=sub_fonts_dir,
        )

    plan = _select_and_download_gameplay_clips(audio_dur, bg_tmp_dir)
    segments = plan["segments"]

    print(
        f"[Video] FFmpeg দিয়ে ভিডিও তৈরি হচ্ছে: {w}x{h} @ {fps}fps "
        f"(preset={FFMPEG_PRESET}, crf={FFMPEG_CRF}, mirror={'ON' if MIRROR_BG_VIDEO else 'OFF'}, "
        f"subtitles={'ON' if ass_file else 'OFF'})",
        flush=True,
    )

    cmd = ["ffmpeg", "-y", "-nostdin"]
    filter_parts = []
    concat_inputs = []
    mirror_filter = ",hflip" if MIRROR_BG_VIDEO else ""

    for idx, seg in enumerate(segments):
        if seg["start_sec"] > 0:
            cmd.extend(["-ss", f"{seg['start_sec']:.3f}"])
        cmd.extend(["-i", str(seg["local_path"])])

        extra = 0.15 if idx == len(segments) - 1 else 0.0
        dur = f"{seg['used_duration'] + extra:.3f}"
        filter_parts.append(
            f"[{idx}:v]setpts=PTS-STARTPTS,trim=duration={dur},setpts=PTS-STARTPTS,"
            f"fps={fps},scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h},setsar=1{mirror_filter}[v{idx}]"
        )
        concat_inputs.append(f"[v{idx}]")

    audio_idx = len(segments)
    cmd.extend(["-i", str(mastered_wav)])

    if ass_file is not None:
        esc_ass = _escape_ffmpeg_path(ass_file)
        esc_fonts = _escape_ffmpeg_path(sub_fonts_dir)
        filter_complex = (
            ";".join(filter_parts)
            + f";{''.join(concat_inputs)}concat=n={len(segments)}:v=1:a=0[vcat];"
            + f"[vcat]ass=filename='{esc_ass}':fontsdir='{esc_fonts}',format=yuv420p[outv]"
        )
    else:
        filter_complex = (
            ";".join(filter_parts)
            + f";{''.join(concat_inputs)}concat=n={len(segments)}:v=1:a=0,format=yuv420p[outv]"
        )

    cmd.extend([
        "-filter_complex",
        filter_complex,
        "-map",
        "[outv]",
        "-map",
        f"{audio_idx}:a",
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
