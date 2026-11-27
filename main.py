import shutil
import sys

from src.ai_writer import generate_monologue_script
from src.channel_fetcher import (
    is_channel_url,
    load_history,
    pick_next_popular_video_from_channel,
    record_video_in_history,
)
from src.config import (
    CLEANUP_AFTER_GDRIVE,
    LINK_FILE,
    OUTPUT_DIR,
    SKIP_UPLOAD,
    SOURCE_CHANNEL_URL,
    UPLOAD_TO_YOUTUBE,
)
from src.gdrive_uploader import (
    list_bg_videos,
    permanently_delete_bg_video,
    upload_to_gdrive,
)
from src.thumbnail_maker import create_modern_thumbnail
from src.transcript import extract_video_id, get_transcript
from src.tts_engine import synthesize_speech
from src.video_maker import create_gameplay_video
from src.youtube_uploader import upload_to_youtube


def _read_link_entries() -> list[tuple[int, str]]:
    if not LINK_FILE.exists():
        return []
    lines = LINK_FILE.read_text(encoding="utf-8").splitlines()
    entries = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            entries.append((idx, stripped))
    return entries


def _remove_line_from_link_file(target_idx: int) -> None:
    if not LINK_FILE.exists():
        return
    lines = LINK_FILE.read_text(encoding="utf-8").splitlines()
    remaining = [line for i, line in enumerate(lines) if i != target_idx]
    content = "\n".join(remaining).strip()
    LINK_FILE.write_text((content + "\n") if content else "", encoding="utf-8")
    print(f"[Cleanup] ব্যবহৃত ভিডিও লিংকটি '{LINK_FILE.name}' থেকে ডিলিট করা হয়েছে।", flush=True)


def _select_next_target(history: dict) -> dict | None:
    processed_ids = set(history.get("processed_video_ids") or [])
    entries = _read_link_entries()

    video_entries: list[tuple[int, str]] = []
    channel_entries: list[tuple[int, str]] = []

    for idx, url in entries:
        if is_channel_url(url):
            channel_entries.append((idx, url))
        else:
            video_entries.append((idx, url))

    # ১. Link.txt-এ আলাদা কোনো ভিডিও লিংক থাকলে আগে সেটি নেওয়া (ডুপ্লিকেট হলে স্কিপ)
    for idx, v_url in video_entries:
        try:
            vid = extract_video_id(v_url)
        except Exception:
            vid = ""
        if vid and vid in processed_ids:
            print(f"[Main] '{v_url}' ইতিমধ্যে history.json-এ আছে (ডুপ্লিকেট)। স্কিপ করে রিমুভ করা হচ্ছে...", flush=True)
            _remove_line_from_link_file(idx)
            continue
        return {
            "mode": "direct_video",
            "url": v_url,
            "link_line_idx": idx,
            "channel_key": None,
            "rank": None,
        }

    # ২. Link.txt-এ চ্যানেল লিংক অথবা SOURCE_CHANNEL_URL থাকলে সেখান থেকে মোস্ট পপুলার পরবর্তী ভিডিও নেওয়া
    channel_urls = [u for _, u in channel_entries]
    if SOURCE_CHANNEL_URL and SOURCE_CHANNEL_URL not in channel_urls:
        channel_urls.append(SOURCE_CHANNEL_URL)

    for ch_url in channel_urls:
        print(f"[Main] চ্যানেল থেকে পরবর্তী Most Popular ভিডিও খোঁজা হচ্ছে: {ch_url}", flush=True)
        picked = pick_next_popular_video_from_channel(ch_url, history)
        if picked:
            print(
                f"[Main] নির্বাচিত পপুলার ভিডিও -> Rank #{picked['rank']} | "
                f"Views: {picked['view_count']:,} | {picked['title']}",
                flush=True,
            )
            return {
                "mode": "channel_popular",
                "url": picked["url"],
                "link_line_idx": None,  # চ্যানেল লিংক Link.txt থেকে ডিলিট হবে না
                "channel_key": picked["channel_key"],
                "rank": picked["rank"],
            }

    return None


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history = load_history()

    target = _select_next_target(history)
    if not target:
        print("[Main] Link.txt বা চ্যানেলে প্রসেস করার মতো নতুন কোনো ভিডিও পাওয়া যায়নি। কাজ শেষ।", flush=True)
        return 0

    bg_videos = list_bg_videos()
    if not bg_videos:
        print("[Main] Google Drive-এর গেমপ্লে ফোল্ডারে কোনো ভিডিও নেই। কাজ শেষ।", flush=True)
        return 0

    target_url = target["url"]
    print(f"[Main] প্রসেস করা হচ্ছে: {target_url}", flush=True)
    print(f"[Main] আপলোড মোড: {'YouTube (Live)' if UPLOAD_TO_YOUTUBE else 'Google Drive (Testing / Review)'}", flush=True)

    # ১. হিন্দি ট্রান্সক্রিপ্ট সংগ্রহ
    transcript_data = get_transcript(target_url)
    video_id = transcript_data["video_id"]

    # ২. ইংরেজি স্ক্রিপ্ট + SEO Title/Description/Tags + Thumbnail Slogan
    content = generate_monologue_script(transcript_data["text"], video_id)
    print(f"[Main] SEO টাইটেল: {content['title']}", flush=True)
    print(f"[Main] থাম্বনেইল স্লোগান: {content['thumbnail_slogan']}", flush=True)
    print(f"[Main] স্ক্রিপ্ট শব্দ সংখ্যা: {len(content['script'].split())}", flush=True)

    # ৩. ভয়েসওভার অডিও জেনারেট
    audio_path = OUTPUT_DIR / f"{video_id}.wav"
    synthesize_speech(content["script"], audio_path)

    # ৪. অডিও মাস্টারিং + Word-by-Word সাবটাইটেল + গেমপ্লে ভিডিও তৈরি
    video_path = OUTPUT_DIR / f"{video_id}.mp4"
    bg_tmp_dir = OUTPUT_DIR / f"bg_{video_id}"
    bg_plan = create_gameplay_video(
        audio_wav=audio_path,
        output_mp4=video_path,
        bg_tmp_dir=bg_tmp_dir,
        script_text=content["script"],
    )

    # ৫. ৮০% ব্লার ডার্ক ব্যাকগ্রাউন্ড + রেনডম এক-কালার মডার্ন লাইট টেক্সট থাম্বনেইল তৈরি
    thumb_path = OUTPUT_DIR / f"{video_id}_thumb.jpg"
    create_modern_thumbnail(
        video_path=bg_plan["thumb_source_video"],
        slogan=content["thumbnail_slogan"],
        output_jpg=thumb_path,
        seek_sec=bg_plan["thumb_seek_sec"],
    )

    if SKIP_UPLOAD:
        print("[Main] SKIP_UPLOAD=true থাকায় আপলোড, হিস্টরি আপডেট ও কোনো ফাইল ডিলিট করা হলো না।", flush=True)
        return 0

    # ৬. UPLOAD_TO_YOUTUBE=true হলে YouTube-এ, নইলে Google Drive-এ আপলোড
    should_cleanup = False
    if UPLOAD_TO_YOUTUBE:
        uploaded_id = upload_to_youtube(
            video_path=video_path,
            title=content["title"],
            description=content["description"],
            tags=content["tags"],
            thumbnail_path=thumb_path,
        )
        should_cleanup = True
        print(f"[Main] YouTube-এ আপলোড সফল! নতুন ভিডিও ID: {uploaded_id}", flush=True)
    else:
        drive_folder = upload_to_gdrive(
            video_path=video_path,
            thumbnail_path=thumb_path,
            source_url=target_url,
            video_id=video_id,
            title=content["title"],
            thumbnail_slogan=content["thumbnail_slogan"],
            description=content["description"],
            tags=content["tags"],
            script=content["script"],
        )
        should_cleanup = CLEANUP_AFTER_GDRIVE
        print(f"[Main] Google Drive-এ আপলোড সফল! ফোল্ডার: {drive_folder}", flush=True)

    # ৭. সফল আপলোডের পর history.json আপডেট এবং শর্ত অনুযায়ী ক্লিনআপ
    if should_cleanup:
        record_video_in_history(
            history=history,
            video_id=video_id,
            url=target_url,
            title=content["title"],
            channel_key=target["channel_key"],
            rank=target["rank"],
        )
        for item in bg_plan["to_delete"]:
            permanently_delete_bg_video(item["remote_name"])
        for kept in bg_plan["to_keep"]:
            print(
                f"[Cleanup-GDrive] অক্ষত রাখা হলো (<= 60s ব্যবহৃত): {kept['remote_name']} "
                f"({kept['used_sec']:.1f}s / {kept['full_sec']:.1f}s)",
                flush=True,
            )
        if target["link_line_idx"] is not None:
            _remove_line_from_link_file(target["link_line_idx"])
    else:
        print(
            "[Cleanup] Google Drive টেস্ট মোড (UPLOAD_TO_YOUTUBE=false): Link.txt, history.json ও গেমপ্লে ভিডিও অক্ষত রাখা হলো।",
            flush=True,
        )

    for tmp in (audio_path, video_path, thumb_path):
        if tmp.exists():
            tmp.unlink()
    if bg_tmp_dir.exists():
        shutil.rmtree(bg_tmp_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[ERROR] অটোমেশন ব্যর্থ হয়েছে (কোনো লিংক, history বা Google Drive ভিডিও ডিলিট করা হয়নি): {exc}", flush=True)
        sys.exit(1)
