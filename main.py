import shutil
import sys

from src.ai_writer import generate_monologue_script
from src.config import (
    CLEANUP_AFTER_GDRIVE,
    LINK_FILE,
    OUTPUT_DIR,
    SKIP_UPLOAD,
    UPLOAD_TO_YOUTUBE,
)
from src.gdrive_uploader import (
    list_bg_videos,
    permanently_delete_bg_video,
    upload_to_gdrive,
)
from src.thumbnail_maker import create_gold_thumbnail
from src.transcript import get_transcript
from src.tts_engine import synthesize_speech
from src.video_maker import create_gameplay_video
from src.youtube_uploader import upload_to_youtube


def read_next_link() -> tuple[list[str], int, str] | tuple[list[str], None, None]:
    if not LINK_FILE.exists():
        print(f"[Main] '{LINK_FILE.name}' ফাইলটি পাওয়া যায়নি।")
        return [], None, None

    lines = LINK_FILE.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return lines, idx, stripped
    return lines, None, None


def remove_processed_link(lines: list[str], processed_idx: int) -> None:
    remaining = [line for i, line in enumerate(lines) if i != processed_idx]
    content = "\n".join(remaining).strip()
    LINK_FILE.write_text((content + "\n") if content else "", encoding="utf-8")
    print(f"[Cleanup] ব্যবহৃত লিংকটি '{LINK_FILE.name}' থেকে ডিলিট করা হয়েছে।")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lines, target_idx, target_url = read_next_link()
    if target_url is None or target_idx is None:
        print("[Main] Link.txt-এ কোনো প্রসেস করার মতো লিংক নেই। কাজ শেষ।")
        return 0

    bg_videos = list_bg_videos()
    if not bg_videos:
        print("[Main] Google Drive-এর গেমপ্লে ফোল্ডারে কোনো ভিডিও নেই। কাজ শেষ।")
        return 0

    print(f"[Main] প্রসেস করা হচ্ছে: {target_url}")
    print(f"[Main] আপলোড মোড: {'YouTube (Live)' if UPLOAD_TO_YOUTUBE else 'Google Drive (Testing / Review)'}")

    # ১. হিন্দি ট্রান্সক্রিপ্ট সংগ্রহ
    transcript_data = get_transcript(target_url)
    video_id = transcript_data["video_id"]

    # ২. ইংলিশ মনোলোগ স্ক্রিপ্ট + SEO Title/Description/Tags + Thumbnail Slogan
    content = generate_monologue_script(transcript_data["text"], video_id)
    print(f"[Main] SEO টাইটেল: {content['title']}")
    print(f"[Main] থাম্বনেইল স্লোগান: {content['thumbnail_slogan']}")
    print(f"[Main] স্ক্রিপ্ট শব্দ সংখ্যা: {len(content['script'].split())}")

    # ৩. ভয়েসওভার অডিও জেনারেট (Kaggle Dual-GPU Qwen3-TTS / ফলব্যাক)
    audio_path = OUTPUT_DIR / f"{video_id}.wav"
    synthesize_speech(content["script"], audio_path)

    # ৪. Google Drive থেকে রেনডম গেমপ্লে ভিডিও নিয়ে মিউট, ট্রিম ও অডিও যুক্ত করে ভিডিও তৈরি
    video_path = OUTPUT_DIR / f"{video_id}.mp4"
    bg_tmp_dir = OUTPUT_DIR / f"bg_{video_id}"
    bg_plan = create_gameplay_video(audio_path, video_path, bg_tmp_dir)

    # ৫. গেমপ্লে ভিডিওর ফ্রেমের উপর মেটালিক গোল্ড ফন্টের স্লোগান দিয়ে থাম্বনেইল তৈরি
    thumb_path = OUTPUT_DIR / f"{video_id}_thumb.jpg"
    create_gold_thumbnail(
        video_path=bg_plan["thumb_source_video"],
        slogan=content["thumbnail_slogan"],
        output_jpg=thumb_path,
        seek_sec=bg_plan["thumb_seek_sec"],
    )

    if SKIP_UPLOAD:
        print("[Main] SKIP_UPLOAD=true থাকায় আপলোড ও কোনো ফাইল ডিলিট করা হলো না।")
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
        print(f"[Main] YouTube-এ আপলোড সফল! নতুন ভিডিও ID: {uploaded_id}")
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
        print(f"[Main] Google Drive-এ আপলোড সফল! ফোল্ডার: {drive_folder}")

    # ৭. শুধু YouTube আপলোডের পর (অথবা CLEANUP_AFTER_GDRIVE=true হলে) ডিলিট
    if should_cleanup:
        for item in bg_plan["to_delete"]:
            permanently_delete_bg_video(item["remote_name"])
        for kept in bg_plan["to_keep"]:
            print(
                f"[Cleanup-GDrive] অক্ষত রাখা হলো (<= 60s ব্যবহৃত): {kept['remote_name']} "
                f"({kept['used_sec']:.1f}s / {kept['full_sec']:.1f}s)"
            )
        remove_processed_link(lines, target_idx)
    else:
        print("[Cleanup] Google Drive টেস্ট মোড (UPLOAD_TO_YOUTUBE=false): Link.txt ও গেমপ্লে ভিডিও অক্ষত রাখা হলো।")

    # টেম্প লোকাল ফাইল পরিষ্কার
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
        print(f"[ERROR] অটোমেশন ব্যর্থ হয়েছে (কোনো লিংক বা Google Drive ভিডিও ডিলিট করা হয়নি): {exc}")
        sys.exit(1)
