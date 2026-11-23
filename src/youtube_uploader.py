import time
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

from src.config import (
    YOUTUBE_CATEGORY_ID,
    YOUTUBE_CLIENT_ID,
    YOUTUBE_CLIENT_SECRET,
    YOUTUBE_PRIVACY_STATUS,
    YOUTUBE_REFRESH_TOKEN,
)


def _sanitize_youtube_text(text: str, max_len: int) -> str:
    cleaned = text.replace("<", "").replace(">", "").strip()
    return cleaned[:max_len]


def _sanitize_tags(tags: list[str]) -> list[str]:
    result = []
    total_chars = 0
    for tag in tags:
        t = tag.replace("<", "").replace(">", "").strip()
        if not t:
            continue
        if total_chars + len(t) + 1 > 480:
            break
        result.append(t)
        total_chars += len(t) + 1
    return result


def upload_to_youtube(
    video_path: Path,
    title: str,
    description: str,
    tags: list[str],
    thumbnail_path: Path | None = None,
) -> str:
    if not (YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET and YOUTUBE_REFRESH_TOKEN):
        raise ValueError(
            "YouTube OAuth ক্রেডেনশিয়াল সেট করা নেই! "
            "YOUTUBE_CLIENT_ID, YOUTUBE_CLIENT_SECRET এবং YOUTUBE_REFRESH_TOKEN সিক্রেট যোগ করুন।"
        )

    creds = Credentials(
        token=None,
        refresh_token=YOUTUBE_REFRESH_TOKEN,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=YOUTUBE_CLIENT_ID,
        client_secret=YOUTUBE_CLIENT_SECRET,
        scopes=["https://www.googleapis.com/auth/youtube.upload"],
    )
    creds.refresh(Request())

    youtube = build("youtube", "v3", credentials=creds)

    body = {
        "snippet": {
            "title": _sanitize_youtube_text(title, 98),
            "description": _sanitize_youtube_text(description, 4950),
            "tags": _sanitize_tags(tags),
            "categoryId": YOUTUBE_CATEGORY_ID,
            "defaultLanguage": "en",
            "defaultAudioLanguage": "en",
        },
        "status": {
            "privacyStatus": YOUTUBE_PRIVACY_STATUS,
            "selfDeclaredMadeForKids": False,
        },
        "paidProductPlacementDetails": {
            "hasPaidProductPlacement": False,
        },
    }

    media = MediaFileUpload(
        str(video_path),
        mimetype="video/mp4",
        chunksize=8 * 1024 * 1024,
        resumable=True,
    )

    request = youtube.videos().insert(
        part="snippet,status,paidProductPlacementDetails",
        body=body,
        media_body=media,
    )

    print(f"[YouTube] আপলোড শুরু হচ্ছে: '{body['snippet']['title']}' ({YOUTUBE_PRIVACY_STATUS})")
    response = None
    retry = 0
    max_retries = 8

    while response is None:
        try:
            status, response = request.next_chunk()
            if status:
                print(f"[YouTube] আপলোড প্রগ্রেস: {int(status.progress() * 100)}%")
        except HttpError as err:
            if err.resp.status in (500, 502, 503, 504) and retry < max_retries:
                retry += 1
                wait = min(2**retry, 60)
                print(f"[YouTube] সার্ভার এরর ({err.resp.status}), {wait}s পর রিট্রাই ({retry}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
        except Exception as err:
            if retry < max_retries:
                retry += 1
                wait = min(2**retry, 60)
                print(f"[YouTube] কানেকশন সমস্যা ({err}), {wait}s পর রিট্রাই ({retry}/{max_retries})...")
                time.sleep(wait)
            else:
                raise

    video_id = response.get("id", "")
    if not video_id:
        raise RuntimeError(f"আপলোড রেসপন্সে Video ID পাওয়া যায়নি: {response}")

    print(f"[YouTube] ভিডিও আপলোড সফল! লিংক: https://www.youtube.com/watch?v={video_id}")

    # কাস্টম থাম্বনেইল আপলোড
    if thumbnail_path and thumbnail_path.exists():
        try:
            thumb_media = MediaFileUpload(str(thumbnail_path), mimetype="image/jpeg")
            youtube.thumbnails().set(videoId=video_id, media_body=thumb_media).execute()
            print("[YouTube] কাস্টম গোল্ডেন থাম্বনেইল সফলভাবে সেট হয়েছে!")
        except HttpError as thumb_err:
            print(
                f"[YouTube] সতর্কতা: থাম্বনেইল সেট করা যায়নি (HTTP {thumb_err.resp.status})। "
                "আপনার ইউটিউব চ্যানেলটি ফোন-ভেরিফাইড কি না চেক করুন (Intermediate features)।"
            )

    return video_id