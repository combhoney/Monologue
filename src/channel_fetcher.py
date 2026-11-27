import json
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from googleapiclient.discovery import build

from src.config import (
    CHANNEL_MAX_SCAN_VIDEOS,
    CHANNEL_MIN_DURATION_SEC,
    HISTORY_FILE,
    PROXY_URL,
    YOUTUBE_API_KEY,
    YOUTUBE_COOKIES,
)


def is_channel_url(url: str) -> bool:
    u = url.strip()
    if not u:
        return False
    if u.startswith("@"):
        return True
    parsed = urlparse(u)
    host = parsed.netloc.lower()
    if host and "youtube.com" not in host and "youtu.be" not in host:
        return False
    path = parsed.path.strip("/")
    if not path:
        return False
    first = path.split("/")[0]
    return first.startswith("@") or first in ("channel", "c", "user")


def normalize_channel_key(url: str) -> str:
    u = url.strip()
    if u.startswith("@"):
        return f"https://www.youtube.com/{u.split('/')[0]}"
    parsed = urlparse(u)
    parts = [p for p in parsed.path.strip("/").split("/") if p]
    if parts and parts[0].startswith("@"):
        return f"https://www.youtube.com/{parts[0]}"
    if len(parts) >= 2 and parts[0] in ("channel", "c", "user"):
        return f"https://www.youtube.com/{parts[0]}/{parts[1]}"
    return u.rstrip("/")


def load_history() -> dict:
    if not HISTORY_FILE.exists():
        return {"channels": {}, "processed_video_ids": [], "processed_links": []}
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Invalid history format")
        data.setdefault("channels", {})
        data.setdefault("processed_video_ids", [])
        data.setdefault("processed_links", [])
        return data
    except Exception:
        return {"channels": {}, "processed_video_ids": [], "processed_links": []}


def save_history(history: dict) -> None:
    HISTORY_FILE.write_text(
        json.dumps(history, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _parse_iso8601_duration(dur: str) -> int:
    m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", dur or "")
    if not m:
        return 0
    return int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60 + int(m.group(3) or 0)


def _fetch_popular_via_youtube_api(channel_url: str) -> tuple[str, list[dict]]:
    if not YOUTUBE_API_KEY:
        raise RuntimeError("YOUTUBE_API_KEY সেট করা নেই।")

    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    norm_url = normalize_channel_key(channel_url)
    parsed = urlparse(norm_url)
    parts = [p for p in parsed.path.strip("/").split("/") if p]

    ch_resp = None
    if len(parts) >= 2 and parts[0] == "channel":
        ch_resp = youtube.channels().list(part="id,contentDetails,snippet", id=parts[1]).execute()
    elif parts and parts[0].startswith("@"):
        ch_resp = youtube.channels().list(
            part="id,contentDetails,snippet", forHandle=f"@{parts[0].lstrip('@')}"
        ).execute()
    elif len(parts) >= 2 and parts[0] == "user":
        ch_resp = youtube.channels().list(part="id,contentDetails,snippet", forUsername=parts[1]).execute()
    elif len(parts) >= 2 and parts[0] == "c":
        s_resp = youtube.search().list(part="snippet", q=parts[1], type="channel", maxResults=1).execute()
        s_items = s_resp.get("items") or []
        if s_items:
            cid = s_items[0]["snippet"]["channelId"]
            ch_resp = youtube.channels().list(part="id,contentDetails,snippet", id=cid).execute()

    items = (ch_resp or {}).get("items") or []
    if not items:
        raise RuntimeError(f"YouTube API দিয়ে চ্যানেলটি খুঁজে পাওয়া যায়নি: {channel_url}")

    channel_id = items[0]["id"]
    uploads_id = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
    channel_title = items[0]["snippet"].get("title", norm_url)
    print(f"[Channel] চ্যানেল পাওয়া গেছে: '{channel_title}' ({channel_id})", flush=True)

    # চ্যানেলের আপলোড তালিকা থেকে ভিডিও আইডি সংগ্রহ (সর্বোচ্চ CHANNEL_MAX_SCAN_VIDEOS পর্যন্ত)
    video_ids: list[str] = []
    next_page = None
    while len(video_ids) < CHANNEL_MAX_SCAN_VIDEOS:
        pl_resp = (
            youtube.playlistItems()
            .list(
                part="contentDetails",
                playlistId=uploads_id,
                maxResults=50,
                pageToken=next_page,
            )
            .execute()
        )
        for pl_item in pl_resp.get("items") or []:
            vid = (pl_item.get("contentDetails") or {}).get("videoId")
            if vid:
                video_ids.append(vid)
        next_page = pl_resp.get("nextPageToken")
        if not next_page:
            break

    # প্রতিটি ভিডিওর viewCount ও duration সংগ্রহ
    popular_list: list[dict] = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i : i + 50]
        v_resp = (
            youtube.videos()
            .list(part="snippet,contentDetails,statistics", id=",".join(batch))
            .execute()
        )
        for v in v_resp.get("items") or []:
            snippet = v.get("snippet") or {}
            if snippet.get("liveBroadcastContent", "none") != "none":
                continue
            dur_sec = _parse_iso8601_duration((v.get("contentDetails") or {}).get("duration", ""))
            if dur_sec < CHANNEL_MIN_DURATION_SEC:
                continue
            views = int((v.get("statistics") or {}).get("viewCount", 0) or 0)
            vid = v["id"]
            popular_list.append(
                {
                    "video_id": vid,
                    "url": f"https://www.youtube.com/watch?v={vid}",
                    "title": snippet.get("title", ""),
                    "view_count": views,
                    "duration_sec": dur_sec,
                }
            )

    popular_list.sort(key=lambda x: x["view_count"], reverse=True)
    for rank_idx, entry in enumerate(popular_list, start=1):
        entry["rank"] = rank_idx

    print(f"[Channel] মোট {len(popular_list)}টি পপুলার লং ভিডিও সাজানো হয়েছে।", flush=True)
    return channel_id, popular_list


def _fetch_popular_via_yt_dlp(channel_url: str) -> tuple[str, list[dict]]:
    norm_url = normalize_channel_key(channel_url)
    target_url = f"{norm_url}/videos" if not norm_url.endswith("/videos") else norm_url

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            "yt-dlp",
            "--no-warnings",
            "--flat-playlist",
            "--dump-single-json",
            "--playlist-end",
            str(CHANNEL_MAX_SCAN_VIDEOS),
            "--extractor-args",
            "youtube:player_client=web",
        ]
        if shutil.which("node"):
            cmd.extend(["--js-runtimes", "node"])
        if PROXY_URL:
            cmd.extend(["--proxy", PROXY_URL])
        if YOUTUBE_COOKIES:
            c_file = Path(tmpdir) / "cookies.txt"
            c_file.write_text(YOUTUBE_COOKIES, encoding="utf-8")
            cmd.extend(["--cookies", str(c_file)])

        cmd.append(target_url)
        print(f"[Channel] yt-dlp দিয়ে চ্যানেলের ভিডিও তালিকা আনা হচ্ছে: {target_url}", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if proc.returncode != 0 or not proc.stdout.strip():
            raise RuntimeError(
                f"yt-dlp দিয়ে চ্যানেল লিস্ট আনা যায়নি। সহজে সমাধানের জন্য YOUTUBE_API_KEY সিক্রেট যোগ করুন।\n"
                f"বিস্তারিত: {proc.stderr[-800:]}"
            )

        data = json.loads(proc.stdout)
        channel_id = str(data.get("channel_id") or data.get("id") or "")
        popular_list: list[dict] = []

        for e in data.get("entries") or []:
            if not e:
                continue
            vid = str(e.get("id", "")).strip()
            if not vid or len(vid) != 11:
                continue
            dur_sec = int(e.get("duration") or 0)
            if dur_sec and dur_sec < CHANNEL_MIN_DURATION_SEC:
                continue
            views = int(e.get("view_count") or 0)
            popular_list.append(
                {
                    "video_id": vid,
                    "url": f"https://www.youtube.com/watch?v={vid}",
                    "title": str(e.get("title") or ""),
                    "view_count": views,
                    "duration_sec": dur_sec,
                }
            )

        # view_count পাওয়া গেলে বড় থেকে ছোট ক্রমে সাজানো
        if any(x["view_count"] > 0 for x in popular_list):
            popular_list.sort(key=lambda x: x["view_count"], reverse=True)

        for rank_idx, entry in enumerate(popular_list, start=1):
            entry["rank"] = rank_idx

        print(f"[Channel] yt-dlp থেকে {len(popular_list)}টি ভিডিও পাওয়া গেছে।", flush=True)
        return channel_id, popular_list


def _fetch_channel_popular_videos(channel_url: str) -> tuple[str, list[dict]]:
    if YOUTUBE_API_KEY:
        try:
            return _fetch_popular_via_youtube_api(channel_url)
        except Exception as api_err:
            print(f"[Channel] YouTube Data API ব্যর্থ হয়েছে ({api_err}), yt-dlp চেষ্টা করা হচ্ছে...", flush=True)

    return _fetch_popular_via_yt_dlp(channel_url)


def _is_cache_fresh(cached_at_iso: str, max_days: int = 14) -> bool:
    if not cached_at_iso:
        return False
    try:
        dt = datetime.fromisoformat(cached_at_iso.replace("Z", "+00:00"))
        age_days = (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0
        return age_days <= max_days
    except Exception:
        return False


def pick_next_popular_video_from_channel(channel_url: str, history: dict) -> dict | None:
    """
    history.json দেখে চ্যানেলের মোস্ট পপুলার ভিডিওর পরবর্তী (last_popular_rank + 1) ভিডিও নেয়
    এবং আগে তৈরি হওয়া কোনো video_id থাকলে তা অটোমেটিক স্কিপ করে।
    """
    ch_key = normalize_channel_key(channel_url)
    channels_map = history.setdefault("channels", {})
    ch_state = channels_map.get(ch_key) or {}

    last_rank = int(ch_state.get("last_popular_rank", 0) or 0)
    queue: list[dict] = ch_state.get("popular_queue") or []
    processed_ids = set(history.get("processed_video_ids") or [])

    # ক্যাশ পুরোনো হলে বা খালি থাকলে চ্যানেল থেকে রিফ্রেশ করা
    if not queue or not _is_cache_fresh(str(ch_state.get("cached_at", ""))):
        channel_id, queue = _fetch_channel_popular_videos(channel_url)
        ch_state = {
            "channel_id": channel_id,
            "last_popular_rank": last_rank,
            "total_popular_cached": len(queue),
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "popular_queue": queue,
        }
        channels_map[ch_key] = ch_state

    # ১. প্রথমে গতবার যেই র‍্যাংক (last_popular_rank) পর্যন্ত নেওয়া হয়েছে তার পরেরগুলো থেকে খোঁজা
    for item in queue:
        rank = int(item.get("rank", 0))
        vid = str(item.get("video_id", ""))
        if rank > last_rank and vid and vid not in processed_ids:
            return {
                "channel_key": ch_key,
                "rank": rank,
                "video_id": vid,
                "url": item["url"],
                "title": item.get("title", ""),
                "view_count": item.get("view_count", 0),
                "history_ref": history,
            }

    # ২. যদি last_popular_rank-এর পরে কিছু না থাকে, তাহলে পুরো কিউ থেকে আগে না বানানো ভিডিও খোঁজা
    for item in queue:
        vid = str(item.get("video_id", ""))
        if vid and vid not in processed_ids:
            return {
                "channel_key": ch_key,
                "rank": int(item.get("rank", 0)),
                "video_id": vid,
                "url": item["url"],
                "title": item.get("title", ""),
                "view_count": item.get("view_count", 0),
                "history_ref": history,
            }

    return None


def record_video_in_history(
    history: dict,
    video_id: str,
    url: str,
    title: str = "",
    channel_key: str | None = None,
    rank: int | None = None,
) -> None:
    processed_ids = history.setdefault("processed_video_ids", [])
    if video_id not in processed_ids:
        processed_ids.append(video_id)

    processed_links = history.setdefault("processed_links", [])
    processed_links.append(
        {
            "rank": rank,
            "source_channel": channel_key,
            "video_id": video_id,
            "url": url,
            "title": title,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    if channel_key and rank is not None:
        ch_state = history.setdefault("channels", {}).setdefault(channel_key, {})
        prev_rank = int(ch_state.get("last_popular_rank", 0) or 0)
        ch_state["last_popular_rank"] = max(prev_rank, int(rank))

    save_history(history)
    print(
        f"[History] history.json আপডেট হয়েছে -> Video: {video_id}"
        + (f" (Popular Rank #{rank})" if rank else ""),
        flush=True,
    )
