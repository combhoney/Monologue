import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import GenericProxyConfig

from src.config import PROXY_URL, YOUTUBE_COOKIES, load_supadata_keys

PREFERRED_LANGUAGES = ["hi", "hi-IN", "hi-Latn", "en", "en-IN", "en-US", "en-GB"]


def extract_video_id(url: str) -> str:
    url = url.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url

    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.strip("/")

    if "youtu.be" in host and path:
        return path.split("/")[0]

    if "youtube.com" in host or "youtube-nocookie.com" in host:
        if path == "watch":
            qs = parse_qs(parsed.query)
            if "v" in qs and qs["v"]:
                return qs["v"][0]
        for prefix in ("shorts/", "live/", "embed/", "v/"):
            if path.startswith(prefix):
                return path[len(prefix):].split("/")[0]

    match = re.search(r"(?:v=|/)([A-Za-z0-9_-]{11})(?:[?&/]|$)", url)
    if match:
        return match.group(1)

    raise ValueError(f"সঠিক YouTube URL নয়: {url}")


def _clean_transcript_text(text: str) -> str:
    text = re.sub(r"\[(?:Music|Applause|Laughter|संगीत|ताली)\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fetch_via_supadata(video_url: str, video_id: str, keys: list[str]) -> dict:
    errors: list[str] = []
    for idx, key in enumerate(keys, start=1):
        print(f"[Transcript] Supadata API দিয়ে চেষ্টা চলছে (Key#{idx})...")
        try:
            resp = requests.get(
                "https://api.supadata.ai/v1/youtube/transcript",
                params={"url": video_url, "text": "true"},
                headers={"x-api-key": key},
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                text = _clean_transcript_text(str(data.get("content", "")))
                if text:
                    return {
                        "video_id": video_id,
                        "language": str(data.get("lang", "auto")),
                        "text": text,
                        "source": f"supadata (Key#{idx})",
                    }
                errors.append(f"Key#{idx}: খালি ট্রান্সক্রিপ্ট")
            else:
                errors.append(f"Key#{idx}: HTTP {resp.status_code} - {resp.text[:180]}")
        except Exception as exc:
            errors.append(f"Key#{idx}: {exc}")

    raise RuntimeError("Supadata API ব্যর্থ হয়েছে: " + " | ".join(errors))


def _fetch_via_transcript_api(video_id: str) -> dict:
    if PROXY_URL:
        api = YouTubeTranscriptApi(
            proxy_config=GenericProxyConfig(http_url=PROXY_URL, https_url=PROXY_URL)
        )
    else:
        api = YouTubeTranscriptApi()

    transcript_list = api.list(video_id)
    chosen = None

    for finder in (
        transcript_list.find_manually_created_transcript,
        transcript_list.find_generated_transcript,
    ):
        try:
            chosen = finder(PREFERRED_LANGUAGES)
            break
        except Exception:
            continue

    if chosen is None:
        available = list(transcript_list)
        if not available:
            raise RuntimeError("ভিডিওটিতে কোনো ট্রান্সক্রিপ্ট পাওয়া যায়নি।")
        chosen = available[0]

    fetched = chosen.fetch()
    lines = [snippet.text.strip() for snippet in fetched if getattr(snippet, "text", "").strip()]
    full_text = _clean_transcript_text(" ".join(lines))
    if not full_text:
        raise RuntimeError("ট্রান্সক্রিপ্ট খালি এসেছে।")

    return {
        "video_id": video_id,
        "language": chosen.language_code,
        "text": full_text,
        "source": "youtube-transcript-api",
    }


def _clean_vtt(vtt_content: str) -> str:
    lines_out: list[str] = []
    seen_recent: list[str] = []

    for raw_line in vtt_content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE", "STYLE")):
            continue
        if "-->" in line or re.fullmatch(r"\d+", line):
            continue

        line = re.sub(r"<[^>]+>", "", line).strip()
        if not line:
            continue

        if lines_out and line == lines_out[-1]:
            continue
        if line in seen_recent:
            continue

        lines_out.append(line)
        seen_recent.append(line)
        if len(seen_recent) > 5:
            seen_recent.pop(0)

    return _clean_transcript_text(" ".join(lines_out))


def _fetch_via_yt_dlp(video_url: str, video_id: str) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # কুকিজ থাকলে web,mweb,android ব্যবহার করতে হয় (android,ios কুকিজ সাপোর্ট করে না)
        clients = "web,mweb,android" if YOUTUBE_COOKIES else "android,ios,web"
        cmd = [
            "yt-dlp",
            "--no-warnings",
            "--skip-download",
            "--write-auto-subs",
            "--write-subs",
            "--sub-langs",
            "hi.*,en.*,all",
            "--sub-format",
            "vtt",
            "--extractor-args",
            f"youtube:player_client={clients}",
            "-o",
            str(tmp_path / "%(id)s.%(ext)s"),
        ]

        if shutil.which("node"):
            cmd.extend(["--js-runtimes", "node"])

        if PROXY_URL:
            cmd.extend(["--proxy", PROXY_URL])

        if YOUTUBE_COOKIES:
            cookie_file = tmp_path / "cookies.txt"
            cookie_file.write_text(YOUTUBE_COOKIES, encoding="utf-8")
            cmd.extend(["--cookies", str(cookie_file)])

        cmd.append(video_url)

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        vtt_files = list(tmp_path.glob("*.vtt"))
        if not vtt_files:
            raise RuntimeError(
                f"yt-dlp সাবটাইটেল আনতে পারেনি (code={proc.returncode}): {proc.stderr.strip()}"
            )

        vtt_files.sort(
            key=lambda p: (0 if ".hi" in p.name else (1 if ".en" in p.name else 2), p.name)
        )
        chosen_vtt = vtt_files[0]
        cleaned = _clean_vtt(chosen_vtt.read_text(encoding="utf-8", errors="ignore"))
        if not cleaned:
            raise RuntimeError("yt-dlp দিয়ে পাওয়া VTT ফাইল খালি।")

        lang_match = re.search(r"\.([a-zA-Z-]+)\.vtt$", chosen_vtt.name)
        lang = lang_match.group(1) if lang_match else "unknown"

        return {
            "video_id": video_id,
            "language": lang,
            "text": cleaned,
            "source": "yt-dlp",
        }


def get_transcript(video_url: str) -> dict:
    video_id = extract_video_id(video_url)
    print(f"[Transcript] ভিডিও আইডি: {video_id} থেকে ট্রান্সক্রিপ্ট সংগ্রহ শুরু...")

    supadata_keys = load_supadata_keys()
    if supadata_keys:
        try:
            data = _fetch_via_supadata(video_url, video_id, supadata_keys)
            print(f"[Transcript] সফল ({data['source']}, ভাষা: {data['language']}, অক্ষর: {len(data['text'])})")
            return data
        except Exception as supa_err:
            print(f"[Transcript] Supadata ব্যর্থ হয়েছে: {supa_err}")

    try:
        data = _fetch_via_transcript_api(video_id)
        print(f"[Transcript] সফল ({data['source']}, ভাষা: {data['language']}, অক্ষর: {len(data['text'])})")
        return data
    except Exception as api_err:
        print(f"[Transcript] youtube-transcript-api ব্যর্থ হয়েছে: {api_err}")
        print("[Transcript] yt-dlp ফলব্যাক চেষ্টা করা হচ্ছে...")

    try:
        data = _fetch_via_yt_dlp(video_url, video_id)
        print(f"[Transcript] সফল ({data['source']}, ভাষা: {data['language']}, অক্ষর: {len(data['text'])})")
        return data
    except Exception as ytdlp_err:
        raise RuntimeError(
            "ট্রান্সক্রিপ্ট বের করা যায়নি! GitHub Actions ক্লাউড IP ইউটিউব ব্লক করেছে। "
            "সমাধানের জন্য SUPADATA_API_KEYS (সবচেয়ে সহজ, মাসে ১০০টি ফ্রি), PROXY_URL অথবা YOUTUBE_COOKIES সিক্রেট যোগ করুন।\n"
            f"বিস্তারিত: {ytdlp_err}"
        ) from ytdlp_err
