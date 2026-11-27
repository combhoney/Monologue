import base64
import json
import re
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from src.config import (
    GDRIVE_BG_FOLDER_ID,
    GDRIVE_FOLDER_ID,
    RCLONE_CONFIG,
    RCLONE_REMOTE_NAME,
)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".flv"}


def _decode_rclone_config(raw: str) -> str:
    text = raw.strip()
    if not text:
        raise ValueError(
            "RCLONE_CONFIG সিক্রেট খালি! আপনার rclone.conf ফাইলের টেক্সট "
            "অথবা Base64 স্ট্রিং RCLONE_CONFIG সিক্রেটে দিন।"
        )

    if re.search(r"^\s*\[[^\]]+\]", text, re.MULTILINE):
        return text + "\n"

    try:
        decoded = base64.b64decode(text).decode("utf-8").strip()
        if re.search(r"^\s*\[[^\]]+\]", decoded, re.MULTILINE):
            return decoded + "\n"
    except Exception:
        pass

    raise ValueError("RCLONE_CONFIG সঠিক ফরম্যাটে নেই।")


def _detect_remote_name(conf_text: str) -> str:
    if RCLONE_REMOTE_NAME:
        return RCLONE_REMOTE_NAME
    match = re.search(r"^\s*\[([^\]]+)\]", conf_text, re.MULTILINE)
    if not match:
        raise ValueError("rclone.conf-এ কোনো রিমোট সেকশন (যেমন [gdrive]) পাওয়া যায়নি।")
    return match.group(1).strip()


def extract_folder_id(raw_val: str) -> str:
    val = raw_val.strip()
    if not val:
        return ""
    match = re.search(r"folders/([A-Za-z0-9_-]+)", val)
    if match:
        return match.group(1)
    match_id = re.search(r"[?&]id=([A-Za-z0-9_-]+)", val)
    if match_id:
        return match_id.group(1)
    return val


@contextmanager
def _rclone_session():
    if not shutil.which("rclone"):
        raise RuntimeError("সিস্টেমে rclone ইনস্টল করা নেই!")
    conf_text = _decode_rclone_config(RCLONE_CONFIG)
    remote_name = _detect_remote_name(conf_text)
    with tempfile.TemporaryDirectory() as tmpdir:
        conf_file = Path(tmpdir) / "rclone.conf"
        conf_file.write_text(conf_text, encoding="utf-8")
        yield remote_name, conf_file


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextmanager
def serve_bg_folder_http():
    """
    Google Drive গেমপ্লে ফোল্ডারটিকে লোকাল HTTP রেঞ্জ-রিকোয়েস্ট সার্ভার হিসেবে চালু করে,
    যাতে ১০-১২ ঘণ্টার (২০-৩০ GB) ভিডিও পুরোটা ডাউনলোড না করেই যেকোনো অংশ কাটা যায়।
    """
    bg_folder_id = extract_folder_id(GDRIVE_BG_FOLDER_ID)
    if not bg_folder_id:
        raise ValueError("GDRIVE_BG_FOLDER_ID সিক্রেট সেট করা নেই!")

    with _rclone_session() as (remote_name, conf_file):
        port = _find_free_port()
        base_url = f"http://127.0.0.1:{port}"
        cmd = [
            "rclone",
            "serve",
            "http",
            f"{remote_name}:",
            "--drive-root-folder-id",
            bg_folder_id,
            "--config",
            str(conf_file),
            "--addr",
            f"127.0.0.1:{port}",
            "--read-only",
            "--vfs-read-chunk-size",
            "16M",
            "--vfs-read-chunk-size-limit",
            "64M",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            ready = False
            for _ in range(40):
                if proc.poll() is not None:
                    break
                try:
                    with urllib.request.urlopen(base_url, timeout=2):
                        ready = True
                        break
                except Exception:
                    time.sleep(0.35)

            if not ready:
                raise RuntimeError("rclone HTTP স্ট্রিম সার্ভার চালু হতে ব্যর্থ হয়েছে।")

            yield lambda fname: f"{base_url}/{urllib.parse.quote(fname)}"
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def list_bg_videos() -> list[dict]:
    bg_folder_id = extract_folder_id(GDRIVE_BG_FOLDER_ID)
    if not bg_folder_id:
        raise ValueError("GDRIVE_BG_FOLDER_ID সিক্রেট সেট করা নেই!")

    with _rclone_session() as (remote_name, conf_file):
        conf_txt = conf_file.read_text(encoding="utf-8")
        if re.search(r"^\s*scope\s*=\s*drive\.file\b", conf_txt, re.MULTILINE):
            raise RuntimeError(
                "আপনার rclone.conf-এ 'scope = drive.file' দেওয়া আছে! "
                "এতে ব্রাউজার থেকে আপলোড করা গেমপ্লে ভিডিও rclone দেখতে পায় না। "
                "rclone config-এ scope = 1 (drive) দিয়ে নতুন RCLONE_CONFIG দিন।"
            )

        cmd = [
            "rclone",
            "lsjson",
            f"{remote_name}:",
            "--files-only",
            "--drive-root-folder-id",
            bg_folder_id,
            "--config",
            str(conf_file),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Google Drive গেমপ্লে ফোল্ডার লিস্ট করা যায়নি:\n{proc.stderr}")

        items = json.loads(proc.stdout or "[]")
        videos = [
            item
            for item in items
            if not item.get("IsDir")
            and (
                Path(item["Name"]).suffix.lower() in VIDEO_EXTENSIONS
                or str(item.get("MimeType", "")).startswith("video/")
            )
        ]

    print(f"[GDrive-BG] ব্যাকগ্রাউন্ড ফোল্ডারে মোট {len(videos)}টি গেমপ্লে ভিডিও পাওয়া গেছে।", flush=True)
    return videos


def download_bg_video(remote_filename: str, dest_path: Path) -> Path:
    bg_folder_id = extract_folder_id(GDRIVE_BG_FOLDER_ID)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with _rclone_session() as (remote_name, conf_file):
        cmd = [
            "rclone",
            "copyto",
            f"{remote_name}:{remote_filename}",
            str(dest_path),
            "--drive-root-folder-id",
            bg_folder_id,
            "--config",
            str(conf_file),
            "--multi-thread-streams",
            "4",
            "--stats",
            "10s",
            "--stats-one-line",
            "-v",
        ]
        print(f"[GDrive-BG] গেমপ্লে ভিডিও ডাউনলোড হচ্ছে: {remote_filename} ...", flush=True)
        proc = subprocess.run(cmd)
        if proc.returncode != 0 or not dest_path.exists():
            raise RuntimeError(f"'{remote_filename}' ডাউনলোড ব্যর্থ (code={proc.returncode})")

    return dest_path


def permanently_delete_bg_video(remote_filename: str) -> None:
    bg_folder_id = extract_folder_id(GDRIVE_BG_FOLDER_ID)
    with _rclone_session() as (remote_name, conf_file):
        cmd = [
            "rclone",
            "deletefile",
            f"{remote_name}:{remote_filename}",
            "--drive-use-trash=false",
            "--drive-root-folder-id",
            bg_folder_id,
            "--config",
            str(conf_file),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"'{remote_filename}' স্থায়ীভাবে ডিলিট করতে ব্যর্থ "
                f"(rclone scope='drive' আছে কি না দেখুন):\n{proc.stderr}"
            )
    print(f"[Cleanup-GDrive] ট্রাশ ছাড়াই স্থায়ীভাবে ডিলিট হয়েছে: {remote_filename}", flush=True)


def _safe_filename(name: str, max_len: int = 55) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\r\n]+', "", name).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return (cleaned[:max_len].strip("_")) or "monologue"


def _build_metadata_txt(
    source_url: str,
    video_id: str,
    title: str,
    thumbnail_slogan: str,
    description: str,
    tags: list[str],
    script: str,
) -> str:
    tags_csv = ", ".join(tags)
    tags_hash = " ".join(f"#{re.sub(r'[^A-Za-z0-9]', '', t)}" for t in tags if t)
    return (
        "============================================================\n"
        "YOUTUBE MONOLOGUE AUTOMATION - AI GENERATED DETAILS\n"
        "============================================================\n\n"
        f"Source Video URL : {source_url}\n"
        f"Source Video ID  : {video_id}\n\n"
        "------------------------------------------------------------\n"
        "1. TITLE\n"
        "------------------------------------------------------------\n"
        f"{title}\n\n"
        "------------------------------------------------------------\n"
        "2. THUMBNAIL SLOGAN\n"
        "------------------------------------------------------------\n"
        f"{thumbnail_slogan}\n\n"
        "------------------------------------------------------------\n"
        "3. TAGS (Comma Separated)\n"
        "------------------------------------------------------------\n"
        f"{tags_csv}\n\n"
        f"Hashtags: {tags_hash}\n\n"
        "------------------------------------------------------------\n"
        "4. DESCRIPTION\n"
        "------------------------------------------------------------\n"
        f"{description}\n\n"
        "------------------------------------------------------------\n"
        "5. ENGLISH MONOLOGUE SCRIPT\n"
        "------------------------------------------------------------\n"
        f"{script}\n"
    )


def upload_to_gdrive(
    video_path: Path,
    thumbnail_path: Path,
    source_url: str,
    video_id: str,
    title: str,
    thumbnail_slogan: str,
    description: str,
    tags: list[str],
    script: str,
) -> str:
    folder_id = extract_folder_id(GDRIVE_FOLDER_ID)
    safe_base = f"{_safe_filename(title)}_{video_id}"

    with _rclone_session() as (remote_name, conf_file), tempfile.TemporaryDirectory() as tmpdir:
        bundle_dir = Path(tmpdir) / "bundle"
        bundle_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(video_path, bundle_dir / f"{safe_base}.mp4")
        if thumbnail_path.exists():
            shutil.copy2(thumbnail_path, bundle_dir / f"{safe_base}_thumbnail.jpg")

        txt_content = _build_metadata_txt(
            source_url=source_url,
            video_id=video_id,
            title=title,
            thumbnail_slogan=thumbnail_slogan,
            description=description,
            tags=tags,
            script=script,
        )
        (bundle_dir / f"{safe_base}_info.txt").write_text(txt_content, encoding="utf-8")

        cmd = [
            "rclone",
            "copy",
            str(bundle_dir),
            f"{remote_name}:{safe_base}",
            "--config",
            str(conf_file),
            "--drive-chunk-size",
            "128M",
            "--transfers",
            "4",
            "--stats",
            "10s",
            "--stats-one-line",
            "--timeout",
            "5m",
            "--contimeout",
            "60s",
            "-v",
        ]
        if folder_id:
            cmd.extend(["--drive-root-folder-id", folder_id])

        print(
            f"[GDrive-Upload] রিভিউ ফাইল আপলোড হচ্ছে -> Folder ID: '{folder_id or 'Root'}', Subfolder: '{safe_base}'",
            flush=True,
        )
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            raise RuntimeError(f"rclone আপলোড ব্যর্থ (code={proc.returncode})")

    print(f"[GDrive-Upload] ভিডিও, থাম্বনেইল ও TXT সফলভাবে Google Drive-এ সেভ হয়েছে! ({safe_base})", flush=True)
    return safe_base
