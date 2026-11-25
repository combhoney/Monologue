import asyncio
import base64
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import edge_tts
import numpy as np
import soundfile as sf

from src.config import (
    EDGE_TTS_RATE,
    EDGE_TTS_VOICE,
    KAGGLE_API_TOKEN,
    KAGGLE_CACHE_KERNEL_SLUG,
    KAGGLE_KERNEL_SLUG,
    KAGGLE_KEY,
    KAGGLE_TIMEOUT_SEC,
    KAGGLE_USERNAME,
    KOKORO_SPEED,
    KOKORO_VOICE,
    MODELS_DIR,
    QWEN_INSTRUCT,
    QWEN_SPEAKER,
    QWEN_TTS_MODEL,
    TTS_ENGINE,
)

KOKORO_MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/kokoro-v1.0.int8.onnx"
)
KOKORO_VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/voices-v1.0.bin"
)


def _split_text_into_chunks(text: str, max_chars: int = 1600) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(para) <= max_chars:
            if len(current) + len(para) + 1 <= max_chars:
                current = f"{current} {para}".strip()
            else:
                if current:
                    chunks.append(current)
                current = para
        else:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                if len(current) + len(sent) + 1 <= max_chars:
                    current = f"{current} {sent}".strip()
                else:
                    if current:
                        chunks.append(current)
                    current = sent

    if current:
        chunks.append(current)
    return chunks


# ==============================================================================
# ১. Kaggle Authentication & Kernel Status Helpers
# ==============================================================================
def _has_kaggle_credentials() -> bool:
    return bool(KAGGLE_USERNAME and (KAGGLE_API_TOKEN or KAGGLE_KEY))


def _setup_kaggle_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("KAGGLE_KEY", None)
    env.pop("KAGGLE_API_TOKEN", None)

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)

    if KAGGLE_KEY:
        kaggle_json = kaggle_dir / "kaggle.json"
        kaggle_json.write_text(
            json.dumps({"username": KAGGLE_USERNAME, "key": KAGGLE_KEY}),
            encoding="utf-8",
        )
        os.chmod(kaggle_json, 0o600)
        env["KAGGLE_USERNAME"] = KAGGLE_USERNAME
        env["KAGGLE_KEY"] = KAGGLE_KEY

    if KAGGLE_API_TOKEN:
        env["KAGGLE_API_TOKEN"] = KAGGLE_API_TOKEN
        token_file = kaggle_dir / "access_token"
        token_file.write_text(KAGGLE_API_TOKEN, encoding="utf-8")
        os.chmod(token_file, 0o600)

    return env


def _get_kernel_status(kernel_id: str, env: dict[str, str]) -> tuple[str, str]:
    proc = subprocess.run(
        ["kaggle", "kernels", "status", kernel_id],
        env=env,
        capture_output=True,
        text=True,
    )
    combined = f"{proc.stdout} {proc.stderr}".strip()
    if proc.returncode != 0 and ("404" in combined or "not found" in combined.lower()):
        return "NOT_FOUND", combined
    match = re.search(r"KernelWorkerStatus\.(\w+)", combined, re.IGNORECASE)
    status = match.group(1).upper() if match else ("NOT_FOUND" if proc.returncode != 0 else "UNKNOWN")
    return status, combined


def _wait_for_kernel(kernel_id: str, env: dict[str, str], timeout_sec: int, label: str) -> None:
    deadline = time.time() + timeout_sec
    last_status = ""
    time.sleep(8)

    while time.time() < deadline:
        status, raw_msg = _get_kernel_status(kernel_id, env)
        if status != last_status:
            print(f"[{label}] কার্নেল স্ট্যাটাস: {status}", flush=True)
            last_status = status

        if status == "COMPLETE":
            return
        if status in ("ERROR", "CANCEL_ACKNOWLEDGED", "CANCELLED"):
            with tempfile.TemporaryDirectory() as log_tmp:
                subprocess.run(
                    ["kaggle", "kernels", "output", kernel_id, "-p", log_tmp],
                    env=env,
                    capture_output=True,
                    text=True,
                )
                logs = list(Path(log_tmp).rglob("*.log"))
                tail = logs[0].read_text(errors="ignore")[-1500:] if logs else raw_msg
            raise RuntimeError(f"[{label}] রান ব্যর্থ হয়েছে ({status}):\n{tail}")

        time.sleep(15)

    raise TimeoutError(f"[{label}] নির্ধারিত {timeout_sec} সেকেন্ডের মধ্যে শেষ হয়নি।")


# ==============================================================================
# ২. One-Time Persistent Model & Wheel Cache inside User's Kaggle Account
# ==============================================================================
def _build_cache_setup_script() -> str:
    return f'''import subprocess
import sys
from pathlib import Path

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "huggingface_hub"])
from huggingface_hub import snapshot_download

MODEL_ID = {QWEN_TTS_MODEL!r}
MODEL_OUT = Path("/kaggle/working/qwen3_model")
WHEELS_OUT = Path("/kaggle/working/wheels")
MODEL_OUT.mkdir(parents=True, exist_ok=True)
WHEELS_OUT.mkdir(parents=True, exist_ok=True)

print(f"Downloading {{MODEL_ID}} once into Kaggle persistent cache...", flush=True)
snapshot_download(repo_id=MODEL_ID, local_dir=str(MODEL_OUT))

print("Downloading qwen-tts dependency wheels into cache...", flush=True)
subprocess.check_call([
    sys.executable, "-m", "pip", "download",
    "qwen-tts", "soundfile",
    "-d", str(WHEELS_OUT),
])

skip_prefixes = ("torch-", "torchaudio-", "torchvision-", "nvidia_", "triton-", "cuda_")
for item in list(WHEELS_OUT.iterdir()):
    if item.name.lower().startswith(skip_prefixes):
        item.unlink()
    elif item.suffix in (".gz", ".zip") or item.name.endswith(".tar.gz"):
        subprocess.check_call([
            sys.executable, "-m", "pip", "wheel", "--no-deps",
            str(item), "-w", str(WHEELS_OUT),
        ])
        item.unlink()

print("Cached wheels count:", len(list(WHEELS_OUT.glob("*.whl"))), flush=True)
print("Permanent Qwen3-TTS cache ready!", flush=True)
'''


def ensure_kaggle_model_cache(env: dict[str, str] | None = None) -> str:
    if not _has_kaggle_credentials():
        raise RuntimeError("Kaggle ক্রেডেনশিয়াল (KAGGLE_USERNAME ও KAGGLE_API_TOKEN/KAGGLE_KEY) সেট করা নেই।")

    env = env or _setup_kaggle_env()
    cache_kernel_id = f"{KAGGLE_USERNAME}/{KAGGLE_CACHE_KERNEL_SLUG}"
    status, _ = _get_kernel_status(cache_kernel_id, env)

    if status == "COMPLETE":
        print(f"[TTS-Cache] স্থায়ী Kaggle মডেল ক্যাশ পাওয়া গেছে: {cache_kernel_id} (ডাউনলোড লাগবে না!)", flush=True)
        return cache_kernel_id

    if status in ("QUEUED", "RUNNING"):
        print(f"[TTS-Cache] স্থায়ী ক্যাশ কার্নেল ইতিমধ্যে চলছে ({status}), শেষ হওয়া পর্যন্ত অপেক্ষা করা হচ্ছে...", flush=True)
        _wait_for_kernel(cache_kernel_id, env, timeout_sec=1200, label="TTS-Cache")
        return cache_kernel_id

    print(
        f"[TTS-Cache] প্রথমবারের মতো আপনার Kaggle অ্যাকাউন্টে {QWEN_TTS_MODEL} স্থায়ীভাবে সেভ করা হচ্ছে "
        "(CPU মোডে চলবে, GPU কোটা খরচ হবে না)...",
        flush=True,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache_setup"
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "cache_setup.py").write_text(_build_cache_setup_script(), encoding="utf-8")

        metadata = {
            "id": cache_kernel_id,
            "title": KAGGLE_CACHE_KERNEL_SLUG,
            "code_file": "cache_setup.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "false",
            "enable_internet": "true",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        (cache_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        push = subprocess.run(
            ["kaggle", "kernels", "push", "-p", str(cache_dir)],
            env=env,
            capture_output=True,
            text=True,
        )
        if push.returncode != 0:
            raise RuntimeError(f"Kaggle cache kernel push ব্যর্থ:\n{push.stdout}\n{push.stderr}")

    _wait_for_kernel(cache_kernel_id, env, timeout_sec=1200, label="TTS-Cache")
    print(f"[TTS-Cache] মডেল স্থায়ীভাবে আপনার Kaggle অ্যাকাউন্টে সেভ হয়ে গেছে: {cache_kernel_id}", flush=True)
    return cache_kernel_id


# ==============================================================================
# ৩. Dual T4 GPU (cuda:0 + cuda:1) Parallel Inference Worker
# ==============================================================================
def _build_kaggle_dual_gpu_script(script_text: str) -> str:
    payload_b64 = base64.b64encode(script_text.encode("utf-8")).decode("ascii")
    return f'''import base64
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def ensure_runtime_deps():
    if not shutil.which("sox"):
        subprocess.run(["apt-get", "update", "-qq"], check=False)
        subprocess.run(["apt-get", "install", "-y", "-qq", "sox", "libsox-fmt-all"], check=False)

    wheels_dir = None
    for p in Path("/kaggle/input").rglob("wheels"):
        if p.is_dir() and any(p.glob("*.whl")):
            wheels_dir = str(p)
            break

    if wheels_dir:
        print(f"Installing qwen-tts from cached wheels: {{wheels_dir}}", flush=True)
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
            "--find-links", wheels_dir, "qwen-tts", "soundfile",
        ])
    else:
        print("Wheels cache not found, installing qwen-tts from PyPI...", flush=True)
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", "-U", "qwen-tts", "soundfile",
        ])


def find_model_source() -> tuple[str, bool]:
    for cfg in Path("/kaggle/input").rglob("config.json"):
        if cfg.parent.name == "qwen3_model" and (cfg.parent / "speech_tokenizer").exists():
            return str(cfg.parent), True
    return {QWEN_TTS_MODEL!r}, False


def split_chunks(text: str, max_chars: int = 700) -> list[str]:
    paras = [p.strip() for p in re.split(r"\\n+", text) if p.strip()]
    out, cur = [], ""
    for p in paras:
        if len(p) <= max_chars:
            if len(cur) + len(p) + 1 <= max_chars:
                cur = f"{{cur}} {{p}}".strip()
            else:
                if cur:
                    out.append(cur)
                cur = p
        else:
            for s in re.split(r"(?<=[.!?])\\s+", p):
                s = s.strip()
                if not s:
                    continue
                if len(cur) + len(s) + 1 <= max_chars:
                    cur = f"{{cur}} {{s}}".strip()
                else:
                    if cur:
                        out.append(cur)
                    cur = s
    if cur:
        out.append(cur)
    return out


def gpu_worker(gpu_id: int, tasks: list[tuple[int, str]], model_src: str, is_offline: bool, out_dir: str, err_q):
    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        if is_offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        import numpy as np
        import torch
        from qwen_tts import Qwen3TTSModel

        torch.cuda.set_device(gpu_id)
        device = f"cuda:{{gpu_id}}"
        print(f"[GPU {{gpu_id}}] Loading Qwen3-TTS on {{device}}...", flush=True)

        try:
            model = Qwen3TTSModel.from_pretrained(
                model_src,
                device_map=device,
                dtype=torch.float16,
                attn_implementation="sdpa",
                local_files_only=is_offline,
            )
        except Exception:
            model = Qwen3TTSModel.from_pretrained(
                model_src,
                device_map=device,
                dtype=torch.float16,
                local_files_only=is_offline,
            )

        speaker = {QWEN_SPEAKER!r}
        instruct = {QWEN_INSTRUCT!r}

        for chunk_idx, chunk_text in tasks:
            print(f"[GPU {{gpu_id}}] Synthesizing chunk #{{chunk_idx}} ({{len(chunk_text)}} chars)...", flush=True)
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            gen_kwargs = dict(
                temperature=0.75,
                top_p=0.90,
                repetition_penalty=1.08,
                max_new_tokens=4096,
            )
            if "VoiceDesign" in model_src:
                wavs, sr = model.generate_voice_design(
                    text=chunk_text,
                    language="English",
                    instruct=instruct,
                    **gen_kwargs,
                )
            else:
                wavs, sr = model.generate_custom_voice(
                    text=chunk_text,
                    language="English",
                    speaker=speaker,
                    instruct=instruct,
                    **gen_kwargs,
                )
            arr = np.asarray(wavs[0], dtype=np.float32)
            np.save(Path(out_dir) / f"chunk_{{chunk_idx:04d}}.npy", arr)
            with open(Path(out_dir) / f"sr_{{chunk_idx:04d}}.txt", "w") as f:
                f.write(str(int(sr)))
            torch.cuda.empty_cache()

        print(f"[GPU {{gpu_id}}] Finished all {{len(tasks)}} assigned chunks!", flush=True)
    except Exception as exc:
        err_q.put((gpu_id, repr(exc)))


def main():
    ensure_runtime_deps()

    import numpy as np
    import soundfile as sf
    import torch

    model_src, is_offline = find_model_source()
    print(f"Model source resolved: {{model_src}} (offline={{is_offline}})", flush=True)

    text = base64.b64decode("{payload_b64}").decode("utf-8")
    chunks = split_chunks(text, max_chars=700)
    n_gpus = torch.cuda.device_count()
    print(f"Total chunks: {{len(chunks)}} | Available CUDA GPUs: {{n_gpus}}", flush=True)
    for g in range(n_gpus):
        print(f"  -> GPU {{g}}: {{torch.cuda.get_device_name(g)}}", flush=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = mp.get_context("spawn")
        err_q = ctx.Queue()

        if n_gpus >= 2 and len(chunks) >= 2:
            tasks_0 = [(i, chunks[i]) for i in range(0, len(chunks), 2)]
            tasks_1 = [(i, chunks[i]) for i in range(1, len(chunks), 2)]
            p0 = ctx.Process(target=gpu_worker, args=(0, tasks_0, model_src, is_offline, tmpdir, err_q))
            p1 = ctx.Process(target=gpu_worker, args=(1, tasks_1, model_src, is_offline, tmpdir, err_q))
            p0.start()
            p1.start()
            p0.join()
            p1.join()
            if not err_q.empty():
                gid, msg = err_q.get()
                raise RuntimeError(f"GPU {{gid}} worker error: {{msg}}")
            if p0.exitcode != 0 or p1.exitcode != 0:
                raise RuntimeError(f"Worker exit codes: p0={{p0.exitcode}}, p1={{p1.exitcode}}")
        else:
            tasks_all = list(enumerate(chunks))
            gpu_worker(0, tasks_all, model_src, is_offline, tmpdir, err_q)
            if not err_q.empty():
                gid, msg = err_q.get()
                raise RuntimeError(f"GPU {{gid}} worker error: {{msg}}")

        audio_parts = []
        sample_rate = 24000
        for i in range(len(chunks)):
            sr_file = Path(tmpdir) / f"sr_{{i:04d}}.txt"
            if sr_file.exists():
                sample_rate = int(sr_file.read_text().strip())
            arr = np.load(Path(tmpdir) / f"chunk_{{i:04d}}.npy")
            audio_parts.append(arr)
            audio_parts.append(np.zeros(int(sample_rate * 0.28), dtype=np.float32))

        final_audio = np.concatenate(audio_parts) if audio_parts else np.zeros(sample_rate, dtype=np.float32)
        peak = float(np.max(np.abs(final_audio))) if final_audio.size else 1.0
        if peak > 0:
            final_audio = (final_audio / peak) * 0.89

        sf.write("/kaggle/working/voiceover.wav", final_audio, sample_rate)
        print("Saved /kaggle/working/voiceover.wav successfully!", flush=True)


if __name__ == "__main__":
    main()
'''


def _synthesize_kaggle_gpu(script: str, output_wav: Path) -> None:
    if not _has_kaggle_credentials():
        raise RuntimeError("Kaggle ক্রেডেনশিয়াল (KAGGLE_USERNAME ও KAGGLE_API_TOKEN/KAGGLE_KEY) সেট করা নেই।")

    env = _setup_kaggle_env()
    cache_kernel_id = ensure_kaggle_model_cache(env)
    worker_kernel_id = f"{KAGGLE_USERNAME}/{KAGGLE_KERNEL_SLUG}"

    with tempfile.TemporaryDirectory() as tmpdir:
        job_dir = Path(tmpdir) / "kaggle_job"
        out_dir = Path(tmpdir) / "kaggle_out"
        job_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        (job_dir / "tts_runner.py").write_text(_build_kaggle_dual_gpu_script(script), encoding="utf-8")

        metadata = {
            "id": worker_kernel_id,
            "title": KAGGLE_KERNEL_SLUG,
            "code_file": "tts_runner.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [cache_kernel_id],
            "model_sources": [],
        }
        (job_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        print(f"[TTS-Kaggle] ডুয়াল T4 GPU জব পুশ হচ্ছে: {worker_kernel_id} (Mounted cache: {cache_kernel_id})...", flush=True)
        push_proc = subprocess.run(
            ["kaggle", "kernels", "push", "-p", str(job_dir)],
            env=env,
            capture_output=True,
            text=True,
        )
        if push_proc.returncode != 0:
            raise RuntimeError(f"Kaggle worker push ব্যর্থ:\n{push_proc.stdout}\n{push_proc.stderr}")

        _wait_for_kernel(worker_kernel_id, env, timeout_sec=KAGGLE_TIMEOUT_SEC, label="TTS-Kaggle-DualGPU")

        print("[TTS-Kaggle] ডুয়াল GPU থেকে তৈরি voiceover.wav ডাউনলোড হচ্ছে...", flush=True)
        dl_proc = subprocess.run(
            ["kaggle", "kernels", "output", worker_kernel_id, "-p", str(out_dir)],
            env=env,
            capture_output=True,
            text=True,
        )
        if dl_proc.returncode != 0:
            raise RuntimeError(f"Kaggle output ডাউনলোড ব্যর্থ: {dl_proc.stderr}")

        wav_candidates = list(out_dir.rglob("voiceover.wav"))
        if not wav_candidates or wav_candidates[0].stat().st_size < 1024:
            raise RuntimeError("Kaggle আউটপুটে voiceover.wav পাওয়া যায়নি।")

        cmd = [
            "ffmpeg", "-y", "-i", str(wav_candidates[0]),
            "-ar", "24000", "-ac", "1", str(output_wav),
        ]
        ff = subprocess.run(cmd, capture_output=True, text=True)
        if ff.returncode != 0:
            raise RuntimeError(f"FFmpeg WAV কনভার্সন এরর: {ff.stderr}")


# ==============================================================================
# ৪. Edge-TTS & Kokoro-ONNX Fallback
# ==============================================================================
async def _edge_tts_chunk(text: str, out_mp3: Path) -> None:
    communicate = edge_tts.Communicate(text=text, voice=EDGE_TTS_VOICE, rate=EDGE_TTS_RATE)
    await communicate.save(str(out_mp3))


def _synthesize_edge_tts(script: str, output_wav: Path) -> None:
    chunks = _split_text_into_chunks(script, max_chars=1600)
    print(f"[TTS] Edge-TTS দিয়ে {len(chunks)}টি অংশে অডিও তৈরি হচ্ছে (Voice: {EDGE_TTS_VOICE})...", flush=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        part_files: list[Path] = []

        for idx, chunk in enumerate(chunks):
            part_mp3 = tmp_path / f"part_{idx:03d}.mp3"
            last_err = None
            for retry in range(3):
                try:
                    asyncio.run(_edge_tts_chunk(chunk, part_mp3))
                    if part_mp3.exists() and part_mp3.stat().st_size > 0:
                        part_files.append(part_mp3)
                        break
                except Exception as err:
                    last_err = err
                    print(f"[TTS] Edge-TTS অংশ {idx + 1} রিট্রাই ({retry + 1}/3): {err}", flush=True)
            else:
                raise RuntimeError(f"Edge-TTS ব্যর্থ: {last_err}")

        concat_list = tmp_path / "concat.txt"
        concat_list.write_text("\n".join(f"file '{p.as_posix()}'" for p in part_files), encoding="utf-8")

        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list), "-ar", "24000", "-ac", "1", str(output_wav),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg অডিও মার্জ এরর: {proc.stderr}")


def _download_if_missing(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 1024:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[TTS] Kokoro মডেল ডাউনলোড হচ্ছে: {dest.name} ...", flush=True)
    urllib.request.urlretrieve(url, dest)


def _synthesize_kokoro_tts(script: str, output_wav: Path) -> None:
    from kokoro_onnx import Kokoro

    model_path = MODELS_DIR / "kokoro-v1.0.int8.onnx"
    voices_path = MODELS_DIR / "voices-v1.0.bin"
    _download_if_missing(KOKORO_MODEL_URL, model_path)
    _download_if_missing(KOKORO_VOICES_URL, voices_path)

    kokoro = Kokoro(str(model_path), str(voices_path))
    chunks = _split_text_into_chunks(script, max_chars=360)
    print(f"[TTS] Kokoro-ONNX দিয়ে {len(chunks)}টি অংশে অডিও তৈরি হচ্ছে (Voice: {KOKORO_VOICE})...", flush=True)

    all_samples: list[np.ndarray] = []
    sample_rate = 24000
    pause_samples = np.zeros(int(sample_rate * 0.28), dtype=np.float32)

    for chunk in chunks:
        samples, sr = kokoro.create(chunk, voice=KOKORO_VOICE, speed=KOKORO_SPEED, lang="en-us")
        sample_rate = sr
        all_samples.append(np.asarray(samples, dtype=np.float32))
        all_samples.append(pause_samples)

    merged = np.concatenate(all_samples) if all_samples else np.zeros(sample_rate, dtype=np.float32)
    sf.write(str(output_wav), merged, sample_rate)


def synthesize_speech(script: str, output_wav: Path) -> float:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    mode = TTS_ENGINE.lower()

    if mode == "kaggle":
        _synthesize_kaggle_gpu(script, output_wav)
    elif mode == "kokoro":
        _synthesize_kokoro_tts(script, output_wav)
    elif mode == "edge":
        _synthesize_edge_tts(script, output_wav)
    else:
        done = False
        if _has_kaggle_credentials():
            try:
                _synthesize_kaggle_gpu(script, output_wav)
                done = True
            except Exception as kg_err:
                print(f"[TTS] Kaggle ডুয়াল-GPU ব্যর্থ ({kg_err})। ব্যাকআপ TTS চালু হচ্ছে...", flush=True)

        if not done:
            try:
                _synthesize_edge_tts(script, output_wav)
            except Exception as edge_err:
                print(f"[TTS] Edge-TTS কাজ করেনি ({edge_err})। Kokoro-ONNX ফলব্যাক চালু হচ্ছে...", flush=True)
                _synthesize_kokoro_tts(script, output_wav)

    info = sf.info(str(output_wav))
    duration = float(info.duration)
    print(f"[TTS] অডিও তৈরি সম্পন্ন: {output_wav.name} (সময়: {duration:.2f} সেকেন্ড)", flush=True)
    return duration
