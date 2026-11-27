import difflib
import re
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from PIL import ImageFont

from src.config import (
    FONTS_DIR,
    SUBTITLE_FONT_SIZE,
    SUBTITLE_MARGIN_V,
    SUBTITLE_MAX_WORDS,
    SUBTITLE_MODEL,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
)

POPPINS_EXTRABOLD_URL = "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-ExtraBold.ttf"
MONTSERRAT_BOLD_URL = "https://github.com/googlefonts/montserrat/raw/master/fonts/ttf/Montserrat-Bold.ttf"

# মডার্ন কালার কম্বিনেশন (ASS BGR ফরম্যাটে: &HAABBGGRR&)
# - ছোট/কানেক্টর শব্দ: ক্রিস্প স্নো হোয়াইট (#F8FAFC)
# - মূল পাওয়ার/উচ্চারিত শব্দ: ওয়ার্ম গোল্ড-অ্যাম্বার (#FFD166)
# - স্ট্রোক / বর্ডার: ডিপ অবসিডিয়ান চারকোল (#0B0F17)
BASE_COLOR_ASS = "&H00FCFAF8&"
ACTIVE_COLOR_ASS = "&H0066D1FF&"
OUTLINE_COLOR_ASS = "&H00170F0B&"
SHADOW_COLOR_ASS = "&H96000000&"

_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _int_to_words(n: int) -> str:
    if n < 0:
        return "minus " + _int_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        rem = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rem == 0 else " " + _int_to_words(rem))
    if n < 1_000_000:
        rem = n % 1000
        return _int_to_words(n // 1000) + " thousand" + ("" if rem == 0 else " " + _int_to_words(rem))
    return ""


def _token_to_mms_words(tok: str) -> list[str]:
    s = tok.lower().replace("’", "'").replace("‘", "'")
    s = s.replace("-", " ").replace("/", " ").replace("%", " percent ").replace("&", " and ")
    s = re.sub(r"\d+", lambda m: " " + _int_to_words(int(m.group(0))) + " " if len(m.group(0)) <= 6 else " ", s)
    parts = []
    for chunk in s.split():
        cleaned = re.sub(r"[^a-z']", "", chunk).strip("'")
        if cleaned:
            parts.append(cleaned)
    return parts if parts else ["*"]


def _prepare_subtitle_font(fonts_dir_out: Path) -> tuple[str, bool]:
    fonts_dir_out.mkdir(parents=True, exist_ok=True)
    FONTS_DIR.mkdir(parents=True, exist_ok=True)

    chosen_font: Path | None = None
    for candidate in (
        FONTS_DIR / "Poppins-ExtraBold.ttf",
        FONTS_DIR / "Montserrat-Bold.ttf",
        Path("/usr/share/fonts/truetype/montserrat/Montserrat-ExtraBold.ttf"),
        Path("/usr/share/fonts/truetype/montserrat/Montserrat-Bold.ttf"),
    ):
        if candidate.exists() and candidate.stat().st_size > 10_000:
            chosen_font = candidate
            break

    if chosen_font is None:
        for url, fname in (
            (POPPINS_EXTRABOLD_URL, "Poppins-ExtraBold.ttf"),
            (MONTSERRAT_BOLD_URL, "Montserrat-Bold.ttf"),
        ):
            dest = FONTS_DIR / fname
            try:
                print(f"[Subtitle] মডার্ন ফন্ট ডাউনলোড হচ্ছে: {fname} ...", flush=True)
                urllib.request.urlretrieve(url, dest)
                if dest.exists() and dest.stat().st_size > 10_000:
                    chosen_font = dest
                    break
            except Exception as err:
                print(f"[Subtitle] ফন্ট ডাউনলোড ব্যর্থ ({fname}): {err}", flush=True)

    if chosen_font is None:
        for sys_fp in (
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
        ):
            if sys_fp.exists():
                chosen_font = sys_fp
                break

    if chosen_font is None:
        return "Arial", True

    isolated_font = fonts_dir_out / chosen_font.name
    shutil.copy2(chosen_font, isolated_font)

    family_name, style_name = ImageFont.truetype(str(isolated_font), size=24).getname()
    is_bold = any(k in style_name.lower() for k in ("bold", "black", "heavy"))
    print(f"[Subtitle] ফন্ট লোড হয়েছে: {family_name} ({style_name})", flush=True)
    return family_name, is_bold


def _extract_coarse_whisper_anchors(audio_wav: Path, script_tokens: list[str], audio_dur: float) -> list[dict]:
    """
    vad_filter=False রেখে coarse Whisper টাইমস্ট্যাম্প বের করে,
    যাতে পরের ধাপে MMS_FA ২০-২৫ সেকেন্ডের উইন্ডোতে ±২০ms নিখুঁত অ্যালাইন করতে পারে।
    """
    print(f"[Subtitle] Whisper ({SUBTITLE_MODEL}) দিয়ে প্রাথমিক সেগমেন্ট অ্যাংকর তৈরি হচ্ছে...", flush=True)
    model = WhisperModel(SUBTITLE_MODEL, device="cpu", compute_type="int8", cpu_threads=4)
    segments, _ = model.transcribe(
        str(audio_wav),
        language="en",
        word_timestamps=True,
        vad_filter=False,
        condition_on_previous_text=False,
    )

    whisper_words: list[dict] = []
    for seg in segments:
        if not getattr(seg, "words", None):
            continue
        for w in seg.words:
            txt = (w.word or "").strip()
            if txt:
                whisper_words.append({"word": txt, "start": float(w.start), "end": float(w.end)})

    if not whisper_words:
        total_chars = sum(max(1, len(t)) for t in script_tokens)
        cur = 0.0
        out = []
        for t in script_tokens:
            d = (max(1, len(t)) / total_chars) * audio_dur
            out.append({"word": t, "start": cur, "end": cur + d})
            cur += d
        return out

    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    s_norm = [norm(t) for t in script_tokens]
    w_norm = [norm(w["word"]) for w in whisper_words]
    sm = difflib.SequenceMatcher(None, s_norm, w_norm, autojunk=False)
    aligned: list[dict | None] = [None] * len(script_tokens)

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for off in range(i2 - i1):
                ww = whisper_words[j1 + off]
                aligned[i1 + off] = {
                    "word": script_tokens[i1 + off],
                    "start": float(ww["start"]),
                    "end": float(ww["end"]),
                }

    idx = 0
    n = len(script_tokens)
    while idx < n:
        if aligned[idx] is not None:
            idx += 1
            continue
        g0 = idx
        while idx < n and aligned[idx] is None:
            idx += 1
        g1 = idx
        t0 = aligned[g0 - 1]["end"] if g0 > 0 else 0.0
        t1 = aligned[g1]["start"] if g1 < n else audio_dur
        if t1 <= t0:
            t1 = min(audio_dur, t0 + 0.25 * (g1 - g0))
        chars = sum(max(1, len(t)) for t in script_tokens[g0:g1])
        cur = t0
        for k, tok in enumerate(script_tokens[g0:g1]):
            d = (max(1, len(tok)) / chars) * (t1 - t0)
            aligned[g0 + k] = {"word": tok, "start": cur, "end": cur + d}
            cur += d

    res: list[dict] = []
    prev = 0.0
    for item in aligned:
        assert item is not None
        st = max(prev, float(item["start"]))
        en = max(st + 0.05, float(item["end"]))
        res.append({"word": item["word"], "start": st, "end": en})
        prev = en
    return res


def _trim_word_silence_edges(
    mono_16k: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
) -> tuple[float, float]:
    """
    শব্দের ভেতরের শুরু বা শেষের বাড়তি নীরবতা ছেঁটে ঠিক ভয়েস অনসেটে বসাবে।
    """
    if (end_sec - start_sec) <= 0.14:
        return start_sec, end_sec
    s_idx = max(0, int(start_sec * sr))
    e_idx = min(len(mono_16k), int(end_sec * sr))
    seg = np.abs(mono_16k[s_idx:e_idx])
    if seg.size < int(sr * 0.08):
        return start_sec, end_sec

    thresh = max(0.012, float(np.percentile(seg, 90)) * 0.08)
    active = np.where(seg > thresh)[0]
    if active.size == 0:
        return start_sec, end_sec

    pad = int(sr * 0.025)
    new_s = start_sec + max(0, int(active[0]) - pad) / sr
    new_e = start_sec + min(len(seg), int(active[-1]) + pad) / sr
    if new_e - new_s >= 0.06:
        return new_s, new_e
    return start_sec, end_sec


def _forced_align_with_mms(
    audio_wav: Path,
    script_tokens: list[str],
    coarse_words: list[dict],
    audio_dur: float,
) -> list[dict]:
    """
    Meta torchaudio.pipelines.MMS_FA ব্যবহার করে ±২০ms ফোনেম-লেভেল CTC Forced Alignment।
    """
    import torch
    import torchaudio
    from torchaudio.pipelines import MMS_FA as bundle

    print("[Subtitle] Meta MMS_FA দিয়ে ±20ms ফোনেম-লেভেল Forced Alignment চলছে...", flush=True)
    raw_wav, orig_sr = sf.read(str(audio_wav), dtype="float32")
    if raw_wav.ndim > 1:
        raw_wav = np.mean(raw_wav, axis=1)

    target_sr = bundle.sample_rate  # 16000
    wav_tensor = torch.from_numpy(raw_wav).unsqueeze(0)
    if orig_sr != target_sr:
        wav_tensor = torchaudio.functional.resample(wav_tensor, orig_sr, target_sr)
    mono_16k = wav_tensor.squeeze(0).numpy()

    model = bundle.get_model().to("cpu")
    model.eval()
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()

    n = len(script_tokens)
    win_size_words = 42
    final_words: list[dict | None] = [None] * n
    prev_anchor_end = 0.0

    for w_start_idx in range(0, n, win_size_words):
        w_end_idx = min(n, w_start_idx + win_size_words)
        win_coarse = coarse_words[w_start_idx:w_end_idx]
        win_tokens = script_tokens[w_start_idx:w_end_idx]

        t_est_start = max(prev_anchor_end, win_coarse[0]["start"])
        t_est_end = max(t_est_start + 1.0, win_coarse[-1]["end"])

        aligned_window = None
        for pad_sec in (1.5, 3.5):
            slice_start = max(0.0, t_est_start - pad_sec)
            slice_end = min(audio_dur, t_est_end + pad_sec)
            s_sample = int(slice_start * target_sr)
            e_sample = int(slice_end * target_sr)
            if e_sample - s_sample < target_sr * 0.4:
                continue

            sub_wav = wav_tensor[:, s_sample:e_sample]
            per_orig = [_token_to_mms_words(tok) for tok in win_tokens]
            flat_words = ["*"] + [p for sub in per_orig for p in sub] + ["*"]

            try:
                with torch.inference_mode():
                    emission, _ = model(sub_wav)
                    token_spans = aligner(emission[0], tokenizer(flat_words))

                frame_dur = (sub_wav.shape[1] / emission.shape[1]) / float(target_sr)
                inner_spans = token_spans[1:-1]
                cursor = 0
                temp_out: list[dict | None] = []

                for orig_tok, sub in zip(win_tokens, per_orig):
                    cnt = len(sub)
                    sub_sp = inner_spans[cursor : cursor + cnt]
                    cursor += cnt
                    if sub == ["*"] or not sub_sp or not sub_sp[0] or not sub_sp[-1]:
                        temp_out.append(None)
                    else:
                        ws = slice_start + sub_sp[0][0].start * frame_dur
                        we = slice_start + sub_sp[-1][-1].end * frame_dur
                        ws, we = _trim_word_silence_edges(mono_16k, target_sr, ws, we)
                        temp_out.append({"word": orig_tok, "start": ws, "end": we})

                # কোনো "*" গ্যাপ থাকলে অনায়াসে ইন্টারপোলেট করা
                for k in range(len(temp_out)):
                    if temp_out[k] is None:
                        prev_e = temp_out[k - 1]["end"] if k > 0 and temp_out[k - 1] else t_est_start
                        next_s = (
                            temp_out[k + 1]["start"]
                            if k + 1 < len(temp_out) and temp_out[k + 1]
                            else min(audio_dur, prev_e + 0.18)
                        )
                        temp_out[k] = {
                            "word": win_tokens[k],
                            "start": prev_e,
                            "end": max(prev_e + 0.08, next_s),
                        }
                aligned_window = temp_out
                break
            except Exception:
                continue

        if aligned_window is None:
            aligned_window = win_coarse

        for off, item in enumerate(aligned_window):
            assert item is not None
            final_words[w_start_idx + off] = item

        last_valid = aligned_window[-1]
        if last_valid:
            prev_anchor_end = max(prev_anchor_end, float(last_valid["end"]) - 0.15)

    monotonic: list[dict] = []
    prev_end = 0.0
    for idx, item in enumerate(final_words):
        assert item is not None
        st = max(prev_end, float(item["start"]))
        en = max(st + 0.06, float(item["end"]))
        monotonic.append({"word": script_tokens[idx], "start": st, "end": en})
        prev_end = en

    print(f"[Subtitle] MMS_FA দিয়ে {len(monotonic)}টি শব্দ ±20ms নিখুঁতভাবে সিঙ্ক হয়েছে!", flush=True)
    return monotonic


def _group_words_into_cues(words: list[dict], max_words: int = 1, max_chars: int = 20) -> list[list[dict]]:
    if max_words <= 1:
        return [[w] for w in words]

    cues: list[list[dict]] = []
    cur: list[dict] = []
    cur_len = 0

    for w in words:
        txt = w["word"]
        would_len = cur_len + (1 if cur else 0) + len(txt)
        if cur and (len(cur) >= max_words or would_len > max_chars):
            cues.append(cur)
            cur = [w]
            cur_len = len(txt)
        else:
            cur.append(w)
            cur_len = would_len

        if txt.endswith((".", "!", "?", ";", ",")) and len(cur) >= 2:
            cues.append(cur)
            cur = []
            cur_len = 0

    if cur:
        cues.append(cur)
    return cues


def _fmt_ass_time(sec: float) -> str:
    sec = max(0.0, sec)
    cs_total = int(round(sec * 100))
    cs = cs_total % 100
    s_total = cs_total // 100
    s = s_total % 60
    m_total = s_total // 60
    m = m_total % 60
    h = m_total // 60
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _is_power_word(word_text: str) -> bool:
    clean = re.sub(r"[^a-zA-Z0-9]", "", word_text)
    return len(clean) >= 5 or word_text.endswith((".", "!", "?"))


def generate_animated_ass_subtitles(
    audio_wav: Path,
    script_text: str,
    out_ass: Path,
    fonts_dir_out: Path,
) -> Path:
    out_ass.parent.mkdir(parents=True, exist_ok=True)
    audio_dur = float(sf.info(str(audio_wav)).duration)
    script_tokens = [t for t in re.split(r"\s+", script_text.strip()) if t]

    font_family, is_bold = _prepare_subtitle_font(fonts_dir_out)

    try:
        coarse = _extract_coarse_whisper_anchors(audio_wav, script_tokens, audio_dur)
    except Exception as err:
        print(f"[Subtitle] Whisper অ্যাংকর ব্যর্থ ({err}), আনুমানিক ব্যাকআপ ব্যবহার হচ্ছে...", flush=True)
        total_chars = sum(max(1, len(t)) for t in script_tokens) or 1
        cur = 0.0
        coarse = []
        for t in script_tokens:
            d = (max(1, len(t)) / total_chars) * audio_dur
            coarse.append({"word": t, "start": cur, "end": cur + d})
            cur += d

    try:
        aligned_words = _forced_align_with_mms(audio_wav, script_tokens, coarse, audio_dur)
    except Exception as fa_err:
        print(f"[Subtitle] MMS_FA অ্যালাইনমেন্ট চালানো যায়নি ({fa_err}), Whisper অ্যাংকর ব্যবহার হচ্ছে...", flush=True)
        aligned_words = coarse

    cues = _group_words_into_cues(aligned_words, max_words=SUBTITLE_MAX_WORDS, max_chars=22)

    bold_flag = -1 if is_bold else 0
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {VIDEO_WIDTH}\n"
        f"PlayResY: {VIDEO_HEIGHT}\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font_family},{SUBTITLE_FONT_SIZE},{BASE_COLOR_ASS},{ACTIVE_COLOR_ASS},"
        f"{OUTLINE_COLOR_ASS},{SHADOW_COLOR_ASS},{bold_flag},0,0,0,100,100,1.2,0,1,7.0,3.5,2,80,80,{SUBTITLE_MARGIN_V},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    events: list[str] = []
    for idx, cue in enumerate(cues):
        st = cue[0]["start"]
        en = cue[-1]["end"]
        if idx + 1 < len(cues):
            next_st = cues[idx + 1][0]["start"]
            # পরের শব্দের ঠিক আগে পরিষ্কারভাবে শেষ করা (কখনো ওভারল্যাপ করবে না)
            en = min(en + 0.08, max(st + 0.06, next_st - 0.015))
        else:
            en = min(audio_dur, en + 0.12)

        dur_ms = max(70, int(round((en - st) * 1000)))

        # স্মুথ পপ-ইন ও পপ-আউট অ্যানিমেশন
        if dur_ms >= 220:
            in_peak = 65
            in_settle = 135
            out_start = dur_ms - 65
        else:
            in_peak = max(20, int(dur_ms * 0.30))
            in_settle = max(in_peak + 15, int(dur_ms * 0.60))
            out_start = max(in_settle + 5, int(dur_ms * 0.72))

        fade_in = min(40, in_peak)
        fade_out = min(55, max(20, dur_ms - out_start))

        line_anim = (
            f"{{\\fad({fade_in},{fade_out})\\fscx74\\fscy74"
            f"\\t(0,{in_peak},\\fscx112\\fscy112)"
            f"\\t({in_peak},{in_settle},\\fscx100\\fscy100)"
            f"\\t({out_start},{dur_ms},\\fscx86\\fscy86)}}"
        )

        if len(cue) == 1:
            txt = cue[0]["word"].replace("{", "(").replace("}", ")").replace("\\", "")
            color = ACTIVE_COLOR_ASS if _is_power_word(txt) else BASE_COLOR_ASS
            body = f"{{\\1c{color}}}{txt}"
        else:
            word_parts: list[str] = []
            for w in cue:
                t0 = max(0, int(round((w["start"] - st) * 1000)))
                t1 = max(t0 + 25, int(round((w["end"] - st) * 1000)))
                t_in_end = min(t1, t0 + 30)
                t_out_end = min(dur_ms, t1 + 30)
                safe_w = w["word"].replace("{", "(").replace("}", ")").replace("\\", "")
                word_parts.append(
                    f"{{\\1c{BASE_COLOR_ASS}\\t({t0},{t_in_end},\\1c{ACTIVE_COLOR_ASS})"
                    f"\\t({t1},{t_out_end},\\1c{BASE_COLOR_ASS})}}{safe_w}"
                )
            body = " ".join(word_parts)

        events.append(
            f"Dialogue: 0,{_fmt_ass_time(st)},{_fmt_ass_time(en)},Default,,0,0,0,,{line_anim}{body}"
        )

    out_ass.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    print(f"[Subtitle] মোট {len(events)}টি Word-by-Word অ্যানিমেটেড সাবটাইটেল তৈরি হয়েছে -> {out_ass.name}", flush=True)
    return out_ass
