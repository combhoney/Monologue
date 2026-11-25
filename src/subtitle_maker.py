import difflib
import re
import shutil
import urllib.request
from pathlib import Path

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
# - বেস লেখা: ক্রিস্প স্নো হোয়াইট (#F8FAFC)
# - উচ্চারিত শব্দ হাইলাইট: ওয়ার্ম গোল্ড-অ্যাম্বার (#FFD166)
# - স্ট্রোক / বর্ডার: ডিপ অবসিডিয়ান চারকোল (#0B0F17)
BASE_COLOR_ASS = "&H00FCFAF8&"
ACTIVE_COLOR_ASS = "&H0066D1FF&"
OUTLINE_COLOR_ASS = "&H00170F0B&"
SHADOW_COLOR_ASS = "&H96000000&"


def _prepare_subtitle_font(fonts_dir_out: Path) -> tuple[str, bool]:
    fonts_dir_out.mkdir(parents=True, exist_ok=True)
    FONTS_DIR.mkdir(parents=True, exist_ok=True)

    chosen_font: Path | None = None

    # ১. আগে থেকেই caching করা Poppins-ExtraBold.ttf বা অন্য কাস্টম ফন্ট খোঁজা
    for candidate in (
        FONTS_DIR / "Poppins-ExtraBold.ttf",
        FONTS_DIR / "Montserrat-Bold.ttf",
        Path("/usr/share/fonts/truetype/montserrat/Montserrat-ExtraBold.ttf"),
        Path("/usr/share/fonts/truetype/montserrat/Montserrat-Bold.ttf"),
    ):
        if candidate.exists() and candidate.stat().st_size > 10_000:
            chosen_font = candidate
            break

    # ২. না থাকলে Google Fonts থেকে Poppins-ExtraBold.ttf ডাউনলোড করা
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


def _extract_whisper_words(audio_wav: Path, script_text: str) -> list[dict]:
    print(f"[Subtitle] faster-whisper ({SUBTITLE_MODEL}) দিয়ে প্রতিটি শব্দের টাইমস্ট্যাম্প বের হচ্ছে...", flush=True)
    model = WhisperModel(SUBTITLE_MODEL, device="cpu", compute_type="int8", cpu_threads=4)
    segments, _ = model.transcribe(
        str(audio_wav),
        language="en",
        word_timestamps=True,
        initial_prompt=script_text[:400],
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 350},
    )

    words: list[dict] = []
    for seg in segments:
        if not getattr(seg, "words", None):
            continue
        for w in seg.words:
            txt = (w.word or "").strip()
            if txt:
                words.append({"word": txt, "start": float(w.start), "end": float(w.end)})
    print(f"[Subtitle] Whisper থেকে {len(words)}টি শব্দের টাইমস্ট্যাম্প পাওয়া গেছে।", flush=True)
    return words


def _align_words_to_script(script: str, whisper_words: list[dict], total_duration: float) -> list[dict]:
    script_tokens = [t for t in re.split(r"\s+", script.strip()) if t]
    if not script_tokens:
        return []

    if not whisper_words:
        total_chars = sum(max(1, len(t)) for t in script_tokens)
        cur = 0.0
        fallback = []
        for t in script_tokens:
            dur = (max(1, len(t)) / total_chars) * total_duration
            fallback.append({"word": t, "start": cur, "end": cur + dur})
            cur += dur
        return fallback

    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    s_norm = [norm(t) for t in script_tokens]
    w_norm = [norm(w["word"]) for w in whisper_words]

    sm = difflib.SequenceMatcher(None, s_norm, w_norm, autojunk=False)
    aligned: list[dict | None] = [None] * len(script_tokens)

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                ww = whisper_words[j1 + offset]
                aligned[i1 + offset] = {
                    "word": script_tokens[i1 + offset],
                    "start": float(ww["start"]),
                    "end": float(ww["end"]),
                }

    idx = 0
    n = len(script_tokens)
    while idx < n:
        if aligned[idx] is not None:
            idx += 1
            continue
        gap_start = idx
        while idx < n and aligned[idx] is None:
            idx += 1
        gap_end = idx

        t0 = aligned[gap_start - 1]["end"] if gap_start > 0 else 0.0
        t1 = aligned[gap_end]["start"] if gap_end < n else total_duration
        if t1 <= t0:
            t1 = min(total_duration, t0 + 0.25 * (gap_end - gap_start))

        span_tokens = script_tokens[gap_start:gap_end]
        chars = sum(max(1, len(t)) for t in span_tokens)
        cur = t0
        for k, tok in enumerate(span_tokens):
            d = (max(1, len(tok)) / chars) * (t1 - t0)
            aligned[gap_start + k] = {"word": tok, "start": cur, "end": cur + d}
            cur += d

    result: list[dict] = []
    prev_end = 0.0
    for item in aligned:
        assert item is not None
        st = max(prev_end, float(item["start"]))
        en = max(st + 0.06, float(item["end"]))
        result.append({"word": item["word"], "start": st, "end": en})
        prev_end = en
    return result


def _group_words_into_cues(words: list[dict], max_words: int = 3, max_chars: int = 22) -> list[list[dict]]:
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


def generate_animated_ass_subtitles(
    audio_wav: Path,
    script_text: str,
    out_ass: Path,
    fonts_dir_out: Path,
) -> Path:
    out_ass.parent.mkdir(parents=True, exist_ok=True)
    audio_dur = float(sf.info(str(audio_wav)).duration)

    font_family, is_bold = _prepare_subtitle_font(fonts_dir_out)

    try:
        whisper_words = _extract_whisper_words(audio_wav, script_text)
    except Exception as err:
        print(f"[Subtitle] Whisper টাইমস্ট্যাম্প ব্যর্থ ({err}), আনুমানিক টাইমিং ব্যবহার হচ্ছে...", flush=True)
        whisper_words = []

    aligned_words = _align_words_to_script(script_text, whisper_words, audio_dur)
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
        f"{OUTLINE_COLOR_ASS},{SHADOW_COLOR_ASS},{bold_flag},0,0,0,100,100,1.0,0,1,5.5,3.0,2,80,80,{SUBTITLE_MARGIN_V},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    events: list[str] = []
    for idx, cue in enumerate(cues):
        st = cue[0]["start"]
        en = cue[-1]["end"]
        if idx + 1 < len(cues):
            next_st = cues[idx + 1][0]["start"]
            en = min(en + 0.12, max(en, next_st - 0.02))
        else:
            en = min(audio_dur, en + 0.15)

        dur_ms = max(120, int(round((en - st) * 1000)))

        # স্মুথ পপ-ইন ও পপ-আউট অ্যানিমেশন
        if dur_ms >= 260:
            in_peak = 85
            in_settle = 170
            out_start = dur_ms - 80
        else:
            in_peak = max(30, int(dur_ms * 0.32))
            in_settle = max(in_peak + 20, int(dur_ms * 0.62))
            out_start = max(in_settle + 10, int(dur_ms * 0.72))

        fade_in = min(65, in_peak)
        fade_out = min(75, max(30, dur_ms - out_start))

        line_anim = (
            f"{{\\fad({fade_in},{fade_out})\\fscx78\\fscy78"
            f"\\t(0,{in_peak},\\fscx108\\fscy108)"
            f"\\t({in_peak},{in_settle},\\fscx100\\fscy100)"
            f"\\t({out_start},{dur_ms},\\fscx88\\fscy88)}}"
        )

        # প্রতিটি শব্দ বলার মুহূর্তে ওয়ার্ম গোল্ড-অ্যাম্বার হাইলাইট
        word_parts: list[str] = []
        for w in cue:
            t0 = max(0, int(round((w["start"] - st) * 1000)))
            t1 = max(t0 + 25, int(round((w["end"] - st) * 1000)))
            t_in_end = min(t1, t0 + 35)
            t_out_end = min(dur_ms, t1 + 35)
            safe_w = w["word"].replace("{", "(").replace("}", ")").replace("\\", "")
            word_parts.append(
                f"{{\\1c{BASE_COLOR_ASS}\\t({t0},{t_in_end},\\1c{ACTIVE_COLOR_ASS})"
                f"\\t({t1},{t_out_end},\\1c{BASE_COLOR_ASS})}}{safe_w}"
            )

        events.append(
            f"Dialogue: 0,{_fmt_ass_time(st)},{_fmt_ass_time(en)},Default,,0,0,0,,{line_anim}{' '.join(word_parts)}"
        )

    out_ass.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    print(f"[Subtitle] মোট {len(events)}টি অ্যানিমেটেড পপ-আপ সাবটাইটেল তৈরি হয়েছে -> {out_ass.name}", flush=True)
    return out_ass