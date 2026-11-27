import json
import os
import re
import time
import requests

from src.config import load_openrouter_keys, load_openrouter_models

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

NON_CHAT_KEYWORDS = (
    "embed",
    "rerank",
    "lyria",
    "recraft",
    "decider",
    "decision",
    "span-",
    "content-safety",
    "guard",
    "whisper",
)

SYSTEM_PROMPT = """You are a skilled International English YouTube Scriptwriter.
You will receive a raw Hindi (or Hinglish) transcript from a YouTube video.
Your goal is to turn it into an engaging, natural-sounding English YouTube video script for a global audience.

TASK 1 - NATURAL ADAPTATION (NOT LITERAL TRANSLATION, NOT DRAMATIC MONOLOGUE):
1. Read and understand the full message, ideas, stories, examples, and context of the Hindi transcript (fixing any Hindi ASR phonetic errors or Devanagari-transcribed English words naturally).
2. Rewrite it into a completely original ENGLISH script that sounds like a real, relatable human talking naturally to their viewers on YouTube.
3. Script rules:
   - Keep the tone natural, conversational, clear, and authentic—avoid sounding theatrical, overly dramatic, preachy, or like a poetic monologue.
   - Start with an engaging, natural opening that introduces the topic smoothly (never use clichés like "Hey guys, welcome back to my channel").
   - Use everyday, easy-to-follow spoken English with natural transitions and smooth sentence flow.
   - Use normal punctuation (periods, commas, paragraph breaks) so Text-to-Speech sounds completely natural and human.
   - STRICTLY NO markdown (`**`, `*`, `#`), NO stage directions (`[pause]`, `(music)`, `Narrator:`), NO bullet lists, and NO emojis inside `script` — ONLY pure spoken words.
   - Keep the length proportional to the source content (~700 to 1,600 words depending on depth), without unnecessary fluff.

TASK 2 - INTERNATIONAL YOUTUBE SEO & THUMBNAIL SLOGAN:
- `title`: High-CTR, curiosity-driven, SEO-optimized English YouTube title for a global audience (max 85 characters, compelling & honest).
- `thumbnail_slogan`: Ultra-punchy 2 to 4 word slogan in ALL CAPS that triggers curiosity and complements the title without repeating it word-for-word. Max 4 words!
- `description`: SEO-rich 2-3 paragraph English YouTube description with relevant search keywords woven naturally into the opening lines, followed by 4-6 relevant hashtags at the bottom.
- `tags`: List of 12 to 15 relevant high-volume English YouTube search tags matching the video's actual topic.

OUTPUT FORMAT:
Return ONLY valid JSON with these exact keys:
{
  "title": "...",
  "thumbnail_slogan": "...",
  "description": "...",
  "tags": ["tag1", "tag2", "..."],
  "script": "..."
}"""


def _fetch_live_free_models() -> list[str]:
    """
    OpenRouter-এর পাবলিক API থেকে বর্তমানে সচল থাকা সব :free চ্যাট মডেল ডাইনামিকভাবে নিয়ে আসে,
    যাতে কোনো পুরোনো মডেল বন্ধ হলেও 404 এরর না আসে।
    """
    try:
        resp = requests.get(OPENROUTER_MODELS_URL, timeout=15)
        if resp.status_code != 200:
            return []
        data = resp.json().get("data") or []
        live_free: list[str] = []
        for item in data:
            mid = str(item.get("id", "")).strip()
            if not mid.endswith(":free"):
                continue
            low = mid.lower()
            if any(bad in low for bad in NON_CHAT_KEYWORDS):
                continue
            arch = item.get("architecture") or {}
            out_mods = arch.get("output_modalities") or ["text"]
            if "text" not in out_mods:
                continue
            live_free.append(mid)
        return live_free
    except Exception:
        return []


def _resolve_model_list(base_models: list[str]) -> list[str]:
    if os.getenv("OPENROUTER_MODELS", "").strip():
        return base_models

    live_free = _fetch_live_free_models()
    if not live_free:
        return base_models

    live_free_set = set(live_free)
    resolved: list[str] = []

    for m in base_models:
        if m == "openrouter/free":
            continue
        if m.endswith(":free"):
            if m in live_free_set and m not in resolved:
                resolved.append(m)
        else:
            if m not in resolved:
                resolved.append(m)

    # ক্যাটালগে থাকা অন্যান্য নতুন সচল :free মডেলও ব্যাকআপ হিসেবে যোগ করা
    for lm in live_free:
        if lm not in resolved:
            resolved.append(lm)

    resolved.append("openrouter/free")
    print(f"[AI] লাইভ ক্যাটালগ থেকে {len(live_free)}টি সচল :free মডেল পাওয়া গেছে।", flush=True)
    return resolved


def _extract_json_object(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned, re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("AI রেসপন্সে সঠিক JSON অবজেক্ট পাওয়া যায়নি।")

    data = json.loads(cleaned[start : end + 1], strict=False)
    if not isinstance(data, dict):
        raise ValueError("JSON আউটপুট dict ফরম্যাটে নয়।")
    return data


def _sanitize_tts_script(script: str) -> str:
    script = re.sub(r"\[[^\]]{1,40}\]", " ", script)
    script = re.sub(r"\((?:pause|music|sigh|breath)[^)]*\)", " ", script, flags=re.IGNORECASE)
    script = script.replace("**", "").replace("__", "").replace("*", "")
    script = re.sub(r"^#+\s*", "", script, flags=re.MULTILINE)
    script = re.sub(r"[ \t]+", " ", script)
    script = re.sub(r"\n{3,}", "\n\n", script)
    return script.strip()


def _sanitize_slogan(slogan: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9'\-\s]", "", slogan).strip().upper()
    words = [w for w in cleaned.split() if w]
    if not words:
        return "THE REAL TRUTH"
    return " ".join(words[:5])


class OpenRouterRotator:
    def __init__(self, api_keys: list[str], models: list[str]):
        self.api_keys = api_keys
        self.models = models
        self.dead_keys: set[str] = set()
        self.zero_balance_keys: set[str] = set()
        self.key_cursor: int = 0

    @staticmethod
    def _is_free_model(model_id: str) -> bool:
        return model_id.endswith(":free") or model_id == "openrouter/free"

    def _post_chat(self, api_key: str, model: str, messages: list[dict], temperature: float, max_tokens: int):
        return requests.post(
            OPENROUTER_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/youtube-automation",
                "X-Title": "YouTube Content Automation",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=150,
        )

    def chat_completion(self, messages: list[dict], temperature: float = 0.7) -> str:
        n_keys = len(self.api_keys)
        errors: list[str] = []

        for model_idx, model in enumerate(self.models, start=1):
            is_free_model = self._is_free_model(model)
            # ফ্রি রিজনিং মডেল যেন thinking করতেই টোকেন শেষ না করে, তাই 10000 টোকেন দেওয়া হলো
            target_max_tokens = 10000 if is_free_model else 4096

            for offset in range(n_keys):
                key_idx = (self.key_cursor + offset) % n_keys
                api_key = self.api_keys[key_idx]

                if api_key in self.dead_keys:
                    continue
                if not is_free_model and api_key in self.zero_balance_keys:
                    continue

                key_label = f"Key#{key_idx + 1}"
                print(f"[AI] চেষ্টা চলছে -> Model [{model_idx}/{len(self.models)}]: {model} | {key_label}", flush=True)

                try:
                    resp = self._post_chat(api_key, model, messages, temperature, target_max_tokens)
                except requests.RequestException as req_err:
                    msg = f"{model} ({key_label}) নেটওয়ার্ক এরর: {req_err}"
                    print(f"[AI] {msg}", flush=True)
                    errors.append(msg)
                    time.sleep(1.5)
                    continue

                # যদি 402-এ বলে যে কম max_tokens দিয়ে চলবে (যেমন 2666 afford করতে পারে), সাথে সাথে কম টোকেনে রিট্রাই
                if resp.status_code == 402:
                    m_afford = re.search(r"can only afford\s+(\d+)", resp.text, re.IGNORECASE)
                    afford = int(m_afford.group(1)) if m_afford else 0
                    if afford >= 1800:
                        reduced_tokens = min(afford - 120, 3200)
                        print(f"[AI] {key_label} ({model}) max_tokens={reduced_tokens} দিয়ে পুনরায় চেষ্টা করছে...", flush=True)
                        try:
                            resp = self._post_chat(api_key, model, messages, temperature, reduced_tokens)
                        except requests.RequestException as req_err:
                            errors.append(f"{model} ({key_label}) retry error: {req_err}")
                            continue

                status = resp.status_code
                if status == 200:
                    try:
                        payload = resp.json()
                    except ValueError:
                        errors.append(f"{model} ({key_label}) invalid JSON")
                        continue

                    if "error" in payload:
                        errors.append(f"{model} ({key_label}) API error: {payload['error']}")
                        continue

                    choices = payload.get("choices") or []
                    msg_obj = (choices[0].get("message") or {}) if choices else {}
                    content = (msg_obj.get("content") or "").strip()
                    finish_reason = choices[0].get("finish_reason", "") if choices else ""

                    if not content:
                        errors.append(f"{model} ({key_label}) খালি রেসপন্স (finish_reason={finish_reason})")
                        print(f"[AI] {model} ({key_label}) খালি রেসপন্স দিয়েছে (finish_reason={finish_reason})।", flush=True)
                        continue

                    self.key_cursor = (key_idx + 1) % n_keys
                    print(f"[AI] সফল হয়েছে -> {model} ({key_label})", flush=True)
                    return content

                err_snippet = resp.text[:250].replace("\n", " ")
                if status in (401, 403):
                    print(f"[AI] {key_label} বাতিল/ভুল ({status})। বাদ দেওয়া হলো।", flush=True)
                    self.dead_keys.add(api_key)
                elif status == 402:
                    m_afford = re.search(r"can only afford\s+(\d+)", resp.text, re.IGNORECASE)
                    afford = int(m_afford.group(1)) if m_afford else -1
                    if afford == 0 or "never purchased" in resp.text.lower():
                        print(f"[AI] {key_label}-এ কোনো পেইড ব্যালেন্স নেই (0)। ফ্রি মডেলের জন্য রাখা হলো।", flush=True)
                        self.zero_balance_keys.add(api_key)
                    else:
                        print(f"[AI] {key_label}-এ {model} চালানোর মতো ক্রেডিট কম, পরের সস্তা মডেল দেখা হচ্ছে...", flush=True)
                elif status == 429:
                    print(f"[AI] {key_label} রেট-লিমিট (429)। পরের Key/মডেল চেষ্টা হচ্ছে...", flush=True)
                    time.sleep(1.0)
                elif status == 404:
                    print(f"[AI] মডেল {model} পাওয়া যায়নি (404)। পরের মডেলে যাওয়া হচ্ছে...", flush=True)
                    errors.append(f"{model}: 404 Not Found")
                    break
                else:
                    print(f"[AI] HTTP {status} ({model} | {key_label}): {err_snippet}", flush=True)
                    time.sleep(1.0)

                errors.append(f"{model} ({key_label}) -> HTTP {status}: {err_snippet}")

        raise RuntimeError(
            "সব OpenRouter মডেল ও API Key চেষ্টা করেও স্ক্রিপ্ট জেনারেট করা যায়নি!\n"
            + "\n".join(errors[-12:])
        )


def generate_monologue_script(hindi_transcript: str, video_id: str) -> dict:
    api_keys = load_openrouter_keys()
    models = _resolve_model_list(load_openrouter_models())
    rotator = OpenRouterRotator(api_keys=api_keys, models=models)

    max_chars = 90000
    if len(hindi_transcript) > max_chars:
        hindi_transcript = hindi_transcript[:max_chars]

    user_prompt = (
        f"Source Hindi Video ID: {video_id}\n\n"
        f"Hindi Transcript:\n\"\"\"\n{hindi_transcript}\n\"\"\"\n\n"
        "Turn this into a natural, conversational English YouTube script "
        "with high-CTR YouTube SEO title, description, tags, and a punchy 2-4 word thumbnail_slogan. "
        "Return ONLY valid JSON."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    last_err = None
    for attempt in range(1, 3):
        raw = rotator.chat_completion(messages=messages)
        try:
            parsed = _extract_json_object(raw)
            title = str(parsed.get("title", "")).strip()
            thumbnail_slogan = _sanitize_slogan(str(parsed.get("thumbnail_slogan", "")))
            description = str(parsed.get("description", "")).strip()
            tags = parsed.get("tags") or []
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]
            tags = [str(t).strip() for t in tags if str(t).strip()][:15]

            script = _sanitize_tts_script(str(parsed.get("script", "")))
            if len(script.split()) < 40:
                raise ValueError("জেনারেটেড স্ক্রিপ্ট খুব ছোট হয়েছে।")

            if not title:
                title = "What Nobody Tells You About Moving On"
            if not description:
                description = f"{title}\n\n#selfimprovement #mindset #relationships #advice"
            if not tags:
                tags = ["self improvement", "mindset", "life advice", "storytelling", "psychology", "wisdom"]

            return {
                "title": title[:95],
                "thumbnail_slogan": thumbnail_slogan,
                "description": description[:4900],
                "tags": tags,
                "script": script,
            }
        except Exception as parse_err:
            last_err = parse_err
            print(f"[AI] JSON পার্স করতে সমস্যা (চেষ্টা {attempt}/2): {parse_err}", flush=True)

    raise RuntimeError(f"AI আউটপুট পার্স করা যায়নি: {last_err}")
