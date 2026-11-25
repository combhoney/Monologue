import json
import re
import time
import requests

from src.config import load_openrouter_keys, load_openrouter_models

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """You are an elite International English YouTube Monologue Scriptwriter and YouTube SEO Growth Strategist.
You will receive a raw Hindi (or Hinglish) transcript from a YouTube video.
Your goal is to attract a global, international English-speaking audience (USA, UK, Canada, Australia, Worldwide).

TASK 1 - TRANSCREATION (NOT LITERAL TRANSLATION):
1. Understand the complete meaning, emotional arc, life philosophy, stories, metaphors, and psychological insights in the Hindi transcript (intelligently fixing Hindi ASR phonetic errors or Devanagari-transcribed English words from context).
2. Write a completely original, deeply resonant ENGLISH monologue script crafted specifically for engaging spoken audio narration.
3. Script rules:
   - Start immediately with a gripping 2-3 sentence cold-open hook (NEVER say "Hey guys", "Welcome back", or "In this video").
   - Write in vivid, emotionally compelling spoken English with varied sentence rhythm, rhetorical questions, and smooth transitions.
   - Use clean punctuation (periods, commas, paragraph breaks) for natural narration flow without excessive ellipses.
   - STRICTLY NO markdown (`**`, `*`, `#`), NO stage directions (`[pause]`, `(music)`, `Narrator:`), NO bullet lists, and NO emojis inside `script` — ONLY pure spoken words.
   - Target ~700 to 1,600 words depending on source richness.

TASK 2 - INTERNATIONAL YOUTUBE SEO & THUMBNAIL SLOGAN:
- `title`: High-CTR, curiosity-driven, SEO-optimized English YouTube title for a global audience (max 85 characters, compelling & honest).
- `thumbnail_slogan`: Ultra-punchy 2 to 4 word slogan in ALL CAPS that triggers immediate curiosity and complements the title without repeating it word-for-word (e.g., "SILENCE IS POWER", "STOP CHASING THEM", "THE QUIET TRUTH", "NEVER BEG AGAIN"). Max 4 words!
- `description`: SEO-rich 3-paragraph English YouTube description with high-intent search keywords woven naturally into the first two sentences, followed by 4-6 relevant hashtags at the bottom.
- `tags`: List of 12 to 15 high-volume English YouTube search tags.

OUTPUT FORMAT:
Return ONLY valid JSON with these exact keys:
{
  "title": "...",
  "thumbnail_slogan": "...",
  "description": "...",
  "tags": ["tag1", "tag2", "..."],
  "script": "..."
}"""


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
        return "THE QUIET TRUTH"
    return " ".join(words[:5])


class OpenRouterRotator:
    def __init__(self, api_keys: list[str], models: list[str]):
        self.api_keys = api_keys
        self.models = models
        self.dead_keys: set[str] = set()
        self.free_only_keys: set[str] = set()
        self.key_cursor: int = 0

    @staticmethod
    def _is_free_model(model_id: str) -> bool:
        return model_id.endswith(":free") or model_id == "openrouter/free"

    def chat_completion(self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 4096) -> str:
        n_keys = len(self.api_keys)
        errors: list[str] = []

        for model_idx, model in enumerate(self.models, start=1):
            is_free_model = self._is_free_model(model)
            tried_any_key = False

            for offset in range(n_keys):
                key_idx = (self.key_cursor + offset) % n_keys
                api_key = self.api_keys[key_idx]

                if api_key in self.dead_keys:
                    continue
                if not is_free_model and api_key in self.free_only_keys:
                    continue

                tried_any_key = True
                key_label = f"Key#{key_idx + 1}"
                print(f"[AI] চেষ্টা চলছে -> Model [{model_idx}/{len(self.models)}]: {model} | {key_label}", flush=True)

                try:
                    resp = requests.post(
                        OPENROUTER_API_URL,
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                            "HTTP-Referer": "https://github.com/youtube-monologue-automation",
                            "X-Title": "YouTube Monologue Automation",
                        },
                        json={
                            "model": model,
                            "messages": messages,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                        },
                        timeout=120,
                    )
                except requests.RequestException as req_err:
                    msg = f"{model} ({key_label}) নেটওয়ার্ক এরর: {req_err}"
                    print(f"[AI] {msg}", flush=True)
                    errors.append(msg)
                    time.sleep(1.5)
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
                    content = (choices[0].get("message") or {}).get("content", "") if choices else ""
                    if not content or not content.strip():
                        errors.append(f"{model} ({key_label}) খালি রেসপন্স")
                        continue

                    self.key_cursor = (key_idx + 1) % n_keys
                    print(f"[AI] সফল হয়েছে -> {model} ({key_label})", flush=True)
                    return content

                err_snippet = resp.text[:250].replace("\n", " ")
                if status in (401, 403):
                    print(f"[AI] {key_label} বাতিল/ভুল ({status})। বাদ দেওয়া হলো।", flush=True)
                    self.dead_keys.add(api_key)
                elif status == 402:
                    print(f"[AI] {key_label}-এ পেইড ক্রেডিট নেই (402)। ফ্রি মডেলের জন্য রাখা হলো।", flush=True)
                    self.free_only_keys.add(api_key)
                elif status == 429:
                    print(f"[AI] {key_label} রেট-লিমিট (429)। পরের Key চেষ্টা হচ্ছে...", flush=True)
                    time.sleep(1.0)
                elif status == 404:
                    print(f"[AI] মডেল {model} পাওয়া যায়নি (404)। পরের মডেলে যাওয়া হচ্ছে...", flush=True)
                    errors.append(f"{model}: 404 Not Found")
                    break
                else:
                    print(f"[AI] HTTP {status} ({model} | {key_label}): {err_snippet}", flush=True)
                    time.sleep(1.0)

                errors.append(f"{model} ({key_label}) -> HTTP {status}: {err_snippet}")

            if not tried_any_key:
                continue

        raise RuntimeError(
            "সব OpenRouter মডেল ও API Key চেষ্টা করেও স্ক্রিপ্ট জেনারেট করা যায়নি!\n"
            + "\n".join(errors[-10:])
        )


def generate_monologue_script(hindi_transcript: str, video_id: str) -> dict:
    api_keys = load_openrouter_keys()
    models = load_openrouter_models()
    rotator = OpenRouterRotator(api_keys=api_keys, models=models)

    max_chars = 90000
    if len(hindi_transcript) > max_chars:
        hindi_transcript = hindi_transcript[:max_chars]

    user_prompt = (
        f"Source Hindi Video ID: {video_id}\n\n"
        f"Hindi Transcript:\n\"\"\"\n{hindi_transcript}\n\"\"\"\n\n"
        "Transcreate this directly into a captivating international English monologue script "
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
                title = "The Silent Mindset Shift That Changes Everything"
            if not description:
                description = f"{title}\n\n#monologue #mindset #philosophy #selfimprovement"
            if not tags:
                tags = ["mindset", "philosophy", "monologue", "self improvement", "life lessons", "wisdom"]

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
