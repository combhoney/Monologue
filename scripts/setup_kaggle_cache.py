import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.tts_engine import ensure_kaggle_model_cache


def main():
    cache_id = ensure_kaggle_model_cache()
    print(f"সেটআপ সম্পন্ন! স্থায়ী ক্যাশ কার্নেল: {cache_id}")


if __name__ == "__main__":
    main()
