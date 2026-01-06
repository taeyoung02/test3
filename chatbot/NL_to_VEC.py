import json
import os
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv("chatbot/.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DATA_DIR = Path("chatbot/car_data")


def iter_cardata_files(directory: Path) -> Iterable[Path]:
    """car_info*.json 중 *_vector.json 파일은 제외하고 순회."""
    for path in sorted(directory.glob("car_info*.json")):
        if path.stem.endswith("_vector"):
            continue
        yield path


def embed_file(path: Path) -> None:
    print(f"\nProcessing {path.name}")
    try:
        cardata = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        print(f"  ✗ JSON decode failed: {exc}. Skipping this file.")
        return

    for idx, item in enumerate(cardata):
        text = item.get("payload", {}).get("nl", "")
        if not text:
            continue
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        item["vector"] = resp.data[0].embedding
        print(f"  → Processed {idx + 1} / {len(cardata)}")

    output = path.with_name(f"{path.stem}_vector.json")
    output.write_text(json.dumps(cardata, ensure_ascii=False, indent=4))
    print(f"  ✓ Saved {output.name}")


def main() -> None:
    for file_path in iter_cardata_files(DATA_DIR):
        embed_file(file_path)


if __name__ == "__main__":
    main()