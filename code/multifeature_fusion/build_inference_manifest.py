from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "data/inference/images"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data/inference/manifest.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a JSONL manifest for inference images."
    )
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.images_dir.is_dir():
        raise SystemExit(f"Missing images directory: {args.images_dir}")
    image_paths = sorted(
        path
        for path in args.images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    )
    if not image_paths:
        raise SystemExit(f"No image files found in {args.images_dir}")
    sample_ids = [path.stem for path in image_paths]
    duplicate_ids = sorted(
        sample_id
        for sample_id, count in Counter(sample_ids).items()
        if count > 1
    )
    if duplicate_ids:
        raise SystemExit(
            "Multiple images map to the same sample_id stem: "
            + ", ".join(duplicate_ids[:10])
        )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for path in image_paths:
            sample_id = path.stem
            row = {
                "sample_id": sample_id,
                "image_id": sample_id,
                "image_path": str(path).replace("\\", "/"),
                "local_image_path": str(path).replace("\\", "/"),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(image_paths), "output": str(args.output_jsonl)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
