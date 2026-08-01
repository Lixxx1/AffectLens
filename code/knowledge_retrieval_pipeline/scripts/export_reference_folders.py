from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one prepared reference folder per inference sample."
    )
    parser.add_argument("--index-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=10)
    return parser.parse_args()


def safe_name(value: object) -> str:
    text = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", str(value or "Unknown"))
    return text[:80]


def validate_sample_id(value: object) -> str:
    sample_id = str(value or "").strip()
    if (
        not sample_id
        or sample_id in {".", ".."}
        or "/" in sample_id
        or "\\" in sample_id
    ):
        raise SystemExit(f"Unsafe or empty sample_id: {sample_id!r}")
    return sample_id


def image_path(row: dict) -> str | None:
    return row.get("local_image_path") or row.get("image_path")


def extension(path: object) -> str:
    suffix = Path(str(path or "")).suffix
    return suffix or ".jpg"


def copy_image(source: object, destination: Path) -> bool:
    if not source:
        return False
    source_path = Path(str(source))
    if not source_path.is_file():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    return True


def clear_generated_files(folder: Path) -> None:
    for path in folder.iterdir():
        if not path.is_file():
            continue
        if (
            path.name in {"annotation.json", "metadata.json"}
            or path.name.startswith("query_")
            or re.match(r"^rank\d{2}_", path.name)
        ):
            path.unlink()


def main() -> None:
    args = parse_args()
    if args.max_images < 1:
        raise SystemExit("--max-images must be at least 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    folders_written = 0
    missing_query = 0
    missing_retrieved = 0

    with args.index_jsonl.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise SystemExit(f"No rows found in {args.index_jsonl}")

    seen_sample_ids: set[str] = set()
    validated_rows: list[tuple[dict, str, object, list[dict]]] = []
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise SystemExit(f"{args.index_jsonl}:{row_number} is not a JSON object")
        sample_id = validate_sample_id(row.get("sample_id"))
        if sample_id in seen_sample_ids:
            raise SystemExit(
                f"{args.index_jsonl}:{row_number} duplicates sample_id {sample_id!r}"
            )
        seen_sample_ids.add(sample_id)
        query = row.get("query_row") or {}
        query_source = image_path(query)
        if not query_source or not Path(str(query_source)).is_file():
            raise SystemExit(
                f"Missing query image for {sample_id}: {query_source!r}"
            )
        top_k = row.get("top_k")
        if not isinstance(top_k, list):
            raise SystemExit(
                f"{args.index_jsonl}:{row_number} has no top_k list"
            )
        selected = top_k[: args.max_images]
        for rank, item in enumerate(selected, start=1):
            if not isinstance(item, dict):
                raise SystemExit(
                    f"{args.index_jsonl}:{row_number} top_k rank {rank} is not an object"
                )
            source = image_path(item)
            if not source or not Path(str(source)).is_file():
                raise SystemExit(
                    f"Missing retrieved image for {sample_id} rank {rank}: {source!r}"
                )
        validated_rows.append((row, sample_id, query_source, selected))

    for row, sample_id, query_source, selected in validated_rows:
        folder = args.output_dir / sample_id
        folder.mkdir(parents=True, exist_ok=True)
        clear_generated_files(folder)

        if not copy_image(
            query_source,
            folder / f"query_{sample_id}{extension(query_source)}",
        ):
            missing_query += 1

        exported = []
        for output_rank, item in enumerate(selected, start=1):
            source = image_path(item)
            score = float(item.get("score", 0.0))
            emotion = safe_name(item.get("dominant_emotion"))
            vote_count = (item.get("consensus") or {}).get("vote_count")
            vote_text = f"_vote{vote_count}" if vote_count else ""
            filename = (
                f"rank{output_rank:02d}{vote_text}_cos_{score:.4f}_"
                f"{emotion}{extension(source)}"
            )
            copied = copy_image(source, folder / filename)
            if not copied:
                missing_retrieved += 1
            exported_item = dict(item)
            if item.get("rank") != output_rank:
                exported_item["retrieval_rank"] = item.get("rank")
            exported_item["rank"] = output_rank
            exported_item["file"] = filename
            exported_item["copied"] = copied
            exported.append(exported_item)

        metadata = dict(row)
        metadata["top_k"] = exported
        rendered = json.dumps(metadata, ensure_ascii=False, indent=2)
        (folder / "annotation.json").write_text(rendered, encoding="utf-8")
        (folder / "metadata.json").write_text(rendered, encoding="utf-8")
        folders_written += 1

    summary = {
        "index_jsonl": str(args.index_jsonl),
        "output_dir": str(args.output_dir),
        "folders_written": folders_written,
        "missing_query_images": missing_query,
        "missing_retrieved_images": missing_retrieved,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
