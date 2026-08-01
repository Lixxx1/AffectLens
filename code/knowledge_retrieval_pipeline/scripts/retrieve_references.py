from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve top-k reference artworks with aligned NPZ features."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--query-features", type=Path, required=True)
    parser.add_argument("--pool-features", type=Path, required=True)
    parser.add_argument("--query-jsonl", type=Path, required=True)
    parser.add_argument("--pool-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=32)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_id(value: object) -> str:
    return str(value or "").replace("\\", "/").strip()


def build_row_index(rows: list[dict]) -> dict[str, dict]:
    index: dict[str, dict] = {}
    ambiguous: set[str] = set()
    for row in rows:
        for value in (
            row.get("image_id"),
            row.get("image_path"),
            row.get("local_image_path"),
            row.get("sample_id"),
            row.get("request_id"),
        ):
            key = normalize_id(value)
            if key:
                for alias in (key, Path(key).stem):
                    if not alias or alias in ambiguous:
                        continue
                    existing = index.get(alias)
                    if existing is not None and existing is not row:
                        index.pop(alias, None)
                        ambiguous.add(alias)
                    else:
                        index[alias] = row
    return index


def find_row(index: dict[str, dict], image_id: str) -> dict | None:
    image_id = normalize_id(image_id)
    return index.get(image_id) or index.get(Path(image_id).stem)


def canonical_sample_id(row: dict, feature_id: str) -> str:
    explicit = row.get("sample_id")
    if explicit is not None and str(explicit).strip():
        sample_id = str(explicit).strip()
    else:
        fallback = (
            row.get("image_id")
            or row.get("image_path")
            or row.get("local_image_path")
            or feature_id
        )
        sample_id = Path(normalize_id(fallback)).stem
    if (
        not sample_id
        or sample_id in {".", ".."}
        or "/" in sample_id
        or "\\" in sample_id
    ):
        raise SystemExit(f"Unsafe or empty sample_id {sample_id!r} for feature {feature_id!r}")
    return sample_id


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if "x" not in data.files or "image_ids" not in data.files:
        raise KeyError(f"{path} must contain x and image_ids; found {data.files}")

    features = data["x"].astype(np.float32)
    image_ids = np.asarray(data["image_ids"]).astype(str)
    if len(features) != len(image_ids):
        raise ValueError(f"{path}: {len(features)} features != {len(image_ids)} ids")

    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    return features, image_ids


def extract_labels(row: dict | None) -> tuple[object, object, object]:
    row = row or {}
    third = ((row.get("description") or {}).get("third_section") or {})
    emotion = row.get("dominant_emotion") or third.get("dominant_emotion")
    valence = (
        row.get("valence")
        or row.get("emotional_valence")
        or third.get("emotional_valence")
    )
    arousal = (
        row.get("arousal")
        or row.get("emotional_arousal_level")
        or third.get("emotional_arousal_level")
    )
    return emotion, valence, arousal


def describe(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float32)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "max": float(array.max()),
    }


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise SystemExit("--top-k must be at least 1")
    if args.chunk_size < 1:
        raise SystemExit("--chunk-size must be at least 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    query_x, query_ids = load_features(args.query_features)
    pool_x, pool_ids = load_features(args.pool_features)
    if len(query_x) == 0:
        raise SystemExit(f"No query features found in {args.query_features}")
    if len(pool_x) == 0:
        raise SystemExit(f"No pool features found in {args.pool_features}")
    if query_x.shape[1] != pool_x.shape[1]:
        raise ValueError(
            f"Feature dimensions differ: query={query_x.shape[1]}, pool={pool_x.shape[1]}"
        )

    query_index = build_row_index(read_jsonl(args.query_jsonl))
    pool_index = build_row_index(read_jsonl(args.pool_jsonl))
    resolved_query_rows = [
        find_row(query_index, normalize_id(image_id)) for image_id in query_ids
    ]
    resolved_pool_rows = [
        find_row(pool_index, normalize_id(image_id)) for image_id in pool_ids
    ]
    missing_query_ids = [
        normalize_id(image_id)
        for image_id, row in zip(query_ids, resolved_query_rows)
        if row is None
    ]
    missing_pool_ids = [
        normalize_id(image_id)
        for image_id, row in zip(pool_ids, resolved_pool_rows)
        if row is None
    ]
    if missing_query_ids:
        raise SystemExit(
            f"Could not uniquely align {len(missing_query_ids)} query feature IDs "
            f"to {args.query_jsonl}. First missing/ambiguous IDs: {missing_query_ids[:10]}"
        )
    if missing_pool_ids:
        raise SystemExit(
            f"Could not uniquely align {len(missing_pool_ids)} pool feature IDs "
            f"to {args.pool_jsonl}. First missing/ambiguous IDs: {missing_pool_ids[:10]}"
        )

    sample_ids = [
        canonical_sample_id(row, normalize_id(image_id))
        for image_id, row in zip(query_ids, resolved_query_rows)
        if row is not None
    ]
    duplicate_sample_ids = sorted(
        sample_id
        for sample_id, count in Counter(sample_ids).items()
        if count > 1
    )
    if duplicate_sample_ids:
        raise SystemExit(
            f"Duplicate canonical sample_id values: {duplicate_sample_ids[:10]}"
        )

    output_jsonl = args.output_dir / f"{args.model_name}_top{args.top_k}.jsonl"
    output_csv = args.output_dir / f"{args.model_name}_top{args.top_k}_flat.csv"
    top1, top3, topk = [], [], []
    missing_query_rows = 0
    missing_pool_rows = 0

    with output_jsonl.open("w", encoding="utf-8") as json_out, output_csv.open(
        "w", encoding="utf-8", newline=""
    ) as csv_out:
        writer = csv.DictWriter(
            csv_out,
            fieldnames=[
                "sample_id",
                "rank",
                "score",
                "image_id",
                "image_path",
                "local_image_path",
                "dominant_emotion",
                "valence",
                "arousal",
            ],
        )
        writer.writeheader()

        for start in range(0, len(query_x), args.chunk_size):
            end = min(start + args.chunk_size, len(query_x))
            similarity_batch = query_x[start:end] @ pool_x.T

            for offset, scores in enumerate(similarity_batch):
                query_index_number = start + offset
                query_id = normalize_id(query_ids[query_index_number])
                query_row = resolved_query_rows[query_index_number]
                sample_id = sample_ids[query_index_number]

                k = min(args.top_k, len(scores))
                selected = np.argpartition(-scores, k - 1)[:k]
                selected = selected[np.argsort(-scores[selected])]
                items = []
                selected_scores = []

                for rank, pool_position in enumerate(selected, start=1):
                    pool_id = normalize_id(pool_ids[pool_position])
                    pool_row = resolved_pool_rows[pool_position] or {}
                    score = float(scores[pool_position])
                    emotion, valence, arousal = extract_labels(pool_row)
                    item = {
                        "rank": rank,
                        "score": score,
                        "pool_feature_index": int(pool_position),
                        "pool_feature_id": pool_id,
                        "image_id": pool_row.get("image_id") or pool_id,
                        "image_path": pool_row.get("image_path"),
                        "local_image_path": pool_row.get("local_image_path"),
                        "dominant_emotion": emotion,
                        "valence": valence,
                        "arousal": arousal,
                        "annotation": pool_row,
                    }
                    items.append(item)
                    selected_scores.append(score)
                    writer.writerow(
                        {
                            "sample_id": sample_id,
                            "rank": rank,
                            "score": score,
                            "image_id": item["image_id"],
                            "image_path": item["image_path"],
                            "local_image_path": item["local_image_path"],
                            "dominant_emotion": emotion,
                            "valence": valence,
                            "arousal": arousal,
                        }
                    )

                json_out.write(
                    json.dumps(
                        {
                            "model": args.model_name,
                            "sample_id": sample_id,
                            "query_feature_id": query_id,
                            "query_row": query_row,
                            "top_k": items,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                top1.append(selected_scores[0])
                top3.append(float(np.mean(selected_scores[:3])))
                topk.append(float(np.mean(selected_scores)))

            print(f"{args.model_name}: scored {end}/{len(query_x)}", flush=True)

    summary = {
        "model": args.model_name,
        "query_rows": int(len(query_x)),
        "pool_rows": int(len(pool_x)),
        "feature_dim": int(query_x.shape[1]),
        "top_k": args.top_k,
        "missing_query_rows": missing_query_rows,
        "missing_pool_rows": missing_pool_rows,
        "top1_cosine": describe(top1),
        "top3_mean_cosine": describe(top3),
        "topk_mean_cosine": describe(topk),
        "outputs": {"jsonl": str(output_jsonl), "csv": str(output_csv)},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
