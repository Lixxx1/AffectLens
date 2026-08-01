from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a consensus ranking from three retrieval models, with an optional "
            "lower-vote fallback to guarantee enough few-shot references."
        )
    )
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--openai-clip", type=Path, required=True)
    parser.add_argument("--dinov3", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-keep", type=int, default=10)
    parser.add_argument(
        "--min-keep",
        type=int,
        default=0,
        help=(
            "Guarantee at least this many references by filling a short unanimous "
            "consensus with two- and one-model matches. Default: unanimous matches only."
        ),
    )
    return parser.parse_args()


def normalize_id(value: object) -> str:
    return str(value or "").replace("\\", "/")


def load_results(path: Path) -> dict[str, dict]:
    results: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise SystemExit(f"{path}:{line_number} is not a JSON object")
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise SystemExit(f"{path}:{line_number} has no non-empty sample_id")
            if sample_id in results:
                raise SystemExit(f"{path}:{line_number} duplicates sample_id {sample_id!r}")
            results[sample_id] = row
    if not results:
        raise SystemExit(f"No retrieval rows found in {path}")
    return results


def main() -> None:
    args = parse_args()
    if args.max_keep < 1:
        raise SystemExit("--max-keep must be at least 1.")
    if args.min_keep < 0 or args.min_keep > args.max_keep:
        raise SystemExit("--min-keep must be between 0 and --max-keep.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_paths = {
        "clip_h14": args.clip,
        "openai_clip_l14_hf": args.openai_clip,
        "dinov3_vit7b16": args.dinov3,
    }
    model_rows = {name: load_results(path) for name, path in model_paths.items()}
    sample_sets = {name: set(rows) for name, rows in model_rows.items()}
    expected_ids = next(iter(sample_sets.values()))
    mismatches = {
        name: {
            "missing": sorted(expected_ids - ids)[:10],
            "extra": sorted(ids - expected_ids)[:10],
            "missing_count": len(expected_ids - ids),
            "extra_count": len(ids - expected_ids),
        }
        for name, ids in sample_sets.items()
        if ids != expected_ids
    }
    if mismatches:
        raise SystemExit(
            "Retrieval sample sets differ across encoders: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    sample_ids = sorted(expected_ids)

    output = args.output_dir / "reference_consensus.jsonl"
    kept_hist = Counter()
    samples_without_unanimous_match = 0
    samples_using_fallback = 0
    insufficient_samples: list[str] = []
    total_items = 0

    with output.open("w", encoding="utf-8") as handle:
        for sample_id in sample_ids:
            hits: dict[str, list[dict]] = defaultdict(list)
            representative: dict[str, dict] = {}

            for model, rows in model_rows.items():
                for item in rows[sample_id].get("top_k", []):
                    image_id = normalize_id(
                        item.get("image_id") or item.get("pool_feature_id")
                    )
                    if not image_id:
                        continue
                    hits[image_id].append(
                        {
                            "model": model,
                            "rank": int(item.get("rank", 999)),
                            "score": float(item.get("score", 0.0)),
                        }
                    )
                    representative.setdefault(image_id, item)

            candidates = []
            for image_id, votes in hits.items():
                models = {vote["model"] for vote in votes}
                item = dict(representative[image_id])
                item["consensus"] = {
                    "vote_count": len(models),
                    "models": sorted(models),
                    "rank_sum": sum(vote["rank"] for vote in votes),
                    "best_rank": min(vote["rank"] for vote in votes),
                    "scores": {vote["model"]: vote["score"] for vote in votes},
                    "ranks": {vote["model"]: vote["rank"] for vote in votes},
                    "votes": sorted(votes, key=lambda vote: vote["model"]),
                }
                candidates.append(item)

            candidates.sort(
                key=lambda item: (
                    -item["consensus"]["vote_count"],
                    item["consensus"]["rank_sum"],
                    item["consensus"]["best_rank"],
                    -sum(item["consensus"]["scores"].values())
                    / item["consensus"]["vote_count"],
                )
            )
            unanimous_candidates = [
                item for item in candidates if item["consensus"]["vote_count"] == 3
            ]
            selected = unanimous_candidates[: args.max_keep]
            if args.min_keep and len(selected) < args.min_keep:
                selected_ids = {
                    normalize_id(
                        item.get("image_id") or item.get("pool_feature_id")
                    )
                    for item in selected
                }
                fallbacks = [
                    item
                    for item in candidates
                    if normalize_id(
                        item.get("image_id") or item.get("pool_feature_id")
                    )
                    not in selected_ids
                ]
                selected.extend(fallbacks[: args.min_keep - len(selected)])
                if len(selected) > len(unanimous_candidates):
                    samples_using_fallback += 1
            candidates = selected[: args.max_keep]
            kept_hist[len(candidates)] += 1
            total_items += len(candidates)
            if not unanimous_candidates:
                samples_without_unanimous_match += 1
            if len(candidates) < args.min_keep:
                insufficient_samples.append(sample_id)

            query_row = model_rows["clip_h14"][sample_id].get("query_row")
            handle.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "query_row": query_row,
                        "top_k": candidates,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = {
        "definition": (
            "A unanimous match is the same pool image appearing in the top-k lists "
            "of CLIP-H/14, OpenAI CLIP-L/14, and DINOv3."
        ),
        "samples": len(sample_ids),
        "samples_without_unanimous_match": samples_without_unanimous_match,
        "samples_with_unanimous_match": (
            len(sample_ids) - samples_without_unanimous_match
        ),
        "samples_using_fallback": samples_using_fallback,
        "min_keep": args.min_keep,
        "max_keep": args.max_keep,
        "insufficient_samples": insufficient_samples,
        "total_selected_items": total_items,
        "kept_hist": {str(key): value for key, value in sorted(kept_hist.items())},
        "output": str(output),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if insufficient_samples:
        raise SystemExit(
            f"{len(insufficient_samples)} sample(s) have fewer than "
            f"{args.min_keep} unique references."
        )


if __name__ == "__main__":
    main()
