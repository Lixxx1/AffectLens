from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


DEFAULT_LABELS = [
    "Aroused",
    "Excited",
    "Happy",
    "Alarmed",
    "Annoyed",
    "Frustrated",
    "Sad",
    "Bored",
    "Tired",
    "Contentment",
    "Calm",
    "Glad",
]

DEFAULT_RULES = [
    "Aroused:Excited",
    "Annoyed:Frustrated",
    "Sad:Tired",
    "Bored:Tired",
    "Contentment:Calm",
    "Glad:Contentment",
    "Glad:Calm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search pairwise threshold corrections from validation logits.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--rules", default=",".join(DEFAULT_RULES), help="Comma-separated pairs like Aroused:Excited.")
    parser.add_argument("--thresholds", default="-1.5,-1.25,-1.0,-0.75,-0.5,-0.35,-0.25,-0.15,-0.05,0.0,0.05,0.15,0.25,0.35,0.5,0.75,1.0,1.25,1.5")
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--metric", choices=("macro_f1", "accuracy", "weighted_f1"), default="macro_f1")
    parser.add_argument("--objective", choices=("metric", "macro_f1_plus_accuracy"), default="macro_f1_plus_accuracy")
    parser.add_argument("--acc-weight", type=float, default=0.15)
    return parser.parse_args()


def load_predictions(path: Path, labels: list[str]) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, list[str]]:
    label_to_id = {label: index for index, label in enumerate(labels)}
    rows: list[dict[str, Any]] = []
    y_true: list[int] = []
    logits: list[list[float]] = []
    image_ids: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            true_label = row.get("dominant_emotion")
            if true_label not in label_to_id:
                raise ValueError(f"Unknown label {true_label!r}")
            row_logits = row.get("logits")
            if not isinstance(row_logits, dict):
                raise ValueError(f"Missing logits for {row.get('image_id') or row.get('sample_id')}")
            rows.append(row)
            y_true.append(label_to_id[true_label])
            logits.append([float(row_logits[label]) for label in labels])
            image_ids.append(str(row.get("image_id") or row.get("sample_id") or len(image_ids)))
    return rows, np.asarray(y_true, dtype=np.int64), np.asarray(logits, dtype=np.float32), image_ids


def evaluate(labels: list[str], y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=list(range(len(labels))),
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=list(range(len(labels))),
                average="weighted",
                zero_division=0,
            )
        ),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(labels))),
            target_names=labels,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": cm.tolist(),
    }


def apply_rules(logits: np.ndarray, pairs: list[tuple[int, int]], thresholds: list[float]) -> np.ndarray:
    pred = logits.argmax(axis=1).astype(np.int64)
    for (left, right), threshold in zip(pairs, thresholds):
        in_pair = (pred == left) | (pred == right)
        # Positive margin means left is stronger. If margin >= threshold choose left,
        # otherwise choose right. This only changes samples already predicted as one
        # of the two labels, keeping rules local and interpretable.
        margin = logits[:, left] - logits[:, right]
        pred[in_pair & (margin >= threshold)] = left
        pred[in_pair & (margin < threshold)] = right
    return pred


def plot_confusion_matrix(cm: list[list[int]], labels: list[str], path: Path, normalize_rows: bool, title: str) -> None:
    matrix = np.asarray(cm, dtype=np.float32)
    if normalize_rows:
        denom = matrix.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        shown = matrix / denom
    else:
        shown = matrix
    fig, ax = plt.subplots(figsize=(10.5, 8.5), dpi=170)
    image = ax.imshow(shown, cmap="GnBu", vmin=0.0, vmax=1.0 if normalize_rows else None)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    threshold = float(shown.max()) * 0.55 if shown.size else 0.0
    for row in range(len(labels)):
        for col in range(len(labels)):
            value = shown[row, col]
            text = f"{value:.0%}\n{int(matrix[row, col])}" if normalize_rows else str(int(matrix[row, col]))
            ax.text(col, row, text, ha="center", va="center", fontsize=7, color="white" if value > threshold else "#1a1a1a")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.passes < 1:
        raise SystemExit("--passes must be at least 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    label_to_id = {label: index for index, label in enumerate(labels)}
    rules = [rule.strip() for rule in args.rules.split(",") if rule.strip()]
    pairs = []
    for rule in rules:
        if ":" not in rule:
            raise SystemExit(f"Invalid rule {rule!r}; expected Left:Right")
        left, right = [part.strip() for part in rule.split(":", 1)]
        if left not in label_to_id or right not in label_to_id:
            raise SystemExit(f"Unknown label in rule {rule!r}")
        pairs.append((label_to_id[left], label_to_id[right]))
    candidate_thresholds = [float(value) for value in args.thresholds.split(",") if value.strip()]

    rows, y_true, logits, image_ids = load_predictions(args.predictions, labels)
    base_pred = logits.argmax(axis=1).astype(np.int64)
    before = evaluate(labels, y_true, base_pred)
    selected = [0.0 for _ in pairs]

    def score(metrics: dict[str, Any]) -> float:
        if args.objective == "macro_f1_plus_accuracy":
            return float(metrics["macro_f1"]) + args.acc_weight * float(metrics["accuracy"])
        return float(metrics[args.metric])

    best_pred = apply_rules(logits, pairs, selected)
    best_metrics = evaluate(labels, y_true, best_pred)
    best_score = score(best_metrics)
    trace: list[dict[str, Any]] = []
    for pass_id in range(1, args.passes + 1):
        changed = False
        for rule_index, ((left, right), rule_name) in enumerate(zip(pairs, rules)):
            old_threshold = selected[rule_index]
            local_best_threshold = old_threshold
            local_best_metrics = best_metrics
            local_best_score = best_score
            for threshold in candidate_thresholds:
                trial = list(selected)
                trial[rule_index] = threshold
                pred = apply_rules(logits, pairs, trial)
                metrics = evaluate(labels, y_true, pred)
                trial_score = score(metrics)
                if trial_score > local_best_score:
                    local_best_threshold = threshold
                    local_best_metrics = metrics
                    local_best_score = trial_score
            selected[rule_index] = local_best_threshold
            best_metrics = local_best_metrics
            best_score = local_best_score
            changed = changed or abs(old_threshold - local_best_threshold) > 1e-8
            trace.append(
                {
                    "pass": pass_id,
                    "rule": rule_name,
                    "old_threshold": old_threshold,
                    "new_threshold": local_best_threshold,
                    "accuracy": best_metrics["accuracy"],
                    "macro_f1": best_metrics["macro_f1"],
                    "weighted_f1": best_metrics["weighted_f1"],
                    "score": best_score,
                }
            )
        if not changed:
            break

    final_pred = apply_rules(logits, pairs, selected)
    after = evaluate(labels, y_true, final_pred)

    with (args.output_dir / "pairwise_rules.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rule", "threshold"], delimiter="\t")
        writer.writeheader()
        for rule, threshold in zip(rules, selected):
            writer.writerow({"rule": rule, "threshold": threshold})
    with (args.output_dir / "search_trace.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["pass", "rule", "old_threshold", "new_threshold", "accuracy", "macro_f1", "weighted_f1", "score"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(trace)
    with (args.output_dir / "val_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row, pred_id, base_id in zip(rows, final_pred, base_pred):
            output = dict(row)
            output["predicted_emotion_before_pairwise"] = labels[int(base_id)]
            output["predicted_emotion"] = labels[int(pred_id)]
            output["correct"] = output.get("dominant_emotion") == output["predicted_emotion"]
            output["pairwise_rules"] = {rule: threshold for rule, threshold in zip(rules, selected)}
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")

    np.savez_compressed(
        args.output_dir / "pairwise_predictions.npz",
        logits=logits.astype(np.float32),
        y_true=y_true,
        y_pred=final_pred,
        base_pred=base_pred,
        image_ids=np.asarray(image_ids),
        labels=np.asarray(labels),
        rules=np.asarray(rules),
        thresholds=np.asarray(selected, dtype=np.float32),
        source_predictions=str(args.predictions),
    )
    plot_confusion_matrix(after["confusion_matrix"], labels, args.output_dir / "confusion_matrix_counts.png", False, "Pairwise Calibrated Gated Fusion")
    plot_confusion_matrix(
        after["confusion_matrix"],
        labels,
        args.output_dir / "confusion_matrix_normalized.png",
        True,
        "Pairwise Calibrated Gated Fusion (Row-Normalized)",
    )
    summary = {
        "source_predictions": str(args.predictions),
        "rules": {rule: threshold for rule, threshold in zip(rules, selected)},
        "objective": args.objective,
        "metric": args.metric,
        "acc_weight": args.acc_weight,
        "before": before,
        "after": after,
        "outputs": {
            "pairwise_rules": str(args.output_dir / "pairwise_rules.tsv"),
            "search_trace": str(args.output_dir / "search_trace.tsv"),
            "predictions": str(args.output_dir / "val_predictions.jsonl"),
            "predictions_npz": str(args.output_dir / "pairwise_predictions.npz"),
            "confusion_matrix_counts": str(args.output_dir / "confusion_matrix_counts.png"),
            "confusion_matrix_normalized": str(args.output_dir / "confusion_matrix_normalized.png"),
            "summary": str(args.output_dir / "summary.json"),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "before": {
                    "accuracy": before["accuracy"],
                    "macro_f1": before["macro_f1"],
                    "weighted_f1": before["weighted_f1"],
                },
                "after": {
                    "accuracy": after["accuracy"],
                    "macro_f1": after["macro_f1"],
                    "weighted_f1": after["weighted_f1"],
                },
                "rules": summary["rules"],
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
