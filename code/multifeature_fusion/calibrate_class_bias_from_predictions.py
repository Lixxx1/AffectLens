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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search additive class bias from multifeature val_predictions.jsonl.")
    parser.add_argument("--predictions", type=Path, required=True, help="Path to val_predictions.jsonl with logits.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--bias-values", default="-1.0,-0.75,-0.5,-0.25,-0.1,0.0,0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--metric", choices=("macro_f1", "accuracy", "weighted_f1"), default="macro_f1")
    parser.add_argument(
        "--objective",
        choices=("metric", "macro_f1_plus_accuracy"),
        default="metric",
        help="Optimization target. macro_f1_plus_accuracy uses macro_f1 + acc_weight * accuracy.",
    )
    parser.add_argument("--acc-weight", type=float, default=0.1)
    parser.add_argument(
        "--min-class-recall",
        type=float,
        default=0.0,
        help="Reject trials where any non-empty class has recall below this value.",
    )
    parser.add_argument(
        "--max-bias",
        type=float,
        default=1.0,
        help="Clamp each final bias to [-max_bias, max_bias].",
    )
    return parser.parse_args()


def load_predictions(path: Path, labels: list[str]) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, list[str]]:
    rows: list[dict[str, Any]] = []
    y_true: list[int] = []
    logits: list[list[float]] = []
    image_ids: list[str] = []
    label_to_id = {label: index for index, label in enumerate(labels)}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            true_label = row.get("dominant_emotion")
            if true_label not in label_to_id:
                raise ValueError(f"Unknown label {true_label!r} in {path}")
            row_logits = row.get("logits")
            if not isinstance(row_logits, dict):
                raise ValueError(f"Missing logits dict for row {row.get('image_id') or row.get('sample_id')}")
            rows.append(row)
            y_true.append(label_to_id[true_label])
            logits.append([float(row_logits[label]) for label in labels])
            image_ids.append(str(row.get("image_id") or row.get("sample_id") or row.get("id") or len(image_ids)))
    return rows, np.asarray(y_true, dtype=np.int64), np.asarray(logits, dtype=np.float32), image_ids


def evaluate(labels: list[str], y_true: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    y_pred = logits.argmax(axis=1)
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
        "class_recall": {
            label: float(cm[index, index] / cm[index].sum()) if cm[index].sum() else 0.0
            for index, label in enumerate(labels)
        },
        "class_support": {
            label: int(cm[index].sum())
            for index, label in enumerate(labels)
        },
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


def coordinate_search(
    labels: list[str],
    y_true: np.ndarray,
    logits: np.ndarray,
    bias_values: list[float],
    metric: str,
    objective: str,
    acc_weight: float,
    min_class_recall: float,
    passes: int,
    max_bias: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    bias = np.zeros(logits.shape[1], dtype=np.float32)
    rows: list[dict[str, Any]] = []
    best_metrics = evaluate(labels, y_true, logits + bias.reshape(1, -1))

    def score(metrics: dict[str, Any]) -> float:
        if min_class_recall > 0:
            recalls = [
                metrics["class_recall"][label]
                for label in labels
                if metrics["class_support"][label] > 0
            ]
            if recalls and min(recalls) < min_class_recall:
                return -1e9
        if objective == "macro_f1_plus_accuracy":
            return float(metrics["macro_f1"]) + acc_weight * float(metrics["accuracy"])
        return float(metrics[metric])

    best_score = score(best_metrics)
    for pass_id in range(1, passes + 1):
        changed = False
        for class_id, label in enumerate(labels):
            old_value = float(bias[class_id])
            best_value = old_value
            local_best_metrics = best_metrics
            local_best_score = best_score
            for value in bias_values:
                trial = bias.copy()
                trial[class_id] = float(np.clip(value, -max_bias, max_bias))
                metrics = evaluate(labels, y_true, logits + trial.reshape(1, -1))
                trial_score = score(metrics)
                if trial_score > local_best_score:
                    best_value = float(trial[class_id])
                    local_best_metrics = metrics
                    local_best_score = trial_score
            bias[class_id] = best_value
            best_metrics = local_best_metrics
            best_score = local_best_score
            changed = changed or abs(best_value - old_value) > 1e-8
            rows.append(
                {
                    "pass": pass_id,
                    "class": label,
                    "old_bias": old_value,
                    "new_bias": best_value,
                    "accuracy": best_metrics["accuracy"],
                    "macro_f1": best_metrics["macro_f1"],
                    "weighted_f1": best_metrics["weighted_f1"],
                    "score": best_score,
                }
            )
        if not changed:
            break
    return bias, rows


def main() -> None:
    args = parse_args()
    if args.passes < 1:
        raise SystemExit("--passes must be at least 1")
    if not 0.0 <= args.min_class_recall <= 1.0:
        raise SystemExit("--min-class-recall must be between 0 and 1")
    if args.max_bias < 0:
        raise SystemExit("--max-bias must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    bias_values = [float(value) for value in args.bias_values.split(",") if value.strip()]
    rows, y_true, logits, image_ids = load_predictions(args.predictions, labels)

    before = evaluate(labels, y_true, logits)
    bias, search_rows = coordinate_search(
        labels,
        y_true,
        logits,
        bias_values,
        args.metric,
        args.objective,
        args.acc_weight,
        args.min_class_recall,
        args.passes,
        args.max_bias,
    )
    calibrated_logits = logits + bias.reshape(1, -1)
    after = evaluate(labels, y_true, calibrated_logits)
    pred_ids = calibrated_logits.argmax(axis=1)

    np.savez_compressed(
        args.output_dir / "calibrated_logits.npz",
        logits=calibrated_logits.astype(np.float32),
        original_logits=logits.astype(np.float32),
        y_true=y_true,
        image_ids=np.asarray(image_ids),
        labels=np.asarray(labels),
        class_bias=bias.astype(np.float32),
        source_predictions=str(args.predictions),
    )
    with (args.output_dir / "class_bias.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class", "bias"], delimiter="\t")
        writer.writeheader()
        for label, value in zip(labels, bias):
            writer.writerow({"class": label, "bias": float(value)})
    with (args.output_dir / "search_trace.tsv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["pass", "class", "old_bias", "new_bias", "accuracy", "macro_f1", "weighted_f1", "score"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(search_rows)
    with (args.output_dir / "val_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row, pred_id, sample_logits in zip(rows, pred_ids, calibrated_logits):
            output = dict(row)
            output["predicted_emotion_before_calibration"] = row.get("predicted_emotion")
            output["predicted_emotion"] = labels[int(pred_id)]
            output["correct"] = output.get("dominant_emotion") == output["predicted_emotion"]
            output["logits"] = {label: float(sample_logits[index]) for index, label in enumerate(labels)}
            output["class_bias"] = {label: float(bias[index]) for index, label in enumerate(labels)}
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")

    plot_confusion_matrix(after["confusion_matrix"], labels, args.output_dir / "confusion_matrix_counts.png", False, "Class Bias Calibrated Gated Fusion")
    plot_confusion_matrix(
        after["confusion_matrix"],
        labels,
        args.output_dir / "confusion_matrix_normalized.png",
        True,
        "Class Bias Calibrated Gated Fusion (Row-Normalized)",
    )
    summary = {
        "source_predictions": str(args.predictions),
        "metric": args.metric,
        "objective": args.objective,
        "acc_weight": args.acc_weight,
        "min_class_recall": args.min_class_recall,
        "bias_values": bias_values,
        "passes": args.passes,
        "max_bias": args.max_bias,
        "before": before,
        "after": after,
        "class_bias": {label: float(value) for label, value in zip(labels, bias)},
        "outputs": {
            "calibrated_logits": str(args.output_dir / "calibrated_logits.npz"),
            "class_bias": str(args.output_dir / "class_bias.tsv"),
            "search_trace": str(args.output_dir / "search_trace.tsv"),
            "predictions": str(args.output_dir / "val_predictions.jsonl"),
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
                "class_bias": summary["class_bias"],
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
