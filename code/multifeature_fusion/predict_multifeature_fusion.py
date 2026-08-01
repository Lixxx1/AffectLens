from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_multifeature_fusion import (  # noqa: E402
    MultiFeatureFusionClassifier,
    load_npz_index,
    norm_id,
    plot_confusion_matrix,
    predict,
)

OUTPUT_ROOT = PROJECT_ROOT / "outputs/fusion"
FEATURE_ROOT = PROJECT_ROOT / "data/features"
DEFAULT_CHECKPOINT = OUTPUT_ROOT / "best_model.pt"
DEFAULT_CLASS_BIAS = SCRIPT_DIR / "class_bias.tsv"
DEFAULT_PAIRWISE_RULES = SCRIPT_DIR / "pairwise_rules.tsv"
DEFAULT_MANIFEST = PROJECT_ROOT / "data/inference/manifest.jsonl"
DEFAULT_INFERENCE_FEATURES = {
    "clip": FEATURE_ROOT / "inference/clip.npz",
    "eva": FEATURE_ROOT / "inference/eva.npz",
    "dinov3": FEATURE_ROOT / "inference/dinov3.npz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run inference with the trained multi-feature fusion classifier "
            "and calibrated post-processing."
        )
    )
    parser.add_argument("--manifest-jsonl", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--feature",
        action="append",
        default=None,
        help="Feature spec name=/path/to/features.npz. Omit to use the default inference features.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--class-bias", type=Path, default=DEFAULT_CLASS_BIAS)
    parser.add_argument("--pairwise-rules", type=Path, default=DEFAULT_PAIRWISE_RULES)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT / "inference",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--data-parallel",
        action="store_true",
        help="Use all visible CUDA devices. Set CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 before running.",
    )
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise SystemExit(f"Expected a JSON list in {path}")
        return [dict(row) for row in data]
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def row_id(row: dict[str, Any]) -> str:
    for key in ("image_id", "sample_id", "id", "image_path"):
        value = row.get(key)
        if value:
            return norm_id(value)
    raise SystemExit(f"Could not find an id field in row keys: {sorted(row)}")


def parse_feature_specs(specs: list[str] | None) -> dict[str, Path]:
    if not specs:
        return dict(DEFAULT_INFERENCE_FEATURES)
    output: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"Invalid feature spec {spec!r}; expected name=path")
        name, path = spec.split("=", 1)
        name = name.strip()
        if not name:
            raise SystemExit(f"Invalid empty feature name in {spec!r}")
        if name in output:
            raise SystemExit(f"Duplicate feature name {name!r}")
        output[name] = Path(path)
    return output


def align_feature_rows(rows: list[dict[str, Any]], feature_path: Path) -> tuple[np.ndarray, int]:
    x_all, id_to_index, duplicates = load_npz_index(feature_path)
    aligned: list[np.ndarray] = []
    missing: list[str] = []
    for row in rows:
        image_id = row_id(row)
        index = id_to_index.get(image_id)
        if index is None:
            missing.append(image_id)
            continue
        aligned.append(x_all[index])
    if missing:
        preview = ", ".join(missing[:10])
        raise SystemExit(f"{feature_path} is missing {len(missing)} rows. First missing: {preview}")
    return np.asarray(aligned, dtype=np.float32), duplicates


def load_tsv_mapping(path: Path, key_field: str, value_field: str) -> dict[str, float]:
    mapping: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            mapping[str(row[key_field])] = float(row[value_field])
    return mapping


def standardize_features(features: dict[str, np.ndarray], checkpoint: dict[str, Any]) -> dict[str, np.ndarray]:
    scalers = checkpoint.get("scalers")
    if not scalers:
        raise SystemExit("Checkpoint does not contain feature scalers.")
    output: dict[str, np.ndarray] = {}
    for name, values in features.items():
        if name not in scalers:
            raise SystemExit(f"Checkpoint does not contain scaler for feature {name!r}.")
        mean = np.asarray(scalers[name]["mean"], dtype=np.float32)
        std = np.asarray(scalers[name]["std"], dtype=np.float32)
        std = np.where(std < 1e-6, 1.0, std)
        output[name] = ((values - mean) / std).astype(np.float32)
    return output


def make_model(checkpoint: dict[str, Any]) -> MultiFeatureFusionClassifier:
    config = checkpoint["config"]
    return MultiFeatureFusionClassifier(
        feature_dims={str(k): int(v) for k, v in checkpoint["feature_dims"].items()},
        fusion=str(config["fusion"]),
        hidden_size=int(config.get("hidden_size", 512)),
        num_classes=len(checkpoint["labels"]),
        dropout=float(config.get("dropout", 0.3)),
        attention_heads=int(config.get("attention_heads", 8)),
        modality_dropout=float(config.get("modality_dropout", 0.0)),
        transformer_layers=int(config.get("transformer_layers", 2)),
        use_modality_embedding=bool(config.get("use_modality_embedding", False)),
        head_type=str(config.get("head_type", "mlp")),
        head_scale=float(config.get("head_scale", config.get("scale", 30.0))),
        head_margin=float(config.get("head_margin", config.get("margin", 0.2))),
    )


def add_class_bias(logits: np.ndarray, labels: list[str], class_bias: dict[str, float]) -> np.ndarray:
    adjusted = logits.copy()
    for index, label in enumerate(labels):
        adjusted[:, index] += float(class_bias.get(label, 0.0))
    return adjusted


def ordered_pairwise_rules(path: Path) -> list[tuple[str, float]]:
    mapping = load_tsv_mapping(path, "rule", "threshold")
    return [(rule, float(threshold)) for rule, threshold in mapping.items()]


def apply_pairwise_rules(logits: np.ndarray, labels: list[str], rules: list[tuple[str, float]]) -> np.ndarray:
    pred = logits.argmax(axis=1).astype(np.int64)
    label_to_id = {label: index for index, label in enumerate(labels)}
    for rule, threshold in rules:
        if ":" not in rule:
            raise SystemExit(f"Invalid pairwise rule {rule!r}; expected Left:Right")
        left, right = [part.strip() for part in rule.split(":", 1)]
        if left not in label_to_id or right not in label_to_id:
            raise SystemExit(f"Unknown label in pairwise rule {rule!r}")
        left_id = label_to_id[left]
        right_id = label_to_id[right]
        in_pair = (pred == left_id) | (pred == right_id)
        margin = logits[:, left_id] - logits[:, right_id]
        pred[in_pair & (margin >= float(threshold))] = left_id
        pred[in_pair & (margin < float(threshold))] = right_id
    return pred


def labels_if_available(rows: list[dict[str, Any]], labels: list[str]) -> np.ndarray | None:
    label_to_id = {label: index for index, label in enumerate(labels)}
    y_true: list[int] = []
    for row in rows:
        label = row.get("dominant_emotion")
        if label not in label_to_id:
            return None
        y_true.append(label_to_id[str(label)])
    return np.asarray(y_true, dtype=np.int64)


def evaluate_predictions(labels: list[str], y_true: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    true_names = [labels[int(index)] for index in y_true]
    pred_names = [labels[int(index)] for index in pred]
    accuracy = float(accuracy_score(true_names, pred_names))
    macro_f1 = float(f1_score(true_names, pred_names, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(true_names, pred_names, labels=labels, average="weighted", zero_division=0))
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "task_score": 0.5 * (accuracy + macro_f1),
        "classification_report": classification_report(
            true_names,
            pred_names,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(true_names, pred_names, labels=labels).tolist(),
        "labels": labels,
    }


def save_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    labels: list[str],
    base_logits: np.ndarray,
    biased_logits: np.ndarray,
    base_pred: np.ndarray,
    biased_pred: np.ndarray,
    final_pred: np.ndarray,
    class_bias: dict[str, float],
    pairwise_rules: list[tuple[str, float]],
    gate_weights: np.ndarray | None,
    feature_names: list[str],
    y_true: np.ndarray | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_ids = [row_id(row) for row in rows]

    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            output = dict(row)
            output["predicted_emotion_base_model"] = labels[int(base_pred[index])]
            output["predicted_emotion_before_calibration"] = labels[int(base_pred[index])]
            output["predicted_emotion_before_pairwise"] = labels[int(biased_pred[index])]
            output["predicted_emotion"] = labels[int(final_pred[index])]
            if row.get("dominant_emotion") in labels:
                output["correct"] = output.get("dominant_emotion") == output["predicted_emotion"]
            output["logits"] = {label: float(biased_logits[index, class_id]) for class_id, label in enumerate(labels)}
            output["base_logits"] = {label: float(base_logits[index, class_id]) for class_id, label in enumerate(labels)}
            output["class_bias"] = {label: float(class_bias.get(label, 0.0)) for label in labels}
            output["pairwise_rules"] = {rule: float(threshold) for rule, threshold in pairwise_rules}
            if gate_weights is not None:
                output["gate_weights"] = {
                    name: float(gate_weights[index, feature_id]) for feature_id, name in enumerate(feature_names)
                }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")

    with (output_dir / "emotion_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "predicted_emotion"])
        writer.writeheader()
        for image_id, pred_id in zip(image_ids, final_pred):
            writer.writerow({"image_id": image_id, "predicted_emotion": labels[int(pred_id)]})

    arrays: dict[str, Any] = {
        "base_logits": base_logits.astype(np.float32),
        "biased_logits": biased_logits.astype(np.float32),
        "base_pred": base_pred.astype(np.int64),
        "biased_pred": biased_pred.astype(np.int64),
        "final_pred": final_pred.astype(np.int64),
        "image_ids": np.asarray(image_ids, dtype=object),
        "labels": np.asarray(labels, dtype=object),
        "class_bias": np.asarray([class_bias.get(label, 0.0) for label in labels], dtype=np.float32),
        "pairwise_rules": np.asarray([rule for rule, _ in pairwise_rules], dtype=object),
        "pairwise_thresholds": np.asarray([threshold for _, threshold in pairwise_rules], dtype=np.float32),
    }
    if gate_weights is not None:
        arrays["gate_weights"] = gate_weights.astype(np.float32)
        arrays["feature_names"] = np.asarray(feature_names, dtype=object)
    if y_true is not None:
        arrays["y_true"] = y_true.astype(np.int64)
    np.savez_compressed(output_dir / "predictions.npz", **arrays)


def main() -> None:
    args = parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    rows = read_manifest(args.manifest_jsonl)
    if not rows:
        raise SystemExit(f"No manifest rows found in {args.manifest_jsonl}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    labels = [str(label) for label in checkpoint["labels"]]
    feature_names = [str(name) for name in checkpoint["feature_names"]]
    feature_paths = parse_feature_specs(args.feature)
    missing_features = sorted(set(feature_names) - set(feature_paths))
    if missing_features:
        raise SystemExit(f"Missing required features for checkpoint: {missing_features}")

    raw_features: dict[str, np.ndarray] = {}
    duplicate_counts: dict[str, int] = {}
    for name in feature_names:
        raw_features[name], duplicate_counts[name] = align_feature_rows(rows, feature_paths[name])
    features = standardize_features(raw_features, checkpoint)

    model = make_model(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    if args.data_parallel and device.startswith("cuda") and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    base_logits, gate_weights = predict(model, features, args.batch_size, device)
    class_bias = load_tsv_mapping(args.class_bias, "class", "bias")
    pairwise_rules = ordered_pairwise_rules(args.pairwise_rules)
    biased_logits = add_class_bias(base_logits, labels, class_bias)
    base_pred = base_logits.argmax(axis=1).astype(np.int64)
    biased_pred = biased_logits.argmax(axis=1).astype(np.int64)
    final_pred = apply_pairwise_rules(biased_logits, labels, pairwise_rules)
    y_true = labels_if_available(rows, labels)

    save_outputs(
        args.output_dir,
        rows,
        labels,
        base_logits,
        biased_logits,
        base_pred,
        biased_pred,
        final_pred,
        class_bias,
        pairwise_rules,
        gate_weights,
        feature_names,
        y_true,
    )

    metrics: dict[str, Any] = {}
    if y_true is not None:
        metrics = {
            "base_model": evaluate_predictions(labels, y_true, base_pred),
            "after_class_bias": evaluate_predictions(labels, y_true, biased_pred),
            "after_pairwise": evaluate_predictions(labels, y_true, final_pred),
        }
        plot_confusion_matrix(
            metrics["after_pairwise"]["confusion_matrix"],
            labels,
            args.output_dir / "confusion_matrix_counts.png",
            normalize_rows=False,
            title="Multi-feature fusion with calibrated post-processing",
        )
        plot_confusion_matrix(
            metrics["after_pairwise"]["confusion_matrix"],
            labels,
            args.output_dir / "confusion_matrix_normalized.png",
            normalize_rows=True,
            title="Multi-feature fusion with calibrated post-processing (row-normalized)",
        )

    summary = {
        "manifest_jsonl": str(args.manifest_jsonl),
        "rows": len(rows),
        "checkpoint": str(args.checkpoint),
        "feature_paths": {name: str(feature_paths[name]) for name in feature_names},
        "feature_duplicate_ids": duplicate_counts,
        "class_bias_tsv": str(args.class_bias),
        "pairwise_rules_tsv": str(args.pairwise_rules),
        "class_bias": {label: float(class_bias.get(label, 0.0)) for label in labels},
        "pairwise_rules": {rule: float(threshold) for rule, threshold in pairwise_rules},
        "metrics": metrics,
        "outputs": {
            "predictions_jsonl": str(args.output_dir / "predictions.jsonl"),
            "predictions_csv": str(args.output_dir / "emotion_predictions.csv"),
            "predictions_npz": str(args.output_dir / "predictions.npz"),
            "summary": str(args.output_dir / "summary.json"),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "rows": len(rows),
                "output_dir": str(args.output_dir),
                "metrics": {
                    name: {
                        "accuracy": values["accuracy"],
                        "macro_f1": values["macro_f1"],
                        "weighted_f1": values["weighted_f1"],
                        "task_score": values["task_score"],
                    }
                    for name, values in metrics.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
