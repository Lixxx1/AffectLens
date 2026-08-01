from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


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

DEFAULT_GROUPS = {
    "Aroused": "positive_high",
    "Excited": "positive_high",
    "Happy": "positive_high",
    "Alarmed": "negative_high",
    "Annoyed": "negative_high",
    "Frustrated": "negative_high",
    "Sad": "negative_low",
    "Bored": "negative_low",
    "Tired": "negative_low",
    "Contentment": "positive_low",
    "Calm": "positive_low",
    "Glad": "positive_low",
}

DEFAULT_GROUP_NAMES = ["positive_high", "negative_high", "negative_low", "positive_low"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-feature image fusion classifier.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument(
        "--train-feature",
        action="append",
        required=True,
        help="Feature spec in the form name=/path/to/train_features.npz",
    )
    parser.add_argument(
        "--val-feature",
        action="append",
        required=True,
        help="Feature spec in the form name=/path/to/val_features.npz",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def norm_id(value: Any) -> str:
    text = str(value).replace("\\", "/").strip()
    if "/Images/" in text:
        text = "Images/" + text.split("/Images/", 1)[1]
    text = text.lstrip("./")
    if text and not text.startswith("Images/") and "/" in text:
        text = "Images/" + text
    return text


def parse_feature_specs(specs: list[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"Invalid feature spec {spec!r}; expected name=path")
        name, path = spec.split("=", 1)
        name = name.strip()
        if not name:
            raise SystemExit(f"Invalid empty feature name in spec {spec!r}")
        if name in output:
            raise SystemExit(f"Duplicate feature name {name!r}")
        output[name] = Path(path)
    return output


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_npz_index(path: Path) -> tuple[np.ndarray, dict[str, int], int]:
    data = np.load(path, allow_pickle=True)
    if "x" not in data.files or "image_ids" not in data.files:
        raise SystemExit(f"{path} must contain x and image_ids; found {data.files}")
    x = data["x"].astype(np.float32)
    ids = [norm_id(value) for value in data["image_ids"]]
    if x.ndim != 2:
        raise SystemExit(f"{path} x must be a 2D array; found shape {x.shape}")
    if len(x) != len(ids):
        raise SystemExit(f"{path} has {len(x)} feature rows but {len(ids)} image_ids")
    if any(not image_id for image_id in ids):
        raise SystemExit(f"{path} contains empty image_ids")
    id_to_index: dict[str, int] = {}
    duplicates = 0
    duplicate_ids: list[str] = []
    for index, image_id in enumerate(ids):
        if image_id in id_to_index:
            duplicates += 1
            if len(duplicate_ids) < 10:
                duplicate_ids.append(image_id)
            continue
        id_to_index[image_id] = index
    if duplicates:
        raise SystemExit(
            f"{path} contains {duplicates} duplicate normalized image_ids. "
            f"First duplicates: {duplicate_ids}"
        )
    return x, id_to_index, duplicates


def align_feature(
    rows: list[dict[str, Any]],
    feature_path: Path,
) -> tuple[np.ndarray, int, int]:
    x_all, id_to_index, duplicates = load_npz_index(feature_path)
    xs: list[np.ndarray] = []
    missing = 0
    for row in rows:
        image_id = norm_id(row["image_id"])
        if image_id not in id_to_index:
            missing += 1
            continue
        xs.append(x_all[id_to_index[image_id]])
    if missing:
        raise SystemExit(f"{feature_path} is missing {missing} rows from manifest.")
    return np.asarray(xs, dtype=np.float32), missing, duplicates


def label_array(rows: list[dict[str, Any]], labels: list[str]) -> np.ndarray:
    label_to_id = {label: index for index, label in enumerate(labels)}
    ys: list[int] = []
    bad: list[tuple[str, Any]] = []
    for row in rows:
        label = row.get("dominant_emotion")
        if label not in label_to_id:
            bad.append((norm_id(row.get("image_id")), label))
            continue
        ys.append(label_to_id[str(label)])
    if bad:
        raise SystemExit(f"Found {len(bad)} bad labels. First 10: {bad[:10]}")
    return np.asarray(ys, dtype=np.int64)


def group_array(rows: list[dict[str, Any]], group_names: list[str], group_map: dict[str, str]) -> np.ndarray:
    group_to_id = {group: index for index, group in enumerate(group_names)}
    ys: list[int] = []
    bad: list[tuple[str, Any]] = []
    for row in rows:
        emotion = row.get("dominant_emotion")
        group = group_map.get(str(emotion))
        if group not in group_to_id:
            bad.append((norm_id(row.get("image_id")), emotion))
            continue
        ys.append(group_to_id[group])
    if bad:
        raise SystemExit(f"Found {len(bad)} labels without groups. First 10: {bad[:10]}")
    return np.asarray(ys, dtype=np.int64)


def standardize_pair(train_x: np.ndarray, val_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train_x - mean) / std, (val_x - mean) / std, mean.astype(np.float32), std.astype(np.float32)


def class_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return weights.astype(np.float32)


class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, class_counts: np.ndarray, label_smoothing: float = 0.0) -> None:
        super().__init__()
        counts = np.asarray(class_counts, dtype=np.float32)
        counts[counts <= 0] = 1.0
        self.register_buffer("log_class_counts", torch.log(torch.from_numpy(counts)))
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        adjusted_logits = logits + self.log_class_counts.to(logits.device).reshape(1, -1)
        return nn.functional.cross_entropy(adjusted_logits, target, label_smoothing=self.label_smoothing)


def sample_weights(y: np.ndarray, power: float) -> np.ndarray:
    counts = np.bincount(y).astype(np.float32)
    weights = np.asarray([(1.0 / counts[label]) ** power for label in y], dtype=np.float64)
    return weights / weights.mean()


class FeatureProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ClassifierHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        dropout: float,
        head_type: str = "mlp",
        scale: float = 30.0,
        margin: float = 0.2,
    ) -> None:
        super().__init__()
        self.head_type = head_type
        self.scale = scale
        self.margin = margin
        if head_type == "mlp":
            self.net = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_classes),
            )
        elif head_type == "deep_mlp":
            self.net = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_classes),
            )
        elif head_type == "residual_mlp":
            self.net = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                ResidualMLPBlock(hidden_size, dropout),
                ResidualMLPBlock(hidden_size, dropout),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, num_classes),
            )
        elif head_type == "linear":
            self.net = nn.Linear(hidden_size, num_classes)
        elif head_type in {"cosine", "cosface", "arcface"}:
            self.weight = nn.Parameter(torch.empty(num_classes, hidden_size))
            nn.init.xavier_uniform_(self.weight)
        else:
            raise ValueError(f"Unknown classifier head: {head_type}")

    def cosine_logits(self, fused: torch.Tensor) -> torch.Tensor:
        fused = nn.functional.normalize(fused, dim=1)
        weight = nn.functional.normalize(self.weight, dim=1)
        return nn.functional.linear(fused, weight).clamp(-1.0 + 1e-7, 1.0 - 1e-7)

    def forward(self, fused: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        if self.head_type in {"mlp", "deep_mlp", "residual_mlp", "linear"}:
            return self.net(fused)
        cosine = self.cosine_logits(fused)
        if target is None or self.head_type == "cosine":
            return self.scale * cosine
        adjusted = cosine.clone()
        rows = torch.arange(cosine.size(0), device=cosine.device)
        if self.head_type == "cosface":
            adjusted[rows, target] = adjusted[rows, target] - self.margin
        elif self.head_type == "arcface":
            theta = torch.acos(cosine[rows, target])
            adjusted[rows, target] = torch.cos(theta + self.margin)
        else:
            raise ValueError(f"Unknown margin head: {self.head_type}")
        return self.scale * adjusted


class MultiFeatureFusionClassifier(nn.Module):
    def __init__(
        self,
        feature_dims: dict[str, int],
        fusion: str,
        hidden_size: int,
        num_classes: int,
        dropout: float,
        attention_heads: int,
        modality_dropout: float = 0.0,
        transformer_layers: int = 2,
        use_modality_embedding: bool = False,
        head_type: str = "mlp",
        head_scale: float = 30.0,
        head_margin: float = 0.2,
    ) -> None:
        super().__init__()
        self.feature_names = list(feature_dims)
        self.fusion = fusion
        self.modality_dropout = modality_dropout
        self.use_modality_embedding = use_modality_embedding
        self.projectors = nn.ModuleDict(
            {
                name: FeatureProjector(input_dim=feature_dims[name], hidden_size=hidden_size, dropout=dropout)
                for name in self.feature_names
            }
        )
        if fusion == "projected_concat":
            self.fuse = nn.Sequential(
                nn.Linear(hidden_size * len(self.feature_names), hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        elif fusion == "gated_sum":
            self.gate = nn.Sequential(
                nn.Linear(hidden_size * len(self.feature_names), hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, len(self.feature_names)),
            )
        elif fusion == "attention":
            if use_modality_embedding:
                self.modality_embedding = nn.Parameter(torch.zeros(1, len(self.feature_names), hidden_size))
            self.attn = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=attention_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.attn_norm = nn.LayerNorm(hidden_size)
            self.fuse = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        elif fusion == "transformer_cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            self.modality_embedding = nn.Parameter(torch.zeros(1, len(self.feature_names) + 1, hidden_size))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=attention_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
            self.encoder_norm = nn.LayerNorm(hidden_size)
        else:
            raise ValueError(f"Unknown fusion: {fusion}")
        self.head = ClassifierHead(
            hidden_size=hidden_size,
            num_classes=num_classes,
            dropout=dropout,
            head_type=head_type,
            scale=head_scale,
            margin=head_margin,
        )

    def project(self, batch: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        return [self.projectors[name](batch[name]) for name in self.feature_names]

    def apply_modality_dropout(self, projected: list[torch.Tensor]) -> list[torch.Tensor]:
        if not self.training or self.modality_dropout <= 0 or len(projected) <= 1:
            return projected
        batch_size = projected[0].shape[0]
        keep = []
        for tensor in projected:
            mask = (torch.rand(batch_size, 1, device=tensor.device) > self.modality_dropout).to(tensor.dtype)
            keep.append(tensor * mask)
        stacked_keep = torch.stack([(tensor.abs().sum(dim=1) > 0) for tensor in keep], dim=1)
        all_dropped = ~stacked_keep.any(dim=1)
        if all_dropped.any():
            restore_indices = torch.randint(0, len(projected), (int(all_dropped.sum().item()),), device=projected[0].device)
            row_indices = all_dropped.nonzero(as_tuple=False).flatten()
            for feature_index, rows in enumerate([(restore_indices == i).nonzero(as_tuple=False).flatten() for i in range(len(projected))]):
                if len(rows) == 0:
                    continue
                keep[feature_index][row_indices[rows]] = projected[feature_index][row_indices[rows]]
        scale = 1.0 / max(1.0 - self.modality_dropout, 1e-6)
        return [tensor * scale for tensor in keep]

    def fuse_features(self, projected: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
        projected = self.apply_modality_dropout(projected)
        if self.fusion == "projected_concat":
            return self.fuse(torch.cat(projected, dim=1)), None
        if self.fusion == "gated_sum":
            gate_logits = self.gate(torch.cat(projected, dim=1))
            weights = torch.softmax(gate_logits, dim=1)
            stacked = torch.stack(projected, dim=1)
            fused = (weights.unsqueeze(-1) * stacked).sum(dim=1)
            return fused, weights
        tokens = torch.stack(projected, dim=1)
        if self.fusion == "transformer_cls":
            cls = self.cls_token.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat([cls, tokens], dim=1) + self.modality_embedding
            tokens = self.encoder_norm(self.encoder(tokens))
            return tokens[:, 0], None
        if self.use_modality_embedding:
            tokens = tokens + self.modality_embedding
        attended, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.attn_norm(tokens + attended)
        pooled_mean = tokens.mean(dim=1)
        pooled_max = tokens.max(dim=1).values
        return self.fuse(torch.cat([pooled_mean, pooled_max], dim=1)), None

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        fused, weights = self.fuse_features(self.project(batch))
        return self.head(fused, target), weights


class MultiFeatureDataset(torch.utils.data.Dataset[Any]):
    def __init__(self, features: dict[str, np.ndarray], labels: np.ndarray, group_labels: np.ndarray | None = None) -> None:
        self.features = features
        self.labels = labels
        self.group_labels = group_labels
        self.names = list(features)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        batch = {name: torch.from_numpy(self.features[name][index]) for name in self.names}
        label = torch.tensor(self.labels[index], dtype=torch.long)
        if self.group_labels is None:
            return batch, label
        return batch, label, torch.tensor(self.group_labels[index], dtype=torch.long)


def to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def predict(
    model: MultiFeatureFusionClassifier,
    features: dict[str, np.ndarray],
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    model.eval()
    logits_out: list[np.ndarray] = []
    weights_out: list[np.ndarray] = []
    names = list(features)
    with torch.no_grad():
        for start in range(0, len(next(iter(features.values()))), batch_size):
            batch = {
                name: torch.from_numpy(features[name][start : start + batch_size]).to(device)
                for name in names
            }
            logits, weights = model(batch)
            logits_out.append(logits.detach().cpu().numpy())
            if weights is not None:
                weights_out.append(weights.detach().cpu().numpy())
    logits_np = np.concatenate(logits_out, axis=0)
    if weights_out:
        return logits_np, np.concatenate(weights_out, axis=0)
    return logits_np, None


def evaluate(labels: list[str], y_true: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    pred = logits.argmax(axis=1)
    y_true_names = [labels[int(index)] for index in y_true]
    y_pred_names = [labels[int(index)] for index in pred]
    accuracy = float(accuracy_score(y_true_names, y_pred_names))
    macro_f1 = float(f1_score(y_true_names, y_pred_names, labels=labels, average="macro", zero_division=0))
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "selection_score": 0.5 * accuracy + 0.5 * macro_f1,
        "weighted_f1": float(
            f1_score(y_true_names, y_pred_names, labels=labels, average="weighted", zero_division=0)
        ),
        "classification_report": classification_report(
            y_true_names,
            y_pred_names,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true_names, y_pred_names, labels=labels).tolist(),
        "labels": labels,
    }


def plot_history(history: list[dict[str, float | int]], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    epochs = [int(row["epoch"]) for row in history]
    train_loss = [float(row["train_loss"]) for row in history]
    val_macro_f1 = [float(row["val_macro_f1"]) for row in history]
    val_accuracy = [float(row["val_accuracy"]) for row in history]
    val_selection_score = [float(row.get("val_selection_score", 0.5 * row["val_accuracy"] + 0.5 * row["val_macro_f1"])) for row in history]

    fig, ax_loss = plt.subplots(figsize=(9, 5))
    ax_loss.plot(epochs, train_loss, label="train loss", color="#1f77b4", linewidth=2)
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("cross entropy loss")
    ax_loss.grid(True, alpha=0.25)
    ax_metric = ax_loss.twinx()
    ax_metric.plot(epochs, val_macro_f1, label="val macro F1", color="#d62728", linewidth=2)
    ax_metric.plot(epochs, val_accuracy, label="val accuracy", color="#2ca02c", linewidth=2)
    ax_metric.plot(epochs, val_selection_score, label="0.5 acc + 0.5 F1", color="#7f3c8d", linewidth=2)
    ax_metric.set_ylabel("validation metric")
    ax_metric.set_ylim(0.0, 1.0)
    lines, labels = ax_loss.get_legend_handles_labels()
    lines2, labels2 = ax_metric.get_legend_handles_labels()
    ax_loss.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_confusion_matrix(
    matrix: list[list[int]],
    labels: list[str],
    output_path: Path,
    *,
    normalize_rows: bool,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        return
    counts = np.asarray(matrix, dtype=np.int64)
    if normalize_rows:
        values = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
        vmax = 1.0
    else:
        values = counts.astype(np.float32)
        vmax = max(float(np.percentile(values, 98)) if values.size else 1.0, 1.0)

    cmap = LinearSegmentedColormap.from_list(
        "soft_teal",
        ["#f8faf9", "#d7ece7", "#7db6ad", "#266b6d", "#12353d"],
    )
    fig, ax = plt.subplots(figsize=(11, 9.5))
    image = ax.imshow(values, interpolation="nearest", cmap=cmap, vmin=0.0, vmax=vmax)
    cbar = fig.colorbar(image, ax=ax, fraction=0.038, pad=0.03)
    cbar.set_label("row percentage" if normalize_rows else "count", rotation=270, labelpad=18)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, pad=14, fontsize=15)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row in range(counts.shape[0]):
        for col in range(counts.shape[1]):
            count = int(counts[row, col])
            value = float(values[row, col])
            if count == 0:
                text = "0"
                alpha = 0.35
            elif normalize_rows:
                text = f"{value * 100:.0f}%\n{count}"
                alpha = 1.0
            else:
                text = str(count)
                alpha = 1.0
            color = "white" if value > (0.55 if normalize_rows else vmax * 0.55) else "#111111"
            ax.text(col, row, text, ha="center", va="center", color=color, fontsize=7.5, alpha=alpha)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_gate_statistics(
    output_dir: Path,
    feature_names: list[str],
    weights: np.ndarray | None,
    rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if weights is None:
        return None
    overall = {
        feature_names[index]: {
            "mean": float(weights[:, index].mean()),
            "std": float(weights[:, index].std()),
            "min": float(weights[:, index].min()),
            "max": float(weights[:, index].max()),
        }
        for index in range(len(feature_names))
    }
    by_emotion: dict[str, dict[str, dict[str, float]]] = {}
    emotions = sorted({str(row.get("dominant_emotion")) for row in rows})
    for emotion in emotions:
        indices = [index for index, row in enumerate(rows) if str(row.get("dominant_emotion")) == emotion]
        if not indices:
            continue
        local = weights[indices]
        by_emotion[emotion] = {
            feature_names[index]: {
                "mean": float(local[:, index].mean()),
                "std": float(local[:, index].std()),
            }
            for index in range(len(feature_names))
        }

    stats = {"overall": overall, "by_emotion": by_emotion}
    json_path = output_dir / "gate_statistics.json"
    csv_path = output_dir / "gate_statistics.csv"
    json_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scope", "emotion", "feature", "mean", "std", "min", "max"])
        for feature, values in overall.items():
            writer.writerow(["overall", "", feature, values["mean"], values["std"], values["min"], values["max"]])
        for emotion, feature_values in by_emotion.items():
            for feature, values in feature_values.items():
                writer.writerow(["by_emotion", emotion, feature, values["mean"], values["std"], "", ""])
    return {"json": str(json_path), "csv": str(csv_path)}


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    labels = [str(label) for label in config.get("labels", DEFAULT_LABELS)]
    seed = int(config.get("seed", 42))
    set_seed(seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_feature_paths = parse_feature_specs(args.train_feature)
    val_feature_paths = parse_feature_specs(args.val_feature)
    if set(train_feature_paths) != set(val_feature_paths):
        raise SystemExit("Train and validation feature names must match.")
    feature_names = list(train_feature_paths)

    train_rows = read_jsonl(args.train_jsonl)
    val_rows = read_jsonl(args.val_jsonl)
    if not train_rows:
        raise SystemExit(f"No training rows found in {args.train_jsonl}")
    if not val_rows:
        raise SystemExit(f"No validation rows found in {args.val_jsonl}")
    train_y = label_array(train_rows, labels)
    val_y = label_array(val_rows, labels)
    group_map = {str(key): str(value) for key, value in config.get("group_map", DEFAULT_GROUPS).items()}
    group_names = [str(value) for value in config.get("group_names", DEFAULT_GROUP_NAMES)]
    group_aux_weight = float(config.get("group_aux_weight", 0.0))
    train_group_y = group_array(train_rows, group_names, group_map) if group_aux_weight > 0 else None

    train_features: dict[str, np.ndarray] = {}
    val_features: dict[str, np.ndarray] = {}
    scalers: dict[str, dict[str, np.ndarray]] = {}
    feature_dims: dict[str, int] = {}
    feature_diagnostics: dict[str, Any] = {}

    for name in feature_names:
        train_x, train_missing, train_duplicates = align_feature(train_rows, train_feature_paths[name])
        val_x, val_missing, val_duplicates = align_feature(val_rows, val_feature_paths[name])
        train_x, val_x, mean, std = standardize_pair(train_x, val_x)
        train_features[name] = train_x
        val_features[name] = val_x
        scalers[name] = {"mean": mean, "std": std}
        feature_dims[name] = int(train_x.shape[1])
        feature_diagnostics[name] = {
            "train_path": str(train_feature_paths[name]),
            "val_path": str(val_feature_paths[name]),
            "dim": int(train_x.shape[1]),
            "train_missing": train_missing,
            "val_missing": val_missing,
            "train_duplicate_feature_ids": train_duplicates,
            "val_duplicate_feature_ids": val_duplicates,
        }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiFeatureFusionClassifier(
        feature_dims=feature_dims,
        fusion=str(config["fusion"]),
        hidden_size=int(config.get("hidden_size", 512)),
        num_classes=len(labels),
        dropout=float(config.get("dropout", 0.3)),
        attention_heads=int(config.get("attention_heads", 8)),
        modality_dropout=float(config.get("modality_dropout", 0.0)),
        transformer_layers=int(config.get("transformer_layers", 2)),
        use_modality_embedding=bool(config.get("use_modality_embedding", False)),
        head_type=str(config.get("head_type", "mlp")),
        head_scale=float(config.get("head_scale", config.get("scale", 30.0))),
        head_margin=float(config.get("head_margin", config.get("margin", 0.2))),
    ).to(device)

    group_head: nn.Module | None = None
    if group_aux_weight > 0:
        group_head = nn.Sequential(
            nn.LayerNorm(int(config.get("hidden_size", 512))),
            nn.Linear(int(config.get("hidden_size", 512)), len(group_names)),
        ).to(device)

    loss_type = str(config.get("loss_type", "cross_entropy"))
    train_class_counts = np.bincount(train_y, minlength=len(labels)).astype(np.float32)
    weight = None
    if loss_type == "balanced_softmax":
        criterion = BalancedSoftmaxLoss(
            train_class_counts,
            label_smoothing=float(config.get("label_smoothing", 0.0)),
        )
    elif loss_type == "cross_entropy":
        if bool(config.get("class_weight", True)):
            weight = torch.from_numpy(class_weights(train_y, len(labels))).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=float(config.get("label_smoothing", 0.0)))
    else:
        raise SystemExit(f"Unknown loss_type {loss_type!r}; expected cross_entropy or balanced_softmax.")
    criterion = criterion.to(device)
    group_criterion = nn.CrossEntropyLoss(label_smoothing=float(config.get("group_label_smoothing", 0.0)))
    optimizer_params = list(model.parameters())
    if group_head is not None:
        optimizer_params.extend(group_head.parameters())
    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=float(config.get("lr", 3e-4)),
        weight_decay=float(config.get("weight_decay", 1e-3)),
    )

    batch_size = int(config.get("batch_size", 512))
    train_dataset = MultiFeatureDataset(train_features, train_y, train_group_y)
    sampler = None
    shuffle = True
    oversample_power = float(config.get("oversample_power", 0.0))
    if oversample_power > 0:
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights(train_y, oversample_power)),
            num_samples=len(train_y),
            replacement=True,
        )
        shuffle = False
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=0)

    epochs = int(config.get("epochs", 60))
    patience = int(config.get("patience", 8))
    best_score = -1.0
    best_accuracy = -1.0
    best_macro_f1 = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_group_head_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for batch_item in train_loader:
            if len(batch_item) == 3:
                batch, y_batch, group_batch = batch_item
                group_batch = group_batch.to(device)
            else:
                batch, y_batch = batch_item
                group_batch = None
            batch = to_device(batch, device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            if group_head is not None and group_batch is not None:
                fused, gate_weights = model.fuse_features(model.project(batch))
                logits = model.head(fused, y_batch)
                group_logits = group_head(fused)
            else:
                logits, gate_weights = model(batch, y_batch)
                group_logits = None
            loss = criterion(logits, y_batch)
            if group_logits is not None and group_batch is not None:
                loss = loss + group_aux_weight * group_criterion(group_logits, group_batch)
            gate_entropy_weight = float(config.get("gate_entropy_weight", 0.0))
            if gate_entropy_weight > 0 and gate_weights is not None:
                entropy = -(gate_weights * torch.log(gate_weights.clamp_min(1e-8))).sum(dim=1).mean()
                loss = loss - gate_entropy_weight * entropy
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(y_batch)
            total_seen += len(y_batch)

        val_logits, _ = predict(model, val_features, batch_size, device)
        metrics = evaluate(labels, val_y, val_logits)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_seen, 1),
            "val_accuracy": metrics["accuracy"],
            "val_macro_f1": metrics["macro_f1"],
            "val_selection_score": metrics["selection_score"],
            "val_weighted_f1": metrics["weighted_f1"],
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

        if metrics["selection_score"] > best_score:
            best_score = float(metrics["selection_score"])
            best_accuracy = float(metrics["accuracy"])
            best_macro_f1 = float(metrics["macro_f1"])
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            if group_head is not None:
                best_group_head_state = {
                    key: value.detach().cpu().clone()
                    for key, value in group_head.state_dict().items()
                }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if group_head is not None and best_group_head_state is not None:
        group_head.load_state_dict(best_group_head_state)

    final_logits, gate_weights = predict(model, val_features, batch_size, device)
    final_metrics = evaluate(labels, val_y, final_logits)
    pred_ids = final_logits.argmax(axis=1)

    predictions_path = args.output_dir / "val_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for index, (row, pred_id, logits) in enumerate(zip(val_rows, pred_ids, final_logits)):
            output = dict(row)
            output["predicted_emotion"] = labels[int(pred_id)]
            output["correct"] = output.get("dominant_emotion") == output["predicted_emotion"]
            output["logits"] = {label: float(logits[label_index]) for label_index, label in enumerate(labels)}
            if gate_weights is not None:
                output["gate_weights"] = {
                    name: float(gate_weights[index, name_index]) for name_index, name in enumerate(feature_names)
                }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")

    history_path = args.output_dir / "history.jsonl"
    with history_path.open("w", encoding="utf-8") as handle:
        for row in history:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    model_path = args.output_dir / "best_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "labels": labels,
            "feature_names": feature_names,
            "feature_dims": feature_dims,
            "scalers": scalers,
            "group_head_state_dict": group_head.state_dict() if group_head is not None else None,
            "group_names": group_names if group_head is not None else None,
            "group_map": group_map if group_head is not None else None,
            "best_epoch": best_epoch,
            "best_selection_score": best_score,
            "best_accuracy": best_accuracy,
            "best_macro_f1": best_macro_f1,
        },
        model_path,
    )

    curve_path = args.output_dir / "training_curve.png"
    cm_counts_path = args.output_dir / "confusion_matrix_counts.png"
    cm_norm_path = args.output_dir / "confusion_matrix_normalized.png"
    plot_history(history, curve_path)
    plot_confusion_matrix(
        final_metrics["confusion_matrix"],
        labels,
        cm_counts_path,
        normalize_rows=False,
        title="Emotion Confusion Matrix (Counts)",
    )
    plot_confusion_matrix(
        final_metrics["confusion_matrix"],
        labels,
        cm_norm_path,
        normalize_rows=True,
        title="Emotion Confusion Matrix (Row-Normalized)",
    )
    gate_outputs = save_gate_statistics(args.output_dir, feature_names, gate_weights, val_rows)

    summary_path = args.output_dir / "summary.json"
    summary = {
        "config": config,
        "train_jsonl": str(args.train_jsonl),
        "val_jsonl": str(args.val_jsonl),
        "device": device,
        "feature_names": feature_names,
        "feature_dims": feature_dims,
        "feature_diagnostics": feature_diagnostics,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_counts": dict(Counter(labels[int(index)] for index in train_y)),
        "val_counts": dict(Counter(labels[int(index)] for index in val_y)),
        "train_class_counts": {label: int(train_class_counts[index]) for index, label in enumerate(labels)},
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "best_accuracy": best_accuracy,
        "best_macro_f1": best_macro_f1,
        "metrics": final_metrics,
        "outputs": {
            "model": str(model_path),
            "predictions": str(predictions_path),
            "history": str(history_path),
            "training_curve": str(curve_path),
            "confusion_matrix_counts": str(cm_counts_path),
            "confusion_matrix_normalized": str(cm_norm_path),
            "gate_statistics": gate_outputs,
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "selection_score": final_metrics["selection_score"],
                "accuracy": final_metrics["accuracy"],
                "macro_f1": final_metrics["macro_f1"],
                "weighted_f1": final_metrics["weighted_f1"],
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
