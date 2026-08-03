from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract normalized image embeddings into an AffectLens NPZ file."
    )
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=("resnet50", "resnet101", "huggingface", "timm"), required=True
    )
    parser.add_argument("--model", help="Model id for the huggingface or timm backend.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--max-side", type=int, default=1024)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise SystemExit(f"{path}:{line_number} is not a JSON object")
            image_id = row.get("image_id") or row.get("sample_id")
            image_path = row.get("local_image_path") or row.get("image_path")
            if not image_id or not image_path:
                raise SystemExit(
                    f"{path}:{line_number} requires image_id/sample_id and "
                    "local_image_path/image_path"
                )
            rows.append({**row, "_image_id": str(image_id), "_image_path": str(image_path)})
    if not rows:
        raise SystemExit(f"Manifest is empty: {path}")
    counts = Counter(row["_image_id"] for row in rows)
    duplicate_ids = sorted(image_id for image_id, count in counts.items() if count > 1)
    if duplicate_ids:
        raise SystemExit(f"Manifest contains duplicate image ids: {duplicate_ids[:10]}")
    return rows


def resolve_image_path(manifest: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest.parent / path


def open_rgb(path: Path, max_side: int) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    if max_side > 0:
        image.thumbnail((max_side, max_side))
    return image


def resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"CUDA was requested but is unavailable: {requested}")
    return requested


def extract_huggingface_features(model: Any, inputs: dict[str, Any]) -> Any:
    get_image_features = getattr(model, "get_image_features", None)
    if callable(get_image_features):
        return get_image_features(**inputs)
    output = model(**inputs)
    pooled = getattr(output, "pooler_output", None)
    if pooled is not None:
        return pooled
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None:
        raise TypeError(f"Unsupported model output: {type(output)!r}")
    return hidden[:, 0]


def load_backend(args: argparse.Namespace) -> tuple[Any, Callable, Callable, Any, str, str]:
    import torch

    device = resolve_device(torch, args.device)
    if args.backend in {"resnet50", "resnet101"}:
        from torch import nn
        from torchvision import models

        weights = (
            models.ResNet50_Weights.DEFAULT
            if args.backend == "resnet50"
            else models.ResNet101_Weights.DEFAULT
        )
        base = (
            models.resnet50(weights=weights)
            if args.backend == "resnet50"
            else models.resnet101(weights=weights)
        )
        model = nn.Sequential(*list(base.children())[:-1]).to(device).eval()
        preprocess = weights.transforms()

        def encode(images: list[Image.Image]) -> Any:
            return model(torch.stack([preprocess(image) for image in images]).to(device)).flatten(1)

        return model, encode, preprocess, torch, device, args.backend

    if not args.model:
        raise SystemExit("--model is required for the huggingface and timm backends")
    if args.backend == "huggingface":
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(args.model)
        model = AutoModel.from_pretrained(args.model).to(device).eval()

        def encode(images: list[Image.Image]) -> Any:
            inputs = {key: value.to(device) for key, value in processor(images=images, return_tensors="pt").items()}
            return extract_huggingface_features(model, inputs)

        return model, encode, processor, torch, device, args.model

    import timm
    from timm.data import create_transform, resolve_data_config

    model = timm.create_model(args.model, pretrained=True, num_classes=0).to(device).eval()
    preprocess = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

    def encode(images: list[Image.Image]) -> Any:
        features = model(torch.stack([preprocess(image) for image in images]).to(device))
        return features.flatten(1) if features.ndim > 2 else features

    return model, encode, preprocess, torch, device, args.model


def normalize_features(features: Any, torch: Any) -> np.ndarray:
    return (
        torch.nn.functional.normalize(features.float(), p=2, dim=1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    rows = read_manifest(args.manifest_jsonl)
    _, encode, _, torch, device, model_name = load_backend(args)
    embeddings: list[np.ndarray] = []
    image_ids: list[str] = []
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start : start + args.batch_size]
            images = [
                open_rgb(resolve_image_path(args.manifest_jsonl, row["_image_path"]), args.max_side)
                for row in batch_rows
            ]
            embeddings.append(normalize_features(encode(images), torch))
            image_ids.extend(row["_image_id"] for row in batch_rows)
            print(f"encoded {len(image_ids)}/{len(rows)}", flush=True)

    x = np.concatenate(embeddings, axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, x=x, image_ids=np.asarray(image_ids))
    summary = {
        "manifest": str(args.manifest_jsonl),
        "output": str(args.output),
        "backend": args.backend,
        "model": model_name,
        "device": device,
        "rows": len(image_ids),
        "feature_dim": int(x.shape[1]),
        "normalized": True,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
