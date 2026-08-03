# Image Feature Extraction

This utility converts an image manifest into the aligned NPZ format used by
the fusion and retrieval pipelines. The output contains:

- `x`: a two-dimensional `float32` embedding matrix;
- `image_ids`: one stable identifier per row in `x`.

Each JSONL manifest row must provide `image_id` (or `sample_id`) and
`local_image_path` (or `image_path`). Relative image paths are resolved from
the manifest directory.

```json
{"image_id": "artwork-001", "local_image_path": "images/artwork-001.jpg"}
```

## Examples

ResNet-50:

```bash
python3 code/feature_extraction/extract_image_features.py \
  --manifest-jsonl data/train.jsonl \
  --output data/features/train/resnet50.npz \
  --backend resnet50
```

Hugging Face DINOv2:

```bash
python3 code/feature_extraction/extract_image_features.py \
  --manifest-jsonl data/train.jsonl \
  --output data/features/train/dinov2.npz \
  --backend huggingface \
  --model facebook/dinov2-large \
  --device cuda
```

Any image model supported by `timm` can be selected with `--backend timm` and
`--model`. Install the matching optional dependencies from
`requirements-feature-extraction.txt`.

Extract train, validation, inference, and retrieval-pool features separately.
Always use the same model and preprocessing for query and pool features.
