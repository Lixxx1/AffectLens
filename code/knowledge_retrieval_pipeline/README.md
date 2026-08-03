# Multi-Encoder Knowledge Retrieval

This pipeline retrieves visually similar labeled artworks independently with
three encoders, then ranks references by cross-encoder agreement. Feature rows
are aligned through `image_ids`; NPZ and manifest row order need not match.

## Run the complete pipeline

Copy `configs/retrieval.example.json`, update its paths, and run:

```bash
python3 code/knowledge_retrieval_pipeline/run_retrieval_pipeline.py \
  --config configs/retrieval.example.json
```

Use `--dry-run` to validate the configuration and inspect the generated
commands without reading feature files.

Each encoder writes a top-k JSONL file, a flat CSV file, and similarity
statistics. The consensus stage first selects references returned by all three
encoders. If `min_keep` is greater than zero, two- and one-encoder matches fill
any remaining slots in rank order.

## Inputs

Every feature NPZ must contain `x` and `image_ids`. Query and pool features for
an encoder must have the same dimension and preprocessing. The corresponding
JSONL manifests should contain stable image identifiers; pool rows may also
provide `dominant_emotion`, `valence`, and `arousal` annotations.

Generate compatible embeddings with
`code/feature_extraction/extract_image_features.py`, or with another extractor
that preserves the same NPZ contract.

Retrieved labels are evidence, not ground truth for the query. Weak or
conflicting neighbors should not override direct visual evidence.
