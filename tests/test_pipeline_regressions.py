from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


retrieve = load_module(
    "test_retrieve_references",
    PROJECT_ROOT / "code/knowledge_retrieval_pipeline/scripts/retrieve_references.py",
)
consensus = load_module(
    "test_build_reference_consensus",
    PROJECT_ROOT
    / "code/knowledge_retrieval_pipeline/scripts/build_reference_consensus.py",
)
exporter = load_module(
    "test_export_reference_folders",
    PROJECT_ROOT
    / "code/knowledge_retrieval_pipeline/scripts/export_reference_folders.py",
)
runner = load_module(
    "test_run_inference",
    PROJECT_ROOT / "code/final_inference/run_inference.py",
)
extractor = load_module(
    "test_extract_image_features",
    PROJECT_ROOT / "code/feature_extraction/extract_image_features.py",
)
retrieval_pipeline = load_module(
    "test_run_retrieval_pipeline",
    PROJECT_ROOT / "code/knowledge_retrieval_pipeline/run_retrieval_pipeline.py",
)


class PipelineRegressionTests(unittest.TestCase):
    def test_huggingface_clip_uses_image_features(self):
        class FakeClipModel:
            def __init__(self):
                self.inputs = None

            def get_image_features(self, **inputs):
                self.inputs = inputs
                return "image-features"

            def __call__(self, **inputs):
                raise AssertionError("CLIP forward should not be used for image-only inputs")

        model = FakeClipModel()
        result = extractor.extract_huggingface_features(model, {"pixel_values": "pixels"})
        self.assertEqual(result, "image-features")
        self.assertEqual(model.inputs, {"pixel_values": "pixels"})

    def test_feature_manifest_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.jsonl"
            manifest.write_text(
                "\n".join(
                    [
                        json.dumps({"image_id": "same", "image_path": "a.jpg"}),
                        json.dumps({"image_id": "same", "image_path": "b.jpg"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SystemExit, "duplicate image ids"):
                extractor.read_manifest(manifest)

    def test_retrieval_config_builds_three_retrievals_and_consensus(self):
        config_path = PROJECT_ROOT / "configs/retrieval.example.json"
        config = retrieval_pipeline.load_config(config_path)
        commands = retrieval_pipeline.build_commands(config_path, config)
        self.assertEqual(len(commands), 4)
        self.assertTrue(all("retrieve_references.py" in command[1] for command in commands[:3]))
        self.assertIn("build_reference_consensus.py", commands[3][1])

    def test_retrieval_config_rejects_invalid_values(self):
        cases = [
            ("non-string field", [("encoders", 0, "query_features", 123)], "non-empty string fields"),
            ("unsafe name", [("encoders", 0, "name", "../outside")], "safe single path segment"),
            ("case-insensitive duplicate", [("encoders", 1, "name", "CLIP_H14")], "unique, ignoring case"),
            ("reserved name", [("encoders", 0, "name", "Consensus")], "reserved output directories"),
            ("invalid keep bounds", [("min_keep", 5), ("max_keep", 4)], "between 0 and max_keep"),
        ]
        example = PROJECT_ROOT / "configs/retrieval.example.json"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            for description, updates, error in cases:
                with self.subTest(description=description):
                    config = json.loads(example.read_text())
                    for update in updates:
                        if update[0] == "encoders":
                            config["encoders"][update[1]][update[2]] = update[3]
                        else:
                            config[update[0]] = update[1]
                    path.write_text(json.dumps(config), encoding="utf-8")
                    with self.assertRaisesRegex(SystemExit, error):
                        retrieval_pipeline.load_config(path)

    def test_default_inference_output_is_results_json_only(self):
        old_argv = sys.argv
        try:
            sys.argv = ["run_inference.py"]
            args = runner.parse_args()
        finally:
            sys.argv = old_argv

        self.assertEqual(args.output, PROJECT_ROOT / "results.json")
        self.assertIsNone(args.raw_output)
        self.assertIsNone(args.run_log)
        self.assertIsNone(args.stats_output)
        self.assertFalse(hasattr(args, "zip_output"))

    def test_ambiguous_stem_alias_is_removed(self):
        first = {"image_id": "style_a/shared.jpg"}
        second = {"image_id": "style_b/shared.jpg"}
        index = retrieve.build_row_index([first, second])

        self.assertNotIn("shared", index)
        self.assertIs(index["style_a/shared.jpg"], first)
        self.assertIs(index["style_b/shared.jpg"], second)

    def test_retrieval_uses_manifest_sample_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            query_features = root / "query.npz"
            pool_features = root / "pool.npz"
            query_jsonl = root / "query.jsonl"
            pool_jsonl = root / "pool.jsonl"
            output_dir = root / "out"

            np.savez(
                query_features,
                x=np.asarray([[1.0, 0.0]], dtype=np.float32),
                image_ids=np.asarray(["data/inference/images/query.jpg"]),
            )
            np.savez(
                pool_features,
                x=np.asarray([[1.0, 0.0]], dtype=np.float32),
                image_ids=np.asarray(["pool/reference.jpg"]),
            )
            query_jsonl.write_text(
                json.dumps(
                    {
                        "sample_id": "query",
                        "image_id": "query",
                        "image_path": "data/inference/images/query.jpg",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            pool_jsonl.write_text(
                json.dumps(
                    {
                        "image_id": "reference",
                        "image_path": "pool/reference.jpg",
                        "dominant_emotion": "Calm",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            old_argv = sys.argv
            try:
                sys.argv = [
                    "retrieve_references.py",
                    "--model-name",
                    "test",
                    "--query-features",
                    str(query_features),
                    "--pool-features",
                    str(pool_features),
                    "--query-jsonl",
                    str(query_jsonl),
                    "--pool-jsonl",
                    str(pool_jsonl),
                    "--output-dir",
                    str(output_dir),
                    "--top-k",
                    "1",
                ]
                with contextlib.redirect_stdout(io.StringIO()):
                    retrieve.main()
            finally:
                sys.argv = old_argv

            row = json.loads(
                (output_dir / "test_top1.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(row["sample_id"], "query")
            self.assertEqual(
                row["query_feature_id"],
                "data/inference/images/query.jpg",
            )

    def test_consensus_rejects_mismatched_sample_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip = root / "clip.jsonl"
            openai = root / "openai.jsonl"
            dino = root / "dino.jsonl"
            clip.write_text(
                "\n".join(
                    [
                        json.dumps({"sample_id": "s1", "query_row": {}, "top_k": []}),
                        json.dumps({"sample_id": "s2", "query_row": {}, "top_k": []}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            for path in (openai, dino):
                path.write_text(
                    json.dumps({"sample_id": "s1", "query_row": {}, "top_k": []})
                    + "\n",
                    encoding="utf-8",
                )

            old_argv = sys.argv
            try:
                sys.argv = [
                    "build_reference_consensus.py",
                    "--clip",
                    str(clip),
                    "--openai-clip",
                    str(openai),
                    "--dinov3",
                    str(dino),
                    "--output-dir",
                    str(root / "out"),
                ]
                with self.assertRaisesRegex(SystemExit, "sample sets differ"):
                    consensus.main()
            finally:
                sys.argv = old_argv

    def test_export_rerun_clears_stale_rank_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            query = root / "query.jpg"
            ref_a = root / "ref_a.jpg"
            ref_b = root / "ref_b.jpg"
            for path in (query, ref_a, ref_b):
                path.touch()
            index_path = root / "index.jsonl"
            output_dir = root / "out"

            def run_export(reference: Path, emotion: str) -> None:
                index_path.write_text(
                    json.dumps(
                        {
                            "sample_id": "sample",
                            "query_row": {"local_image_path": str(query)},
                            "top_k": [
                                {
                                    "rank": 1,
                                    "score": 0.9,
                                    "local_image_path": str(reference),
                                    "dominant_emotion": emotion,
                                }
                            ],
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                old_argv = sys.argv
                try:
                    sys.argv = [
                        "export_reference_folders.py",
                        "--index-jsonl",
                        str(index_path),
                        "--output-dir",
                        str(output_dir),
                        "--max-images",
                        "1",
                    ]
                    with contextlib.redirect_stdout(io.StringIO()):
                        exporter.main()
                finally:
                    sys.argv = old_argv

            run_export(ref_a, "Calm")
            run_export(ref_b, "Sad")

            rank_files = list((output_dir / "sample").glob("rank01_*"))
            self.assertEqual(len(rank_files), 1)
            self.assertIn("Sad", rank_files[0].name)

    def test_all_supported_target_formats_are_discovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp)
            images = input_dir / "images"
            images.mkdir()
            for name in ("a.jpg", "b.jpeg", "c.png", "d.webp", "e.bmp"):
                (images / name).touch()

            self.assertEqual(
                [path.name for path in runner.list_samples(input_dir)],
                ["a.jpg", "b.jpeg", "c.png", "d.webp", "e.bmp"],
            )

    def test_dry_run_does_not_delete_existing_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [
                root / name
                for name in (
                    "out.json",
                    "raw.jsonl",
                    "run.jsonl",
                    "stats.json",
                )
            ]
            for path in paths:
                path.write_text("keep", encoding="utf-8")
            args = argparse.Namespace(
                output=paths[0],
                raw_output=paths[1],
                run_log=paths[2],
                stats_output=paths[3],
                overwrite=True,
                resume=False,
                dry_run=True,
            )

            runner.prepare_outputs(args)

            self.assertTrue(all(path.read_text(encoding="utf-8") == "keep" for path in paths))

    def test_duplicate_auxiliary_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {"sample_id": "same", "predicted_emotion": "Calm"}
                        ),
                        json.dumps(
                            {"image_id": "same.jpg", "predicted_emotion": "Sad"}
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(SystemExit, "duplicates sample_id"):
                runner.load_auxiliary_predictions(path)

    def test_missing_auxiliary_coverage_is_rejected(self):
        selected = [Path("first.jpg"), Path("second.png")]
        with self.assertRaisesRegex(SystemExit, "missing auxiliary predictions"):
            runner.validate_auxiliary_coverage(
                {"first": {"predicted_emotion": "Calm"}},
                selected,
                Path("predictions.jsonl"),
            )

    def test_missing_named_reference_does_not_fall_back_to_stale_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "rank01_old.jpg").touch()

            resolved = runner.reference_image_path(
                folder,
                {"file": "rank01_new.jpg"},
                1,
            )

            self.assertIsNone(resolved)

    def test_final_inference_dry_run_validates_prepared_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            images = input_dir / "images"
            sample_folder = root / "references" / "sample"
            images.mkdir(parents=True)
            sample_folder.mkdir(parents=True)
            Image.new("RGB", (8, 8), "white").save(images / "sample.png")

            top_k = []
            for rank in range(1, 5):
                file_name = f"rank{rank:02d}_ref.jpg"
                Image.new("RGB", (8, 8), "white").save(sample_folder / file_name)
                top_k.append(
                    {
                        "rank": rank,
                        "file": file_name,
                        "dominant_emotion": "Calm",
                        "valence": "Positive",
                        "arousal": "Low",
                    }
                )
            (sample_folder / "annotation.json").write_text(
                json.dumps({"sample_id": "sample", "top_k": top_k}),
                encoding="utf-8",
            )

            old_argv = sys.argv
            try:
                sys.argv = [
                    "run_inference.py",
                    "--input-dir",
                    str(input_dir),
                    "--reference-dir",
                    str(root / "references"),
                    "--run-log",
                    str(root / "dry_run.jsonl"),
                    "--output",
                    str(root / "results.json"),
                    "--raw-output",
                    str(root / "raw.jsonl"),
                    "--stats-output",
                    str(root / "summary.json"),
                    "--all",
                    "--dry-run",
                    "--workers",
                    "1",
                ]
                with contextlib.redirect_stdout(io.StringIO()):
                    runner.main()
            finally:
                sys.argv = old_argv

            log_row = json.loads(
                (root / "dry_run.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(log_row["sample_id"], "sample")
            self.assertEqual(log_row["status"], "dry_run")
            self.assertFalse((root / "results.json").exists())


if __name__ == "__main__":
    unittest.main()
