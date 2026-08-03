from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent / "scripts"
REQUIRED_ROLES = ("clip", "openai_clip", "dinov3")
SAFE_ENCODER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
ENCODER_STRING_FIELDS = ("role", "name", "query_features", "pool_features")
CONFIG_PATH_FIELDS = ("query_manifest", "pool_manifest", "output_dir")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run three-encoder retrieval and build a consensus reference set."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise SystemExit("Retrieval config must be a JSON object")
    encoders = config.get("encoders")
    if not isinstance(encoders, list) or len(encoders) != 3:
        raise SystemExit("Retrieval config requires exactly three encoders")
    for index, encoder in enumerate(encoders):
        if not isinstance(encoder, dict):
            raise SystemExit(f"Encoder {index} must be a JSON object")
        invalid = [
            key
            for key in ENCODER_STRING_FIELDS
            if not isinstance(encoder.get(key), str) or not encoder[key].strip()
        ]
        if invalid:
            raise SystemExit(
                f"Encoder {index} requires non-empty string fields: {', '.join(invalid)}"
            )
        name = encoder["name"]
        if name in {".", ".."} or SAFE_ENCODER_NAME.fullmatch(name) is None:
            raise SystemExit(
                f"Encoder name must be a safe single path segment: {name!r}"
            )
    roles = [encoder["role"] for encoder in encoders]
    if sorted(roles) != sorted(REQUIRED_ROLES):
        raise SystemExit(f"Encoder roles must be exactly: {', '.join(REQUIRED_ROLES)}")
    for key in CONFIG_PATH_FIELDS:
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise SystemExit(f"Retrieval config requires a non-empty string {key!r}")
    for key, default in (("top_k", 10), ("min_keep", 0), ("max_keep", 10)):
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise SystemExit(f"{key} must be an integer")
    top_k = config.get("top_k", 10)
    min_keep = config.get("min_keep", 0)
    max_keep = config.get("max_keep", 10)
    if top_k < 1:
        raise SystemExit("top_k must be at least 1")
    if max_keep < 1:
        raise SystemExit("max_keep must be at least 1")
    if min_keep < 0 or min_keep > max_keep:
        raise SystemExit("min_keep must be between 0 and max_keep")
    return config


def resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def build_commands(config_path: Path, config: dict[str, Any]) -> list[list[str]]:
    output_root = resolve(config_path, config["output_dir"])
    query_manifest = resolve(config_path, config["query_manifest"])
    pool_manifest = resolve(config_path, config["pool_manifest"])
    top_k = config.get("top_k", 10)
    min_keep = config.get("min_keep", 0)
    max_keep = config.get("max_keep", 10)

    commands: list[list[str]] = []
    result_paths: dict[str, Path] = {}
    for encoder in config["encoders"]:
        encoder_output = output_root / encoder["name"]
        commands.append(
            [
                sys.executable,
                str(SCRIPT_DIR / "retrieve_references.py"),
                "--model-name",
                encoder["name"],
                "--query-features",
                str(resolve(config_path, encoder["query_features"])),
                "--pool-features",
                str(resolve(config_path, encoder["pool_features"])),
                "--query-jsonl",
                str(query_manifest),
                "--pool-jsonl",
                str(pool_manifest),
                "--output-dir",
                str(encoder_output),
                "--top-k",
                str(top_k),
            ]
        )
        result_paths[encoder["role"]] = encoder_output / f"{encoder['name']}_top{top_k}.jsonl"

    commands.append(
        [
            sys.executable,
            str(SCRIPT_DIR / "build_reference_consensus.py"),
            "--clip",
            str(result_paths["clip"]),
            "--openai-clip",
            str(result_paths["openai_clip"]),
            "--dinov3",
            str(result_paths["dinov3"]),
            "--output-dir",
            str(output_root / "consensus"),
            "--min-keep",
            str(min_keep),
            "--max-keep",
            str(max_keep),
        ]
    )
    return commands


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    commands = build_commands(args.config.resolve(), config)
    for command in commands:
        print(json.dumps(command, ensure_ascii=False), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
