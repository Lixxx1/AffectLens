from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent / "scripts"
REQUIRED_ROLES = ("clip", "openai_clip", "dinov3")


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
    roles = [encoder.get("role") for encoder in encoders if isinstance(encoder, dict)]
    if sorted(roles) != sorted(REQUIRED_ROLES):
        raise SystemExit(f"Encoder roles must be exactly: {', '.join(REQUIRED_ROLES)}")
    for encoder in encoders:
        missing = [key for key in ("name", "query_features", "pool_features") if not encoder.get(key)]
        if missing:
            raise SystemExit(f"Encoder {encoder.get('role')!r} is missing: {', '.join(missing)}")
    for key in ("query_manifest", "pool_manifest", "output_dir"):
        if not config.get(key):
            raise SystemExit(f"Retrieval config is missing {key!r}")
    return config


def resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def build_commands(config_path: Path, config: dict[str, Any]) -> list[list[str]]:
    output_root = resolve(config_path, config["output_dir"])
    query_manifest = resolve(config_path, config["query_manifest"])
    pool_manifest = resolve(config_path, config["pool_manifest"])
    top_k = int(config.get("top_k", 10))
    if top_k < 1:
        raise SystemExit("top_k must be at least 1")

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
            str(int(config.get("min_keep", 0))),
            "--max-keep",
            str(int(config.get("max_keep", 10))),
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
