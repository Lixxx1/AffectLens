from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


REPO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = REPO_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metrics import (
    OUTPUT_AROUSAL_LABELS,
    OUTPUT_EMOTION_LABELS,
    OUTPUT_VALENCE_LABELS,
    emotion_binding_mismatch,
    normalize_task_label,
    to_output_emotion,
)
from src.prompts import SYSTEM_PROMPT, build_analysis_prompt

DEFAULT_INPUT_DIR = PROJECT_ROOT / "data/inference"
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "outputs/references"
DEFAULT_REFERENCE_COUNT = 4
DEFAULT_OUTPUT = PROJECT_ROOT / "results.json"
DEFAULT_SCHEMA = REPO_ROOT / "configs/analysis_response_schema.json"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_CODEX_MODEL = "gpt-5.5"

EMOTIONS = OUTPUT_EMOTION_LABELS
VALENCES = OUTPUT_VALENCE_LABELS
AROUSALS = OUTPUT_AROUSAL_LABELS
TEXT_FIELDS = ("overall_caption", "brushstroke", "composition", "color", "line", "light")
RESULT_FIELDS = (
    "sample_id",
    "emotion",
    "emotional_valence",
    "emotional_arousal_level",
    *TEXT_FIELDS,
)
RETRY_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
ALLOWED_REFERENCE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_TARGET_SUFFIXES = ALLOWED_REFERENCE_SUFFIXES


class RunError(RuntimeError):
    pass


ACTIVE_PROCESSES: set[subprocess.Popen[str]] = set()
ACTIVE_PROCESSES_LOCK = threading.Lock()


def register_active_process(process: subprocess.Popen[str]) -> None:
    with ACTIVE_PROCESSES_LOCK:
        ACTIVE_PROCESSES.add(process)


def unregister_active_process(process: subprocess.Popen[str]) -> None:
    with ACTIVE_PROCESSES_LOCK:
        ACTIVE_PROCESSES.discard(process)


def terminate_active_processes() -> None:
    with ACTIVE_PROCESSES_LOCK:
        processes = list(ACTIVE_PROCESSES)
    for process in processes:
        terminate_process_tree(process)


def install_interrupt_handler() -> None:
    previous_handler = signal.getsignal(signal.SIGINT)

    def handle_sigint(signum: int, frame: object) -> None:
        terminate_active_processes()
        if callable(previous_handler):
            previous_handler(signum, frame)
        else:
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_sigint)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run artwork emotion inference and write results.json."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=DEFAULT_REFERENCE_DIR,
        help=(
            "Directory containing one prepared reference folder per sample. "
            "Used as the Codex CLI working directory so prompt context can "
            "refer to query/rank images and annotation.json."
        ),
    )
    parser.add_argument(
        "--reference-count",
        type=int,
        default=DEFAULT_REFERENCE_COUNT,
        help="Number of ranked reference images to attach after the target image for Codex CLI.",
    )
    parser.add_argument(
        "--reference-attachment-mode",
        choices=("framed", "raw"),
        default="raw",
        help=(
            "How to attach reference images for Codex CLI. "
            "'framed' draws NOT TARGET cards; 'raw' attaches the original rank image files directly."
        ),
    )
    parser.add_argument(
        "--auxiliary-predictions",
        type=Path,
        default=None,
        help=(
            "Optional JSONL predictions from the multi-feature fusion model. "
            "The predicted emotion is supplied as a non-binding calibration prior."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=None,
        help="Optional JSONL file for raw model responses.",
    )
    parser.add_argument(
        "--run-log",
        type=Path,
        default=None,
        help="Optional JSONL file for per-sample run metadata.",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=None,
        help="Optional JSON file for aggregate run statistics.",
    )
    parser.add_argument(
        "--schema",
        "--codex-output-schema",
        dest="codex_output_schema",
        type=Path,
        default=DEFAULT_SCHEMA,
        help="JSON schema passed to codex exec --output-schema.",
    )
    parser.add_argument(
        "--codex-log-dir",
        type=Path,
        default=None,
        help="Optional directory for per-sample Codex CLI debug logs.",
    )
    parser.add_argument("--provider", choices=("openai-compatible", "codex-cli"), default="codex-cli")
    parser.add_argument("--model", default=os.getenv("AFFECTIVEART_MODEL") or os.getenv("OPENAI_MODEL"))
    parser.add_argument("--base-url", default=os.getenv("AFFECTIVEART_BASE_URL") or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=os.getenv("AFFECTIVEART_API_KEY") or os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--codex-command", default=os.getenv("CODEX_COMMAND") or "codex")
    parser.add_argument("--codex-model", default=os.getenv("AFFECTIVEART_CODEX_MODEL") or os.getenv("CODEX_MODEL") or DEFAULT_CODEX_MODEL)
    parser.add_argument(
        "--search",
        action="store_true",
        help="Pass Codex CLI --search so web_search is available during codex runs.",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help=(
            "Number of samples to process concurrently. Defaults to 8. "
            "Lower this value if the provider rate-limits concurrent requests."
        ),
    )
    parser.add_argument("--image-max-side", type=int, default=1024)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=450)
    parser.add_argument("--no-response-format", action="store_true")
    parser.add_argument(
        "--dangerously-bypass-approvals-and-sandbox",
        action="store_true",
        help=(
            "Pass Codex CLI --dangerously-bypass-approvals-and-sandbox instead of "
            "--sandbox read-only."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.start < 0:
        raise SystemExit("--start must be >= 0")
    if not args.all and args.limit <= 0:
        raise SystemExit("--limit must be positive unless --all is used")
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite cannot be used together")
    if args.provider == "codex-cli" and not args.codex_output_schema.exists():
        raise SystemExit(f"Schema does not exist: {args.codex_output_schema}")
    if args.provider == "codex-cli" and args.reference_count != DEFAULT_REFERENCE_COUNT:
        raise SystemExit(f"--reference-count must be {DEFAULT_REFERENCE_COUNT} to match the prompt attachment order")
    if args.auxiliary_predictions is not None and not args.auxiliary_predictions.is_file():
        raise SystemExit(
            f"Auxiliary predictions do not exist: {args.auxiliary_predictions}"
        )
    if args.workers <= 0:
        raise SystemExit("--workers must be a positive integer")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    if args.max_retries < 0:
        raise SystemExit("--max-retries must be non-negative")
    if args.retry_backoff < 0:
        raise SystemExit("--retry-backoff must be non-negative")
    if args.sleep < 0:
        raise SystemExit("--sleep must be non-negative")
    if args.image_max_side <= 0:
        raise SystemExit("--image-max-side must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("--jpeg-quality must be between 1 and 100")
    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be positive")
    if args.provider == "openai-compatible" and not args.dry_run:
        if not args.model:
            raise SystemExit("Set --model or AFFECTIVEART_MODEL/OPENAI_MODEL.")
        if not args.api_key:
            raise SystemExit("Set --api-key or AFFECTIVEART_API_KEY/OPENAI_API_KEY.")


def list_samples(input_dir: Path) -> list[Path]:
    images_dir = input_dir / "images"
    if not images_dir.exists():
        raise SystemExit(f"Missing images directory: {images_dir}")
    samples = sorted(
        (
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.casefold() in ALLOWED_TARGET_SUFFIXES
        ),
        key=lambda path: (path.stem, path.name.casefold()),
    )
    if not samples:
        raise SystemExit(f"No supported images found in {images_dir}")
    counts = Counter(path.stem for path in samples)
    duplicate_stems = sorted(stem for stem, count in counts.items() if count > 1)
    if duplicate_stems:
        raise SystemExit(
            "Multiple target images map to the same sample_id stem: "
            + ", ".join(duplicate_stems[:10])
        )
    return samples


def selected_samples(samples: list[Path], args: argparse.Namespace) -> list[Path]:
    remaining = samples[args.start :]
    return remaining if args.all else remaining[: args.limit]


def prepare_outputs(args: argparse.Namespace) -> None:
    output_paths = [
        path
        for path in (
            args.output,
            args.raw_output,
            args.run_log,
            args.stats_output,
        )
        if path is not None
    ]
    if args.dry_run:
        if args.run_log is not None:
            args.run_log.parent.mkdir(parents=True, exist_ok=True)
        return
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in output_paths:
            if path.exists():
                path.unlink()
    elif not args.resume:
        existing = [str(path) for path in output_paths if path.exists()]
        if existing:
            raise SystemExit("Output exists. Use --resume or --overwrite: " + ", ".join(existing))


def load_existing_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise SystemExit(f"Existing results are not a JSON list: {path}")
    return value


def write_jsonl_row(path: Path | None, row: dict[str, Any]) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda row: str(row["sample_id"]))
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def image_to_jpeg_bytes(image_path: Path, max_side: int, jpeg_quality: int) -> tuple[bytes, dict[str, Any]]:
    image = Image.open(image_path)
    original_size = image.size
    image.thumbnail((max_side, max_side))
    if image.mode != "RGB":
        image = image.convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    return buffer.getvalue(), {
        "original_width": original_size[0],
        "original_height": original_size[1],
        "encoded_width": image.width,
        "encoded_height": image.height,
        "encoded_bytes": len(buffer.getvalue()),
    }


def image_file_metadata(image_path: Path) -> dict[str, Any]:
    with Image.open(image_path) as image:
        return {
            "path": str(image_path.resolve()),
            "original_width": image.width,
            "original_height": image.height,
            "sent_as": "original_file",
        }


def score_label(score: Any) -> str:
    if score is None:
        return "score n/a"
    try:
        return f"score {float(score):.6f}"
    except (TypeError, ValueError):
        return f"score {score}"


def reference_label_parts(item: dict[str, Any]) -> list[str]:
    parts: list[str] = []
    labels = item.get("labels") if isinstance(item.get("labels"), dict) else {}
    for key in (
        "dominant_emotion",
        "emotion",
        "label",
        "valence",
        "emotional_valence",
        "arousal",
        "emotional_arousal_level",
    ):
        value = item.get(key) or labels.get(key)
        if value is not None:
            parts.append(f"{key} {value}")
    return parts


def selected_reference_items(annotation_path: Path, count: int) -> list[dict[str, Any]]:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    top_k = data.get("top_k")
    if not isinstance(top_k, list):
        raise RunError(f"annotation.json does not contain a top_k list: {annotation_path}")
    selected = [
        item
        for item in sorted(top_k, key=lambda value: value.get("rank", 10**9) if isinstance(value, dict) else 10**9)
        if isinstance(item, dict) and item.get("rank") is not None
    ]
    if len(selected) < count:
        raise RunError(f"annotation.json has only {len(selected)} ranked references, need {count}: {annotation_path}")
    selected = selected[:count]
    ranks = [int(item["rank"]) for item in selected]
    if ranks != list(range(1, count + 1)):
        raise RunError(
            f"annotation.json must contain unique consecutive ranks 1-{count}; "
            f"found {ranks}: {annotation_path}"
        )
    return selected


def find_rank_image(folder: Path, rank: int) -> Path | None:
    prefix = f"rank{rank:02d}_"
    matches = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.name.startswith(prefix) and path.suffix.lower() in ALLOWED_REFERENCE_SUFFIXES
    ]
    if len(matches) > 1:
        names = ", ".join(sorted(path.name for path in matches))
        raise RunError(f"Multiple files found for rank{rank:02d} in {folder}: {names}")
    return matches[0] if matches else None


def reference_image_path(folder: Path, item: dict[str, Any], rank: int) -> Path | None:
    file_name = item.get("file") or item.get("copied_file")
    if file_name:
        path = folder / str(file_name)
        return path if path.is_file() else None
    return find_rank_image(folder, rank)


def reference_images_for_sample(args: argparse.Namespace, sample_id: str, sample_context_dir: Path) -> list[dict[str, Any]]:
    annotation_path = sample_context_dir / "annotation.json"
    if not annotation_path.exists():
        raise RunError(f"Missing annotation.json for {sample_id}: {annotation_path}")
    items = selected_reference_items(annotation_path, args.reference_count)
    references: list[dict[str, Any]] = []
    for item in items:
        rank = int(item["rank"])
        image_path = reference_image_path(sample_context_dir, item, rank)
        if image_path is None:
            raise RunError(f"Missing rank{rank:02d} reference image for {sample_id} in {sample_context_dir}")
        references.append({"rank": rank, "path": image_path.resolve(), "annotation": item})
    return references


def load_reference_fonts() -> tuple[ImageFont.ImageFont, ImageFont.ImageFont]:
    try:
        return (
            ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 30),
            ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 23),
        )
    except OSError:
        return ImageFont.load_default(), ImageFont.load_default()


def fit_reference_image(path: Path, width: int, height: int) -> Image.Image:
    resample = getattr(Image, "Resampling", Image).LANCZOS
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((width, height), resample)
        canvas = Image.new("RGB", (width, height), (252, 252, 250))
        x = (width - image.width) // 2
        y = (height - image.height) // 2
        canvas.paste(image, (x, y))
        return canvas


def draw_reference_frame(reference: dict[str, Any], output_path: Path, quality: int = 88) -> None:
    rank = int(reference["rank"])
    image_path = reference["path"]

    margin = 18
    header_h = 64
    image_w = 860
    image_h = 640
    footer_h = 58
    width = margin * 2 + image_w
    height = margin * 2 + header_h + image_h + footer_h

    font_header, font_footer = load_reference_fonts()
    frame = Image.new("RGB", (width, height), (244, 244, 240))
    draw = ImageDraw.Draw(frame)
    draw.text(
        (margin, margin + 14),
        "REFERENCE TOP 4 - NOT TARGET",
        fill=(20, 20, 20),
        font=font_header,
    )

    image_y = margin + header_h
    frame.paste(fit_reference_image(image_path, image_w, image_h), (margin, image_y))

    footer_y = image_y + image_h
    draw.rectangle([margin, footer_y, margin + image_w - 1, footer_y + footer_h - 1], fill=(28, 28, 28))
    footer_text = f"REF rank{rank:02d}"
    draw.text((margin + 14, footer_y + 17), footer_text, fill=(255, 255, 255), font=font_footer)

    frame.save(output_path, format="JPEG", quality=quality, optimize=True, progressive=True)


def jpeg_bytes_to_data_url(image_bytes: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii")


def endpoint_from_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    return normalized if normalized.endswith("/chat/completions") else f"{normalized}/chat/completions"


def post_openai_compatible(args: argparse.Namespace, image_bytes: bytes, user_prompt: str) -> str:
    body: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": jpeg_bytes_to_data_url(image_bytes)}},
                ],
            },
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if not args.no_response_format:
        body["response_format"] = {"type": "json_object"}

    payload = json.dumps(body).encode("utf-8")
    headers = {"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"}
    endpoint = endpoint_from_base_url(args.base_url)
    for attempt in range(args.max_retries + 1):
        request = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
            return value["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code in RETRY_STATUS_CODES and attempt < args.max_retries:
                time.sleep(args.retry_backoff * (2**attempt))
                continue
            raise RunError(f"HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            if attempt < args.max_retries:
                time.sleep(args.retry_backoff * (2**attempt))
                continue
            raise RunError(f"Model request failed: {exc}") from exc
    raise RunError("Model request failed after retries")


def resolve_codex_command(codex_command: str) -> str:
    path = Path(codex_command)
    if path.exists():
        return str(path)
    resolved = shutil.which(codex_command)
    if resolved:
        return resolved
    if os.name == "nt":
        appdata = os.getenv("APPDATA")
        if appdata:
            npm_dir = Path(appdata) / "npm"
            for suffix in (".cmd", ".exe", ".bat", ".ps1", ""):
                candidate = npm_dir / f"{codex_command}{suffix}"
                if candidate.exists():
                    return str(candidate)
    return codex_command


def build_codex_cli_prompt(user_prompt: str) -> str:
    return f"{SYSTEM_PROMPT}\n\n{user_prompt}"


def compact_reference_labels(annotation: dict[str, Any]) -> dict[str, Any]:
    labels = annotation.get("labels") if isinstance(annotation.get("labels"), dict) else {}
    return {
        "dominant_emotion": labels.get("dominant_emotion") or annotation.get("dominant_emotion"),
        "valence": labels.get("valence") or annotation.get("valence"),
        "arousal": labels.get("arousal") or annotation.get("arousal"),
    }


def attachment_manifest_prompt(
    sample_id: str,
    target_image_path: Path,
    sample_context_dir: Path,
    reference_images: list[dict[str, Any]],
    reference_attachment_mode: str,
) -> str:
    lines = [
        "Attachment manifest for this exact run:",
        f"- sample_id: {sample_id}",
        f"- sample folder: {sample_context_dir}",
        f"- reference attachment mode: {reference_attachment_mode}",
        f"- Image attachment 1 is the TARGET artwork. TARGET path: {target_image_path.resolve()}",
        "- Images attachment 2 and later are REFERENCES only, not target artworks.",
    ]
    for attachment_index, reference in enumerate(reference_images, start=2):
        rank = int(reference["rank"])
        labels = compact_reference_labels(reference["annotation"])
        lines.append(
            f"- Image attachment {attachment_index} is REFERENCE rank{rank:02d}. "
            f"Reference path: {Path(reference['path']).resolve()}. "
            f"Reference labels from annotation.json: {json.dumps(labels, ensure_ascii=False, sort_keys=True)}"
        )
    lines.extend(
        [
            "Use the reference paths and labels only for calibration.",
            "Do not describe any reference image in overall_caption, brushstroke, composition, color, line, or light.",
            "The final JSON must describe and classify only Image attachment 1, the TARGET artwork.",
        ]
    )
    return "\n".join(lines)


def run_codex_cli(
    args: argparse.Namespace,
    target_image_path: Path,
    user_prompt: str,
    sample_id: str,
    sample_context_dir: Path,
    reference_images: list[dict[str, Any]],
    debug_dir: Path | None = None,
) -> str:
    manifest_prompt = attachment_manifest_prompt(
        sample_id=sample_id,
        target_image_path=target_image_path,
        sample_context_dir=sample_context_dir,
        reference_images=reference_images,
        reference_attachment_mode=args.reference_attachment_mode,
    )
    prompt = build_codex_cli_prompt(f"{manifest_prompt}\n\n{user_prompt}")
    with TemporaryDirectory(prefix="affectlens_inference_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        output_path = tmp_path / "codex_output.txt"
        reference_attachment_paths: list[Path] = []
        if args.reference_attachment_mode == "framed":
            for index, reference in enumerate(reference_images, start=2):
                frame_path = tmp_path / f"attachment_{index:02d}_reference_rank{int(reference['rank']):02d}.jpg"
                draw_reference_frame(reference, frame_path)
                reference_attachment_paths.append(frame_path)
        else:
            reference_attachment_paths = [Path(reference["path"]) for reference in reference_images]

        command = [
            resolve_codex_command(args.codex_command),
        ]
        if args.search:
            command.append("--search")
        command.extend(
            [
                "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "-C",
            str(sample_context_dir),
            "--image",
            str(target_image_path.resolve()),
            ]
        )
        for reference_attachment_path in reference_attachment_paths:
            command.extend(["--image", str(reference_attachment_path)])
        command.extend(
            [
                "--output-schema",
                str(args.codex_output_schema.resolve()),
                "--output-last-message",
                str(output_path),
            ]
        )
        if args.dangerously_bypass_approvals_and_sandbox:
            command.append("--dangerously-bypass-approvals-and-sandbox")
        else:
            command.extend(["--sandbox", "read-only"])
        if args.codex_model:
            command.extend(["--model", args.codex_model])
        command.append("-")

        try:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                start_new_session=os.name != "nt",
            )
            register_active_process(process)
            try:
                stdout, stderr = process.communicate(input=prompt, timeout=args.timeout)
            except subprocess.TimeoutExpired as exc:
                terminate_process_tree(process)
                try:
                    stdout, stderr = process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, stderr = process.communicate()
                raise RunError(f"Codex CLI timed out after {args.timeout} seconds.") from exc
            finally:
                unregister_active_process(process)
            completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        except FileNotFoundError as exc:
            raise RunError(f"Codex CLI command was not found: {args.codex_command}") from exc

        if completed.returncode != 0:
            write_codex_debug_log(
                debug_dir=debug_dir,
                command=command,
                prompt=prompt,
                stdout=completed.stdout,
                stderr=completed.stderr,
                output_path=output_path,
                returncode=completed.returncode,
            )
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RunError(f"Codex CLI failed: {detail}")
        output = output_path.read_text(encoding="utf-8").strip() if output_path.exists() else completed.stdout.strip()
        write_codex_debug_log(
            debug_dir=debug_dir,
            command=command,
            prompt=prompt,
            stdout=completed.stdout,
            stderr=completed.stderr,
            output_path=output_path,
            returncode=completed.returncode,
        )
        if not output:
            raise RunError("Codex CLI returned empty output")
        return output


def write_codex_debug_log(
    debug_dir: Path | None,
    command: list[str],
    prompt: str,
    stdout: str,
    stderr: str,
    output_path: Path,
    returncode: int,
) -> None:
    if debug_dir is None:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    last_message = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
    (debug_dir / "command.json").write_text(
        json.dumps(
            {
                "command": command,
                "cwd": str(REPO_ROOT),
                "returncode": returncode,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (debug_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (debug_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
    (debug_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
    (debug_dir / "last_message.txt").write_text(last_message, encoding="utf-8")


def terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except OSError:
            process.kill()


def codex_debug_dir_for_sample(
    base_dir: Path | None,
    offset: int,
    sample_id: str,
) -> Path | None:
    if base_dir is None:
        return None
    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:12]
    return base_dir / f"{offset:06d}_{digest}"


def sample_context_dir_for_sample(args: argparse.Namespace, sample_id: str) -> Path:
    sample_context_dir = args.reference_dir / sample_id
    if not sample_context_dir.exists():
        raise RunError(
            f"Missing prepared reference folder for {sample_id}: {sample_context_dir}"
        )
    if not sample_context_dir.is_dir():
        raise RunError(
            f"Reference path is not a directory for {sample_id}: {sample_context_dir}"
        )
    return sample_context_dir.resolve()


def auxiliary_row_id(row: dict[str, Any]) -> str | None:
    explicit = row.get("sample_id")
    if explicit is not None and str(explicit).strip():
        sample_id = str(explicit).strip()
        if (
            sample_id in {".", ".."}
            or "/" in sample_id
            or "\\" in sample_id
        ):
            raise SystemExit(f"Unsafe auxiliary sample_id: {sample_id!r}")
        return sample_id
    for key in ("image_id", "image_path", "local_image_path"):
        value = row.get(key)
        if value:
            return Path(str(value).replace("\\", "/")).stem
    return None


def load_auxiliary_predictions(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    output: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise SystemExit(f"{path}:{line_number} is not a JSON object.")
            sample_id = auxiliary_row_id(row)
            if not sample_id:
                raise SystemExit(
                    f"{path}:{line_number} has no sample_id, image_id, or image path."
                )
            if sample_id in output:
                raise SystemExit(
                    f"{path}:{line_number} duplicates sample_id {sample_id!r}."
                )
            emotion = (
                row.get("predicted_emotion")
                or row.get("emotion")
                or row.get("dominant_emotion")
            )
            if not emotion:
                raise SystemExit(
                    f"{path}:{line_number} has no predicted emotion."
                )
            try:
                normalize_task_label("dominant_emotion", emotion)
            except ValueError as exc:
                raise SystemExit(
                    f"{path}:{line_number} has invalid predicted emotion {emotion!r}."
                ) from exc
            output[sample_id] = row
    return output


def validate_auxiliary_coverage(
    prediction_index: dict[str, dict[str, Any]],
    selected: list[Path],
    source_path: Path | None,
) -> None:
    if source_path is None:
        return
    expected = {path.stem for path in selected}
    missing = sorted(expected - set(prediction_index))
    if missing:
        raise SystemExit(
            f"{source_path} is missing auxiliary predictions for "
            f"{len(missing)} selected sample(s): {missing[:10]}"
        )


def auxiliary_prompt_block(row: dict[str, Any] | None) -> str:
    if not row:
        return ""
    emotion = (
        row.get("predicted_emotion")
        or row.get("emotion")
        or row.get("dominant_emotion")
    )
    if not emotion:
        return ""
    return (
        "\n\nAuxiliary classifier calibration:\n"
        f"- The separately trained visual classifier predicts {emotion}.\n"
        "- Treat this only as a non-binding prior. Target-artwork evidence, "
        "style-skill calibration, labeled references, and the required "
        "valence-arousal binding remain authoritative.\n"
    )


def process_sample(offset: int, image_path: Path, args: argparse.Namespace, user_prompt: str) -> dict[str, Any]:
    sample_id = image_path.stem
    started = time.time()
    codex_debug_dir: Path | None = None
    sample_context_dir: Path | None = None
    reference_images: list[dict[str, Any]] = []
    try:
        auxiliary_row = getattr(args, "auxiliary_prediction_index", {}).get(
            sample_id
        )
        sample_prompt = user_prompt + auxiliary_prompt_block(auxiliary_row)
        if args.provider == "codex-cli":
            image_bytes = None
            image_meta = image_file_metadata(image_path)
            sample_context_dir = sample_context_dir_for_sample(args, sample_id)
            reference_images = reference_images_for_sample(args, sample_id, sample_context_dir)
        else:
            image_bytes, image_meta = image_to_jpeg_bytes(
                image_path,
                args.image_max_side,
                args.jpeg_quality,
            )
        request_meta = {
            "provider": args.provider,
            "model": args.codex_model if args.provider == "codex-cli" else args.model,
            "codex_command": args.codex_command if args.provider == "codex-cli" else None,
            "codex_output_schema": (
                str(args.codex_output_schema.resolve()) if args.provider == "codex-cli" else None
            ),
            "codex_working_dir": (
                str(sample_context_dir) if sample_context_dir is not None else None
            ),
            "reference_dir": (
                str(args.reference_dir) if args.provider == "codex-cli" else None
            ),
            "reference_images": (
                [
                    {
                        "attachment": index,
                        "rank": reference["rank"],
                        "path": str(reference["path"]),
                        "attachment_mode": args.reference_attachment_mode,
                        "annotation": reference["annotation"],
                    }
                    for index, reference in enumerate(reference_images, start=2)
                ]
                if args.provider == "codex-cli"
                else None
            ),
            "target_image": (
                str(image_path.resolve()) if args.provider == "codex-cli" else None
            ),
            "codex_image_order": (
                [f"image attachment 1: target artwork original file {image_path.name}"]
                + [
                    (
                        f"image attachment {index}: "
                        f"{args.reference_attachment_mode} reference rank{int(reference['rank']):02d}, NOT TARGET"
                    )
                    for index, reference in enumerate(reference_images, start=2)
                ]
                if args.provider == "codex-cli"
                else None
            ),
            "dangerously_bypass_approvals_and_sandbox": (
                args.dangerously_bypass_approvals_and_sandbox
                if args.provider == "codex-cli"
                else None
            ),
            "prompt_source": (
                "src.prompts.SYSTEM_PROMPT + src.prompts.build_analysis_prompt"
                if args.provider == "codex-cli"
                else "src.prompts.build_analysis_prompt"
            ),
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "auxiliary_prediction": (
                (
                    auxiliary_row.get("predicted_emotion")
                    or auxiliary_row.get("emotion")
                    or auxiliary_row.get("dominant_emotion")
                )
                if auxiliary_row
                else None
            ),
        }
        codex_debug_dir = codex_debug_dir_for_sample(
            args.codex_log_dir if args.provider == "codex-cli" else None,
            offset,
            sample_id,
        )
        if codex_debug_dir is not None:
            request_meta["codex_log_dir"] = str(args.codex_log_dir)
        if args.dry_run:
            return {
                "kind": "dry_run",
                "offset": offset,
                "sample_id": sample_id,
                "run_log": {
                    "sample_id": sample_id,
                    "status": "dry_run",
                    "image": image_meta,
                    "request": request_meta,
                },
            }

        raw_text = (
            run_codex_cli(
                args,
                image_path.resolve(),
                sample_prompt,
                sample_id,
                sample_context_dir,
                reference_images,
                debug_dir=codex_debug_dir,
            )
            if args.provider == "codex-cli"
            else post_openai_compatible(args, image_bytes, sample_prompt)
        )
        prediction = normalize_prediction(sample_id, extract_json_object(raw_text))
        latency_seconds = round(time.time() - started, 3)
        raw_record = {
            "sample_id": sample_id,
            "status": "ok",
            "provider": args.provider,
            "model": args.codex_model if args.provider == "codex-cli" else args.model,
            "prompt_source": request_meta["prompt_source"],
            "raw_response": raw_text,
            "parsed": prediction,
            "image": image_meta,
            "latency_seconds": latency_seconds,
        }
        run_log = {
            "sample_id": sample_id,
            "status": "ok",
            "request": request_meta,
            "latency_seconds": latency_seconds,
        }
        if codex_debug_dir is not None:
            raw_record["codex_debug_dir"] = str(codex_debug_dir)
            run_log["codex_debug_dir"] = str(codex_debug_dir)
        return {
            "kind": "ok",
            "offset": offset,
            "sample_id": sample_id,
            "prediction": prediction,
            "raw_record": raw_record,
            "run_log": run_log,
        }
    except Exception as exc:
        error_record = {
            "sample_id": sample_id,
            "status": "error",
            "error_type": exc.__class__.__name__,
            "error": str(exc),
            "latency_seconds": round(time.time() - started, 3),
        }
        if codex_debug_dir is not None:
            error_record["codex_debug_dir"] = str(codex_debug_dir)
        return {
            "kind": "error",
            "offset": offset,
            "sample_id": sample_id,
            "error": exc,
            "error_record": error_record,
        }


def write_result(args: argparse.Namespace, rows: list[dict[str, Any]], done_ids: set[str], result: dict[str, Any]) -> None:
    kind = result["kind"]
    if kind == "dry_run":
        write_jsonl_row(args.run_log, result["run_log"])
        return
    if kind == "ok":
        rows.append(result["prediction"])
        done_ids.add(result["sample_id"])
        write_results(args.output, rows)
        write_jsonl_row(args.raw_output, result["raw_record"])
        write_jsonl_row(args.run_log, result["run_log"])
        return
    if kind == "error":
        if not args.dry_run:
            write_jsonl_row(args.raw_output, result["error_record"])
        write_jsonl_row(args.run_log, result["error_record"])


def submit_sample(
    executor: ThreadPoolExecutor,
    offset: int,
    image_path: Path,
    args: argparse.Namespace,
    user_prompt: str,
) -> Future[dict[str, Any]]:
    print(f"[submit] index={offset} sample_id={image_path.stem}")
    return executor.submit(process_sample, offset, image_path, args, user_prompt)


def extract_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model output is not an object")
    return value


def normalize_prediction(sample_id: str, payload: dict[str, Any]) -> dict[str, str]:
    aliases = {
        "dominant_emotion": "emotion",
        "valence": "emotional_valence",
        "arousal": "emotional_arousal_level",
        "arousal_level": "emotional_arousal_level",
    }
    normalized: dict[str, Any] = {}
    for key, value in payload.items():
        canonical = aliases.get(str(key), str(key))
        normalized[canonical] = value

    row: dict[str, str] = {"sample_id": sample_id}
    try:
        model_emotion = normalize_task_label(
            "dominant_emotion", normalized.get("emotion", "")
        )
        row["emotion"] = to_output_emotion(model_emotion)
    except ValueError as exc:
        raise ValueError(
            f"Invalid emotion for {sample_id}: {normalized.get('emotion')!r}"
        ) from exc

    try:
        row["emotional_valence"] = normalize_task_label(
            "valence", normalized.get("emotional_valence", "")
        )
    except ValueError as exc:
        raise ValueError(
            f"Invalid emotional_valence for {sample_id}: "
            f"{normalized.get('emotional_valence')!r}"
        ) from exc

    try:
        row["emotional_arousal_level"] = normalize_task_label(
            "arousal", normalized.get("emotional_arousal_level", "")
        )
    except ValueError as exc:
        raise ValueError(
            f"Invalid emotional_arousal_level for {sample_id}: "
            f"{normalized.get('emotional_arousal_level')!r}"
        ) from exc

    binding_error = emotion_binding_mismatch(
        model_emotion,
        row["emotional_valence"],
        row["emotional_arousal_level"],
    )
    if binding_error:
        raise ValueError(f"Invalid emotion binding for {sample_id}: {binding_error}")

    row["overall_caption"] = required_text(
        sample_id,
        "overall_caption",
        payload,
        "overall_caption",
    )
    row["brushstroke"] = required_text(
        sample_id,
        "brushstroke",
        payload,
        "brushstroke",
    )
    row["composition"] = required_text(
        sample_id,
        "composition",
        payload,
        "composition",
    )
    row["color"] = required_text(
        sample_id,
        "color",
        payload,
        "color",
    )
    row["line"] = required_text(
        sample_id,
        "line",
        payload,
        "line",
    )
    row["light"] = required_text(
        sample_id,
        "light",
        payload,
        "light",
    )
    return row


def required_text(
    sample_id: str,
    output_field: str,
    source: dict[str, Any],
    *source_fields: str,
) -> str:
    for source_field in source_fields:
        value = source.get(source_field)
        if isinstance(value, str) and value.strip():
            return compact_text(value)
    expected = ", ".join(source_fields)
    raise ValueError(
        f"Missing {output_field} for {sample_id}; expected non-empty field: {expected}"
    )


def compact_text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected a non-empty text value.")
    return " ".join(value.strip().split())


def validate_results(rows: list[dict[str, Any]], expected_ids: set[str] | None = None) -> None:
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        if set(row) != set(RESULT_FIELDS):
            raise ValueError(f"Row {index} has wrong fields: {sorted(row)}")
        sample_id = row["sample_id"]
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"Row {index} has invalid sample_id")
        if sample_id in seen:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        if row["emotion"] not in EMOTIONS:
            raise ValueError(f"Invalid emotion for {sample_id}: {row['emotion']!r}")
        if row["emotional_valence"] not in VALENCES:
            raise ValueError(f"Invalid valence for {sample_id}: {row['emotional_valence']!r}")
        if row["emotional_arousal_level"] not in AROUSALS:
            raise ValueError(f"Invalid arousal for {sample_id}: {row['emotional_arousal_level']!r}")
        model_emotion = normalize_task_label("dominant_emotion", row["emotion"])
        binding_error = emotion_binding_mismatch(
            model_emotion,
            row["emotional_valence"],
            row["emotional_arousal_level"],
        )
        if binding_error:
            raise ValueError(f"Invalid emotion binding for {sample_id}: {binding_error}")
        for field in TEXT_FIELDS:
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"Missing {field} for {sample_id}")
    if expected_ids is not None:
        missing = sorted(expected_ids - seen)
        extra = sorted(seen - expected_ids)
        if missing or extra:
            raise ValueError(f"ID mismatch. missing={missing[:5]} extra={extra[:5]}")


def count_values(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field, "")) for row in rows).items()))


def build_result_stats(
    rows: list[dict[str, Any]],
    samples: list[Path],
    selected: list[Path],
    processed: int,
    skipped: int,
    failed: int,
) -> dict[str, Any]:
    all_ids = {path.stem for path in samples}
    selected_ids = {path.stem for path in selected}
    result_ids = {str(row.get("sample_id")) for row in rows}
    missing_selected = sorted(selected_ids - result_ids)
    extra_result_ids = sorted(result_ids - all_ids)
    emotion_by_quadrant: dict[str, dict[str, int]] = {}
    for row in rows:
        quadrant = f"{row.get('emotional_valence', '')}/{row.get('emotional_arousal_level', '')}"
        emotion = str(row.get("emotion", ""))
        emotion_by_quadrant.setdefault(quadrant, {})
        emotion_by_quadrant[quadrant][emotion] = emotion_by_quadrant[quadrant].get(emotion, 0) + 1

    return {
        "total_test_images": len(samples),
        "selected_images": len(selected),
        "result_rows": len(rows),
        "processed_this_run": processed,
        "skipped_this_run": skipped,
        "failed_this_run": failed,
        "complete_selected": not missing_selected and failed == 0,
        "missing_selected_count": len(missing_selected),
        "missing_selected_sample_ids": missing_selected[:50],
        "extra_result_count": len(extra_result_ids),
        "extra_result_sample_ids": extra_result_ids[:50],
        "emotion_counts": count_values(rows, "emotion"),
        "valence_counts": count_values(rows, "emotional_valence"),
        "arousal_counts": count_values(rows, "emotional_arousal_level"),
        "emotion_by_valence_arousal": {
            quadrant: dict(sorted(counts.items()))
            for quadrant, counts in sorted(emotion_by_quadrant.items())
        },
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    install_interrupt_handler()
    args.auxiliary_prediction_index = load_auxiliary_predictions(
        args.auxiliary_predictions
    )
    user_prompt = build_analysis_prompt(
        include_references=args.provider == "codex-cli",
        include_local_skills=args.provider == "codex-cli",
    )
    samples = list_samples(args.input_dir)
    selected = selected_samples(samples, args)
    if not selected:
        raise SystemExit("The requested start/limit selection contains no images.")
    validate_auxiliary_coverage(
        args.auxiliary_prediction_index,
        selected,
        args.auxiliary_predictions,
    )
    expected_ids = {path.stem for path in samples}
    prepare_outputs(args)

    rows = load_existing_results(args.output) if args.resume else []
    done_ids = {str(row.get("sample_id")) for row in rows}
    print(f"Loaded {len(samples)} input images from {args.input_dir / 'images'}")
    print(f"Selected {len(selected)} image(s), start={args.start}, all={args.all}")
    print(f"Provider: {args.provider}")
    if args.auxiliary_predictions is not None:
        print(
            "Auxiliary predictions: "
            f"{args.auxiliary_predictions} "
            f"({len(args.auxiliary_prediction_index)} rows)"
        )
    if args.provider == "codex-cli":
        print(f"Codex model: {args.codex_model}")
        print(f"Reference directory: {args.reference_dir}")
        print(f"Reference images: attachments 2-{args.reference_count + 1} from annotation.json rank order")
        print("Prompt source: src.prompts.SYSTEM_PROMPT + src.prompts.build_analysis_prompt")
    else:
        print(f"Model: {args.model}")
        print("Prompt source: src.prompts.build_analysis_prompt")
    print(f"Workers: {args.workers}")

    processed = 0
    skipped = 0
    failed = 0
    pending_samples: list[tuple[int, Path]] = []
    for offset, image_path in enumerate(selected, start=args.start):
        sample_id = image_path.stem
        if sample_id in done_ids:
            skipped += 1
            print(f"[skip] {sample_id}")
            continue
        pending_samples.append((offset, image_path))

    if args.workers == 1:
        for offset, image_path in pending_samples:
            print(f"[run] index={offset} sample_id={image_path.stem}")
            result = process_sample(offset, image_path, args, user_prompt)
            write_result(args, rows, done_ids, result)
            if result["kind"] in {"ok", "dry_run"}:
                processed += 1
                print(f"[done] index={offset} sample_id={image_path.stem}")
            else:
                failed += 1
                print(f"[error] {image_path.stem}: {result['error']}", file=sys.stderr)
            if args.sleep > 0:
                time.sleep(args.sleep)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures: dict[Future[dict[str, Any]], tuple[int, str]] = {}
            next_sample_index = 0
            while next_sample_index < len(pending_samples) and len(futures) < args.workers:
                offset, image_path = pending_samples[next_sample_index]
                future = submit_sample(executor, offset, image_path, args, user_prompt)
                futures[future] = (offset, image_path.stem)
                next_sample_index += 1

            while futures:
                done, _ = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
                if not done:
                    continue
                for future in done:
                    offset, sample_id = futures.pop(future)
                    result = future.result()
                    write_result(args, rows, done_ids, result)
                    if result["kind"] in {"ok", "dry_run"}:
                        processed += 1
                        print(f"[done] index={offset} sample_id={sample_id}")
                    else:
                        failed += 1
                        print(f"[error] {sample_id}: {result['error']}", file=sys.stderr)

                while next_sample_index < len(pending_samples) and len(futures) < args.workers:
                    offset, image_path = pending_samples[next_sample_index]
                    if args.sleep > 0 and futures:
                        time.sleep(args.sleep)
                    future = submit_sample(executor, offset, image_path, args, user_prompt)
                    futures[future] = (offset, image_path.stem)
                    next_sample_index += 1

    if not args.dry_run and rows:
        full_expected = expected_ids if failed == 0 and args.all and args.start == 0 else None
        validate_results(rows, expected_ids=full_expected)
        if args.stats_output is not None:
            stats = build_result_stats(
                rows,
                samples,
                selected,
                processed,
                skipped,
                failed,
            )
            write_json(args.stats_output, stats)
            print(f"Summary JSON: {args.stats_output}")
        print(f"Results JSON: {args.output}")
    print(
        f"Finished. processed={processed} skipped={skipped} failed={failed} "
        f"total_result_rows={len(rows)}"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
