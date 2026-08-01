from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


DEFAULT_DATA_ROOT = Path("data/raw/emoart_130k")


@dataclass(frozen=True)
class ImageLocation:
    archive_path: Path
    member_name: str


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def read_json_array(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError(f"Expected JSON array at {path}")

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(value, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Expected object at {path}[{index}]")
        rows.append(row)
    return rows


def normalize_split_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    if not isinstance(normalized.get("image_id"), str):
        train_image_id = normalized.get("train_image_id")
        if isinstance(train_image_id, str):
            normalized["image_id"] = train_image_id

    if not isinstance(normalized.get("image_path"), str):
        for key in ("train_image_path", "member_name"):
            value = normalized.get(key)
            if isinstance(value, str):
                normalized["image_path"] = value
                break

    return normalized


def normalize_archive_image_path(image_path: str) -> str:
    parts = [part for part in image_path.replace("\\", "/").strip().split("/") if part]
    if parts and parts[0].casefold() == "images":
        parts = parts[1:]
    return "/".join(parts)


def read_split_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.casefold() == ".json":
        rows = read_json_array(path)
    else:
        rows = read_jsonl(path)
    return [normalize_split_row(row) for row in rows]


def style_from_image_path(image_path: str) -> str:
    parts = [part for part in normalize_archive_image_path(image_path).split("/") if part]
    if len(parts) >= 2:
        return parts[0]
    return ""


def image_location(
    image_path: str,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> ImageLocation:
    normalized = normalize_archive_image_path(image_path)
    style = style_from_image_path(normalized)
    if not style:
        raise ValueError(f"Cannot infer style from image_path: {image_path!r}")
    return ImageLocation(
        archive_path=data_root / f"{style}.tar.gz",
        member_name=normalized,
    )


def image_exists(image_path: str, data_root: Path = DEFAULT_DATA_ROOT) -> bool:
    location = image_location(image_path, data_root=data_root)
    if not location.archive_path.exists():
        return False
    with tarfile.open(location.archive_path, "r:gz") as archive:
        try:
            archive.getmember(location.member_name)
        except KeyError:
            return False
    return True


def open_image(image_path: str, data_root: Path = DEFAULT_DATA_ROOT) -> Any:
    from PIL import Image

    location = image_location(image_path, data_root=data_root)
    with tarfile.open(location.archive_path, "r:gz") as archive:
        member = archive.getmember(location.member_name)
        handle = archive.extractfile(member)
        if handle is None:
            raise FileNotFoundError(location.member_name)
        with handle:
            image = Image.open(handle)
            return image.copy()


class AffectiveArtDataset:
    def __init__(
        self,
        split_path: str | Path,
        data_root: str | Path = DEFAULT_DATA_ROOT,
        load_images: bool = False,
    ) -> None:
        self.split_path = Path(split_path)
        self.data_root = Path(data_root)
        self.load_images = load_images
        self.rows = read_split_rows(self.split_path)

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = dict(self.rows[index])
        location = image_location(row["image_path"], data_root=self.data_root)
        row["image_location"] = {
            "archive_path": str(location.archive_path),
            "member_name": location.member_name,
        }
        if self.load_images:
            row["image"] = open_image(row["image_path"], data_root=self.data_root)
        return row

    def get_labels(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "dominant_emotion": row["dominant_emotion"],
            "valence": row["valence"],
            "arousal": row["arousal"],
        }
