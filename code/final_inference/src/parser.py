from __future__ import annotations

import json
import re
from typing import Any

from src.metrics import TASKS, emotion_binding_mismatch, normalize_task_label


class ModelOutputParseError(ValueError):
    pass


TASK_KEY_ALIASES = {
    "dominant_emotion": "dominant_emotion",
    "emotion": "dominant_emotion",
    "dominantemotion": "dominant_emotion",
    "dominant emotion": "dominant_emotion",
    "valence": "valence",
    "emotional_valence": "valence",
    "emotional valence": "valence",
    "arousal": "arousal",
    "arousal_level": "arousal",
    "arousal level": "arousal",
    "emotional_arousal_level": "arousal",
    "emotional arousal level": "arousal",
}

def extract_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if not text:
        raise ModelOutputParseError("Model output is empty.")

    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = json.loads(_first_balanced_json_object(text))

    if not isinstance(payload, dict):
        raise ModelOutputParseError("Model output JSON is not an object.")
    return payload


def _first_balanced_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise ModelOutputParseError("Model output does not contain a JSON object.")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise ModelOutputParseError("Model output contains an incomplete JSON object.")


def parse_model_output(raw_text: str) -> dict[str, str]:
    payload = extract_json_object(raw_text)
    normalized = _normalize_keys(payload)
    prediction: dict[str, str] = {}

    for task in TASKS:
        if task not in normalized:
            raise ModelOutputParseError(f"Missing required field: {task}")
        prediction[task] = _normalize_label(task, normalized[task])

    mismatch = emotion_binding_mismatch(
        prediction["dominant_emotion"],
        prediction["valence"],
        prediction["arousal"],
    )
    if mismatch:
        raise ModelOutputParseError(mismatch)

    return prediction


def _normalize_keys(payload: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in payload.items():
        key_text = str(key).strip()
        aliases = (
            key_text,
            key_text.casefold(),
            key_text.replace("_", " ").casefold(),
        )
        canonical = next(
            (TASK_KEY_ALIASES[alias] for alias in aliases if alias in TASK_KEY_ALIASES),
            None,
        )
        if canonical:
            normalized[canonical] = value
    return normalized


def _normalize_label(task: str, value: Any) -> str:
    try:
        return normalize_task_label(task, value)
    except ValueError as exc:
        raise ModelOutputParseError(str(exc)) from exc
