from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any


TASKS = ("dominant_emotion", "valence", "arousal")

TASK_LABELS: dict[str, tuple[str, ...]] = {
    "dominant_emotion": (
        "Alarmed",
        "Annoyed",
        "Aroused",
        "Bored",
        "Calm",
        "Contentment",
        "Excited",
        "Frustrated",
        "Glad",
        "Happy",
        "Sad",
        "Tired",
    ),
    "valence": ("Positive", "Negative"),
    "arousal": ("Low", "High"),
}

MODEL_TO_OUTPUT_EMOTION: dict[str, str] = {
    label: "content" if label == "Contentment" else label.casefold()
    for label in TASK_LABELS["dominant_emotion"]
}

OUTPUT_TO_MODEL_EMOTION: dict[str, str] = {
    label: emotion for emotion, label in MODEL_TO_OUTPUT_EMOTION.items()
}

OUTPUT_EMOTION_LABELS: tuple[str, ...] = tuple(
    MODEL_TO_OUTPUT_EMOTION[label] for label in TASK_LABELS["dominant_emotion"]
)
OUTPUT_VALENCE_LABELS = TASK_LABELS["valence"]
OUTPUT_AROUSAL_LABELS = TASK_LABELS["arousal"]

TASK_LABEL_ALIASES: dict[tuple[str, str], str] = {
    ("dominant_emotion", output_label.casefold()): model_label
    for output_label, model_label in OUTPUT_TO_MODEL_EMOTION.items()
}

EMOTION_BINDINGS: dict[tuple[str, str], tuple[str, ...]] = {
    ("High", "Positive"): ("Excited", "Happy", "Aroused"),
    ("High", "Negative"): ("Alarmed", "Annoyed", "Frustrated"),
    ("Low", "Negative"): ("Bored", "Sad", "Tired"),
    ("Low", "Positive"): ("Calm", "Contentment", "Glad"),
}

EMOTION_TO_BINDING: dict[str, tuple[str, str]] = {
    emotion: (arousal, valence)
    for (arousal, valence), emotions in EMOTION_BINDINGS.items()
    for emotion in emotions
}


@dataclass(frozen=True)
class PredictionIssue:
    kind: str
    image_id: str | None
    message: str
    task: str | None = None
    value: Any = None
    line_number: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("Cannot compute accuracy on an empty sequence")
    correct = sum(expected == actual for expected, actual in zip(y_true, y_pred))
    return correct / len(y_true)


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("Cannot compute macro F1 on an empty sequence")

    labels = sorted(set(y_true) | set(y_pred))
    scores: list[float] = []
    for label in labels:
        true_positive = sum(
            expected == label and actual == label
            for expected, actual in zip(y_true, y_pred)
        )
        false_positive = sum(
            expected != label and actual == label
            for expected, actual in zip(y_true, y_pred)
        )
        false_negative = sum(
            expected == label and actual != label
            for expected, actual in zip(y_true, y_pred)
        )

        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = (
            true_positive / precision_denominator
            if precision_denominator
            else 0.0
        )
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        score = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        scores.append(score)

    return sum(scores) / len(scores)


def validate_tasks(tasks: Sequence[str]) -> None:
    unknown = [task for task in tasks if task not in TASK_LABELS]
    if unknown:
        raise ValueError(f"Unknown task names: {unknown}")


def normalize_task_label(task: str, value: Any) -> str:
    if task not in TASK_LABELS:
        raise ValueError(f"Unknown task name: {task!r}")
    if isinstance(value, dict) and "label" in value:
        value = value["label"]
    if not isinstance(value, str):
        raise ValueError(f"{task} must be a string label.")

    cleaned = value.strip()
    alias = TASK_LABEL_ALIASES.get((task, cleaned.casefold()))
    if alias:
        return alias

    for label in TASK_LABELS[task]:
        if cleaned.casefold() == label.casefold():
            return label

    raise ValueError(
        f"Invalid {task} label {value!r}. Allowed labels: {TASK_LABELS[task]}"
    )


def to_output_emotion(value: Any) -> str:
    model_label = normalize_task_label("dominant_emotion", value)
    return MODEL_TO_OUTPUT_EMOTION[model_label]


def expected_emotions_for_binding(arousal: str, valence: str) -> tuple[str, ...]:
    return EMOTION_BINDINGS[(arousal, valence)]


def emotion_binding_mismatch(
    dominant_emotion: str,
    valence: str,
    arousal: str,
) -> str | None:
    expected = EMOTION_TO_BINDING.get(dominant_emotion)
    if expected is None:
        return None

    expected_arousal, expected_valence = expected
    if arousal == expected_arousal and valence == expected_valence:
        return None

    expected_emotions = EMOTION_BINDINGS.get((arousal, valence))
    if expected_emotions is None:
        return None

    return (
        f"dominant_emotion {dominant_emotion!r} is bound to "
        f"arousal={expected_arousal!r}, valence={expected_valence!r}; "
        f"for arousal={arousal!r}, valence={valence!r}, dominant_emotion "
        f"must be one of {expected_emotions!r}."
    )


def build_prediction_index(
    prediction_rows: Sequence[dict[str, Any]],
    tasks: Sequence[str] = TASKS,
) -> tuple[dict[str, dict[str, Any]], list[PredictionIssue]]:
    validate_tasks(tasks)
    prediction_by_id: dict[str, dict[str, Any]] = {}
    issues: list[PredictionIssue] = []

    for line_number, row in enumerate(prediction_rows, start=1):
        image_id_value = row.get("image_id")
        image_id = image_id_value if isinstance(image_id_value, str) else None
        if not image_id:
            issues.append(
                PredictionIssue(
                    kind="missing_image_id",
                    image_id=None,
                    line_number=line_number,
                    message="Prediction row is missing a string image_id.",
                )
            )
            continue

        if image_id in prediction_by_id:
            issues.append(
                PredictionIssue(
                    kind="duplicate_prediction",
                    image_id=image_id,
                    line_number=line_number,
                    message="Prediction file contains duplicate image_id rows.",
                )
            )
            continue

        prediction_by_id[image_id] = row
        for task in tasks:
            value = row.get(task)
            if not isinstance(value, str) or not value:
                issues.append(
                    PredictionIssue(
                        kind="missing_prediction_label",
                        image_id=image_id,
                        task=task,
                        value=value,
                        line_number=line_number,
                        message=f"Prediction is missing a string label for {task}.",
                    )
                )
                continue
            if value not in TASK_LABELS[task]:
                issues.append(
                    PredictionIssue(
                        kind="invalid_prediction_label",
                        image_id=image_id,
                        task=task,
                        value=value,
                        line_number=line_number,
                        message=f"Prediction label is not valid for {task}.",
                    )
                )
        if all(task in row for task in TASKS):
            mismatch = emotion_binding_mismatch(
                str(row["dominant_emotion"]),
                str(row["valence"]),
                str(row["arousal"]),
            )
            if mismatch:
                issues.append(
                    PredictionIssue(
                        kind="invalid_prediction_binding",
                        image_id=image_id,
                        task="dominant_emotion",
                        value={
                            "dominant_emotion": row["dominant_emotion"],
                            "valence": row["valence"],
                            "arousal": row["arousal"],
                        },
                        line_number=line_number,
                        message=mismatch,
                    )
                )

    return prediction_by_id, issues


def validate_ground_truth_rows(
    ground_truth_rows: Sequence[dict[str, Any]],
    tasks: Sequence[str] = TASKS,
) -> list[PredictionIssue]:
    validate_tasks(tasks)
    issues: list[PredictionIssue] = []
    seen_image_ids: set[str] = set()

    for line_number, row in enumerate(ground_truth_rows, start=1):
        image_id_value = row.get("image_id")
        image_id = image_id_value if isinstance(image_id_value, str) else None
        if not image_id:
            issues.append(
                PredictionIssue(
                    kind="missing_ground_truth_image_id",
                    image_id=None,
                    line_number=line_number,
                    message="Ground-truth row is missing a string image_id.",
                )
            )
            continue

        if image_id in seen_image_ids:
            issues.append(
                PredictionIssue(
                    kind="duplicate_ground_truth",
                    image_id=image_id,
                    line_number=line_number,
                    message="Ground truth contains duplicate image_id rows.",
                )
            )
        seen_image_ids.add(image_id)

        for task in tasks:
            value = row.get(task)
            if value not in TASK_LABELS[task]:
                issues.append(
                    PredictionIssue(
                        kind="invalid_ground_truth_label",
                        image_id=image_id,
                        task=task,
                        value=value,
                        line_number=line_number,
                        message=f"Ground-truth label is not valid for {task}.",
                    )
                )

    return issues


def validate_alignment(
    ground_truth_rows: Sequence[dict[str, Any]],
    prediction_by_id: dict[str, dict[str, Any]],
) -> list[PredictionIssue]:
    ground_truth_ids = {
        row["image_id"]
        for row in ground_truth_rows
        if isinstance(row.get("image_id"), str)
    }
    issues: list[PredictionIssue] = []

    for row in ground_truth_rows:
        image_id = row.get("image_id")
        if isinstance(image_id, str) and image_id not in prediction_by_id:
            issues.append(
                PredictionIssue(
                    kind="missing_prediction",
                    image_id=image_id,
                    message="No prediction row found for this ground-truth image_id.",
                )
            )

    for image_id in sorted(set(prediction_by_id) - ground_truth_ids):
        issues.append(
            PredictionIssue(
                kind="extra_prediction",
                image_id=image_id,
                message="Prediction image_id is not present in the ground truth.",
            )
        )

    return issues


def evaluate_predictions(
    ground_truth_rows: Sequence[dict[str, Any]],
    prediction_rows: Sequence[dict[str, Any]],
    tasks: Sequence[str] = TASKS,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    prediction_by_id, issues = build_prediction_index(prediction_rows, tasks=tasks)
    issues.extend(validate_ground_truth_rows(ground_truth_rows, tasks=tasks))
    issues.extend(validate_alignment(ground_truth_rows, prediction_by_id))
    if issues:
        return {}, [], [issue.to_dict() for issue in issues]

    metrics: dict[str, Any] = {
        "num_samples": len(ground_truth_rows),
        "tasks": {},
    }
    error_cases: list[dict[str, Any]] = []

    for task in tasks:
        y_true = [str(row[task]) for row in ground_truth_rows]
        y_pred = [
            str(prediction_by_id[str(row["image_id"])][task])
            for row in ground_truth_rows
        ]
        metrics["tasks"][task] = {
            "accuracy": accuracy(y_true, y_pred),
            "macro_f1": macro_f1(y_true, y_pred),
            "support": len(y_true),
        }

    for row in ground_truth_rows:
        image_id = str(row["image_id"])
        prediction = prediction_by_id[image_id]
        errors: dict[str, dict[str, str]] = {}
        for task in tasks:
            expected = str(row[task])
            actual = str(prediction[task])
            if expected != actual:
                errors[task] = {
                    "expected": expected,
                    "predicted": actual,
                }
        if errors:
            error_cases.append(
                {
                    "image_id": image_id,
                    "image_path": row.get("image_path"),
                    "errors": errors,
                }
            )

    return metrics, error_cases, []
