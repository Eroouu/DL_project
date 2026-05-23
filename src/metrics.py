

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
)


DEFAULT_WEATHER_CLASSES = ["fog", "night", "rain", "snow"]


@dataclass(frozen=True)
class ThresholdTuningResult:
    thresholds: list[float]
    per_class_f1: dict[str, float]


def compute_weather_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute metrics for the 4-class single-label weather head."""
    class_names = class_names or DEFAULT_WEATHER_CLASSES
    labels = list(range(len(class_names)))

    per_class_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class_f1": {
            class_name: float(score)
            for class_name, score in zip(class_names, per_class_f1, strict=True)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
        "class_names": class_names,
    }


def compute_object_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | list[float] | float = 0.5,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute metrics for the multi-label object-presence head."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    class_names = class_names or [f"class_{idx}" for idx in range(y_true.shape[1])]

    threshold_array = _as_threshold_array(thresholds, y_true.shape[1])
    y_pred = (y_prob >= threshold_array).astype(int)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )
    per_class_ap = _average_precision_per_class(y_true, y_prob)
    valid_ap = [score for score in per_class_ap if score is not None]

    return {
        "mAP": float(np.mean(valid_ap)) if valid_ap else None,
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "macro_f1": float(np.mean(f1)),
        "thresholds": {
            class_name: float(threshold)
            for class_name, threshold in zip(class_names, threshold_array, strict=True)
        },
        "per_class": {
            class_name: {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
                "average_precision": None if per_class_ap[idx] is None else float(per_class_ap[idx]),
            }
            for idx, class_name in enumerate(class_names)
        },
    }


def tune_object_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str] | None = None,
    threshold_grid: np.ndarray | None = None,
) -> ThresholdTuningResult:
    """Select one threshold per class by maximizing validation F1."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    class_names = class_names or [f"class_{idx}" for idx in range(y_true.shape[1])]
    threshold_grid = threshold_grid if threshold_grid is not None else np.arange(0.05, 1.0, 0.05)

    best_thresholds: list[float] = []
    best_f1: dict[str, float] = {}

    for class_idx, class_name in enumerate(class_names):
        class_true = y_true[:, class_idx]
        class_prob = y_prob[:, class_idx]

        best_score = -1.0
        best_threshold = 0.5
        for threshold in threshold_grid:
            class_pred = (class_prob >= threshold).astype(int)
            score = f1_score(class_true, class_pred, zero_division=0)
            if score > best_score:
                best_score = float(score)
                best_threshold = float(threshold)

        best_thresholds.append(best_threshold)
        best_f1[class_name] = best_score

    return ThresholdTuningResult(thresholds=best_thresholds, per_class_f1=best_f1)


def compute_per_weather_object_metrics(
    weather_true: np.ndarray,
    object_true: np.ndarray,
    object_prob: np.ndarray,
    thresholds: np.ndarray | list[float] | float = 0.5,
    weather_classes: list[str] | None = None,
    object_classes: list[str] | None = None,
) -> dict[str, Any]:
    """Compute object metrics separately for each weather condition."""
    weather_classes = weather_classes or DEFAULT_WEATHER_CLASSES
    result: dict[str, Any] = {}

    for weather_idx, weather_name in enumerate(weather_classes):
        mask = weather_true == weather_idx
        if not np.any(mask):
            result[weather_name] = {"num_samples": 0, "metrics": None}
            continue

        result[weather_name] = {
            "num_samples": int(np.sum(mask)),
            "metrics": compute_object_metrics(
                object_true[mask],
                object_prob[mask],
                thresholds=thresholds,
                class_names=object_classes,
            ),
        }

    return result


def _as_threshold_array(thresholds: np.ndarray | list[float] | float, num_classes: int) -> np.ndarray:
    threshold_array = np.asarray(thresholds, dtype=float)
    if threshold_array.ndim == 0:
        return np.full(num_classes, float(threshold_array))
    if threshold_array.shape != (num_classes,):
        raise ValueError(f"Expected {num_classes} thresholds, got shape {threshold_array.shape}")
    return threshold_array


def _average_precision_per_class(y_true: np.ndarray, y_prob: np.ndarray) -> list[float | None]:
    scores: list[float | None] = []
    for class_idx in range(y_true.shape[1]):
        class_true = y_true[:, class_idx]
        if class_true.sum() == 0:
            scores.append(None)
            continue
        scores.append(float(average_precision_score(class_true, y_prob[:, class_idx])))
    return scores
