"""Evaluation entry point for the ACDC dual-head classifier.

The file already works in dummy mode, so metrics can be tested before the
training code and checkpoints are ready.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from src.metrics import (
        DEFAULT_WEATHER_CLASSES,
        compute_object_metrics,
        compute_per_weather_object_metrics,
        compute_weather_metrics,
        tune_object_thresholds,
    )
except ModuleNotFoundError:
    from metrics import (
        DEFAULT_WEATHER_CLASSES,
        compute_object_metrics,
        compute_per_weather_object_metrics,
        compute_weather_metrics,
        tune_object_thresholds,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ACDC dual-head classifier outputs.")
    parser.add_argument("--metadata", type=Path, help="CSV with image paths and ground-truth labels.")
    parser.add_argument("--predictions", type=Path, help="NPZ with weather_logits/object_logits or probabilities.")
    parser.add_argument("--class-map", type=Path, default=Path("configs/class_map.example.json"))
    parser.add_argument("--out", type=Path, default=Path("reports/metrics/dummy_eval.json"))
    parser.add_argument("--dummy", action="store_true", help="Generate random predictions for pipeline testing.")
    parser.add_argument("--tune-thresholds", action="store_true", help="Tune per-class object thresholds on this split.")
    parser.add_argument("--num-samples", type=int, default=128, help="Dummy sample count when metadata is absent.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_map = load_class_map(args.class_map)
    weather_classes = class_map["weather_classes"]
    object_classes = class_map["object_classes"]

    rng = np.random.default_rng(args.seed)
    weather_true, object_true = load_ground_truth(
        metadata_path=args.metadata,
        weather_classes=weather_classes,
        object_classes=object_classes,
        num_samples=args.num_samples,
        rng=rng,
    )

    weather_prob, object_prob = load_predictions(
        predictions_path=args.predictions,
        dummy=args.dummy,
        num_samples=len(weather_true),
        num_weather_classes=len(weather_classes),
        num_object_classes=len(object_classes),
        rng=rng,
    )

    weather_pred = np.argmax(weather_prob, axis=1)

    threshold_result = None
    object_thresholds: float | list[float] = 0.5
    if args.tune_thresholds:
        threshold_result = tune_object_thresholds(object_true, object_prob, class_names=object_classes)
        object_thresholds = threshold_result.thresholds

    result: dict[str, Any] = {
        "num_samples": int(len(weather_true)),
        "weather": compute_weather_metrics(weather_true, weather_pred, class_names=weather_classes),
        "objects": compute_object_metrics(
            object_true,
            object_prob,
            thresholds=object_thresholds,
            class_names=object_classes,
        ),
        "objects_by_weather": compute_per_weather_object_metrics(
            weather_true,
            object_true,
            object_prob,
            thresholds=object_thresholds,
            weather_classes=weather_classes,
            object_classes=object_classes,
        ),
    }
    if threshold_result is not None:
        result["threshold_tuning"] = {
            "thresholds": {
                class_name: float(threshold)
                for class_name, threshold in zip(object_classes, threshold_result.thresholds, strict=True)
            },
            "per_class_f1": threshold_result.per_class_f1,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_metric_tables(result, args.out)
    print_metric_summary(result)
    print(f"Saved evaluation metrics to {args.out}")


def load_class_map(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Class map not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return {"weather_classes": DEFAULT_WEATHER_CLASSES, "object_classes": raw}

    object_classes = raw.get("object_classes")
    if not object_classes:
        raise ValueError("class_map must contain object_classes")

    return {
        "weather_classes": raw.get("weather_classes", DEFAULT_WEATHER_CLASSES),
        "object_classes": object_classes,
    }


def load_ground_truth(
    metadata_path: Path | None,
    weather_classes: list[str],
    object_classes: list[str],
    num_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if metadata_path is None:
        weather_true = rng.integers(0, len(weather_classes), size=num_samples)
        object_true = rng.integers(0, 2, size=(num_samples, len(object_classes)))
        return weather_true, object_true

    rows = read_csv(metadata_path)
    weather_true = np.array(
        [parse_weather_label(row["weather_label"], weather_classes) for row in rows],
        dtype=int,
    )
    object_true = np.array(
        [[int(row[class_name]) for class_name in object_classes] for row in rows],
        dtype=int,
    )
    return weather_true, object_true


def load_predictions(
    predictions_path: Path | None,
    dummy: bool,
    num_samples: int,
    num_weather_classes: int,
    num_object_classes: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if dummy:
        weather_logits = rng.normal(size=(num_samples, num_weather_classes))
        object_logits = rng.normal(size=(num_samples, num_object_classes))
        return softmax(weather_logits), sigmoid(object_logits)

    if predictions_path is None:
        raise ValueError("Pass --dummy or provide --predictions")

    data = np.load(predictions_path)
    if "weather_prob" in data and "object_prob" in data:
        return data["weather_prob"], data["object_prob"]
    if "weather_logits" in data and "object_logits" in data:
        return softmax(data["weather_logits"]), sigmoid(data["object_logits"])

    raise ValueError("Predictions NPZ must contain weather/object logits or probabilities")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def parse_weather_label(value: str, weather_classes: list[str]) -> int:
    if value.isdigit():
        return int(value)
    return weather_classes.index(value)


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def write_metric_tables(result: dict[str, Any], json_path: Path) -> None:
    summary_path = json_path.with_suffix(".summary.md")
    per_class_path = json_path.with_suffix(".per_class.csv")
    per_weather_path = json_path.with_suffix(".per_weather.csv")

    weather = result["weather"]
    objects = result["objects"]
    object_map = objects["mAP"]
    object_map_text = "n/a" if object_map is None else f"{object_map:.6f}"

    summary_path.write_text(
        "\n".join(
            [
                "# Evaluation Summary",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| Samples | {result['num_samples']} |",
                f"| Weather accuracy | {weather['accuracy']:.6f} |",
                f"| Weather macro F1 | {weather['macro_f1']:.6f} |",
                f"| Object mAP | {object_map_text} |",
                f"| Object macro F1 | {objects['macro_f1']:.6f} |",
                f"| Object Hamming loss | {objects['hamming_loss']:.6f} |",
                "",
                "## Weather Confusion Matrix",
                "",
                _format_confusion_matrix_markdown(weather["class_names"], weather["confusion_matrix"]),
                "",
            ]
        ),
        encoding="utf-8",
    )

    with per_class_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["class", "precision", "recall", "f1", "support", "average_precision", "threshold"],
        )
        writer.writeheader()
        for class_name, class_metrics in objects["per_class"].items():
            writer.writerow(
                {
                    "class": class_name,
                    "precision": class_metrics["precision"],
                    "recall": class_metrics["recall"],
                    "f1": class_metrics["f1"],
                    "support": class_metrics["support"],
                    "average_precision": class_metrics["average_precision"],
                    "threshold": objects["thresholds"][class_name],
                }
            )

    with per_weather_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["weather", "num_samples", "object_mAP", "object_macro_f1", "object_hamming_loss"],
        )
        writer.writeheader()
        for weather_name, weather_result in result["objects_by_weather"].items():
            metrics = weather_result["metrics"]
            writer.writerow(
                {
                    "weather": weather_name,
                    "num_samples": weather_result["num_samples"],
                    "object_mAP": None if metrics is None else metrics["mAP"],
                    "object_macro_f1": None if metrics is None else metrics["macro_f1"],
                    "object_hamming_loss": None if metrics is None else metrics["hamming_loss"],
                }
            )


def _format_confusion_matrix_markdown(class_names: list[str], matrix: list[list[int]]) -> str:
    header = "| true \\ pred | " + " | ".join(class_names) + " |"
    separator = "|---|" + "|".join(["---:"] * len(class_names)) + "|"
    rows = [
        f"| {class_name} | " + " | ".join(str(value) for value in matrix[idx]) + " |"
        for idx, class_name in enumerate(class_names)
    ]
    return "\n".join([header, separator, *rows])


def print_metric_summary(result: dict[str, Any]) -> None:
    weather = result["weather"]
    objects = result["objects"]

    print("\nEvaluation summary")
    print("------------------")
    print(f"Samples: {result['num_samples']}")
    print(f"Weather accuracy: {weather['accuracy']:.4f}")
    print(f"Weather macro F1: {weather['macro_f1']:.4f}")

    object_map = objects["mAP"]
    object_map_text = "n/a" if object_map is None else f"{object_map:.4f}"
    print(f"Object mAP: {object_map_text}")
    print(f"Object macro F1: {objects['macro_f1']:.4f}")
    print(f"Object Hamming loss: {objects['hamming_loss']:.4f}")

    print("\nPer-class object F1")
    for class_name, class_metrics in objects["per_class"].items():
        print(f"- {class_name}: {class_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
