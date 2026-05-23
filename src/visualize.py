"""Build report figures from eval.py metric JSON output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create figures from evaluation metrics JSON.")
    parser.add_argument("--metrics", type=Path, required=True, help="Path to eval.py JSON output.")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--prefix", default=None, help="Figure filename prefix. Defaults to metrics stem.")
    parser.add_argument("--top-k", type=int, default=18, help="Max object classes to show in bar charts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or args.metrics.stem

    paths = [
        plot_weather_confusion_matrix(metrics, args.out_dir / f"{prefix}_weather_confusion.png"),
        plot_object_metric_bars(metrics, args.out_dir / f"{prefix}_object_f1_ap.png", top_k=args.top_k),
        plot_per_weather_metrics(metrics, args.out_dir / f"{prefix}_per_weather.png"),
    ]

    print("Saved figures:")
    for path in paths:
        print(f"- {path}")


def plot_weather_confusion_matrix(metrics: dict[str, Any], out_path: Path) -> Path:
    weather = metrics["weather"]
    class_names = weather["class_names"]
    matrix = np.asarray(weather["confusion_matrix"], dtype=int)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("Weather Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(class_names)), labels=class_names, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)

    threshold = matrix.max() / 2 if matrix.size else 0
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            color = "white" if value > threshold else "black"
            ax.text(col_idx, row_idx, str(value), ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_object_metric_bars(metrics: dict[str, Any], out_path: Path, top_k: int) -> Path:
    per_class = metrics["objects"]["per_class"]
    class_names = list(per_class.keys())[:top_k]
    f1_scores = [per_class[class_name]["f1"] for class_name in class_names]
    ap_scores = [
        0.0 if per_class[class_name]["average_precision"] is None else per_class[class_name]["average_precision"]
        for class_name in class_names
    ]

    x = np.arange(len(class_names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(9, len(class_names) * 0.55), 5.5))
    ax.bar(x - width / 2, f1_scores, width=width, label="F1", color="#4C78A8")
    ax.bar(x + width / 2, ap_scores, width=width, label="AP", color="#F58518")

    ax.set_title("Object Metrics by Class")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x, labels=class_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_per_weather_metrics(metrics: dict[str, Any], out_path: Path) -> Path:
    by_weather = metrics["objects_by_weather"]
    weather_names = list(by_weather.keys())
    map_scores = []
    f1_scores = []

    for weather_name in weather_names:
        weather_metrics = by_weather[weather_name]["metrics"]
        if weather_metrics is None:
            map_scores.append(0.0)
            f1_scores.append(0.0)
            continue
        map_scores.append(0.0 if weather_metrics["mAP"] is None else weather_metrics["mAP"])
        f1_scores.append(weather_metrics["macro_f1"])

    x = np.arange(len(weather_names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, map_scores, width=width, label="mAP", color="#54A24B")
    ax.bar(x + width / 2, f1_scores, width=width, label="Macro F1", color="#E45756")

    ax.set_title("Object Performance by Weather")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x, labels=weather_names)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    main()
