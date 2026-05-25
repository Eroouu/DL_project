from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a comparison table from eval.py metric JSON files.")
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--names", nargs="+", help="Optional model names in the same order as --metrics.")
    parser.add_argument("--out-csv", type=Path, default=Path("reports/metrics/model_comparison.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("reports/metrics/model_comparison.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.names is not None and len(args.names) != len(args.metrics):
        raise ValueError("--names length must match --metrics length")

    rows = []
    for idx, metrics_path in enumerate(args.metrics):
        model_name = args.names[idx] if args.names else metrics_path.stem
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(flatten_metrics(model_name, metrics))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out_csv)
    write_markdown(rows, args.out_md)

    print(f"Saved comparison CSV to {args.out_csv}")
    print(f"Saved comparison Markdown to {args.out_md}")


def flatten_metrics(model_name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    weather = metrics["weather"]
    objects = metrics["objects"]
    return {
        "model": model_name,
        "num_samples": metrics["num_samples"],
        "weather_accuracy": weather["accuracy"],
        "weather_macro_f1": weather["macro_f1"],
        "object_mAP": objects["mAP"],
        "object_macro_f1": objects["macro_f1"],
        "object_hamming_loss": objects["hamming_loss"],
    }


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], out_path: Path) -> None:
    columns = list(rows[0].keys())
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = [format_value(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if value is None:
        return "n/a"
    return str(value)


if __name__ == "__main__":
    main()
