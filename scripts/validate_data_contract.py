from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval import load_class_map


REQUIRED_COLUMNS = {
    "image_path",
    "weather_label",
    "weather_id",
    "official_split",
    "split",
    "sequence",
    "frame",
    "mask_path",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate metadata/classmap/train CSV contract.")
    parser.add_argument("--metadata-dir", type=Path, default=Path("metadata"))
    parser.add_argument("--classmap", type=Path, default=Path("configs/classmap.json"))
    parser.add_argument("--train-csv", type=Path)
    parser.add_argument("--val-csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_csv = args.train_csv or first_existing_path(
        args.metadata_dir / "metadata_train.csv",
        args.metadata_dir / "head2_presence_train.csv",
    )
    val_csv = args.val_csv or first_existing_path(
        args.metadata_dir / "metadata_val.csv",
        args.metadata_dir / "head2_presence_val.csv",
    )
    classmap = load_class_map(args.classmap)
    expected_object_columns = classmap["object_columns"]

    for csv_path, expected_split in [(train_csv, "train"), (val_csv, "val")]:
        df = pd.read_csv(csv_path)
        object_columns = [column for column in df.columns if column.startswith("has_")]

        missing = sorted(REQUIRED_COLUMNS - set(df.columns))
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {missing}")
        if object_columns != expected_object_columns:
            raise ValueError(
                f"{csv_path} object columns do not match {args.classmap}: "
                f"{json.dumps(object_columns, ensure_ascii=False)}"
            )
        if "has_road" in object_columns:
            raise ValueError(f"{csv_path} must not contain has_road")
        if set(df["split"]) != {expected_split}:
            raise ValueError(f"{csv_path} expected split={expected_split}, got {sorted(df['split'].unique())}")
        if sorted(df["weather_label"].unique()) != classmap["weather_classes"]:
            raise ValueError(f"{csv_path} weather labels do not match classmap")
        if not df["image_path"].map(lambda value: Path(value).exists()).all():
            missing_count = int((~df["image_path"].map(lambda value: Path(value).exists())).sum())
            raise FileNotFoundError(f"{csv_path} contains {missing_count} missing image paths")
        if not df["mask_path"].map(lambda value: Path(value).exists()).all():
            missing_count = int((~df["mask_path"].map(lambda value: Path(value).exists())).sum())
            raise FileNotFoundError(f"{csv_path} contains {missing_count} missing mask paths")

        print(
            f"{csv_path}: ok | rows={len(df)} | object_columns={len(object_columns)} | "
            f"weather={sorted(df['weather_label'].unique())}"
        )


def first_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


if __name__ == "__main__":
    main()
