from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


DEFAULT_CLASSMAP_PATH = Path("configs/classmap.json")
DEFAULT_INTERNAL_VAL_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42


@dataclass(frozen=True)
class AcdcPaths:
    data_root: Path
    rgb_root: Path
    gt_root: Path


def load_classmap(path: Path = DEFAULT_CLASSMAP_PATH) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    weather_to_id = data.get("weather_to_id")
    object_classes = data.get("object_classes")
    object_columns = data.get("object_columns")

    if not weather_to_id:
        raise ValueError("classmap must contain weather_to_id")
    if not object_classes:
        raise ValueError("classmap must contain object_classes")
    if not object_columns:
        data["object_columns"] = [f"has_{name}" for name in object_classes]
    if len(data["object_columns"]) != len(object_classes):
        raise ValueError("object_columns and object_classes must have the same length")

    return data


def resolve_acdc_paths(data_root: Path) -> AcdcPaths:
    return AcdcPaths(
        data_root=data_root,
        rgb_root=data_root / "rgb_anon_trainvaltest" / "rgb_anon",
        gt_root=data_root / "gt_trainval" / "gt",
    )


def get_acdc_key(path: Path) -> tuple[str, str, str, str]:
    """Return (weather, official_split, sequence, frame) for ACDC RGB or GT files."""
    weather = path.parts[-4]
    official_split = path.parts[-3]
    sequence = path.parts[-2]
    frame = path.name.split("_frame_")[1].split("_")[0]
    return weather, official_split, sequence, frame


def mask_to_multihot(mask_path: Path, num_classes: int) -> np.ndarray:
    mask = np.array(Image.open(mask_path))
    class_ids = np.unique(mask)
    class_ids = class_ids[(0 <= class_ids) & (class_ids < num_classes)]

    target = np.zeros(num_classes, dtype=np.int64)
    target[class_ids.astype(np.int64)] = 1
    return target


def build_labeled_metadata(
    data_root: Path,
    classmap: dict[str, Any],
    include_objects: bool = True,
) -> pd.DataFrame:
    paths = resolve_acdc_paths(data_root)
    weather_to_id = classmap["weather_to_id"]
    object_columns = classmap["object_columns"]

    rgb_files = sorted(paths.rgb_root.rglob("*_rgb_anon.png"))
    rgb_index = {get_acdc_key(path): path for path in rgb_files}
    if len(rgb_index) != len(rgb_files):
        raise ValueError("Some RGB files produced duplicate ACDC keys")

    mask_index: dict[tuple[str, str, str, str], Path] = {}
    if include_objects:
        mask_files = sorted(paths.gt_root.rglob("*_gt_labelTrainIds.png"))
        mask_index = {get_acdc_key(path): path for path in mask_files}
        if len(mask_index) != len(mask_files):
            raise ValueError("Some mask files produced duplicate ACDC keys")

    rows: list[dict[str, Any]] = []
    for key, rgb_path in sorted(rgb_index.items()):
        weather_label, official_split, sequence, frame = key
        if official_split not in {"train", "val"}:
            continue
        if weather_label not in weather_to_id:
            continue

        row: dict[str, Any] = {
            "image_path": str(rgb_path),
            "weather_label": weather_label,
            "weather_id": int(weather_to_id[weather_label]),
            "official_split": official_split,
            "sequence": sequence,
            "frame": frame,
        }

        if include_objects:
            mask_path = mask_index.get(key)
            if mask_path is None:
                raise FileNotFoundError(f"Mask not found for RGB image: {rgb_path}")
            row["mask_path"] = str(mask_path)
            target = mask_to_multihot(mask_path, num_classes=len(object_columns))
            for column, value in zip(object_columns, target, strict=True):
                row[column] = int(value)

        rows.append(row)

    return pd.DataFrame(rows)


def make_internal_splits(
    df: pd.DataFrame,
    internal_val_size: float = DEFAULT_INTERNAL_VAL_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    official_train = df[df["official_split"] == "train"].copy().reset_index(drop=True)
    test_df = df[df["official_split"] == "val"].copy().reset_index(drop=True)

    train_df, val_df = train_test_split(
        official_train,
        test_size=internal_val_size,
        random_state=random_state,
        stratify=official_train["weather_id"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    return train_df, val_df, test_df, all_df


def limit_rows_per_split(df: pd.DataFrame, max_rows: int | None, random_state: int) -> pd.DataFrame:
    if max_rows is None:
        return df

    pieces = []
    for split_name, split_df in df.groupby("split", sort=False):
        if len(split_df) > max_rows:
            split_df = split_df.sample(n=max_rows, random_state=random_state)
        pieces.append(split_df.sort_index())
    return pd.concat(pieces, ignore_index=True)


def write_metadata_splits(
    all_df: pd.DataFrame,
    out_dir: Path,
    prefix: str = "metadata",
    quick_limit_per_split: int | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, Path] = {}

    export_df = limit_rows_per_split(all_df, quick_limit_per_split, random_state)
    for split_name in ["train", "val", "test"]:
        split_df = export_df[export_df["split"] == split_name].reset_index(drop=True)
        path = out_dir / f"{prefix}_{split_name}.csv"
        split_df.to_csv(path, index=False)
        output_paths[split_name] = path

    all_path = out_dir / f"{prefix}_all.csv"
    export_df.to_csv(all_path, index=False)
    output_paths["all"] = all_path
    return output_paths


def make_smoke_subset_indices(dataframe: pd.DataFrame, subset_size: int, seed: int = DEFAULT_RANDOM_STATE) -> list[int]:
    if subset_size >= len(dataframe):
        return list(range(len(dataframe)))
    return dataframe.sample(n=subset_size, random_state=seed).index.tolist()


class ACDCWeatherDataset:
    def __init__(self, dataframe: pd.DataFrame, transform: Any = None) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        import torch

        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(int(row["weather_id"]), dtype=torch.long)


class ACDCDualHeadDataset:
    def __init__(self, dataframe: pd.DataFrame, object_columns: list[str], transform: Any = None) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.object_columns = list(object_columns)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[Any, Any, Any]:
        import torch

        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        weather_id = torch.tensor(int(row["weather_id"]), dtype=torch.long)
        object_target = row[self.object_columns].to_numpy(dtype=np.float32)
        object_target = torch.tensor(object_target, dtype=torch.float32)
        return image, weather_id, object_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build standardized ACDC metadata CSV files.")
    parser.add_argument("--data-root", type=Path, required=True, help="Root with rgb_anon_trainvaltest/ and gt_trainval/.")
    parser.add_argument("--classmap", type=Path, default=DEFAULT_CLASSMAP_PATH)
    parser.add_argument("--out-dir", type=Path, default=Path("metadata"))
    parser.add_argument("--prefix", default="metadata")
    parser.add_argument("--internal-val-size", type=float, default=DEFAULT_INTERNAL_VAL_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--quick-limit-per-split", type=int, help="Write at most N rows per split for fast notebook runs.")
    parser.add_argument("--weather-only", action="store_true", help="Skip gt_labelTrainIds masks and has_* columns.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classmap = load_classmap(args.classmap)
    base_df = build_labeled_metadata(
        data_root=args.data_root,
        classmap=classmap,
        include_objects=not args.weather_only,
    )
    _, _, _, all_df = make_internal_splits(
        base_df,
        internal_val_size=args.internal_val_size,
        random_state=args.seed,
    )
    output_paths = write_metadata_splits(
        all_df,
        out_dir=args.out_dir,
        prefix=args.prefix,
        quick_limit_per_split=args.quick_limit_per_split,
        random_state=args.seed,
    )

    print(f"Built {len(all_df)} labeled rows")
    for split_name, path in output_paths.items():
        print(f"{split_name}: {path}")


if __name__ == "__main__":
    main()
