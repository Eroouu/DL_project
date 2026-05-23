from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


WEATHER_TO_ID = {
    "fog": 0,
    "night": 1,
    "rain": 2,
    "snow": 3,
}
ID_TO_WEATHER = {value: key for key, value in WEATHER_TO_ID.items()}

TRAIN_ID_TO_NAME = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
}
OBJECT_LABEL_COLUMNS = [
    f"has_{TRAIN_ID_TO_NAME[i].replace(' ', '_')}" for i in range(len(TRAIN_ID_TO_NAME))
]


def build_transforms(image_size: int = 224, train: bool = False) -> transforms.Compose:
    transform_steps = [transforms.Resize((image_size, image_size))]
    if train:
        transform_steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
            ]
        )
    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return transforms.Compose(transform_steps)


def infer_object_label_columns(dataframe: pd.DataFrame) -> list[str]:
    columns = [column for column in dataframe.columns if column.startswith("has_")]
    if not columns:
        raise ValueError("No object label columns found. Expected columns with 'has_' prefix.")
    return columns


def _read_csv(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return pd.read_csv(csv_path)


def _resolve_image_path(path_value: str | Path, image_root: str | Path | None = None) -> Path:
    image_path = Path(path_value)
    if image_path.exists():
        return image_path
    if image_root is not None:
        rooted_path = Path(image_root) / image_path
        if rooted_path.exists():
            return rooted_path
    return image_path


def _weather_id_from_row(row: pd.Series, weather_to_id: dict[str, int]) -> int:
    if "weather_id" in row and not pd.isna(row["weather_id"]):
        return int(row["weather_id"])
    if "weather" in row and row["weather"] in weather_to_id:
        return weather_to_id[str(row["weather"])]
    if "weather_label" in row and row["weather_label"] in weather_to_id:
        return weather_to_id[str(row["weather_label"])]
    raise KeyError("CSV row must contain weather_id or weather/weather_label.")


class ACDCWeatherDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        transform=None,
        image_root: str | Path | None = None,
        weather_to_id: dict[str, int] | None = None,
    ):
        self.df = _read_csv(csv_path).reset_index(drop=True)
        self.transform = transform
        self.image_root = image_root
        self.weather_to_id = weather_to_id or WEATHER_TO_ID

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = _resolve_image_path(row["image_path"], self.image_root)
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        weather_id = _weather_id_from_row(row, self.weather_to_id)
        return image, torch.tensor(weather_id, dtype=torch.long)


class ACDCObjectPresenceDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        label_columns: Iterable[str] | None = None,
        transform=None,
        image_root: str | Path | None = None,
    ):
        self.df = _read_csv(csv_path).reset_index(drop=True)
        self.label_columns = list(label_columns) if label_columns is not None else infer_object_label_columns(self.df)
        self.transform = transform
        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = _resolve_image_path(row["image_path"], self.image_root)
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        target = row[self.label_columns].to_numpy(dtype=np.float32)
        return image, torch.tensor(target, dtype=torch.float32)


class ACDCDualHeadDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        label_columns: Iterable[str] | None = None,
        transform=None,
        image_root: str | Path | None = None,
        weather_to_id: dict[str, int] | None = None,
    ):
        self.df = _read_csv(csv_path).reset_index(drop=True)
        self.label_columns = list(label_columns) if label_columns is not None else infer_object_label_columns(self.df)
        self.transform = transform
        self.image_root = image_root
        self.weather_to_id = weather_to_id or WEATHER_TO_ID

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = _resolve_image_path(row["image_path"], self.image_root)
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        weather_id = _weather_id_from_row(row, self.weather_to_id)
        object_target = row[self.label_columns].to_numpy(dtype=np.float32)

        return (
            image,
            torch.tensor(weather_id, dtype=torch.long),
            torch.tensor(object_target, dtype=torch.float32),
        )
