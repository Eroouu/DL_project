from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import first_existing_path
from src.data.datasets import ACDCDualHeadDataset, WEATHER_TO_ID, build_transforms, infer_object_label_columns
from src.models import create_dual_head_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export validation predictions to NPZ for eval.py.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, default=Path("metadata"))
    parser.add_argument("--val-csv", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config", {})
    label_columns = checkpoint.get("label_columns")

    val_csv = args.val_csv or first_existing_path(
        args.metadata_dir / "metadata_val.csv",
        args.metadata_dir / "head2_presence_val.csv",
    )
    val_df_head = pd.read_csv(val_csv, nrows=1)
    if label_columns is None:
        label_columns = infer_object_label_columns(val_df_head)

    image_size = args.image_size or int(config.get("image_size", 224))
    transform = build_transforms(image_size, train=False)
    dataset = ACDCDualHeadDataset(
        val_csv,
        label_columns=label_columns,
        transform=transform,
        image_root=args.image_root or config.get("image_root"),
    )
    if args.limit_val is not None and args.limit_val < len(dataset):
        dataset = Subset(dataset, list(range(args.limit_val)))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    requested_device = args.device
    device = torch.device("cuda" if requested_device == "auto" and torch.cuda.is_available() else requested_device)
    if requested_device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")

    model = create_dual_head_model(
        model_name=config.get("model_name", "mamba_t"),
        num_weather_classes=len(checkpoint.get("weather_to_id", WEATHER_TO_ID)),
        num_object_classes=len(label_columns),
        hidden_dim=int(config.get("hidden_dim", 256)),
        dropout=float(config.get("dropout", 0.1)),
        freeze_backbone=False,
        pretrained_backbone=bool(config.get("pretrained_backbone", True)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    weather_logits = []
    object_logits = []
    weather_true = []
    object_true = []

    with torch.no_grad():
        for images, weather_ids, object_targets in loader:
            outputs = model(images.to(device))
            weather_logits.append(outputs["weather_logits"].cpu().numpy())
            object_logits.append(outputs["object_logits"].cpu().numpy())
            weather_true.append(weather_ids.numpy())
            object_true.append(object_targets.numpy())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        weather_logits=np.concatenate(weather_logits, axis=0),
        object_logits=np.concatenate(object_logits, axis=0),
        weather_true=np.concatenate(weather_true, axis=0),
        object_true=np.concatenate(object_true, axis=0),
        label_columns=np.array(label_columns),
    )
    print(f"Saved predictions to {args.out}")
    print(json.dumps({"samples": len(dataset), "label_columns": list(label_columns)}, indent=2))


if __name__ == "__main__":
    main()
