from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import ACDCDualHeadDataset, WEATHER_TO_ID, build_transforms, infer_object_label_columns
from src.models import MODEL_CHOICES, create_dual_head_model
from src.utils.checkpoint import save_checkpoint
from src.utils.config import apply_overrides, load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a dual-head ACDC classifier.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--metadata-dir", type=Path, default=None)
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--val-csv", type=Path, default=None)
    parser.add_argument("--model", dest="model_name", choices=MODEL_CHOICES, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--freeze-backbone", dest="freeze_backbone", action="store_true", default=None)
    parser.add_argument("--unfreeze-backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument("--pretrained-backbone", dest="pretrained_backbone", action="store_true", default=None)
    parser.add_argument("--random-backbone", dest="pretrained_backbone", action="store_false")
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> dict:
    config = load_yaml_config(args.config)
    overrides = {
        "metadata_dir": args.metadata_dir,
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "image_size": args.image_size,
        "num_workers": args.num_workers,
        "output_dir": args.output_dir,
        "image_root": args.image_root,
        "limit_train": args.limit_train,
        "limit_val": args.limit_val,
        "device": args.device,
        "freeze_backbone": args.freeze_backbone,
        "pretrained_backbone": args.pretrained_backbone,
    }
    config = apply_overrides(config, overrides)

    config.setdefault("metadata_dir", "/kaggle/working/metadata")
    config.setdefault("model_name", "mamba_t")
    config.setdefault("batch_size", 8)
    config.setdefault("lr", 1e-3)
    config.setdefault("weight_decay", 1e-4)
    config.setdefault("epochs", 1)
    config.setdefault("image_size", 224)
    config.setdefault("num_workers", 2)
    config.setdefault("output_dir", "checkpoints")
    config.setdefault("image_root", None)
    config.setdefault("hidden_dim", 256)
    config.setdefault("dropout", 0.1)
    config.setdefault("freeze_backbone", True)
    config.setdefault("pretrained_backbone", True)
    config.setdefault("weather_loss_weight", 1.0)
    config.setdefault("object_loss_weight", 1.0)
    config.setdefault("device", "auto")
    config.setdefault("seed", 42)
    return config


def resolve_csv_paths(config: dict) -> tuple[Path, Path]:
    metadata_dir = Path(config["metadata_dir"])
    if config.get("train_csv"):
        train_csv = Path(config["train_csv"])
    else:
        train_csv = first_existing_path(
            metadata_dir / "metadata_train.csv",
            metadata_dir / "head2_presence_train.csv",
        )

    if config.get("val_csv"):
        val_csv = Path(config["val_csv"])
    else:
        val_csv = first_existing_path(
            metadata_dir / "metadata_val.csv",
            metadata_dir / "head2_presence_val.csv",
        )

    return train_csv, val_csv


def first_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def limit_dataset(dataset, limit: int | None):
    if limit is None or limit >= len(dataset):
        return dataset
    return Subset(dataset, list(range(limit)))


def move_batch_to_device(batch, device: torch.device):
    images, weather_ids, object_targets = batch
    return images.to(device), weather_ids.to(device), object_targets.to(device)


def compute_loss(outputs, weather_ids, object_targets, weather_loss_fn, object_loss_fn, config):
    weather_loss = weather_loss_fn(outputs["weather_logits"], weather_ids)
    object_loss = object_loss_fn(outputs["object_logits"], object_targets)
    total_loss = (
        config["weather_loss_weight"] * weather_loss
        + config["object_loss_weight"] * object_loss
    )
    return total_loss, weather_loss, object_loss


def run_epoch(model, loader, optimizer, device, weather_loss_fn, object_loss_fn, config, train: bool) -> dict:
    model.train(train)
    totals = {"total_loss": 0.0, "weather_loss": 0.0, "object_loss": 0.0}
    sample_count = 0

    for batch in loader:
        images, weather_ids, object_targets = move_batch_to_device(batch, device)

        with torch.set_grad_enabled(train):
            outputs = model(images)
            total_loss, weather_loss, object_loss = compute_loss(
                outputs,
                weather_ids,
                object_targets,
                weather_loss_fn,
                object_loss_fn,
                config,
            )

            if train:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()

        batch_size = images.shape[0]
        totals["total_loss"] += float(total_loss.detach().cpu()) * batch_size
        totals["weather_loss"] += float(weather_loss.detach().cpu()) * batch_size
        totals["object_loss"] += float(object_loss.detach().cpu()) * batch_size
        sample_count += batch_size

    return {key: value / max(sample_count, 1) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    config = resolve_config(args)
    torch.manual_seed(int(config["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["seed"]))

    train_csv, val_csv = resolve_csv_paths(config)
    label_columns = infer_object_label_columns(pd.read_csv(train_csv, nrows=1))
    train_transform = build_transforms(int(config["image_size"]), train=True)
    val_transform = build_transforms(int(config["image_size"]), train=False)

    train_dataset = ACDCDualHeadDataset(
        train_csv,
        label_columns=label_columns,
        transform=train_transform,
        image_root=config.get("image_root"),
    )
    val_dataset = ACDCDualHeadDataset(
        val_csv,
        label_columns=label_columns,
        transform=val_transform,
        image_root=config.get("image_root"),
    )
    train_dataset = limit_dataset(train_dataset, config.get("limit_train"))
    val_dataset = limit_dataset(val_dataset, config.get("limit_val"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )

    requested_device = config["device"]
    device = torch.device("cuda" if requested_device == "auto" and torch.cuda.is_available() else requested_device)
    if requested_device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")

    model = create_dual_head_model(
        model_name=config["model_name"],
        num_weather_classes=len(WEATHER_TO_ID),
        num_object_classes=len(label_columns),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
        freeze_backbone=bool(config["freeze_backbone"]),
        pretrained_backbone=bool(config["pretrained_backbone"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    weather_loss_fn = nn.CrossEntropyLoss()
    object_loss_fn = nn.BCEWithLogitsLoss()
    history = []

    print("Training config:")
    print(json.dumps({key: str(value) for key, value in config.items()}, indent=2))
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | Device: {device}")

    for epoch in range(1, int(config["epochs"]) + 1):
        train_metrics = run_epoch(
            model, train_loader, optimizer, device, weather_loss_fn, object_loss_fn, config, train=True
        )
        val_metrics = run_epoch(
            model, val_loader, optimizer, device, weather_loss_fn, object_loss_fn, config, train=False
        )

        record = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(record)
        print(
            f"epoch {epoch:02d} | "
            f"train total={record['train_total_loss']:.4f}, weather={record['train_weather_loss']:.4f}, objects={record['train_object_loss']:.4f} | "
            f"val total={record['val_total_loss']:.4f}, weather={record['val_weather_loss']:.4f}, objects={record['val_object_loss']:.4f}"
        )

        checkpoint_path = save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "label_columns": label_columns,
                "weather_to_id": WEATHER_TO_ID,
                "history": history,
            },
            config["output_dir"],
            config["model_name"],
            epoch,
        )
        print(f"saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
