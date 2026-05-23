from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(payload: dict, output_dir: str | Path, model_name: str, epoch: int) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{model_name}_epoch{epoch:02d}.pt"
    torch.save(payload, checkpoint_path)
    return checkpoint_path
