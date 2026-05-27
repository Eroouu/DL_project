# Smoke Run Guide

This document explains how to run and verify the short smoke-training path for Egor's dual-head part of the ACDC project.

## Purpose

The smoke run is not final model training. It is a fast end-to-end check that verifies:

- metadata CSV files are readable;
- the dual-head dataset returns image, weather label, and object-presence target;
- the selected backbone and both heads can run forward;
- the multitask loss is computed;
- backward pass and optimizer step work;
- epoch logging works;
- a checkpoint is written to disk.

## MambaVision Smoke On Kaggle

Use this path when the Kaggle environment has GPU and Internet enabled.

First, prepare metadata in `/kaggle/working/metadata`. This can be done by running the data-preparation cells in `dl-project.ipynb` or by running the repository script after the ACDC archives are available:

```bash
python -m src.prepare_acdc \
  --gt-zip gt_trainval.zip \
  --rgb-zip rgb_anon_trainvaltest.zip \
  --data-root /kaggle/working/data/acdc \
  --metadata-dir /kaggle/working/metadata \
  --prefix metadata
```

Then run the MambaVision-T smoke config:

```bash
python scripts/train.py --config configs/mamba_t_smoke.yaml
```

The smoke config uses:

```text
model_name: mamba_t
epochs: 1
batch_size: 8
limit_train: 64
limit_val: 32
freeze_backbone: true
output_dir: /kaggle/working/checkpoints
```

Expected checkpoint:

```text
/kaggle/working/checkpoints/mamba_t_epoch01.pt
```

## Local Smoke Fallback

If MambaVision dependencies are unavailable locally, use the ResNet-50 smoke config to verify the same training pipeline without `mamba-ssm`:

```bash
python scripts/train.py --config configs/resnet50_smoke.yaml
```

Expected checkpoint:

```text
checkpoints/resnet50_epoch01.pt
```

This fallback verifies the repository training path, checkpoint saving, CSV-backed dataset loading, and multitask loss. It does not validate MambaVision-specific CUDA dependencies.

## Checkpoint Verification

After a smoke run, verify the checkpoint can be loaded:

```python
from pathlib import Path
import torch

checkpoint_path = Path("/kaggle/working/checkpoints/mamba_t_epoch01.pt")
assert checkpoint_path.exists(), checkpoint_path

checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
print(checkpoint.keys())
print("epoch:", checkpoint["epoch"])
print("history:", checkpoint["history"])
print("labels:", len(checkpoint["label_columns"]))
```

For the local ResNet fallback, replace the path with:

```python
checkpoint_path = Path("checkpoints/resnet50_epoch01.pt")
```

## Expected Log Shape

A successful run should print one epoch summary similar to:

```text
epoch 01 | train total=..., weather=..., objects=... | val total=..., weather=..., objects=...
saved checkpoint: /kaggle/working/checkpoints/mamba_t_epoch01.pt
```

The exact loss values are not important for smoke. The important part is that the command reaches `saved checkpoint`.

## What To Commit

Commit source/config/documentation changes:

```text
configs/mamba_t_smoke.yaml
configs/resnet50_smoke.yaml
scripts/train.py
src/
reports/smoke_run.md
```

Do not commit local datasets or downloaded archives:

```text
data/
gt_trainval.zip
rgb_anon_trainvaltest.zip
```

Checkpoint files are large. If the team requires checkpoint files in GitHub, commit them through Git LFS:

```bash
git lfs install
git lfs track "checkpoints/*.pt"
git add .gitattributes
git add -f checkpoints/*.pt
git commit -m "Add smoke checkpoints"
```

Otherwise, record the checkpoint path and the successful smoke log in the report.

