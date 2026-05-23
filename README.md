# ACDC Dual-Head Training

This repository contains the original Kaggle notebook and a script-based training path for the ACDC image-level dual-head classifier.

Egor's part covers the training infrastructure: datasets, MambaVision dual-head model, configs, and a simple checkpoint-producing `train.py`.

## Environment

Use the existing local environment:

```powershell
C:\Users\egor0\anaconda3\envs\diplom314\python.exe -m pip install -r requirements.txt
```

On this machine, `diplom314` already has the needed training packages installed.

## Project Structure

```text
src/
  data/datasets.py              CSV-backed ACDC datasets and transforms
  models/mamba_dual_head.py     MambaVision backbone + weather/object heads
  utils/                        config and checkpoint helpers
scripts/
  train.py                      simple training entrypoint
configs/
  mamba_t_smoke.yaml            short MambaVision-T run on a small subset
  mamba_s_smoke.yaml            short MambaVision-S architecture check
  mamba_t_full.yaml             first longer MambaVision-T config
```

## Expected Metadata

Run the notebook preprocessing cells first. Training reads ready CSV files from `metadata_dir`:

```text
head2_presence_train.csv
head2_presence_val.csv
```

Each CSV must contain:

- `image_path`
- either `weather_id` or `weather` / `weather_label`
- object columns with the `has_` prefix, for example `has_road`, `has_car`, `has_sky`

## Train

Kaggle smoke run after metadata is created:

```bash
python scripts/train.py --config configs/mamba_t_smoke.yaml
```

First longer run:

```bash
python scripts/train.py --config configs/mamba_t_full.yaml
```

MambaVision-S architecture/config check:

```bash
python scripts/train.py --config configs/mamba_s_smoke.yaml
```

Direct CLI overrides:

```bash
python scripts/train.py \
  --metadata-dir /kaggle/working/metadata \
  --model mamba_t \
  --batch-size 8 \
  --lr 0.001 \
  --epochs 1
```

Checkpoints are saved as:

```text
checkpoints/mamba_t_epoch01.pt
checkpoints/mamba_t_epoch02.pt
...
```

On Kaggle configs they are written to `/kaggle/working/checkpoints/`.

## Config Fields

- `metadata_dir`: directory with the ready CSV files
- `train_csv`, `val_csv`: optional explicit CSV paths
- `image_root`: optional root for relative image paths
- `model_name`: `mamba_t` or `mamba_s`
- `batch_size`, `lr`, `weight_decay`, `epochs`
- `image_size`, `num_workers`
- `hidden_dim`, `dropout`
- `freeze_backbone`: whether to train only the two heads first
- `limit_train`, `limit_val`: optional small-subset smoke run limits
- `output_dir`: checkpoint directory

## Implemented Scope

- `ACDCWeatherDataset`, `ACDCObjectPresenceDataset`, and `ACDCDualHeadDataset` read ready CSV metadata.
- `MambaVisionDualHead` loads a shared MambaVision backbone and exposes `freeze_backbone()` / `unfreeze_backbone()`.
- `train.py` supports config + CLI overrides, logs epoch losses to stdout, and saves epoch checkpoints.
- Training code is cleaned into `src/`, `scripts/`, and `configs/` so the next stage can consume checkpoints directly.

## Left For The Metrics/Eval Owner

- Build or finish `eval.py`.
- Produce summary `.md` and `.csv` metric tables in `reports/metrics/`.
- Run dummy/checkpoint evaluation checks.
- Add plots and choose the best-looking visualizations for the report.
