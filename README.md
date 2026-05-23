# ACDC Dual-Head Classification Project

Image-level classification project on ACDC adverse-weather driving scenes.

The repository now contains three script-based parts:

- data preparation and standardized metadata generation;
- dual-head MambaVision training;
- evaluation metrics for weather and object heads.

## Environment

With `uv`:

```bash
uv sync
```

On Egor's local machine, the existing Conda environment can also install the training requirements:

```powershell
C:\Users\egor0\anaconda3\envs\diplom314\python.exe -m pip install -r requirements.txt
```

## Project Structure

```text
configs/
  classmap.json                fixed weather/object mapping and CSV label columns
  mamba_t_smoke.yaml           short MambaVision-T run on a small subset
  mamba_s_smoke.yaml           short MambaVision-S architecture check
  mamba_t_full.yaml            first longer MambaVision-T config
reports/
  data.md                      data layout, split logic, CSV contract, build command
  evaluation_plan.md           evaluation plan and expected inputs
scripts/
  train.py                     training entrypoint
src/
  data/datasets.py             CSV-backed ACDC datasets and transforms
  dataset_builder.py           ACDC metadata builder and reusable dataset helpers
  eval.py                      evaluation entrypoint
  metrics.py                   pure metric functions
  models/mamba_dual_head.py    MambaVision backbone + weather/object heads
  prepare_acdc.py              one-step ZIP extraction + metadata build
  utils/                       config and checkpoint helpers
```

## Data Preparation

Put the downloaded ACDC archives into the project root:

```text
gt_trainval.zip
rgb_anon_trainvaltest.zip
```

Then prepare the project dataset layout and metadata in one step:

```bash
uv run python -m src.prepare_acdc \
  --gt-zip gt_trainval.zip \
  --rgb-zip rgb_anon_trainvaltest.zip \
  --data-root data/acdc \
  --metadata-dir metadata \
  --prefix metadata
```

The script extracts only labeled `train`/`val` RGB images and `gt_labelTrainIds` masks, then creates:

```text
data/acdc/gt_trainval/gt/...
data/acdc/rgb_anon_trainvaltest/rgb_anon/...
metadata/metadata_train.csv
metadata/metadata_val.csv
metadata/metadata_test.csv
metadata/metadata_all.csv
```

For quick notebook smoke runs:

```bash
uv run python -m src.prepare_acdc \
  --quick-limit-per-split 32 \
  --metadata-dir metadata_quick
```

If the files are already unpacked, rebuild only standardized metadata:

```bash
uv run python -m src.dataset_builder \
  --data-root data/acdc \
  --classmap configs/classmap.json \
  --out-dir metadata \
  --prefix metadata
```

The CSV contract is fixed in `reports/data.md`. Core columns are `image_path`, `weather_label`, `weather_id`, `official_split`, `split`, `sequence`, `frame`, `mask_path`, and object columns such as `has_road`, `has_car`, `has_bicycle`.

## Train

Smoke run after metadata is created:

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
  --metadata-dir metadata \
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

## Evaluation

Dummy evaluation check:

```bash
uv run python src/eval.py --dummy --tune-thresholds --out reports/metrics/dummy_eval.json
```

Evaluation on prepared validation metadata with dummy predictions:

```bash
uv run python src/eval.py \
  --metadata metadata/metadata_val.csv \
  --dummy \
  --tune-thresholds \
  --out reports/metrics/metadata_val_dummy_eval.json
```

Expected prediction `.npz` keys for real model outputs:

```text
weather_logits: N x 4
object_logits: N x C
```

or:

```text
weather_prob: N x 4
object_prob: N x C
```

## Implemented Scope

- `src.prepare_acdc` extracts the ACDC ZIP files into the project layout and builds metadata.
- `configs/classmap.json` fixes weather ids and object `has_*` columns.
- `ACDCWeatherDataset`, `ACDCObjectPresenceDataset`, and `ACDCDualHeadDataset` read ready CSV metadata.
- `MambaVisionDualHead` loads a shared MambaVision backbone and exposes `freeze_backbone()` / `unfreeze_backbone()`.
- `train.py` supports config + CLI overrides, logs epoch losses to stdout, and saves epoch checkpoints.
- `eval.py` computes weather/object metrics and writes JSON, Markdown, and CSV tables.
