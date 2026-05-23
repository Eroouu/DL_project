# ACDC Dual-head Classification Project

Image-level classification project on ACDC adverse-weather driving scenes.

## Current Evaluation Architecture

Vasilij owns the evaluation pipeline:

- `src/metrics.py` - pure metric functions for weather and object heads.
- `src/eval.py` - command-line evaluation entry point.
- `src/dataset_builder.py` - ACDC metadata builder and reusable dataset helpers.
- `configs/classmap.json` - fixed weather/object mapping and CSV label columns.
- `reports/evaluation_plan.md` - evaluation plan and required inputs from teammates.
- `reports/data.md` - data layout, split logic, CSV contract, and metadata build command.
- `reports/metrics/` - generated metric JSON files.

## Data Metadata

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

If the files are already unpacked, rebuild only standardized metadata from the unpacked ACDC dataset:

```bash
uv run python -m src.dataset_builder \
  --data-root data/acdc \
  --classmap configs/classmap.json \
  --out-dir metadata \
  --prefix metadata
```

The CSV contract is fixed in `reports/data.md`. Core columns are `image_path`, `weather_label`, `weather_id`, `official_split`, `split`, `sequence`, `frame`, `mask_path`, and object columns such as `has_road`, `has_car`, `has_bicycle`.

## Dummy Evaluation Check

Run this before real data/model are ready:

```bash
uv run python src/eval.py --dummy --tune-thresholds --out reports/metrics/dummy_eval.json
```

Expected output:

```text
Saved evaluation metrics to reports/metrics/dummy_eval.json
```

## Expected Real Inputs

From the data pipeline:

- `metadata_val.csv`
- final `configs/classmap.json`
- one fixed `has_*` object-label column per object class

From the model pipeline:

- prediction `.npz` file with either:
  - `weather_logits`, `object_logits`
  - or `weather_prob`, `object_prob`

The expected shapes are:

```text
weather_logits: N x 4
object_logits: N x C
```
