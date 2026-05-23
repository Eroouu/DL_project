# ACDC Data Contract

## Dataset Layout

The metadata builder expects the unpacked ACDC folders under one root:

```text
<data-root>/
  rgb_anon_trainvaltest/
    rgb_anon/
      fog|night|rain|snow/
        train|val|test/
          <sequence>/*_rgb_anon.png
  gt_trainval/
    gt/
      fog|night|rain|snow/
        train|val/
          <sequence>/*_gt_labelTrainIds.png
```

RGB files are matched with `gt_labelTrainIds` masks by `(weather_label, official_split, sequence, frame)`.

## Splits

ACDC has official `train`, `val`, and `test` folders.

- Official `train` is split into internal `train` and `val` with stratification by `weather_id`.
- Official `val` is kept as internal `test` for final local evaluation.
- Official `test` is not used for supervised training or metric calculation because labels/masks are not available.

## Class Mapping

The canonical mapping lives in `configs/classmap.json`.

Weather labels are fixed as:

| weather_label | weather_id |
|---|---:|
| fog | 0 |
| night | 1 |
| rain | 2 |
| snow | 3 |

Object presence labels are generated from ACDC train ids `0..18`. Column names are `has_<class>`, for example `has_road`, `has_sidewalk`, `has_car`, and `has_bicycle`.

## CSV Format

Generated CSV files use these common columns:

| Column | Meaning |
|---|---|
| `image_path` | Absolute or dataset-local path to the RGB image. |
| `weather_label` | One of `fog`, `night`, `rain`, `snow`. |
| `weather_id` | Integer weather id from `configs/classmap.json`. |
| `official_split` | Original ACDC split: `train` or `val`. |
| `sequence` | ACDC video/drive sequence folder. |
| `frame` | Frame id parsed from the file name. |
| `split` | Internal split: `train`, `val`, or `test`. |
| `mask_path` | Path to `gt_labelTrainIds` mask, present unless `--weather-only` is used. |
| `has_*` | Binary object-presence labels derived from mask train ids. |

The full dual-head target is:

- weather head: `weather_id`;
- object head: all `has_*` columns in the order from `configs/classmap.json`.

## One-step ZIP Preparation

After downloading ACDC, place these archives in the project root:

```text
gt_trainval.zip
rgb_anon_trainvaltest.zip
```

Prepare the project data layout and metadata CSV files with:

```bash
uv run python -m src.prepare_acdc \
  --gt-zip gt_trainval.zip \
  --rgb-zip rgb_anon_trainvaltest.zip \
  --data-root data/acdc \
  --metadata-dir metadata \
  --prefix metadata
```

The script extracts only the files used by this project:

- RGB `train` and `val` images;
- `gt_labelTrainIds` masks for `train` and `val`;
- no official RGB `test` images.

For a quick notebook/debug metadata sample:

```bash
uv run python -m src.prepare_acdc \
  --quick-limit-per-split 32 \
  --metadata-dir metadata_quick
```

## Metadata-only Build Command

If the data is already unpacked under `data/acdc`, rebuild all metadata CSV files with:

```bash
uv run python -m src.dataset_builder \
  --data-root data/acdc \
  --classmap configs/classmap.json \
  --out-dir metadata \
  --prefix metadata
```

This writes:

```text
metadata/metadata_train.csv
metadata/metadata_val.csv
metadata/metadata_test.csv
metadata/metadata_all.csv
```

For a quick notebook/debug run:

```bash
uv run python -m src.dataset_builder \
  --data-root data/acdc \
  --out-dir metadata_quick \
  --quick-limit-per-split 32
```
