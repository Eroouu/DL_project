# ACDC Dual-head Classification Project

Image-level classification project on ACDC adverse-weather driving scenes.

## Current Evaluation Architecture

Vasilij owns the evaluation pipeline:

- `src/metrics.py` - pure metric functions for weather and object heads.
- `src/eval.py` - command-line evaluation entry point.
- `configs/class_map.example.json` - temporary class list until the final data mapping is fixed.
- `reports/evaluation_plan.md` - evaluation plan and required inputs from teammates.
- `reports/metrics/` - generated metric JSON files.

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
- final `class_map.json`
- one fixed object-label column per object class

From the model pipeline:

- prediction `.npz` file with either:
  - `weather_logits`, `object_logits`
  - or `weather_prob`, `object_prob`

The expected shapes are:

```text
weather_logits: N x 4
object_logits: N x C
```
