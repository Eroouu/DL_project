# Evaluation Plan

## Owner

Vasilij: evaluation pipeline, metrics, threshold tuning, result tables, and figures.

## Weather Head

Task: single-label classification.

Current classes: `fog`, `night`, `rain`, `snow`.

Metrics:
- accuracy
- macro F1
- per-class F1
- confusion matrix

## Object Head

Task: multi-label image-level object presence classification.

The object class list and `has_*` label columns are read from `configs/classmap.json`.

Metrics:
- mAP
- Hamming loss
- macro F1
- per-class precision / recall / F1
- per-class average precision

## Threshold Tuning

Default threshold is `0.5`.

For final validation results, tune one threshold per object class by maximizing F1 on the validation split.

## Per-weather Breakdown

Object metrics should also be reported separately for:
- fog
- night
- rain
- snow

## Expected Inputs From Teammates

From data pipeline:
- `metadata_val.csv`
- final `configs/classmap.json`
- fixed `has_*` object label columns in the same order as `configs/classmap.json`

From model pipeline:
- checkpoint, or prediction file
- weather logits/probabilities with shape `N x 4`
- object logits/probabilities with shape `N x C`
