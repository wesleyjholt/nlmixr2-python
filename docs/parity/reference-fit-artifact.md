# Reference Fit Artifact

Status: active parity leaf

## Purpose

Use Gautschi-backed `nlmixr2` runs to produce stable JSON artifacts that the Python package can load with `nlmixr2(..., est="reference", reference_fit_path=...)`.

## Current artifact

- Fixture path: `tests/fixtures/reference-theophylline-fit.json`
- Source script: `scripts/reference_theophylline_fit.R`
- Latest successful run: `gautschi-theo-ref3-20260324-160922`
- Job id: `8531604`
- Estimator: `focei`
- Dataset: `theo_sd`

## Current Python behavior

- `est="reference"` requires `reference_fit_path`
- The artifact populates `NLMIXRFit.estimator`, `n_observations`, `columns`, `objective`
- Reference parameters and run metadata are stored in `fit.table`

## Limitations

- The Python package does not yet reproduce FOCEi numerically.
- This is parity by ingestion of validated upstream output, not by reimplementation of the optimizer.
