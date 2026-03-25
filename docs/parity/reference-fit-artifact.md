# Reference Fit Artifact

Status: active parity leaf

## Purpose

Use Gautschi-backed `nlmixr2` runs to produce stable JSON artifacts that the Python package can load with `nlmixr2(..., est="reference", reference_fit_path=...)`.

## Current artifacts

- `tests/fixtures/reference-theophylline-fit.json`
  - Source script: `scripts/reference_theophylline_fit.R`
  - Run / job: `gautschi-theo-ref3-20260324-160922` / `8531604`
  - Estimator / dataset: `focei` / `theo_sd`
- `tests/fixtures/reference-warfarin-foce-fit.json`
  - Source script: `scripts/reference_warfarin_foce_fit.R`
  - Run / job: `gautschi-warfarin-foce-ref2-20260324-203630` / `8532818`
  - Estimator / dataset: `focei` / `warfarin`
- `tests/fixtures/reference-warfarin-saem-fit.json`
  - Source script: `scripts/reference_warfarin_saem_fit.R`
  - Run / job: `gautschi-warfarin-saem-ref2-20260324-203630` / `8532817`
  - Estimator / dataset: `saem` / `warfarin`
- `tests/fixtures/reference-pk-oral1comp-fit.json`
  - Source script: `scripts/reference_pk_oral1comp_fit.R`
  - Run / job: `gautschi-oral1comp-ref2-20260324-203630` / `8532816`
  - Estimator / dataset: `focei` / `Oral_1CPT`

## Current Python behavior

- `est="reference"` requires `reference_fit_path`
- The artifact populates `NLMIXRFit.estimator`, `n_observations`, `columns`, `objective`
- Reference parameters and run metadata are stored in `fit.table`
- `tests/test_api.py` now loads all checked-in Gautschi fixtures as regression coverage

## Limitations

- Native FOCE estimation is now implemented in `estimators.py` via `est="foce"`, but the reference artifact path (`est="reference"`) remains the primary mechanism for validating against upstream R outputs.
- The native FOCE implementation has not yet been validated for numerical equivalence against the upstream R FOCEi/SAEM results in these artifacts.
- The `warfarin` FOCE artifact currently converges to a poor local optimum; treat it as loader coverage unless and until the reference fit is tuned.
- The reference artifact path serves as a parity target for validating the native estimator once numerical alignment work is done.
