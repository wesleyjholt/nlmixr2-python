# Phase 0 Bootstrap

Status: in progress

## Goal

Turn an empty instruction-only repository into a runnable Python package with documented continuity.

## Tasks

- [x] Create `pyproject.toml`, package layout, tests, and docs hubs
- [x] Define a narrow first public API slice with TDD
- [x] Add a Gautschi smoke script for real `nlmixr2`
- [x] Get Gautschi execution working through `cluster-slurm`
- [x] Validate the uploaded `reference_smoke.R` on Gautschi `standby`
- [x] Add first parity-backed behavior beyond the mock fit summary
- [x] Stress-test the implemented Python slice and harden input validation / save semantics
- [x] Run the full test suite and record pass/fail
- [ ] Expand parity-backed coverage beyond the single `theo_sd` FOCEi artifact

## Dependencies

- Native estimator parity now depends on producing and consuming richer Gautschi artifacts.
