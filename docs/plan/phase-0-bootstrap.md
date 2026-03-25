# Phase 0 Bootstrap

Status: mostly complete (SAEM and expanded parity artifacts remain)

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
- [x] Implement model DSL parser with ODE, algebraic assignment, and residual error support (`parser.py`)
- [x] Implement diffrax-based ODE solver with PK dosing events (`ode.py`)
- [x] Implement omega block-diagonal covariance matrix construction and eta sampling (`omega.py`)
- [x] Implement NONMEM-style event table builder (`event_table.py`)
- [x] Implement native FOCE estimator with JAX autodiff (`estimators.py`)
- [x] Wire FOCE into `nlmixr2()` dispatch via `est="foce"` (`api.py`)
- [ ] Expand parity-backed coverage beyond the single `theo_sd` FOCEi artifact
- [ ] Implement SAEM estimator (test stubs exist in `test_saem.py`)

## Dependencies

- Broader parity validation depends on producing and consuming richer Gautschi artifacts.
- SAEM implementation can build on the existing omega/parser/ODE infrastructure.
