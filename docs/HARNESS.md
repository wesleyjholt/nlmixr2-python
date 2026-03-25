# Harness

Status: Python `nlmixr2` package with native estimators plus Gautschi-backed reference-artifact coverage for `theo_sd`, `warfarin`, and `Oral_1CPT`.

## Local commands

- Install editable package: `python3 -m pip install -e .[dev]`
- Run the full test suite: `python3 -m pytest`
- Run one file: `python3 -m pytest tests/test_api.py`

## Repo layout

- `src/nlmixr2/`: Python package under active development
  - `api.py`: core `nlmixr2()` entry point, `ini()`, `model()`, data coercion, mock/reference/FOCE dispatch
  - `estimators.py`: native FOCE estimator with JAX autodiff (Adam optimiser, inner/outer alternation)
  - `parser.py`: model DSL parser (ODE `d/dt` statements, algebraic assignments, residual error `~` specs)
  - `ode.py`: diffrax-based ODE solver with PK dosing events (bolus and zero-order infusion)
  - `omega.py`: omega block-diagonal covariance matrix construction, Cholesky factorisation, eta sampling
  - `event_table.py`: NONMEM-style event table builder with dosing, sampling, and repeat semantics
  - `install.py`: environment/install helpers (nlmixr2CheckInstall, nlmixr2packages, nlmixr2deps, etc.)
- `tests/`: unit tests; every exported function needs several tests
  - `test_api.py`, `test_api_foce.py`, `test_control.py`, `test_estimators.py`, `test_event_table.py`, `test_install.py`, `test_ode.py`, `test_omega.py`, `test_parser.py`, `test_saem.py`, `test_stress.py`
- `docs/`: state, plan, parity, decisions, and session log
- `scripts/reference_smoke.R`: first Gautschi smoke script for real `nlmixr2`
- `scripts/reference_theophylline_fit.R`: Gautschi FOCEi reference artifact producer for `theo_sd`
- `scripts/reference_fit_helpers.R`: shared JSON artifact extractor for Gautschi reference fits
- `scripts/reference_warfarin_foce_fit.R`: Gautschi FOCEi artifact producer for `warfarin`
- `scripts/reference_warfarin_saem_fit.R`: Gautschi SAEM artifact producer for `warfarin`
- `scripts/reference_pk_oral1comp_fit.R`: Gautschi FOCEi artifact producer for the single-dose `Oral_1CPT` slice used as `pk.oral1comp`

## Gautschi reference runs

Use the Gautschi / `cluster-slurm` skill for all real R-side `nlmixr2` runs. Do not duplicate module or environment setup here. Always submit to `standby`.

Active helper path in this repo:

- `python3 .agents/skills/cluster-slurm/scripts/cluster_slurm.py`

Active Gautschi profile notes:

- Use `--profile gautschi-cpu` explicitly. The global `cluster-slurm` default profile on this machine still points to Bell.
- The Gautschi CPU and GPU profiles now load `/scratch/gautschi/$USER/codex-envs/nlmixr2-ref/bin` through profile-managed setup commands.
- The reference env contains `Rscript` and `nlmixr2 5.0.0`.

Current repo-local smoke target:

- `Rscript scripts/reference_smoke.R`
- `Rscript reference_theophylline_fit.R`
- `Rscript reference_warfarin_foce_fit.R`
- `Rscript reference_warfarin_saem_fit.R`
- `Rscript reference_pk_oral1comp_fit.R`

Validated smoke sequence on 2026-03-24:

- `init-run --profile gautschi-cpu --prefix gautschi-ref-smoke`
- `upload --run-id <RUN_ID> --local-path scripts/reference_smoke.R`
- `render-job --run-id <RUN_ID> --command 'Rscript reference_smoke.R' --header qos=standby`
- `submit-job --run-id <RUN_ID> --script-name job.slurm`
- `status --run-id <RUN_ID>`
- `logs --run-id <RUN_ID> --tail 200`

Validated parity sequence on 2026-03-24:

- `init-run --profile gautschi-cpu --prefix gautschi-theo-ref3`
- `upload --run-id <RUN_ID> --local-path scripts/reference_theophylline_fit.R`
- `render-job --run-id <RUN_ID> --command 'Rscript reference_theophylline_fit.R' --header qos=standby`
- `submit-job --run-id <RUN_ID> --script-name job.slurm`
- `download --run-id <RUN_ID> --remote-path reference-theophylline-fit.json --local-path tests/fixtures/reference-theophylline-fit.json`

Additional validated artifact runs on 2026-03-24:

- `gautschi-warfarin-foce-ref2-20260324-203630` / job `8532818` -> `tests/fixtures/reference-warfarin-foce-fit.json`
- `gautschi-warfarin-saem-ref2-20260324-203630` / job `8532817` -> `tests/fixtures/reference-warfarin-saem-fit.json`
- `gautschi-oral1comp-ref2-20260324-203630` / job `8532816` -> `tests/fixtures/reference-pk-oral1comp-fit.json`

## Next harness upgrade

- Expand parity tests against Gautschi-generated artifacts (multiple models and estimators).
- Validate native FOCE output against the reference theophylline artifact for numerical equivalence.
- Improve the `warfarin` FOCE artifact so it can support stricter numerical parity checks.
- Add a repo-local wrapper for the low-level smoke submission flow.
