# Harness

Status: bootstrap harness for a Python `nlmixr2` package with a tested first API slice.

## Local commands

- Install editable package: `python3 -m pip install -e .[dev]`
- Run the full test suite: `python3 -m pytest`
- Run one file: `python3 -m pytest tests/test_api.py`

## Repo layout

- `src/nlmixr2/`: Python package under active development
- `tests/`: unit tests; every exported function needs several tests
- `docs/`: state, plan, parity, decisions, and session log
- `scripts/reference_smoke.R`: first Gautschi smoke script for real `nlmixr2`
- `scripts/reference_theophylline_fit.R`: Gautschi FOCEi reference artifact producer

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

## Next harness upgrade

- Expand parity tests against Gautschi-generated artifacts.
- Add a repo-local wrapper for the low-level smoke submission flow.
