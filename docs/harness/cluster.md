# Cluster Harness

Use the Gautschi / `cluster-slurm` skill for all real R reference runs.

Repo-specific target:

- `Rscript scripts/reference_smoke.R`

Submission policy:

- Gautschi only
- `standby` qos only

Helper path used in this repo:

- `python3 .agents/skills/cluster-slurm/scripts/cluster_slurm.py`

Current working profile state:

- `gautschi-cpu` and `gautschi-gpu` pass `doctor`
- Both profiles load `module use /scratch/gautschi/$USER/codex-modules`
- Both profiles load `conda-env/nlmixr2-ref-py3.12.11`
- Both profiles export `PATH=/scratch/gautschi/$USER/codex-envs/nlmixr2-ref/bin:$PATH`
- `gautschi-cpu` account: `lilly-agentic-cpu`
- `gautschi-gpu` account: `lilly-agentic-gpu`

Smoke result:

- Run id: `gautschi-ref-smoke-20260324-155625`
- Job id: `8531512`
- Status: completed
- Stdout:
  `nlmixr2 version: 5.0.0`
  `reference smoke ok`

Reference-fit results:

- Run id: `gautschi-theo-ref3-20260324-160922`
  Job id: `8531604`
  Status: completed
  Produced artifact: `reference-theophylline-fit.json`
  Downloaded fixture: `tests/fixtures/reference-theophylline-fit.json`
- Run id: `gautschi-warfarin-foce-ref2-20260324-203630`
  Job id: `8532818`
  Status: completed
  Produced artifact: `reference-warfarin-foce-fit.json`
  Downloaded fixture: `tests/fixtures/reference-warfarin-foce-fit.json`
- Run id: `gautschi-warfarin-saem-ref2-20260324-203630`
  Job id: `8532817`
  Status: completed
  Produced artifact: `reference-warfarin-saem-fit.json`
  Downloaded fixture: `tests/fixtures/reference-warfarin-saem-fit.json`
- Run id: `gautschi-oral1comp-ref2-20260324-203630`
  Job id: `8532816`
  Status: completed
  Produced artifact: `reference-pk-oral1comp-fit.json`
  Downloaded fixture: `tests/fixtures/reference-pk-oral1comp-fit.json`

Operational note:

- The machine-wide `cluster-slurm` default profile still points to Bell, so repo workflows should pass `--profile gautschi-cpu` explicitly until that global default is changed on purpose.
