# State

- Phase: bootstrap / first API slice
- Branch: `master` (new local git repo)
- Package status: Python package has stress-tested mock and reference-artifact `nlmixr2()` paths, stricter data validation, stable save round-trips, and helper-function parity improvements
- Parity status: top-level package surface is bootstrapped; `est="reference"` now ingests a real Gautschi FOCEi artifact from `theo_sd`
- Gap analysis: coverage matrix added for upstream-vs-Python feature parity; current status is package-surface bootstrap, not estimator parity
- Cluster status: Gautschi CPU/GPU profiles pass `doctor`; profile-managed env `/scratch/gautschi/$USER/codex-envs/nlmixr2-ref` has `Rscript` and `nlmixr2 5.0.0`; smoke and FOCEi artifact jobs completed on `standby`
- Blockers: no native Python FOCEi/SAEM implementation yet; global `cluster-slurm` default profile still points to Bell, so Gautschi jobs must stay explicit for now
- Latest full test run: `python3 -m pytest` -> 35 passed on 2026-03-24
- Next action: use the coverage matrix to drive richer Gautschi reference artifacts beyond one model/fit, then replace the mock estimator path with native implementations
