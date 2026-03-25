# State

- Phase: parity hardening / validation
- Branch: `main`
- Package status: the implementation checklist in `TODO.md` is fully checked off, including the Gautschi-backed reference artifact task for `warfarin` FOCE, `warfarin` SAEM, and `Oral_1CPT`
- Reference artifacts available locally:
  - `tests/fixtures/reference-theophylline-fit.json`
  - `tests/fixtures/reference-warfarin-foce-fit.json`
  - `tests/fixtures/reference-warfarin-saem-fit.json`
  - `tests/fixtures/reference-pk-oral1comp-fit.json`
- Verification: `python3 -m pytest` passes locally (`913 passed` on 2026-03-24)
- Cluster status: Gautschi CPU/GPU profiles pass `doctor`; profile-managed env `/scratch/gautschi/$USER/codex-envs/nlmixr2-ref` has `Rscript` and `nlmixr2 5.0.0`; all reference jobs in this repo still require explicit `--profile gautschi-cpu` because the global `cluster-slurm` default profile points to Bell
- Current risk: the `warfarin` FOCE artifact is reproducible but lands in a poor local optimum, so it is suitable for loader/regression coverage now but should be tuned before using it as a strict numeric parity target
- Next action: promote the new Gautschi artifacts into deeper native-vs-reference comparison tests and improve the `warfarin` FOCE reference fit quality if that artifact becomes part of tighter numerical assertions
