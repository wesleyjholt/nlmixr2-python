# nlmixr2 Package Surface

Status: active parity leaf

## Upstream target

The first parity target is the public surface exported by the upstream R `nlmixr2` package, inferred from:

- Official docs for `nlmixr2()`: <https://nlmixr2.org/reference/nlmixr2.html>
- Upstream `NAMESPACE`: <https://raw.githubusercontent.com/nlmixr2/nlmixr2/master/NAMESPACE>

## Exported functions tracked here

- `nlmixr2`
- `nlmixr2CheckInstall`
- `nlmixr2conflicts`
- `nlmixr2deps`
- `nlmixr2packages`
- `nlmixr2update`

Also included in the Python package for model construction and estimation:

- `ini`
- `model`
- `parse_model` / `ParsedModel`
- `estimate_foce` / `EstimationResult`
- `solve_ode`
- `omega` / `OmegaBlock` / `sample_etas`
- `et` / `EventTable`

## Current Python status

- `nlmixr2()` currently returns a structured model spec, a JAX-backed mock fit summary, a parity-backed fit loaded from a Gautschi-generated reference artifact via `est="reference"`, or a native FOCE fit via `est="foce"`.
- FOCE estimation is implemented natively using JAX autodiff with an Adam optimiser and inner/outer alternation.
- The model parser (`parser.py`) handles ODE definitions (`d/dt(state) = expr`), algebraic assignments, and residual error specifications (`add()`, `prop()`, combined).
- The ODE solver (`ode.py`) uses diffrax Dormand-Prince (Dopri5) with adaptive stepping and supports bolus and zero-order infusion dosing events.
- Omega utilities (`omega.py`) provide block-diagonal covariance matrix construction, Cholesky factorisation, and multivariate normal eta sampling.
- Event table construction (`event_table.py`) provides a NONMEM-style dataset builder with dosing, sampling, and repeat semantics.
- Install and conflict helpers are implemented as Python environment helpers, not R package managers.
- SAEM and other estimators are not yet implemented natively (test stubs exist for SAEM).

## Notes

- `.nlmixr2attach` appears in the R `NAMESPACE` but is treated as internal because of its leading dot.
- The inclusion of `ini` and `model` is based on the official modeling documentation, even though they are not exported from the top-level `NAMESPACE`. This is an explicit parity inference, not a direct export match.
- The current parity-backed estimator path is artifact ingestion, not in-process estimation. See [reference fit artifact](reference-fit-artifact.md).
