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

Also included in the Python package for model construction:

- `ini`
- `model`

## Current Python status

- `nlmixr2()` currently returns a structured model spec, a JAX-backed mock fit summary, or a parity-backed fit loaded from a Gautschi-generated reference artifact via `est="reference"`.
- Install and conflict helpers are implemented as Python environment helpers, not R package managers.
- Real estimators such as FOCEi or SAEM are not implemented natively yet.

## Notes

- `.nlmixr2attach` appears in the R `NAMESPACE` but is treated as internal because of its leading dot.
- The inclusion of `ini` and `model` is based on the official modeling documentation, even though they are not exported from the top-level `NAMESPACE`. This is an explicit parity inference, not a direct export match.
- The current parity-backed estimator path is artifact ingestion, not in-process estimation. See [reference fit artifact](reference-fit-artifact.md).
