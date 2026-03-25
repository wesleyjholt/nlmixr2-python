# Coverage Matrix

Status: active parity leaf

Date: 2026-03-24 (updated)

Purpose: summarize what upstream `nlmixr2` functionality is present in this Python port, what is only partially covered, and what is still missing.

Primary upstream evidence:

- Official package reference index: <https://nlmixr2.org/reference/index.html>
- Core `nlmixr2()` reference: <https://nlmixr2.org/reference/nlmixr2.html>
- Upstream `NAMESPACE`: <https://raw.githubusercontent.com/nlmixr2/nlmixr2/master/NAMESPACE>
- Representative upstream helper/control pages:
  - `foceiControl()`: <https://nlmixr2.org/reference/foceiControl.html>
  - `saemControl()`: <https://nlmixr2.org/reference/saemControl.html>
  - `tableControl()`: <https://nlmixr2.org/reference/tableControl.html>
  - `setOfv()`: <https://nlmixr2.org/reference/setOfv.html>
  - `bootstrapFit()`: <https://nlmixr2.org/reference/bootstrapFit.html>
  - `traceplot()`: <https://nlmixr2.org/reference/traceplot.html>
  - `nlmixr2CheckInstall()`: <https://nlmixr2.org/reference/nlmixr2CheckInstall.html>
  - Censoring article: <https://nlmixr2.org/articles/censoring.html>

## Overall assessment

The Python port now covers the package-surface bootstrap plus a native FOCE estimator, a model DSL parser with ODE and residual error support, omega (random-effects) matrix utilities, and NONMEM-style event table construction. It is no longer limited to mock/reference artifact paths.

Practical summary:

- Package-entry helper names: present
- Model DSL: functional for ODE models, algebraic assignments, and residual error (additive/proportional/combined)
- Native estimation: FOCE implemented (JAX/Adam-based); SAEM absent
- Random-effects specification: omega block-diagonal matrices with Cholesky factorisation and eta sampling
- ODE solving: diffrax-based Dormand-Prince integrator with PK dosing events (bolus and infusion)
- Event tables: NONMEM-style event dataset builder with dosing, sampling, and repeat semantics
- Fit-object semantics: basic structure present (estimator, objective, parameters, model, control, table)
- Post-fit utilities and controls: absent
- Upstream ecosystem integration: absent or only represented as metadata

## Matrix

| Area | Upstream `nlmixr2` capability | Python status | Notes |
| --- | --- | --- | --- |
| Package entry points | `nlmixr2`, `nlmixr2CheckInstall`, `nlmixr2conflicts`, `nlmixr2deps`, `nlmixr2packages`, `nlmixr2update` exported from the R package namespace | Present | Names are present with Python-adapted semantics |
| Model entry/inspection | `nlmixr2(object)` can parse/check models before fitting and returns a uniform model object | Present | Python can build, parse, and return a structured model object; parser handles ODE, assignment, and error statements |
| `ini` block semantics | Numeric initials, bounds, reserved-name restrictions, variance-scale conventions | Present | Python supports scalars, `c(est)`, `c(lower, est)`, `c(lower, est, upper)`-like tuples and reserved-prefix validation |
| Random-effects specification | `eta` / omega syntax using `~`, including correlated blocks | Present | `omega.py` provides named block-diagonal covariance matrices with off-diagonal (correlation) support, Cholesky factorisation, and multivariate normal eta sampling via `sample_etas()` |
| Residual-error specification | `add()`, `prop()`, combined residual error, `norm()` | Partial | Parser recognises `add()`, `prop()`, and `add() + prop()` (combined) error models; `norm()` is not yet supported |
| ODE model DSL | RxODE-style differential equations inside the model block | Present | Parser handles `d/dt(state) = expr` statements and algebraic assignments; ODE solver uses diffrax Dormand-Prince (Dopri5) with adaptive stepping |
| ODE solver / PK dosing | ODE integration with bolus doses and infusions | Present | `ode.py` provides piecewise integration with bolus events and zero-order infusion rate functions via diffrax |
| Solved PK systems | `linCmt()` solved systems for 1/2/3 compartment models with absorption/lag support | Missing | No solved-system support |
| Covariates / mu-referencing | Upstream model conventions for covariates and SAEM-friendly parameterization | Missing | No covariate semantics beyond raw input columns |
| Dataset semantics | Upstream nlmixr/rxode2-compatible event datasets | Present | `event_table.py` provides NONMEM-style event table construction (`et()`) with dosing, sampling, repeat, and export to dict/JAX arrays; `api.py` validates columnar and record data |
| Censoring support | `DV`, `CENS`, `LIMIT` conventions and censored residual handling | Missing | No censoring-aware likelihood or output |
| Estimator dispatch | Upstream `nlmixr2()` dispatches to estimation backends and methods listed by `nlmixr2AllEst()` | Partial | Python supports `est="mock"`, `est="reference"`, and `est="foce"`; other estimators raise `NotImplementedError` |
| FOCEi estimation | Native FOCEi fitting and its controls | Present | `estimators.py` implements FOCE with JAX autodiff, Adam optimiser, inner/outer alternation; controls: maxiter, lr, lr_eta, tol, inner_steps, sigma |
| SAEM estimation | Native SAEM fitting and its controls | Missing | Explicitly absent; test stubs exist in `test_saem.py` |
| NLME / NLM / other estimators | Upstream estimator/control support visible in docs and helper pages | Missing | No native Python implementation |
| Estimation controls | `foceiControl()`, `saemControl()`, `nlmControl()`, `nlmixr2NlmeControl()`, `tableControl()` | Partial | Python accepts `control` dicts for FOCE (maxiter, lr, tol, etc.) but does not implement the formal control constructors or full upstream semantics |
| Fit object richness | Upstream fit carries objective info, covariance/correlation, theta/eta outputs, timing, per-subject information, data-frame-like behavior | Partial | Python fit object contains estimator, n_observations, columns, parameter_count, objective, model, control, and table; FOCE fits also include fixed_params, n_iterations, converged, and etas in the table |
| Objective-function handling | Upstream `setOfv()` and related AIC/BIC/objective type behavior | Missing | No equivalent |
| Post-fit utilities | `bootstrapFit()`, `traceplot()`, fit-print/inspection helpers | Missing | No equivalent |
| Ecosystem package checks | Upstream install checks report core and optional R packages | Present | Python reports Python package versions and Gautschi reference-runtime version |
| Persistence format | Upstream can save/load fit objects in R-native forms | Present | Python can save/load JSON fit summaries with round-trip semantics |
| Reference parity artifacts | Compare against real upstream outputs | Partial | One tested `theo_sd` FOCEi JSON artifact is supported; coverage is currently a single model/fit |
| Omega / between-subject variability | `lotri`-style block-diagonal covariance specification | Present | `omega.py` provides `OmegaBlock`, `omega()` constructor, `block_diagonal()`, `cholesky_factor()`, and `sample_etas()` |
| Event table construction | `rxode2::et()` for dosing regimens and observation times | Present | `event_table.py` provides `EventTable` class and `et()` factory with dosing, sampling, repeat, and NONMEM column output |
| Model parsing | Internal model compilation from DSL to executable functions | Present | `parser.py` converts model block statements into JAX-callable ODE and output functions with math namespace (exp, log, sqrt, etc.) |

## What Is Actually Covered Today

Covered with tests in this repo:

- top-level helper names (nlmixr2, nlmixr2CheckInstall, nlmixr2conflicts, nlmixr2deps, nlmixr2packages, nlmixr2update)
- `ini()` block construction with bounds, validation, and reserved-prefix checks
- `model()` block construction from statement lists or mappings
- `nlmixr2()` returning:
  - a Python model object (no data)
  - a mock fit summary from supplied data
  - a reference-backed fit loaded from a Gautschi-produced FOCEi artifact
  - a native FOCE fit from supplied data with JAX-based estimation
- `parse_model()` handling ODE d/dt statements, algebraic assignments, and residual error specifications
- `solve_ode()` with diffrax integration, bolus dosing, and zero-order infusions
- `omega()` block-diagonal matrix construction with off-diagonal covariance, positive-definiteness validation
- `block_diagonal()` combining multiple omega blocks
- `cholesky_factor()` and `sample_etas()` for random-effects sampling
- `EventTable` / `et()` with dosing, sampling, repeat, and export to dict/arrays
- `estimate_foce()` with inner/outer optimisation, convergence detection, and control parameters
- stress tests for edge cases, save round-trips, and repeated artifact loads

This means the current package is best described as:

- a functional FOCE estimator with JAX autodiff
- a model DSL parser for ODE-based PK/PD models
- an omega (random-effects) specification and sampling system
- a NONMEM-style event table builder
- a parity harness and reference-artifact consumer

It is progressing toward a usable replacement for upstream `nlmixr2` model fitting, with FOCE as the first working estimator.

## Highest-Priority Missing Work

If the goal is substantive parity rather than namespace parity, the biggest missing items are:

1. SAEM estimator (test stubs exist)
2. Solved PK systems (linCmt) for 1/2/3 compartment models
3. Proper formal control objects (foceiControl, saemControl constructors)
4. Richer fit-object structure (covariance matrix, standard errors, AIC/BIC, per-subject info)
5. Post-fit utilities (bootstrapFit, traceplot)
6. Censoring support (DV/CENS/LIMIT)
7. Covariate and mu-referencing semantics
8. More Gautschi parity artifacts across multiple models and estimators

## Recommended sequencing

Recommended next implementation order:

1. Expand reference-artifact coverage so native work has more parity targets
2. Implement SAEM estimator with the existing model/omega infrastructure
3. Add solved PK systems (linCmt) to complement ODE-based models
4. Fill in formal control objects and richer fit outputs
5. Add post-fit utilities after estimator coverage is broader

## Confidence notes

- The Python port now has a functioning native FOCE estimator, which is a significant advance beyond pure package-surface parity.
- The model parser, ODE solver, omega utilities, and event table modules are tested and functional but have not been validated against a broad range of upstream reference outputs.
- SAEM remains explicitly absent. Test stubs in `test_saem.py` indicate it is a planned next target.
