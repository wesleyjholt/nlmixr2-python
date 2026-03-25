# Parity

Current parity contract:

- [nlmixr2 package surface](parity/nlmixr2-package.md)
- [reference fit artifact](parity/reference-fit-artifact.md)
- [coverage matrix](parity/coverage-matrix.md)

Current implementation status:

- Native FOCE estimator is implemented and wired into `nlmixr2(est="foce")`
- Model DSL parser handles ODE definitions, algebraic assignments, and residual error models
- ODE solver (diffrax) supports bolus and zero-order infusion dosing events
- Omega utilities support block-diagonal covariance, Cholesky factorisation, and eta sampling
- Event table builder provides NONMEM-style dosing/sampling/repeat semantics
- SAEM and other estimators are not yet implemented

Near-term rule:

- Match exported upstream behavior where feasible.
- Keep Python semantics explicit when exact R behavior is not yet implemented.
- Validate native estimators against Gautschi reference artifacts.
- Prioritize SAEM implementation and broader parity artifact coverage.
