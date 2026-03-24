# ADR 001: Initial Scope

Date: 2026-03-24

## Decision

Bootstrap the Python port by matching the documented top-level `nlmixr2` package surface and a lightweight model-building API before attempting real estimation algorithms.

## Rationale

- The repository started empty, so continuity and harness quality were the immediate risks.
- The upstream package surface is small enough to make a tested first slice realistic in one session.
- Real estimator parity depends on Gautschi-backed R reference runs, which are not reproducible until the cluster helper path is resolved.

## Consequences

- The first `nlmixr2()` implementation is intentionally a mock fit summary, not an estimator.
- Future sessions should replace mock behavior with parity-backed algorithms instead of widening the helper API first.
