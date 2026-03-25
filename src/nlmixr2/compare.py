"""Model comparison utilities for comparing multiple pharmacometric fits."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import jax
import jax.numpy as jnp

from .api import NLMIXRFit
from .diagnostics import compute_aic, compute_bic


@dataclass(frozen=True)
class ComparisonTable:
    """Summary table comparing multiple model fits."""

    models: list[str]
    objectives: list[float]
    aics: list[float]
    bics: list[float]
    n_params: list[int]
    n_obs: list[int]
    best_aic: str
    best_bic: str


@dataclass(frozen=True)
class LRTResult:
    """Result of a likelihood ratio test."""

    statistic: float
    df: int
    p_value: float
    significant: bool


def compare_fits(
    fits: Sequence[NLMIXRFit],
    names: Sequence[str] | None = None,
) -> ComparisonTable:
    """Compare multiple NLMIXRFit objects on information criteria.

    Parameters
    ----------
    fits : list of NLMIXRFit
        The fit objects to compare.
    names : list of str, optional
        Model names. Defaults to "Model 1", "Model 2", etc.

    Returns
    -------
    ComparisonTable
    """
    if names is None:
        names = [f"Model {i + 1}" for i in range(len(fits))]

    model_names = list(names)
    objectives = [fit.objective for fit in fits]
    n_params = [fit.parameter_count for fit in fits]
    n_obs = [fit.n_observations for fit in fits]
    aics = [
        compute_aic(fit.objective, fit.parameter_count)
        for fit in fits
    ]
    bics = [
        compute_bic(fit.objective, fit.parameter_count, max(fit.n_observations, 1))
        for fit in fits
    ]

    best_aic_idx = min(range(len(aics)), key=lambda i: aics[i])
    best_bic_idx = min(range(len(bics)), key=lambda i: bics[i])

    return ComparisonTable(
        models=model_names,
        objectives=objectives,
        aics=aics,
        bics=bics,
        n_params=n_params,
        n_obs=n_obs,
        best_aic=model_names[best_aic_idx],
        best_bic=model_names[best_bic_idx],
    )


def likelihood_ratio_test(
    fit_full: NLMIXRFit,
    fit_reduced: NLMIXRFit,
    df: int,
) -> LRTResult:
    """Perform a likelihood ratio test between nested models.

    Parameters
    ----------
    fit_full : NLMIXRFit
        The full (more complex) model fit.
    fit_reduced : NLMIXRFit
        The reduced (simpler) model fit.
    df : int
        Degrees of freedom (difference in number of parameters).

    Returns
    -------
    LRTResult
    """
    if df <= 0:
        raise ValueError("degrees of freedom must be positive")

    # -2LL difference: reduced objective minus full objective
    statistic = fit_reduced.objective - fit_full.objective

    # Chi-squared survival function: P(X > statistic)
    p_value = _chi2_sf(statistic, df)

    return LRTResult(
        statistic=statistic,
        df=df,
        p_value=p_value,
        significant=p_value < 0.05,
    )


def _chi2_sf(x: float, df: int) -> float:
    """Compute chi-squared survival function P(X > x) using regularized gamma."""
    if x <= 0:
        return 1.0
    # P(X <= x) = regularized lower incomplete gamma: gammainc(df/2, x/2)
    # sf = 1 - cdf = upper incomplete gamma
    a = df / 2.0
    z = x / 2.0
    # Use the series expansion of the regularized lower incomplete gamma
    return 1.0 - _regularized_lower_gamma(a, z)


def _regularized_lower_gamma(a: float, x: float) -> float:
    """Regularized lower incomplete gamma function P(a, x) via series expansion."""
    if x < 0:
        return 0.0
    if x == 0:
        return 0.0

    # Series: P(a, x) = e^{-x} * x^a * sum_{n=0}^{inf} x^n / Gamma(a+n+1)
    # Which equals sum_{n=0}^{inf} e^{-x} * x^{a+n} / Gamma(a+n+1)
    # = gamma(a,x) / Gamma(a)
    # Use: gamma(a,x)/Gamma(a) = e^{-x} * x^a / Gamma(a) * sum x^n / (a*(a+1)*...*(a+n))
    term = 1.0 / a
    total = term
    for n in range(1, 300):
        term *= x / (a + n)
        total += term
        if abs(term) < 1e-15 * abs(total):
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def bootstrap_comparison(
    fits: Sequence[NLMIXRFit],
    data: dict[str, Any],
    n_bootstrap: int = 100,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Bootstrap confidence intervals for objective function differences.

    This computes bootstrap CIs by resampling observation indices and
    recomputing a simple objective proxy (sum of squared DV deviations)
    for each model's parameter count penalty.

    Parameters
    ----------
    fits : list of NLMIXRFit
        Model fits to compare.
    data : dict
        Dataset with at least a "dv" key.
    n_bootstrap : int
        Number of bootstrap replicates.
    seed : int
        Random seed.

    Returns
    -------
    dict mapping pair labels to dicts with ci_lower, ci_upper, mean_diff.
    """
    dv = jnp.asarray(data.get("dv", []))
    n = len(dv)
    key = jax.random.PRNGKey(seed)

    results: dict[str, dict[str, float]] = {}

    for i in range(len(fits)):
        for j in range(i + 1, len(fits)):
            diffs = []
            for b in range(n_bootstrap):
                key, subkey = jax.random.split(key)
                indices = jax.random.randint(subkey, shape=(n,), minval=0, maxval=n)
                boot_dv = dv[indices]
                centered = boot_dv - jnp.mean(boot_dv)
                base_obj = float(jnp.mean(jnp.square(centered)))
                # Penalize by parameter count difference as AIC-like proxy
                obj_i = base_obj + 2 * fits[i].parameter_count / max(n, 1)
                obj_j = base_obj + 2 * fits[j].parameter_count / max(n, 1)
                diffs.append(obj_i - obj_j)

            diffs_arr = sorted(diffs)
            lo_idx = max(0, int(0.025 * n_bootstrap))
            hi_idx = min(n_bootstrap - 1, int(0.975 * n_bootstrap))
            pair_label = f"fit{i + 1}_vs_fit{j + 1}"
            results[pair_label] = {
                "ci_lower": diffs_arr[lo_idx],
                "ci_upper": diffs_arr[hi_idx],
                "mean_diff": sum(diffs) / len(diffs),
            }

    return results


def format_comparison(table: ComparisonTable) -> str:
    """Format a ComparisonTable as a readable string.

    Parameters
    ----------
    table : ComparisonTable

    Returns
    -------
    str
    """
    # Column widths
    name_w = max(len(m) for m in table.models + ["Model"])
    lines: list[str] = []

    header = (
        f"{'Model':<{name_w}s}  "
        f"{'Objective':>12s}  "
        f"{'AIC':>12s}  "
        f"{'BIC':>12s}  "
        f"{'nPar':>5s}  "
        f"{'nObs':>6s}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for i, name in enumerate(table.models):
        row = (
            f"{name:<{name_w}s}  "
            f"{table.objectives[i]:>12.4f}  "
            f"{table.aics[i]:>12.4f}  "
            f"{table.bics[i]:>12.4f}  "
            f"{table.n_params[i]:>5d}  "
            f"{table.n_obs[i]:>6d}"
        )
        lines.append(row)

    lines.append("")
    lines.append(f"Best AIC: {table.best_aic}")
    lines.append(f"Best BIC: {table.best_bic}")

    return "\n".join(lines)
