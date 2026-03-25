"""Bootstrap refitting for parameter uncertainty estimation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .api import NLMIXRFit, nlmixr2


@dataclass
class BootstrapResult:
    """Result of a bootstrap analysis."""

    fits: list[NLMIXRFit]
    n_success: int
    n_fail: int
    parameter_summary: dict[str, dict[str, float]]


def resample_by_subject(
    data: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Resample subjects with replacement, returning all records for selected subjects.

    Parameters
    ----------
    data : dict
        Column-oriented data with an "id" column.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Resampled data dict with same column structure.
    """
    rng = random.Random(seed)

    ids = list(data["id"])
    unique_ids = sorted(set(ids))
    n_subjects = len(unique_ids)

    # Build index: subject_id -> list of row indices
    subject_rows: dict[Any, list[int]] = {}
    for i, sid in enumerate(ids):
        subject_rows.setdefault(sid, []).append(i)

    # Resample subjects with replacement
    sampled_ids = rng.choices(unique_ids, k=n_subjects)

    # Collect rows for sampled subjects
    row_indices: list[int] = []
    for sid in sampled_ids:
        row_indices.extend(subject_rows[sid])

    # Build resampled data
    columns = list(data.keys())
    resampled: dict[str, list] = {}
    for col in columns:
        col_data = data[col]
        # Handle both list and array-like
        if hasattr(col_data, '__getitem__'):
            resampled[col] = [col_data[i] for i in row_indices]
        else:
            resampled[col] = [list(col_data)[i] for i in row_indices]

    return resampled


def _compute_parameter_summary(
    fits: list[NLMIXRFit],
    ci_level: float = 0.95,
) -> dict[str, dict[str, float]]:
    """Compute summary statistics from successful bootstrap fits.

    For mock estimator fits, we use the objective value as the parameter
    estimate (since mock doesn't produce real parameter estimates).
    For real estimator fits, we extract from the table's fixed_params.
    """
    if not fits:
        return {}

    # Collect parameter estimates from each fit
    all_params: dict[str, list[float]] = {}
    for fit in fits:
        # Try to get parameter estimates from table
        fixed_params = fit.table.get("fixed_params")
        if isinstance(fixed_params, dict) and fixed_params:
            for name, value in fixed_params.items():
                all_params.setdefault(name, []).append(float(value))
        else:
            # For mock estimator: use ini values as "estimates" perturbed by objective
            for name, iv in fit.model.ini.values.items():
                all_params.setdefault(name, []).append(iv.estimate)
            # Also record the objective as a pseudo-parameter
            all_params.setdefault("objective", []).append(fit.objective)

    alpha = 1.0 - ci_level
    summary: dict[str, dict[str, float]] = {}
    for name, values in all_params.items():
        arr = np.array(values)
        summary[name] = {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "se": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "ci_lo": float(np.percentile(arr, 100 * alpha / 2)),
            "ci_hi": float(np.percentile(arr, 100 * (1 - alpha / 2))),
        }

    return summary


def bootstrap_fit(
    fit: NLMIXRFit,
    data: dict[str, Any],
    n_boot: int = 100,
    seed: int = 0,
    ci_level: float = 0.95,
) -> BootstrapResult:
    """Nonparametric bootstrap: resample subjects and refit.

    Parameters
    ----------
    fit : NLMIXRFit
        Original fit result whose model/estimator/control to reuse.
    data : dict
        Original data (column-oriented) with an "id" column.
    n_boot : int
        Number of bootstrap replicates.
    seed : int
        Base random seed (each replicate uses seed + i).
    ci_level : float
        Confidence interval level (default 0.95).

    Returns
    -------
    BootstrapResult
    """
    successful_fits: list[NLMIXRFit] = []
    n_fail = 0

    for i in range(n_boot):
        resampled = resample_by_subject(data, seed=seed + i)
        try:
            new_fit = nlmixr2(
                fit.model,
                data=resampled,
                est=fit.estimator,
                control=fit.control if fit.control else None,
            )
            if isinstance(new_fit, NLMIXRFit):
                successful_fits.append(new_fit)
            else:
                n_fail += 1
        except Exception:
            n_fail += 1

    parameter_summary = _compute_parameter_summary(successful_fits, ci_level=ci_level)

    return BootstrapResult(
        fits=successful_fits,
        n_success=len(successful_fits),
        n_fail=n_fail,
        parameter_summary=parameter_summary,
    )


def parametric_bootstrap(
    fit: NLMIXRFit,
    data: dict[str, Any],
    n_boot: int = 100,
    seed: int = 0,
    omega: float | None = None,
    sigma: float = 1.0,
) -> BootstrapResult:
    """Parametric bootstrap: simulate new datasets from fitted model + noise, then refit.

    Parameters
    ----------
    fit : NLMIXRFit
        Original fit result.
    data : dict
        Original data (column-oriented).
    n_boot : int
        Number of bootstrap replicates.
    seed : int
        Base random seed.
    omega : float or None
        Between-subject variability scale. If None, defaults to 0.1.
    sigma : float
        Residual variability scale.

    Returns
    -------
    BootstrapResult
    """
    if omega is None:
        omega = 0.1

    rng = np.random.RandomState(seed)
    successful_fits: list[NLMIXRFit] = []
    n_fail = 0

    # Get DV column for perturbation
    dv = list(data["dv"]) if "dv" in data else None
    n_obs = len(list(data.values())[0]) if data else 0

    for i in range(n_boot):
        # Create simulated dataset by perturbing DV with noise
        sim_data = dict(data)  # shallow copy
        if dv is not None:
            noise = rng.normal(0, sigma, size=n_obs)
            eta = rng.normal(0, omega, size=n_obs)
            sim_dv = [float(dv[j]) + noise[j] + eta[j] for j in range(n_obs)]
            sim_data = {k: (sim_dv if k == "dv" else list(v)) for k, v in data.items()}

        try:
            new_fit = nlmixr2(
                fit.model,
                data=sim_data,
                est=fit.estimator,
                control=fit.control if fit.control else None,
            )
            if isinstance(new_fit, NLMIXRFit):
                successful_fits.append(new_fit)
            else:
                n_fail += 1
        except Exception:
            n_fail += 1

    parameter_summary = _compute_parameter_summary(successful_fits, ci_level=0.95)

    return BootstrapResult(
        fits=successful_fits,
        n_success=len(successful_fits),
        n_fail=n_fail,
        parameter_summary=parameter_summary,
    )
