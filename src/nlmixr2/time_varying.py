"""Time-varying covariate support for ODE integration.

Provides utilities to:
- Define time-varying covariates with LOCF or linear interpolation
- Build callable covariate functions for use during ODE solving
- Extract time-varying covariates from tabular data
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import jax.numpy as jnp
import numpy as np


@dataclass
class TimeVaryingCovariate:
    """A covariate whose value changes at specified time points.

    Attributes
    ----------
    name : str
        Covariate name (e.g., ``"wt"``).
    times : jnp.ndarray
        1-D array of time points where the covariate value changes.
        Must be sorted in ascending order.
    values : jnp.ndarray
        1-D array of covariate values at each time point.
        Must have the same length as *times*.
    method : str
        Interpolation method: ``"locf"`` (last observation carried forward,
        default) or ``"linear"`` (linear interpolation between time points,
        clamped at boundaries).
    """

    name: str
    times: jnp.ndarray
    values: jnp.ndarray
    method: str = "locf"


def interpolate_covariate(tvc: TimeVaryingCovariate, t: float) -> float:
    """Return the covariate value at time *t*.

    Parameters
    ----------
    tvc : TimeVaryingCovariate
        The time-varying covariate specification.
    t : float
        The query time.

    Returns
    -------
    float
        Interpolated covariate value.
    """
    times = jnp.asarray(tvc.times, dtype=jnp.float32)
    values = jnp.asarray(tvc.values, dtype=jnp.float32)

    if tvc.method == "locf":
        # Find the index of the last time point <= t.
        # If t is before all time points, use the first value.
        mask = times <= t
        # If no time <= t, use index 0 (first value)
        idx = jnp.where(jnp.any(mask), jnp.sum(mask) - 1, 0)
        return values[idx]

    elif tvc.method == "linear":
        n = len(times)
        # Clamp: before first -> first value, after last -> last value
        t_clamped = jnp.clip(t, times[0], times[-1])

        # Find the rightmost index where times <= t_clamped
        mask = times <= t_clamped
        idx_lo = jnp.clip(jnp.sum(mask) - 1, 0, n - 2)
        idx_hi = idx_lo + 1

        t_lo = times[idx_lo]
        t_hi = times[idx_hi]
        v_lo = values[idx_lo]
        v_hi = values[idx_hi]

        # Linear interpolation; avoid division by zero if t_lo == t_hi
        dt = t_hi - t_lo
        frac = jnp.where(dt > 0, (t_clamped - t_lo) / dt, 0.0)
        return v_lo + frac * (v_hi - v_lo)

    else:
        raise ValueError(f"Unknown interpolation method: {tvc.method!r}")


def build_covariate_function(
    tvcs: List[TimeVaryingCovariate],
) -> Callable[[float], Dict[str, Any]]:
    """Build a function that returns covariate values at a given time.

    Parameters
    ----------
    tvcs : list of TimeVaryingCovariate
        The time-varying covariates.

    Returns
    -------
    callable
        A function ``f(t) -> dict`` mapping covariate names to values at time *t*.
    """

    def cov_fn(t: float) -> dict[str, Any]:
        return {tvc.name: interpolate_covariate(tvc, t) for tvc in tvcs}

    return cov_fn


def extract_time_varying(
    data: dict[str, Any],
    covariate_names: list[str],
    id_column: str = "id",
) -> dict[Any, list[TimeVaryingCovariate]]:
    """Detect time-varying covariates from tabular data.

    A covariate is considered time-varying for a given subject if its values
    differ across time points within that subject.

    Parameters
    ----------
    data : dict[str, array-like]
        Tabular data with at least columns for *id_column*, ``"time"``,
        and each name in *covariate_names*.
    covariate_names : list of str
        Names of covariates to check.
    id_column : str
        Name of the subject identifier column (default ``"id"``).

    Returns
    -------
    dict
        Mapping of subject ID to a list of :class:`TimeVaryingCovariate`
        objects.  Subjects with no time-varying covariates are omitted.
    """
    ids = np.asarray(data[id_column])
    times = np.asarray(data["time"])
    unique_ids = np.unique(ids)

    result: dict[Any, list[TimeVaryingCovariate]] = {}

    for subj_id in unique_ids:
        mask = ids == subj_id
        subj_times = times[mask]
        # Sort by time
        sort_idx = np.argsort(subj_times)
        subj_times_sorted = subj_times[sort_idx]

        tvcs: list[TimeVaryingCovariate] = []
        for cov_name in covariate_names:
            cov_vals = np.asarray(data[cov_name])[mask][sort_idx]
            # Check if the covariate varies within this subject
            if not np.all(cov_vals == cov_vals[0]):
                tvcs.append(
                    TimeVaryingCovariate(
                        name=cov_name,
                        times=jnp.array(subj_times_sorted),
                        values=jnp.array(cov_vals),
                        method="locf",
                    )
                )

        if tvcs:
            # Convert subj_id to Python int if it's a numpy scalar
            key = int(subj_id) if np.issubdtype(type(subj_id), np.integer) else subj_id
            result[key] = tvcs

    return result
