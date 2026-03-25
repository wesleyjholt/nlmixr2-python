"""Visual Predictive Check (VPC) data generation.

Simulates replicate datasets from a pharmacometric model and computes
prediction-interval quantiles for comparison with observed data.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class VPCResult:
    """Container for VPC output.

    Attributes
    ----------
    observed : dict
        ``{"time": array, "dv": array}`` from the original data.
    simulated_quantiles : dict
        ``{"time": array, "lo": array, "median": array, "hi": array}``
        where lo/median/hi correspond to the prediction interval quantiles.
    pi : tuple[float, float, float]
        The quantile levels used (e.g. (0.05, 0.5, 0.95)).
    n_sim : int
        Number of simulation replicates.
    """

    observed: Dict[str, np.ndarray]
    simulated_quantiles: Dict[str, np.ndarray]
    pi: Tuple[float, float, float]
    n_sim: int


def bin_times(
    times: jax.Array,
    method: str = "time",
    n_bins: int = 10,
) -> np.ndarray:
    """Compute bin centers from a time array.

    Parameters
    ----------
    times
        1-D array of time values.
    method
        ``"time"`` for equal-width bins, ``"ntile"`` for equal-count bins.
    n_bins
        Number of bins.

    Returns
    -------
    np.ndarray
        Array of bin centers with length ``n_bins``.

    Raises
    ------
    ValueError
        If *method* is not ``"time"`` or ``"ntile"``.
    """
    times_np = np.asarray(times).ravel()

    if method == "time":
        t_min, t_max = float(times_np.min()), float(times_np.max())
        edges = np.linspace(t_min, t_max, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers

    if method == "ntile":
        quantile_edges = np.linspace(0.0, 100.0, n_bins + 1)
        edges = np.percentile(times_np, quantile_edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers

    raise ValueError(
        f"Unknown binning method '{method}'. Supported: 'time', 'ntile'."
    )


def compute_quantiles(
    simulated_dvs: jax.Array,
    quantiles: Tuple[float, ...],
) -> Dict[float, np.ndarray]:
    """Compute quantiles across simulation replicates.

    Parameters
    ----------
    simulated_dvs
        Array of shape ``(n_sim, n_bins)`` — simulated summary values per bin.
    quantiles
        Tuple of quantile levels, e.g. ``(0.05, 0.5, 0.95)``.

    Returns
    -------
    dict[float, np.ndarray]
        Mapping from quantile level to 1-D array of length ``n_bins``.
    """
    arr = np.asarray(simulated_dvs)
    result = {}
    for q in quantiles:
        result[q] = np.quantile(arr, q, axis=0)
    return result


def _assign_bins(times_np: np.ndarray, bin_centers: np.ndarray) -> np.ndarray:
    """Assign each time point to the nearest bin center.

    Returns an integer index array of the same length as *times_np*.
    """
    diffs = np.abs(times_np[:, None] - bin_centers[None, :])
    return np.argmin(diffs, axis=1)


def _trace_param_names(model_func: Callable, sample_times: jax.Array) -> list[str]:
    """Discover parameter names by calling the model with a tracing dict."""

    class _ParamTracer(dict):
        """Dict subclass that records which keys are accessed."""

        def __init__(self):
            super().__init__()
            self._accessed: list[str] = []

        def __getitem__(self, key):
            if key not in self._accessed:
                self._accessed.append(key)
            return 0.0

        def __contains__(self, key):
            return True

    tracer = _ParamTracer()
    try:
        model_func(tracer, sample_times)
    except Exception:
        pass
    return tracer._accessed


def vpc(
    fit_or_model_func: Callable,
    data: Dict[str, jax.Array],
    n_sim: int = 200,
    omega: jax.Array | None = None,
    sigma: float = 1.0,
    seed: int = 0,
    pi: Tuple[float, float, float] = (0.05, 0.5, 0.95),
    bin_method: str = "time",
    n_bins: int = 10,
) -> VPCResult:
    """Generate VPC data by simulating replicates from the model.

    For each replicate, new individual random effects (etas) are sampled
    from ``MVN(0, omega)``, predictions are computed via the model, and
    additive residual error ``N(0, sigma)`` is added.  The simulated
    observations are then binned by time and quantiles are computed across
    replicates.

    Parameters
    ----------
    fit_or_model_func
        A callable ``(params_dict, times) -> predictions``.
    data
        Must contain ``"id"``, ``"time"``, ``"dv"`` arrays.
    n_sim
        Number of simulation replicates.
    omega
        Between-subject covariance matrix, shape ``(n_etas, n_etas)``.
        If *None* a 1x1 zero matrix is used (no random effects).
    sigma
        Residual error variance (additive normal).
    seed
        PRNG seed for reproducibility.
    pi
        Prediction interval quantiles ``(lo, median, hi)``.
    bin_method
        Binning strategy: ``"time"`` or ``"ntile"``.
    n_bins
        Number of time bins.

    Returns
    -------
    VPCResult
    """
    model_func = fit_or_model_func
    times_jnp = jnp.asarray(data["time"])
    dv_jnp = jnp.asarray(data["dv"])
    ids_jnp = jnp.asarray(data["id"])
    times_np = np.asarray(times_jnp)
    dv_np = np.asarray(dv_jnp)

    if omega is None:
        omega = jnp.zeros((1, 1))

    n_etas = omega.shape[0]
    unique_ids = jnp.unique(ids_jnp)
    n_subjects = int(unique_ids.shape[0])

    # Discover parameter names by tracing a call to the model
    param_names = _trace_param_names(model_func, times_jnp[:1])
    if not param_names:
        param_names = [f"p{i}" for i in range(n_etas)]

    # Estimate baseline parameter values from observed data.
    # First param is set to median(dv), rest to 0 — a simple heuristic
    # that works well when the first param is a scale/amplitude term.
    median_dv = float(np.median(dv_np))
    base_params = {
        name: (median_dv if i == 0 else 0.0)
        for i, name in enumerate(param_names)
    }

    # Validate that the model runs with our base params; fall back to zeros
    try:
        test_pred = model_func(base_params, times_jnp[:3])
        _ = float(jnp.mean(test_pred))
    except Exception:
        base_params = {name: 0.0 for name in param_names}

    # --- Compute bin structure ---
    effective_n_bins = min(n_bins, len(np.unique(times_np)))
    bin_centers = bin_times(times_jnp, method=bin_method, n_bins=effective_n_bins)
    actual_n_bins = len(bin_centers)
    bin_indices = _assign_bins(times_np, bin_centers)

    # --- Simulate replicates ---
    key = jax.random.PRNGKey(seed)
    L = jnp.linalg.cholesky(omega + jnp.eye(n_etas) * 1e-12)

    # sim_bin_dvs[sim_idx, bin_idx] = median DV in that bin for that replicate
    sim_bin_dvs = np.zeros((n_sim, actual_n_bins))

    for sim_i in range(n_sim):
        key, key_eta, key_eps = jax.random.split(key, 3)

        # Sample etas for each subject from MVN(0, omega)
        z = jax.random.normal(key_eta, shape=(n_subjects, n_etas))
        etas = z @ L.T  # (n_subjects, n_etas)

        # Compute predictions for all subjects
        all_preds = []
        for subj_idx in range(n_subjects):
            subj_id = unique_ids[subj_idx]
            mask = ids_jnp == subj_id
            subj_times = times_jnp[mask]

            # Individual params = base + eta offset for the first n_etas params
            indiv_params = {}
            for j, name in enumerate(param_names):
                if j < n_etas:
                    indiv_params[name] = base_params[name] + float(etas[subj_idx, j])
                else:
                    indiv_params[name] = base_params[name]

            pred = model_func(indiv_params, subj_times)
            all_preds.append(np.asarray(pred))

        all_preds_arr = np.concatenate(all_preds)

        # Add residual error ~ N(0, sigma)
        eps = np.asarray(
            jax.random.normal(key_eps, shape=all_preds_arr.shape)
        ) * np.sqrt(sigma)
        sim_dv = all_preds_arr + eps

        # Aggregate into bins (median per bin)
        for b in range(actual_n_bins):
            bin_mask = bin_indices == b
            if np.any(bin_mask):
                sim_bin_dvs[sim_i, b] = np.median(sim_dv[bin_mask])

    # --- Compute quantiles across simulations ---
    q_lo, q_med, q_hi = pi
    quantile_dict = compute_quantiles(
        sim_bin_dvs, quantiles=(q_lo, q_med, q_hi)
    )

    return VPCResult(
        observed={"time": times_np, "dv": dv_np},
        simulated_quantiles={
            "time": bin_centers,
            "lo": quantile_dict[q_lo],
            "median": quantile_dict[q_med],
            "hi": quantile_dict[q_hi],
        },
        pi=pi,
        n_sim=n_sim,
    )
