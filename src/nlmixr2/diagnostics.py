"""Post-fit diagnostic utilities for nlmixr2 fit objects."""

from __future__ import annotations

import math
from typing import Any, Callable

import jax.numpy as jnp
import jax.scipy.special as jspecial

from .api import NLMIXRFit


def compute_predictions(
    fit: NLMIXRFit,
    model_func: Callable,
    data: dict[str, jnp.ndarray],
    *,
    thetas: jnp.ndarray,
    etas: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """Compute population/individual predictions and residuals.

    Parameters
    ----------
    fit : NLMIXRFit
        The fit object (used for metadata if needed).
    model_func : callable
        ``model_func(thetas, etas, data) -> jnp.ndarray`` returning
        predictions for each observation.
    data : dict
        Must contain ``"dv"`` (dependent variable) at minimum.
    thetas : jnp.ndarray
        Fixed-effect parameter vector.
    etas : jnp.ndarray
        Per-observation random-effect values (1-D, same length as dv).

    Returns
    -------
    dict with keys: pred, ipred, res, ires, wres
    """
    dv = data["dv"]

    # Population predictions: etas zeroed out
    zero_etas = jnp.zeros_like(etas)
    pred = model_func(thetas, zero_etas, data)

    # Individual predictions: full etas
    ipred = model_func(thetas, etas, data)

    # Residuals
    res = dv - pred
    ires = dv - ipred

    # Weighted residuals: scale by std of residuals (avoid division by zero)
    res_std = jnp.std(res)
    res_std = jnp.where(res_std > 0.0, res_std, 1.0)
    wres = res / res_std

    return {
        "pred": pred,
        "ipred": ipred,
        "res": res,
        "ires": ires,
        "wres": wres,
    }


def compute_cwres(
    dv: jnp.ndarray,
    pred: jnp.ndarray,
    ipred: jnp.ndarray,
    sigma: float,
) -> jnp.ndarray:
    """Compute conditional weighted residuals (CWRES).

    Simplified formula when no full covariance is available:
    ``CWRES_i = (DV_i - IPRED_i) / sigma``

    Parameters
    ----------
    dv : jnp.ndarray
        Observed dependent variable values.
    pred : jnp.ndarray
        Population predictions (unused in simplified form, kept for API
        consistency with the full CWRES formula).
    ipred : jnp.ndarray
        Individual predictions.
    sigma : float
        Residual standard deviation.

    Returns
    -------
    jnp.ndarray
        CWRES values with the same shape as *dv*.
    """
    return (dv - ipred) / sigma


def compute_npde(
    dv: jnp.ndarray,
    simulated_dvs: jnp.ndarray,
) -> jnp.ndarray:
    """Compute normalised prediction distribution errors (NPDE).

    For each observation the fraction of simulated values that fall below the
    observed value is computed, then transformed to standard-normal quantiles
    via the inverse normal CDF (``ndtri``).

    Parameters
    ----------
    dv : jnp.ndarray
        Observed dependent variable values, shape ``(n_obs,)``.
    simulated_dvs : jnp.ndarray
        Simulated dependent variable values, shape ``(n_sim, n_obs)``.

    Returns
    -------
    jnp.ndarray
        NPDE values with the same shape as *dv*.
    """
    n_sim = simulated_dvs.shape[0]
    # Fraction of simulations strictly less than the observed value
    fraction = jnp.sum(simulated_dvs < dv[None, :], axis=0) / n_sim
    # Clamp to (0, 1) to avoid infinities from ndtri at 0 or 1
    eps = 1.0 / (2.0 * n_sim)
    fraction = jnp.clip(fraction, eps, 1.0 - eps)
    # Inverse normal CDF (probit)
    npde = jspecial.ndtri(fraction)
    return npde


def compute_wres(
    dv: jnp.ndarray,
    pred: jnp.ndarray,
    sigma: float,
) -> jnp.ndarray:
    """Compute weighted residuals.

    ``WRES_i = (DV_i - PRED_i) / sigma``

    Parameters
    ----------
    dv : jnp.ndarray
        Observed dependent variable values.
    pred : jnp.ndarray
        Population predictions.
    sigma : float
        Residual standard deviation.

    Returns
    -------
    jnp.ndarray
        WRES values with the same shape as *dv*.
    """
    return (dv - pred) / sigma


def compute_aic(objective: float, n_params: int) -> float:
    """AIC = objective + 2 * n_params."""
    return objective + 2 * n_params


def compute_bic(objective: float, n_params: int, n_obs: int) -> float:
    """BIC = objective + n_params * ln(n_obs)."""
    return objective + n_params * math.log(n_obs)


def compute_condition_number(hessian: jnp.ndarray) -> float:
    """Ratio of largest to smallest absolute eigenvalue of *hessian*."""
    eigenvalues = jnp.linalg.eigvalsh(hessian)
    abs_eig = jnp.abs(eigenvalues)
    return float(jnp.max(abs_eig) / jnp.min(abs_eig))


def compute_shrinkage(etas: jnp.ndarray, omega: jnp.ndarray) -> jnp.ndarray:
    """Eta shrinkage per random effect: 1 - var(etas) / omega_diag.

    Parameters
    ----------
    etas : jnp.ndarray
        Shape ``(n_subjects, n_random_effects)``.
    omega : jnp.ndarray
        Omega matrix (square).  Only the diagonal is used.

    Returns
    -------
    jnp.ndarray of shape ``(n_random_effects,)``
    """
    omega_diag = jnp.diag(omega)
    eta_var = jnp.var(etas, axis=0)
    # Where omega_diag is zero, shrinkage is 0 by convention (no random effect)
    shrinkage = jnp.where(
        omega_diag > 0.0,
        1.0 - eta_var / omega_diag,
        0.0,
    )
    return shrinkage


def compute_per_subject_predictions(
    model_func: Callable,
    data: dict[str, jnp.ndarray],
    fixed_params: dict[str, float],
    etas: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """Compute per-subject PRED (population) and IPRED (individual) predictions.

    Parameters
    ----------
    model_func : callable
        ``(params_dict, times) -> predictions`` array.
    data : dict
        Must contain ``"id"``, ``"time"``, ``"dv"`` arrays.
    fixed_params : dict
        Population (fixed-effect) parameters.
    etas : jnp.ndarray
        Per-subject random effects, shape ``(n_subjects, n_etas)``.

    Returns
    -------
    dict with keys: id, time, dv, pred, ipred, res, ires
    """
    ids = data["id"]
    times = data["time"]
    dv = data["dv"]
    n_obs = ids.shape[0]
    param_names = list(fixed_params.keys())
    n_etas = etas.shape[1]

    unique_ids = jnp.unique(ids)

    pred_all = jnp.zeros(n_obs)
    ipred_all = jnp.zeros(n_obs)

    for i in range(unique_ids.shape[0]):
        subj_id = unique_ids[i]
        mask = ids == subj_id
        subj_times = times[mask]

        # Population prediction: etas = 0
        pop_params = dict(fixed_params)
        pred_subj = model_func(pop_params, subj_times)
        pred_all = pred_all.at[mask].set(pred_subj)

        # Individual prediction: fixed + eta offsets
        indiv_params = {}
        for j, name in enumerate(param_names):
            if j < n_etas:
                indiv_params[name] = fixed_params[name] + float(etas[i, j])
            else:
                indiv_params[name] = fixed_params[name]
        ipred_subj = model_func(indiv_params, subj_times)
        ipred_all = ipred_all.at[mask].set(ipred_subj)

    res = dv - pred_all
    ires = dv - ipred_all

    return {
        "id": ids,
        "time": times,
        "dv": dv,
        "pred": pred_all,
        "ipred": ipred_all,
        "res": res,
        "ires": ires,
    }


def compute_phi(
    model_func: Callable,
    data: dict[str, jnp.ndarray],
    fixed_params: dict[str, float],
    etas: jnp.ndarray,
    sigma: float,
) -> dict[float, float]:
    """Compute per-subject individual log-likelihood (phi).

    Parameters
    ----------
    model_func : callable
        ``(params_dict, times) -> predictions`` array.
    data : dict
        Must contain ``"id"``, ``"time"``, ``"dv"`` arrays.
    fixed_params : dict
        Population (fixed-effect) parameters.
    etas : jnp.ndarray
        Per-subject random effects, shape ``(n_subjects, n_etas)``.
    sigma : float
        Residual error standard deviation.

    Returns
    -------
    dict mapping subject_id (float) to individual log-likelihood value.
        phi_i = -0.5 * sum_j [(dv_ij - ipred_ij)^2 / sigma^2 + log(2*pi*sigma^2)]
    """
    ids = data["id"]
    times = data["time"]
    dv = data["dv"]
    param_names = list(fixed_params.keys())
    n_etas = etas.shape[1]
    unique_ids = jnp.unique(ids)

    phi_dict: dict[float, float] = {}
    sigma_sq = sigma ** 2

    for i in range(unique_ids.shape[0]):
        subj_id = unique_ids[i]
        mask = ids == subj_id
        subj_times = times[mask]
        subj_dv = dv[mask]

        # Individual params: fixed + eta offsets
        indiv_params = {}
        for j, name in enumerate(param_names):
            if j < n_etas:
                indiv_params[name] = fixed_params[name] + float(etas[i, j])
            else:
                indiv_params[name] = fixed_params[name]

        ipred = model_func(indiv_params, subj_times)
        resid_sq = jnp.sum((subj_dv - ipred) ** 2)
        n_obs = subj_times.shape[0]

        phi_i = -0.5 * float(
            resid_sq / sigma_sq + n_obs * jnp.log(2.0 * jnp.pi * sigma_sq)
        )
        phi_dict[float(subj_id)] = phi_i

    return phi_dict


def summarize_fit(
    fit: NLMIXRFit,
    predictions: dict[str, Any] | None = None,
) -> str:
    """Return a formatted summary string similar to R's ``print.nlmixr2``.

    Parameters
    ----------
    fit : NLMIXRFit
        Completed fit object.
    predictions : dict, optional
        May contain ``"shrinkage"`` (array) and ``"shrinkage_labels"`` (list of str).
    """
    aic = compute_aic(fit.objective, fit.parameter_count)
    bic = compute_bic(fit.objective, fit.parameter_count, max(fit.n_observations, 1))

    lines: list[str] = []
    lines.append("nlmixr2 Fit Summary")
    lines.append("=" * 40)
    lines.append(f"Estimator:       {fit.estimator}")
    lines.append(f"Observations:    {fit.n_observations}")
    lines.append(f"Parameters:      {fit.parameter_count}")
    lines.append("")
    lines.append(f"Objective:       {fit.objective:.4f}")
    lines.append(f"AIC:             {aic:.4f}")
    lines.append(f"BIC:             {bic:.4f}")
    lines.append("")
    lines.append("Parameter Estimates")
    lines.append("-" * 40)
    for name, init_val in fit.model.ini.values.items():
        lines.append(f"  {name:<16s} {init_val.estimate:.4f}")

    if predictions and "shrinkage" in predictions:
        lines.append("")
        lines.append("Eta Shrinkage")
        lines.append("-" * 40)
        shrinkage = predictions["shrinkage"]
        labels = predictions.get("shrinkage_labels")
        for i in range(len(shrinkage)):
            label = labels[i] if labels and i < len(labels) else f"eta.{i}"
            lines.append(f"  {label:<16s} {float(shrinkage[i]):.2%}")

    lines.append("")
    return "\n".join(lines)
