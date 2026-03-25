"""FOCE (First-Order Conditional Estimation) estimator using JAX.

Provides an approximate -2 log-likelihood objective function and an
iterative optimiser that alternates between:
  - Inner problem: optimise individual random effects (etas) per subject
  - Outer problem: optimise population fixed-effect parameters

Gradients are computed via ``jax.grad``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Set, Tuple

import jax
import jax.numpy as jnp


@dataclass
class EstimationResult:
    """Container for FOCE estimation output.

    Attributes
    ----------
    fixed_params : dict[str, float]
        Estimated population (fixed-effect) parameters.
    etas : jax.Array
        Estimated individual random effects, shape ``(n_subjects, n_etas)``.
    objective : float
        Final FOCE objective function value (approximate -2LL).
    n_iterations : int
        Number of outer iterations performed.
    converged : bool
        Whether the optimiser satisfied the convergence criterion.
    """

    fixed_params: Dict[str, float]
    etas: jax.Array
    objective: float
    n_iterations: int
    converged: bool


# ---------------------------------------------------------------------------
# FOCE objective function
# ---------------------------------------------------------------------------

def foce_objective(
    fixed_params: Dict[str, float],
    etas: jax.Array,
    omega: jax.Array,
    sigma: float,
    model_func: Callable,
    data: Dict[str, jax.Array],
) -> jax.Array:
    """Compute the FOCE approximate -2 log-likelihood.

    Parameters
    ----------
    fixed_params : dict
        Population parameters (e.g. ``{"A": 10.0, "ke": 0.5}``).
    etas : array, shape (n_subjects, n_etas)
        Individual random effects.
    omega : array, shape (n_etas, n_etas)
        Between-subject variability covariance matrix.
    sigma : float
        Residual error variance.
    model_func : callable
        ``(params_dict, times) -> predictions`` array.
    data : dict
        Must contain ``"id"``, ``"time"``, ``"dv"`` arrays.

    Returns
    -------
    jax.Array
        Scalar FOCE objective value.
    """
    ids = data["id"]
    times = data["time"]
    dv = data["dv"]

    n_etas = etas.shape[1]
    omega_inv = jnp.linalg.inv(omega)
    _, log_det_omega = jnp.linalg.slogdet(omega)

    unique_ids = jnp.unique(ids)
    n_subjects = unique_ids.shape[0]

    param_names = list(fixed_params.keys())

    total = jnp.array(0.0)

    for i in range(n_subjects):
        subj_id = unique_ids[i]
        mask = ids == subj_id
        subj_times = times[mask]
        subj_dv = dv[mask]
        n_obs = subj_times.shape[0]

        eta_i = etas[i]  # (n_etas,)

        # Build individual params: fixed + eta offsets
        indiv_params = {}
        for j, name in enumerate(param_names):
            if j < n_etas:
                indiv_params[name] = fixed_params[name] + eta_i[j]
            else:
                indiv_params[name] = fixed_params[name]

        pred = model_func(indiv_params, subj_times)
        resid = subj_dv - pred

        # -2LL contribution from residual error (normal)
        ll_resid = (
            n_obs * jnp.log(2.0 * jnp.pi * sigma)
            + jnp.sum(resid ** 2) / sigma
        )

        # -2LL contribution from random effects prior
        ll_eta = (
            log_det_omega
            + eta_i @ omega_inv @ eta_i
            + n_etas * jnp.log(2.0 * jnp.pi)
        )

        total = total + ll_resid + ll_eta

    return total


# ---------------------------------------------------------------------------
# FOCEi objective function (FOCE with interaction)
# ---------------------------------------------------------------------------

def focei_objective(
    fixed_params: Dict[str, float],
    etas: jax.Array,
    omega: jax.Array,
    sigma: float,
    model_func: Callable,
    data: Dict[str, jax.Array],
) -> jax.Array:
    """Compute the FOCEi approximate -2 log-likelihood.

    Unlike FOCE, FOCEi includes the eta-epsilon interaction: the residual
    variance for each observation depends on the individual random effects
    through the Jacobian df/deta.

    The -2LL for subject *i* is::

        log|Sigma_i| + (y_i - f_i)^T Sigma_i^{-1} (y_i - f_i)
        + log|Omega| + eta_i^T Omega^{-1} eta_i + const

    where ``Sigma_i = diag(sigma * (1 + (df/deta)^2))`` captures the
    interaction between random effects and residual error.

    Parameters
    ----------
    fixed_params : dict
        Population parameters (e.g. ``{"A": 10.0, "ke": 0.5}``).
    etas : array, shape (n_subjects, n_etas)
        Individual random effects.
    omega : array, shape (n_etas, n_etas)
        Between-subject variability covariance matrix.
    sigma : float
        Residual error variance.
    model_func : callable
        ``(params_dict, times) -> predictions`` array.
    data : dict
        Must contain ``"id"``, ``"time"``, ``"dv"`` arrays.

    Returns
    -------
    jax.Array
        Scalar FOCEi objective value.
    """
    ids = data["id"]
    times = data["time"]
    dv = data["dv"]

    n_etas = etas.shape[1]
    omega_inv = jnp.linalg.inv(omega)
    _, log_det_omega = jnp.linalg.slogdet(omega)

    unique_ids = jnp.unique(ids)
    n_subjects = unique_ids.shape[0]

    param_names = list(fixed_params.keys())

    total = jnp.array(0.0)

    for i in range(n_subjects):
        subj_id = unique_ids[i]
        mask = ids == subj_id
        subj_times = times[mask]
        subj_dv = dv[mask]
        n_obs = subj_times.shape[0]

        eta_i = etas[i]  # (n_etas,)

        # Build individual params: fixed + eta offsets
        indiv_params = {}
        for j, name in enumerate(param_names):
            if j < n_etas:
                indiv_params[name] = fixed_params[name] + eta_i[j]
            else:
                indiv_params[name] = fixed_params[name]

        pred = model_func(indiv_params, subj_times)
        resid = subj_dv - pred

        # Compute df/deta via JAX Jacobian for the interaction term
        def _pred_from_eta(eta_vec, _fp=dict(fixed_params), _st=subj_times):
            p = {}
            for j2, name2 in enumerate(param_names):
                if j2 < n_etas:
                    p[name2] = _fp[name2] + eta_vec[j2]
                else:
                    p[name2] = _fp[name2]
            return model_func(p, _st)

        # jac shape: (n_obs, n_etas)
        jac = jax.jacobian(_pred_from_eta)(eta_i)

        # Interaction variance per observation:
        # Sigma_i_diag = sigma * (1 + sum_k (df/deta_k)^2)
        # This scales the residual variance by the sensitivity to etas
        interaction = jnp.sum(jac ** 2, axis=1)  # (n_obs,)
        sigma_i_diag = sigma * (1.0 + interaction)  # (n_obs,)

        # -2LL contribution from residual error with interaction
        ll_resid = (
            jnp.sum(jnp.log(2.0 * jnp.pi * sigma_i_diag))
            + jnp.sum(resid ** 2 / sigma_i_diag)
        )

        # -2LL contribution from random effects prior
        ll_eta = (
            log_det_omega
            + eta_i @ omega_inv @ eta_i
            + n_etas * jnp.log(2.0 * jnp.pi)
        )

        total = total + ll_resid + ll_eta

    return total


# ---------------------------------------------------------------------------
# Internal helpers for array <-> dict conversion
# ---------------------------------------------------------------------------

def _params_to_array(params: Dict[str, float]) -> tuple[list[str], jax.Array]:
    """Convert parameter dict to (ordered names, values array)."""
    names = list(params.keys())
    values = jnp.array([float(params[n]) for n in names])
    return names, values


def _array_to_params(names: list[str], values: jax.Array) -> Dict[str, float]:
    """Convert ordered names + values array back to a dict."""
    return {n: float(values[i]) for i, n in enumerate(names)}


def _clip_param_array(
    param_arr: jax.Array,
    param_names: list[str],
    bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]],
) -> jax.Array:
    """Clip parameter array according to bounds dict.

    Parameters
    ----------
    param_arr : array
        Current parameter values.
    param_names : list[str]
        Ordered parameter names matching ``param_arr``.
    bounds : dict or None
        Mapping of param name to ``(lower, upper)`` tuple.
        Either bound may be ``None`` to indicate no constraint on that side.

    Returns
    -------
    jax.Array
        Clipped parameter values.
    """
    if bounds is None:
        return param_arr
    lower = jnp.full_like(param_arr, -jnp.inf)
    upper = jnp.full_like(param_arr, jnp.inf)
    for i, name in enumerate(param_names):
        if name in bounds:
            lo, hi = bounds[name]
            if lo is not None:
                lower = lower.at[i].set(lo)
            if hi is not None:
                upper = upper.at[i].set(hi)
    return jnp.clip(param_arr, lower, upper)


# ---------------------------------------------------------------------------
# Differentiable wrappers for JAX grad
# ---------------------------------------------------------------------------

def _make_objective_wrt_etas(
    fixed_params: Dict[str, float],
    omega: jax.Array,
    sigma: float,
    model_func: Callable,
    data: Dict[str, jax.Array],
):
    """Return a function ``f(etas) -> scalar`` closed over fixed params."""

    def fn(etas):
        return foce_objective(fixed_params, etas, omega, sigma, model_func, data)

    return fn


def _make_objective_wrt_fixed(
    param_names: list[str],
    etas: jax.Array,
    omega: jax.Array,
    sigma: float,
    model_func: Callable,
    data: Dict[str, jax.Array],
):
    """Return a function ``f(param_array) -> scalar`` closed over etas."""

    def fn(param_arr):
        fp = {n: param_arr[i] for i, n in enumerate(param_names)}
        return foce_objective(fp, etas, omega, sigma, model_func, data)

    return fn


# ---------------------------------------------------------------------------
# FOCEi differentiable wrappers
# ---------------------------------------------------------------------------

def _make_focei_objective_wrt_etas(
    fixed_params: Dict[str, float],
    omega: jax.Array,
    sigma: float,
    model_func: Callable,
    data: Dict[str, jax.Array],
):
    """Return a function ``f(etas) -> scalar`` using FOCEi objective."""

    def fn(etas):
        return focei_objective(fixed_params, etas, omega, sigma, model_func, data)

    return fn


def _make_focei_objective_wrt_fixed(
    param_names: list[str],
    etas: jax.Array,
    omega: jax.Array,
    sigma: float,
    model_func: Callable,
    data: Dict[str, jax.Array],
):
    """Return a function ``f(param_array) -> scalar`` using FOCEi objective."""

    def fn(param_arr):
        fp = {n: param_arr[i] for i, n in enumerate(param_names)}
        return focei_objective(fp, etas, omega, sigma, model_func, data)

    return fn


# ---------------------------------------------------------------------------
# Main estimation routine (FOCE)
# ---------------------------------------------------------------------------

def estimate_foce(
    model: Callable,
    data: Dict[str, jax.Array],
    ini_values: Dict[str, float],
    omega: jax.Array,
    control: Optional[Dict[str, Any]] = None,
    bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
    fixed_names: Optional[Set[str]] = None,
) -> EstimationResult:
    """Run FOCE estimation via alternating gradient descent.

    Parameters
    ----------
    model : callable
        ``(params_dict, times) -> predictions``.
    data : dict
        Must contain ``"id"``, ``"time"``, ``"dv"``.
    ini_values : dict
        Starting values for fixed-effect parameters.
    omega : array
        Between-subject covariance matrix.
    control : dict, optional
        Optimisation settings:
        - ``maxiter`` (int, default 100): max outer iterations.
        - ``lr`` (float, default 0.01): learning rate for fixed effects.
        - ``lr_eta`` (float, default 0.01): learning rate for etas.
        - ``tol`` (float, default 1e-4): convergence tolerance on
          relative change in objective.
        - ``inner_steps`` (int, default 10): gradient steps on etas
          per outer iteration.
        - ``sigma`` (float, default 1.0): residual error variance.
    bounds : dict, optional
        Mapping of parameter name to ``(lower, upper)`` tuple.
        Either bound may be ``None`` for one-sided constraints.
        Parameters are clipped to these bounds after each optimisation step.

    Returns
    -------
    EstimationResult
    """
    ctrl = dict(control or {})
    maxiter = int(ctrl.get("maxiter", 100))
    lr = float(ctrl.get("lr", 0.01))
    lr_eta = float(ctrl.get("lr_eta", 0.01))
    tol = float(ctrl.get("tol", 1e-4))
    inner_steps = int(ctrl.get("inner_steps", 10))
    sigma = float(ctrl.get("sigma", 1.0))

    # Determine number of subjects
    ids = data["id"]
    unique_ids = jnp.unique(ids)
    n_subjects = int(unique_ids.shape[0])
    n_etas = omega.shape[0]

    # Build mask for non-fixed parameters (1.0 = free, 0.0 = fixed)
    _fixed = fixed_names or set()

    # Initialise
    param_names, param_arr = _params_to_array(ini_values)
    free_mask = jnp.array([0.0 if n in _fixed else 1.0 for n in param_names])
    etas = jnp.zeros((n_subjects, n_etas))

    prev_obj = jnp.inf
    converged = False
    iteration = 0

    # Adam optimiser state for fixed params
    m_fixed = jnp.zeros_like(param_arr)
    v_fixed = jnp.zeros_like(param_arr)
    # Adam state for etas
    m_eta = jnp.zeros_like(etas)
    v_eta = jnp.zeros_like(etas)
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_eps = 1e-8

    for iteration in range(1, maxiter + 1):
        # --- Inner problem: optimise etas given current fixed params ---
        current_fixed = _array_to_params(param_names, param_arr)
        eta_obj_fn = _make_objective_wrt_etas(
            current_fixed, omega, sigma, model, data
        )
        eta_grad_fn = jax.grad(eta_obj_fn)

        for inner_it in range(inner_steps):
            g_eta = eta_grad_fn(etas)
            # Adam update for etas
            step_t = iteration * inner_steps + inner_it + 1
            m_eta = adam_beta1 * m_eta + (1 - adam_beta1) * g_eta
            v_eta = adam_beta2 * v_eta + (1 - adam_beta2) * g_eta ** 2
            m_hat = m_eta / (1 - adam_beta1 ** step_t)
            v_hat = v_eta / (1 - adam_beta2 ** step_t)
            etas = etas - lr_eta * m_hat / (jnp.sqrt(v_hat) + adam_eps)

        # --- Outer problem: optimise fixed params given current etas ---
        fixed_obj_fn = _make_objective_wrt_fixed(
            param_names, etas, omega, sigma, model, data
        )
        fixed_grad_fn = jax.grad(fixed_obj_fn)
        g_fixed = fixed_grad_fn(param_arr)
        # Zero out gradients for fixed parameters
        g_fixed = g_fixed * free_mask
        # Adam update for fixed params
        m_fixed = adam_beta1 * m_fixed + (1 - adam_beta1) * g_fixed
        v_fixed = adam_beta2 * v_fixed + (1 - adam_beta2) * g_fixed ** 2
        m_hat_f = m_fixed / (1 - adam_beta1 ** iteration)
        v_hat_f = v_fixed / (1 - adam_beta2 ** iteration)
        param_arr = param_arr - lr * m_hat_f / (jnp.sqrt(v_hat_f) + adam_eps)

        # --- Enforce parameter bounds ---
        param_arr = _clip_param_array(param_arr, param_names, bounds)

        # --- Check convergence ---
        current_fixed = _array_to_params(param_names, param_arr)
        obj_val = foce_objective(
            current_fixed, etas, omega, sigma, model, data
        )

        rel_change = jnp.abs(obj_val - prev_obj) / (jnp.abs(prev_obj) + 1e-10)
        if rel_change < tol and iteration > 1:
            converged = True
            break

        prev_obj = obj_val

    final_params = _array_to_params(param_names, param_arr)
    final_obj = float(
        foce_objective(final_params, etas, omega, sigma, model, data)
    )

    return EstimationResult(
        fixed_params=final_params,
        etas=etas,
        objective=final_obj,
        n_iterations=iteration,
        converged=converged,
    )


# ---------------------------------------------------------------------------
# FOCEi estimation routine
# ---------------------------------------------------------------------------

def estimate_focei(
    model: Callable,
    data: Dict[str, jax.Array],
    ini_values: Dict[str, float],
    omega: jax.Array,
    control: Optional[Dict[str, Any]] = None,
    bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
    fixed_names: Optional[Set[str]] = None,
) -> EstimationResult:
    """Run FOCEi estimation via alternating gradient descent.

    Identical to :func:`estimate_foce` except that the objective function
    includes the eta-epsilon interaction term (FOCEi).

    Parameters
    ----------
    model : callable
        ``(params_dict, times) -> predictions``.
    data : dict
        Must contain ``"id"``, ``"time"``, ``"dv"``.
    ini_values : dict
        Starting values for fixed-effect parameters.
    omega : array
        Between-subject covariance matrix.
    control : dict, optional
        Optimisation settings (same as FOCE):
        - ``maxiter`` (int, default 100): max outer iterations.
        - ``lr`` (float, default 0.01): learning rate for fixed effects.
        - ``lr_eta`` (float, default 0.01): learning rate for etas.
        - ``tol`` (float, default 1e-4): convergence tolerance on
          relative change in objective.
        - ``inner_steps`` (int, default 10): gradient steps on etas
          per outer iteration.
        - ``sigma`` (float, default 1.0): residual error variance.
    bounds : dict, optional
        Mapping of parameter name to ``(lower, upper)`` tuple.
    fixed_names : set, optional
        Parameters to hold fixed during estimation.

    Returns
    -------
    EstimationResult
    """
    ctrl = dict(control or {})
    maxiter = int(ctrl.get("maxiter", 100))
    lr = float(ctrl.get("lr", 0.01))
    lr_eta = float(ctrl.get("lr_eta", 0.01))
    tol = float(ctrl.get("tol", 1e-4))
    inner_steps = int(ctrl.get("inner_steps", 10))
    sigma = float(ctrl.get("sigma", 1.0))

    # Determine number of subjects
    ids = data["id"]
    unique_ids = jnp.unique(ids)
    n_subjects = int(unique_ids.shape[0])
    n_etas = omega.shape[0]

    # Build mask for non-fixed parameters (1.0 = free, 0.0 = fixed)
    _fixed = fixed_names or set()

    # Initialise
    param_names, param_arr = _params_to_array(ini_values)
    free_mask = jnp.array([0.0 if n in _fixed else 1.0 for n in param_names])
    etas = jnp.zeros((n_subjects, n_etas))

    prev_obj = jnp.inf
    converged = False
    iteration = 0

    # Adam optimiser state for fixed params
    m_fixed = jnp.zeros_like(param_arr)
    v_fixed = jnp.zeros_like(param_arr)
    # Adam state for etas
    m_eta = jnp.zeros_like(etas)
    v_eta = jnp.zeros_like(etas)
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_eps = 1e-8

    for iteration in range(1, maxiter + 1):
        # --- Inner problem: optimise etas given current fixed params ---
        current_fixed = _array_to_params(param_names, param_arr)
        eta_obj_fn = _make_focei_objective_wrt_etas(
            current_fixed, omega, sigma, model, data
        )
        eta_grad_fn = jax.grad(eta_obj_fn)

        for inner_it in range(inner_steps):
            g_eta = eta_grad_fn(etas)
            # Adam update for etas
            step_t = iteration * inner_steps + inner_it + 1
            m_eta = adam_beta1 * m_eta + (1 - adam_beta1) * g_eta
            v_eta = adam_beta2 * v_eta + (1 - adam_beta2) * g_eta ** 2
            m_hat = m_eta / (1 - adam_beta1 ** step_t)
            v_hat = v_eta / (1 - adam_beta2 ** step_t)
            etas = etas - lr_eta * m_hat / (jnp.sqrt(v_hat) + adam_eps)

        # --- Outer problem: optimise fixed params given current etas ---
        fixed_obj_fn = _make_focei_objective_wrt_fixed(
            param_names, etas, omega, sigma, model, data
        )
        fixed_grad_fn = jax.grad(fixed_obj_fn)
        g_fixed = fixed_grad_fn(param_arr)
        # Zero out gradients for fixed parameters
        g_fixed = g_fixed * free_mask
        # Adam update for fixed params
        m_fixed = adam_beta1 * m_fixed + (1 - adam_beta1) * g_fixed
        v_fixed = adam_beta2 * v_fixed + (1 - adam_beta2) * g_fixed ** 2
        m_hat_f = m_fixed / (1 - adam_beta1 ** iteration)
        v_hat_f = v_fixed / (1 - adam_beta2 ** iteration)
        param_arr = param_arr - lr * m_hat_f / (jnp.sqrt(v_hat_f) + adam_eps)

        # --- Enforce parameter bounds ---
        param_arr = _clip_param_array(param_arr, param_names, bounds)

        # --- Check convergence ---
        current_fixed = _array_to_params(param_names, param_arr)
        obj_val = focei_objective(
            current_fixed, etas, omega, sigma, model, data
        )

        rel_change = jnp.abs(obj_val - prev_obj) / (jnp.abs(prev_obj) + 1e-10)
        if rel_change < tol and iteration > 1:
            converged = True
            break

        prev_obj = obj_val

    final_params = _array_to_params(param_names, param_arr)
    final_obj = float(
        focei_objective(final_params, etas, omega, sigma, model, data)
    )

    return EstimationResult(
        fixed_params=final_params,
        etas=etas,
        objective=final_obj,
        n_iterations=iteration,
        converged=converged,
    )


# ---------------------------------------------------------------------------
# Laplacian approximation objective function
# ---------------------------------------------------------------------------

def laplacian_objective(
    fixed_params: Dict[str, float],
    etas: jax.Array,
    omega: jax.Array,
    sigma: float,
    model_func: Callable,
    data: Dict[str, jax.Array],
) -> jax.Array:
    """Compute the full Laplacian approximation -2 log-likelihood.

    Unlike FOCE, the Laplacian method includes the Hessian correction
    ``log|H_i|`` for each subject, where *H_i* is the Hessian of the
    individual objective with respect to ``eta_i``.

    The per-subject contribution is::

        n_i*log(2*pi) + log|Sigma_i| + (y-f)^T Sigma^{-1} (y-f)
        + log|Omega| + eta^T Omega^{-1} eta + log|H_i|

    Parameters
    ----------
    fixed_params : dict
        Population parameters.
    etas : array, shape (n_subjects, n_etas)
        Individual random effects.
    omega : array, shape (n_etas, n_etas)
        Between-subject variability covariance matrix.
    sigma : float
        Residual error variance.
    model_func : callable
        ``(params_dict, times) -> predictions`` array.
    data : dict
        Must contain ``"id"``, ``"time"``, ``"dv"`` arrays.

    Returns
    -------
    jax.Array
        Scalar Laplacian objective value.
    """
    ids = data["id"]
    times = data["time"]
    dv = data["dv"]

    n_etas = etas.shape[1]
    omega_inv = jnp.linalg.inv(omega)
    _, log_det_omega = jnp.linalg.slogdet(omega)

    unique_ids = jnp.unique(ids)
    n_subjects = unique_ids.shape[0]

    param_names = list(fixed_params.keys())

    total = jnp.array(0.0)

    for i in range(n_subjects):
        subj_id = unique_ids[i]
        mask = ids == subj_id
        subj_times = times[mask]
        subj_dv = dv[mask]
        n_obs = subj_times.shape[0]

        eta_i = etas[i]  # (n_etas,)

        # Build individual params: fixed + eta offsets
        indiv_params = {}
        for j, name in enumerate(param_names):
            if j < n_etas:
                indiv_params[name] = fixed_params[name] + eta_i[j]
            else:
                indiv_params[name] = fixed_params[name]

        pred = model_func(indiv_params, subj_times)
        resid = subj_dv - pred

        # -2LL contribution from residual error (normal)
        ll_resid = (
            n_obs * jnp.log(2.0 * jnp.pi * sigma)
            + jnp.sum(resid ** 2) / sigma
        )

        # -2LL contribution from random effects prior
        ll_eta = (
            log_det_omega
            + eta_i @ omega_inv @ eta_i
            + n_etas * jnp.log(2.0 * jnp.pi)
        )

        # Hessian correction: compute d^2(individual_obj)/d(eta_i)^2
        def _individual_obj(eta_vec, _fp=dict(fixed_params), _st=subj_times,
                            _dv=subj_dv):
            p = {}
            for j2, name2 in enumerate(param_names):
                if j2 < n_etas:
                    p[name2] = _fp[name2] + eta_vec[j2]
                else:
                    p[name2] = _fp[name2]
            pred_i = model_func(p, _st)
            res_i = _dv - pred_i
            obj_resid = jnp.sum(res_i ** 2) / sigma
            obj_eta = eta_vec @ omega_inv @ eta_vec
            return obj_resid + obj_eta

        hessian_i = jax.hessian(_individual_obj)(eta_i)  # (n_etas, n_etas)
        _, log_det_hessian = jnp.linalg.slogdet(hessian_i)

        total = total + ll_resid + ll_eta + log_det_hessian

    return total


# ---------------------------------------------------------------------------
# Laplacian differentiable wrappers
# ---------------------------------------------------------------------------

def _make_laplacian_objective_wrt_etas(
    fixed_params: Dict[str, float],
    omega: jax.Array,
    sigma: float,
    model_func: Callable,
    data: Dict[str, jax.Array],
):
    """Return a function ``f(etas) -> scalar`` using Laplacian objective."""

    def fn(etas):
        return laplacian_objective(fixed_params, etas, omega, sigma, model_func, data)

    return fn


def _make_laplacian_objective_wrt_fixed(
    param_names: list[str],
    etas: jax.Array,
    omega: jax.Array,
    sigma: float,
    model_func: Callable,
    data: Dict[str, jax.Array],
):
    """Return a function ``f(param_array) -> scalar`` using Laplacian objective."""

    def fn(param_arr):
        fp = {n: param_arr[i] for i, n in enumerate(param_names)}
        return laplacian_objective(fp, etas, omega, sigma, model_func, data)

    return fn


# ---------------------------------------------------------------------------
# Laplacian estimation routine
# ---------------------------------------------------------------------------

def estimate_laplacian(
    model: Callable,
    data: Dict[str, jax.Array],
    ini_values: Dict[str, float],
    omega: jax.Array,
    control: Optional[Dict[str, Any]] = None,
    bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
    fixed_names: Optional[Set[str]] = None,
) -> EstimationResult:
    """Run Laplacian estimation via alternating gradient descent.

    Identical in structure to :func:`estimate_foce` but uses the full
    Laplacian approximation objective that includes the Hessian correction
    term ``log|H_i|``.

    Parameters
    ----------
    model : callable
        ``(params_dict, times) -> predictions``.
    data : dict
        Must contain ``"id"``, ``"time"``, ``"dv"``.
    ini_values : dict
        Starting values for fixed-effect parameters.
    omega : array
        Between-subject covariance matrix.
    control : dict, optional
        Optimisation settings (same as FOCE):
        - ``maxiter`` (int, default 100): max outer iterations.
        - ``lr`` (float, default 0.01): learning rate for fixed effects.
        - ``lr_eta`` (float, default 0.01): learning rate for etas.
        - ``tol`` (float, default 1e-4): convergence tolerance.
        - ``inner_steps`` (int, default 10): gradient steps on etas.
        - ``sigma`` (float, default 1.0): residual error variance.
    bounds : dict, optional
        Mapping of parameter name to ``(lower, upper)`` tuple.
    fixed_names : set, optional
        Parameters to hold fixed during estimation.

    Returns
    -------
    EstimationResult
    """
    ctrl = dict(control or {})
    maxiter = int(ctrl.get("maxiter", 100))
    lr = float(ctrl.get("lr", 0.01))
    lr_eta = float(ctrl.get("lr_eta", 0.01))
    tol = float(ctrl.get("tol", 1e-4))
    inner_steps = int(ctrl.get("inner_steps", 10))
    sigma = float(ctrl.get("sigma", 1.0))

    # Determine number of subjects
    ids = data["id"]
    unique_ids = jnp.unique(ids)
    n_subjects = int(unique_ids.shape[0])
    n_etas = omega.shape[0]

    # Build mask for non-fixed parameters
    _fixed = fixed_names or set()

    # Initialise
    param_names, param_arr = _params_to_array(ini_values)
    free_mask = jnp.array([0.0 if n in _fixed else 1.0 for n in param_names])
    etas = jnp.zeros((n_subjects, n_etas))

    prev_obj = jnp.inf
    converged = False
    iteration = 0

    # Adam optimiser state
    m_fixed = jnp.zeros_like(param_arr)
    v_fixed = jnp.zeros_like(param_arr)
    m_eta = jnp.zeros_like(etas)
    v_eta = jnp.zeros_like(etas)
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_eps = 1e-8

    for iteration in range(1, maxiter + 1):
        # --- Inner problem: optimise etas ---
        current_fixed = _array_to_params(param_names, param_arr)
        eta_obj_fn = _make_laplacian_objective_wrt_etas(
            current_fixed, omega, sigma, model, data
        )
        eta_grad_fn = jax.grad(eta_obj_fn)

        for inner_it in range(inner_steps):
            g_eta = eta_grad_fn(etas)
            step_t = iteration * inner_steps + inner_it + 1
            m_eta = adam_beta1 * m_eta + (1 - adam_beta1) * g_eta
            v_eta = adam_beta2 * v_eta + (1 - adam_beta2) * g_eta ** 2
            m_hat = m_eta / (1 - adam_beta1 ** step_t)
            v_hat = v_eta / (1 - adam_beta2 ** step_t)
            etas = etas - lr_eta * m_hat / (jnp.sqrt(v_hat) + adam_eps)

        # --- Outer problem: optimise fixed params ---
        fixed_obj_fn = _make_laplacian_objective_wrt_fixed(
            param_names, etas, omega, sigma, model, data
        )
        fixed_grad_fn = jax.grad(fixed_obj_fn)
        g_fixed = fixed_grad_fn(param_arr)
        g_fixed = g_fixed * free_mask
        m_fixed = adam_beta1 * m_fixed + (1 - adam_beta1) * g_fixed
        v_fixed = adam_beta2 * v_fixed + (1 - adam_beta2) * g_fixed ** 2
        m_hat_f = m_fixed / (1 - adam_beta1 ** iteration)
        v_hat_f = v_fixed / (1 - adam_beta2 ** iteration)
        param_arr = param_arr - lr * m_hat_f / (jnp.sqrt(v_hat_f) + adam_eps)

        # --- Enforce parameter bounds ---
        param_arr = _clip_param_array(param_arr, param_names, bounds)

        # --- Check convergence ---
        current_fixed = _array_to_params(param_names, param_arr)
        obj_val = laplacian_objective(
            current_fixed, etas, omega, sigma, model, data
        )

        rel_change = jnp.abs(obj_val - prev_obj) / (jnp.abs(prev_obj) + 1e-10)
        if rel_change < tol and iteration > 1:
            converged = True
            break

        prev_obj = obj_val

    final_params = _array_to_params(param_names, param_arr)
    final_obj = float(
        laplacian_objective(final_params, etas, omega, sigma, model, data)
    )

    return EstimationResult(
        fixed_params=final_params,
        etas=etas,
        objective=final_obj,
        n_iterations=iteration,
        converged=converged,
    )


# ---------------------------------------------------------------------------
# SAEM (Stochastic Approximation Expectation-Maximization) estimator
# ---------------------------------------------------------------------------

def _log_posterior_individual(
    eta_i: jax.Array,
    fixed_params: Dict[str, float],
    param_names: list[str],
    n_etas: int,
    omega_inv: jax.Array,
    log_det_omega: jax.Array,
    sigma: float,
    model_func: Callable,
    subj_times: jax.Array,
    subj_dv: jax.Array,
) -> float:
    """Compute the unnormalised log-posterior for one subject's random effects.

    Returns the negative log-posterior (so lower is better) for use in
    Metropolis-Hastings acceptance.
    """
    # Build individual params
    indiv_params = {}
    for j, name in enumerate(param_names):
        if j < n_etas:
            indiv_params[name] = fixed_params[name] + eta_i[j]
        else:
            indiv_params[name] = fixed_params[name]

    pred = model_func(indiv_params, subj_times)
    resid = subj_dv - pred
    n_obs = subj_times.shape[0]

    # Log-likelihood of observations (normal residual)
    ll_resid = -0.5 * (
        n_obs * jnp.log(2.0 * jnp.pi * sigma)
        + jnp.sum(resid ** 2) / sigma
    )

    # Log-prior on etas (multivariate normal)
    ll_eta = -0.5 * (
        log_det_omega
        + eta_i @ omega_inv @ eta_i
        + n_etas * jnp.log(2.0 * jnp.pi)
    )

    return ll_resid + ll_eta  # log-posterior (higher is better)


def _mcmc_sample_etas(
    key: jax.Array,
    etas: jax.Array,
    fixed_params: Dict[str, float],
    param_names: list[str],
    n_etas: int,
    omega_inv: jax.Array,
    log_det_omega: jax.Array,
    sigma: float,
    model_func: Callable,
    data: Dict[str, jax.Array],
    proposal_scale: float = 0.1,
    n_mcmc_steps: int = 5,
) -> tuple[jax.Array, jax.Array]:
    """Run Metropolis-Hastings MCMC to sample etas for each subject.

    Returns updated etas array, shape (n_subjects, n_etas).
    """
    ids = data["id"]
    times = data["time"]
    dv = data["dv"]
    unique_ids = jnp.unique(ids)
    n_subjects = unique_ids.shape[0]

    new_etas = []
    for i in range(n_subjects):
        subj_id = unique_ids[i]
        mask = ids == subj_id
        subj_times = times[mask]
        subj_dv = dv[mask]

        eta_i = etas[i]

        current_lp = _log_posterior_individual(
            eta_i, fixed_params, param_names, n_etas,
            omega_inv, log_det_omega, sigma, model_func,
            subj_times, subj_dv,
        )

        for _ in range(n_mcmc_steps):
            key, k_prop, k_accept = jax.random.split(key, 3)
            proposal = eta_i + jax.random.normal(k_prop, shape=eta_i.shape) * proposal_scale
            proposal_lp = _log_posterior_individual(
                proposal, fixed_params, param_names, n_etas,
                omega_inv, log_det_omega, sigma, model_func,
                subj_times, subj_dv,
            )
            log_alpha = proposal_lp - current_lp
            u = jax.random.uniform(k_accept)
            accept = jnp.log(u) < log_alpha
            eta_i = jnp.where(accept, proposal, eta_i)
            current_lp = jnp.where(accept, proposal_lp, current_lp)

        new_etas.append(eta_i)

    return key, jnp.stack(new_etas)


def estimate_saem(
    model: Callable,
    data: Dict[str, jax.Array],
    ini_values: Dict[str, float],
    omega: jax.Array,
    control: Optional[Dict[str, Any]] = None,
    bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
    fixed_names: Optional[Set[str]] = None,
) -> EstimationResult:
    """Run SAEM estimation.

    Implements the Stochastic Approximation EM algorithm:
      - E-step: sample individual random effects via MCMC (Metropolis-Hastings)
      - M-step: update population parameters using stochastic approximation

    Two phases:
      1. Burn-in: fixed step size for exploration
      2. Convergence: decreasing step size (1/k) for convergence

    Parameters
    ----------
    model : callable
        ``(params_dict, times) -> predictions``.
    data : dict
        Must contain ``"id"``, ``"time"``, ``"dv"``.
    ini_values : dict
        Starting values for fixed-effect parameters.
    omega : array
        Initial between-subject covariance matrix.
    control : dict, optional
        Algorithm settings:
        - ``n_burn`` (int, default 50): burn-in iterations.
        - ``n_em`` (int, default 100): convergence phase iterations.
        - ``n_mcmc_steps`` (int, default 5): MCMC steps per E-step.
        - ``step_size`` (float, default 0.5): initial SA step size.
        - ``sigma`` (float, default 1.0): residual error variance.
        - ``proposal_scale`` (float, default 0.1): MCMC proposal std dev.
        - ``tol`` (float, default 1e-3): convergence tolerance on
          relative change in parameters.
        - ``seed`` (int, default 0): PRNG seed.
    bounds : dict, optional
        Mapping of parameter name to ``(lower, upper)`` tuple.
        Either bound may be ``None`` for one-sided constraints.
        Fixed effects are clipped to these bounds after each M-step update.

    Returns
    -------
    EstimationResult
    """
    ctrl = dict(control or {})
    n_burn = int(ctrl.get("n_burn", 50))
    n_em = int(ctrl.get("n_em", 100))
    n_mcmc_steps = int(ctrl.get("n_mcmc_steps", 5))
    step_size = float(ctrl.get("step_size", 0.5))
    sigma = float(ctrl.get("sigma", 1.0))
    proposal_scale = float(ctrl.get("proposal_scale", 0.1))
    tol = float(ctrl.get("tol", 1e-3))
    seed = int(ctrl.get("seed", 0))

    # Setup
    ids = data["id"]
    unique_ids = jnp.unique(ids)
    n_subjects = int(unique_ids.shape[0])
    n_etas = omega.shape[0]

    _fixed = fixed_names or set()
    param_names = list(ini_values.keys())
    # Current parameter estimates (mutable copies)
    current_fixed = {k: float(v) for k, v in ini_values.items()}
    current_omega = jnp.array(omega, dtype=jnp.float32)

    etas = jnp.zeros((n_subjects, n_etas))
    key = jax.random.PRNGKey(seed)

    total_iter = n_burn + n_em
    converged = False

    # Running sufficient statistics for stochastic approximation
    # S1 = running mean of etas (for fixed effect update)
    # S2 = running mean of eta * eta^T (for omega update)
    S1 = jnp.zeros(n_etas)
    S2 = jnp.zeros((n_etas, n_etas))

    prev_params_arr = jnp.array([current_fixed[n] for n in param_names])

    for iteration in range(1, total_iter + 1):
        is_burn = iteration <= n_burn

        # --- E-step: sample etas via MCMC ---
        omega_inv = jnp.linalg.inv(current_omega)
        _, log_det_omega = jnp.linalg.slogdet(current_omega)

        key, etas = _mcmc_sample_etas(
            key, etas, current_fixed, param_names, n_etas,
            omega_inv, log_det_omega, sigma, model, data,
            proposal_scale=proposal_scale,
            n_mcmc_steps=n_mcmc_steps,
        )

        # --- Compute sufficient statistics from sampled etas ---
        mean_eta = jnp.mean(etas, axis=0)
        # Empirical second moment: (1/N) sum eta_i eta_i^T
        eta_outer = jnp.mean(
            jnp.einsum("ij,ik->ijk", etas, etas), axis=0
        )

        # --- Stochastic approximation step ---
        if is_burn:
            gamma = step_size  # fixed step during burn-in
        else:
            k = iteration - n_burn
            gamma = step_size / k  # decreasing step during convergence

        # Update sufficient statistics
        S1 = S1 + gamma * (mean_eta - S1)
        S2 = S2 + gamma * (eta_outer - S2)

        # --- M-step: update population parameters ---
        # Update fixed effects: shift by the running mean of etas
        # Skip parameters marked as fixed (held constant).
        for j, name in enumerate(param_names):
            if j < n_etas and name not in _fixed:
                current_fixed[name] = float(
                    ini_values[name] + S1[j]
                )

        # Enforce parameter bounds on fixed effects
        if bounds is not None:
            for name in param_names:
                if name in bounds:
                    lo, hi = bounds[name]
                    val = current_fixed[name]
                    if lo is not None and val < lo:
                        current_fixed[name] = lo
                    if hi is not None and val > hi:
                        current_fixed[name] = hi

        # Update omega: empirical covariance from sufficient statistics
        # Omega = S2 - S1 * S1^T (covariance = E[eta eta^T] - E[eta]E[eta]^T)
        new_omega = S2 - jnp.outer(S1, S1)
        # Ensure positive definiteness with a small ridge
        new_omega = new_omega + jnp.eye(n_etas) * 1e-6
        current_omega = new_omega

        # --- Check convergence in EM phase ---
        if not is_burn and iteration > n_burn + 5:
            current_arr = jnp.array([current_fixed[n] for n in param_names])
            rel_change = jnp.max(
                jnp.abs(current_arr - prev_params_arr)
                / (jnp.abs(prev_params_arr) + 1e-10)
            )
            if rel_change < tol:
                converged = True
            prev_params_arr = current_arr

    # Compute final objective using FOCE objective for comparability
    final_obj = float(
        foce_objective(current_fixed, etas, current_omega, sigma, model, data)
    )

    return EstimationResult(
        fixed_params=current_fixed,
        etas=etas,
        objective=final_obj,
        n_iterations=total_iter,
        converged=converged,
    )


# ---------------------------------------------------------------------------
# Posthoc (Empirical Bayes) estimator
# ---------------------------------------------------------------------------

def _individual_neg_log_posterior(
    eta_i: jax.Array,
    fixed_params: Dict[str, float],
    param_names: list[str],
    n_etas: int,
    omega_inv: jax.Array,
    sigma: float,
    model_func: Callable,
    subj_times: jax.Array,
    subj_dv: jax.Array,
) -> jax.Array:
    """Compute the negative log-posterior for one subject's etas.

    This is the objective to minimise per-subject in posthoc estimation.
    """
    indiv_params = {}
    for j, name in enumerate(param_names):
        if j < n_etas:
            indiv_params[name] = fixed_params[name] + eta_i[j]
        else:
            indiv_params[name] = fixed_params[name]

    pred = model_func(indiv_params, subj_times)
    resid = subj_dv - pred
    n_obs = subj_times.shape[0]

    # Negative log-likelihood of residuals
    nll_resid = 0.5 * (
        n_obs * jnp.log(2.0 * jnp.pi * sigma)
        + jnp.sum(resid ** 2) / sigma
    )

    # Negative log-prior on etas
    nll_eta = 0.5 * eta_i @ omega_inv @ eta_i

    return nll_resid + nll_eta


def estimate_posthoc(
    model_func: Callable,
    data: Dict[str, jax.Array],
    fixed_params: Dict[str, float],
    omega: jax.Array,
    sigma: float = 1.0,
    control: Optional[Dict[str, Any]] = None,
) -> EstimationResult:
    """Compute posthoc (empirical Bayes) estimates of individual etas.

    For each subject, optimise etas to maximise the individual posterior
    (i.e. minimise the negative log-posterior), conditional on known
    population fixed-effect parameters and omega.

    Parameters
    ----------
    model_func : callable
        ``(params_dict, times) -> predictions``.
    data : dict
        Must contain ``"id"``, ``"time"``, ``"dv"``.
    fixed_params : dict
        Known population parameters (not estimated, returned unchanged).
    omega : array, shape (n_etas, n_etas)
        Between-subject variability covariance matrix.
    sigma : float
        Residual error variance.
    control : dict, optional
        - ``maxiter`` (int, default 100): max gradient descent iterations.
        - ``lr`` (float, default 0.01): learning rate.
        - ``tol`` (float, default 1e-6): convergence tolerance on
          gradient norm.

    Returns
    -------
    EstimationResult
    """
    ctrl = dict(control or {})
    maxiter = int(ctrl.get("maxiter", 100))
    lr = float(ctrl.get("lr", 0.01))
    tol = float(ctrl.get("tol", 1e-6))

    ids = data["id"]
    times = data["time"]
    dv = data["dv"]

    unique_ids = jnp.unique(ids)
    n_subjects = int(unique_ids.shape[0])
    n_etas = omega.shape[0]

    omega_inv = jnp.linalg.inv(omega)
    param_names = list(fixed_params.keys())

    all_etas = []
    total_obj = 0.0
    converged = True
    total_iters = 0

    for i in range(n_subjects):
        subj_id = unique_ids[i]
        mask = ids == subj_id
        subj_times = times[mask]
        subj_dv = dv[mask]

        eta_i = jnp.zeros(n_etas)

        def obj_fn(eta, st=subj_times, sd=subj_dv):
            return _individual_neg_log_posterior(
                eta, fixed_params, param_names, n_etas,
                omega_inv, sigma, model_func, st, sd,
            )

        grad_fn = jax.grad(obj_fn)

        subj_converged = False
        for it in range(1, maxiter + 1):
            g = grad_fn(eta_i)
            eta_i = eta_i - lr * g
            if jnp.linalg.norm(g) < tol:
                subj_converged = True
                total_iters = max(total_iters, it)
                break
        else:
            total_iters = max(total_iters, maxiter)

        if not subj_converged:
            converged = False

        total_obj += float(obj_fn(eta_i))
        all_etas.append(eta_i)

    etas = jnp.stack(all_etas)

    return EstimationResult(
        fixed_params=dict(fixed_params),
        etas=etas,
        objective=total_obj,
        n_iterations=total_iters,
        converged=converged,
    )


# ---------------------------------------------------------------------------
# Nelder-Mead (NLM) simplex estimator
# ---------------------------------------------------------------------------

def _nelder_mead_step(
    simplex: jax.Array,
    f_values: jax.Array,
    obj_fn: Callable,
    alpha: float = 1.0,
    gamma: float = 2.0,
    rho: float = 0.5,
    sigma_shrink: float = 0.5,
) -> tuple[jax.Array, jax.Array]:
    """Perform one Nelder-Mead simplex iteration.

    Parameters
    ----------
    simplex : array, shape (n+1, n)
        Current simplex vertices.
    f_values : array, shape (n+1,)
        Objective values at each vertex.
    obj_fn : callable
        Objective function mapping an array of length n to a scalar.
    alpha, gamma, rho, sigma_shrink : float
        Standard Nelder-Mead coefficients for reflection, expansion,
        contraction, and shrink.

    Returns
    -------
    simplex, f_values : updated simplex and objective values.
    """
    n = simplex.shape[1]
    # Sort vertices by objective value
    order = jnp.argsort(f_values)
    simplex = simplex[order]
    f_values = f_values[order]

    best = simplex[0]
    worst = simplex[-1]
    second_worst = simplex[-2]
    f_best = f_values[0]
    f_worst = f_values[-1]
    f_second_worst = f_values[-2]

    # Centroid of all points except the worst
    centroid = jnp.mean(simplex[:-1], axis=0)

    # Reflection
    x_r = centroid + alpha * (centroid - worst)
    f_r = obj_fn(x_r)

    if f_r < f_second_worst and f_r >= f_best:
        # Accept reflection
        simplex = simplex.at[-1].set(x_r)
        f_values = f_values.at[-1].set(f_r)
        return simplex, f_values

    if f_r < f_best:
        # Try expansion
        x_e = centroid + gamma * (x_r - centroid)
        f_e = obj_fn(x_e)
        if f_e < f_r:
            simplex = simplex.at[-1].set(x_e)
            f_values = f_values.at[-1].set(f_e)
        else:
            simplex = simplex.at[-1].set(x_r)
            f_values = f_values.at[-1].set(f_r)
        return simplex, f_values

    # Contraction
    if f_r < f_worst:
        # Outside contraction
        x_c = centroid + rho * (x_r - centroid)
        f_c = obj_fn(x_c)
        if f_c <= f_r:
            simplex = simplex.at[-1].set(x_c)
            f_values = f_values.at[-1].set(f_c)
            return simplex, f_values
    else:
        # Inside contraction
        x_c = centroid - rho * (centroid - worst)
        f_c = obj_fn(x_c)
        if f_c < f_worst:
            simplex = simplex.at[-1].set(x_c)
            f_values = f_values.at[-1].set(f_c)
            return simplex, f_values

    # Shrink: move all vertices toward the best vertex
    for i in range(1, n + 1):
        simplex = simplex.at[i].set(best + sigma_shrink * (simplex[i] - best))
        f_values = f_values.at[i].set(obj_fn(simplex[i]))

    return simplex, f_values


def estimate_nlm(
    model_func: Callable,
    data: Dict[str, jax.Array],
    ini_values: Dict[str, float],
    omega: jax.Array,
    control: Optional[Dict[str, Any]] = None,
) -> EstimationResult:
    """Run Nelder-Mead (NLM) simplex estimation.

    A derivative-free optimiser that alternates between:
      - Inner problem: optimise individual random effects (etas) per subject
        using gradient descent.
      - Outer problem: one Nelder-Mead simplex step on the population
        fixed-effect parameters.

    The objective is the FOCE approximate -2 log-likelihood.

    Parameters
    ----------
    model_func : callable
        ``(params_dict, times) -> predictions``.
    data : dict
        Must contain ``"id"``, ``"time"``, ``"dv"``.
    ini_values : dict
        Starting values for fixed-effect parameters.
    omega : array
        Between-subject covariance matrix.
    control : dict, optional
        Optimisation settings:
        - ``maxiter`` (int, default 500): max outer (simplex) iterations.
        - ``tol`` (float, default 1e-6): convergence tolerance on the
          range of objective values across the simplex.
        - ``sigma`` (float, default 1.0): residual error variance.
        - ``inner_steps`` (int, default 10): gradient descent steps on
          etas per outer iteration.
        - ``lr_eta`` (float, default 0.01): learning rate for inner eta
          optimisation.
        - ``simplex_scale`` (float, default 0.05): scale for initial
          simplex perturbations.

    Returns
    -------
    EstimationResult
    """
    ctrl = dict(control or {})
    maxiter = int(ctrl.get("maxiter", 500))
    tol = float(ctrl.get("tol", 1e-6))
    sigma = float(ctrl.get("sigma", 1.0))
    inner_steps = int(ctrl.get("inner_steps", 10))
    lr_eta = float(ctrl.get("lr_eta", 0.01))
    simplex_scale = float(ctrl.get("simplex_scale", 0.05))

    # Setup
    ids = data["id"]
    unique_ids = jnp.unique(ids)
    n_subjects = int(unique_ids.shape[0])
    n_etas = omega.shape[0]

    param_names, param_arr = _params_to_array(ini_values)
    n_params = len(param_names)
    etas = jnp.zeros((n_subjects, n_etas))

    # --- Helper: optimise etas for given fixed param array ---
    def _optimise_etas(fixed_arr, current_etas):
        """Run inner_steps of gradient descent on etas with gradient clipping."""
        fp = _array_to_params(param_names, fixed_arr)
        eta_fn = _make_objective_wrt_etas(fp, omega, sigma, model_func, data)
        eta_grad_fn = jax.grad(eta_fn)
        e = current_etas
        for _ in range(inner_steps):
            g = eta_grad_fn(e)
            # Clip gradients to prevent explosion
            g_norm = jnp.linalg.norm(g)
            max_grad = 10.0
            g = jnp.where(g_norm > max_grad, g * max_grad / g_norm, g)
            e = e - lr_eta * g
        return e

    # --- Helper: full objective (optimise etas then evaluate) ---
    def _full_objective(fixed_arr, current_etas):
        """Optimise etas, then return FOCE objective."""
        e = _optimise_etas(fixed_arr, current_etas)
        fp = _array_to_params(param_names, fixed_arr)
        obj = foce_objective(fp, e, omega, sigma, model_func, data)
        return float(obj), e

    # Build initial simplex: (n_params + 1) vertices
    simplex = jnp.zeros((n_params + 1, n_params))
    simplex = simplex.at[0].set(param_arr)
    for i in range(n_params):
        delta = jnp.zeros(n_params)
        # Keep a meaningful minimum perturbation for small-magnitude parameters.
        scale = max(abs(float(param_arr[i])) * simplex_scale, simplex_scale)
        delta = delta.at[i].set(scale)
        simplex = simplex.at[i + 1].set(param_arr + delta)

    # Evaluate initial simplex
    f_values = jnp.zeros(n_params + 1)
    etas_cache = [etas] * (n_params + 1)
    for i in range(n_params + 1):
        obj_val, new_etas = _full_objective(simplex[i], etas)
        f_values = f_values.at[i].set(obj_val)
        etas_cache[i] = new_etas

    converged = False
    iteration = 0

    for iteration in range(1, maxiter + 1):
        # Sort by objective
        order = jnp.argsort(f_values)
        simplex = simplex[order]
        f_values = f_values[order]
        etas_cache = [etas_cache[int(order[j])] for j in range(n_params + 1)]

        # Use the best etas as the reference for inner optimisation
        best_etas = etas_cache[0]

        # Check convergence: range of f_values
        f_range = float(f_values[-1] - f_values[0])
        if f_range < tol and iteration > 1:
            converged = True
            break

        # Nelder-Mead coefficients
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_shrink = 0.5

        worst = simplex[-1]
        centroid = jnp.mean(simplex[:-1], axis=0)
        f_best = float(f_values[0])
        f_worst = float(f_values[-1])
        f_second_worst = float(f_values[-2])

        # Reflection
        x_r = centroid + alpha * (centroid - worst)
        f_r, etas_r = _full_objective(x_r, best_etas)

        if f_r < f_second_worst and f_r >= f_best:
            simplex = simplex.at[-1].set(x_r)
            f_values = f_values.at[-1].set(f_r)
            etas_cache[-1] = etas_r
            continue

        if f_r < f_best:
            # Expansion
            x_e = centroid + gamma * (x_r - centroid)
            f_e, etas_e = _full_objective(x_e, best_etas)
            if f_e < f_r:
                simplex = simplex.at[-1].set(x_e)
                f_values = f_values.at[-1].set(f_e)
                etas_cache[-1] = etas_e
            else:
                simplex = simplex.at[-1].set(x_r)
                f_values = f_values.at[-1].set(f_r)
                etas_cache[-1] = etas_r
            continue

        # Contraction
        if f_r < f_worst:
            # Outside contraction
            x_c = centroid + rho * (x_r - centroid)
            f_c, etas_c = _full_objective(x_c, best_etas)
            if f_c <= f_r:
                simplex = simplex.at[-1].set(x_c)
                f_values = f_values.at[-1].set(f_c)
                etas_cache[-1] = etas_c
                continue
        else:
            # Inside contraction
            x_c = centroid - rho * (centroid - worst)
            f_c, etas_c = _full_objective(x_c, best_etas)
            if f_c < f_worst:
                simplex = simplex.at[-1].set(x_c)
                f_values = f_values.at[-1].set(f_c)
                etas_cache[-1] = etas_c
                continue

        # Shrink toward best
        best_vertex = simplex[0]
        for i in range(1, n_params + 1):
            new_vertex = best_vertex + sigma_shrink * (simplex[i] - best_vertex)
            simplex = simplex.at[i].set(new_vertex)
            f_val, new_e = _full_objective(new_vertex, best_etas)
            f_values = f_values.at[i].set(f_val)
            etas_cache[i] = new_e

    # Final: pick the best vertex
    best_idx = int(jnp.argmin(f_values))
    final_arr = simplex[best_idx]
    final_etas = etas_cache[best_idx]
    final_params = _array_to_params(param_names, final_arr)
    final_obj = float(
        foce_objective(final_params, final_etas, omega, sigma, model_func, data)
    )

    return EstimationResult(
        fixed_params=final_params,
        etas=final_etas,
        objective=final_obj,
        n_iterations=iteration,
        converged=converged,
    )


# ---------------------------------------------------------------------------
# NLME (Nonlinear Mixed Effects via linearization) estimator
# ---------------------------------------------------------------------------

def estimate_nlme(
    model_func: Callable,
    data: Dict[str, jax.Array],
    ini_values: Dict[str, float],
    omega: jax.Array,
    control: Optional[Dict[str, Any]] = None,
) -> EstimationResult:
    """Run NLME estimation via iterative linearization.

    At each iteration the nonlinear model is linearized around the current
    parameter estimates (fixed effects + individual etas).  The resulting
    linear mixed-effects problem is solved with a GLS-like update for the
    fixed effects and a posterior-mode update for the random effects.  Omega
    is re-estimated from the empirical covariance of the etas.

    Parameters
    ----------
    model_func : callable
        ``(params_dict, times) -> predictions``.
    data : dict
        Must contain ``"id"``, ``"time"``, ``"dv"``.
    ini_values : dict
        Starting values for fixed-effect parameters.
    omega : array
        Between-subject covariance matrix.
    control : dict, optional
        Optimisation settings:
        - ``maxiter`` (int, default 100): max outer iterations.
        - ``tol`` (float, default 1e-4): convergence tolerance on
          relative change in the fixed-effect parameter vector.
        - ``sigma`` (float, default 1.0): residual error variance.

    Returns
    -------
    EstimationResult
    """
    ctrl = dict(control or {})
    maxiter = int(ctrl.get("maxiter", 100))
    tol = float(ctrl.get("tol", 1e-4))
    sigma = float(ctrl.get("sigma", 1.0))

    ids = data["id"]
    times = data["time"]
    dv = data["dv"]

    unique_ids = jnp.unique(ids)
    n_subjects = int(unique_ids.shape[0])
    n_etas = omega.shape[0]

    param_names, param_arr = _params_to_array(ini_values)
    n_params = len(param_names)

    # Current estimates
    beta = param_arr  # fixed effects vector
    etas = jnp.zeros((n_subjects, n_etas))
    omega_est = omega.copy()

    converged = False
    iteration = 0

    for iteration in range(1, maxiter + 1):
        beta_old = beta
        fixed_params = _array_to_params(param_names, beta)

        # Per-subject: compute predictions, Jacobians, and solve linearized problem
        # Accumulate GLS normal-equation components
        # H = sum_i J_i^T R_i^{-1} J_i + prior_precision  (for fixed effects)
        # g = sum_i J_i^T R_i^{-1} (dv_i - f_i + J_i @ beta)
        # For etas: posterior mode given current fixed effects

        omega_inv = jnp.linalg.inv(omega_est + jnp.eye(n_etas) * 1e-8)

        # Accumulate normal equations for beta update
        JtRiJ = jnp.zeros((n_params, n_params))
        JtRi_r = jnp.zeros(n_params)

        new_etas = jnp.zeros_like(etas)

        for i in range(n_subjects):
            subj_id = unique_ids[i]
            mask = ids == subj_id
            subj_times = times[mask]
            subj_dv = dv[mask]
            eta_i = etas[i]

            # Individual parameters: fixed + eta
            indiv_params = {}
            for j, name in enumerate(param_names):
                if j < n_etas:
                    indiv_params[name] = fixed_params[name] + float(eta_i[j])
                else:
                    indiv_params[name] = fixed_params[name]

            # Predictions at current individual parameters
            pred_i = model_func(indiv_params, subj_times)

            # Jacobian of predictions w.r.t. fixed effects (beta)
            def _pred_from_beta(b, _eta=eta_i, _st=subj_times):
                p = {}
                for j2, name2 in enumerate(param_names):
                    if j2 < n_etas:
                        p[name2] = b[j2] + _eta[j2]
                    else:
                        p[name2] = b[j2]
                return model_func(p, _st)

            # J_beta shape: (n_obs_i, n_params)
            J_beta = jax.jacobian(_pred_from_beta)(beta)

            # Jacobian w.r.t. eta for posterior mode
            def _pred_from_eta(e, _b=beta, _st=subj_times):
                p = {}
                for j2, name2 in enumerate(param_names):
                    if j2 < n_etas:
                        p[name2] = _b[j2] + e[j2]
                    else:
                        p[name2] = _b[j2]
                return model_func(p, _st)

            # J_eta shape: (n_obs_i, n_etas)
            J_eta = jax.jacobian(_pred_from_eta)(eta_i)

            # Residuals
            resid_i = subj_dv - pred_i  # (n_obs_i,)

            # R_inv = (1/sigma) * I
            ri_inv = 1.0 / sigma

            # Accumulate GLS for beta: J_beta^T R_i^{-1} J_beta
            JtRiJ = JtRiJ + ri_inv * (J_beta.T @ J_beta)
            # Linearized residual for beta update:
            # r_lin = resid_i + J_beta @ (beta - beta)  -- but beta=beta so just resid
            JtRi_r = JtRi_r + ri_inv * (J_beta.T @ resid_i)

            # Posterior mode for eta_i:
            # Minimize 0.5 * (resid - J_eta @ delta_eta)^T R^{-1} (resid - J_eta @ delta_eta)
            #        + 0.5 * (eta_i + delta_eta)^T Omega^{-1} (eta_i + delta_eta)
            # => (J_eta^T R^{-1} J_eta + Omega^{-1}) delta_eta = J_eta^T R^{-1} resid - Omega^{-1} eta_i
            H_eta = ri_inv * (J_eta.T @ J_eta) + omega_inv
            g_eta = ri_inv * (J_eta.T @ resid_i) - omega_inv @ eta_i
            delta_eta = jnp.linalg.solve(H_eta + jnp.eye(n_etas) * 1e-8, g_eta)
            new_etas = new_etas.at[i].set(eta_i + delta_eta)

        # GLS update for beta: delta_beta = (J^T R^{-1} J)^{-1} J^T R^{-1} r
        delta_beta = jnp.linalg.solve(
            JtRiJ + jnp.eye(n_params) * 1e-8, JtRi_r
        )
        beta = beta + delta_beta
        etas = new_etas

        # Update omega from empirical eta covariance
        if n_subjects > 1:
            eta_centered = etas - jnp.mean(etas, axis=0, keepdims=True)
            omega_est = (eta_centered.T @ eta_centered) / n_subjects
            # Ensure positive-definite
            omega_est = omega_est + jnp.eye(n_etas) * 1e-6
        else:
            # With one subject, keep omega as-is but add etas contribution
            omega_est = jnp.diag(jnp.maximum(etas[0] ** 2, 1e-6))

        # Check convergence
        param_change = float(jnp.linalg.norm(beta - beta_old))
        param_scale = float(jnp.linalg.norm(beta_old)) + 1e-8
        if param_change / param_scale < tol:
            converged = True
            break

    # Final objective: FOCE -2LL
    final_params = _array_to_params(param_names, beta)
    final_obj = float(
        foce_objective(final_params, etas, omega_est, sigma, model_func, data)
    )

    return EstimationResult(
        fixed_params=final_params,
        etas=etas,
        objective=final_obj,
        n_iterations=iteration,
        converged=converged,
    )
