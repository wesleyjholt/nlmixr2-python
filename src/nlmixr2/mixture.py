"""Mixture (latent class) model support for pharmacometric modeling.

Subjects may belong to unknown sub-populations (classes).  This module
provides:

- ``MixtureSpec`` – specification of a mixture model.
- ``mixture_log_likelihood`` – marginal log-likelihood over classes.
- ``classify_subjects`` – posterior class assignment via Bayes' theorem.
- ``estimate_mixture`` – EM algorithm to fit a mixture model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class MixtureSpec:
    """Specification of a finite mixture model.

    Parameters
    ----------
    n_classes : int
        Number of mixture components.
    class_params : dict
        Mapping from class index to parameter override dict.
    mixing_probs : tuple of float
        Prior class probabilities; must sum to 1.
    """

    n_classes: int
    class_params: Dict[int, Dict[str, Any]]
    mixing_probs: Tuple[float, ...]

    def __post_init__(self) -> None:
        if abs(sum(self.mixing_probs) - 1.0) > 1e-8:
            raise ValueError(
                f"mixing_probs must sum to 1, got {sum(self.mixing_probs)}"
            )


@dataclass
class MixtureResult:
    """Container for mixture model estimation output.

    Attributes
    ----------
    params_per_class : list of dict
        Estimated parameters for each class.
    mixing_probs : tuple of float
        Estimated mixing probabilities.
    classifications : dict
        Mapping subject_id -> (most_likely_class, posterior_probs).
    objective : float
        Final marginal log-likelihood.
    n_iterations : int
        Number of EM iterations performed.
    converged : bool
        Whether the EM algorithm satisfied the convergence criterion.
    """

    params_per_class: List[Dict[str, float]]
    mixing_probs: Tuple[float, ...]
    classifications: Dict[Any, Tuple[int, Tuple[float, ...]]]
    objective: float
    n_iterations: int
    converged: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normal_log_likelihood_obs(dv: np.ndarray, pred: np.ndarray, sigma: float) -> np.ndarray:
    """Per-observation normal log-likelihood."""
    return -0.5 * (np.log(2.0 * np.pi * sigma) + (dv - pred) ** 2 / sigma)


def _logsumexp(a: np.ndarray) -> float:
    """Numerically stable log-sum-exp over a 1-D array."""
    a_max = np.max(a)
    return float(a_max + np.log(np.sum(np.exp(a - a_max))))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mixture_log_likelihood(
    dv: np.ndarray,
    pred_per_class: List[np.ndarray],
    sigma: float,
    mixing_probs: Tuple[float, ...],
) -> float:
    """Compute the marginal mixture log-likelihood.

    .. math::

        \\ell = \\sum_j \\log \\sum_k \\pi_k \\, L_k(y_j \\mid \\theta_k)

    Uses the log-sum-exp trick for numerical stability.

    Parameters
    ----------
    dv : array
        Observed dependent variable.
    pred_per_class : list of arrays
        Predicted values for each class.
    sigma : float
        Residual error variance (common across classes).
    mixing_probs : tuple of float
        Class probabilities.

    Returns
    -------
    float
        Marginal log-likelihood.
    """
    dv = np.asarray(dv, dtype=float)
    n_obs = dv.shape[0]
    n_classes = len(pred_per_class)

    total_ll = 0.0
    for j in range(n_obs):
        # log(pi_k * L_k) for each class at observation j
        log_terms = np.empty(n_classes)
        for k in range(n_classes):
            ll_obs = _normal_log_likelihood_obs(
                dv[j:j+1],
                np.asarray(pred_per_class[k])[j:j+1],
                sigma,
            )[0]
            log_terms[k] = np.log(mixing_probs[k]) + ll_obs
        total_ll += _logsumexp(log_terms)

    return float(total_ll)


def classify_subjects(
    data: Dict[str, np.ndarray],
    model_func: Callable,
    params_per_class: List[Dict[str, float]],
    mixing_probs: Tuple[float, ...],
    sigma: float,
) -> Dict[Any, Tuple[int, Tuple[float, ...]]]:
    """Classify subjects into mixture components via Bayes' theorem.

    For each subject, computes the posterior probability of belonging to
    each class, and returns the MAP assignment.

    Parameters
    ----------
    data : dict
        Must contain ``"id"``, ``"time"``, ``"dv"`` arrays.
    model_func : callable
        ``(params_dict, times) -> predictions`` array.
    params_per_class : list of dict
        Parameters for each class.
    mixing_probs : tuple of float
        Prior class probabilities.
    sigma : float
        Residual error variance.

    Returns
    -------
    dict
        Mapping ``subject_id -> (most_likely_class, posterior_probs)``.
    """
    ids = np.asarray(data["id"])
    times = np.asarray(data["time"])
    dv = np.asarray(data["dv"])
    unique_ids = np.unique(ids)
    n_classes = len(params_per_class)

    result: Dict[Any, Tuple[int, Tuple[float, ...]]] = {}

    for subj_id in unique_ids:
        mask = ids == subj_id
        subj_times = times[mask]
        subj_dv = dv[mask]

        # Log-likelihood for each class
        log_posts = np.empty(n_classes)
        for k in range(n_classes):
            pred_k = np.asarray(model_func(params_per_class[k], subj_times))
            ll_k = np.sum(_normal_log_likelihood_obs(subj_dv, pred_k, sigma))
            log_posts[k] = np.log(mixing_probs[k]) + ll_k

        # Normalise via log-sum-exp to get posteriors
        log_norm = _logsumexp(log_posts)
        posteriors = tuple(float(np.exp(lp - log_norm)) for lp in log_posts)
        best_class = int(np.argmax(posteriors))
        result[subj_id] = (best_class, posteriors)

    return result


def estimate_mixture(
    model_func: Callable,
    data: Dict[str, np.ndarray],
    ini_values: Dict[str, float],
    n_classes: int = 2,
    control: Optional[Dict[str, Any]] = None,
) -> MixtureResult:
    """Fit a mixture model using the EM algorithm.

    E-step: compute posterior class probabilities for each subject.
    M-step: update class-specific parameters and mixing probabilities.

    Parameters
    ----------
    model_func : callable
        ``(params_dict, times) -> predictions`` array.
    data : dict
        Must contain ``"id"``, ``"time"``, ``"dv"`` arrays.
    ini_values : dict
        Initial population parameter values.
    n_classes : int
        Number of mixture components (default 2).
    control : dict, optional
        Algorithm settings: ``maxiter``, ``tol``, ``sigma``.

    Returns
    -------
    MixtureResult
    """
    ctrl = control or {}
    maxiter = int(ctrl.get("maxiter", 100))
    tol = float(ctrl.get("tol", 1e-4))
    sigma = float(ctrl.get("sigma", 1.0))

    ids = np.asarray(data["id"])
    times = np.asarray(data["time"])
    dv = np.asarray(data["dv"])
    unique_ids = np.unique(ids)
    n_subjects = len(unique_ids)
    param_names = list(ini_values.keys())

    # Initialise class params with small perturbations
    rng = np.random.default_rng(1234)
    params_per_class: List[Dict[str, float]] = []
    for k in range(n_classes):
        p = {}
        for name in param_names:
            # Spread initial values to help separation
            offset = (k - (n_classes - 1) / 2.0) * 0.5 * abs(ini_values[name])
            p[name] = ini_values[name] + offset + rng.normal(0, 0.01)
        params_per_class.append(p)

    mixing_probs: Tuple[float, ...] = tuple(1.0 / n_classes for _ in range(n_classes))

    # Pre-extract per-subject data
    subj_data = []
    for subj_id in unique_ids:
        mask = ids == subj_id
        subj_data.append((subj_id, times[mask], dv[mask]))

    prev_ll = -np.inf
    converged = False
    n_iter = 0

    for iteration in range(maxiter):
        n_iter = iteration + 1

        # --- E-step: compute posterior class probabilities per subject ---
        # responsibilities[i, k] = P(class k | subject i, params)
        responsibilities = np.zeros((n_subjects, n_classes))
        for i, (_, s_times, s_dv) in enumerate(subj_data):
            log_posts = np.empty(n_classes)
            for k in range(n_classes):
                pred_k = np.asarray(model_func(params_per_class[k], s_times))
                ll_k = np.sum(_normal_log_likelihood_obs(s_dv, pred_k, sigma))
                log_posts[k] = np.log(mixing_probs[k]) + ll_k
            log_norm = _logsumexp(log_posts)
            responsibilities[i] = np.exp(log_posts - log_norm)

        # --- M-step: update mixing probabilities ---
        nk = responsibilities.sum(axis=0)  # effective class sizes
        mixing_probs = tuple(float(nk[k] / n_subjects) for k in range(n_classes))

        # --- M-step: update class-specific parameters ---
        for k in range(n_classes):
            weights_k = responsibilities[:, k]

            def _neg_weighted_ll(param_vec: np.ndarray, class_k: int = k, w: np.ndarray = weights_k) -> float:
                params = {name: float(param_vec[j]) for j, name in enumerate(param_names)}
                total = 0.0
                for i, (_, s_times, s_dv) in enumerate(subj_data):
                    pred = np.asarray(model_func(params, s_times))
                    ll_i = np.sum(_normal_log_likelihood_obs(s_dv, pred, sigma))
                    total += w[i] * ll_i
                return -total

            x0 = np.array([params_per_class[k][name] for name in param_names])
            res = minimize(_neg_weighted_ll, x0, method="Nelder-Mead",
                           options={"maxiter": 200, "xatol": 1e-6, "fatol": 1e-6})
            for j, name in enumerate(param_names):
                params_per_class[k][name] = float(res.x[j])

        # --- Compute marginal log-likelihood ---
        current_ll = 0.0
        for i, (_, s_times, s_dv) in enumerate(subj_data):
            log_terms = np.empty(n_classes)
            for k in range(n_classes):
                pred_k = np.asarray(model_func(params_per_class[k], s_times))
                ll_k = np.sum(_normal_log_likelihood_obs(s_dv, pred_k, sigma))
                log_terms[k] = np.log(mixing_probs[k]) + ll_k
            current_ll += _logsumexp(log_terms)

        # Check convergence
        if abs(current_ll - prev_ll) < tol * (abs(prev_ll) + 1e-10):
            converged = True
            break
        prev_ll = current_ll

    # Final classification
    classifications = classify_subjects(
        data, model_func, params_per_class, mixing_probs, sigma,
    )

    return MixtureResult(
        params_per_class=params_per_class,
        mixing_probs=mixing_probs,
        classifications=classifications,
        objective=float(current_ll),
        n_iterations=n_iter,
        converged=converged,
    )
