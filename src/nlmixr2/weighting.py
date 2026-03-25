"""Observation-level likelihood weighting for NLME objectives.

Provides per-observation weighting schemes that can be composed with
any objective function (FOCE, FOCEi, Laplacian, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Union

import jax
import jax.numpy as jnp


@dataclass
class WeightingScheme:
    """Container for observation-level weights.

    Attributes
    ----------
    weights : array or callable
        Either a JAX array of per-observation weights, or a callable
        ``(dv, pred) -> weights`` that computes weights dynamically.
    """

    weights: Union[jax.Array, Callable]


def apply_weights(residuals: jax.Array, weights: jax.Array) -> jax.Array:
    """Apply observation-level weights to residuals.

    Weighted residuals are ``residuals * sqrt(weights)`` so that the
    sum of squared weighted residuals equals
    ``sum(weights * residuals^2)``.

    Parameters
    ----------
    residuals : array
        Raw residuals (y - f).
    weights : array
        Per-observation weights (same length as *residuals*).

    Returns
    -------
    jax.Array
        Weighted residuals.
    """
    return residuals * jnp.sqrt(weights)


def inverse_variance_weights(
    pred: jax.Array, a: float, b: float
) -> jax.Array:
    """Compute inverse-variance weights common in PK modelling.

    The variance model is ``Var = (a + b * pred)^2``, so the weights
    are ``1 / (a + b * pred)^2``.

    Parameters
    ----------
    pred : array
        Model predictions.
    a : float
        Additive error coefficient.
    b : float
        Proportional error coefficient.

    Returns
    -------
    jax.Array
        Per-observation weights.
    """
    return 1.0 / (a + b * pred) ** 2


def weighted_objective(
    objective_fn: Callable,
    weights: Union[jax.Array, Callable],
) -> Callable:
    """Wrap an objective function with per-observation weights.

    The wrapper multiplies residuals (``dv - pred``) by ``sqrt(weights)``
    before they enter the objective.  This is achieved by replacing
    ``data["dv"]`` with adjusted values such that the squared residual
    term in the objective becomes ``sum(w_i * (y_i - f_i)^2)``.

    Parameters
    ----------
    objective_fn : callable
        Original objective with signature
        ``(fixed_params, etas, omega, sigma, model_func, data) -> scalar``.
    weights : array or callable
        Per-observation weight array, or ``(dv, pred) -> weights``.

    Returns
    -------
    callable
        New objective function with the same signature.
    """

    def _weighted(fixed_params, etas, omega, sigma, model_func, data):
        # Compute predictions to resolve callable weights and to build
        # the weight-adjusted DV.  We use a simple population prediction
        # (eta=0 for first subject) to evaluate callable weights.  For
        # array weights this is a no-op lookup.
        if callable(weights):
            pred = model_func(fixed_params, data["time"])
            w = weights(data["dv"], pred)
        else:
            w = weights

        # Build a weight-adjusted model: wrap model_func so that
        # residuals are automatically scaled.
        def weighted_model(params, times):
            return model_func(params, times)

        # Scale dv so that (dv_adj - f)^2 = w * (dv - f)^2
        # dv_adj = f + sqrt(w) * (dv - f)
        # But f depends on individual params (etas) which we don't have
        # here.  Instead, scale the objective by modifying sigma:
        # sum((dv-f)^2 / sigma) with weights becomes
        # sum(w*(dv-f)^2 / sigma).
        #
        # The cleanest approach: wrap the model to return sqrt(w)*pred
        # and adjust dv to sqrt(w)*dv, so residuals become
        # sqrt(w)*(dv-pred).

        sqrt_w = jnp.sqrt(w)

        def scaled_model(params, times):
            return sqrt_w * model_func(params, times)

        scaled_data = dict(data)
        scaled_data["dv"] = sqrt_w * data["dv"]

        return objective_fn(fixed_params, etas, omega, sigma, scaled_model, scaled_data)

    return _weighted
