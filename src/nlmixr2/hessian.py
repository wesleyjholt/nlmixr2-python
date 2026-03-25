"""Hessian computation and standard errors for the estimation pipeline.

Provides utilities to compute the parameter covariance matrix from
the Hessian of the objective function, along with derived quantities
(standard errors, correlation matrix, relative standard errors).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp


@dataclass
class CovarianceResult:
    """Container for covariance-step output.

    Attributes
    ----------
    hessian : jax.Array
        Hessian matrix of the objective function at the estimates.
    covariance : jax.Array
        Parameter covariance matrix (inverse of the Hessian).
    correlation : jax.Array
        Parameter correlation matrix.
    standard_errors : jax.Array
        Standard error per parameter (sqrt of diagonal of covariance).
    rse : jax.Array
        Relative standard error (%) per parameter.
    condition_number : float
        Condition number of the Hessian matrix.
    """

    hessian: jax.Array
    covariance: jax.Array
    correlation: jax.Array
    standard_errors: jax.Array
    rse: jax.Array
    condition_number: float


def compute_hessian(
    objective_fn: Callable[[jax.Array], jax.Array],
    params: jax.Array,
) -> jax.Array:
    """Compute the Hessian matrix of *objective_fn* at *params* using JAX.

    Parameters
    ----------
    objective_fn : callable
        A scalar-valued function ``f(params) -> scalar``.
    params : jax.Array
        Parameter vector at which to evaluate the Hessian.

    Returns
    -------
    jax.Array
        Hessian matrix, shape ``(len(params), len(params))``.
    """
    return jax.hessian(objective_fn)(params)


def compute_covariance(hessian: jax.Array) -> jax.Array:
    """Compute the parameter covariance matrix as the inverse of the Hessian.

    For singular or near-singular Hessians the result may contain ``inf``
    or ``nan`` values rather than raising an exception.

    Parameters
    ----------
    hessian : jax.Array
        Hessian matrix, shape ``(p, p)``.

    Returns
    -------
    jax.Array
        Covariance matrix, shape ``(p, p)``.
    """
    return jnp.linalg.inv(hessian)


def compute_standard_errors(covariance: jax.Array) -> jax.Array:
    """Compute per-parameter standard errors from a covariance matrix.

    Parameters
    ----------
    covariance : jax.Array
        Covariance matrix, shape ``(p, p)``.

    Returns
    -------
    jax.Array
        Standard errors, shape ``(p,)``.
    """
    return jnp.sqrt(jnp.diag(covariance))


def compute_correlation(covariance: jax.Array) -> jax.Array:
    """Compute the correlation matrix from a covariance matrix.

    Parameters
    ----------
    covariance : jax.Array
        Covariance matrix, shape ``(p, p)``.

    Returns
    -------
    jax.Array
        Correlation matrix, shape ``(p, p)``.
    """
    se = jnp.sqrt(jnp.diag(covariance))
    outer = jnp.outer(se, se)
    return covariance / outer


def compute_rse(
    standard_errors: jax.Array,
    estimates: jax.Array,
) -> jax.Array:
    """Compute relative standard errors as a percentage.

    ``RSE_i = (SE_i / |estimate_i|) * 100``.

    Parameters
    ----------
    standard_errors : jax.Array
        Standard errors, shape ``(p,)``.
    estimates : jax.Array
        Point estimates, shape ``(p,)``.

    Returns
    -------
    jax.Array
        Relative standard errors (%), shape ``(p,)``.
    """
    return (standard_errors / jnp.abs(estimates)) * 100.0
