"""Censoring / BLQ (below limit of quantification) support for pharmacometric data.

Implements the M3 method (Beal 2001) for handling censored observations
in nonlinear mixed-effects models, along with utilities for detecting and
extracting censoring information from datasets.

Censoring codes follow the NONMEM convention:
  - 0  = uncensored (normal observation)
  - 1  = left-censored (below LOQ / BLQ)
  - -1 = right-censored (above upper limit)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import jax.numpy as jnp
import jax.scipy.stats.norm as jax_norm


@dataclass
class CensoringSpec:
    """Specification for censoring columns in a dataset.

    Attributes
    ----------
    cens_column : str
        Name of the column containing censoring flags
        (0 = uncensored, 1 = left-censored, -1 = right-censored).
    limit_column : str
        Name of the column containing the censoring limit value (e.g. LOQ).
    """

    cens_column: str = "cens"
    limit_column: str = "limit"


def censored_normal_log_likelihood(
    dv: jnp.ndarray,
    pred: jnp.ndarray,
    sigma: float,
    cens: jnp.ndarray,
    limit: jnp.ndarray,
) -> jnp.ndarray:
    """Compute element-wise log-likelihood accounting for censoring.

    Parameters
    ----------
    dv : array
        Observed dependent variable values.
    pred : array
        Model-predicted values.
    sigma : float
        Residual standard deviation (not variance).
    cens : array
        Censoring flags: 0 = uncensored, 1 = left-censored, -1 = right-censored.
    limit : array
        Censoring limit values (e.g. LOQ for BLQ observations).

    Returns
    -------
    jnp.ndarray
        Log-likelihood for each observation.
    """
    # Standardised residual for censoring limit
    z_limit = (limit - pred) / sigma

    # Uncensored: standard normal log-likelihood
    ll_uncensored = (
        -0.5 * jnp.log(2.0 * jnp.pi * sigma ** 2)
        - 0.5 * ((dv - pred) / sigma) ** 2
    )

    # Left-censored (BLQ): log(Phi((limit - pred) / sigma))
    ll_left = jnp.log(jnp.clip(jax_norm.cdf(z_limit), min=1e-30))

    # Right-censored: log(1 - Phi((limit - pred) / sigma))
    ll_right = jnp.log(jnp.clip(1.0 - jax_norm.cdf(z_limit), min=1e-30))

    # Select based on censoring flag
    result = jnp.where(cens == 0, ll_uncensored, jnp.where(cens == 1, ll_left, ll_right))
    return result


def apply_censoring(
    data: Dict[str, jnp.ndarray],
    censoring_spec: CensoringSpec,
) -> Dict[str, jnp.ndarray]:
    """Extract censoring and limit arrays from a dataset.

    Parameters
    ----------
    data : dict
        Dataset with column-name keys and array values.
    censoring_spec : CensoringSpec
        Specifies which columns hold the censoring flag and limit.

    Returns
    -------
    dict
        ``{"cens": array, "limit": array}`` extracted (or defaulted to zeros).
    """
    # Determine number of rows from any column present
    n = None
    for v in data.values():
        n = jnp.asarray(v).shape[0]
        break

    if censoring_spec.cens_column in data:
        cens = jnp.asarray(data[censoring_spec.cens_column])
    else:
        cens = jnp.zeros(n, dtype=jnp.int32)

    if censoring_spec.limit_column in data:
        limit = jnp.asarray(data[censoring_spec.limit_column])
    else:
        limit = jnp.zeros(n, dtype=jnp.float32)

    return {"cens": cens, "limit": limit}


def has_censoring(data: Dict[str, jnp.ndarray]) -> bool:
    """Check whether a dataset contains any censored observations.

    Parameters
    ----------
    data : dict
        Dataset with column-name keys and array values.

    Returns
    -------
    bool
        True if a ``"cens"`` column exists and contains any non-zero values.
    """
    if "cens" not in data:
        return False
    cens = jnp.asarray(data["cens"])
    return bool(jnp.any(cens != 0))


def m3_method(
    dv: jnp.ndarray,
    pred: jnp.ndarray,
    sigma: float,
    cens: jnp.ndarray,
    limit: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the M3 method total log-likelihood (Beal 2001).

    Combines continuous (uncensored) and censored likelihood contributions
    into a single scalar log-likelihood value.

    For uncensored observations the standard normal log-likelihood is used.
    For left-censored observations (cens=1) the CDF-based likelihood is used.
    For right-censored observations (cens=-1) the survival-function-based
    likelihood is used.

    Parameters
    ----------
    dv : array
        Observed dependent variable values.
    pred : array
        Model-predicted values.
    sigma : float
        Residual standard deviation.
    cens : array
        Censoring flags (0, 1, -1).
    limit : array
        Censoring limit values.

    Returns
    -------
    jnp.ndarray
        Scalar total log-likelihood.
    """
    ll = censored_normal_log_likelihood(dv, pred, sigma, cens, limit)
    return jnp.sum(ll)
