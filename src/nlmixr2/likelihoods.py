"""Non-Gaussian endpoint likelihoods for count and categorical data.

Provides per-observation log-likelihood functions for common pharmacometric
endpoint distributions: Poisson (count), negative binomial (overdispersed
count), binomial (binary/proportion), and ordinal categorical.

All functions accept and return JAX arrays, and are compatible with
``jax.grad`` for use in gradient-based estimation.
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
import jax.scipy.special as jsp

from nlmixr2.censoring import censored_normal_log_likelihood


# ---------------------------------------------------------------------------
# Poisson
# ---------------------------------------------------------------------------

def poisson_log_likelihood(
    dv: jnp.ndarray,
    lambda_pred: jnp.ndarray,
) -> jnp.ndarray:
    """Per-observation Poisson log-likelihood.

    log P(Y=k | lambda) = k * log(lambda) - lambda - log(k!)

    Parameters
    ----------
    dv : array
        Observed counts (non-negative integers, stored as floats).
    lambda_pred : array
        Predicted Poisson rate parameter (must be > 0).

    Returns
    -------
    jnp.ndarray
        Log-likelihood for each observation.
    """
    k = dv
    return k * jnp.log(lambda_pred) - lambda_pred - jsp.gammaln(k + 1.0)


# ---------------------------------------------------------------------------
# Negative binomial
# ---------------------------------------------------------------------------

def negative_binomial_log_likelihood(
    dv: jnp.ndarray,
    mu: jnp.ndarray,
    size: jnp.ndarray,
) -> jnp.ndarray:
    """Per-observation negative binomial log-likelihood.

    Parameterised by mean ``mu`` and dispersion ``size`` (often called ``r``
    or ``theta``).  As ``size -> inf``, the NB converges to Poisson(mu).

    log P(Y=k | mu, size) =
        gammaln(k + size) - gammaln(size) - gammaln(k + 1)
        + size * log(size / (size + mu))
        + k * log(mu / (size + mu))

    Parameters
    ----------
    dv : array
        Observed counts.
    mu : array
        Predicted mean (must be > 0).
    size : array
        Dispersion parameter (must be > 0).  Larger values mean less
        overdispersion.

    Returns
    -------
    jnp.ndarray
        Log-likelihood for each observation.
    """
    k = dv
    return (
        jsp.gammaln(k + size)
        - jsp.gammaln(size)
        - jsp.gammaln(k + 1.0)
        + size * jnp.log(size / (size + mu))
        + k * jnp.log(mu / (size + mu))
    )


# ---------------------------------------------------------------------------
# Binomial
# ---------------------------------------------------------------------------

def binomial_log_likelihood(
    dv: jnp.ndarray,
    n: jnp.ndarray,
    p: jnp.ndarray,
) -> jnp.ndarray:
    """Per-observation binomial log-likelihood.

    log P(Y=k | n, p) = log C(n,k) + k*log(p) + (n-k)*log(1-p)

    Boundary-safe: uses ``jnp.where`` to handle p=0 and p=1 without NaN.

    Parameters
    ----------
    dv : array
        Observed successes (0 <= dv <= n).
    n : array
        Number of trials.
    p : array
        Success probability (0 <= p <= 1).

    Returns
    -------
    jnp.ndarray
        Log-likelihood for each observation.
    """
    k = dv
    log_comb = jsp.gammaln(n + 1.0) - jsp.gammaln(k + 1.0) - jsp.gammaln(n - k + 1.0)

    # Safe log computation: avoid log(0) by clamping, then zero-out via where
    eps = 1e-30
    safe_log_p = jnp.log(jnp.clip(p, min=eps))
    safe_log_1mp = jnp.log(jnp.clip(1.0 - p, min=eps))

    # k * log(p): when p=0 and k=0, contribution should be 0
    term_p = jnp.where(k == 0, 0.0, k * safe_log_p)
    # (n-k) * log(1-p): when p=1 and k=n, contribution should be 0
    term_1mp = jnp.where(n - k == 0, 0.0, (n - k) * safe_log_1mp)

    return log_comb + term_p + term_1mp


# ---------------------------------------------------------------------------
# Ordinal categorical
# ---------------------------------------------------------------------------

def ordinal_log_likelihood(
    dv: jnp.ndarray,
    cumulative_probs: jnp.ndarray,
) -> jnp.ndarray:
    """Per-observation ordinal categorical log-likelihood.

    P(Y=k) = F(k) - F(k-1), where F are cumulative probabilities and
    F(-1) = 0 by convention.

    Parameters
    ----------
    dv : array, shape (n_obs,)
        Observed categories as integer indices (0-based).
    cumulative_probs : array, shape (n_obs, n_categories)
        Cumulative probabilities for each observation.  The last column
        should be 1.0.

    Returns
    -------
    jnp.ndarray
        Log-likelihood for each observation, shape (n_obs,).
    """
    n_obs = dv.shape[0]
    idx = dv.astype(jnp.int32)

    # F(k) for each observation
    f_k = cumulative_probs[jnp.arange(n_obs), idx]

    # F(k-1): 0 when k=0
    f_k_minus1 = jnp.where(
        idx == 0,
        0.0,
        cumulative_probs[jnp.arange(n_obs), idx - 1],
    )

    prob = jnp.clip(f_k - f_k_minus1, min=1e-30)
    return jnp.log(prob)


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------

def select_likelihood(family: str) -> Callable:
    """Return the log-likelihood function for a given distribution family.

    Parameters
    ----------
    family : str
        One of ``"normal"``, ``"poisson"``, ``"negbin"``, ``"binomial"``,
        ``"ordinal"``.

    Returns
    -------
    Callable
        The corresponding log-likelihood function.

    Raises
    ------
    ValueError
        If *family* is not recognised.
    """
    _registry = {
        "normal": censored_normal_log_likelihood,
        "poisson": poisson_log_likelihood,
        "negbin": negative_binomial_log_likelihood,
        "binomial": binomial_log_likelihood,
        "ordinal": ordinal_log_likelihood,
    }
    if family not in _registry:
        raise ValueError(
            f"Unknown likelihood family {family!r}. "
            f"Choose from: {', '.join(sorted(_registry))}"
        )
    return _registry[family]
