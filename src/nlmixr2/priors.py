"""Prior specification for Bayesian/MAP estimation.

Provides prior distribution classes and utilities for computing
log-prior contributions and MAP objective functions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


class Prior:
    """Base class for prior distributions.

    Subclasses must implement ``log_density(value)`` returning the
    log probability density evaluated at *value*.
    """

    def log_density(self, value: Any) -> float:
        raise NotImplementedError("Subclasses must implement log_density")


@dataclass(frozen=True)
class NormalPrior(Prior):
    """Normal (Gaussian) prior N(mean, sd^2).

    Parameters
    ----------
    mean : float
        Prior mean.
    sd : float
        Prior standard deviation (> 0).
    """

    mean: float
    sd: float

    def log_density(self, value: float) -> float:
        return -0.5 * math.log(2 * math.pi) - math.log(self.sd) - 0.5 * ((value - self.mean) / self.sd) ** 2


@dataclass(frozen=True)
class LogNormalPrior(Prior):
    """Log-normal prior.  If X ~ LogNormal(meanlog, sdlog) then log(X) ~ N(meanlog, sdlog^2).

    Parameters
    ----------
    meanlog : float
        Mean of log(X).
    sdlog : float
        Standard deviation of log(X) (> 0).
    """

    meanlog: float
    sdlog: float

    def log_density(self, value: float) -> float:
        if value <= 0.0:
            return -math.inf
        log_val = math.log(value)
        return (
            -0.5 * math.log(2 * math.pi)
            - math.log(self.sdlog)
            - math.log(value)
            - 0.5 * ((log_val - self.meanlog) / self.sdlog) ** 2
        )


@dataclass(frozen=True)
class UniformPrior(Prior):
    """Uniform prior on [lower, upper].

    Parameters
    ----------
    lower : float
        Lower bound.
    upper : float
        Upper bound (must be > lower).
    """

    lower: float
    upper: float

    def log_density(self, value: float) -> float:
        if value < self.lower or value > self.upper:
            return -math.inf
        return -math.log(self.upper - self.lower)


@dataclass(frozen=True)
class HalfNormalPrior(Prior):
    """Half-normal prior for non-negative parameters (e.g. variance components).

    The density is proportional to N(0, sd^2) restricted to [0, inf).

    Parameters
    ----------
    sd : float
        Scale parameter (> 0).
    """

    sd: float

    def log_density(self, value: float) -> float:
        if value < 0.0:
            return -math.inf
        # log density of half-normal: log(2) + log(N(0, sd^2))
        return (
            math.log(2.0)
            - 0.5 * math.log(2 * math.pi)
            - math.log(self.sd)
            - 0.5 * (value / self.sd) ** 2
        )


@dataclass(frozen=True)
class InverseWishartPrior(Prior):
    """Inverse-Wishart prior for covariance (omega) matrices.

    Parameters
    ----------
    df : int
        Degrees of freedom (must be > p - 1 where p is the matrix dimension).
    scale : array-like, shape (p, p)
        Scale matrix (positive definite).
    """

    df: int
    scale: Any  # numpy array

    def log_density(self, value: Any) -> float:
        """Evaluate the log density of the Inverse-Wishart distribution.

        Parameters
        ----------
        value : array-like, shape (p, p)
            A symmetric positive-definite matrix.
        """
        X = np.asarray(value, dtype=float)
        S = np.asarray(self.scale, dtype=float)
        p = X.shape[0]
        nu = self.df

        # Check positive definiteness via Cholesky
        try:
            np.linalg.cholesky(X)
        except np.linalg.LinAlgError:
            return -math.inf

        # log|X|
        sign, log_det_X = np.linalg.slogdet(X)
        if sign <= 0:
            return -math.inf

        # log|S|
        sign_s, log_det_S = np.linalg.slogdet(S)

        # tr(S @ X^{-1})
        X_inv = np.linalg.inv(X)
        trace_term = np.trace(S @ X_inv)

        # Multivariate log-gamma
        log_gamma_p = (p * (p - 1) / 4.0) * math.log(math.pi)
        for j in range(1, p + 1):
            log_gamma_p += math.lgamma(0.5 * (nu + 1 - j))

        log_density = (
            0.5 * nu * log_det_S
            - 0.5 * (nu + p + 1) * log_det_X
            - 0.5 * trace_term
            - 0.5 * nu * p * math.log(2.0)
            - log_gamma_p
        )

        return float(log_density)


@dataclass
class PriorSpec:
    """Collection of priors for named parameters.

    Attributes
    ----------
    priors : dict[str, Prior]
        Mapping from parameter name to Prior object.
    """

    priors: Dict[str, Prior] = field(default_factory=dict)


def compute_prior_contribution(prior_spec: PriorSpec, params: Dict[str, float]) -> float:
    """Compute the total log-prior density for all parameters in *prior_spec*.

    Parameters that appear in *params* but not in *prior_spec* are ignored.
    Parameters in *prior_spec* that are missing from *params* are skipped.

    Returns
    -------
    float
        Sum of log-prior densities.  Returns 0.0 if the spec is empty.
    """
    if not prior_spec.priors:
        return 0.0

    total = 0.0
    for name, prior in prior_spec.priors.items():
        if name in params:
            total += prior.log_density(params[name])
    return total


def map_objective(
    base_objective: float,
    prior_spec: PriorSpec,
    params: Dict[str, float],
) -> float:
    """Compute the MAP objective function.

    MAP objective = base_objective - 2 * log_prior

    The base objective is typically the -2 log-likelihood from a
    frequentist estimator (e.g. FOCE).  Subtracting 2 * log_prior
    penalises parameter values that deviate from the prior, yielding
    the maximum a posteriori (MAP) estimate when minimised.

    Parameters
    ----------
    base_objective : float
        The base -2LL objective value.
    prior_spec : PriorSpec
        Prior specification.
    params : dict[str, float]
        Current parameter values.

    Returns
    -------
    float
        MAP objective value.
    """
    log_prior = compute_prior_contribution(prior_spec, params)
    return base_objective - 2.0 * log_prior
