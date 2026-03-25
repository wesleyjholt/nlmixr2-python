"""Adaptive optimizer utilities for nlmixr2 estimators.

Provides an Adam optimizer with proper state management, Armijo backtracking
line search, convergence monitoring, and parameter scaling helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Adam optimizer
# ---------------------------------------------------------------------------

@dataclass
class AdamState:
    """Mutable state for the Adam optimizer.

    Attributes
    ----------
    m : jnp.ndarray
        First moment estimate (momentum).
    v : jnp.ndarray
        Second moment estimate (velocity / uncentered variance).
    t : int
        Timestep counter (number of updates applied so far).
    """

    m: jnp.ndarray
    v: jnp.ndarray
    t: int


def adam_step(
    grad: jnp.ndarray,
    state: AdamState,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> Tuple[jnp.ndarray, AdamState]:
    """Compute a single Adam update.

    Parameters
    ----------
    grad : jnp.ndarray
        Gradient of the objective with respect to the parameters.
    state : AdamState
        Current optimizer state.
    lr : float
        Learning rate.
    beta1 : float
        Exponential decay rate for the first moment.
    beta2 : float
        Exponential decay rate for the second moment.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    update : jnp.ndarray
        The parameter update to *subtract* from the current parameters
        (i.e. ``params_new = params - update``).
    new_state : AdamState
        Updated optimizer state.
    """
    t = state.t + 1
    m = beta1 * state.m + (1 - beta1) * grad
    v = beta2 * state.v + (1 - beta2) * grad ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    update = lr * m_hat / (jnp.sqrt(v_hat) + eps)
    return update, AdamState(m=m, v=v, t=t)


# ---------------------------------------------------------------------------
# Line search
# ---------------------------------------------------------------------------

def line_search(
    objective_fn: Callable[[jnp.ndarray], float],
    params: jnp.ndarray,
    direction: jnp.ndarray,
    max_step: float = 1.0,
    c1: float = 1e-4,
    shrink: float = 0.5,
    min_step: float = 1e-12,
) -> float:
    """Armijo backtracking line search.

    Finds a step size ``alpha`` such that the *sufficient decrease*
    (Armijo) condition is satisfied::

        f(params + alpha * direction) <= f(params) + c1 * alpha * grad^T direction

    where ``grad^T direction`` is approximated by the directional derivative
    implied by *direction* (assumed to be a descent direction, e.g.
    ``-gradient``).

    Parameters
    ----------
    objective_fn : callable
        Scalar objective ``f(params) -> float``.
    params : jnp.ndarray
        Current parameter vector.
    direction : jnp.ndarray
        Search direction (should be a descent direction for convergence).
    max_step : float
        Initial (maximum) step size to try.
    c1 : float
        Sufficient decrease parameter (typically 1e-4).
    shrink : float
        Factor by which the step size is reduced on each iteration.
    min_step : float
        Minimum step size before giving up.

    Returns
    -------
    float
        The accepted step size.
    """
    f0 = objective_fn(params)
    # Directional derivative approximation: use a finite-difference
    # or, when direction = -grad, slope = -||grad||^2.  We compute it
    # via a small forward difference to stay generic.
    delta = 1e-7
    f_probe = objective_fn(params + delta * direction)
    slope = (f_probe - f0) / delta  # directional derivative

    alpha = max_step
    while alpha > min_step:
        f_new = objective_fn(params + alpha * direction)
        if f_new <= f0 + c1 * alpha * slope:
            return alpha
        alpha *= shrink
    return alpha  # return the smallest tried


# ---------------------------------------------------------------------------
# Convergence monitor
# ---------------------------------------------------------------------------

class ConvergenceMonitor:
    """Track objective values and detect convergence / divergence.

    Usage::

        mon = ConvergenceMonitor()
        for step in range(max_steps):
            obj = compute_objective(params)
            mon.update(obj, params)
            if mon.is_converged(tol=1e-4):
                break
            if mon.is_diverging():
                raise RuntimeError("Objective is diverging")
    """

    def __init__(self) -> None:
        self._history: List[Tuple[float, jnp.ndarray]] = []

    # -- public interface ---------------------------------------------------

    def update(self, objective: float, params: jnp.ndarray) -> None:
        """Record a new (objective, params) observation."""
        self._history.append((float(objective), params))

    def is_converged(self, tol: float, patience: int = 5) -> bool:
        """Return ``True`` if the objective change is below *tol* for the
        last *patience* consecutive steps.

        We measure absolute change between successive objectives.
        """
        if len(self._history) < patience + 1:
            return False
        recent = self._history[-(patience + 1):]
        for i in range(1, len(recent)):
            if abs(recent[i][0] - recent[i - 1][0]) >= tol:
                return False
        return True

    def is_diverging(self, max_increase: float = 10) -> bool:
        """Return ``True`` if the latest objective is more than
        *max_increase* times larger than the best objective seen so far.
        """
        if len(self._history) < 2:
            return False
        best = min(obj for obj, _ in self._history)
        latest = self._history[-1][0]
        return latest > max_increase * best

    @property
    def history(self) -> List[Tuple[float, jnp.ndarray]]:
        """List of ``(objective, params)`` tuples in recording order."""
        return list(self._history)


# ---------------------------------------------------------------------------
# Parameter scaling
# ---------------------------------------------------------------------------

def scale_parameters(params: jnp.ndarray, scales: jnp.ndarray) -> jnp.ndarray:
    """Normalize *params* by dividing element-wise by *scales*.

    This brings parameters of very different magnitudes to a similar
    scale, improving optimizer conditioning.
    """
    return params / scales


def unscale_parameters(normalized: jnp.ndarray, scales: jnp.ndarray) -> jnp.ndarray:
    """Recover original-scale parameters from *normalized* values."""
    return normalized * scales
