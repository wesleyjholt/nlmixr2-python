"""Sensitivity equations for gradient-based optimization of ODE model parameters.

Provides forward sensitivity analysis (augmented ODE system) and adjoint-method
gradients via JAX autodiff through diffrax solvers.  Also includes Fisher
Information Matrix computation from parameter sensitivities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import diffrax
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class SensitivityResult:
    """Container for ODE solution with parameter sensitivities.

    Attributes
    ----------
    states : jnp.ndarray
        ODE state trajectory, shape ``(n_times, n_states)``.
    sensitivities : jnp.ndarray
        Parameter sensitivities ``dy_i/dp_j`` at each saved time,
        shape ``(n_times, n_states, n_params)``.
    times : jnp.ndarray
        Time points at which the solution was saved, shape ``(n_times,)``.
    """

    states: jnp.ndarray
    sensitivities: jnp.ndarray
    times: jnp.ndarray


# ---------------------------------------------------------------------------
# Forward sensitivity solver
# ---------------------------------------------------------------------------

def solve_with_sensitivities(
    rhs_func: Callable,
    t_span: Tuple[float, float],
    y0: jnp.ndarray,
    params: jnp.ndarray,
    t_eval: jnp.ndarray,
    dosing_events: List[Dict[str, Any]] | None = None,
) -> SensitivityResult:
    """Solve an ODE system augmented with forward sensitivity equations.

    The sensitivity matrix *S* satisfies ``dS/dt = (df/dy) S + df/dp``,
    where ``f`` is the right-hand side.  Jacobians ``df/dy`` and ``df/dp``
    are computed automatically using ``jax.jacobian``.

    Parameters
    ----------
    rhs_func : callable
        Right-hand side ``f(t, y, params) -> dy/dt``.  ``params`` must be
        a 1-D ``jnp.ndarray`` (not a dict) so that JAX can differentiate
        through it.
    t_span : (t0, tf)
        Integration interval.
    y0 : jnp.ndarray
        Initial state, shape ``(n_states,)``.
    params : jnp.ndarray
        Parameter vector, shape ``(n_params,)``.
    t_eval : jnp.ndarray
        Times at which to report the solution.
    dosing_events : list of dict, optional
        PK dosing events (bolus only for the sensitivity system).

    Returns
    -------
    SensitivityResult
    """
    params = jnp.asarray(params, dtype=jnp.float64)
    y0 = jnp.asarray(y0, dtype=jnp.float64)
    t_eval = jnp.asarray(t_eval, dtype=jnp.float64)

    n_states = y0.shape[0]
    n_params = params.shape[0]

    # Initial sensitivity matrix is zero (y0 does not depend on params)
    S0 = jnp.zeros((n_states, n_params), dtype=jnp.float64)

    # Flatten: augmented state = [y (n_states), S_flat (n_states*n_params)]
    aug_y0 = jnp.concatenate([y0, S0.ravel()])

    # Apply bolus dosing events to the augmented initial state
    if dosing_events:
        for ev in dosing_events:
            if ev.get("duration", 0.0) <= 0.0:
                cpt = ev["compartment"]
                amt = ev["amount"]
                aug_y0 = aug_y0.at[cpt].add(amt)

    # Jacobian functions (compiled once via JAX)
    # df/dy: (n_states,) -> (n_states,), differentiate w.r.t. y
    def _dfdy(t, y, p):
        return jax.jacobian(lambda y_: rhs_func(t, y_, p))(y)

    # df/dp: (n_params,) -> (n_states,), differentiate w.r.t. p
    def _dfdp(t, y, p):
        return jax.jacobian(lambda p_: rhs_func(t, y, p_))(p)

    def augmented_rhs(t, aug_y, p):
        y = aug_y[:n_states]
        S = aug_y[n_states:].reshape(n_states, n_params)

        dydt = rhs_func(t, y, p)
        dfdy = _dfdy(t, y, p)   # (n_states, n_states)
        dfdp = _dfdp(t, y, p)   # (n_states, n_params)

        dSdt = dfdy @ S + dfdp  # (n_states, n_params)

        return jnp.concatenate([dydt, dSdt.ravel()])

    # Solve the augmented system
    term = diffrax.ODETerm(lambda t, y, args: augmented_rhs(t, y, args))
    solver = diffrax.Dopri5()
    sc = diffrax.PIDController(rtol=1e-8, atol=1e-8)
    saveat = diffrax.SaveAt(ts=t_eval)

    t0, t1 = t_span
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=min(0.01, (t1 - t0) / 10.0),
        y0=aug_y0,
        args=params,
        saveat=saveat,
        stepsize_controller=sc,
        max_steps=100_000,
    )

    aug_ys = sol.ys  # (n_times, n_states + n_states*n_params)
    states = aug_ys[:, :n_states]
    sens_flat = aug_ys[:, n_states:]
    sensitivities = sens_flat.reshape(-1, n_states, n_params)

    return SensitivityResult(
        states=states,
        sensitivities=sensitivities,
        times=t_eval,
    )


# ---------------------------------------------------------------------------
# Fisher Information Matrix
# ---------------------------------------------------------------------------

def compute_fim(
    sensitivities: jnp.ndarray,
    sigma: float,
) -> jnp.ndarray:
    """Compute the Fisher Information Matrix from parameter sensitivities.

    Assumes the observation function is the identity (all states observed)
    with constant residual variance ``sigma^2``.

    ``FIM = sum_t  (dh/dp)^T  (1/sigma^2)  (dh/dp)``

    where ``dh/dp`` at time *t* has shape ``(n_states, n_params)``.

    Parameters
    ----------
    sensitivities : jnp.ndarray
        Shape ``(n_times, n_states, n_params)`` — typically from
        ``SensitivityResult.sensitivities``.
    sigma : float
        Standard deviation of the residual error.

    Returns
    -------
    jnp.ndarray
        Fisher Information Matrix, shape ``(n_params, n_params)``.
    """
    # sensitivities: (T, S, P)
    # For each time point: J_t^T @ J_t, then sum over time and scale
    # Efficient einsum: sum_t sum_s  sens[t,s,i] * sens[t,s,j] / sigma^2
    fim = jnp.einsum("tsi,tsj->ij", sensitivities, sensitivities) / (sigma ** 2)
    return fim


# ---------------------------------------------------------------------------
# Adjoint gradient via JAX autodiff
# ---------------------------------------------------------------------------

def jax_adjoint_gradient(
    objective_fn: Callable[[jnp.ndarray], float],
    rhs_func: Callable,
    params: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the gradient of an objective function w.r.t. parameters.

    Uses ``jax.grad`` which, when the objective contains a diffrax
    ``diffeqsolve`` call, automatically applies the adjoint method for
    reverse-mode differentiation through the ODE solve.

    Parameters
    ----------
    objective_fn : callable
        Scalar-valued function ``f(params) -> float``.  Should internally
        call ``diffrax.diffeqsolve`` so that JAX can differentiate through it.
    rhs_func : callable
        The ODE right-hand side (unused directly; included for API
        consistency — the caller embeds it in *objective_fn*).
    params : jnp.ndarray
        Parameter vector at which to evaluate the gradient.

    Returns
    -------
    jnp.ndarray
        Gradient array of same shape as *params*.
    """
    grad_fn = jax.grad(objective_fn)
    return grad_fn(params)
