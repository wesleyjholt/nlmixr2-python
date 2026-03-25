"""Steady-state solving for pharmacokinetic models.

Provides functions to repeatedly apply doses until the system reaches
steady state, using either ODE simulation or analytical superposition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

import jax.numpy as jnp
import numpy as np

from .ode import solve_ode


@dataclass
class SteadyStateResult:
    """Result of a steady-state computation.

    Attributes
    ----------
    trough : float
        Steady-state trough (pre-dose) concentration.
    peak : float
        Steady-state peak concentration over one dosing interval.
    auc_ss : float
        AUC over one dosing interval at steady state (trapezoidal).
    n_doses : int
        Number of doses administered to reach steady state.
    accumulation_ratio : float
        Ratio of steady-state trough to first-dose trough (Css_trough / C1_trough).
    converged : bool
        Whether the convergence criterion was met.
    """

    trough: float
    peak: float
    auc_ss: float
    n_doses: int
    accumulation_ratio: float
    converged: bool


def find_steady_state(
    model_func: Callable,
    dose_event: Dict[str, Any],
    params: Dict[str, Any],
    tol: float = 1e-4,
    max_doses: int = 1000,
) -> SteadyStateResult:
    """Find steady state by repeatedly simulating dosing intervals via ODE.

    Parameters
    ----------
    model_func : callable
        ODE right-hand side ``f(t, y, params) -> dy/dt``.
    dose_event : dict
        Must contain ``amt``, ``ii``, ``cmt``.  Optionally ``dur`` for
        zero-order infusion (default 0 = bolus).
    params : dict
        Parameter dictionary forwarded to *model_func*.
    tol : float
        Relative convergence tolerance on trough concentration.
    max_doses : int
        Maximum number of dosing cycles before giving up.

    Returns
    -------
    SteadyStateResult
    """
    amt = dose_event["amt"]
    ii = dose_event["ii"]
    cmt = dose_event["cmt"]
    dur = dose_event.get("dur", 0.0)

    # Determine number of compartments by probing model_func
    # We start with a 1-element state; if the model returns more we adapt.
    # For safety, assume cmt index tells us the minimum size.
    n_cpt = cmt + 1

    state = jnp.zeros(n_cpt)
    n_eval = 50  # internal evaluation points per interval

    first_trough = None
    prev_trough = 0.0
    converged = False
    n_doses_applied = 0

    # Store last-interval profile for AUC/peak computation
    last_times = None
    last_conc = None

    for dose_idx in range(max_doses):
        # Build dosing event for this interval
        ev = {
            "time": 0.0,
            "amount": amt,
            "compartment": cmt,
        }
        if dur > 0.0:
            ev["duration"] = dur

        t_eval = jnp.linspace(0.0, ii, n_eval + 1)

        sol = solve_ode(
            rhs=model_func,
            t_span=(0.0, ii),
            y0=state,
            params=params,
            t_eval=t_eval,
            dosing_events=[ev],
        )
        # sol shape: (n_eval+1, n_cpt)
        conc = sol[:, cmt]

        # State at end of interval becomes initial state for next
        state = sol[-1]
        n_doses_applied += 1

        # Trough = concentration at end of interval (pre-next-dose)
        current_trough = float(conc[-1])

        if first_trough is None:
            first_trough = current_trough

        # Check convergence (skip first dose)
        if dose_idx > 0 and current_trough > 0.0:
            rel_change = abs(current_trough - prev_trough) / abs(current_trough)
            if rel_change < tol:
                converged = True
                last_times = np.asarray(t_eval)
                last_conc = np.asarray(conc)
                break

        prev_trough = current_trough
        last_times = np.asarray(t_eval)
        last_conc = np.asarray(conc)

    # Compute summary statistics from last interval
    trough = float(last_conc[-1])
    peak = float(np.max(last_conc))
    auc_ss = float(np.trapezoid(last_conc, last_times))

    if first_trough is not None and first_trough > 0.0:
        accumulation_ratio = trough / first_trough
    else:
        accumulation_ratio = 1.0

    return SteadyStateResult(
        trough=trough,
        peak=peak,
        auc_ss=auc_ss,
        n_doses=n_doses_applied,
        accumulation_ratio=accumulation_ratio,
        converged=converged,
    )


def steady_state_profile(
    model_func: Callable,
    dose_event: Dict[str, Any],
    params: Dict[str, Any],
    n_points: int = 100,
) -> Dict[str, np.ndarray]:
    """Return the concentration-time profile over one dosing interval at SS.

    First calls :func:`find_steady_state` to obtain the steady-state
    initial conditions, then simulates one more interval at the
    requested resolution.

    Parameters
    ----------
    model_func : callable
        ODE right-hand side.
    dose_event : dict
        Dosing event specification (same as :func:`find_steady_state`).
    params : dict
        Model parameters.
    n_points : int
        Number of time points in the returned profile.

    Returns
    -------
    dict
        ``{"time": np.ndarray, "concentration": np.ndarray}``
    """
    amt = dose_event["amt"]
    ii = dose_event["ii"]
    cmt = dose_event["cmt"]
    dur = dose_event.get("dur", 0.0)

    # Run to steady state first
    ss_result = find_steady_state(model_func, dose_event, params)

    # Reconstruct the SS initial state by running to convergence and
    # capturing the state at the end of the last interval.
    # We re-run the simulation to get the pre-dose state.
    n_cpt = cmt + 1
    state = jnp.zeros(n_cpt)
    for _ in range(ss_result.n_doses):
        ev = {"time": 0.0, "amount": amt, "compartment": cmt}
        if dur > 0.0:
            ev["duration"] = dur
        sol = solve_ode(
            rhs=model_func,
            t_span=(0.0, ii),
            y0=state,
            params=params,
            t_eval=jnp.array([ii]),
            dosing_events=[ev],
        )
        state = sol[-1]

    # Now simulate one more interval at high resolution
    ev = {"time": 0.0, "amount": amt, "compartment": cmt}
    if dur > 0.0:
        ev["duration"] = dur

    t_eval = jnp.linspace(0.0, ii, n_points)
    sol = solve_ode(
        rhs=model_func,
        t_span=(0.0, ii),
        y0=state,
        params=params,
        t_eval=t_eval,
        dosing_events=[ev],
    )

    return {
        "time": np.asarray(t_eval),
        "concentration": np.asarray(sol[:, cmt]),
    }


def superposition_to_ss(
    single_dose_func: Callable,
    dose: float,
    ii: float,
    params: Dict[str, Any],
    tol: float = 1e-4,
    max_doses: int = 500,
) -> SteadyStateResult:
    """Find steady state using analytical superposition for linCmt models.

    This is faster than ODE-based :func:`find_steady_state` because it
    evaluates the analytical single-dose function and sums contributions
    from all previous doses.

    Parameters
    ----------
    single_dose_func : callable
        ``f(dose, times) -> concentrations`` for a single dose at t=0.
    dose : float
        Dose amount.
    ii : float
        Dosing interval.
    params : dict
        Model parameters (not used directly; kept for API consistency).
    tol : float
        Relative convergence tolerance on trough concentration.
    max_doses : int
        Maximum number of doses before giving up.

    Returns
    -------
    SteadyStateResult
    """
    n_eval = 50
    eval_offsets = jnp.linspace(0.0, ii, n_eval + 1)

    first_trough = None
    prev_trough = 0.0
    converged = False
    n_doses_applied = 0
    last_conc = None

    for n in range(1, max_doses + 1):
        # Dose times: 0, ii, 2*ii, ..., (n-1)*ii
        dose_times = jnp.arange(n) * ii
        doses = jnp.full(n, dose)

        # Evaluate at offset from last dose: last_dose_time + eval_offsets
        absolute_times = dose_times[-1] + eval_offsets

        # Sum contributions from all doses
        total = jnp.zeros_like(eval_offsets)
        for i in range(n):
            dt = absolute_times - dose_times[i]
            contrib = single_dose_func(dose, jnp.maximum(dt, 0.0))
            contrib = jnp.where(dt >= 0.0, contrib, 0.0)
            total = total + contrib

        last_conc = np.asarray(total)
        current_trough = float(total[-1])
        n_doses_applied = n

        if first_trough is None:
            first_trough = current_trough

        if n > 1 and current_trough > 0.0:
            rel_change = abs(current_trough - prev_trough) / abs(current_trough)
            if rel_change < tol:
                converged = True
                break

        prev_trough = current_trough

    trough = float(last_conc[-1])
    peak = float(np.max(last_conc))
    times_arr = np.asarray(eval_offsets)
    auc_ss = float(np.trapezoid(last_conc, times_arr))

    if first_trough is not None and first_trough > 0.0:
        accumulation_ratio = trough / first_trough
    else:
        accumulation_ratio = 1.0

    return SteadyStateResult(
        trough=trough,
        peak=peak,
        auc_ss=auc_ss,
        n_doses=n_doses_applied,
        accumulation_ratio=accumulation_ratio,
        converged=converged,
    )
