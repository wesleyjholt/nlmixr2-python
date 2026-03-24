"""Diffrax-based ODE solver for pharmacokinetic/pharmacodynamic models.

This module provides ``solve_ode``, which wraps diffrax's adaptive-step
Dormand-Prince (Dopri5) integrator and adds support for PK dosing events
(bolus and zero-order infusions).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Tuple

import diffrax
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def solve_ode(
    rhs: Callable[[float, jnp.ndarray, Dict[str, Any]], jnp.ndarray],
    t_span: Tuple[float, float],
    y0: jnp.ndarray,
    params: Dict[str, Any],
    t_eval: jnp.ndarray,
    dosing_events: List[Dict[str, Any]] | None = None,
) -> jnp.ndarray:
    """Solve an ODE system with PK dosing events.

    Parameters
    ----------
    rhs : callable
        Right-hand side ``f(t, y, params) -> dy/dt``.  Must return a
        JAX array of the same shape as *y*.
    t_span : (t0, tf)
        Start and end of the integration interval.
    y0 : jnp.ndarray
        Initial state vector (shape ``(n_cpt,)``).
    params : dict
        Parameter dictionary forwarded to *rhs*.
    t_eval : jnp.ndarray
        1-D array of time points at which to report the solution.
    dosing_events : list of dict, optional
        Each dict must have keys ``"time"``, ``"amount"``, ``"compartment"``.
        Optionally ``"duration"`` (seconds) for zero-order infusion; if
        absent the dose is treated as an instantaneous bolus.

    Returns
    -------
    jnp.ndarray
        Solution array of shape ``(len(t_eval), n_cpt)``.
    """
    if dosing_events is None:
        dosing_events = []

    n_cpt = y0.shape[0]

    # Separate bolus and infusion events
    bolus_events = []
    infusion_events = []
    for ev in dosing_events:
        if ev.get("duration", 0.0) > 0.0:
            infusion_events.append(ev)
        else:
            bolus_events.append(ev)

    # Sort bolus events by time
    bolus_events.sort(key=lambda e: e["time"])

    # Build infusion rate function: sum of active zero-order infusions
    def _infusion_rate(t: float) -> jnp.ndarray:
        rate = jnp.zeros(n_cpt)
        for ev in infusion_events:
            t_start = ev["time"]
            dur = ev["duration"]
            cpt = ev["compartment"]
            r = ev["amount"] / dur
            # active when t_start <= t < t_start + dur
            active = jnp.where(
                (t >= t_start) & (t < t_start + dur), r, 0.0
            )
            rate = rate.at[cpt].add(active)
        return rate

    # Augmented RHS that includes infusion input
    def _augmented_rhs(t, y, params):
        dydt = rhs(t, y, params)
        return dydt + _infusion_rate(t)

    # Collect unique bolus times within t_span
    bolus_times = sorted({ev["time"] for ev in bolus_events})

    # Build ordered list of integration segments.
    # At each bolus time we pause, apply the bolus, then resume.
    t0, tf = t_span

    # Segments are (seg_start, seg_end) pairs
    breakpoints = sorted({t0, tf} | {bt for bt in bolus_times if t0 <= bt <= tf})

    # Solve piecewise, collecting results for t_eval points in each segment
    state = jnp.array(y0, dtype=jnp.float64)
    results = []  # list of (t_points, y_values) pairs

    for i in range(len(breakpoints) - 1):
        seg_start = breakpoints[i]
        seg_end = breakpoints[i + 1]

        # Apply any bolus dose at segment start
        for ev in bolus_events:
            if abs(ev["time"] - seg_start) < 1e-12:
                state = state.at[ev["compartment"]].add(ev["amount"])

        # Determine which t_eval points fall in [seg_start, seg_end)
        # For the last segment, include seg_end
        if i == len(breakpoints) - 2:
            mask = (t_eval >= seg_start) & (t_eval <= seg_end)
        else:
            mask = (t_eval >= seg_start) & (t_eval < seg_end)

        seg_t_eval = t_eval[mask]

        if len(seg_t_eval) == 0:
            # No evaluation points in this segment, but we still need
            # to propagate state to seg_end.
            if seg_start < seg_end:
                sol = _solve_segment(
                    _augmented_rhs,
                    seg_start,
                    seg_end,
                    state,
                    params,
                    jnp.array([seg_end]),
                )
                state = sol[0]
            continue

        sol = _solve_segment(
            _augmented_rhs, seg_start, seg_end, state, params, seg_t_eval
        )
        results.append(sol)

        # Advance state to end of segment
        if seg_t_eval[-1] < seg_end:
            end_sol = _solve_segment(
                _augmented_rhs,
                seg_start,
                seg_end,
                state,
                params,
                jnp.array([seg_end]),
            )
            state = end_sol[0]
        else:
            state = sol[-1]

    if len(results) == 0:
        return jnp.zeros((len(t_eval), n_cpt))

    return jnp.concatenate(results, axis=0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _solve_segment(
    rhs_fn: Callable,
    t0: float,
    t1: float,
    y0: jnp.ndarray,
    params: Dict[str, Any],
    t_eval: jnp.ndarray,
) -> jnp.ndarray:
    """Integrate a single ODE segment with diffrax and return states at *t_eval*.

    Returns array of shape ``(len(t_eval), n_cpt)``.
    """
    if t0 >= t1 or len(t_eval) == 0:
        # Nothing to integrate; replicate initial state
        return jnp.tile(y0, (len(t_eval), 1))

    # If t_eval contains t0 exactly, diffrax may not include it in SaveAt.
    # We handle the t0 point separately and solve the rest.
    at_start_mask = jnp.abs(t_eval - t0) < 1e-12
    n_at_start = int(jnp.sum(at_start_mask))

    inner_t = t_eval[~at_start_mask]

    if len(inner_t) == 0:
        return jnp.tile(y0, (n_at_start, 1))

    term = diffrax.ODETerm(lambda t, y, args: rhs_fn(t, y, args))
    solver = diffrax.Dopri5()
    stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)

    saveat = diffrax.SaveAt(ts=inner_t)

    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=min(0.01, (t1 - t0) / 10.0),
        y0=y0,
        args=params,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=100_000,
    )

    ys = solution.ys  # (n_save, n_cpt)

    if n_at_start > 0:
        start_rows = jnp.tile(y0, (n_at_start, 1))
        ys = jnp.concatenate([start_rows, ys], axis=0)

    return ys
