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

    t_eval = jnp.asarray(t_eval)
    state = jnp.asarray(y0)

    # Expand ADDL/II: any event with addl > 0 is replaced by explicit copies
    dosing_events = _expand_addl_dosing(dosing_events)

    # Pre-process dosing events: apply bioavailability and lag_time
    processed_events = []
    for ev in dosing_events:
        ev = dict(ev)  # shallow copy
        F = ev.pop("bioavailability", 1.0)
        tlag = ev.pop("lag_time", 0.0)
        ev["amount"] = ev["amount"] * F
        ev["time"] = ev["time"] + tlag
        processed_events.append(ev)
    dosing_events = processed_events

    n_cpt = state.shape[0]

    # Resolve rate/duration: if rate > 0 and no duration, compute duration
    resolved_events = []
    for ev in dosing_events:
        ev = dict(ev)  # shallow copy
        rate = ev.pop("rate", 0.0)
        dur = ev.get("duration", 0.0)
        if dur > 0.0:
            # Explicit duration takes precedence; ignore rate
            pass
        elif rate > 0.0 and ev.get("amount", 0.0) > 0.0:
            # Compute duration from rate
            ev["duration"] = ev["amount"] / rate
        elif rate == -1 or rate == -2:
            # Model-specified rate/duration: not yet supported,
            # fall through as bolus (no duration set)
            pass
        resolved_events.append(ev)
    dosing_events = resolved_events

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
        rate = jnp.zeros(n_cpt, dtype=state.dtype)
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
        dydt = jnp.asarray(rhs(t, y, params), dtype=y.dtype)
        return dydt + jnp.asarray(_infusion_rate(t), dtype=y.dtype)

    # Collect unique bolus times within t_span
    bolus_times = sorted({ev["time"] for ev in bolus_events})

    # Build ordered list of integration segments.
    # At each bolus time we pause, apply the bolus, then resume.
    t0, tf = t_span

    # Segments are (seg_start, seg_end) pairs
    breakpoints = sorted({t0, tf} | {bt for bt in bolus_times if t0 <= bt <= tf})

    # Solve piecewise, collecting results for t_eval points in each segment
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
                    jnp.array([seg_end], dtype=t_eval.dtype),
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
                jnp.array([seg_end], dtype=t_eval.dtype),
            )
            state = end_sol[0]
        else:
            state = sol[-1]

    if len(results) == 0:
        return jnp.zeros((len(t_eval), n_cpt), dtype=state.dtype)

    return jnp.concatenate(results, axis=0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _expand_addl_dosing(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Expand ADDL/II fields in dosing-event dicts into explicit dose records."""
    expanded: list[dict[str, Any]] = []
    for ev in events:
        addl = ev.get("addl", 0)
        ii = ev.get("ii", 0.0)
        if addl > 0:
            if ii <= 0:
                raise ValueError(
                    f"ii must be positive when addl > 0 (got ii={ii}, addl={addl})"
                )
            base = {k: v for k, v in ev.items() if k not in ("addl", "ii")}
            expanded.append(base)
            for k in range(1, addl + 1):
                copy = dict(base)
                copy["time"] = ev["time"] + k * ii
                expanded.append(copy)
        else:
            # Strip addl/ii keys if present so downstream code doesn't see them
            clean = {k: v for k, v in ev.items() if k not in ("addl", "ii")}
            expanded.append(clean)
    return expanded


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


# ---------------------------------------------------------------------------
# Transit compartment helper
# ---------------------------------------------------------------------------

def transit_compartments(
    n_transit: int,
    ktr: float,
    dose_compartment: int = 0,
) -> Callable:
    """Return a function that adds *n_transit* transit-chain terms to *dydt*.

    The transit chain occupies state indices ``[dose_compartment, dose_compartment + n_transit)``.
    The first transit compartment (index ``dose_compartment``) receives the
    bolus dose.  The last transit compartment feeds into the compartment at
    index ``dose_compartment + n_transit`` (e.g., a depot or absorption
    compartment) which the caller is responsible for defining.

    Usage::

        transit_fn = transit_compartments(3, ktr=2.0, dose_compartment=0)

        def rhs(t, y, params):
            dydt = jnp.zeros_like(y)
            dydt = transit_fn(t, y, dydt)
            # ... add depot/central dynamics for y[3], y[4], etc.
            return dydt

    Parameters
    ----------
    n_transit : int
        Number of transit compartments in the chain.
    ktr : float
        Transit rate constant (same for all compartments).
    dose_compartment : int
        Index of the first transit compartment in the state vector.

    Returns
    -------
    callable
        ``fn(t, y, dydt) -> dydt`` that mutates and returns *dydt*.
    """
    def _transit_rhs(t, y, dydt):
        for i in range(n_transit):
            idx = dose_compartment + i
            if i == 0:
                inflow = 0.0  # dose is applied via bolus event
            else:
                inflow = ktr * y[dose_compartment + i - 1]
            outflow = ktr * y[idx]
            dydt = dydt.at[idx].add(inflow - outflow)
        # Feed last transit into the next compartment (depot)
        depot_idx = dose_compartment + n_transit
        dydt = dydt.at[depot_idx].add(ktr * y[dose_compartment + n_transit - 1])
        return dydt

    return _transit_rhs
