"""Simulation module — Python equivalent of rxode2's rxSolve().

Simulates PK/PD models with between-subject variability (BSV) and
residual unexplained variability (RUV).  Supports both analytical
(closed-form) model functions and ODE-based models via ``solve_ode``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np

from nlmixr2.event_table import EventTable
from nlmixr2.ode import solve_ode
from nlmixr2.omega import OmegaBlock, sample_etas


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """Container for simulation output.

    Attributes
    ----------
    subjects : list of dict
        Each dict has keys ``"id"``, ``"time"``, ``"pred"``, ``"ipred"``,
        ``"dv"``.  ``pred`` uses population parameters, ``ipred`` uses
        individual parameters (with etas applied), and ``dv`` adds residual
        error on top of ``ipred``.
    population_params : dict
        The population-level parameter dictionary used for the simulation.
    omega : OmegaBlock or None
        The omega matrix used for between-subject variability, or ``None``.
    sigma : float
        Standard deviation of the additive residual error.
    """

    subjects: List[Dict[str, Any]]
    population_params: Dict[str, Any]
    omega: Optional[OmegaBlock]
    sigma: float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simulate(
    model_func: Callable,
    params: Dict[str, Any],
    event_table: Union[EventTable, Dict[str, Any]],
    n_subjects: int = 1,
    omega: Optional[OmegaBlock] = None,
    sigma: float = 0.0,
    seed: int = 0,
    *,
    y0: Optional[jnp.ndarray] = None,
    ode_t_span: Optional[tuple] = None,
    ode_cmt_index: int = 0,
) -> SimulationResult:
    """Simulate a PK/PD model for one or more subjects.

    Parameters
    ----------
    model_func : callable
        Either an analytical model ``(params, times) -> predictions`` or an
        ODE right-hand side ``(t, y, params) -> dy/dt``.  When *y0* is
        provided the function is treated as an ODE RHS.
    params : dict
        Population parameter values.
    event_table : EventTable or dict
        Dosing/sampling event specification.  If an ``EventTable``, observation
        times are extracted automatically.  If a dict, must contain at least
        ``"time"`` and ``"evid"`` arrays.
    n_subjects : int
        Number of subjects to simulate.
    omega : OmegaBlock, optional
        Between-subject variability covariance structure.  Eta names in the
        omega block should match parameter names with an ``"eta."`` prefix
        (e.g. ``"eta.ke"`` maps to ``params["ke"]``).  Etas are applied
        multiplicatively: ``param_i = param_pop * exp(eta_i)``.
    sigma : float
        Standard deviation of additive residual error.
    seed : int
        JAX PRNG seed for reproducibility.
    y0 : jnp.ndarray, optional
        Initial state for ODE models.  When provided, *model_func* is
        interpreted as an ODE RHS.
    ode_t_span : tuple of (float, float), optional
        Integration time span for ODE models.
    ode_cmt_index : int
        Which ODE compartment to report as prediction (default 0).

    Returns
    -------
    SimulationResult
    """
    key = jax.random.PRNGKey(seed)
    is_ode = y0 is not None

    # --- Extract observation times from event_table -----------------------
    obs_times = _extract_obs_times(event_table)

    # --- Population prediction (no BSV) -----------------------------------
    if is_ode:
        t_span = ode_t_span if ode_t_span is not None else (float(obs_times[0]), float(obs_times[-1]))
        sol = solve_ode(model_func, t_span, y0, params, obs_times)
        pred = sol[:, ode_cmt_index]
    else:
        pred = model_func(params, obs_times)

    pred = jnp.asarray(pred)

    # --- Sample etas if omega provided ------------------------------------
    if omega is not None:
        key, eta_key = jax.random.split(key)
        etas = sample_etas(omega, n_subjects, eta_key)  # (n_subjects, p)
    else:
        etas = None

    # --- Simulate each subject --------------------------------------------
    subjects: List[Dict[str, Any]] = []
    for i in range(n_subjects):
        # Build individual params
        if etas is not None:
            ind_params = _apply_etas(params, omega, etas[i])
        else:
            ind_params = params

        # Individual prediction
        if etas is not None:
            if is_ode:
                t_span_i = ode_t_span if ode_t_span is not None else (float(obs_times[0]), float(obs_times[-1]))
                sol_i = solve_ode(model_func, t_span_i, y0, ind_params, obs_times)
                ipred = sol_i[:, ode_cmt_index]
            else:
                ipred = model_func(ind_params, obs_times)
            ipred = jnp.asarray(ipred)
        else:
            ipred = pred

        # Residual error
        if sigma > 0.0:
            key, eps_key = jax.random.split(key)
            eps = jax.random.normal(eps_key, shape=ipred.shape) * sigma
            dv = ipred + eps
        else:
            dv = ipred

        subjects.append({
            "id": i,
            "time": np.array(obs_times),
            "pred": np.array(pred),
            "ipred": np.array(ipred),
            "dv": np.array(dv),
        })

    return SimulationResult(
        subjects=subjects,
        population_params=params,
        omega=omega,
        sigma=sigma,
    )


def to_dataframe_dict(result: SimulationResult) -> Dict[str, list]:
    """Flatten a SimulationResult into a dict suitable for tabular output.

    All subjects are concatenated in order.  Returns plain Python lists
    so the result can be passed directly to ``pandas.DataFrame(...)`` or
    similar.

    Parameters
    ----------
    result : SimulationResult

    Returns
    -------
    dict
        Keys: ``"id"``, ``"time"``, ``"pred"``, ``"ipred"``, ``"dv"``.
    """
    ids: list = []
    times: list = []
    preds: list = []
    ipreds: list = []
    dvs: list = []

    for subj in result.subjects:
        n = len(subj["time"])
        ids.extend([subj["id"]] * n)
        times.extend(subj["time"].tolist() if hasattr(subj["time"], "tolist") else list(subj["time"]))
        preds.extend(subj["pred"].tolist() if hasattr(subj["pred"], "tolist") else list(subj["pred"]))
        ipreds.extend(subj["ipred"].tolist() if hasattr(subj["ipred"], "tolist") else list(subj["ipred"]))
        dvs.extend(subj["dv"].tolist() if hasattr(subj["dv"], "tolist") else list(subj["dv"]))

    return {
        "id": ids,
        "time": times,
        "pred": preds,
        "ipred": ipreds,
        "dv": dvs,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_obs_times(event_table: Union[EventTable, Dict[str, Any]]) -> jnp.ndarray:
    """Pull observation time points from an EventTable or dict."""
    if isinstance(event_table, EventTable):
        d = event_table.to_arrays()
        evid = d["evid"]
        times = d["time"]
        # Observations have evid == 0
        obs_mask = evid == 0
        return times[obs_mask]
    else:
        times = jnp.asarray(event_table["time"])
        if "evid" in event_table:
            evid = jnp.asarray(event_table["evid"], dtype=jnp.int32)
            obs_mask = evid == 0
            return times[obs_mask]
        return times


def _apply_etas(
    params: Dict[str, Any],
    omega_block: OmegaBlock,
    eta_values: jnp.ndarray,
) -> Dict[str, Any]:
    """Apply random effects to population parameters.

    For each eta named ``"eta.X"`` in the omega block, the corresponding
    parameter ``"X"`` is transformed as ``param * exp(eta)``.
    """
    ind_params = dict(params)
    for j, eta_name in enumerate(omega_block.names):
        # Strip "eta." prefix to get the parameter name
        if eta_name.startswith("eta."):
            param_name = eta_name[4:]
        else:
            param_name = eta_name

        if param_name in ind_params:
            ind_params[param_name] = float(
                ind_params[param_name] * jnp.exp(eta_values[j])
            )

    return ind_params
