"""Tests for steady-state solving functionality."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from nlmixr2.steady_state import (
    SteadyStateResult,
    find_steady_state,
    steady_state_profile,
    superposition_to_ss,
)
from nlmixr2.lincmt import one_cmt_bolus
from nlmixr2.ode import solve_ode


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def _one_cmt_conc_rhs(t, y, params):
    """1-compartment IV bolus ODE RHS in concentration space: dC/dt = -ke * C.

    When using this with solve_ode, the dosing amount should be dose/V so
    that the state directly represents concentration.
    """
    ke = params["ke"]
    return jnp.array([-ke * y[0]])


def _one_cmt_single_dose_func(dose, times):
    """Analytical single-dose function for 1-cpt bolus (ke=0.1, V=10)."""
    ke = 0.1
    V = 10.0
    return (dose / V) * jnp.exp(-ke * times)


# Standard parameters used across tests
_STD_PARAMS = {"ke": 0.1, "V": 10.0}
# For ODE tests, amt is dose/V (=10) because solve_ode adds amt directly to
# state, and the RHS works in concentration space.
_STD_DOSE_EVENT_ODE = {"amt": 10.0, "ii": 12.0, "cmt": 0, "dur": 0.0}


# ---------------------------------------------------------------------------
# Tests for find_steady_state
# ---------------------------------------------------------------------------

class TestFindSteadyState:
    """Tests for the ODE-based find_steady_state function."""

    def test_converges_for_1cmt(self):
        """find_steady_state should converge for a simple 1-cpt model."""
        result = find_steady_state(
            model_func=_one_cmt_conc_rhs,
            dose_event=_STD_DOSE_EVENT_ODE,
            params=_STD_PARAMS,
        )
        assert isinstance(result, SteadyStateResult)
        assert result.converged is True
        assert result.trough > 0.0
        assert result.peak > 0.0
        assert result.auc_ss > 0.0

    def test_accumulation_ratio_gt_1(self):
        """Accumulation ratio must be > 1 for a drug that accumulates."""
        result = find_steady_state(
            model_func=_one_cmt_conc_rhs,
            dose_event=_STD_DOSE_EVENT_ODE,
            params=_STD_PARAMS,
        )
        assert result.accumulation_ratio > 1.0

    def test_n_doses_reasonable(self):
        """Number of doses to reach SS should be reasonable (< 100 for 1-cpt)."""
        result = find_steady_state(
            model_func=_one_cmt_conc_rhs,
            dose_event=_STD_DOSE_EVENT_ODE,
            params=_STD_PARAMS,
        )
        assert result.n_doses >= 2
        assert result.n_doses < 100

    def test_trough_less_than_peak(self):
        """Trough concentration must be less than peak."""
        result = find_steady_state(
            model_func=_one_cmt_conc_rhs,
            dose_event=_STD_DOSE_EVENT_ODE,
            params=_STD_PARAMS,
        )
        assert result.trough < result.peak

    def test_max_doses_triggers_not_converged(self):
        """When max_doses is too small, converged should be False."""
        result = find_steady_state(
            model_func=_one_cmt_conc_rhs,
            dose_event=_STD_DOSE_EVENT_ODE,
            params=_STD_PARAMS,
            tol=1e-12,
            max_doses=2,
        )
        assert result.converged is False

    def test_different_dosing_interval(self):
        """Should work with a different dosing interval (e.g. q24h)."""
        dose_event_q24 = {"amt": 10.0, "ii": 24.0, "cmt": 0, "dur": 0.0}
        result = find_steady_state(
            model_func=_one_cmt_conc_rhs,
            dose_event=dose_event_q24,
            params=_STD_PARAMS,
        )
        assert result.converged is True
        # Longer interval => less accumulation
        result_short = find_steady_state(
            model_func=_one_cmt_conc_rhs,
            dose_event=_STD_DOSE_EVENT_ODE,
            params=_STD_PARAMS,
        )
        assert result.accumulation_ratio < result_short.accumulation_ratio


# ---------------------------------------------------------------------------
# Tests for steady_state_profile
# ---------------------------------------------------------------------------

class TestSteadyStateProfile:
    """Tests for steady_state_profile."""

    def test_returns_correct_shape(self):
        """Profile should return time and conc arrays of length n_points."""
        n_points = 50
        profile = steady_state_profile(
            model_func=_one_cmt_conc_rhs,
            dose_event=_STD_DOSE_EVENT_ODE,
            params=_STD_PARAMS,
            n_points=n_points,
        )
        assert "time" in profile
        assert "concentration" in profile
        assert len(profile["time"]) == n_points
        assert len(profile["concentration"]) == n_points

    def test_profile_time_range(self):
        """Profile times should span one dosing interval [0, ii]."""
        ii = 12.0
        dose_event = {"amt": 10.0, "ii": ii, "cmt": 0, "dur": 0.0}
        profile = steady_state_profile(
            model_func=_one_cmt_conc_rhs,
            dose_event=dose_event,
            params=_STD_PARAMS,
            n_points=100,
        )
        times = np.asarray(profile["time"])
        assert times[0] >= 0.0
        assert times[-1] <= ii

    def test_profile_concentrations_positive(self):
        """All concentrations in the SS profile should be positive."""
        profile = steady_state_profile(
            model_func=_one_cmt_conc_rhs,
            dose_event=_STD_DOSE_EVENT_ODE,
            params=_STD_PARAMS,
        )
        conc = np.asarray(profile["concentration"])
        assert np.all(conc > 0.0)


# ---------------------------------------------------------------------------
# Tests for superposition_to_ss
# ---------------------------------------------------------------------------

class TestSuperpositionToSS:
    """Tests for the analytical superposition-based steady-state solver."""

    def test_converges(self):
        """superposition_to_ss should converge for a 1-cpt bolus model."""
        result = superposition_to_ss(
            single_dose_func=_one_cmt_single_dose_func,
            dose=100.0,
            ii=12.0,
            params={"ke": 0.1, "V": 10.0},
        )
        assert result.converged is True
        assert result.trough > 0.0

    def test_matches_find_steady_state(self):
        """superposition_to_ss and find_steady_state should agree on trough."""
        ode_result = find_steady_state(
            model_func=_one_cmt_conc_rhs,
            dose_event=_STD_DOSE_EVENT_ODE,
            params=_STD_PARAMS,
        )
        sup_result = superposition_to_ss(
            single_dose_func=_one_cmt_single_dose_func,
            dose=100.0,
            ii=12.0,
            params={"ke": 0.1, "V": 10.0},
        )
        # Allow 5% relative tolerance between ODE and analytical
        np.testing.assert_allclose(
            float(sup_result.trough), float(ode_result.trough), rtol=0.05
        )
        np.testing.assert_allclose(
            float(sup_result.peak), float(ode_result.peak), rtol=0.05
        )

    def test_accumulation_ratio_gt_1(self):
        """Accumulation ratio from superposition should be > 1."""
        result = superposition_to_ss(
            single_dose_func=_one_cmt_single_dose_func,
            dose=100.0,
            ii=12.0,
            params={"ke": 0.1, "V": 10.0},
        )
        assert result.accumulation_ratio > 1.0

    def test_different_interval(self):
        """superposition_to_ss should work with q24h dosing."""
        result_q12 = superposition_to_ss(
            single_dose_func=_one_cmt_single_dose_func,
            dose=100.0,
            ii=12.0,
            params={"ke": 0.1, "V": 10.0},
        )
        result_q24 = superposition_to_ss(
            single_dose_func=_one_cmt_single_dose_func,
            dose=100.0,
            ii=24.0,
            params={"ke": 0.1, "V": 10.0},
        )
        # Longer interval => less accumulation
        assert result_q24.accumulation_ratio < result_q12.accumulation_ratio
