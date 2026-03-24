"""Tests for the diffrax-based ODE solver module."""

import jax
import jax.numpy as jnp
import pytest

from nlmixr2.ode import solve_ode


# ---------------------------------------------------------------------------
# Helper: analytical solutions
# ---------------------------------------------------------------------------

def analytical_1cpt_bolus(t, dose, V, k):
    """C(t) = (Dose / V) * exp(-k * t), amount = Dose * exp(-k * t)."""
    return dose * jnp.exp(-k * t)


def analytical_1cpt_infusion(t, dose, V, k, dur):
    """1-compartment amount during and after zero-order infusion.

    Rate = dose / dur applied from t=0 to t=dur.
    During infusion (t <= dur):
        A(t) = (Rate / k) * (1 - exp(-k * t))
    After infusion (t > dur):
        A(t) = (Rate / k) * (1 - exp(-k * dur)) * exp(-k * (t - dur))
    """
    rate = dose / dur
    during = (rate / k) * (1.0 - jnp.exp(-k * t))
    after = (rate / k) * (1.0 - jnp.exp(-k * dur)) * jnp.exp(-k * (t - dur))
    return jnp.where(t <= dur, during, after)


# ---------------------------------------------------------------------------
# 1-compartment IV bolus
# ---------------------------------------------------------------------------

class TestOneCptBolus:
    """1-compartment model with IV bolus dosing."""

    def _rhs_1cpt(self, t, y, params):
        """dy/dt = -k * y  for a single central compartment."""
        k = params["k"]
        return jnp.array([-k * y[0]])

    def test_single_dose_matches_analytical(self):
        dose = 100.0
        V = 10.0
        k = 0.1
        params = {"k": k}
        t_eval = jnp.linspace(0.0, 24.0, 50)
        dosing_events = [{"time": 0.0, "amount": dose, "compartment": 0}]

        result = solve_ode(
            rhs=self._rhs_1cpt,
            t_span=(0.0, 24.0),
            y0=jnp.array([0.0]),
            params=params,
            t_eval=t_eval,
            dosing_events=dosing_events,
        )

        expected = analytical_1cpt_bolus(t_eval, dose, V, k)
        assert result.shape == (50, 1)
        jnp.allclose(result[:, 0], expected, atol=1e-3, rtol=1e-3)
        # explicit element-wise check
        assert jnp.max(jnp.abs(result[:, 0] - expected)) < 0.5

    def test_output_shape(self):
        params = {"k": 0.1}
        t_eval = jnp.linspace(0.0, 10.0, 20)
        dosing_events = [{"time": 0.0, "amount": 50.0, "compartment": 0}]

        result = solve_ode(
            rhs=self._rhs_1cpt,
            t_span=(0.0, 10.0),
            y0=jnp.array([0.0]),
            params=params,
            t_eval=t_eval,
            dosing_events=dosing_events,
        )

        assert result.shape == (20, 1)

    def test_amount_decreases_monotonically(self):
        params = {"k": 0.5}
        t_eval = jnp.linspace(0.0, 10.0, 30)
        dosing_events = [{"time": 0.0, "amount": 100.0, "compartment": 0}]

        result = solve_ode(
            rhs=self._rhs_1cpt,
            t_span=(0.0, 10.0),
            y0=jnp.array([0.0]),
            params=params,
            t_eval=t_eval,
            dosing_events=dosing_events,
        )

        diffs = jnp.diff(result[:, 0])
        assert jnp.all(diffs <= 1e-6)

    def test_single_time_point(self):
        params = {"k": 0.1}
        t_eval = jnp.array([5.0])
        dosing_events = [{"time": 0.0, "amount": 100.0, "compartment": 0}]

        result = solve_ode(
            rhs=self._rhs_1cpt,
            t_span=(0.0, 5.0),
            y0=jnp.array([0.0]),
            params=params,
            t_eval=t_eval,
            dosing_events=dosing_events,
        )

        assert result.shape == (1, 1)
        expected = 100.0 * jnp.exp(-0.1 * 5.0)
        assert jnp.abs(result[0, 0] - expected) < 0.5


# ---------------------------------------------------------------------------
# Multiple dosing events
# ---------------------------------------------------------------------------

class TestMultipleDosing:
    """Multiple bolus doses at different times."""

    def _rhs_1cpt(self, t, y, params):
        k = params["k"]
        return jnp.array([-k * y[0]])

    def test_two_doses(self):
        k = 0.1
        dose = 100.0
        params = {"k": k}
        t_eval = jnp.linspace(0.0, 48.0, 100)
        dosing_events = [
            {"time": 0.0, "amount": dose, "compartment": 0},
            {"time": 24.0, "amount": dose, "compartment": 0},
        ]

        result = solve_ode(
            rhs=self._rhs_1cpt,
            t_span=(0.0, 48.0),
            y0=jnp.array([0.0]),
            params=params,
            t_eval=t_eval,
            dosing_events=dosing_events,
        )

        assert result.shape == (100, 1)

        # Right after second dose (at t=24), the amount should jump
        # Before 2nd dose: 100*exp(-0.1*24) ≈ 9.07
        # After 2nd dose:  ~109.07
        idx_just_after_24 = jnp.argmin(jnp.abs(t_eval - 24.5))
        val_after = result[int(idx_just_after_24), 0]
        pre_dose = dose * jnp.exp(-k * 24.0)
        # After second dose plus small elimination
        assert val_after > pre_dose + 50.0  # well above pre-dose level

    def test_three_doses_q8h(self):
        k = 0.3
        dose = 200.0
        params = {"k": k}
        t_eval = jnp.linspace(0.0, 32.0, 80)
        dosing_events = [
            {"time": 0.0, "amount": dose, "compartment": 0},
            {"time": 8.0, "amount": dose, "compartment": 0},
            {"time": 16.0, "amount": dose, "compartment": 0},
        ]

        result = solve_ode(
            rhs=self._rhs_1cpt,
            t_span=(0.0, 32.0),
            y0=jnp.array([0.0]),
            params=params,
            t_eval=t_eval,
            dosing_events=dosing_events,
        )

        assert result.shape == (80, 1)
        # All values should be non-negative
        assert jnp.all(result >= -1e-6)


# ---------------------------------------------------------------------------
# 2-compartment model
# ---------------------------------------------------------------------------

class TestTwoCptModel:
    """2-compartment model: central + peripheral."""

    def _rhs_2cpt(self, t, y, params):
        """
        y[0] = central, y[1] = peripheral
        dy0/dt = -(k10 + k12)*y0 + k21*y1
        dy1/dt = k12*y0 - k21*y1
        """
        k10 = params["k10"]
        k12 = params["k12"]
        k21 = params["k21"]
        dy0 = -(k10 + k12) * y[0] + k21 * y[1]
        dy1 = k12 * y[0] - k21 * y[1]
        return jnp.array([dy0, dy1])

    def test_output_shape_2cpt(self):
        params = {"k10": 0.1, "k12": 0.05, "k21": 0.03}
        t_eval = jnp.linspace(0.0, 48.0, 60)
        dosing_events = [{"time": 0.0, "amount": 500.0, "compartment": 0}]

        result = solve_ode(
            rhs=self._rhs_2cpt,
            t_span=(0.0, 48.0),
            y0=jnp.array([0.0, 0.0]),
            params=params,
            t_eval=t_eval,
            dosing_events=dosing_events,
        )

        assert result.shape == (60, 2)

    def test_mass_conservation_no_elimination(self):
        """With k10=0, total mass in both compartments should be conserved."""
        params = {"k10": 0.0, "k12": 0.2, "k21": 0.1}
        t_eval = jnp.linspace(0.0, 50.0, 100)
        dose = 1000.0
        dosing_events = [{"time": 0.0, "amount": dose, "compartment": 0}]

        result = solve_ode(
            rhs=self._rhs_2cpt,
            t_span=(0.0, 50.0),
            y0=jnp.array([0.0, 0.0]),
            params=params,
            t_eval=t_eval,
            dosing_events=dosing_events,
        )

        total = result[:, 0] + result[:, 1]
        assert jnp.allclose(total, dose, atol=1.0)

    def test_peripheral_rises_then_falls(self):
        """Peripheral compartment should rise as drug distributes, then fall."""
        params = {"k10": 0.1, "k12": 0.3, "k21": 0.1}
        t_eval = jnp.linspace(0.0, 48.0, 200)
        dosing_events = [{"time": 0.0, "amount": 500.0, "compartment": 0}]

        result = solve_ode(
            rhs=self._rhs_2cpt,
            t_span=(0.0, 48.0),
            y0=jnp.array([0.0, 0.0]),
            params=params,
            t_eval=t_eval,
            dosing_events=dosing_events,
        )

        periph = result[:, 1]
        # Should start at 0, go up, then come back toward 0
        max_idx = int(jnp.argmax(periph))
        assert max_idx > 0  # rises from zero
        assert max_idx < len(t_eval) - 1  # peaks before the end
        assert periph[max_idx] > periph[0]
        assert periph[-1] < periph[max_idx]

    def test_dose_to_peripheral(self):
        """Dose directly into peripheral compartment."""
        params = {"k10": 0.1, "k12": 0.05, "k21": 0.03}
        t_eval = jnp.linspace(0.0, 24.0, 50)
        dosing_events = [{"time": 0.0, "amount": 500.0, "compartment": 1}]

        result = solve_ode(
            rhs=self._rhs_2cpt,
            t_span=(0.0, 24.0),
            y0=jnp.array([0.0, 0.0]),
            params=params,
            t_eval=t_eval,
            dosing_events=dosing_events,
        )

        # Peripheral should start high and central should start at 0
        # (first eval point after t=0 will show peripheral > central)
        assert result[0, 1] > result[0, 0]


# ---------------------------------------------------------------------------
# Infusion dosing
# ---------------------------------------------------------------------------

class TestInfusionDosing:
    """Zero-order infusion (drug input over a duration)."""

    def _rhs_1cpt(self, t, y, params):
        k = params["k"]
        return jnp.array([-k * y[0]])

    def test_infusion_matches_analytical(self):
        dose = 100.0
        k = 0.1
        dur = 2.0
        params = {"k": k}
        t_eval = jnp.linspace(0.0, 24.0, 100)
        dosing_events = [
            {"time": 0.0, "amount": dose, "compartment": 0, "duration": dur}
        ]

        result = solve_ode(
            rhs=self._rhs_1cpt,
            t_span=(0.0, 24.0),
            y0=jnp.array([0.0]),
            params=params,
            t_eval=t_eval,
            dosing_events=dosing_events,
        )

        expected = analytical_1cpt_infusion(t_eval, dose, 1.0, k, dur)
        assert result.shape == (100, 1)
        # Allow some tolerance for numerical ODE vs analytical
        assert jnp.max(jnp.abs(result[:, 0] - expected)) < 1.0

    def test_infusion_peak_near_end_of_infusion(self):
        """For 1-cpt, peak amount should occur near the end of infusion."""
        dose = 500.0
        k = 0.05
        dur = 4.0
        params = {"k": k}
        t_eval = jnp.linspace(0.0, 24.0, 200)
        dosing_events = [
            {"time": 0.0, "amount": dose, "compartment": 0, "duration": dur}
        ]

        result = solve_ode(
            rhs=self._rhs_1cpt,
            t_span=(0.0, 24.0),
            y0=jnp.array([0.0]),
            params=params,
            t_eval=t_eval,
            dosing_events=dosing_events,
        )

        peak_time = t_eval[int(jnp.argmax(result[:, 0]))]
        # Peak should be at or near end of infusion
        assert jnp.abs(peak_time - dur) < 1.0

    def test_infusion_starts_at_zero(self):
        """Amount should start near 0 and build up during infusion."""
        dose = 100.0
        dur = 5.0
        params = {"k": 0.1}
        t_eval = jnp.linspace(0.0, 20.0, 80)
        dosing_events = [
            {"time": 0.0, "amount": dose, "compartment": 0, "duration": dur}
        ]

        result = solve_ode(
            rhs=self._rhs_1cpt,
            t_span=(0.0, 20.0),
            y0=jnp.array([0.0]),
            params=params,
            t_eval=t_eval,
            dosing_events=dosing_events,
        )

        # At t=0, amount should be ~0
        assert result[0, 0] < 5.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def _rhs_1cpt(self, t, y, params):
        k = params["k"]
        return jnp.array([-k * y[0]])

    def test_no_dosing_events(self):
        """With nonzero initial conditions and no dosing, should still work."""
        params = {"k": 0.1}
        t_eval = jnp.linspace(0.0, 10.0, 20)

        result = solve_ode(
            rhs=self._rhs_1cpt,
            t_span=(0.0, 10.0),
            y0=jnp.array([100.0]),
            params=params,
            t_eval=t_eval,
            dosing_events=[],
        )

        expected = 100.0 * jnp.exp(-0.1 * t_eval)
        assert result.shape == (20, 1)
        assert jnp.max(jnp.abs(result[:, 0] - expected)) < 0.5

    def test_zero_elimination(self):
        """k=0 means no elimination; amount should stay constant."""
        params = {"k": 0.0}
        t_eval = jnp.linspace(0.0, 100.0, 50)
        dosing_events = [{"time": 0.0, "amount": 42.0, "compartment": 0}]

        result = solve_ode(
            rhs=self._rhs_1cpt,
            t_span=(0.0, 100.0),
            y0=jnp.array([0.0]),
            params=params,
            t_eval=t_eval,
            dosing_events=dosing_events,
        )

        assert jnp.allclose(result[:, 0], 42.0, atol=0.1)

    def test_returns_jax_array(self):
        params = {"k": 0.1}
        t_eval = jnp.linspace(0.0, 5.0, 10)
        dosing_events = [{"time": 0.0, "amount": 50.0, "compartment": 0}]

        result = solve_ode(
            rhs=self._rhs_1cpt,
            t_span=(0.0, 5.0),
            y0=jnp.array([0.0]),
            params=params,
            t_eval=t_eval,
            dosing_events=dosing_events,
        )

        assert isinstance(result, jax.Array)
