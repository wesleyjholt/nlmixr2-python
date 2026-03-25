"""Tests for RATE and DUR column support in ODE solver and EventTable."""

import jax
import jax.numpy as jnp
import pytest

from nlmixr2.ode import solve_ode
from nlmixr2.event_table import EventTable, et


# ---------------------------------------------------------------------------
# Helper: 1-cpt RHS and analytical infusion solution
# ---------------------------------------------------------------------------

def _rhs_1cpt(t, y, params):
    k = params["k"]
    return jnp.array([-k * y[0]])


def analytical_1cpt_infusion(t, dose, k, dur):
    """Analytical 1-cpt amount for zero-order infusion of given dose over dur."""
    rate = dose / dur
    during = (rate / k) * (1.0 - jnp.exp(-k * t))
    after = (rate / k) * (1.0 - jnp.exp(-k * dur)) * jnp.exp(-k * (t - dur))
    return jnp.where(t <= dur, during, after)


# ---------------------------------------------------------------------------
# ODE solver: RATE > 0 computes correct DUR
# ---------------------------------------------------------------------------

class TestRatePositiveComputesDur:
    """When rate > 0, the solver should compute dur = amt / rate."""

    def test_rate_specified_matches_dur_specified(self):
        """An infusion specified via rate should match one specified via dur."""
        dose = 100.0
        k = 0.1
        rate = 50.0  # dur should be 100/50 = 2.0
        dur = dose / rate  # 2.0
        t_eval = jnp.linspace(0.0, 24.0, 100)

        # Using rate
        result_rate = solve_ode(
            rhs=_rhs_1cpt,
            t_span=(0.0, 24.0),
            y0=jnp.array([0.0]),
            params={"k": k},
            t_eval=t_eval,
            dosing_events=[{"time": 0.0, "amount": dose, "compartment": 0, "rate": rate}],
        )

        # Using duration
        result_dur = solve_ode(
            rhs=_rhs_1cpt,
            t_span=(0.0, 24.0),
            y0=jnp.array([0.0]),
            params={"k": k},
            t_eval=t_eval,
            dosing_events=[{"time": 0.0, "amount": dose, "compartment": 0, "duration": dur}],
        )

        assert jnp.max(jnp.abs(result_rate[:, 0] - result_dur[:, 0])) < 0.5

    def test_rate_computes_correct_dur(self):
        """Rate=25, amt=100 => dur=4. Check against analytical solution."""
        dose = 100.0
        k = 0.1
        rate = 25.0
        dur = dose / rate  # 4.0
        t_eval = jnp.linspace(0.0, 24.0, 100)

        result = solve_ode(
            rhs=_rhs_1cpt,
            t_span=(0.0, 24.0),
            y0=jnp.array([0.0]),
            params={"k": k},
            t_eval=t_eval,
            dosing_events=[{"time": 0.0, "amount": dose, "compartment": 0, "rate": rate}],
        )

        expected = analytical_1cpt_infusion(t_eval, dose, k, dur)
        assert jnp.max(jnp.abs(result[:, 0] - expected)) < 1.0


# ---------------------------------------------------------------------------
# DUR > 0 already works (existing behavior)
# ---------------------------------------------------------------------------

class TestDurPositiveWorks:
    """Duration-based infusions should continue to work as before."""

    def test_dur_specified_infusion(self):
        dose = 100.0
        k = 0.1
        dur = 2.0
        t_eval = jnp.linspace(0.0, 24.0, 100)

        result = solve_ode(
            rhs=_rhs_1cpt,
            t_span=(0.0, 24.0),
            y0=jnp.array([0.0]),
            params={"k": k},
            t_eval=t_eval,
            dosing_events=[{"time": 0.0, "amount": dose, "compartment": 0, "duration": dur}],
        )

        expected = analytical_1cpt_infusion(t_eval, dose, k, dur)
        assert jnp.max(jnp.abs(result[:, 0] - expected)) < 1.0


# ---------------------------------------------------------------------------
# RATE and DUR both specified: DUR takes precedence
# ---------------------------------------------------------------------------

class TestRateAndDurBothSpecified:
    """When both rate and dur are specified, dur takes precedence."""

    def test_dur_takes_precedence_over_rate(self):
        dose = 100.0
        k = 0.1
        dur = 2.0
        rate = 10.0  # would imply dur=10, but explicit dur=2 wins
        t_eval = jnp.linspace(0.0, 24.0, 100)

        result = solve_ode(
            rhs=_rhs_1cpt,
            t_span=(0.0, 24.0),
            y0=jnp.array([0.0]),
            params={"k": k},
            t_eval=t_eval,
            dosing_events=[{
                "time": 0.0, "amount": dose, "compartment": 0,
                "duration": dur, "rate": rate,
            }],
        )

        # Should match dur=2 analytical solution, not dur=10
        expected = analytical_1cpt_infusion(t_eval, dose, k, dur)
        assert jnp.max(jnp.abs(result[:, 0] - expected)) < 1.0


# ---------------------------------------------------------------------------
# RATE in EventTable
# ---------------------------------------------------------------------------

class TestRateInEventTable:
    """EventTable should accept and store rate."""

    def test_add_dosing_with_rate(self):
        ev = et().add_dosing(amt=100, time=0, rate=50.0)
        d = ev.to_dict()
        assert d["rate"] == [50.0]
        assert d["amt"] == [100.0]

    def test_add_dosing_default_rate_zero(self):
        ev = et().add_dosing(amt=100, time=0)
        d = ev.to_dict()
        assert d["rate"] == [0.0]

    def test_rate_with_dur(self):
        ev = et().add_dosing(amt=100, time=0, rate=50.0, dur=2.0)
        d = ev.to_dict()
        assert d["rate"] == [50.0]
        assert d["dur"] == [2.0]

    def test_sampling_has_zero_rate(self):
        ev = et().add_sampling([1.0, 2.0])
        d = ev.to_dict()
        assert d["rate"] == [0.0, 0.0]


# ---------------------------------------------------------------------------
# to_dict includes rate column
# ---------------------------------------------------------------------------

class TestToDictIncludesRate:
    def test_rate_key_present(self):
        ev = et().add_dosing(amt=100, time=0, rate=25.0)
        d = ev.to_dict()
        assert "rate" in d

    def test_to_arrays_includes_rate(self):
        ev = et().add_dosing(amt=100, time=0, rate=25.0)
        arrays = ev.to_arrays()
        assert "rate" in arrays
        assert isinstance(arrays["rate"], jax.Array)
        assert jnp.allclose(arrays["rate"], jnp.array([25.0], dtype=jnp.float32))

    def test_empty_table_has_rate(self):
        ev = et()
        d = ev.to_dict()
        assert "rate" in d
        assert d["rate"] == []


# ---------------------------------------------------------------------------
# ODE integration: rate-specified matches dur-specified
# ---------------------------------------------------------------------------

class TestODEIntegrationRateMatchesDur:
    """End-to-end: solve_ode with rate= gives same result as duration=."""

    def test_integration_equivalence(self):
        dose = 200.0
        k = 0.05
        rate = 40.0  # dur = 200/40 = 5
        dur = dose / rate
        t_eval = jnp.linspace(0.0, 48.0, 200)

        result_rate = solve_ode(
            rhs=_rhs_1cpt,
            t_span=(0.0, 48.0),
            y0=jnp.array([0.0]),
            params={"k": k},
            t_eval=t_eval,
            dosing_events=[{"time": 0.0, "amount": dose, "compartment": 0, "rate": rate}],
        )

        result_dur = solve_ode(
            rhs=_rhs_1cpt,
            t_span=(0.0, 48.0),
            y0=jnp.array([0.0]),
            params={"k": k},
            t_eval=t_eval,
            dosing_events=[{"time": 0.0, "amount": dose, "compartment": 0, "duration": dur}],
        )

        assert jnp.max(jnp.abs(result_rate[:, 0] - result_dur[:, 0])) < 0.5

    def test_rate_negative_one_ignored(self):
        """rate=-1 means model-specified rate; treated as bolus if no model rate."""
        dose = 100.0
        k = 0.1
        t_eval = jnp.linspace(0.0, 24.0, 50)

        # rate=-1 with no duration => bolus
        result = solve_ode(
            rhs=_rhs_1cpt,
            t_span=(0.0, 24.0),
            y0=jnp.array([0.0]),
            params={"k": k},
            t_eval=t_eval,
            dosing_events=[{"time": 0.0, "amount": dose, "compartment": 0, "rate": -1}],
        )

        # Should behave as bolus
        expected = dose * jnp.exp(-k * t_eval)
        assert jnp.max(jnp.abs(result[:, 0] - expected)) < 0.5

    def test_rate_negative_two_ignored(self):
        """rate=-2 means model-specified duration; treated as bolus if no model dur."""
        dose = 100.0
        k = 0.1
        t_eval = jnp.linspace(0.0, 24.0, 50)

        # rate=-2 with no duration => bolus
        result = solve_ode(
            rhs=_rhs_1cpt,
            t_span=(0.0, 24.0),
            y0=jnp.array([0.0]),
            params={"k": k},
            t_eval=t_eval,
            dosing_events=[{"time": 0.0, "amount": dose, "compartment": 0, "rate": -2}],
        )

        expected = dose * jnp.exp(-k * t_eval)
        assert jnp.max(jnp.abs(result[:, 0] - expected)) < 0.5
