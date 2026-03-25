"""Tests for ADDL/II dose expansion in EventTable and ODE solver."""

import jax.numpy as jnp
import pytest

from nlmixr2.event_table import EventTable, et, expand_addl
from nlmixr2.ode import solve_ode


# ---------------------------------------------------------------------------
# expand_addl() free function
# ---------------------------------------------------------------------------

class TestExpandAddlFunction:
    """Tests for the standalone expand_addl() function."""

    def test_single_dose_addl2_ii12_creates_3_doses(self):
        """A dose at t=0 with addl=2, ii=12 should produce 3 doses at t=0, 12, 24."""
        tbl = et().add_dosing(amt=100.0, time=0.0, ii=12.0, addl=2)
        expanded = expand_addl(tbl)
        d = expanded.to_dict()

        dose_times = [t for t, evid in zip(d["time"], d["evid"]) if evid > 0]
        assert dose_times == [0.0, 12.0, 24.0]

    def test_expansion_preserves_dose_amount(self):
        tbl = et().add_dosing(amt=250.0, time=0.0, ii=8.0, addl=1)
        expanded = expand_addl(tbl)
        d = expanded.to_dict()

        dose_amts = [a for a, evid in zip(d["amt"], d["evid"]) if evid > 0]
        assert all(a == 250.0 for a in dose_amts)

    def test_expansion_preserves_compartment(self):
        tbl = et().add_dosing(amt=100.0, time=0.0, cmt=2, ii=12.0, addl=1)
        expanded = expand_addl(tbl)
        d = expanded.to_dict()

        cmts = [c for c, evid in zip(d["cmt"], d["evid"]) if evid > 0]
        assert all(c == 2 for c in cmts)

    def test_expansion_preserves_observations(self):
        tbl = (
            et()
            .add_dosing(amt=100.0, time=0.0, ii=12.0, addl=2)
            .add_sampling([6.0, 18.0, 30.0])
        )
        expanded = expand_addl(tbl)
        d = expanded.to_dict()

        obs_times = [t for t, evid in zip(d["time"], d["evid"]) if evid == 0]
        assert obs_times == [6.0, 18.0, 30.0]

    def test_addl_zero_leaves_event_unchanged(self):
        tbl = et().add_dosing(amt=100.0, time=5.0, addl=0)
        expanded = expand_addl(tbl)
        d = expanded.to_dict()

        assert d["time"] == [5.0]
        assert d["amt"] == [100.0]
        assert d["addl"] == [0]

    def test_ii_zero_with_addl_raises_error(self):
        tbl = et().add_dosing(amt=100.0, time=0.0, ii=0.0, addl=3)
        with pytest.raises(ValueError, match="ii.*must be.*positive"):
            expand_addl(tbl)

    def test_multiple_doses_different_addl_ii(self):
        tbl = (
            et()
            .add_dosing(amt=100.0, time=0.0, ii=12.0, addl=1)
            .add_dosing(amt=200.0, time=6.0, ii=24.0, addl=2)
        )
        expanded = expand_addl(tbl)
        d = expanded.to_dict()

        dose_rows = [
            (t, a) for t, a, evid in zip(d["time"], d["amt"], d["evid"]) if evid > 0
        ]
        expected = [
            (0.0, 100.0),
            (6.0, 200.0),
            (12.0, 100.0),
            (30.0, 200.0),
            (54.0, 200.0),
        ]
        assert dose_rows == expected

    def test_expanded_doses_have_addl_zero_ii_zero(self):
        tbl = et().add_dosing(amt=100.0, time=0.0, ii=12.0, addl=2)
        expanded = expand_addl(tbl)
        d = expanded.to_dict()

        assert all(a == 0 for a in d["addl"])
        assert all(i == 0.0 for i in d["ii"])

    def test_result_sorted_by_time(self):
        tbl = (
            et()
            .add_dosing(amt=100.0, time=24.0, ii=12.0, addl=1)
            .add_dosing(amt=200.0, time=0.0, ii=6.0, addl=2)
            .add_sampling([3.0, 15.0, 40.0])
        )
        expanded = expand_addl(tbl)
        d = expanded.to_dict()

        times = d["time"]
        assert times == sorted(times)


# ---------------------------------------------------------------------------
# EventTable.expand() method
# ---------------------------------------------------------------------------

class TestExpandMethod:
    """Tests for the expand() method on EventTable."""

    def test_expand_returns_new_event_table(self):
        tbl = et().add_dosing(amt=100.0, time=0.0, ii=12.0, addl=2)
        expanded = tbl.expand()
        assert isinstance(expanded, EventTable)
        assert expanded is not tbl

    def test_expand_matches_expand_addl(self):
        tbl = (
            et()
            .add_dosing(amt=100.0, time=0.0, ii=12.0, addl=2)
            .add_sampling([6.0, 18.0])
        )
        assert tbl.expand().to_dict() == expand_addl(tbl).to_dict()


# ---------------------------------------------------------------------------
# ODE solver integration
# ---------------------------------------------------------------------------

class TestAddlODEIntegration:
    """Expanded ADDL doses should integrate correctly via solve_ode."""

    def _rhs_1cpt(self, t, y, params):
        k = params["k"]
        return jnp.array([-k * y[0]])

    def test_addl_doses_integrate_same_as_explicit(self):
        """ODE solution with addl=2, ii=12 should match 3 explicit bolus doses."""
        k = 0.1
        dose = 100.0
        params = {"k": k}
        t_eval = jnp.linspace(0.0, 48.0, 100)

        # Explicit 3 doses
        explicit_events = [
            {"time": 0.0, "amount": dose, "compartment": 0},
            {"time": 12.0, "amount": dose, "compartment": 0},
            {"time": 24.0, "amount": dose, "compartment": 0},
        ]
        result_explicit = solve_ode(
            rhs=self._rhs_1cpt,
            t_span=(0.0, 48.0),
            y0=jnp.array([0.0]),
            params=params,
            t_eval=t_eval,
            dosing_events=explicit_events,
        )

        # ADDL-based: single dose record with addl=2, ii=12
        addl_events = [
            {
                "time": 0.0,
                "amount": dose,
                "compartment": 0,
                "addl": 2,
                "ii": 12.0,
            },
        ]
        result_addl = solve_ode(
            rhs=self._rhs_1cpt,
            t_span=(0.0, 48.0),
            y0=jnp.array([0.0]),
            params=params,
            t_eval=t_eval,
            dosing_events=addl_events,
        )

        assert jnp.allclose(result_explicit, result_addl, atol=0.5)
