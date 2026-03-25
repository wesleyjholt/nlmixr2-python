"""Tests for the event table module (rxode2 et() equivalent)."""

import jax
import jax.numpy as jnp
import pytest

from nlmixr2.event_table import EventTable, et


# ---------------------------------------------------------------------------
# et() factory function
# ---------------------------------------------------------------------------

class TestEtFactory:
    def test_creates_event_table(self):
        ev = et()
        assert isinstance(ev, EventTable)

    def test_empty_table(self):
        ev = et()
        d = ev.to_dict()
        assert d == {
            "time": [],
            "amt": [],
            "evid": [],
            "cmt": [],
            "dur": [],
            "ii": [],
            "addl": [],
            "rate": [],
        }


# ---------------------------------------------------------------------------
# add_dosing()
# ---------------------------------------------------------------------------

class TestAddDosing:
    def test_single_bolus(self):
        ev = et().add_dosing(amt=100, time=0)
        d = ev.to_dict()
        assert d["time"] == [0.0]
        assert d["amt"] == [100.0]
        assert d["evid"] == [1]
        assert d["cmt"] == [1]
        assert d["dur"] == [0.0]
        assert d["ii"] == [0.0]
        assert d["addl"] == [0]

    def test_bolus_at_nonzero_time(self):
        ev = et().add_dosing(amt=50, time=2.0)
        d = ev.to_dict()
        assert d["time"] == [2.0]
        assert d["amt"] == [50.0]

    def test_dose_to_specific_compartment(self):
        ev = et().add_dosing(amt=100, time=0, cmt=2)
        d = ev.to_dict()
        assert d["cmt"] == [2]

    def test_infusion(self):
        ev = et().add_dosing(amt=100, time=0, dur=1.5)
        d = ev.to_dict()
        assert d["dur"] == [1.5]
        assert d["amt"] == [100.0]

    def test_dose_with_evid(self):
        ev = et().add_dosing(amt=0, time=5, evid=2)
        d = ev.to_dict()
        assert d["evid"] == [2]

    def test_steady_state_ii_addl(self):
        ev = et().add_dosing(amt=100, time=0, ii=12, addl=6)
        d = ev.to_dict()
        assert d["ii"] == [12.0]
        assert d["addl"] == [6]

    def test_multiple_doses_at_different_times(self):
        ev = et().add_dosing(amt=100, time=0).add_dosing(amt=200, time=24)
        d = ev.to_dict()
        assert d["time"] == [0.0, 24.0]
        assert d["amt"] == [100.0, 200.0]
        assert d["evid"] == [1, 1]


# ---------------------------------------------------------------------------
# add_sampling()
# ---------------------------------------------------------------------------

class TestAddSampling:
    def test_single_observation(self):
        ev = et().add_sampling([1.0])
        d = ev.to_dict()
        assert d["time"] == [1.0]
        assert d["amt"] == [0.0]
        assert d["evid"] == [0]
        assert d["cmt"] == [0]
        assert d["dur"] == [0.0]

    def test_multiple_observations(self):
        ev = et().add_sampling([0.5, 1, 2, 4, 8, 12, 24])
        d = ev.to_dict()
        assert len(d["time"]) == 7
        assert all(e == 0 for e in d["evid"])
        assert all(a == 0.0 for a in d["amt"])

    def test_observations_preserve_order(self):
        ev = et().add_sampling([3, 1, 2])
        d = ev.to_dict()
        # to_dict returns sorted by time
        assert d["time"] == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Sorting by time
# ---------------------------------------------------------------------------

class TestSorting:
    def test_doses_and_obs_sorted_by_time(self):
        ev = (
            et()
            .add_dosing(amt=100, time=0)
            .add_sampling([0.5, 1, 2, 4])
            .add_dosing(amt=100, time=12)
            .add_sampling([12.5, 13, 14])
        )
        d = ev.to_dict()
        times = d["time"]
        assert times == sorted(times)

    def test_dose_before_obs_at_same_time(self):
        """When a dose and observation share the same time, dose (evid=1) comes first."""
        ev = (
            et()
            .add_sampling([0.0])
            .add_dosing(amt=100, time=0.0)
        )
        d = ev.to_dict()
        assert d["time"] == [0.0, 0.0]
        # Dose event should come first at same time
        assert d["evid"] == [1, 0]


# ---------------------------------------------------------------------------
# repeat()
# ---------------------------------------------------------------------------

class TestRepeat:
    def test_repeat_single_dose(self):
        ev = et().add_dosing(amt=100, time=0).repeat(n=3, interval=24)
        d = ev.to_dict()
        # Original at t=0, then repeats at t=24, t=48, t=72
        expected_times = [0.0, 24.0, 48.0, 72.0]
        assert d["time"] == expected_times
        assert all(a == 100.0 for a in d["amt"])
        assert all(e == 1 for e in d["evid"])

    def test_repeat_dose_and_obs(self):
        ev = (
            et()
            .add_dosing(amt=100, time=0)
            .add_sampling([1, 2, 4])
            .repeat(n=2, interval=12)
        )
        d = ev.to_dict()
        # Original: dose at 0, obs at 1,2,4
        # Repeat 1: dose at 12, obs at 13,14,16
        # Repeat 2: dose at 24, obs at 25,26,28
        assert len(d["time"]) == 12  # 4 events * 3 cycles
        assert d["time"] == sorted(d["time"])
        # Check dose times
        dose_times = [t for t, e in zip(d["time"], d["evid"]) if e == 1]
        assert dose_times == [0.0, 12.0, 24.0]

    def test_repeat_zero_is_noop(self):
        ev = et().add_dosing(amt=100, time=0).repeat(n=0, interval=24)
        d = ev.to_dict()
        assert len(d["time"]) == 1


# ---------------------------------------------------------------------------
# Method chaining
# ---------------------------------------------------------------------------

class TestMethodChaining:
    def test_chaining_returns_event_table(self):
        ev = (
            et()
            .add_dosing(amt=100, time=0)
            .add_sampling([0.5, 1, 2, 4, 8, 12])
            .add_dosing(amt=100, time=12)
            .add_sampling([12.5, 13, 14, 16, 20, 24])
        )
        assert isinstance(ev, EventTable)
        d = ev.to_dict()
        assert len(d["time"]) == 14

    def test_chaining_does_not_mutate_original(self):
        """Each method returns a new EventTable, leaving the original unchanged."""
        ev1 = et().add_dosing(amt=100, time=0)
        ev2 = ev1.add_sampling([1, 2, 3])
        assert len(ev1.to_dict()["time"]) == 1
        assert len(ev2.to_dict()["time"]) == 4


# ---------------------------------------------------------------------------
# to_dict()
# ---------------------------------------------------------------------------

class TestToDict:
    def test_all_keys_present(self):
        ev = et().add_dosing(amt=100, time=0)
        d = ev.to_dict()
        expected_keys = {"time", "amt", "evid", "cmt", "dur", "ii", "addl", "rate"}
        assert set(d.keys()) == expected_keys

    def test_all_lists_same_length(self):
        ev = (
            et()
            .add_dosing(amt=100, time=0)
            .add_sampling([1, 2, 3])
        )
        d = ev.to_dict()
        lengths = [len(v) for v in d.values()]
        assert len(set(lengths)) == 1  # all same length


# ---------------------------------------------------------------------------
# to_arrays()
# ---------------------------------------------------------------------------

class TestToArrays:
    def test_returns_jax_arrays(self):
        ev = et().add_dosing(amt=100, time=0).add_sampling([1, 2])
        arrays = ev.to_arrays()
        for key, arr in arrays.items():
            assert isinstance(arr, jax.Array), f"{key} is not a jax.Array"

    def test_array_shapes(self):
        ev = et().add_dosing(amt=100, time=0).add_sampling([1, 2, 4])
        arrays = ev.to_arrays()
        for key, arr in arrays.items():
            assert arr.shape == (4,), f"{key} shape is {arr.shape}, expected (4,)"

    def test_array_values_match_dict(self):
        ev = et().add_dosing(amt=100, time=0).add_sampling([1, 2])
        d = ev.to_dict()
        arrays = ev.to_arrays()
        for key in d:
            expected = jnp.array(d[key], dtype=jnp.float32)
            assert jnp.allclose(arrays[key], expected), f"Mismatch for {key}"

    def test_all_keys_present(self):
        ev = et().add_dosing(amt=100, time=0)
        arrays = ev.to_arrays()
        expected_keys = {"time", "amt", "evid", "cmt", "dur", "ii", "addl", "rate"}
        assert set(arrays.keys()) == expected_keys

    def test_empty_table_arrays(self):
        ev = et()
        arrays = ev.to_arrays()
        for key, arr in arrays.items():
            assert arr.shape == (0,), f"{key} shape is {arr.shape} for empty table"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_duplicate_sampling_times(self):
        ev = et().add_sampling([1, 1, 2])
        d = ev.to_dict()
        assert d["time"] == [1.0, 1.0, 2.0]
        assert len(d["evid"]) == 3

    def test_large_number_of_events(self):
        ev = et()
        for i in range(100):
            ev = ev.add_dosing(amt=100, time=float(i * 24))
        d = ev.to_dict()
        assert len(d["time"]) == 100

    def test_float_time_precision(self):
        ev = et().add_sampling([0.1, 0.2, 0.3])
        d = ev.to_dict()
        assert len(d["time"]) == 3
