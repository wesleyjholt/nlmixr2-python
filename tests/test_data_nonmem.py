"""Tests for NONMEM $DATA-style filtering: SKIP, ACCEPT, and record selection."""

import pytest

from nlmixr2.data import (
    expand_doses,
    filter_dataset,
    infer_evid,
    reconstruct_doses,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(**cols):
    """Build a dict-of-lists dataset from keyword columns."""
    return {k: list(v) for k, v in cols.items()}


# ---------------------------------------------------------------------------
# filter_dataset – accept
# ---------------------------------------------------------------------------

class TestFilterAccept:
    def test_accept_equality_keeps_matching(self):
        data = _make_data(id=[1, 2, 3], time=[0, 0, 0], dv=[1.0, 2.0, 3.0])
        out = filter_dataset(data, accept={"id": 2})
        assert out["id"] == [2]
        assert out["dv"] == [2.0]

    def test_accept_range_tuple(self):
        data = _make_data(id=[1, 2, 3, 4, 5], time=[0]*5, dv=[10, 20, 30, 40, 50])
        out = filter_dataset(data, accept={"id": (2, 4)})
        assert out["id"] == [2, 3, 4]

    def test_accept_callable(self):
        data = _make_data(
            id=[1, 1, 2, 2], time=[0, 1, 0, 1],
            dv=[0.0, 1.0, 0.0, 2.0], evid=[1, 0, 1, 0],
        )
        out = filter_dataset(data, accept={"evid": lambda x: x in (0, 1)})
        # All rows match, nothing removed
        assert out["id"] == [1, 1, 2, 2]

    def test_accept_multiple_predicates_all_must_match(self):
        data = _make_data(id=[1, 1, 2, 2], time=[0, 1, 0, 1], dv=[5, 10, 15, 20])
        out = filter_dataset(data, accept={"id": 1, "time": 1})
        assert out["id"] == [1]
        assert out["dv"] == [10]


# ---------------------------------------------------------------------------
# filter_dataset – skip
# ---------------------------------------------------------------------------

class TestFilterSkip:
    def test_skip_equality_removes_matching(self):
        data = _make_data(id=[1, 2, 3], time=[0, 0, 0], dv=[1.0, 2.0, 3.0])
        out = filter_dataset(data, skip={"id": 2})
        assert out["id"] == [1, 3]

    def test_skip_range_tuple(self):
        data = _make_data(id=[1, 2, 3, 4, 5], time=[0]*5, dv=[10, 20, 30, 40, 50])
        out = filter_dataset(data, skip={"id": (2, 4)})
        assert out["id"] == [1, 5]

    def test_skip_callable(self):
        data = _make_data(id=[1, 2, 99], time=[0, 0, 0], dv=[1.0, 2.0, 3.0])
        out = filter_dataset(data, skip={"id": lambda x: x == 99})
        assert out["id"] == [1, 2]


# ---------------------------------------------------------------------------
# filter_dataset – combined accept + skip
# ---------------------------------------------------------------------------

class TestFilterCombined:
    def test_accept_and_skip_together(self):
        data = _make_data(
            id=[1, 1, 2, 2, 99, 99],
            time=[0, 1, 0, 1, 0, 1],
            dv=[1, 2, 3, 4, 5, 6],
            evid=[0, 0, 0, 0, 0, 0],
        )
        out = filter_dataset(
            data,
            accept={"evid": lambda x: x in (0, 1)},
            skip={"id": 99},
        )
        assert out["id"] == [1, 1, 2, 2]
        assert out["dv"] == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# expand_doses
# ---------------------------------------------------------------------------

class TestExpandDoses:
    def test_expand_addl_ii(self):
        data = _make_data(
            id=[1],
            time=[0.0],
            dv=[0.0],
            amt=[100.0],
            evid=[1],
            addl=[2],
            ii=[12.0],
        )
        out = expand_doses(data)
        # Original dose + 2 additional = 3 dose records total
        assert len(out["time"]) == 3
        assert out["time"] == [0.0, 12.0, 24.0]
        assert out["amt"] == [100.0, 100.0, 100.0]
        assert out["evid"] == [1, 1, 1]
        # ADDL should be 0 for all expanded rows
        assert out["addl"] == [0, 0, 0]

    def test_expand_preserves_obs(self):
        data = _make_data(
            id=[1, 1, 1],
            time=[0.0, 6.0, 12.0],
            dv=[0.0, 5.0, 3.0],
            amt=[100.0, 0.0, 0.0],
            evid=[1, 0, 0],
            addl=[1, 0, 0],
            ii=[12.0, 0.0, 0.0],
        )
        out = expand_doses(data)
        # 1 original dose + 1 addl dose + 2 obs = 4 records
        assert len(out["time"]) == 4
        # Expanded dose inserted right after its source row
        assert out["evid"] == [1, 1, 0, 0]
        assert out["time"] == [0.0, 12.0, 6.0, 12.0]

    def test_expand_no_addl_unchanged(self):
        data = _make_data(
            id=[1], time=[0.0], dv=[0.0], amt=[100.0], evid=[1],
        )
        out = expand_doses(data)
        assert len(out["time"]) == 1


# ---------------------------------------------------------------------------
# infer_evid
# ---------------------------------------------------------------------------

class TestInferEvid:
    def test_infer_evid_from_amt(self):
        data = _make_data(
            id=[1, 1, 1], time=[0, 1, 2], dv=[0, 5, 3], amt=[100, 0, 0],
        )
        out = infer_evid(data)
        assert out["evid"] == [1, 0, 0]

    def test_infer_evid_existing_unchanged(self):
        data = _make_data(
            id=[1, 1], time=[0, 1], dv=[0, 5], amt=[100, 0], evid=[4, 0],
        )
        out = infer_evid(data)
        # Should not overwrite existing EVID
        assert out["evid"] == [4, 0]

    def test_infer_evid_no_amt(self):
        data = _make_data(id=[1, 1], time=[0, 1], dv=[5, 3])
        out = infer_evid(data)
        # No AMT column, no EVID column to infer
        assert "evid" not in out


# ---------------------------------------------------------------------------
# reconstruct_doses
# ---------------------------------------------------------------------------

class TestReconstructDoses:
    def test_compute_dur_from_amt_rate(self):
        data = _make_data(
            id=[1, 1], time=[0, 1], dv=[0, 5],
            amt=[100.0, 0.0], rate=[25.0, 0.0], evid=[1, 0],
        )
        out = reconstruct_doses(data)
        assert "dur" in out
        assert out["dur"][0] == pytest.approx(4.0)  # 100/25
        assert out["dur"][1] == pytest.approx(0.0)

    def test_compute_rate_from_amt_dur(self):
        data = _make_data(
            id=[1, 1], time=[0, 1], dv=[0, 5],
            amt=[100.0, 0.0], dur=[4.0, 0.0], evid=[1, 0],
        )
        out = reconstruct_doses(data)
        assert "rate" in out
        assert out["rate"][0] == pytest.approx(25.0)  # 100/4
        assert out["rate"][1] == pytest.approx(0.0)

    def test_reconstruct_no_rate_no_dur_unchanged(self):
        data = _make_data(
            id=[1], time=[0], dv=[0], amt=[100.0], evid=[1],
        )
        out = reconstruct_doses(data)
        assert "rate" not in out
        assert "dur" not in out
