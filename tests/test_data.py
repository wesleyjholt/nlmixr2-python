"""Tests for the NONMEM-style data validation and handling module."""

import jax.numpy as jnp
import numpy as np
import pytest

from nlmixr2.data import (
    ValidatedDataset,
    get_doses,
    get_observations,
    merge_datasets,
    split_by_subject,
    validate_dataset,
)


# ---------------------------------------------------------------------------
# Basic validation
# ---------------------------------------------------------------------------

class TestBasicValidation:
    def test_minimal_dataset_dict_of_arrays(self):
        data = {
            "ID": [1, 1, 2, 2],
            "TIME": [0.0, 1.0, 0.0, 1.0],
            "DV": [0.0, 5.0, 0.0, 3.0],
        }
        ds = validate_dataset(data)
        assert isinstance(ds, ValidatedDataset)
        assert ds.n_subjects == 2
        assert ds.n_observations == 4  # no EVID => all observations
        assert ds.n_doses == 0

    def test_minimal_dataset_list_of_dicts(self):
        data = [
            {"ID": 1, "TIME": 0.0, "DV": 0.0},
            {"ID": 1, "TIME": 1.0, "DV": 5.0},
            {"ID": 2, "TIME": 0.0, "DV": 0.0},
        ]
        ds = validate_dataset(data)
        assert ds.n_subjects == 2
        assert ds.n_observations == 3

    def test_returns_validated_dataset_fields(self):
        data = {
            "ID": [1, 1],
            "TIME": [0.0, 1.0],
            "DV": [0.0, 5.0],
        }
        ds = validate_dataset(data)
        assert "id" in ds.column_names
        assert "time" in ds.column_names
        assert "dv" in ds.column_names
        assert ds.subject_ids == (1,)


# ---------------------------------------------------------------------------
# Case-insensitive column names
# ---------------------------------------------------------------------------

class TestCaseInsensitive:
    def test_uppercase(self):
        data = {"ID": [1], "TIME": [0.0], "DV": [1.0]}
        ds = validate_dataset(data)
        assert "id" in ds.column_names

    def test_lowercase(self):
        data = {"id": [1], "time": [0.0], "dv": [1.0]}
        ds = validate_dataset(data)
        assert "id" in ds.column_names

    def test_mixed_case(self):
        data = {"Id": [1], "Time": [0.0], "Dv": [1.0]}
        ds = validate_dataset(data)
        assert "id" in ds.column_names


# ---------------------------------------------------------------------------
# Column normalization
# ---------------------------------------------------------------------------

class TestColumnNormalization:
    def test_all_columns_lowercase(self):
        data = {
            "ID": [1],
            "TIME": [0.0],
            "DV": [1.0],
            "AMT": [100.0],
            "EVID": [1],
            "CMT": [1],
            "MDV": [1],
            "WT": [70.0],
        }
        ds = validate_dataset(data)
        for col in ds.column_names:
            assert col == col.lower(), f"Column {col} not lowercased"


# ---------------------------------------------------------------------------
# Sorting by ID then TIME
# ---------------------------------------------------------------------------

class TestSorting:
    def test_sorted_by_id_then_time(self):
        data = {
            "ID": [2, 1, 2, 1],
            "TIME": [1.0, 0.0, 0.0, 1.0],
            "DV": [3.0, 0.0, 0.0, 5.0],
        }
        ds = validate_dataset(data)
        ids = np.array(ds.columns["id"])
        times = np.array(ds.columns["time"])
        # Should be sorted: (1,0), (1,1), (2,0), (2,1)
        np.testing.assert_array_equal(ids, [1, 1, 2, 2])
        np.testing.assert_array_equal(times, [0.0, 1.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# Counts: n_subjects, n_observations, n_doses
# ---------------------------------------------------------------------------

class TestCounts:
    def test_counts_with_evid(self):
        data = {
            "ID": [1, 1, 1, 2, 2, 2],
            "TIME": [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
            "DV": [0.0, 5.0, 8.0, 0.0, 3.0, 6.0],
            "EVID": [1, 0, 0, 1, 0, 0],
            "AMT": [100.0, 0.0, 0.0, 100.0, 0.0, 0.0],
        }
        ds = validate_dataset(data)
        assert ds.n_subjects == 2
        assert ds.n_observations == 4
        assert ds.n_doses == 2

    def test_counts_without_evid(self):
        data = {
            "ID": [1, 1, 2],
            "TIME": [0.0, 1.0, 0.0],
            "DV": [0.0, 5.0, 0.0],
        }
        ds = validate_dataset(data)
        assert ds.n_observations == 3
        assert ds.n_doses == 0

    def test_subject_ids(self):
        data = {
            "ID": [3, 1, 2, 1],
            "TIME": [0.0, 0.0, 0.0, 1.0],
            "DV": [0.0, 0.0, 0.0, 5.0],
        }
        ds = validate_dataset(data)
        assert ds.subject_ids == (1, 2, 3)


# ---------------------------------------------------------------------------
# split_by_subject
# ---------------------------------------------------------------------------

class TestSplitBySubject:
    def test_split(self):
        data = {
            "ID": [1, 1, 2, 2],
            "TIME": [0.0, 1.0, 0.0, 1.0],
            "DV": [0.0, 5.0, 0.0, 3.0],
        }
        ds = validate_dataset(data)
        splits = split_by_subject(ds)
        assert set(splits.keys()) == {1, 2}
        assert splits[1].n_subjects == 1
        assert splits[2].n_subjects == 1
        np.testing.assert_array_equal(np.array(splits[1].columns["dv"]), [0.0, 5.0])
        np.testing.assert_array_equal(np.array(splits[2].columns["dv"]), [0.0, 3.0])


# ---------------------------------------------------------------------------
# get_observations / get_doses
# ---------------------------------------------------------------------------

class TestGetObservations:
    def test_observation_filtering(self):
        data = {
            "ID": [1, 1, 1],
            "TIME": [0.0, 0.5, 1.0],
            "DV": [0.0, 5.0, 8.0],
            "EVID": [1, 0, 0],
            "AMT": [100.0, 0.0, 0.0],
        }
        ds = validate_dataset(data)
        obs = get_observations(ds)
        assert obs.n_observations == 2
        assert obs.n_doses == 0
        np.testing.assert_array_equal(np.array(obs.columns["time"]), [0.5, 1.0])

    def test_observations_without_evid(self):
        data = {
            "ID": [1, 1],
            "TIME": [0.0, 1.0],
            "DV": [0.0, 5.0],
        }
        ds = validate_dataset(data)
        obs = get_observations(ds)
        assert obs.n_observations == 2


class TestGetDoses:
    def test_dose_filtering(self):
        data = {
            "ID": [1, 1, 1],
            "TIME": [0.0, 0.5, 1.0],
            "DV": [0.0, 5.0, 8.0],
            "EVID": [1, 0, 0],
            "AMT": [100.0, 0.0, 0.0],
        }
        ds = validate_dataset(data)
        doses = get_doses(ds)
        assert doses.n_doses == 1
        assert doses.n_observations == 0
        np.testing.assert_array_equal(np.array(doses.columns["time"]), [0.0])

    def test_doses_without_evid(self):
        """No EVID column => no dose records."""
        data = {
            "ID": [1, 1],
            "TIME": [0.0, 1.0],
            "DV": [0.0, 5.0],
        }
        ds = validate_dataset(data)
        doses = get_doses(ds)
        assert doses.n_doses == 0


# ---------------------------------------------------------------------------
# merge_datasets
# ---------------------------------------------------------------------------

class TestMergeDatasets:
    def test_merge_two(self):
        d1 = validate_dataset({
            "ID": [1, 1],
            "TIME": [0.0, 1.0],
            "DV": [0.0, 5.0],
        })
        d2 = validate_dataset({
            "ID": [2, 2],
            "TIME": [0.0, 1.0],
            "DV": [0.0, 3.0],
        })
        merged = merge_datasets([d1, d2])
        assert merged.n_subjects == 2
        assert merged.n_observations == 4
        # Should be sorted
        ids = np.array(merged.columns["id"])
        np.testing.assert_array_equal(ids, [1, 1, 2, 2])

    def test_merge_preserves_columns(self):
        d1 = validate_dataset({
            "ID": [1], "TIME": [0.0], "DV": [0.0], "AMT": [100.0], "EVID": [1],
        })
        d2 = validate_dataset({
            "ID": [2], "TIME": [0.0], "DV": [5.0], "AMT": [0.0], "EVID": [0],
        })
        merged = merge_datasets([d1, d2])
        assert "amt" in merged.column_names
        assert "evid" in merged.column_names


# ---------------------------------------------------------------------------
# MDV handling
# ---------------------------------------------------------------------------

class TestMDV:
    def test_mdv_excludes_from_observations(self):
        data = {
            "ID": [1, 1, 1],
            "TIME": [0.0, 1.0, 2.0],
            "DV": [0.0, 5.0, 8.0],
            "MDV": [1, 0, 0],
        }
        ds = validate_dataset(data)
        # MDV=1 records are not counted as observations
        assert ds.n_observations == 2

    def test_mdv_with_evid(self):
        data = {
            "ID": [1, 1, 1],
            "TIME": [0.0, 1.0, 2.0],
            "DV": [0.0, 5.0, 8.0],
            "EVID": [1, 0, 0],
            "MDV": [1, 0, 0],
            "AMT": [100.0, 0.0, 0.0],
        }
        ds = validate_dataset(data)
        assert ds.n_observations == 2
        assert ds.n_doses == 1

    def test_get_observations_respects_mdv(self):
        data = {
            "ID": [1, 1, 1],
            "TIME": [0.0, 1.0, 2.0],
            "DV": [0.0, 5.0, 8.0],
            "MDV": [1, 0, 0],
        }
        ds = validate_dataset(data)
        obs = get_observations(ds)
        assert obs.n_observations == 2
        np.testing.assert_array_equal(np.array(obs.columns["time"]), [1.0, 2.0])


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

class TestErrors:
    def test_missing_id(self):
        with pytest.raises(ValueError, match="(?i)id"):
            validate_dataset({"TIME": [0.0], "DV": [1.0]})

    def test_missing_time(self):
        with pytest.raises(ValueError, match="(?i)time"):
            validate_dataset({"ID": [1], "DV": [1.0]})

    def test_missing_dv(self):
        with pytest.raises(ValueError, match="(?i)dv"):
            validate_dataset({"ID": [1], "TIME": [0.0]})

    def test_nan_in_id(self):
        with pytest.raises(ValueError, match="(?i)nan.*id"):
            validate_dataset({"ID": [1, float("nan")], "TIME": [0.0, 1.0], "DV": [0.0, 1.0]})

    def test_nan_in_time(self):
        with pytest.raises(ValueError, match="(?i)nan.*time"):
            validate_dataset({"ID": [1, 1], "TIME": [0.0, float("nan")], "DV": [0.0, 1.0]})


# ---------------------------------------------------------------------------
# Dosing records with AMT/EVID
# ---------------------------------------------------------------------------

class TestDosingRecords:
    def test_full_dosing_dataset(self):
        data = {
            "ID": [1, 1, 1, 1],
            "TIME": [0.0, 0.5, 1.0, 2.0],
            "DV": [0.0, 5.0, 8.0, 6.0],
            "AMT": [100.0, 0.0, 0.0, 0.0],
            "EVID": [1, 0, 0, 0],
            "CMT": [1, 1, 1, 1],
        }
        ds = validate_dataset(data)
        assert ds.n_doses == 1
        assert ds.n_observations == 3
        assert "amt" in ds.column_names
        assert "cmt" in ds.column_names

    def test_multiple_doses(self):
        data = {
            "ID": [1, 1, 1, 1, 1],
            "TIME": [0.0, 0.5, 6.0, 6.5, 12.0],
            "DV": [0.0, 5.0, 0.0, 3.0, 2.0],
            "AMT": [100.0, 0.0, 50.0, 0.0, 0.0],
            "EVID": [1, 0, 1, 0, 0],
        }
        ds = validate_dataset(data)
        assert ds.n_doses == 2
        assert ds.n_observations == 3
