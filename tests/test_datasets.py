"""Tests for the built-in datasets module."""

import pytest
from nlmixr2.datasets import theo_sd, warfarin, pheno_sd, list_datasets, load_dataset


class TestTheoSD:
    """Tests for Theophylline single-dose PK dataset."""

    def test_returns_dict(self):
        data = theo_sd()
        assert isinstance(data, dict)

    def test_expected_columns(self):
        data = theo_sd()
        expected = {"id", "time", "dv", "amt", "evid", "wt"}
        assert set(data.keys()) == expected

    def test_column_lengths_consistent(self):
        data = theo_sd()
        lengths = [len(v) for v in data.values()]
        assert len(set(lengths)) == 1, f"Inconsistent lengths: {lengths}"

    def test_values_are_plain_lists(self):
        data = theo_sd()
        for key, val in data.items():
            assert isinstance(val, list), f"Column {key} is {type(val)}, expected list"

    def test_12_subjects(self):
        data = theo_sd()
        assert len(set(data["id"])) == 12

    def test_id_values_are_integers(self):
        data = theo_sd()
        for v in data["id"]:
            assert isinstance(v, int)

    def test_time_non_negative(self):
        data = theo_sd()
        assert all(t >= 0 for t in data["time"])

    def test_time_sorted_within_subjects(self):
        data = theo_sd()
        ids = data["id"]
        times = data["time"]
        current_id = None
        prev_time = -1.0
        for i in range(len(ids)):
            if ids[i] != current_id:
                current_id = ids[i]
                prev_time = -1.0
            assert times[i] >= prev_time, (
                f"Time not sorted for subject {current_id} at index {i}: "
                f"{times[i]} < {prev_time}"
            )
            prev_time = times[i]

    def test_amt_positive_only_for_dosing(self):
        data = theo_sd()
        for i in range(len(data["evid"])):
            if data["evid"][i] == 1:
                assert data["amt"][i] > 0, f"Dosing record at {i} has amt <= 0"
            else:
                assert data["amt"][i] == 0, f"Non-dosing record at {i} has amt != 0"

    def test_dv_positive_for_observations(self):
        data = theo_sd()
        for i in range(len(data["evid"])):
            if data["evid"][i] == 0:
                assert data["dv"][i] > 0, f"Observation at {i} has dv <= 0"

    def test_roughly_10_timepoints_per_subject(self):
        data = theo_sd()
        n_records = len(data["id"])
        n_subjects = len(set(data["id"]))
        avg = n_records / n_subjects
        assert 8 <= avg <= 14, f"Average records per subject: {avg}"


class TestWarfarin:
    """Tests for Warfarin PK dataset."""

    def test_returns_dict(self):
        data = warfarin()
        assert isinstance(data, dict)

    def test_expected_columns(self):
        data = warfarin()
        expected = {"id", "time", "dv", "amt", "evid", "wt", "age", "sex"}
        assert set(data.keys()) == expected

    def test_column_lengths_consistent(self):
        data = warfarin()
        lengths = [len(v) for v in data.values()]
        assert len(set(lengths)) == 1

    def test_values_are_plain_lists(self):
        data = warfarin()
        for key, val in data.items():
            assert isinstance(val, list), f"Column {key} is {type(val)}, expected list"

    def test_about_32_subjects(self):
        data = warfarin()
        n = len(set(data["id"]))
        assert 30 <= n <= 34, f"Expected ~32 subjects, got {n}"

    def test_id_values_are_integers(self):
        data = warfarin()
        for v in data["id"]:
            assert isinstance(v, int)

    def test_time_non_negative(self):
        data = warfarin()
        assert all(t >= 0 for t in data["time"])

    def test_time_sorted_within_subjects(self):
        data = warfarin()
        ids = data["id"]
        times = data["time"]
        current_id = None
        prev_time = -1.0
        for i in range(len(ids)):
            if ids[i] != current_id:
                current_id = ids[i]
                prev_time = -1.0
            assert times[i] >= prev_time
            prev_time = times[i]

    def test_amt_positive_only_for_dosing(self):
        data = warfarin()
        for i in range(len(data["evid"])):
            if data["evid"][i] == 1:
                assert data["amt"][i] > 0
            else:
                assert data["amt"][i] == 0

    def test_dv_positive_for_observations(self):
        data = warfarin()
        for i in range(len(data["evid"])):
            if data["evid"][i] == 0:
                assert data["dv"][i] > 0

    def test_sex_values(self):
        data = warfarin()
        assert all(s in (0, 1) for s in data["sex"])

    def test_age_reasonable(self):
        data = warfarin()
        assert all(18 <= a <= 90 for a in data["age"])


class TestPhenoSD:
    """Tests for Phenobarbital neonatal PK dataset."""

    def test_returns_dict(self):
        data = pheno_sd()
        assert isinstance(data, dict)

    def test_expected_columns(self):
        data = pheno_sd()
        expected = {"id", "time", "dv", "amt", "evid", "wt", "apgr"}
        assert set(data.keys()) == expected

    def test_column_lengths_consistent(self):
        data = pheno_sd()
        lengths = [len(v) for v in data.values()]
        assert len(set(lengths)) == 1

    def test_values_are_plain_lists(self):
        data = pheno_sd()
        for key, val in data.items():
            assert isinstance(val, list), f"Column {key} is {type(val)}, expected list"

    def test_about_59_subjects(self):
        data = pheno_sd()
        n = len(set(data["id"]))
        assert 55 <= n <= 63, f"Expected ~59 subjects, got {n}"

    def test_id_values_are_integers(self):
        data = pheno_sd()
        for v in data["id"]:
            assert isinstance(v, int)

    def test_time_non_negative(self):
        data = pheno_sd()
        assert all(t >= 0 for t in data["time"])

    def test_time_sorted_within_subjects(self):
        data = pheno_sd()
        ids = data["id"]
        times = data["time"]
        current_id = None
        prev_time = -1.0
        for i in range(len(ids)):
            if ids[i] != current_id:
                current_id = ids[i]
                prev_time = -1.0
            assert times[i] >= prev_time
            prev_time = times[i]

    def test_amt_positive_only_for_dosing(self):
        data = pheno_sd()
        for i in range(len(data["evid"])):
            if data["evid"][i] == 1:
                assert data["amt"][i] > 0
            else:
                assert data["amt"][i] == 0

    def test_dv_positive_for_observations(self):
        data = pheno_sd()
        for i in range(len(data["evid"])):
            if data["evid"][i] == 0:
                assert data["dv"][i] > 0

    def test_apgr_reasonable(self):
        data = pheno_sd()
        assert all(0 <= a <= 10 for a in data["apgr"])


class TestListDatasets:
    """Tests for list_datasets function."""

    def test_returns_list(self):
        result = list_datasets()
        assert isinstance(result, list)

    def test_contains_all_datasets(self):
        result = list_datasets()
        assert "theo_sd" in result
        assert "warfarin" in result
        assert "pheno_sd" in result

    def test_length(self):
        result = list_datasets()
        assert len(result) >= 3


class TestLoadDataset:
    """Tests for load_dataset function."""

    def test_load_theo_sd(self):
        data = load_dataset("theo_sd")
        assert "id" in data
        assert "time" in data

    def test_load_warfarin(self):
        data = load_dataset("warfarin")
        assert "age" in data
        assert "sex" in data

    def test_load_pheno_sd(self):
        data = load_dataset("pheno_sd")
        assert "apgr" in data

    def test_invalid_name_raises(self):
        with pytest.raises((ValueError, KeyError)):
            load_dataset("nonexistent_dataset")
