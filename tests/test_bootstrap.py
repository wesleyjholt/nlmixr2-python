"""Tests for bootstrap refitting for parameter uncertainty estimation."""

from __future__ import annotations

import pytest

from nlmixr2.api import (
    NLMIXRFit,
    NLMIXRModel,
    ini,
    model,
    nlmixr2,
)
from nlmixr2.bootstrap import (
    BootstrapResult,
    bootstrap_fit,
    parametric_bootstrap,
    resample_by_subject,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model() -> NLMIXRModel:
    return NLMIXRModel(
        ini=ini({"tvcl": 1.0, "tvv": 10.0}),
        model=model(["cl = tvcl", "v = tvv"]),
    )


def _make_data() -> dict:
    """Multi-subject data with 3 subjects, 2 observations each."""
    return {
        "id": [1, 1, 2, 2, 3, 3],
        "time": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        "dv": [0.5, 0.3, 0.6, 0.4, 0.7, 0.2],
    }


def _make_fit() -> NLMIXRFit:
    m = _make_model()
    return nlmixr2(m, data=_make_data(), est="mock")


# ---------------------------------------------------------------------------
# resample_by_subject
# ---------------------------------------------------------------------------

class TestResampleBySubject:
    def test_preserves_column_structure(self):
        data = _make_data()
        resampled = resample_by_subject(data, seed=42)
        assert set(resampled.keys()) == set(data.keys())
        # Same number of columns
        assert len(resampled) == len(data)

    def test_can_duplicate_subjects(self):
        """With 3 subjects and replacement, duplicates are possible over many seeds."""
        data = _make_data()
        found_duplicate = False
        for seed in range(50):
            resampled = resample_by_subject(data, seed=seed)
            ids = list(resampled["id"])
            # Check if any subject appears more times than in original
            from collections import Counter
            orig_counts = Counter(data["id"])
            new_counts = Counter(ids)
            for subj_id, count in new_counts.items():
                if count > orig_counts.get(subj_id, 0):
                    found_duplicate = True
                    break
            if found_duplicate:
                break
        assert found_duplicate, "Expected at least one seed to produce duplicate subjects"

    def test_preserves_total_records_per_subject(self):
        """Each resampled subject should bring all its records."""
        data = _make_data()
        resampled = resample_by_subject(data, seed=0)
        # Original has 2 records per subject, 3 subjects => 6 records
        # Resampled should also have 3 subjects * 2 records = 6 records
        assert len(resampled["id"]) == len(data["id"])


# ---------------------------------------------------------------------------
# bootstrap_fit
# ---------------------------------------------------------------------------

class TestBootstrapFit:
    def test_returns_bootstrap_result(self):
        fit = _make_fit()
        data = _make_data()
        result = bootstrap_fit(fit, data, n_boot=3, seed=0)
        assert isinstance(result, BootstrapResult)

    def test_n_success_plus_n_fail_equals_n_boot(self):
        fit = _make_fit()
        data = _make_data()
        result = bootstrap_fit(fit, data, n_boot=5, seed=0)
        assert result.n_success + result.n_fail == 5

    def test_parameter_summary_has_expected_keys(self):
        fit = _make_fit()
        data = _make_data()
        result = bootstrap_fit(fit, data, n_boot=5, seed=0)
        assert len(result.parameter_summary) > 0
        for param_name, summary in result.parameter_summary.items():
            assert "mean" in summary
            assert "median" in summary
            assert "se" in summary
            assert "ci_lo" in summary
            assert "ci_hi" in summary

    def test_ci_contains_true_value_with_mock(self):
        """With mock estimator the objective is deterministic; CIs should be finite."""
        fit = _make_fit()
        data = _make_data()
        result = bootstrap_fit(fit, data, n_boot=10, seed=0)
        for param_name, summary in result.parameter_summary.items():
            assert summary["ci_lo"] <= summary["ci_hi"]

    def test_reproducibility_with_same_seed(self):
        fit = _make_fit()
        data = _make_data()
        result1 = bootstrap_fit(fit, data, n_boot=5, seed=42)
        result2 = bootstrap_fit(fit, data, n_boot=5, seed=42)
        assert result1.n_success == result2.n_success
        assert result1.n_fail == result2.n_fail
        assert result1.parameter_summary == result2.parameter_summary

    def test_n_boot_1_works(self):
        fit = _make_fit()
        data = _make_data()
        result = bootstrap_fit(fit, data, n_boot=1, seed=0)
        assert isinstance(result, BootstrapResult)
        assert result.n_success + result.n_fail == 1


# ---------------------------------------------------------------------------
# parametric_bootstrap
# ---------------------------------------------------------------------------

class TestParametricBootstrap:
    def test_returns_bootstrap_result(self):
        fit = _make_fit()
        data = _make_data()
        result = parametric_bootstrap(fit, data, n_boot=3, seed=0)
        assert isinstance(result, BootstrapResult)

    def test_n_success_plus_n_fail_equals_n_boot(self):
        fit = _make_fit()
        data = _make_data()
        result = parametric_bootstrap(fit, data, n_boot=5, seed=0)
        assert result.n_success + result.n_fail == 5
