"""Tests for model comparison utilities."""

from __future__ import annotations

import math
import pytest

from nlmixr2.api import NLMIXRFit, NLMIXRModel, IniBlock, InitValue, ModelBlock
from nlmixr2.compare import (
    ComparisonTable,
    LRTResult,
    compare_fits,
    likelihood_ratio_test,
    bootstrap_comparison,
    format_comparison,
)


def _make_fit(
    objective: float = 100.0,
    n_params: int = 3,
    n_obs: int = 50,
) -> NLMIXRFit:
    """Helper to create a minimal NLMIXRFit for testing."""
    values = {f"p{i}": InitValue(estimate=float(i)) for i in range(n_params)}
    ini = IniBlock(values=values)
    model_block = ModelBlock(statements=("y = p0",))
    nlmixr_model = NLMIXRModel(ini=ini, model=model_block)
    return NLMIXRFit(
        estimator="mock",
        n_observations=n_obs,
        columns=("time", "dv"),
        parameter_count=n_params,
        objective=objective,
        model=nlmixr_model,
        control={},
        table={},
    )


class TestCompareFits:
    def test_two_fits_basic(self):
        fit1 = _make_fit(objective=100.0, n_params=3, n_obs=50)
        fit2 = _make_fit(objective=90.0, n_params=4, n_obs=50)
        table = compare_fits([fit1, fit2], names=["Base", "Full"])

        assert table.models == ["Base", "Full"]
        assert table.objectives == [100.0, 90.0]
        assert table.n_params == [3, 4]
        assert table.n_obs == [50, 50]
        assert len(table.aics) == 2
        assert len(table.bics) == 2

    def test_identifies_best_aic(self):
        fit1 = _make_fit(objective=100.0, n_params=3, n_obs=50)
        fit2 = _make_fit(objective=80.0, n_params=4, n_obs=50)
        table = compare_fits([fit1, fit2], names=["Base", "Full"])

        # AIC = obj + 2*k: Base=106, Full=88
        assert table.best_aic == "Full"

    def test_identifies_best_bic(self):
        # With n_obs=50, ln(50)~3.91, so BIC penalty is larger
        fit1 = _make_fit(objective=100.0, n_params=3, n_obs=50)
        fit2 = _make_fit(objective=80.0, n_params=4, n_obs=50)
        table = compare_fits([fit1, fit2], names=["Base", "Full"])

        # BIC: Base=100+3*ln50=111.73, Full=80+4*ln50=95.63
        assert table.best_bic == "Full"

    def test_single_fit(self):
        fit1 = _make_fit(objective=100.0, n_params=3, n_obs=50)
        table = compare_fits([fit1], names=["Only"])

        assert table.models == ["Only"]
        assert table.best_aic == "Only"
        assert table.best_bic == "Only"

    def test_default_model_names(self):
        fit1 = _make_fit(objective=100.0)
        fit2 = _make_fit(objective=90.0)
        fit3 = _make_fit(objective=80.0)
        table = compare_fits([fit1, fit2, fit3])

        assert table.models == ["Model 1", "Model 2", "Model 3"]

    def test_aic_values_correct(self):
        fit = _make_fit(objective=100.0, n_params=5, n_obs=30)
        table = compare_fits([fit], names=["M"])
        expected_aic = 100.0 + 2 * 5
        assert table.aics[0] == pytest.approx(expected_aic)

    def test_bic_values_correct(self):
        fit = _make_fit(objective=100.0, n_params=5, n_obs=30)
        table = compare_fits([fit], names=["M"])
        expected_bic = 100.0 + 5 * math.log(30)
        assert table.bics[0] == pytest.approx(expected_bic)


class TestLikelihoodRatioTest:
    def test_known_statistic(self):
        fit_full = _make_fit(objective=90.0, n_params=5)
        fit_reduced = _make_fit(objective=100.0, n_params=3)
        result = likelihood_ratio_test(fit_full, fit_reduced, df=2)

        assert result.statistic == pytest.approx(10.0)
        assert result.df == 2
        assert isinstance(result.p_value, float)
        assert 0 <= result.p_value <= 1

    def test_significance_flag(self):
        # Large statistic -> significant
        fit_full = _make_fit(objective=50.0, n_params=5)
        fit_reduced = _make_fit(objective=100.0, n_params=3)
        result = likelihood_ratio_test(fit_full, fit_reduced, df=2)

        assert result.statistic == pytest.approx(50.0)
        assert result.significant is True

    def test_not_significant(self):
        # Tiny difference -> not significant
        fit_full = _make_fit(objective=99.5, n_params=5)
        fit_reduced = _make_fit(objective=100.0, n_params=3)
        result = likelihood_ratio_test(fit_full, fit_reduced, df=2)

        assert result.significant is False

    def test_zero_df_raises(self):
        fit_full = _make_fit(objective=90.0)
        fit_reduced = _make_fit(objective=100.0)
        with pytest.raises(ValueError, match="degrees of freedom"):
            likelihood_ratio_test(fit_full, fit_reduced, df=0)


class TestFormatComparison:
    def test_returns_readable_string(self):
        fit1 = _make_fit(objective=100.0, n_params=3, n_obs=50)
        fit2 = _make_fit(objective=90.0, n_params=4, n_obs=50)
        table = compare_fits([fit1, fit2], names=["Base", "Full"])
        text = format_comparison(table)

        assert isinstance(text, str)
        assert "Base" in text
        assert "Full" in text
        assert "AIC" in text
        assert "BIC" in text
        assert "Objective" in text


class TestBootstrapComparison:
    def test_returns_dict_with_expected_keys(self):
        fit1 = _make_fit(objective=100.0, n_params=3, n_obs=50)
        fit2 = _make_fit(objective=90.0, n_params=4, n_obs=50)
        data = {"time": list(range(50)), "dv": [float(i) for i in range(50)]}
        result = bootstrap_comparison([fit1, fit2], data, n_bootstrap=10, seed=0)

        assert isinstance(result, dict)
        # Should contain pairwise comparison keys
        assert len(result) > 0
        for key, val in result.items():
            assert "ci_lower" in val
            assert "ci_upper" in val
            assert "mean_diff" in val
