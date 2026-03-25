"""Tests for enriched NLMIXRFit fields: AIC, BIC, etas, predictions, shrinkage, timing."""

from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

from nlmixr2 import ini, model, nlmixr2
from nlmixr2.api import NLMIXRFit, NLMIXRModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _algebraic_model():
    """A simple algebraic (non-ODE) model: y = A * exp(-ke * t)."""
    return NLMIXRModel(
        ini=ini({"A": 10.0, "ke": 0.5}),
        model=model([
            "cp = A * exp(-ke * t)",
            "cp ~ add(A)",
        ]),
    )


def _make_data(n_subjects=2, n_times=10):
    """Generate simple data with required columns."""
    all_ids = []
    all_times = []
    all_dv = []
    for subj in range(n_subjects):
        times = jnp.linspace(0.5, 5.0, n_times)
        dv = 10.0 * jnp.exp(-0.5 * times) + 0.1 * subj
        all_ids.append(jnp.full(n_times, subj, dtype=jnp.int32))
        all_times.append(times)
        all_dv.append(dv)
    return {
        "id": jnp.concatenate(all_ids),
        "time": jnp.concatenate(all_times),
        "dv": jnp.concatenate(all_dv),
    }


# ---------------------------------------------------------------------------
# FOCE enrichment tests
# ---------------------------------------------------------------------------

class TestFoceEnrichment:
    @pytest.fixture(scope="class")
    def foce_fit(self):
        data = _make_data(n_subjects=3, n_times=8)
        return nlmixr2(
            _algebraic_model(),
            data=data,
            est="foce",
            control={"maxiter": 5},
        )

    def test_aic_is_finite(self, foce_fit):
        """FOCE fit should include a finite AIC."""
        assert foce_fit.aic is not None
        assert math.isfinite(foce_fit.aic)

    def test_bic_is_finite(self, foce_fit):
        """FOCE fit should include a finite BIC."""
        assert foce_fit.bic is not None
        assert math.isfinite(foce_fit.bic)

    def test_aic_formula(self, foce_fit):
        """AIC should equal objective + 2 * n_params."""
        expected = foce_fit.objective + 2 * foce_fit.parameter_count
        assert foce_fit.aic == pytest.approx(expected)

    def test_bic_formula(self, foce_fit):
        """BIC should equal objective + n_params * ln(n_obs)."""
        expected = foce_fit.objective + foce_fit.parameter_count * math.log(
            foce_fit.n_observations
        )
        assert foce_fit.bic == pytest.approx(expected)

    def test_etas_present_and_correct_shape(self, foce_fit):
        """etas dict should contain an array with shape (n_subjects, n_params)."""
        assert foce_fit.etas is not None
        assert "values" in foce_fit.etas
        arr = foce_fit.etas["values"]
        # 3 subjects, 2 params
        assert arr.shape == (3, 2)

    def test_elapsed_seconds_positive(self, foce_fit):
        """elapsed_seconds should be a positive number."""
        assert foce_fit.elapsed_seconds is not None
        assert foce_fit.elapsed_seconds > 0.0

    def test_shrinkage_present(self, foce_fit):
        """shrinkage dict should be present for foce fit."""
        assert foce_fit.shrinkage is not None
        assert isinstance(foce_fit.shrinkage, dict)

    def test_to_dict_includes_new_fields(self, foce_fit):
        """to_dict output should contain all enrichment fields."""
        d = foce_fit.to_dict()
        assert "aic" in d
        assert "bic" in d
        assert "etas" in d
        assert "predictions" in d
        assert "shrinkage" in d
        assert "elapsed_seconds" in d


# ---------------------------------------------------------------------------
# Backward-compatibility: mock fit
# ---------------------------------------------------------------------------

class TestMockFitBackwardCompat:
    def test_mock_fit_has_none_enrichment_fields(self):
        """A mock fit (no real estimation) should have None for enrichment fields."""
        data = _make_data(n_subjects=2, n_times=5)
        fit = nlmixr2(
            _algebraic_model(),
            data=data,
            est="mock",
        )
        assert fit.aic is None
        assert fit.bic is None
        assert fit.etas is None
        assert fit.predictions is None
        assert fit.shrinkage is None
        assert fit.elapsed_seconds is None


# ---------------------------------------------------------------------------
# SAEM enrichment tests
# ---------------------------------------------------------------------------

class TestSaemEnrichment:
    @pytest.fixture(scope="class")
    def saem_fit(self):
        data = _make_data(n_subjects=3, n_times=8)
        return nlmixr2(
            _algebraic_model(),
            data=data,
            est="saem",
            control={"n_burn": 3, "n_em": 5},
        )

    def test_aic_is_finite(self, saem_fit):
        """SAEM fit should include a finite AIC."""
        assert saem_fit.aic is not None
        assert math.isfinite(saem_fit.aic)

    def test_bic_is_finite(self, saem_fit):
        """SAEM fit should include a finite BIC."""
        assert saem_fit.bic is not None
        assert math.isfinite(saem_fit.bic)

    def test_aic_formula(self, saem_fit):
        """AIC should equal objective + 2 * n_params."""
        expected = saem_fit.objective + 2 * saem_fit.parameter_count
        assert saem_fit.aic == pytest.approx(expected)

    def test_elapsed_seconds_positive(self, saem_fit):
        """elapsed_seconds should be a positive number."""
        assert saem_fit.elapsed_seconds is not None
        assert saem_fit.elapsed_seconds > 0.0

    def test_etas_present(self, saem_fit):
        """SAEM fit should include etas."""
        assert saem_fit.etas is not None
        assert "values" in saem_fit.etas
