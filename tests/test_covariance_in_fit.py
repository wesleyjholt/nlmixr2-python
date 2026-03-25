"""Tests for Hessian/covariance integration into the fit pipeline."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from nlmixr2 import ini, model, nlmixr2
from nlmixr2.api import NLMIXRFit, NLMIXRModel
from nlmixr2.hessian import CovarianceResult


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


def _make_data(n_subjects=3, n_times=10):
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
# Tests
# ---------------------------------------------------------------------------

class TestFOCECovarianceInFit:
    """Test that FOCE fit automatically includes covariance_result."""

    def test_foce_fit_has_covariance_result(self):
        """FOCE fit should have a covariance_result attribute."""
        data = _make_data(n_subjects=2, n_times=8)
        fit = nlmixr2(
            _algebraic_model(),
            data=data,
            est="foce",
            control={"maxiter": 5},
        )
        assert isinstance(fit, NLMIXRFit)
        assert fit.covariance_result is not None
        assert isinstance(fit.covariance_result, CovarianceResult)

    def test_covariance_result_has_standard_errors(self):
        """covariance_result should contain standard_errors array."""
        data = _make_data(n_subjects=2, n_times=8)
        fit = nlmixr2(
            _algebraic_model(),
            data=data,
            est="foce",
            control={"maxiter": 5},
        )
        se = fit.covariance_result.standard_errors
        assert se is not None
        assert se.shape == (2,)  # 2 parameters: A, ke
        # Standard errors should be positive
        assert jnp.all(se > 0)

    def test_covariance_result_has_correlation_matrix(self):
        """covariance_result should contain a correlation matrix."""
        data = _make_data(n_subjects=2, n_times=8)
        fit = nlmixr2(
            _algebraic_model(),
            data=data,
            est="foce",
            control={"maxiter": 5},
        )
        corr = fit.covariance_result.correlation
        assert corr is not None
        assert corr.shape == (2, 2)
        # Diagonal should be 1.0
        np.testing.assert_allclose(jnp.diag(corr), 1.0, atol=1e-5)
        # Off-diagonal should be between -1 and 1
        assert jnp.all(corr >= -1.0 - 1e-5)
        assert jnp.all(corr <= 1.0 + 1e-5)

    def test_rse_values_are_finite(self):
        """RSE values should be finite numbers."""
        data = _make_data(n_subjects=2, n_times=8)
        fit = nlmixr2(
            _algebraic_model(),
            data=data,
            est="foce",
            control={"maxiter": 5},
        )
        rse = fit.covariance_result.rse
        assert rse is not None
        assert rse.shape == (2,)
        assert jnp.all(jnp.isfinite(rse))

    def test_condition_number_is_finite(self):
        """Condition number should be a finite positive number."""
        data = _make_data(n_subjects=2, n_times=8)
        fit = nlmixr2(
            _algebraic_model(),
            data=data,
            est="foce",
            control={"maxiter": 5},
        )
        cn = fit.covariance_result.condition_number
        assert math.isfinite(cn)
        assert cn > 0

    def test_mock_fit_has_none_covariance_result(self):
        """Mock estimator should produce a fit with covariance_result=None."""
        data = _make_data(n_subjects=2, n_times=8)
        fit = nlmixr2(
            _algebraic_model(),
            data=data,
            est="mock",
        )
        assert isinstance(fit, NLMIXRFit)
        assert fit.covariance_result is None


class TestToDictCovariance:
    """Test that to_dict() includes covariance info."""

    def test_to_dict_includes_covariance_info(self):
        """to_dict() should include SE and RSE per parameter."""
        data = _make_data(n_subjects=2, n_times=8)
        fit = nlmixr2(
            _algebraic_model(),
            data=data,
            est="foce",
            control={"maxiter": 5},
        )
        d = fit.to_dict()
        assert "covariance" in d
        cov_info = d["covariance"]
        assert "standard_errors" in cov_info
        assert "rse" in cov_info
        assert "condition_number" in cov_info
        # SE and RSE should be dicts keyed by parameter name
        assert isinstance(cov_info["standard_errors"], dict)
        assert isinstance(cov_info["rse"], dict)
        assert "A" in cov_info["standard_errors"]
        assert "ke" in cov_info["standard_errors"]

    def test_to_dict_none_covariance(self):
        """to_dict() should include covariance=None for mock fits."""
        data = _make_data(n_subjects=2, n_times=8)
        fit = nlmixr2(
            _algebraic_model(),
            data=data,
            est="mock",
        )
        d = fit.to_dict()
        assert "covariance" in d
        assert d["covariance"] is None
