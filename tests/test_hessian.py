"""Tests for hessian module: Hessian computation and standard errors."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from nlmixr2.hessian import (
    CovarianceResult,
    compute_correlation,
    compute_covariance,
    compute_hessian,
    compute_rse,
    compute_standard_errors,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _simple_quadratic(params):
    """f(x, y) = 3*x^2 + 2*y^2 + x*y.  Hessian is [[6, 1], [1, 4]]."""
    x, y = params[0], params[1]
    return 3.0 * x**2 + 2.0 * y**2 + x * y


def _diagonal_quadratic(params):
    """f(x, y) = 5*x^2 + 10*y^2.  Hessian is [[10, 0], [0, 20]]."""
    return 5.0 * params[0] ** 2 + 10.0 * params[1] ** 2


# ---------------------------------------------------------------------------
# Tests: compute_hessian
# ---------------------------------------------------------------------------

class TestComputeHessian:
    def test_simple_quadratic(self):
        params = jnp.array([1.0, 2.0])
        H = compute_hessian(_simple_quadratic, params)
        expected = jnp.array([[6.0, 1.0], [1.0, 4.0]])
        assert jnp.allclose(H, expected, atol=1e-5)

    def test_diagonal_quadratic(self):
        params = jnp.array([0.0, 0.0])
        H = compute_hessian(_diagonal_quadratic, params)
        expected = jnp.array([[10.0, 0.0], [0.0, 20.0]])
        assert jnp.allclose(H, expected, atol=1e-5)

    def test_hessian_is_symmetric(self):
        params = jnp.array([1.0, 2.0])
        H = compute_hessian(_simple_quadratic, params)
        assert jnp.allclose(H, H.T, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: compute_covariance
# ---------------------------------------------------------------------------

class TestComputeCovariance:
    def test_inverse_of_hessian(self):
        H = jnp.array([[6.0, 1.0], [1.0, 4.0]])
        cov = compute_covariance(H)
        # cov should be the inverse of H
        identity = H @ cov
        assert jnp.allclose(identity, jnp.eye(2), atol=1e-5)

    def test_inverse_diagonal(self):
        H = jnp.array([[10.0, 0.0], [0.0, 20.0]])
        cov = compute_covariance(H)
        expected = jnp.array([[0.1, 0.0], [0.0, 0.05]])
        assert jnp.allclose(cov, expected, atol=1e-6)

    def test_singular_hessian_returns_nan(self):
        """A singular Hessian should be handled gracefully (NaN/Inf, no crash)."""
        H = jnp.array([[1.0, 1.0], [1.0, 1.0]])  # rank 1
        cov = compute_covariance(H)
        # Should not raise; result will contain inf or nan
        assert cov.shape == (2, 2)


# ---------------------------------------------------------------------------
# Tests: compute_standard_errors
# ---------------------------------------------------------------------------

class TestComputeStandardErrors:
    def test_sqrt_of_diagonal(self):
        cov = jnp.array([[4.0, 1.0], [1.0, 9.0]])
        se = compute_standard_errors(cov)
        expected = jnp.array([2.0, 3.0])
        assert jnp.allclose(se, expected, atol=1e-6)

    def test_single_param(self):
        cov = jnp.array([[16.0]])
        se = compute_standard_errors(cov)
        assert jnp.allclose(se, jnp.array([4.0]), atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: compute_correlation
# ---------------------------------------------------------------------------

class TestComputeCorrelation:
    def test_diagonal_is_ones(self):
        cov = jnp.array([[4.0, 1.0], [1.0, 9.0]])
        corr = compute_correlation(cov)
        assert jnp.allclose(jnp.diag(corr), jnp.ones(2), atol=1e-6)

    def test_off_diagonal_bounded(self):
        cov = jnp.array([[4.0, 1.0], [1.0, 9.0]])
        corr = compute_correlation(cov)
        # off-diagonal should be 1/(2*3) = 1/6
        assert jnp.allclose(corr[0, 1], 1.0 / 6.0, atol=1e-6)
        assert jnp.allclose(corr[1, 0], 1.0 / 6.0, atol=1e-6)

    def test_identity_covariance(self):
        cov = jnp.eye(3)
        corr = compute_correlation(cov)
        assert jnp.allclose(corr, jnp.eye(3), atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: compute_rse
# ---------------------------------------------------------------------------

class TestComputeRSE:
    def test_rse_calculation(self):
        se = jnp.array([0.5, 1.0])
        estimates = jnp.array([5.0, 10.0])
        rse = compute_rse(se, estimates)
        expected = jnp.array([10.0, 10.0])  # (0.5/5)*100, (1/10)*100
        assert jnp.allclose(rse, expected, atol=1e-6)

    def test_rse_zero_estimate(self):
        """RSE with a zero estimate should not crash."""
        se = jnp.array([0.5])
        estimates = jnp.array([0.0])
        rse = compute_rse(se, estimates)
        assert rse.shape == (1,)
        # Result will be inf, which is acceptable


# ---------------------------------------------------------------------------
# Tests: CovarianceResult
# ---------------------------------------------------------------------------

class TestCovarianceResult:
    def test_fields(self):
        H = jnp.eye(2)
        cov = jnp.eye(2)
        corr = jnp.eye(2)
        se = jnp.ones(2)
        rse = jnp.ones(2) * 100.0
        result = CovarianceResult(
            hessian=H,
            covariance=cov,
            correlation=corr,
            standard_errors=se,
            rse=rse,
            condition_number=1.0,
        )
        assert result.hessian is H
        assert result.covariance is cov
        assert result.correlation is corr
        assert result.standard_errors is se
        assert result.rse is rse
        assert result.condition_number == 1.0

    def test_condition_number_value(self):
        """Condition number of identity is 1.0."""
        H = jnp.eye(3) * 5.0
        cond = float(jnp.linalg.cond(H))
        assert pytest.approx(cond, abs=1e-4) == 1.0

    def test_condition_number_ill_conditioned(self):
        """An ill-conditioned Hessian should have a large condition number."""
        H = jnp.array([[1.0, 0.0], [0.0, 1e-8]])
        cond = float(jnp.linalg.cond(H))
        assert cond > 1e6
