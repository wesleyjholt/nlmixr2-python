"""Tests for censoring/BLQ support in pharmacometric data."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.stats.norm as jax_norm
import numpy as np
import pytest

from nlmixr2.censoring import (
    CensoringSpec,
    apply_censoring,
    censored_normal_log_likelihood,
    has_censoring,
    m3_method,
)


# ---------------------------------------------------------------------------
# Helper: manual normal log-likelihood for a single uncensored observation
# ---------------------------------------------------------------------------

def _manual_normal_ll(dv, pred, sigma):
    """Standard normal log-likelihood: log N(dv | pred, sigma^2)."""
    return -0.5 * jnp.log(2.0 * jnp.pi * sigma ** 2) - 0.5 * ((dv - pred) / sigma) ** 2


# ---------------------------------------------------------------------------
# censored_normal_log_likelihood
# ---------------------------------------------------------------------------

class TestCensoredNormalLogLikelihood:
    """Tests for censored_normal_log_likelihood."""

    def test_uncensored_gives_standard_normal_ll(self):
        dv = jnp.array([5.0, 3.0, 7.0])
        pred = jnp.array([4.8, 3.1, 6.9])
        sigma = 1.0
        cens = jnp.array([0, 0, 0])
        limit = jnp.array([0.0, 0.0, 0.0])

        result = censored_normal_log_likelihood(dv, pred, sigma, cens, limit)
        expected = jax.vmap(lambda d, p: _manual_normal_ll(d, p, sigma))(dv, pred)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_left_censored_blq(self):
        """Left censored (cens=1): log(Phi((limit - pred) / sigma))."""
        dv = jnp.array([0.0])  # DV doesn't matter for censored
        pred = jnp.array([2.0])
        sigma = 1.0
        cens = jnp.array([1])
        limit = jnp.array([1.0])  # LOQ = 1.0

        result = censored_normal_log_likelihood(dv, pred, sigma, cens, limit)
        expected = jnp.log(jax_norm.cdf((1.0 - 2.0) / 1.0))

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_right_censored(self):
        """Right censored (cens=-1): log(1 - Phi((limit - pred) / sigma))."""
        dv = jnp.array([0.0])
        pred = jnp.array([2.0])
        sigma = 1.0
        cens = jnp.array([-1])
        limit = jnp.array([5.0])

        result = censored_normal_log_likelihood(dv, pred, sigma, cens, limit)
        expected = jnp.log(1.0 - jax_norm.cdf((5.0 - 2.0) / 1.0))

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_mixed_censored_uncensored(self):
        """Mix of uncensored, left-censored, and right-censored."""
        dv = jnp.array([5.0, 0.0, 0.0, 3.0])
        pred = jnp.array([4.8, 2.0, 2.0, 3.1])
        sigma = 0.5
        cens = jnp.array([0, 1, -1, 0])
        limit = jnp.array([0.0, 1.0, 5.0, 0.0])

        result = censored_normal_log_likelihood(dv, pred, sigma, cens, limit)

        # Manually compute each
        expected_0 = _manual_normal_ll(5.0, 4.8, 0.5)
        expected_1 = jnp.log(jax_norm.cdf((1.0 - 2.0) / 0.5))
        expected_2 = jnp.log(jnp.clip(1.0 - jax_norm.cdf((5.0 - 2.0) / 0.5), min=1e-30))
        expected_3 = _manual_normal_ll(3.0, 3.1, 0.5)
        expected = jnp.array([expected_0, expected_1, expected_2, expected_3])

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_limit_zero_edge_case(self):
        """Left censored with limit=0 (common for BLQ with LOQ=0)."""
        dv = jnp.array([0.0])
        pred = jnp.array([1.0])
        sigma = 1.0
        cens = jnp.array([1])
        limit = jnp.array([0.0])

        result = censored_normal_log_likelihood(dv, pred, sigma, cens, limit)
        expected = jnp.log(jax_norm.cdf((0.0 - 1.0) / 1.0))

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_all_observations_censored(self):
        """All observations are left-censored."""
        dv = jnp.array([0.0, 0.0, 0.0])
        pred = jnp.array([2.0, 3.0, 1.5])
        sigma = 1.0
        cens = jnp.array([1, 1, 1])
        limit = jnp.array([1.0, 1.0, 1.0])

        result = censored_normal_log_likelihood(dv, pred, sigma, cens, limit)
        for i in range(3):
            expected_i = jnp.log(jax_norm.cdf((limit[i] - pred[i]) / sigma))
            np.testing.assert_allclose(float(result[i]), float(expected_i), rtol=1e-5)


# ---------------------------------------------------------------------------
# has_censoring
# ---------------------------------------------------------------------------

class TestHasCensoring:
    """Tests for has_censoring."""

    def test_no_cens_column(self):
        data = {"id": jnp.array([1, 1]), "time": jnp.array([0.0, 1.0]), "dv": jnp.array([5.0, 3.0])}
        assert has_censoring(data) is False

    def test_cens_column_all_zero(self):
        data = {
            "id": jnp.array([1, 1]),
            "time": jnp.array([0.0, 1.0]),
            "dv": jnp.array([5.0, 3.0]),
            "cens": jnp.array([0, 0]),
        }
        assert has_censoring(data) is False

    def test_cens_column_with_censored_data(self):
        data = {
            "id": jnp.array([1, 1]),
            "time": jnp.array([0.0, 1.0]),
            "dv": jnp.array([5.0, 0.0]),
            "cens": jnp.array([0, 1]),
        }
        assert has_censoring(data) is True

    def test_cens_column_with_right_censored(self):
        data = {
            "id": jnp.array([1, 1]),
            "time": jnp.array([0.0, 1.0]),
            "dv": jnp.array([5.0, 0.0]),
            "cens": jnp.array([0, -1]),
        }
        assert has_censoring(data) is True


# ---------------------------------------------------------------------------
# apply_censoring
# ---------------------------------------------------------------------------

class TestApplyCensoring:
    """Tests for apply_censoring."""

    def test_extracts_correct_arrays(self):
        data = {
            "id": jnp.array([1, 1, 2]),
            "time": jnp.array([0.0, 1.0, 0.0]),
            "dv": jnp.array([5.0, 0.0, 3.0]),
            "cens": jnp.array([0, 1, 0]),
            "limit": jnp.array([0.0, 1.0, 0.0]),
        }
        spec = CensoringSpec(cens_column="cens", limit_column="limit")
        result = apply_censoring(data, spec)

        np.testing.assert_array_equal(result["cens"], jnp.array([0, 1, 0]))
        np.testing.assert_array_equal(result["limit"], jnp.array([0.0, 1.0, 0.0]))

    def test_defaults_when_columns_missing(self):
        data = {
            "id": jnp.array([1, 1]),
            "time": jnp.array([0.0, 1.0]),
            "dv": jnp.array([5.0, 3.0]),
        }
        spec = CensoringSpec()
        result = apply_censoring(data, spec)

        # No cens column => all uncensored (zeros)
        np.testing.assert_array_equal(result["cens"], jnp.array([0, 0]))
        np.testing.assert_array_equal(result["limit"], jnp.array([0.0, 0.0]))

    def test_custom_column_names(self):
        data = {
            "id": jnp.array([1]),
            "time": jnp.array([0.0]),
            "dv": jnp.array([5.0]),
            "my_cens": jnp.array([1]),
            "my_loq": jnp.array([0.5]),
        }
        spec = CensoringSpec(cens_column="my_cens", limit_column="my_loq")
        result = apply_censoring(data, spec)

        np.testing.assert_array_equal(result["cens"], jnp.array([1]))
        np.testing.assert_array_equal(result["limit"], jnp.array([0.5]))


# ---------------------------------------------------------------------------
# m3_method
# ---------------------------------------------------------------------------

class TestM3Method:
    """Tests for M3 method (Beal 2001)."""

    def test_m3_uncensored_matches_normal_ll(self):
        """With no censoring, M3 should reduce to normal log-likelihood."""
        dv = jnp.array([5.0, 3.0])
        pred = jnp.array([4.8, 3.1])
        sigma = 1.0
        cens = jnp.array([0, 0])
        limit = jnp.array([0.0, 0.0])

        result = m3_method(dv, pred, sigma, cens, limit)
        expected = jnp.sum(jax.vmap(lambda d, p: _manual_normal_ll(d, p, sigma))(dv, pred))

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_m3_mixed_matches_manual(self):
        """M3 with mixed data should sum continuous + censored contributions."""
        dv = jnp.array([5.0, 0.0])
        pred = jnp.array([4.8, 2.0])
        sigma = 1.0
        cens = jnp.array([0, 1])
        limit = jnp.array([0.0, 1.0])

        result = m3_method(dv, pred, sigma, cens, limit)

        # Continuous contribution for obs 0
        ll_cont = _manual_normal_ll(5.0, 4.8, 1.0)
        # Censored contribution for obs 1
        ll_cens = jnp.log(jax_norm.cdf((1.0 - 2.0) / 1.0))
        expected = ll_cont + ll_cens

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_m3_all_censored(self):
        """M3 with all censored observations."""
        dv = jnp.array([0.0, 0.0])
        pred = jnp.array([2.0, 3.0])
        sigma = 1.0
        cens = jnp.array([1, 1])
        limit = jnp.array([1.0, 1.0])

        result = m3_method(dv, pred, sigma, cens, limit)
        expected = (
            jnp.log(jax_norm.cdf((1.0 - 2.0) / 1.0))
            + jnp.log(jax_norm.cdf((1.0 - 3.0) / 1.0))
        )

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_m3_right_censored(self):
        """M3 with right-censored observation."""
        dv = jnp.array([0.0])
        pred = jnp.array([2.0])
        sigma = 1.0
        cens = jnp.array([-1])
        limit = jnp.array([5.0])

        result = m3_method(dv, pred, sigma, cens, limit)
        expected = jnp.log(1.0 - jax_norm.cdf((5.0 - 2.0) / 1.0))

        np.testing.assert_allclose(result, expected, rtol=1e-5)
