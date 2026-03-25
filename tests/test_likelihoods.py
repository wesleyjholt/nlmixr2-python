"""Tests for non-Gaussian endpoint likelihoods (count, categorical data)."""

from __future__ import annotations

import math

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from nlmixr2.likelihoods import (
    binomial_log_likelihood,
    negative_binomial_log_likelihood,
    ordinal_log_likelihood,
    poisson_log_likelihood,
    select_likelihood,
)


# ---------------------------------------------------------------------------
# Poisson log-likelihood
# ---------------------------------------------------------------------------

class TestPoissonLogLikelihood:
    """Tests for poisson_log_likelihood."""

    def test_known_lambda(self):
        """Compare to manual computation: k*log(lam) - lam - log(k!)."""
        dv = jnp.array([0.0, 1.0, 2.0, 5.0])
        lam = jnp.array([3.0, 3.0, 3.0, 3.0])

        result = poisson_log_likelihood(dv, lam)

        for i in range(4):
            k = int(dv[i])
            expected = k * math.log(3.0) - 3.0 - math.lgamma(k + 1)
            np.testing.assert_allclose(float(result[i]), expected, rtol=1e-5)

    def test_scipy_comparison(self):
        """Compare to scipy.stats.poisson if available."""
        try:
            from scipy.stats import poisson as sp_poisson
        except ImportError:
            pytest.skip("scipy not available")

        dv = jnp.array([0.0, 1.0, 3.0, 7.0, 10.0])
        lam = jnp.array([2.0, 2.0, 5.0, 5.0, 10.0])
        result = poisson_log_likelihood(dv, lam)

        for i in range(5):
            expected = sp_poisson.logpmf(int(dv[i]), float(lam[i]))
            np.testing.assert_allclose(float(result[i]), expected, rtol=1e-5)

    def test_log_likelihood_is_negative(self):
        """Log-likelihood of a probability must be <= 0."""
        dv = jnp.array([0.0, 1.0, 5.0, 10.0, 20.0])
        lam = jnp.array([1.0, 3.0, 5.0, 10.0, 15.0])
        result = poisson_log_likelihood(dv, lam)
        assert jnp.all(result <= 0.0 + 1e-10)

    def test_returns_jax_array(self):
        dv = jnp.array([1.0])
        lam = jnp.array([2.0])
        result = poisson_log_likelihood(dv, lam)
        assert isinstance(result, jax.Array)


# ---------------------------------------------------------------------------
# Negative binomial log-likelihood
# ---------------------------------------------------------------------------

class TestNegativeBinomialLogLikelihood:
    """Tests for negative_binomial_log_likelihood."""

    def test_reduces_to_poisson_large_size(self):
        """As size -> inf, NB(mu, size) -> Poisson(mu)."""
        dv = jnp.array([0.0, 1.0, 3.0, 5.0, 10.0])
        mu = jnp.array([3.0, 3.0, 3.0, 3.0, 10.0])
        size = jnp.array([1e6, 1e6, 1e6, 1e6, 1e6])

        nb_ll = negative_binomial_log_likelihood(dv, mu, size)
        pois_ll = poisson_log_likelihood(dv, mu)

        np.testing.assert_allclose(nb_ll, pois_ll, atol=1e-3)

    def test_log_likelihood_is_negative(self):
        dv = jnp.array([0.0, 1.0, 5.0])
        mu = jnp.array([2.0, 2.0, 5.0])
        size = jnp.array([1.0, 1.0, 2.0])
        result = negative_binomial_log_likelihood(dv, mu, size)
        assert jnp.all(result <= 0.0 + 1e-10)

    def test_returns_jax_array(self):
        dv = jnp.array([1.0])
        mu = jnp.array([2.0])
        size = jnp.array([1.0])
        result = negative_binomial_log_likelihood(dv, mu, size)
        assert isinstance(result, jax.Array)


# ---------------------------------------------------------------------------
# Binomial log-likelihood
# ---------------------------------------------------------------------------

class TestBinomialLogLikelihood:
    """Tests for binomial_log_likelihood."""

    def test_p_half(self):
        """With p=0.5, n=1, P(k) = 0.5 for k in {0,1}, so log(P) = log(0.5)."""
        dv = jnp.array([0.0, 1.0])
        n = jnp.array([1.0, 1.0])
        p = jnp.array([0.5, 0.5])

        result = binomial_log_likelihood(dv, n, p)
        expected = jnp.array([math.log(0.5), math.log(0.5)])
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_boundary_p_zero(self):
        """p=0 => P(k=0)=1, log(P)=0; P(k=1)=-inf."""
        dv = jnp.array([0.0])
        n = jnp.array([1.0])
        p = jnp.array([0.0])
        result = binomial_log_likelihood(dv, n, p)
        np.testing.assert_allclose(float(result[0]), 0.0, atol=1e-5)

    def test_boundary_p_one(self):
        """p=1 => P(k=n)=1, log(P)=0."""
        dv = jnp.array([5.0])
        n = jnp.array([5.0])
        p = jnp.array([1.0])
        result = binomial_log_likelihood(dv, n, p)
        np.testing.assert_allclose(float(result[0]), 0.0, atol=1e-5)

    def test_known_values(self):
        """Binomial(n=10, k=3, p=0.3): manual computation."""
        k, n_val, p_val = 3, 10, 0.3
        # log(C(10,3)) + 3*log(0.3) + 7*log(0.7)
        expected = (
            math.lgamma(11) - math.lgamma(4) - math.lgamma(8)
            + 3 * math.log(0.3) + 7 * math.log(0.7)
        )
        dv = jnp.array([3.0])
        n = jnp.array([10.0])
        p = jnp.array([0.3])
        result = binomial_log_likelihood(dv, n, p)
        np.testing.assert_allclose(float(result[0]), expected, rtol=1e-5)

    def test_returns_jax_array(self):
        dv = jnp.array([1.0])
        n = jnp.array([1.0])
        p = jnp.array([0.5])
        result = binomial_log_likelihood(dv, n, p)
        assert isinstance(result, jax.Array)


# ---------------------------------------------------------------------------
# Ordinal log-likelihood
# ---------------------------------------------------------------------------

class TestOrdinalLogLikelihood:
    """Tests for ordinal_log_likelihood."""

    def test_known_probabilities(self):
        """With 3 categories, cumulative_probs = [0.2, 0.7, 1.0].
        P(Y=0) = 0.2, P(Y=1) = 0.5, P(Y=2) = 0.3.
        """
        cumulative_probs = jnp.array([
            [0.2, 0.7, 1.0],
            [0.2, 0.7, 1.0],
            [0.2, 0.7, 1.0],
        ])
        dv = jnp.array([0, 1, 2])

        result = ordinal_log_likelihood(dv, cumulative_probs)

        expected = jnp.array([
            math.log(0.2),
            math.log(0.5),
            math.log(0.3),
        ])
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_two_categories(self):
        """Binary case: cumulative_probs = [0.6, 1.0]."""
        cumulative_probs = jnp.array([
            [0.6, 1.0],
            [0.6, 1.0],
        ])
        dv = jnp.array([0, 1])

        result = ordinal_log_likelihood(dv, cumulative_probs)
        expected = jnp.array([math.log(0.6), math.log(0.4)])
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_returns_jax_array(self):
        cumulative_probs = jnp.array([[0.5, 1.0]])
        dv = jnp.array([0])
        result = ordinal_log_likelihood(dv, cumulative_probs)
        assert isinstance(result, jax.Array)


# ---------------------------------------------------------------------------
# select_likelihood
# ---------------------------------------------------------------------------

class TestSelectLikelihood:
    """Tests for select_likelihood."""

    def test_returns_poisson(self):
        fn = select_likelihood("poisson")
        assert fn is poisson_log_likelihood

    def test_returns_negbin(self):
        fn = select_likelihood("negbin")
        assert fn is negative_binomial_log_likelihood

    def test_returns_binomial(self):
        fn = select_likelihood("binomial")
        assert fn is binomial_log_likelihood

    def test_returns_ordinal(self):
        fn = select_likelihood("ordinal")
        assert fn is ordinal_log_likelihood

    def test_returns_normal(self):
        """Normal family should return a callable (the censored_normal_log_likelihood)."""
        fn = select_likelihood("normal")
        assert callable(fn)

    def test_invalid_family_raises(self):
        with pytest.raises(ValueError, match="Unknown likelihood family"):
            select_likelihood("gamma")
