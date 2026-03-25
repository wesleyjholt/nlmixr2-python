"""Tests for nlmixr2.vpc — Visual Predictive Check data generation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nlmixr2.vpc import VPCResult, bin_times, compute_quantiles, vpc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _constant_model(params, times):
    """Model that always returns a constant value (params['A'])."""
    return jnp.full_like(times, params["A"])


def _linear_model(params, times):
    """Model: y = A + B * t."""
    return params["A"] + params["B"] * times


def _make_simple_data(n_subjects=5, n_times=10):
    """Create simple dataset with multiple subjects."""
    ids = jnp.repeat(jnp.arange(n_subjects), n_times)
    times = jnp.tile(jnp.linspace(0.0, 10.0, n_times), n_subjects)
    dv = jnp.ones_like(times) * 5.0  # constant observed values
    return {"id": ids, "time": times, "dv": dv}


# ---------------------------------------------------------------------------
# Tests for vpc()
# ---------------------------------------------------------------------------

class TestVPC:
    def test_returns_vpc_result(self):
        data = _make_simple_data()
        omega = jnp.eye(1) * 0.01
        result = vpc(_constant_model, data, n_sim=10, omega=omega, sigma=0.01, seed=42)
        assert isinstance(result, VPCResult)

    def test_simulated_quantiles_has_correct_keys(self):
        data = _make_simple_data()
        omega = jnp.eye(1) * 0.01
        result = vpc(_constant_model, data, n_sim=10, omega=omega, sigma=0.01, seed=42)
        assert "time" in result.simulated_quantiles
        assert "lo" in result.simulated_quantiles
        assert "median" in result.simulated_quantiles
        assert "hi" in result.simulated_quantiles

    def test_median_between_lo_and_hi(self):
        data = _make_simple_data()
        omega = jnp.eye(1) * 0.01
        result = vpc(_constant_model, data, n_sim=50, omega=omega, sigma=0.1, seed=42)
        lo = result.simulated_quantiles["lo"]
        med = result.simulated_quantiles["median"]
        hi = result.simulated_quantiles["hi"]
        assert np.all(np.array(lo) <= np.array(med) + 1e-10)
        assert np.all(np.array(med) <= np.array(hi) + 1e-10)

    def test_n_sim_preserved(self):
        data = _make_simple_data()
        omega = jnp.eye(1) * 0.01
        result = vpc(_constant_model, data, n_sim=77, omega=omega, sigma=0.01, seed=0)
        assert result.n_sim == 77

    def test_pi_preserved(self):
        data = _make_simple_data()
        omega = jnp.eye(1) * 0.01
        pi = (0.1, 0.5, 0.9)
        result = vpc(_constant_model, data, n_sim=10, omega=omega, sigma=0.01, seed=0, pi=pi)
        assert result.pi == pi

    def test_observed_data_preserved(self):
        data = _make_simple_data()
        omega = jnp.eye(1) * 0.01
        result = vpc(_constant_model, data, n_sim=10, omega=omega, sigma=0.01, seed=0)
        np.testing.assert_allclose(result.observed["time"], np.array(data["time"]))
        np.testing.assert_allclose(result.observed["dv"], np.array(data["dv"]))

    def test_constant_model_quantiles_close(self):
        """With zero omega and zero sigma, all quantiles should equal the constant."""
        data = _make_simple_data()
        omega = jnp.eye(1) * 1e-10
        result = vpc(
            _constant_model, data, n_sim=50, omega=omega, sigma=1e-10, seed=0,
            pi=(0.05, 0.5, 0.95),
        )
        med = np.array(result.simulated_quantiles["median"])
        np.testing.assert_allclose(med, 5.0, atol=0.1)
        lo = np.array(result.simulated_quantiles["lo"])
        hi = np.array(result.simulated_quantiles["hi"])
        np.testing.assert_allclose(lo, 5.0, atol=0.1)
        np.testing.assert_allclose(hi, 5.0, atol=0.1)

    def test_more_sims_tighter_intervals(self):
        """With many simulations, the median should be closer to truth."""
        data = _make_simple_data()
        omega = jnp.eye(1) * 0.1
        r_few = vpc(_constant_model, data, n_sim=20, omega=omega, sigma=0.5, seed=0)
        r_many = vpc(_constant_model, data, n_sim=500, omega=omega, sigma=0.5, seed=0)
        # With more sims, the estimate of the median should be more stable (closer to 5.0)
        med_few = np.array(r_few.simulated_quantiles["median"])
        med_many = np.array(r_many.simulated_quantiles["median"])
        err_few = np.mean(np.abs(med_few - 5.0))
        err_many = np.mean(np.abs(med_many - 5.0))
        # The many-sim median should be at least as good (or close)
        assert err_many < err_few + 0.5  # generous tolerance

    def test_custom_seed_reproducibility(self):
        data = _make_simple_data()
        omega = jnp.eye(1) * 0.1
        r1 = vpc(_constant_model, data, n_sim=20, omega=omega, sigma=0.5, seed=123)
        r2 = vpc(_constant_model, data, n_sim=20, omega=omega, sigma=0.5, seed=123)
        np.testing.assert_allclose(
            r1.simulated_quantiles["median"],
            r2.simulated_quantiles["median"],
        )

    def test_with_multi_param_model(self):
        """Test with a model that has 2 parameters and 2x2 omega."""
        data = _make_simple_data()
        # Overwrite dv to match linear model: A=1, B=0.5
        times = np.array(data["time"])
        data["dv"] = jnp.array(1.0 + 0.5 * times)
        omega = jnp.eye(2) * 0.01
        result = vpc(_linear_model, data, n_sim=30, omega=omega, sigma=0.01, seed=0)
        assert isinstance(result, VPCResult)
        assert len(result.simulated_quantiles["time"]) > 0


# ---------------------------------------------------------------------------
# Tests for bin_times()
# ---------------------------------------------------------------------------

class TestBinTimes:
    def test_time_method_returns_correct_count(self):
        times = jnp.linspace(0.0, 10.0, 100)
        centers = bin_times(times, method="time", n_bins=5)
        assert len(centers) == 5

    def test_time_method_covers_range(self):
        times = jnp.linspace(0.0, 10.0, 100)
        centers = bin_times(times, method="time", n_bins=5)
        assert float(centers[0]) >= 0.0
        assert float(centers[-1]) <= 10.0

    def test_time_method_sorted(self):
        times = jnp.array([5.0, 1.0, 9.0, 3.0, 7.0])
        centers = bin_times(times, method="time", n_bins=3)
        for i in range(len(centers) - 1):
            assert centers[i] < centers[i + 1]

    def test_ntile_method_returns_correct_count(self):
        times = jnp.linspace(0.0, 10.0, 100)
        centers = bin_times(times, method="ntile", n_bins=5)
        assert len(centers) == 5

    def test_ntile_method_with_uneven_data(self):
        # Heavily skewed times: many early, few late
        times = jnp.concatenate([jnp.zeros(90), jnp.linspace(5.0, 10.0, 10)])
        centers_time = bin_times(times, method="time", n_bins=5)
        centers_ntile = bin_times(times, method="ntile", n_bins=5)
        # ntile bins should have more bins in the dense region
        # The first ntile center should be near 0
        assert float(centers_ntile[0]) < float(centers_time[0]) + 1.0

    def test_invalid_method_raises(self):
        times = jnp.linspace(0.0, 10.0, 10)
        with pytest.raises(ValueError, match="method"):
            bin_times(times, method="invalid")


# ---------------------------------------------------------------------------
# Tests for compute_quantiles()
# ---------------------------------------------------------------------------

class TestComputeQuantiles:
    def test_known_quantiles(self):
        # 1000 values uniformly spread: quantiles should be predictable
        rng = np.random.default_rng(0)
        sim_dvs = rng.normal(loc=5.0, scale=1.0, size=(1000, 20))
        sim_dvs = jnp.array(sim_dvs)
        q = compute_quantiles(sim_dvs, quantiles=(0.05, 0.5, 0.95))
        assert len(q) == 3
        # Median of normals should be near 5.0
        np.testing.assert_allclose(np.array(q[0.5]), 5.0, atol=0.15)

    def test_quantile_ordering(self):
        rng = np.random.default_rng(1)
        sim_dvs = jnp.array(rng.normal(loc=0.0, scale=2.0, size=(500, 15)))
        q = compute_quantiles(sim_dvs, quantiles=(0.1, 0.5, 0.9))
        lo = np.array(q[0.1])
        med = np.array(q[0.5])
        hi = np.array(q[0.9])
        assert np.all(lo <= med + 1e-10)
        assert np.all(med <= hi + 1e-10)

    def test_constant_input(self):
        """All simulations produce same value → all quantiles equal."""
        sim_dvs = jnp.ones((100, 10)) * 3.0
        q = compute_quantiles(sim_dvs, quantiles=(0.05, 0.5, 0.95))
        np.testing.assert_allclose(np.array(q[0.05]), 3.0)
        np.testing.assert_allclose(np.array(q[0.5]), 3.0)
        np.testing.assert_allclose(np.array(q[0.95]), 3.0)
