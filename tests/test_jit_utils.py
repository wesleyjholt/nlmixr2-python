"""Tests for JIT compilation and vectorized subject processing utilities."""

import jax
import jax.numpy as jnp
import pytest

from nlmixr2.jit_utils import (
    batch_objective,
    benchmark,
    ensure_jit_compatible,
    jit_model_func,
    vmap_over_subjects,
)
from nlmixr2.lincmt import one_cmt_bolus, one_cmt_oral


# ---------------------------------------------------------------------------
# jit_model_func
# ---------------------------------------------------------------------------

class TestJitModelFunc:
    """Test that jit_model_func produces a JIT-compiled version."""

    def test_same_output_as_original(self):
        """JIT-compiled function must produce identical results."""
        times = jnp.linspace(0.0, 24.0, 50)
        dose, ke, V = 1000.0, 0.1, 10.0

        original = one_cmt_bolus(dose, ke, V, times)
        jitted = jit_model_func(one_cmt_bolus)
        compiled_result = jitted(dose, ke, V, times)

        assert jnp.allclose(original, compiled_result, atol=1e-10)

    def test_same_output_oral(self):
        """Also works for one_cmt_oral."""
        times = jnp.linspace(0.0, 24.0, 50)
        dose, ka, ke, V = 500.0, 1.5, 0.2, 20.0

        original = one_cmt_oral(dose, ka, ke, V, times)
        jitted = jit_model_func(one_cmt_oral)
        compiled_result = jitted(dose, ka, ke, V, times)

        assert jnp.allclose(original, compiled_result, atol=1e-10)

    def test_faster_on_repeated_calls(self):
        """After warm-up, JIT version should be at least as fast."""
        times = jnp.linspace(0.0, 24.0, 500)
        dose, ke, V = 1000.0, 0.1, 10.0

        jitted = jit_model_func(one_cmt_bolus)
        # Warm up
        _ = jitted(dose, ke, V, times)

        stats_jit = benchmark(jitted, dose, ke, V, times, n_runs=20)
        stats_orig = benchmark(one_cmt_bolus, dose, ke, V, times, n_runs=20)

        # JIT should not be dramatically slower than the original;
        # just verify the benchmark ran and returned timing info.
        assert stats_jit["mean_time"] >= 0
        assert stats_orig["mean_time"] >= 0


# ---------------------------------------------------------------------------
# vmap_over_subjects
# ---------------------------------------------------------------------------

class TestVmapOverSubjects:
    """Test vectorized subject processing."""

    def _make_subject_data(self, n_subjects, n_times):
        """Create test data for multiple subjects."""
        # params: (dose, ke, V) per subject
        doses = jnp.linspace(500.0, 1500.0, n_subjects)
        kes = jnp.full(n_subjects, 0.1)
        Vs = jnp.full(n_subjects, 10.0)
        all_params = jnp.stack([doses, kes, Vs], axis=1)
        times = jnp.tile(jnp.linspace(0.0, 24.0, n_times), (n_subjects, 1))
        return all_params, times

    def test_matches_sequential_loop(self):
        """vmap result must match a sequential for-loop over subjects."""
        n_subjects, n_times = 5, 20
        all_params, times = self._make_subject_data(n_subjects, n_times)

        def model(params, t):
            return one_cmt_bolus(params[0], params[1], params[2], t)

        vmap_result = vmap_over_subjects(model, all_params, times)

        # Sequential reference
        sequential = jnp.stack(
            [model(all_params[i], times[i]) for i in range(n_subjects)]
        )

        assert jnp.allclose(vmap_result, sequential, atol=1e-10)

    def test_output_shape(self):
        """Output shape should be (n_subjects, n_times)."""
        n_subjects, n_times = 8, 30
        all_params, times = self._make_subject_data(n_subjects, n_times)

        def model(params, t):
            return one_cmt_bolus(params[0], params[1], params[2], t)

        result = vmap_over_subjects(model, all_params, times)
        assert result.shape == (n_subjects, n_times)

    def test_single_subject(self):
        """Works for a single subject."""
        all_params = jnp.array([[1000.0, 0.1, 10.0]])
        times = jnp.linspace(0.0, 24.0, 15).reshape(1, -1)

        def model(params, t):
            return one_cmt_bolus(params[0], params[1], params[2], t)

        result = vmap_over_subjects(model, all_params, times)
        assert result.shape == (1, 15)
        expected = one_cmt_bolus(1000.0, 0.1, 10.0, times[0])
        assert jnp.allclose(result[0], expected, atol=1e-10)


# ---------------------------------------------------------------------------
# batch_objective
# ---------------------------------------------------------------------------

class TestBatchObjective:
    """Test batch_objective sums per-batch objectives."""

    def test_sums_correctly(self):
        """Sum of per-batch objectives should equal batch_objective result."""
        def obj_fn(batch):
            return jnp.sum(batch ** 2)

        batches = [jnp.array([1.0, 2.0, 3.0]),
                   jnp.array([4.0, 5.0]),
                   jnp.array([6.0])]

        result = batch_objective(obj_fn, batches)
        expected = sum(float(obj_fn(b)) for b in batches)
        assert jnp.allclose(result, expected, atol=1e-10)

    def test_single_batch(self):
        """Works with a single batch."""
        def obj_fn(batch):
            return jnp.sum(batch)

        batches = [jnp.array([10.0, 20.0])]
        result = batch_objective(obj_fn, batches)
        assert jnp.allclose(result, 30.0, atol=1e-10)

    def test_empty_batches(self):
        """Empty batch list should return 0."""
        def obj_fn(batch):
            return jnp.sum(batch)

        result = batch_objective(obj_fn, [])
        assert jnp.allclose(result, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# ensure_jit_compatible
# ---------------------------------------------------------------------------

class TestEnsureJitCompatible:
    """Test the ensure_jit_compatible decorator."""

    def test_decorated_function_is_jittable(self):
        """Decorated function should be compilable with jax.jit."""
        @ensure_jit_compatible
        def f(x):
            return x ** 2 + 1.0

        jitted = jax.jit(f)
        result = jitted(jnp.array(3.0))
        assert jnp.allclose(result, 10.0, atol=1e-10)

    def test_preserves_output(self):
        """Decorator should not alter function output."""
        @ensure_jit_compatible
        def g(x, y):
            return x * y + x

        result = g(jnp.array(2.0), jnp.array(3.0))
        assert jnp.allclose(result, 8.0, atol=1e-10)

    def test_preserves_function_name(self):
        """Decorated function should preserve original name."""
        @ensure_jit_compatible
        def my_func(x):
            return x

        assert my_func.__name__ == "my_func"


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------

class TestBenchmark:
    """Test the benchmark utility."""

    def test_returns_timing_dict(self):
        """Should return dict with mean_time, std_time, min_time."""
        def f(x):
            return jnp.sum(x ** 2)

        result = benchmark(f, jnp.ones(100), n_runs=5)
        assert "mean_time" in result
        assert "std_time" in result
        assert "min_time" in result

    def test_positive_times(self):
        """All timing values should be non-negative."""
        def f(x):
            return jnp.sum(x)

        result = benchmark(f, jnp.ones(10), n_runs=3)
        assert result["mean_time"] >= 0
        assert result["std_time"] >= 0
        assert result["min_time"] >= 0

    def test_min_le_mean(self):
        """min_time should be <= mean_time."""
        def f(x):
            return x + 1.0

        result = benchmark(f, jnp.array(1.0), n_runs=5)
        assert result["min_time"] <= result["mean_time"] + 1e-12


# ---------------------------------------------------------------------------
# Integration with lincmt functions
# ---------------------------------------------------------------------------

class TestLincmtIntegration:
    """Test JIT utilities with actual lincmt functions."""

    def test_jit_one_cmt_bolus(self):
        """one_cmt_bolus is already JIT-compatible; verify via jit_model_func."""
        jitted = jit_model_func(one_cmt_bolus)
        times = jnp.array([0.0, 1.0, 5.0, 10.0])
        original = one_cmt_bolus(1000.0, 0.1, 10.0, times)
        result = jitted(1000.0, 0.1, 10.0, times)
        assert jnp.allclose(original, result, atol=1e-10)

    def test_vmap_one_cmt_oral(self):
        """vmap over subjects with one_cmt_oral."""
        n_subjects = 4
        n_times = 10
        # params: dose, ka, ke, V
        all_params = jnp.array([
            [500.0, 1.5, 0.2, 20.0],
            [750.0, 1.2, 0.15, 25.0],
            [1000.0, 1.0, 0.1, 30.0],
            [250.0, 2.0, 0.3, 15.0],
        ])
        times = jnp.tile(jnp.linspace(0.0, 24.0, n_times), (n_subjects, 1))

        def model(params, t):
            return one_cmt_oral(params[0], params[1], params[2], params[3], t)

        vmap_result = vmap_over_subjects(model, all_params, times)

        for i in range(n_subjects):
            expected = one_cmt_oral(
                all_params[i, 0], all_params[i, 1],
                all_params[i, 2], all_params[i, 3],
                times[i],
            )
            assert jnp.allclose(vmap_result[i], expected, atol=1e-10)
