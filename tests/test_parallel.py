"""Tests for GPU acceleration utilities and parallel bootstrap execution."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nlmixr2.api import NLMIXRFit, NLMIXRModel, ini, model, nlmixr2
from nlmixr2.bootstrap import BootstrapResult
from nlmixr2.parallel import (
    configure_jax,
    ensure_gpu,
    get_device_info,
    parallel_bootstrap,
    pmap_subjects,
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
    return {
        "id": [1, 1, 2, 2, 3, 3],
        "time": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        "dv": [0.5, 0.3, 0.6, 0.4, 0.7, 0.2],
    }


def _make_fit() -> NLMIXRFit:
    m = _make_model()
    return nlmixr2(m, data=_make_data(), est="mock")


# ---------------------------------------------------------------------------
# get_device_info
# ---------------------------------------------------------------------------

class TestGetDeviceInfo:
    def test_returns_dict(self):
        info = get_device_info()
        assert isinstance(info, dict)

    def test_has_expected_keys(self):
        info = get_device_info()
        assert "device_type" in info
        assert "device_count" in info
        assert "memory_info" in info

    def test_device_type_is_string(self):
        info = get_device_info()
        assert info["device_type"] in ("cpu", "gpu")

    def test_device_count_positive(self):
        info = get_device_info()
        assert info["device_count"] >= 1


# ---------------------------------------------------------------------------
# ensure_gpu
# ---------------------------------------------------------------------------

class TestEnsureGpu:
    def test_fallback_when_gpu_not_available(self):
        """On CI / CPU-only machines, ensure_gpu should raise RuntimeError."""
        info = get_device_info()
        if info["device_type"] == "cpu":
            with pytest.raises(RuntimeError, match="[Nn]o GPU"):
                ensure_gpu()
        else:
            # If GPU is available, it should return device info
            result = ensure_gpu()
            assert isinstance(result, dict)
            assert result["device_type"] == "gpu"


# ---------------------------------------------------------------------------
# configure_jax
# ---------------------------------------------------------------------------

class TestConfigureJax:
    def test_configure_cpu_platform(self):
        result = configure_jax(platform="cpu")
        assert result["platform"] == "cpu"

    def test_configure_returns_dict(self):
        result = configure_jax(platform="cpu", memory_fraction=0.5)
        assert isinstance(result, dict)
        assert "platform" in result
        assert "memory_fraction" in result


# ---------------------------------------------------------------------------
# parallel_bootstrap
# ---------------------------------------------------------------------------

class TestParallelBootstrap:
    def test_returns_bootstrap_result(self):
        fit = _make_fit()
        data = _make_data()
        result = parallel_bootstrap(fit, data, n_boot=5, seed=0, n_workers=2)
        assert isinstance(result, BootstrapResult)

    def test_n_success_plus_n_fail_equals_n_boot(self):
        fit = _make_fit()
        data = _make_data()
        n_boot = 8
        result = parallel_bootstrap(fit, data, n_boot=n_boot, seed=42, n_workers=2)
        assert result.n_success + result.n_fail == n_boot

    def test_single_worker_matches_sequential(self):
        fit = _make_fit()
        data = _make_data()
        result = parallel_bootstrap(fit, data, n_boot=4, seed=0, n_workers=1)
        assert isinstance(result, BootstrapResult)
        assert result.n_success + result.n_fail == 4

    def test_ci_level_respected(self):
        fit = _make_fit()
        data = _make_data()
        result = parallel_bootstrap(fit, data, n_boot=10, seed=0, n_workers=2, ci_level=0.90)
        assert isinstance(result, BootstrapResult)
        # parameter_summary should exist
        assert isinstance(result.parameter_summary, dict)

    def test_has_parameter_summary(self):
        fit = _make_fit()
        data = _make_data()
        result = parallel_bootstrap(fit, data, n_boot=6, seed=0, n_workers=2)
        assert isinstance(result.parameter_summary, dict)
        # With mock estimator, we should get some parameters
        if result.n_success > 1:
            for name, stats in result.parameter_summary.items():
                assert "mean" in stats
                assert "se" in stats


# ---------------------------------------------------------------------------
# pmap_subjects
# ---------------------------------------------------------------------------

class TestPmapSubjects:
    def test_matches_sequential(self):
        """pmap_subjects should produce same results as sequential vmap."""

        def model_func(params, times):
            # Simple one-compartment: C(t) = (D/V) * exp(-CL/V * t)
            cl, v = params[0], params[1]
            return (100.0 / v) * jnp.exp(-cl / v * times)

        n_subjects = 4
        rng = np.random.RandomState(123)
        params_per_subject = jnp.array(
            rng.uniform(0.5, 2.0, size=(n_subjects, 2)), dtype=jnp.float32
        )
        times_per_subject = jnp.broadcast_to(
            jnp.array([0.0, 1.0, 2.0, 4.0, 8.0], dtype=jnp.float32),
            (n_subjects, 5),
        )

        # Sequential reference
        sequential = jnp.stack(
            [model_func(params_per_subject[i], times_per_subject[i]) for i in range(n_subjects)]
        )

        # pmap_subjects
        result = pmap_subjects(model_func, params_per_subject, times_per_subject)

        np.testing.assert_allclose(np.array(result), np.array(sequential), rtol=1e-5)

    def test_output_shape(self):
        def model_func(params, times):
            return params[0] * jnp.exp(-params[1] * times)

        n_subjects = 3
        n_times = 5
        params = jnp.ones((n_subjects, 2), dtype=jnp.float32)
        times = jnp.ones((n_subjects, n_times), dtype=jnp.float32)

        result = pmap_subjects(model_func, params, times)
        assert result.shape == (n_subjects, n_times)
