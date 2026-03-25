"""Tests for the NLME (Nonlinear Mixed Effects via linearization) estimator."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from nlmixr2.estimators import (
    EstimationResult,
    estimate_nlme,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mono_exp_model(params, times):
    """Simple mono-exponential: y = A * exp(-ke * t)."""
    A = params["A"]
    ke = params["ke"]
    return A * jnp.exp(-ke * times)


def _simulate_single_subject(
    A=10.0, ke=0.5, sigma=0.1, n_times=20, seed=0
):
    """Generate data for one subject from the mono-exponential model."""
    key = jax.random.PRNGKey(seed)
    times = jnp.linspace(0.5, 10.0, n_times)
    true_pred = A * jnp.exp(-ke * times)
    noise = jax.random.normal(key, shape=(n_times,)) * jnp.sqrt(sigma)
    dv = true_pred + noise
    data = {
        "id": jnp.zeros(n_times, dtype=jnp.int32),
        "time": times,
        "dv": dv,
    }
    return data, times


def _simulate_multi_subject(
    A=10.0, ke=0.5, omega_A=0.3, omega_ke=0.01, sigma=0.1,
    n_subjects=5, n_times=15, seed=42,
):
    """Generate data for multiple subjects with between-subject variability."""
    key = jax.random.PRNGKey(seed)
    all_ids, all_times, all_dv = [], [], []

    for i in range(n_subjects):
        key, k1, k2, k3 = jax.random.split(key, 4)
        eta_A = jax.random.normal(k1) * jnp.sqrt(omega_A)
        eta_ke = jax.random.normal(k2) * jnp.sqrt(omega_ke)
        subj_A = A + eta_A
        subj_ke = ke + eta_ke
        times = jnp.linspace(0.5, 10.0, n_times)
        pred = subj_A * jnp.exp(-subj_ke * times)
        noise = jax.random.normal(k3, shape=(n_times,)) * jnp.sqrt(sigma)
        dv = pred + noise

        all_ids.append(jnp.full(n_times, i, dtype=jnp.int32))
        all_times.append(times)
        all_dv.append(dv)

    data = {
        "id": jnp.concatenate(all_ids),
        "time": jnp.concatenate(all_times),
        "dv": jnp.concatenate(all_dv),
    }
    return data


# ---------------------------------------------------------------------------
# estimate_nlme
# ---------------------------------------------------------------------------

class TestEstimateNlme:
    def test_returns_estimation_result(self):
        """estimate_nlme must return an EstimationResult."""
        data, _ = _simulate_single_subject()
        ini = {"A": 8.0, "ke": 0.3}
        omega = jnp.eye(2) * 0.1

        result = estimate_nlme(_mono_exp_model, data, ini, omega)
        assert isinstance(result, EstimationResult)

    def test_objective_is_finite(self):
        """Objective should be finite for reasonable inputs."""
        data, _ = _simulate_single_subject()
        ini = {"A": 8.0, "ke": 0.3}
        omega = jnp.eye(2) * 0.1

        result = estimate_nlme(_mono_exp_model, data, ini, omega)
        assert jnp.isfinite(result.objective)

    def test_parameter_recovery(self):
        """With enough data and reasonable starts, should get close to truth."""
        data, _ = _simulate_single_subject(
            A=10.0, ke=0.5, sigma=0.05, n_times=50, seed=123
        )
        ini = {"A": 8.0, "ke": 0.3}
        omega = jnp.eye(2) * 0.1

        result = estimate_nlme(
            _mono_exp_model, data, ini, omega,
            control={"maxiter": 100, "sigma": 0.05},
        )
        assert result.fixed_params["A"] == pytest.approx(10.0, abs=3.0)
        assert result.fixed_params["ke"] == pytest.approx(0.5, abs=0.3)

    def test_convergence_flag(self):
        """Result should have a boolean converged flag."""
        data, _ = _simulate_single_subject()
        ini = {"A": 10.0, "ke": 0.5}
        omega = jnp.eye(2) * 0.1

        result = estimate_nlme(_mono_exp_model, data, ini, omega)
        assert isinstance(result.converged, bool)

    def test_multiple_subjects(self):
        """Should handle multiple subjects and return correct eta shape."""
        n_subj = 5
        data = _simulate_multi_subject(n_subjects=n_subj, n_times=10, seed=77)
        ini = {"A": 9.0, "ke": 0.4}
        omega = jnp.eye(2) * 0.2

        result = estimate_nlme(_mono_exp_model, data, ini, omega)
        assert isinstance(result, EstimationResult)
        assert result.etas.shape == (n_subj, 2)
        assert jnp.isfinite(result.objective)

    def test_result_fields(self):
        """EstimationResult should have all expected fields."""
        data, _ = _simulate_single_subject()
        ini = {"A": 8.0, "ke": 0.3}
        omega = jnp.eye(2) * 0.1

        result = estimate_nlme(_mono_exp_model, data, ini, omega)
        assert "A" in result.fixed_params
        assert "ke" in result.fixed_params
        assert result.etas.shape == (1, 2)
        assert isinstance(result.objective, float)
        assert isinstance(result.n_iterations, int)
        assert result.n_iterations > 0


# ---------------------------------------------------------------------------
# End-to-end via nlmixr2() API
# ---------------------------------------------------------------------------

class TestNlmeEndToEnd:
    def test_nlmixr2_nlme(self):
        """nlmixr2(..., est='nlme') should return an NLMIXRFit."""
        from nlmixr2.api import NLMIXRFit, NLMIXRModel, ini, model, nlmixr2

        ini_block = ini({"A": 10.0, "ke": 0.5})
        model_block = model([
            "cp = A * exp(-ke * t)",
            "cp ~ add(A)",
        ])
        mdl = NLMIXRModel(ini=ini_block, model=model_block)

        raw = _simulate_single_subject(A=10.0, ke=0.5, sigma=0.1, seed=0)[0]
        data = {"id": raw["id"], "time": raw["time"], "dv": raw["dv"]}

        fit = nlmixr2(
            mdl, data, est="nlme",
            control={"maxiter": 10, "sigma": 0.1},
        )
        assert isinstance(fit, NLMIXRFit)
        assert fit.estimator == "nlme"
        assert jnp.isfinite(fit.objective)
        assert fit.n_observations == int(raw["dv"].shape[0])
