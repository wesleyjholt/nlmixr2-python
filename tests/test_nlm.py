"""Tests for the Nelder-Mead (NLM) estimator module (TDD -- written before implementation)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from nlmixr2.estimators import EstimationResult, estimate_nlm


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
# estimate_nlm returns EstimationResult
# ---------------------------------------------------------------------------

class TestNlmReturnsEstimationResult:
    def test_returns_estimation_result(self):
        data, _ = _simulate_single_subject()
        ini = {"A": 8.0, "ke": 0.3}
        omega = jnp.eye(2) * 0.1
        result = estimate_nlm(
            model_func=_mono_exp_model,
            data=data,
            ini_values=ini,
            omega=omega,
            control={"maxiter": 10},
        )
        assert isinstance(result, EstimationResult)

    def test_result_has_fixed_params(self):
        data, _ = _simulate_single_subject()
        ini = {"A": 8.0, "ke": 0.3}
        omega = jnp.eye(2) * 0.1
        result = estimate_nlm(
            model_func=_mono_exp_model,
            data=data,
            ini_values=ini,
            omega=omega,
            control={"maxiter": 10},
        )
        assert "A" in result.fixed_params
        assert "ke" in result.fixed_params

    def test_result_has_etas(self):
        data, _ = _simulate_single_subject()
        ini = {"A": 8.0, "ke": 0.3}
        omega = jnp.eye(2) * 0.1
        result = estimate_nlm(
            model_func=_mono_exp_model,
            data=data,
            ini_values=ini,
            omega=omega,
            control={"maxiter": 10},
        )
        assert result.etas.shape == (1, 2)


# ---------------------------------------------------------------------------
# Objective is finite
# ---------------------------------------------------------------------------

class TestNlmObjectiveFinite:
    def test_objective_finite_single_subject(self):
        data, _ = _simulate_single_subject()
        ini = {"A": 8.0, "ke": 0.3}
        omega = jnp.eye(2) * 0.1
        result = estimate_nlm(
            model_func=_mono_exp_model,
            data=data,
            ini_values=ini,
            omega=omega,
            control={"maxiter": 20},
        )
        assert jnp.isfinite(result.objective)

    def test_objective_finite_multi_subject(self):
        data = _simulate_multi_subject(n_subjects=3)
        ini = {"A": 8.0, "ke": 0.3}
        omega = jnp.eye(2) * 0.1
        result = estimate_nlm(
            model_func=_mono_exp_model,
            data=data,
            ini_values=ini,
            omega=omega,
            control={"maxiter": 20},
        )
        assert jnp.isfinite(result.objective)


# ---------------------------------------------------------------------------
# Parameter recovery on simple model
# ---------------------------------------------------------------------------

class TestNlmParameterRecovery:
    def test_recovers_approximate_params(self):
        """NLM should move params toward true values (A=10, ke=0.5)."""
        data, _ = _simulate_single_subject(A=10.0, ke=0.5, sigma=0.01, n_times=10, seed=1)
        ini = {"A": 8.0, "ke": 0.3}
        omega = jnp.eye(2) * 0.1
        result = estimate_nlm(
            model_func=_mono_exp_model,
            data=data,
            ini_values=ini,
            omega=omega,
            control={"maxiter": 50, "tol": 1e-8, "inner_steps": 3},
        )
        # Should get closer to true values than initial
        assert abs(result.fixed_params["A"] - 10.0) < abs(8.0 - 10.0)
        assert abs(result.fixed_params["ke"] - 0.5) < abs(0.3 - 0.5)


# ---------------------------------------------------------------------------
# Convergence flag
# ---------------------------------------------------------------------------

class TestNlmConvergence:
    def test_converged_with_enough_iterations(self):
        data, _ = _simulate_single_subject(sigma=0.01, n_times=8)
        ini = {"A": 9.5, "ke": 0.45}
        omega = jnp.eye(2) * 0.1
        result = estimate_nlm(
            model_func=_mono_exp_model,
            data=data,
            ini_values=ini,
            omega=omega,
            control={"maxiter": 150, "tol": 0.1, "inner_steps": 3, "simplex_scale": 0.5},
        )
        assert result.converged is True

    def test_not_converged_with_few_iterations(self):
        data, _ = _simulate_single_subject()
        ini = {"A": 1.0, "ke": 5.0}
        omega = jnp.eye(2) * 0.1
        result = estimate_nlm(
            model_func=_mono_exp_model,
            data=data,
            ini_values=ini,
            omega=omega,
            control={"maxiter": 2, "tol": 1e-12},
        )
        # With only 2 iterations and very tight tol, should not converge
        assert result.converged is False


# ---------------------------------------------------------------------------
# nlmixr2(..., est="nlm") end-to-end
# ---------------------------------------------------------------------------

class TestNlmEndToEnd:
    def test_nlmixr2_nlm_returns_fit(self):
        from nlmixr2.api import nlmixr2, ini, model, NLMIXRModel, NLMIXRFit

        ini_block = ini({"A": 8.0, "ke": 0.3})
        model_block = model([
            "cp = A * exp(-ke * t)",
            "cp ~ add(A)",
        ])
        mdl = NLMIXRModel(ini=ini_block, model=model_block)

        data = {
            "id": [0] * 10,
            "time": [float(x) for x in range(1, 11)],
            "dv": [10.0 * jnp.exp(-0.5 * t).item() for t in range(1, 11)],
        }

        fit = nlmixr2(mdl, data, est="nlm", control={"maxiter": 5, "inner_steps": 3})
        assert isinstance(fit, NLMIXRFit)
        assert fit.estimator == "nlm"
        assert jnp.isfinite(fit.objective)


# ---------------------------------------------------------------------------
# maxiter control option
# ---------------------------------------------------------------------------

class TestNlmMaxiter:
    def test_respects_maxiter(self):
        data, _ = _simulate_single_subject()
        ini = {"A": 8.0, "ke": 0.3}
        omega = jnp.eye(2) * 0.1
        result = estimate_nlm(
            model_func=_mono_exp_model,
            data=data,
            ini_values=ini,
            omega=omega,
            control={"maxiter": 5, "tol": 1e-15},
        )
        assert result.n_iterations <= 5

    def test_default_maxiter(self):
        data, _ = _simulate_single_subject(n_times=8)
        ini = {"A": 9.9, "ke": 0.49}
        omega = jnp.eye(2) * 0.1
        # With near-true values and low maxiter, should complete quickly
        result = estimate_nlm(
            model_func=_mono_exp_model,
            data=data,
            ini_values=ini,
            omega=omega,
            control={"maxiter": 30, "inner_steps": 3},
        )
        assert result.n_iterations <= 30


# ---------------------------------------------------------------------------
# Single subject
# ---------------------------------------------------------------------------

class TestNlmSingleSubject:
    def test_single_subject(self):
        data, _ = _simulate_single_subject()
        ini = {"A": 8.0, "ke": 0.3}
        omega = jnp.eye(2) * 0.1
        result = estimate_nlm(
            model_func=_mono_exp_model,
            data=data,
            ini_values=ini,
            omega=omega,
            control={"maxiter": 20},
        )
        assert result.etas.shape[0] == 1
        assert jnp.isfinite(result.objective)


# ---------------------------------------------------------------------------
# Multiple subjects
# ---------------------------------------------------------------------------

class TestNlmMultipleSubjects:
    def test_multiple_subjects(self):
        data = _simulate_multi_subject(n_subjects=4, n_times=8)
        ini = {"A": 8.0, "ke": 0.3}
        omega = jnp.eye(2) * 0.1
        result = estimate_nlm(
            model_func=_mono_exp_model,
            data=data,
            ini_values=ini,
            omega=omega,
            control={"maxiter": 10, "inner_steps": 3},
        )
        assert result.etas.shape[0] == 4
        assert jnp.isfinite(result.objective)

    def test_multi_subject_objective_improves(self):
        data = _simulate_multi_subject(n_subjects=3, sigma=0.01, seed=7, n_times=8)
        ini = {"A": 8.0, "ke": 0.3}
        omega = jnp.eye(2) * 0.1

        # Compute objective at initial values for comparison
        from nlmixr2.estimators import foce_objective
        ini_etas = jnp.zeros((3, 2))
        initial_obj = float(foce_objective(ini, ini_etas, omega, 1.0, _mono_exp_model, data))

        result = estimate_nlm(
            model_func=_mono_exp_model,
            data=data,
            ini_values=ini,
            omega=omega,
            control={"maxiter": 30, "inner_steps": 3},
        )
        # Final objective should be better (lower) than initial
        assert result.objective < initial_obj
