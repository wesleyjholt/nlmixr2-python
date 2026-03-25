"""Tests for the Laplacian approximation estimator."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from nlmixr2.estimators import (
    EstimationResult,
    foce_objective,
    laplacian_objective,
    estimate_laplacian,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mono_exp_model(params, times):
    """Simple mono-exponential: y = A * exp(-ke * t)."""
    A = params["A"]
    ke = params["ke"]
    return A * jnp.exp(-ke * times)


def _simulate_single_subject(A=10.0, ke=0.5, sigma=0.1, n_times=20, seed=0):
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
    return data


def _simulate_multi_subject(
    A=10.0, ke=0.5, omega_A=0.3, omega_ke=0.01, sigma=0.1,
    n_subjects=5, n_times=15, seed=42,
):
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
# Tests for laplacian_objective
# ---------------------------------------------------------------------------

class TestLaplacianObjective:
    """Test the Laplacian objective function."""

    def test_returns_scalar(self):
        """Objective should return a scalar."""
        data = _simulate_single_subject()
        omega = jnp.eye(2) * 0.1
        etas = jnp.zeros((1, 2))
        fixed = {"A": 10.0, "ke": 0.5}
        result = laplacian_objective(fixed, etas, omega, 0.1, _mono_exp_model, data)
        assert result.shape == (), "Objective must be a scalar"

    def test_finite_value(self):
        """Objective should be finite for reasonable inputs."""
        data = _simulate_single_subject()
        omega = jnp.eye(2) * 0.1
        etas = jnp.zeros((1, 2))
        fixed = {"A": 10.0, "ke": 0.5}
        result = laplacian_objective(fixed, etas, omega, 0.1, _mono_exp_model, data)
        assert jnp.isfinite(result), "Objective must be finite"

    def test_positive_value(self):
        """The -2LL objective should typically be positive for reasonable data."""
        data = _simulate_single_subject()
        omega = jnp.eye(2) * 0.1
        etas = jnp.zeros((1, 2))
        fixed = {"A": 10.0, "ke": 0.5}
        result = laplacian_objective(fixed, etas, omega, 0.1, _mono_exp_model, data)
        assert result > 0, "Objective should be positive for this data"

    def test_multi_subject(self):
        """Should handle multiple subjects."""
        data = _simulate_multi_subject(n_subjects=3)
        omega = jnp.diag(jnp.array([0.3, 0.01]))
        etas = jnp.zeros((3, 2))
        fixed = {"A": 10.0, "ke": 0.5}
        result = laplacian_objective(fixed, etas, omega, 0.1, _mono_exp_model, data)
        assert jnp.isfinite(result)

    def test_differs_from_foce_at_zero_etas(self):
        """At eta=0 the Laplacian includes the Hessian correction, so it
        should differ from the plain FOCE objective."""
        data = _simulate_single_subject()
        omega = jnp.eye(2) * 0.1
        etas = jnp.zeros((1, 2))
        fixed = {"A": 10.0, "ke": 0.5}

        lap_val = laplacian_objective(fixed, etas, omega, 0.1, _mono_exp_model, data)
        foce_val = foce_objective(fixed, etas, omega, 0.1, _mono_exp_model, data)

        # They should differ because of the log|H_i| Hessian correction
        assert not jnp.allclose(lap_val, foce_val, atol=1e-6), (
            "Laplacian and FOCE should differ due to Hessian correction"
        )

    def test_worse_params_give_higher_objective(self):
        """Misspecified parameters should produce a higher objective."""
        data = _simulate_single_subject(A=10.0, ke=0.5)
        omega = jnp.eye(2) * 0.1
        etas = jnp.zeros((1, 2))

        good = laplacian_objective(
            {"A": 10.0, "ke": 0.5}, etas, omega, 0.1, _mono_exp_model, data,
        )
        bad = laplacian_objective(
            {"A": 5.0, "ke": 2.0}, etas, omega, 0.1, _mono_exp_model, data,
        )
        assert bad > good, "Bad params should give higher -2LL"

    def test_jax_differentiable(self):
        """Laplacian objective must be differentiable w.r.t. etas."""
        data = _simulate_single_subject()
        omega = jnp.eye(2) * 0.1
        fixed = {"A": 10.0, "ke": 0.5}

        def obj(etas):
            return laplacian_objective(fixed, etas, omega, 0.1, _mono_exp_model, data)

        etas = jnp.zeros((1, 2))
        grad = jax.grad(obj)(etas)
        assert grad.shape == etas.shape
        assert jnp.all(jnp.isfinite(grad))


# ---------------------------------------------------------------------------
# Tests for estimate_laplacian
# ---------------------------------------------------------------------------

class TestEstimateLaplacian:
    """Test the full Laplacian estimation routine."""

    def test_returns_estimation_result(self):
        data = _simulate_multi_subject(n_subjects=3, n_times=10)
        omega = jnp.diag(jnp.array([0.3, 0.01]))
        ini = {"A": 10.0, "ke": 0.5}
        result = estimate_laplacian(
            _mono_exp_model, data, ini, omega,
            control={"maxiter": 5, "inner_steps": 3},
        )
        assert isinstance(result, EstimationResult)

    def test_has_correct_fields(self):
        data = _simulate_multi_subject(n_subjects=3, n_times=10)
        omega = jnp.diag(jnp.array([0.3, 0.01]))
        ini = {"A": 10.0, "ke": 0.5}
        result = estimate_laplacian(
            _mono_exp_model, data, ini, omega,
            control={"maxiter": 5, "inner_steps": 3},
        )
        assert "A" in result.fixed_params
        assert "ke" in result.fixed_params
        assert result.etas.shape == (3, 2)
        assert isinstance(result.objective, float)
        assert isinstance(result.n_iterations, int)
        assert isinstance(result.converged, bool)

    def test_objective_decreases(self):
        """The objective after estimation should be <= the initial objective."""
        data = _simulate_multi_subject(n_subjects=3, n_times=10)
        omega = jnp.diag(jnp.array([0.3, 0.01]))
        ini = {"A": 10.0, "ke": 0.5}

        initial_obj = float(laplacian_objective(
            ini, jnp.zeros((3, 2)), omega, 1.0, _mono_exp_model, data,
        ))
        result = estimate_laplacian(
            _mono_exp_model, data, ini, omega,
            control={"maxiter": 20, "inner_steps": 5},
        )
        assert result.objective <= initial_obj + 1e-3, (
            "Estimation should not increase the objective significantly"
        )
