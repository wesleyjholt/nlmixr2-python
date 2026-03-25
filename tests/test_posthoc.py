"""Tests for posthoc (empirical Bayes) estimation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from nlmixr2.estimators import EstimationResult, estimate_posthoc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mono_exp_model(params, times):
    """Simple mono-exponential: y = A * exp(-ke * t)."""
    A = params["A"]
    ke = params["ke"]
    return A * jnp.exp(-ke * times)


def _simulate_single_subject(A=10.0, ke=0.5, sigma=0.1, n_times=20, seed=0):
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
    return data


def _simulate_multi_subject(
    A=10.0, ke=0.5, omega_A=0.3, omega_ke=0.01, sigma=0.1,
    n_subjects=5, n_times=15, seed=42,
):
    """Generate data for multiple subjects with between-subject variability."""
    key = jax.random.PRNGKey(seed)
    all_ids, all_times, all_dv = [], [], []
    true_etas = []

    for i in range(n_subjects):
        key, k1, k2, k3 = jax.random.split(key, 4)
        eta_A = float(jax.random.normal(k1) * jnp.sqrt(omega_A))
        eta_ke = float(jax.random.normal(k2) * jnp.sqrt(omega_ke))
        true_etas.append([eta_A, eta_ke])
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
    return data, jnp.array(true_etas)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPosthocReturnsEstimationResult:
    def test_returns_estimation_result(self):
        data = _simulate_single_subject()
        fixed_params = {"A": 10.0, "ke": 0.5}
        omega = jnp.eye(2) * 0.1
        result = estimate_posthoc(
            _mono_exp_model, data, fixed_params, omega, sigma=1.0,
        )
        assert isinstance(result, EstimationResult)

    def test_fixed_params_unchanged(self):
        data = _simulate_single_subject()
        fixed_params = {"A": 10.0, "ke": 0.5}
        omega = jnp.eye(2) * 0.1
        result = estimate_posthoc(
            _mono_exp_model, data, fixed_params, omega, sigma=1.0,
        )
        assert result.fixed_params == fixed_params


class TestPosthocEtasShape:
    def test_single_subject_shape(self):
        data = _simulate_single_subject()
        fixed_params = {"A": 10.0, "ke": 0.5}
        omega = jnp.eye(2) * 0.1
        result = estimate_posthoc(
            _mono_exp_model, data, fixed_params, omega, sigma=1.0,
        )
        assert result.etas.shape == (1, 2)

    def test_multi_subject_shape(self):
        n_subjects = 5
        data, _ = _simulate_multi_subject(n_subjects=n_subjects)
        fixed_params = {"A": 10.0, "ke": 0.5}
        omega = jnp.eye(2) * 0.1
        result = estimate_posthoc(
            _mono_exp_model, data, fixed_params, omega, sigma=1.0,
        )
        assert result.etas.shape == (n_subjects, 2)


class TestPosthocEtasNearZero:
    def test_etas_near_zero_when_data_matches_population(self):
        """When data is generated exactly from population params, etas ~ 0."""
        times = jnp.linspace(0.5, 10.0, 30)
        A, ke = 10.0, 0.5
        # No noise -- DV is exactly the population prediction
        dv = A * jnp.exp(-ke * times)
        data = {
            "id": jnp.zeros(30, dtype=jnp.int32),
            "time": times,
            "dv": dv,
        }
        fixed_params = {"A": A, "ke": ke}
        omega = jnp.eye(2) * 0.1
        result = estimate_posthoc(
            _mono_exp_model, data, fixed_params, omega, sigma=1.0,
            control={"maxiter": 200, "lr": 0.01},
        )
        # Etas should be very close to zero
        assert jnp.allclose(result.etas, 0.0, atol=0.05), (
            f"Expected etas near zero, got {result.etas}"
        )


class TestPosthocEtasShift:
    def test_etas_shift_for_deviant_subject(self):
        """When individual data deviates, etas should shift away from zero."""
        times = jnp.linspace(0.5, 10.0, 30)
        # Population params
        A_pop, ke_pop = 10.0, 0.5
        # Individual has higher A
        A_indiv = 12.0
        dv = A_indiv * jnp.exp(-ke_pop * times)
        data = {
            "id": jnp.zeros(30, dtype=jnp.int32),
            "time": times,
            "dv": dv,
        }
        fixed_params = {"A": A_pop, "ke": ke_pop}
        omega = jnp.eye(2) * 10.0  # wide prior so eta can move
        result = estimate_posthoc(
            _mono_exp_model, data, fixed_params, omega, sigma=1.0,
            control={"maxiter": 1000, "lr": 0.001},
        )
        # eta for A should be positive (individual A > population A)
        eta_A = float(result.etas[0, 0])
        assert eta_A > 0.5, f"Expected positive eta_A shift, got {eta_A}"


class TestPosthocMultiSubject:
    def test_different_etas_per_subject(self):
        """Each subject should get different etas when data differs."""
        data, true_etas = _simulate_multi_subject(
            n_subjects=3, omega_A=1.0, omega_ke=0.05, sigma=0.5,
        )
        fixed_params = {"A": 10.0, "ke": 0.5}
        omega = jnp.diag(jnp.array([1.0, 0.05]))
        result = estimate_posthoc(
            _mono_exp_model, data, fixed_params, omega, sigma=0.5,
            control={"maxiter": 500, "lr": 0.001},
        )
        # Not all subjects should have identical etas
        eta_spread = jnp.std(result.etas[:, 0])
        assert eta_spread > 0.01, (
            f"Expected variation in etas across subjects, got std={eta_spread}"
        )


class TestPosthocObjective:
    def test_objective_is_finite(self):
        data = _simulate_single_subject()
        fixed_params = {"A": 10.0, "ke": 0.5}
        omega = jnp.eye(2) * 0.1
        result = estimate_posthoc(
            _mono_exp_model, data, fixed_params, omega, sigma=1.0,
        )
        assert jnp.isfinite(result.objective)


class TestPosthocEndToEnd:
    def test_via_nlmixr2_api(self):
        """End-to-end: nlmixr2(..., est='posthoc') should work."""
        from nlmixr2.api import nlmixr2, ini, model, NLMIXRModel, NLMIXRFit

        ini_block = ini({"A": 10.0, "ke": 0.5})
        model_block = model([
            "cp = A * exp(-ke * t)",
            "cp ~ add(A)",
        ])
        mod = NLMIXRModel(ini=ini_block, model=model_block)

        times = jnp.linspace(0.5, 10.0, 20)
        dv = 10.0 * jnp.exp(-0.5 * times)
        data = {
            "id": jnp.zeros(20, dtype=jnp.int32),
            "time": times,
            "dv": dv,
        }

        ctrl = {
            "fixed_params": {"A": 10.0, "ke": 0.5},
            "omega": jnp.eye(2) * 0.1,
            "sigma": 1.0,
            "maxiter": 100,
        }

        fit = nlmixr2(mod, data=data, est="posthoc", control=ctrl)
        assert isinstance(fit, NLMIXRFit)
        assert fit.estimator == "posthoc"

    def test_api_requires_fixed_params(self):
        """nlmixr2(..., est='posthoc') must raise if fixed_params missing."""
        from nlmixr2.api import nlmixr2, ini, model, NLMIXRModel

        ini_block = ini({"A": 10.0, "ke": 0.5})
        model_block = model([
            "cp = A * exp(-ke * t)",
            "cp ~ add(A)",
        ])
        mod = NLMIXRModel(ini=ini_block, model=model_block)

        data = {
            "id": jnp.zeros(5, dtype=jnp.int32),
            "time": jnp.linspace(0.5, 5.0, 5),
            "dv": jnp.ones(5),
        }

        with pytest.raises(ValueError, match="fixed_params"):
            nlmixr2(mod, data=data, est="posthoc", control={"omega": jnp.eye(2)})

    def test_api_requires_omega(self):
        """nlmixr2(..., est='posthoc') must raise if omega missing."""
        from nlmixr2.api import nlmixr2, ini, model, NLMIXRModel

        ini_block = ini({"A": 10.0, "ke": 0.5})
        model_block = model([
            "cp = A * exp(-ke * t)",
            "cp ~ add(A)",
        ])
        mod = NLMIXRModel(ini=ini_block, model=model_block)

        data = {
            "id": jnp.zeros(5, dtype=jnp.int32),
            "time": jnp.linspace(0.5, 5.0, 5),
            "dv": jnp.ones(5),
        }

        with pytest.raises(ValueError, match="omega"):
            nlmixr2(mod, data=data, est="posthoc", control={"fixed_params": {"A": 10.0}})
