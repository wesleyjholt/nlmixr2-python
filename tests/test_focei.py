"""Tests for the FOCEi (FOCE with interaction) estimator."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from nlmixr2.estimators import (
    EstimationResult,
    estimate_focei,
    focei_objective,
    foce_objective,
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
# focei_objective
# ---------------------------------------------------------------------------

class TestFoceiObjective:
    def test_returns_scalar(self):
        """focei_objective must return a single scalar value."""
        data, _ = _simulate_single_subject()
        fixed = {"A": 10.0, "ke": 0.5}
        etas = jnp.zeros((1, 2))
        omega = jnp.eye(2) * 0.1
        sigma = 0.1

        obj = focei_objective(fixed, etas, omega, sigma, _mono_exp_model, data)
        assert jnp.ndim(obj) == 0

    def test_finite_value(self):
        """Objective should be finite for reasonable inputs."""
        data, _ = _simulate_single_subject()
        fixed = {"A": 10.0, "ke": 0.5}
        etas = jnp.zeros((1, 2))
        omega = jnp.eye(2) * 0.1
        sigma = 0.1

        obj = focei_objective(fixed, etas, omega, sigma, _mono_exp_model, data)
        assert jnp.isfinite(obj)

    def test_differs_from_foce_objective(self):
        """FOCEi objective should differ from FOCE when etas are non-zero.

        The interaction term (eta-dependent residual variance) is what
        distinguishes FOCEi from FOCE. With non-zero etas the Jacobian
        df/deta is non-zero, so Sigma_i differs from constant sigma.
        """
        data, _ = _simulate_single_subject(A=10.0, ke=0.5, sigma=0.1, seed=7)
        fixed = {"A": 10.0, "ke": 0.5}
        # Use non-zero etas so the interaction term matters
        etas = jnp.array([[1.0, 0.05]])
        omega = jnp.eye(2) * 0.5
        sigma = 0.1

        obj_foce = foce_objective(fixed, etas, omega, sigma, _mono_exp_model, data)
        obj_focei = focei_objective(fixed, etas, omega, sigma, _mono_exp_model, data)
        # They should be different due to the interaction term
        assert float(obj_foce) != pytest.approx(float(obj_focei), abs=1e-6)

    def test_multiple_subjects(self):
        """Objective should work with multiple subjects."""
        data = _simulate_multi_subject(n_subjects=3, n_times=10)
        fixed = {"A": 10.0, "ke": 0.5}
        etas = jnp.zeros((3, 2))
        omega = jnp.eye(2) * 0.1
        sigma = 0.1

        obj = focei_objective(fixed, etas, omega, sigma, _mono_exp_model, data)
        assert jnp.isfinite(obj)
        assert jnp.ndim(obj) == 0

    def test_worse_params_higher_objective(self):
        """Params far from truth should yield a higher objective."""
        data, _ = _simulate_single_subject(A=10.0, ke=0.5, sigma=0.05, seed=7)
        etas = jnp.zeros((1, 2))
        omega = jnp.eye(2) * 0.1
        sigma = 0.05

        obj_good = focei_objective(
            {"A": 10.0, "ke": 0.5}, etas, omega, sigma, _mono_exp_model, data
        )
        obj_bad = focei_objective(
            {"A": 1.0, "ke": 5.0}, etas, omega, sigma, _mono_exp_model, data
        )
        assert obj_bad > obj_good

    def test_nonzero_etas_change_objective(self):
        """Non-zero etas should change the objective value."""
        data, _ = _simulate_single_subject()
        fixed = {"A": 10.0, "ke": 0.5}
        omega = jnp.eye(2) * 0.5
        sigma = 0.1

        obj_zero = focei_objective(
            fixed, jnp.zeros((1, 2)), omega, sigma, _mono_exp_model, data
        )
        obj_nonzero = focei_objective(
            fixed, jnp.array([[1.0, 0.1]]), omega, sigma, _mono_exp_model, data
        )
        assert obj_zero != pytest.approx(float(obj_nonzero), abs=1e-6)


# ---------------------------------------------------------------------------
# estimate_focei
# ---------------------------------------------------------------------------

class TestEstimateFocei:
    def test_returns_estimation_result(self):
        data, _ = _simulate_single_subject()
        ini = {"A": 8.0, "ke": 0.3}
        omega = jnp.eye(2) * 0.1

        result = estimate_focei(_mono_exp_model, data, ini, omega)
        assert isinstance(result, EstimationResult)

    def test_result_fields(self):
        """EstimationResult should have all expected fields."""
        data, _ = _simulate_single_subject()
        ini = {"A": 8.0, "ke": 0.3}
        omega = jnp.eye(2) * 0.1

        result = estimate_focei(_mono_exp_model, data, ini, omega)
        assert "A" in result.fixed_params
        assert "ke" in result.fixed_params
        assert result.etas.shape == (1, 2)
        assert isinstance(result.objective, float)
        assert isinstance(result.n_iterations, int)
        assert isinstance(result.converged, bool)

    def test_single_subject_convergence(self):
        """Should converge and return finite objective for single subject."""
        data, _ = _simulate_single_subject(A=10.0, ke=0.5, sigma=0.1, seed=1)
        ini = {"A": 8.0, "ke": 0.3}
        omega = jnp.eye(2) * 0.1

        result = estimate_focei(_mono_exp_model, data, ini, omega)
        assert jnp.isfinite(result.objective)
        assert result.n_iterations > 0

    def test_multi_subject(self):
        """Should handle multiple subjects."""
        data = _simulate_multi_subject(n_subjects=4, n_times=10, seed=99)
        ini = {"A": 8.0, "ke": 0.3}
        omega = jnp.eye(2) * 0.2

        result = estimate_focei(_mono_exp_model, data, ini, omega)
        assert isinstance(result, EstimationResult)
        assert result.etas.shape[0] == 4
        assert result.etas.shape[1] == 2

    def test_recovers_known_params(self):
        """With enough data and reasonable starting values, should get close to truth."""
        data, _ = _simulate_single_subject(
            A=10.0, ke=0.5, sigma=0.05, n_times=50, seed=123
        )
        ini = {"A": 8.0, "ke": 0.3}
        omega = jnp.eye(2) * 0.1

        result = estimate_focei(
            _mono_exp_model, data, ini, omega,
            control={"maxiter": 500, "lr": 0.005, "lr_eta": 0.005,
                     "inner_steps": 5, "sigma": 0.05},
        )
        # Should recover params to within ~30% for this prototype
        assert abs(result.fixed_params["A"] - 10.0) < 3.0
        assert abs(result.fixed_params["ke"] - 0.5) < 0.3

    def test_objective_decreases(self):
        """The final objective should be <= the initial objective."""
        data, _ = _simulate_single_subject(A=10.0, ke=0.5, sigma=0.1, seed=5)
        ini = {"A": 6.0, "ke": 1.0}
        omega = jnp.eye(2) * 0.1

        etas_init = jnp.zeros((1, 2))
        initial_obj = focei_objective(
            ini, etas_init, omega, 0.1, _mono_exp_model, data
        )

        result = estimate_focei(
            _mono_exp_model, data, ini, omega,
            control={"maxiter": 100},
        )
        assert result.objective <= float(initial_obj) + 1e-6

    def test_control_maxiter(self):
        """Should respect maxiter control option."""
        data, _ = _simulate_single_subject()
        ini = {"A": 8.0, "ke": 0.3}
        omega = jnp.eye(2) * 0.1

        result = estimate_focei(
            _mono_exp_model, data, ini, omega,
            control={"maxiter": 5},
        )
        assert result.n_iterations <= 5

    def test_etas_shape_matches_subjects(self):
        """Returned etas should have one row per subject."""
        n_subj = 6
        data = _simulate_multi_subject(n_subjects=n_subj, n_times=8, seed=77)
        ini = {"A": 9.0, "ke": 0.4}
        omega = jnp.eye(2) * 0.1

        result = estimate_focei(_mono_exp_model, data, ini, omega)
        assert result.etas.shape == (n_subj, 2)


# ---------------------------------------------------------------------------
# End-to-end via nlmixr2() API
# ---------------------------------------------------------------------------

class TestFoceiEndToEnd:
    def test_nlmixr2_focei(self):
        """nlmixr2(..., est='focei') should return an NLMIXRFit."""
        from nlmixr2.api import NLMIXRFit, NLMIXRModel, ini, model, nlmixr2

        ini_block = ini({"A": 10.0, "ke": 0.5})
        model_block = model([
            "cp = A * exp(-ke * t)",
            "cp ~ add(A)",  # error spec required by parser
        ])
        mdl = NLMIXRModel(ini=ini_block, model=model_block)

        raw = _simulate_single_subject(A=10.0, ke=0.5, sigma=0.1, seed=0)[0]
        data = {"id": raw["id"], "time": raw["time"], "dv": raw["dv"]}

        fit = nlmixr2(
            mdl, data, est="focei",
            control={"maxiter": 10, "sigma": 0.1},
        )
        assert isinstance(fit, NLMIXRFit)
        assert fit.estimator == "focei"
        assert fit.n_observations > 0
        assert isinstance(fit.objective, float)
        assert "fixed_params" in fit.table
        assert "converged" in fit.table
        assert "n_iterations" in fit.table
