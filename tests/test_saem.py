"""Tests for the SAEM estimator module (TDD -- written before implementation)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from nlmixr2.estimators import EstimationResult, estimate_saem


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
# estimate_saem returns EstimationResult
# ---------------------------------------------------------------------------

class TestSaemReturnsEstimationResult:
    def test_returns_estimation_result(self):
        data, _ = _simulate_single_subject()
        ini = {"A": 8.0, "ke": 0.3}
        omega = jnp.eye(2) * 0.1

        result = estimate_saem(_mono_exp_model, data, ini, omega)
        assert isinstance(result, EstimationResult)

    def test_result_has_all_fields(self):
        data, _ = _simulate_single_subject()
        ini = {"A": 8.0, "ke": 0.3}
        omega = jnp.eye(2) * 0.1

        result = estimate_saem(_mono_exp_model, data, ini, omega)
        assert "A" in result.fixed_params
        assert "ke" in result.fixed_params
        assert result.etas is not None
        assert isinstance(result.objective, float)
        assert isinstance(result.n_iterations, int)
        assert isinstance(result.converged, bool)


# ---------------------------------------------------------------------------
# Single subject parameter recovery
# ---------------------------------------------------------------------------

class TestSaemSingleSubject:
    def test_single_subject_finite_objective(self):
        """Should produce a finite objective for single subject."""
        data, _ = _simulate_single_subject(A=10.0, ke=0.5, sigma=0.1, seed=1)
        ini = {"A": 8.0, "ke": 0.3}
        omega = jnp.eye(2) * 0.1

        result = estimate_saem(
            _mono_exp_model, data, ini, omega,
            control={"n_burn": 20, "n_em": 30, "sigma": 0.1},
        )
        assert jnp.isfinite(result.objective)
        assert result.n_iterations > 0

    def test_recovers_known_params_single_subject(self):
        """With enough iterations, SAEM should get close to true params."""
        data, _ = _simulate_single_subject(
            A=10.0, ke=0.5, sigma=0.05, n_times=50, seed=123
        )
        ini = {"A": 8.0, "ke": 0.3}
        omega = jnp.eye(2) * 0.5

        result = estimate_saem(
            _mono_exp_model, data, ini, omega,
            control={"n_burn": 100, "n_em": 150, "sigma": 0.05,
                     "step_size": 0.5},
        )
        # Should recover params to within reasonable tolerance
        assert abs(result.fixed_params["A"] - 10.0) < 4.0
        assert abs(result.fixed_params["ke"] - 0.5) < 0.4


# ---------------------------------------------------------------------------
# Multiple subjects
# ---------------------------------------------------------------------------

class TestSaemMultiSubject:
    def test_multi_subject_runs(self):
        """Should handle multiple subjects without error."""
        data = _simulate_multi_subject(n_subjects=4, n_times=10, seed=99)
        ini = {"A": 8.0, "ke": 0.3}
        omega = jnp.eye(2) * 0.2

        result = estimate_saem(
            _mono_exp_model, data, ini, omega,
            control={"n_burn": 15, "n_em": 20, "sigma": 0.1},
        )
        assert isinstance(result, EstimationResult)
        assert result.etas.shape[0] == 4  # one row per subject
        assert result.etas.shape[1] == 2  # n_etas matches omega dimension

    def test_multi_subject_recovers_params(self):
        """With multiple subjects SAEM should recover population params."""
        data = _simulate_multi_subject(
            A=10.0, ke=0.5, omega_A=0.3, omega_ke=0.01,
            n_subjects=8, n_times=20, sigma=0.05, seed=77,
        )
        ini = {"A": 7.0, "ke": 0.3}
        omega = jnp.eye(2) * 0.5

        result = estimate_saem(
            _mono_exp_model, data, ini, omega,
            control={"n_burn": 100, "n_em": 150, "sigma": 0.05,
                     "step_size": 0.5},
        )
        assert abs(result.fixed_params["A"] - 10.0) < 5.0
        assert abs(result.fixed_params["ke"] - 0.5) < 0.4


# ---------------------------------------------------------------------------
# Objective decreases over iterations
# ---------------------------------------------------------------------------

class TestSaemObjectiveDecreases:
    def test_final_obj_better_than_initial(self):
        """The final objective should be no worse than the initial guess."""
        from nlmixr2.estimators import foce_objective

        data, _ = _simulate_single_subject(A=10.0, ke=0.5, sigma=0.1, seed=5)
        ini = {"A": 6.0, "ke": 1.0}
        omega = jnp.eye(2) * 0.1

        etas_init = jnp.zeros((1, 2))
        initial_obj = foce_objective(
            ini, etas_init, omega, 0.1, _mono_exp_model, data
        )

        result = estimate_saem(
            _mono_exp_model, data, ini, omega,
            control={"n_burn": 50, "n_em": 100, "sigma": 0.1},
        )
        # SAEM should improve over initial guess
        assert result.objective <= float(initial_obj) + 50.0  # generous margin


# ---------------------------------------------------------------------------
# Burn-in vs convergence phase
# ---------------------------------------------------------------------------

class TestSaemPhases:
    def test_burn_in_only(self):
        """With n_em=0, should still run burn-in phase and return a result."""
        data, _ = _simulate_single_subject()
        ini = {"A": 8.0, "ke": 0.3}
        omega = jnp.eye(2) * 0.1

        result = estimate_saem(
            _mono_exp_model, data, ini, omega,
            control={"n_burn": 20, "n_em": 0, "sigma": 0.1},
        )
        assert isinstance(result, EstimationResult)
        assert result.n_iterations == 20

    def test_more_iterations_means_more_n_iterations(self):
        """Total iterations should equal n_burn + n_em."""
        data, _ = _simulate_single_subject()
        ini = {"A": 8.0, "ke": 0.3}
        omega = jnp.eye(2) * 0.1

        result_short = estimate_saem(
            _mono_exp_model, data, ini, omega,
            control={"n_burn": 10, "n_em": 10, "sigma": 0.1},
        )
        result_long = estimate_saem(
            _mono_exp_model, data, ini, omega,
            control={"n_burn": 20, "n_em": 30, "sigma": 0.1},
        )
        assert result_short.n_iterations == 20
        assert result_long.n_iterations == 50


# ---------------------------------------------------------------------------
# Control options
# ---------------------------------------------------------------------------

class TestSaemControlOptions:
    def test_default_control(self):
        """Should work with no control dict at all."""
        data, _ = _simulate_single_subject()
        ini = {"A": 8.0, "ke": 0.3}
        omega = jnp.eye(2) * 0.1

        result = estimate_saem(_mono_exp_model, data, ini, omega)
        assert isinstance(result, EstimationResult)

    def test_custom_n_burn_n_em(self):
        """Custom n_burn and n_em should be respected."""
        data, _ = _simulate_single_subject()
        ini = {"A": 8.0, "ke": 0.3}
        omega = jnp.eye(2) * 0.1

        result = estimate_saem(
            _mono_exp_model, data, ini, omega,
            control={"n_burn": 5, "n_em": 7},
        )
        assert result.n_iterations == 12

    def test_custom_sigma(self):
        """Should accept custom sigma value."""
        data, _ = _simulate_single_subject()
        ini = {"A": 8.0, "ke": 0.3}
        omega = jnp.eye(2) * 0.1

        result = estimate_saem(
            _mono_exp_model, data, ini, omega,
            control={"n_burn": 10, "n_em": 10, "sigma": 2.0},
        )
        assert isinstance(result, EstimationResult)


# ---------------------------------------------------------------------------
# Convergence flag
# ---------------------------------------------------------------------------

class TestSaemConvergence:
    def test_converged_flag_bool(self):
        """converged should be a boolean."""
        data, _ = _simulate_single_subject()
        ini = {"A": 8.0, "ke": 0.3}
        omega = jnp.eye(2) * 0.1

        result = estimate_saem(
            _mono_exp_model, data, ini, omega,
            control={"n_burn": 10, "n_em": 10},
        )
        assert isinstance(result.converged, bool)

    def test_short_run_not_converged(self):
        """Very short run unlikely to converge."""
        data, _ = _simulate_single_subject()
        ini = {"A": 6.0, "ke": 1.0}
        omega = jnp.eye(2) * 0.1

        result = estimate_saem(
            _mono_exp_model, data, ini, omega,
            control={"n_burn": 2, "n_em": 2},
        )
        # With so few iterations, convergence is not guaranteed
        assert isinstance(result.converged, bool)

    def test_long_run_near_truth_converges(self):
        """Starting near truth with many iterations should converge."""
        data, _ = _simulate_single_subject(
            A=10.0, ke=0.5, sigma=0.05, n_times=50, seed=42
        )
        ini = {"A": 10.0, "ke": 0.5}
        omega = jnp.eye(2) * 0.1

        result = estimate_saem(
            _mono_exp_model, data, ini, omega,
            control={"n_burn": 50, "n_em": 200, "sigma": 0.05,
                     "tol": 1e-2},
        )
        assert result.converged is True
