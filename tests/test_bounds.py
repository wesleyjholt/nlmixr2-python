"""Tests for parameter bounds enforcement in FOCE and SAEM estimators."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from nlmixr2.estimators import estimate_foce, estimate_saem


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
    n_subjects=3, n_times=10, seed=42,
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
# FOCE bounds tests
# ---------------------------------------------------------------------------

class TestFOCEBounds:
    def test_foce_bounds_keeps_params_in_range(self):
        """FOCE with bounds should keep parameters within the specified range."""
        data = _simulate_single_subject(A=10.0, ke=0.5, seed=10)
        # Start far from truth to push optimizer; bounds should constrain
        ini = {"A": 6.0, "ke": 1.0}
        omega = jnp.eye(2) * 0.1
        bounds = {
            "A": (5.0, 15.0),
            "ke": (0.1, 2.0),
        }

        result = estimate_foce(
            _mono_exp_model, data, ini, omega,
            control={"maxiter": 50, "lr": 0.05},
            bounds=bounds,
        )

        assert result.fixed_params["A"] >= 5.0
        assert result.fixed_params["A"] <= 15.0
        assert result.fixed_params["ke"] >= 0.1
        assert result.fixed_params["ke"] <= 2.0

    def test_foce_tight_bounds_clamp_params(self):
        """With very tight bounds, params should stay at the bound edges."""
        data = _simulate_single_subject(A=10.0, ke=0.5, seed=20)
        # Start at 8.0 for A, but bound to [7.0, 8.5] -- optimizer wants ~10
        ini = {"A": 8.0, "ke": 0.5}
        omega = jnp.eye(2) * 0.1
        bounds = {
            "A": (7.0, 8.5),
            "ke": (0.1, 1.0),
        }

        result = estimate_foce(
            _mono_exp_model, data, ini, omega,
            control={"maxiter": 100, "lr": 0.01},
            bounds=bounds,
        )

        assert result.fixed_params["A"] >= 7.0
        assert result.fixed_params["A"] <= 8.5
        assert result.fixed_params["ke"] >= 0.1
        assert result.fixed_params["ke"] <= 1.0

    def test_foce_no_bounds_backward_compatible(self):
        """Passing bounds=None should work identically to the original behavior."""
        data = _simulate_single_subject(A=10.0, ke=0.5, seed=30)
        ini = {"A": 8.0, "ke": 0.3}
        omega = jnp.eye(2) * 0.1
        ctrl = {"maxiter": 20, "lr": 0.01}

        result_none = estimate_foce(
            _mono_exp_model, data, ini, omega, control=ctrl, bounds=None,
        )
        result_default = estimate_foce(
            _mono_exp_model, data, ini, omega, control=ctrl,
        )

        # Both should produce the same result
        assert result_none.fixed_params["A"] == pytest.approx(
            result_default.fixed_params["A"], abs=1e-6
        )
        assert result_none.fixed_params["ke"] == pytest.approx(
            result_default.fixed_params["ke"], abs=1e-6
        )

    def test_foce_single_sided_lower_only(self):
        """Only lower bound specified; upper is None."""
        data = _simulate_single_subject(A=10.0, ke=0.5, seed=40)
        ini = {"A": 8.0, "ke": 0.3}
        omega = jnp.eye(2) * 0.1
        bounds = {
            "A": (5.0, None),  # lower only
        }

        result = estimate_foce(
            _mono_exp_model, data, ini, omega,
            control={"maxiter": 50, "lr": 0.01},
            bounds=bounds,
        )

        assert result.fixed_params["A"] >= 5.0
        # ke should not be affected by bounds (not in bounds dict)
        assert jnp.isfinite(result.fixed_params["ke"])

    def test_foce_single_sided_upper_only(self):
        """Only upper bound specified; lower is None."""
        data = _simulate_single_subject(A=10.0, ke=0.5, seed=50)
        ini = {"A": 8.0, "ke": 0.3}
        omega = jnp.eye(2) * 0.1
        bounds = {
            "ke": (None, 1.5),  # upper only
        }

        result = estimate_foce(
            _mono_exp_model, data, ini, omega,
            control={"maxiter": 50, "lr": 0.01},
            bounds=bounds,
        )

        assert result.fixed_params["ke"] <= 1.5

    def test_foce_bound_at_initial_value(self):
        """Bounds exactly at the initial value should keep the param there."""
        data = _simulate_single_subject(A=10.0, ke=0.5, seed=60)
        ini = {"A": 8.0, "ke": 0.5}
        omega = jnp.eye(2) * 0.1
        # Pin A to exactly 8.0
        bounds = {
            "A": (8.0, 8.0),
        }

        result = estimate_foce(
            _mono_exp_model, data, ini, omega,
            control={"maxiter": 50, "lr": 0.01},
            bounds=bounds,
        )

        assert result.fixed_params["A"] == pytest.approx(8.0, abs=1e-6)


# ---------------------------------------------------------------------------
# SAEM bounds tests
# ---------------------------------------------------------------------------

class TestSAEMBounds:
    def test_saem_bounds_keeps_params_in_range(self):
        """SAEM with bounds should keep parameters within the specified range."""
        data = _simulate_multi_subject(n_subjects=3, n_times=10, seed=10)
        ini = {"A": 8.0, "ke": 0.3}
        omega = jnp.eye(2) * 0.1
        bounds = {
            "A": (5.0, 15.0),
            "ke": (0.05, 2.0),
        }

        result = estimate_saem(
            _mono_exp_model, data, ini, omega,
            control={"n_burn": 10, "n_em": 20, "seed": 42},
            bounds=bounds,
        )

        assert result.fixed_params["A"] >= 5.0
        assert result.fixed_params["A"] <= 15.0
        assert result.fixed_params["ke"] >= 0.05
        assert result.fixed_params["ke"] <= 2.0

    def test_saem_tight_bounds_clamp_params(self):
        """With tight bounds, SAEM params should stay within range."""
        data = _simulate_multi_subject(n_subjects=3, n_times=10, seed=20)
        ini = {"A": 9.0, "ke": 0.4}
        omega = jnp.eye(2) * 0.1
        bounds = {
            "A": (8.0, 9.5),
            "ke": (0.3, 0.6),
        }

        result = estimate_saem(
            _mono_exp_model, data, ini, omega,
            control={"n_burn": 10, "n_em": 20, "seed": 42},
            bounds=bounds,
        )

        assert result.fixed_params["A"] >= 8.0
        assert result.fixed_params["A"] <= 9.5
        assert result.fixed_params["ke"] >= 0.3
        assert result.fixed_params["ke"] <= 0.6

    def test_saem_no_bounds_backward_compatible(self):
        """Passing bounds=None should work identically to the original behavior."""
        data = _simulate_multi_subject(n_subjects=3, n_times=10, seed=30)
        ini = {"A": 9.0, "ke": 0.4}
        omega = jnp.eye(2) * 0.1
        ctrl = {"n_burn": 5, "n_em": 10, "seed": 42}

        result_none = estimate_saem(
            _mono_exp_model, data, ini, omega, control=ctrl, bounds=None,
        )
        result_default = estimate_saem(
            _mono_exp_model, data, ini, omega, control=ctrl,
        )

        assert result_none.fixed_params["A"] == pytest.approx(
            result_default.fixed_params["A"], abs=1e-6
        )
        assert result_none.fixed_params["ke"] == pytest.approx(
            result_default.fixed_params["ke"], abs=1e-6
        )

    def test_saem_single_sided_lower_only(self):
        """Only lower bound specified for SAEM."""
        data = _simulate_multi_subject(n_subjects=3, n_times=10, seed=40)
        ini = {"A": 9.0, "ke": 0.4}
        omega = jnp.eye(2) * 0.1
        bounds = {
            "A": (6.0, None),
        }

        result = estimate_saem(
            _mono_exp_model, data, ini, omega,
            control={"n_burn": 10, "n_em": 20, "seed": 42},
            bounds=bounds,
        )

        assert result.fixed_params["A"] >= 6.0

    def test_saem_single_sided_upper_only(self):
        """Only upper bound specified for SAEM."""
        data = _simulate_multi_subject(n_subjects=3, n_times=10, seed=50)
        ini = {"A": 9.0, "ke": 0.4}
        omega = jnp.eye(2) * 0.1
        bounds = {
            "ke": (None, 1.0),
        }

        result = estimate_saem(
            _mono_exp_model, data, ini, omega,
            control={"n_burn": 10, "n_em": 20, "seed": 42},
            bounds=bounds,
        )

        assert result.fixed_params["ke"] <= 1.0

    def test_saem_bound_at_initial_value(self):
        """Bounds exactly at the initial value should pin the parameter."""
        data = _simulate_multi_subject(n_subjects=3, n_times=10, seed=60)
        ini = {"A": 9.0, "ke": 0.4}
        omega = jnp.eye(2) * 0.1
        bounds = {
            "A": (9.0, 9.0),
        }

        result = estimate_saem(
            _mono_exp_model, data, ini, omega,
            control={"n_burn": 10, "n_em": 20, "seed": 42},
            bounds=bounds,
        )

        assert result.fixed_params["A"] == pytest.approx(9.0, abs=1e-6)
