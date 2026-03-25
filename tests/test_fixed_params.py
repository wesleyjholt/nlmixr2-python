"""Tests for parameter fixing support (TDD — written before implementation)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from nlmixr2.api import InitValue, IniBlock, ini, _coerce_init_value
from nlmixr2.estimators import estimate_foce, estimate_saem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mono_exp_model(params, times):
    """Simple mono-exponential: y = A * exp(-ke * t)."""
    A = params["A"]
    ke = params["ke"]
    return A * jnp.exp(-ke * times)


def _simulate_multi_subject(
    A=10.0, ke=0.5, sigma=0.1,
    n_subjects=3, n_times=10, seed=42,
):
    """Generate data for multiple subjects."""
    key = jax.random.PRNGKey(seed)
    all_ids, all_times, all_dv = [], [], []

    for i in range(n_subjects):
        key, k1, k2 = jax.random.split(key, 3)
        times = jnp.linspace(0.5, 5.0, n_times)
        pred = A * jnp.exp(-ke * times)
        noise = jax.random.normal(k2, shape=(n_times,)) * jnp.sqrt(sigma)
        dv = pred + noise

        all_ids.append(jnp.full(n_times, i, dtype=jnp.int32))
        all_times.append(times)
        all_dv.append(dv)

    return {
        "id": jnp.concatenate(all_ids),
        "time": jnp.concatenate(all_times),
        "dv": jnp.concatenate(all_dv),
    }


# ---------------------------------------------------------------------------
# InitValue with fixed field
# ---------------------------------------------------------------------------

class TestInitValueFixed:
    def test_default_fixed_is_false(self):
        iv = InitValue(estimate=1.0)
        assert iv.fixed is False

    def test_fixed_true(self):
        iv = InitValue(estimate=1.0, fixed=True)
        assert iv.fixed is True

    def test_to_dict_includes_fixed(self):
        iv = InitValue(estimate=1.0, fixed=True)
        d = iv.to_dict()
        assert d["fixed"] is True
        assert d["estimate"] == 1.0

    def test_to_dict_fixed_false(self):
        iv = InitValue(estimate=2.0)
        d = iv.to_dict()
        assert d["fixed"] is False


# ---------------------------------------------------------------------------
# _coerce_init_value with dict specification
# ---------------------------------------------------------------------------

class TestCoerceInitValueDict:
    def test_dict_with_estimate_and_fixed(self):
        iv = _coerce_init_value({"estimate": 5.0, "fixed": True})
        assert iv.estimate == pytest.approx(5.0)
        assert iv.fixed is True

    def test_dict_with_estimate_only(self):
        iv = _coerce_init_value({"estimate": 3.0})
        assert iv.estimate == pytest.approx(3.0)
        assert iv.fixed is False

    def test_dict_with_bounds_and_fixed(self):
        iv = _coerce_init_value({"estimate": 1.0, "lower": 0.0, "upper": 2.0, "fixed": True})
        assert iv.estimate == pytest.approx(1.0)
        assert iv.lower == pytest.approx(0.0)
        assert iv.upper == pytest.approx(2.0)
        assert iv.fixed is True


# ---------------------------------------------------------------------------
# ini() with fixed parameter specification
# ---------------------------------------------------------------------------

class TestIniFixed:
    def test_ini_with_fixed_dict(self):
        block = ini({
            "A": {"estimate": 10.0, "fixed": True},
            "ke": 0.5,
        })
        assert block.values["A"].fixed is True
        assert block.values["A"].estimate == pytest.approx(10.0)
        assert block.values["ke"].fixed is False

    def test_ini_backward_compatible(self):
        """Scalar and tuple specs still work and default to fixed=False."""
        block = ini({"A": 10.0, "ke": (0.1, 0.5, 1.0)})
        assert block.values["A"].fixed is False
        assert block.values["ke"].fixed is False


# ---------------------------------------------------------------------------
# FOCE holds fixed params constant
# ---------------------------------------------------------------------------

class TestFoceFixedParams:
    def test_foce_fixed_param_unchanged(self):
        """When A is fixed, FOCE must return A at its initial value."""
        data = _simulate_multi_subject(A=10.0, ke=0.5)
        omega = jnp.eye(2) * 0.1

        result = estimate_foce(
            model=_mono_exp_model,
            data=data,
            ini_values={"A": 8.0, "ke": 0.3},
            omega=omega,
            control={"maxiter": 20, "lr": 0.01, "sigma": 0.1},
            fixed_names={"A"},
        )

        # A must stay at 8.0 exactly
        assert result.fixed_params["A"] == pytest.approx(8.0, abs=1e-6)
        # ke should have changed from initial value
        assert result.fixed_params["ke"] != pytest.approx(0.3, abs=1e-3)

    def test_foce_no_fixed_backward_compat(self):
        """Without fixed_names, FOCE behaves as before."""
        data = _simulate_multi_subject()
        omega = jnp.eye(2) * 0.1

        result = estimate_foce(
            model=_mono_exp_model,
            data=data,
            ini_values={"A": 8.0, "ke": 0.3},
            omega=omega,
            control={"maxiter": 10, "lr": 0.01, "sigma": 0.1},
        )

        assert isinstance(result.fixed_params, dict)
        assert "A" in result.fixed_params
        assert "ke" in result.fixed_params

    def test_foce_all_fixed_returns_initial(self):
        """When all params are fixed, result matches initial values."""
        data = _simulate_multi_subject()
        omega = jnp.eye(2) * 0.1

        result = estimate_foce(
            model=_mono_exp_model,
            data=data,
            ini_values={"A": 8.0, "ke": 0.3},
            omega=omega,
            control={"maxiter": 20, "lr": 0.01, "sigma": 0.1},
            fixed_names={"A", "ke"},
        )

        assert result.fixed_params["A"] == pytest.approx(8.0, abs=1e-6)
        assert result.fixed_params["ke"] == pytest.approx(0.3, abs=1e-6)


# ---------------------------------------------------------------------------
# SAEM holds fixed params constant
# ---------------------------------------------------------------------------

class TestSaemFixedParams:
    def test_saem_fixed_param_unchanged(self):
        """When A is fixed, SAEM must return A at its initial value."""
        data = _simulate_multi_subject(A=10.0, ke=0.5)
        omega = jnp.eye(2) * 0.1

        result = estimate_saem(
            model=_mono_exp_model,
            data=data,
            ini_values={"A": 8.0, "ke": 0.3},
            omega=omega,
            control={"n_burn": 5, "n_em": 10, "sigma": 0.1, "seed": 0},
            fixed_names={"A"},
        )

        assert result.fixed_params["A"] == pytest.approx(8.0, abs=1e-6)

    def test_saem_no_fixed_backward_compat(self):
        """Without fixed_names, SAEM behaves as before."""
        data = _simulate_multi_subject()
        omega = jnp.eye(2) * 0.1

        result = estimate_saem(
            model=_mono_exp_model,
            data=data,
            ini_values={"A": 8.0, "ke": 0.3},
            omega=omega,
            control={"n_burn": 5, "n_em": 5, "sigma": 0.1, "seed": 0},
        )

        assert isinstance(result.fixed_params, dict)

    def test_saem_all_fixed_returns_initial(self):
        """When all params are fixed, result matches initial values."""
        data = _simulate_multi_subject()
        omega = jnp.eye(2) * 0.1

        result = estimate_saem(
            model=_mono_exp_model,
            data=data,
            ini_values={"A": 8.0, "ke": 0.3},
            omega=omega,
            control={"n_burn": 5, "n_em": 10, "sigma": 0.1, "seed": 0},
            fixed_names={"A", "ke"},
        )

        assert result.fixed_params["A"] == pytest.approx(8.0, abs=1e-6)
        assert result.fixed_params["ke"] == pytest.approx(0.3, abs=1e-6)


# ---------------------------------------------------------------------------
# Mixed fixed and estimated params
# ---------------------------------------------------------------------------

class TestMixedFixedEstimated:
    def test_foce_mixed(self):
        """With one param fixed and one estimated, only estimated changes."""
        data = _simulate_multi_subject(A=10.0, ke=0.5)
        omega = jnp.eye(2) * 0.1

        result = estimate_foce(
            model=_mono_exp_model,
            data=data,
            ini_values={"A": 10.0, "ke": 0.1},
            omega=omega,
            control={"maxiter": 30, "lr": 0.01, "sigma": 0.1},
            fixed_names={"A"},
        )

        # A fixed at initial
        assert result.fixed_params["A"] == pytest.approx(10.0, abs=1e-6)
        # ke should move toward true value 0.5
        assert result.fixed_params["ke"] != pytest.approx(0.1, abs=0.01)
