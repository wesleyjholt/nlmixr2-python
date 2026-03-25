"""Tests for the FOCE estimator wired through the nlmixr2() API entry point."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from nlmixr2 import ini, model, nlmixr2
from nlmixr2.api import NLMIXRFit, NLMIXRModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _algebraic_model():
    """A simple algebraic (non-ODE) model: y = A * exp(-ke * t)."""
    return NLMIXRModel(
        ini=ini({"A": 10.0, "ke": 0.5}),
        model=model([
            "cp = A * exp(-ke * t)",
            "cp ~ add(A)",  # error spec required by parser
        ]),
    )


def _make_foce_data(n_subjects=2, n_times=10):
    """Generate simple data with required columns for FOCE."""
    all_ids = []
    all_times = []
    all_dv = []
    for subj in range(n_subjects):
        times = jnp.linspace(0.5, 5.0, n_times)
        # True: A=10, ke=0.5 => y = 10 * exp(-0.5 * t)
        dv = 10.0 * jnp.exp(-0.5 * times) + 0.1 * subj
        all_ids.append(jnp.full(n_times, subj, dtype=jnp.int32))
        all_times.append(times)
        all_dv.append(dv)
    return {
        "id": jnp.concatenate(all_ids),
        "time": jnp.concatenate(all_times),
        "dv": jnp.concatenate(all_dv),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNlmixr2Foce:
    def test_returns_nlmixr_fit(self):
        """nlmixr2(..., est='foce') should return an NLMIXRFit."""
        data = _make_foce_data(n_subjects=2, n_times=8)
        fit = nlmixr2(
            _algebraic_model(),
            data=data,
            est="foce",
            control={"maxiter": 5},
        )
        assert isinstance(fit, NLMIXRFit)

    def test_estimator_field_is_foce(self):
        """Returned fit should have estimator='foce'."""
        data = _make_foce_data(n_subjects=2, n_times=8)
        fit = nlmixr2(
            _algebraic_model(),
            data=data,
            est="foce",
            control={"maxiter": 5},
        )
        assert fit.estimator == "foce"

    def test_objective_is_finite(self):
        """Objective value should be finite."""
        data = _make_foce_data(n_subjects=2, n_times=8)
        fit = nlmixr2(
            _algebraic_model(),
            data=data,
            est="foce",
            control={"maxiter": 10},
        )
        assert jnp.isfinite(fit.objective)

    def test_n_observations_correct(self):
        """n_observations should match data length."""
        data = _make_foce_data(n_subjects=3, n_times=5)
        fit = nlmixr2(
            _algebraic_model(),
            data=data,
            est="foce",
            control={"maxiter": 5},
        )
        assert fit.n_observations == 15  # 3 * 5

    def test_parameter_count_correct(self):
        """parameter_count should match number of ini parameters."""
        data = _make_foce_data(n_subjects=2, n_times=5)
        fit = nlmixr2(
            _algebraic_model(),
            data=data,
            est="foce",
            control={"maxiter": 5},
        )
        assert fit.parameter_count == 2  # A, ke

    def test_missing_required_columns_raises(self):
        """Data missing 'id', 'time', or 'dv' should raise ValueError."""
        # Missing 'id'
        data_no_id = {
            "time": jnp.array([1.0, 2.0]),
            "dv": jnp.array([5.0, 3.0]),
        }
        with pytest.raises(ValueError, match="missing.*id"):
            nlmixr2(_algebraic_model(), data=data_no_id, est="foce")

        # Missing 'dv'
        data_no_dv = {
            "id": jnp.array([0, 0]),
            "time": jnp.array([1.0, 2.0]),
        }
        with pytest.raises(ValueError, match="missing.*dv"):
            nlmixr2(_algebraic_model(), data=data_no_dv, est="foce")

        # Missing 'time'
        data_no_time = {
            "id": jnp.array([0, 0]),
            "dv": jnp.array([5.0, 3.0]),
        }
        with pytest.raises(ValueError, match="missing.*time"):
            nlmixr2(_algebraic_model(), data=data_no_time, est="foce")

    def test_control_dict_passed_through(self):
        """Control options should appear in the fit's control dict."""
        data = _make_foce_data(n_subjects=2, n_times=5)
        fit = nlmixr2(
            _algebraic_model(),
            data=data,
            est="foce",
            control={"maxiter": 3, "lr": 0.005, "custom_key": "hello"},
        )
        assert fit.control["maxiter"] == 3
        assert fit.control["lr"] == 0.005
        assert fit.control["custom_key"] == "hello"

    def test_table_contains_foce_results(self):
        """The fit table should contain fixed_params, n_iterations, converged."""
        data = _make_foce_data(n_subjects=2, n_times=8)
        fit = nlmixr2(
            _algebraic_model(),
            data=data,
            est="foce",
            control={"maxiter": 5},
        )
        assert "fixed_params" in fit.table
        assert "n_iterations" in fit.table
        assert "converged" in fit.table
        assert isinstance(fit.table["fixed_params"], dict)
        assert "A" in fit.table["fixed_params"]
        assert "ke" in fit.table["fixed_params"]

    def test_columns_tuple_matches_data(self):
        """columns should reflect the data column names."""
        data = _make_foce_data(n_subjects=2, n_times=5)
        fit = nlmixr2(
            _algebraic_model(),
            data=data,
            est="foce",
            control={"maxiter": 3},
        )
        assert set(fit.columns) == {"id", "time", "dv"}

    def test_model_preserved_in_fit(self):
        """The fit should preserve the original NLMIXRModel."""
        m = _algebraic_model()
        data = _make_foce_data(n_subjects=2, n_times=5)
        fit = nlmixr2(m, data=data, est="foce", control={"maxiter": 3})
        assert fit.model is m

    def test_custom_omega_via_control(self):
        """An omega matrix passed via control should be used."""
        data = _make_foce_data(n_subjects=2, n_times=8)
        omega = jnp.eye(2) * 0.5
        fit = nlmixr2(
            _algebraic_model(),
            data=data,
            est="foce",
            control={"maxiter": 5, "omega": omega},
        )
        assert isinstance(fit, NLMIXRFit)
        assert jnp.isfinite(fit.objective)
