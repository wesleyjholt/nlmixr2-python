"""Tests for the SAEM estimator wired through the nlmixr2() API entry point."""

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


def _make_saem_data(n_subjects=2, n_times=10):
    """Generate simple data with required columns for SAEM."""
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

class TestNlmixr2Saem:
    def test_returns_nlmixr_fit(self):
        """nlmixr2(..., est='saem') should return an NLMIXRFit."""
        data = _make_saem_data(n_subjects=2, n_times=8)
        fit = nlmixr2(
            _algebraic_model(),
            data=data,
            est="saem",
            control={"n_burn": 3, "n_em": 5},
        )
        assert isinstance(fit, NLMIXRFit)

    def test_estimator_field_is_saem(self):
        """Returned fit should have estimator='saem'."""
        data = _make_saem_data(n_subjects=2, n_times=8)
        fit = nlmixr2(
            _algebraic_model(),
            data=data,
            est="saem",
            control={"n_burn": 3, "n_em": 5},
        )
        assert fit.estimator == "saem"

    def test_objective_is_finite(self):
        """Objective value should be finite."""
        data = _make_saem_data(n_subjects=2, n_times=8)
        fit = nlmixr2(
            _algebraic_model(),
            data=data,
            est="saem",
            control={"n_burn": 3, "n_em": 5},
        )
        assert jnp.isfinite(fit.objective)

    def test_missing_required_columns_raises(self):
        """Data missing 'id', 'time', or 'dv' should raise ValueError."""
        # Missing 'id'
        data_no_id = {
            "time": jnp.array([1.0, 2.0]),
            "dv": jnp.array([5.0, 3.0]),
        }
        with pytest.raises(ValueError, match="missing.*id"):
            nlmixr2(_algebraic_model(), data=data_no_id, est="saem")

        # Missing 'dv'
        data_no_dv = {
            "id": jnp.array([0, 0]),
            "time": jnp.array([1.0, 2.0]),
        }
        with pytest.raises(ValueError, match="missing.*dv"):
            nlmixr2(_algebraic_model(), data=data_no_dv, est="saem")

        # Missing 'time'
        data_no_time = {
            "id": jnp.array([0, 0]),
            "dv": jnp.array([5.0, 3.0]),
        }
        with pytest.raises(ValueError, match="missing.*time"):
            nlmixr2(_algebraic_model(), data=data_no_time, est="saem")

    def test_control_dict_passed_through(self):
        """Control options should appear in the fit's control dict."""
        data = _make_saem_data(n_subjects=2, n_times=5)
        fit = nlmixr2(
            _algebraic_model(),
            data=data,
            est="saem",
            control={"n_burn": 2, "n_em": 3, "custom_key": "hello"},
        )
        assert fit.control["n_burn"] == 2
        assert fit.control["n_em"] == 3
        assert fit.control["custom_key"] == "hello"
