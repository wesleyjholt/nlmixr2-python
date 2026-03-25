"""Tests for nlmixr2.diagnostics — post-fit diagnostic utilities."""

from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

from nlmixr2.diagnostics import (
    compute_aic,
    compute_bic,
    compute_condition_number,
    compute_predictions,
    compute_shrinkage,
    summarize_fit,
)
from nlmixr2.api import (
    IniBlock,
    InitValue,
    ModelBlock,
    NLMIXRFit,
    NLMIXRModel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fit(
    estimator="foce",
    n_observations=5,
    parameter_count=3,
    objective=100.0,
    table=None,
) -> NLMIXRFit:
    """Build a minimal NLMIXRFit for testing."""
    ini_block = IniBlock(values={
        "tka": InitValue(estimate=0.45),
        "tcl": InitValue(estimate=1.2),
        "tv": InitValue(estimate=3.1),
    })
    model_block = ModelBlock(statements=("ka = exp(tka)", "cl = exp(tcl)", "v = exp(tv)"))
    mdl = NLMIXRModel(ini=ini_block, model=model_block)
    return NLMIXRFit(
        estimator=estimator,
        n_observations=n_observations,
        columns=("id", "time", "dv"),
        parameter_count=parameter_count,
        objective=objective,
        model=mdl,
        control={},
        table=table or {},
    )


def _simple_model_func(thetas, etas, data):
    """A trivial 'model' that returns thetas[0] + etas[0] for each row."""
    n = data["dv"].shape[0]
    return jnp.full(n, thetas[0]) + etas


# ---------------------------------------------------------------------------
# compute_predictions
# ---------------------------------------------------------------------------

class TestComputePredictions:
    def test_output_keys(self):
        fit = _make_fit()
        data = {"dv": jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        thetas = jnp.array([2.0])
        etas = jnp.array([0.1, -0.1, 0.2, -0.2, 0.0])

        result = compute_predictions(
            fit, _simple_model_func, data, thetas=thetas, etas=etas
        )

        assert set(result.keys()) == {"pred", "ipred", "res", "ires", "wres"}

    def test_output_shapes(self):
        fit = _make_fit()
        data = {"dv": jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        thetas = jnp.array([2.0])
        etas = jnp.array([0.1, -0.1, 0.2, -0.2, 0.0])

        result = compute_predictions(
            fit, _simple_model_func, data, thetas=thetas, etas=etas
        )

        for key in ("pred", "ipred", "res", "ires", "wres"):
            assert result[key].shape == (5,), f"{key} has wrong shape"

    def test_pred_uses_fixed_effects_only(self):
        fit = _make_fit()
        data = {"dv": jnp.array([1.0, 2.0, 3.0])}
        thetas = jnp.array([5.0])
        etas = jnp.array([0.1, -0.1, 0.2])

        result = compute_predictions(
            fit, _simple_model_func, data, thetas=thetas, etas=etas
        )

        # pred should use zero etas → all equal to thetas[0]
        expected_pred = jnp.array([5.0, 5.0, 5.0])
        assert jnp.allclose(result["pred"], expected_pred)

    def test_ipred_uses_individual_effects(self):
        fit = _make_fit()
        data = {"dv": jnp.array([1.0, 2.0, 3.0])}
        thetas = jnp.array([5.0])
        etas = jnp.array([0.1, -0.1, 0.2])

        result = compute_predictions(
            fit, _simple_model_func, data, thetas=thetas, etas=etas
        )

        expected_ipred = jnp.array([5.1, 4.9, 5.2])
        assert jnp.allclose(result["ipred"], expected_ipred, atol=1e-6)

    def test_residuals_equal_dv_minus_pred(self):
        fit = _make_fit()
        dv = jnp.array([1.0, 2.0, 3.0])
        data = {"dv": dv}
        thetas = jnp.array([2.0])
        etas = jnp.array([0.1, -0.1, 0.2])

        result = compute_predictions(
            fit, _simple_model_func, data, thetas=thetas, etas=etas
        )

        expected_res = dv - jnp.array([2.0, 2.0, 2.0])
        assert jnp.allclose(result["res"], expected_res)

    def test_individual_residuals_equal_dv_minus_ipred(self):
        fit = _make_fit()
        dv = jnp.array([1.0, 2.0, 3.0])
        data = {"dv": dv}
        thetas = jnp.array([2.0])
        etas = jnp.array([0.1, -0.1, 0.2])

        result = compute_predictions(
            fit, _simple_model_func, data, thetas=thetas, etas=etas
        )

        expected_ipred = jnp.array([2.1, 1.9, 2.2])
        expected_ires = dv - expected_ipred
        assert jnp.allclose(result["ires"], expected_ires, atol=1e-6)

    def test_single_observation(self):
        fit = _make_fit(n_observations=1)
        data = {"dv": jnp.array([3.0])}
        thetas = jnp.array([2.0])
        etas = jnp.array([0.5])

        result = compute_predictions(
            fit, _simple_model_func, data, thetas=thetas, etas=etas
        )

        assert result["pred"].shape == (1,)
        assert jnp.allclose(result["res"], jnp.array([1.0]))


# ---------------------------------------------------------------------------
# compute_aic / compute_bic
# ---------------------------------------------------------------------------

class TestAIC:
    def test_formula(self):
        # AIC = objective + 2 * n_params
        assert compute_aic(100.0, 3) == pytest.approx(106.0)

    def test_zero_params(self):
        assert compute_aic(50.0, 0) == pytest.approx(50.0)

    def test_negative_objective(self):
        assert compute_aic(-20.0, 5) == pytest.approx(-10.0)


class TestBIC:
    def test_formula(self):
        # BIC = objective + n_params * ln(n_obs)
        n_obs = 100
        n_params = 3
        expected = 200.0 + 3 * math.log(100)
        assert compute_bic(200.0, n_params, n_obs) == pytest.approx(expected)

    def test_single_observation(self):
        # ln(1) = 0, so BIC = objective
        assert compute_bic(50.0, 5, 1) == pytest.approx(50.0)

    def test_zero_params(self):
        assert compute_bic(50.0, 0, 100) == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# compute_condition_number
# ---------------------------------------------------------------------------

class TestConditionNumber:
    def test_identity_matrix(self):
        # All eigenvalues are 1, condition number = 1
        hessian = jnp.eye(3)
        assert compute_condition_number(hessian) == pytest.approx(1.0)

    def test_known_diagonal_matrix(self):
        # eigenvalues 1, 2, 4 → condition number = 4/1 = 4
        hessian = jnp.diag(jnp.array([1.0, 2.0, 4.0]))
        assert compute_condition_number(hessian) == pytest.approx(4.0)

    def test_scalar_matrix(self):
        # 1x1 matrix → condition number = 1
        hessian = jnp.array([[5.0]])
        assert compute_condition_number(hessian) == pytest.approx(1.0)

    def test_large_condition_number(self):
        hessian = jnp.diag(jnp.array([1e-6, 1.0, 1e6]))
        cn = compute_condition_number(hessian)
        assert cn == pytest.approx(1e12, rel=1e-3)


# ---------------------------------------------------------------------------
# compute_shrinkage
# ---------------------------------------------------------------------------

class TestShrinkage:
    def test_no_shrinkage(self):
        # If var(etas) == omega_diag, shrinkage = 0
        etas = jnp.array([[1.0, 2.0], [-1.0, -2.0]])
        # var of column 0: 1.0, var of column 1: 4.0
        omega = jnp.diag(jnp.array([1.0, 4.0]))
        shrinkage = compute_shrinkage(etas, omega)
        assert jnp.allclose(shrinkage, jnp.array([0.0, 0.0]), atol=1e-6)

    def test_full_shrinkage(self):
        # If all etas are zero, var = 0, shrinkage = 1
        etas = jnp.zeros((10, 2))
        omega = jnp.diag(jnp.array([0.5, 1.0]))
        shrinkage = compute_shrinkage(etas, omega)
        assert jnp.allclose(shrinkage, jnp.array([1.0, 1.0]))

    def test_partial_shrinkage(self):
        # var(etas) = 0.5 * omega → shrinkage = 0.5
        # column with var = 0.25, omega_diag = 0.5 → shrinkage = 0.5
        etas = jnp.array([[0.5], [-0.5]])  # var = 0.25
        omega = jnp.diag(jnp.array([0.5]))
        shrinkage = compute_shrinkage(etas, omega)
        assert shrinkage[0] == pytest.approx(0.5, abs=1e-6)

    def test_output_shape(self):
        etas = jnp.ones((5, 3))
        omega = jnp.eye(3)
        shrinkage = compute_shrinkage(etas, omega)
        assert shrinkage.shape == (3,)

    def test_zero_omega_diagonal(self):
        # When omega diagonal is zero, shrinkage should be 0 (no random effect)
        etas = jnp.zeros((5, 2))
        omega = jnp.diag(jnp.array([0.0, 1.0]))
        shrinkage = compute_shrinkage(etas, omega)
        # First element: 0/0 → should return 0.0 (convention)
        assert shrinkage[0] == pytest.approx(0.0)
        # Second element: 0/1 = 0 → shrinkage = 1.0
        assert shrinkage[1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# summarize_fit
# ---------------------------------------------------------------------------

class TestSummarizeFit:
    def test_returns_string(self):
        fit = _make_fit()
        summary = summarize_fit(fit)
        assert isinstance(summary, str)

    def test_contains_estimator(self):
        fit = _make_fit(estimator="foce")
        summary = summarize_fit(fit)
        assert "foce" in summary.lower()

    def test_contains_objective(self):
        fit = _make_fit(objective=123.456)
        summary = summarize_fit(fit)
        assert "123.456" in summary or "123.46" in summary

    def test_contains_aic_and_bic(self):
        fit = _make_fit(objective=100.0, parameter_count=3, n_observations=50)
        summary = summarize_fit(fit)
        assert "AIC" in summary
        assert "BIC" in summary

    def test_contains_parameter_estimates(self):
        fit = _make_fit()
        summary = summarize_fit(fit)
        assert "tka" in summary
        assert "tcl" in summary
        assert "tv" in summary

    def test_contains_shrinkage_when_provided(self):
        fit = _make_fit()
        predictions = {
            "shrinkage_labels": ["eta.ka", "eta.cl"],
            "shrinkage": jnp.array([0.10, 0.25]),
        }
        summary = summarize_fit(fit, predictions=predictions)
        assert "shrinkage" in summary.lower()

    def test_multiline_output(self):
        fit = _make_fit()
        summary = summarize_fit(fit)
        lines = summary.strip().split("\n")
        assert len(lines) > 3, "Summary should span multiple lines"
