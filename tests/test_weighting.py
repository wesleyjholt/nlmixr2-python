"""Tests for the observation-level likelihood weighting module."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from nlmixr2.weighting import (
    WeightingScheme,
    apply_weights,
    inverse_variance_weights,
    weighted_objective,
)


# ---------------------------------------------------------------------------
# Tests for WeightingScheme
# ---------------------------------------------------------------------------

class TestWeightingScheme:
    """Test the WeightingScheme dataclass."""

    def test_create_with_array(self):
        w = jnp.array([1.0, 2.0, 3.0])
        ws = WeightingScheme(weights=w)
        assert jnp.allclose(ws.weights, w)

    def test_create_with_callable(self):
        fn = lambda dv, pred: 1.0 / (1.0 + pred)
        ws = WeightingScheme(weights=fn)
        assert callable(ws.weights)

    def test_callable_produces_weights(self):
        fn = lambda dv, pred: 1.0 / (1.0 + pred)
        ws = WeightingScheme(weights=fn)
        dv = jnp.array([5.0, 10.0])
        pred = jnp.array([4.0, 9.0])
        result = ws.weights(dv, pred)
        expected = 1.0 / (1.0 + pred)
        assert jnp.allclose(result, expected)


# ---------------------------------------------------------------------------
# Tests for apply_weights
# ---------------------------------------------------------------------------

class TestApplyWeights:
    """Test apply_weights function."""

    def test_basic_weighting(self):
        residuals = jnp.array([1.0, 2.0, 3.0])
        weights = jnp.array([0.5, 1.0, 2.0])
        result = apply_weights(residuals, weights)
        expected = residuals * jnp.sqrt(weights)
        assert jnp.allclose(result, expected)

    def test_unit_weights_no_change(self):
        residuals = jnp.array([1.0, 2.0, 3.0])
        weights = jnp.ones(3)
        result = apply_weights(residuals, weights)
        assert jnp.allclose(result, residuals)

    def test_shape_preserved(self):
        residuals = jnp.array([1.0, 2.0, 3.0, 4.0])
        weights = jnp.array([0.5, 1.0, 2.0, 0.1])
        result = apply_weights(residuals, weights)
        assert result.shape == residuals.shape

    def test_zero_weight(self):
        residuals = jnp.array([1.0, 2.0, 3.0])
        weights = jnp.array([1.0, 0.0, 1.0])
        result = apply_weights(residuals, weights)
        assert result[1] == 0.0


# ---------------------------------------------------------------------------
# Tests for inverse_variance_weights
# ---------------------------------------------------------------------------

class TestInverseVarianceWeights:
    """Test the inverse_variance_weights function."""

    def test_basic(self):
        pred = jnp.array([1.0, 2.0, 3.0])
        a = 0.1
        b = 0.2
        result = inverse_variance_weights(pred, a, b)
        expected = 1.0 / (a + b * pred) ** 2
        assert jnp.allclose(result, expected)

    def test_zero_pred(self):
        pred = jnp.array([0.0])
        result = inverse_variance_weights(pred, 0.1, 0.2)
        expected = 1.0 / (0.1) ** 2
        assert jnp.allclose(result, expected)

    def test_positive_weights(self):
        pred = jnp.array([1.0, 5.0, 10.0])
        result = inverse_variance_weights(pred, 0.1, 0.5)
        assert jnp.all(result > 0)

    def test_decreasing_with_pred(self):
        """Weights should decrease as predictions increase (for b > 0)."""
        pred = jnp.array([1.0, 2.0, 3.0])
        result = inverse_variance_weights(pred, 0.1, 0.5)
        assert jnp.all(jnp.diff(result) < 0)


# ---------------------------------------------------------------------------
# Tests for weighted_objective
# ---------------------------------------------------------------------------

def _simple_objective(fixed_params, etas, omega, sigma, model_func, data):
    """A toy objective: sum of squared residuals."""
    pred = model_func(fixed_params, data["time"])
    return jnp.sum((data["dv"] - pred) ** 2)


def _simple_model(params, times):
    return params["a"] * times


class TestWeightedObjective:
    """Test the weighted_objective wrapper."""

    def test_returns_callable(self):
        weights = jnp.ones(5)
        new_obj = weighted_objective(_simple_objective, weights)
        assert callable(new_obj)

    def test_unit_weights_same_as_original(self):
        times = jnp.array([1.0, 2.0, 3.0])
        data = {
            "id": jnp.zeros(3, dtype=jnp.int32),
            "time": times,
            "dv": jnp.array([1.0, 2.0, 3.0]),
        }
        weights = jnp.ones(3)
        fixed = {"a": 1.0}
        omega = jnp.eye(1)

        orig = _simple_objective(fixed, None, omega, 1.0, _simple_model, data)
        wrapped = weighted_objective(_simple_objective, weights)
        weighted_val = wrapped(fixed, None, omega, 1.0, _simple_model, data)

        assert jnp.allclose(orig, weighted_val, atol=1e-6)

    def test_weights_change_objective(self):
        times = jnp.array([1.0, 2.0, 3.0])
        data = {
            "id": jnp.zeros(3, dtype=jnp.int32),
            "time": times,
            "dv": jnp.array([2.0, 4.0, 6.0]),  # pred=a*t, resid nonzero when a!=2
        }
        fixed = {"a": 1.5}  # deliberate mismatch
        omega = jnp.eye(1)
        weights_equal = jnp.ones(3)
        weights_unequal = jnp.array([1.0, 2.0, 3.0])

        obj_eq = weighted_objective(_simple_objective, weights_equal)(
            fixed, None, omega, 1.0, _simple_model, data,
        )
        obj_uneq = weighted_objective(_simple_objective, weights_unequal)(
            fixed, None, omega, 1.0, _simple_model, data,
        )
        assert not jnp.allclose(obj_eq, obj_uneq), (
            "Non-unit weights should change the objective"
        )

    def test_callable_weights(self):
        """weighted_objective should also accept a callable for weights."""
        times = jnp.array([1.0, 2.0, 3.0])
        data = {
            "id": jnp.zeros(3, dtype=jnp.int32),
            "time": times,
            "dv": jnp.array([2.0, 4.0, 6.0]),
        }
        fixed = {"a": 1.5}
        omega = jnp.eye(1)

        # Callable that returns constant weights
        weight_fn = lambda dv, pred: jnp.ones_like(dv)
        wrapped = weighted_objective(_simple_objective, weight_fn)
        result = wrapped(fixed, None, omega, 1.0, _simple_model, data)
        orig = _simple_objective(fixed, None, omega, 1.0, _simple_model, data)
        assert jnp.allclose(result, orig, atol=1e-6)
