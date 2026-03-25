"""Tests for the adaptive optimizer module."""

import jax.numpy as jnp
import pytest

from nlmixr2.optimizer import (
    AdamState,
    ConvergenceMonitor,
    adam_step,
    line_search,
    scale_parameters,
    unscale_parameters,
)


# ---------------------------------------------------------------------------
# adam_step
# ---------------------------------------------------------------------------

class TestAdamStep:
    """adam_step should produce updates that reduce a simple quadratic."""

    def test_reduces_quadratic(self):
        """Repeated Adam steps on f(x) = x^2 should drive x toward 0."""
        x = jnp.array([5.0, -3.0])
        state = AdamState(
            m=jnp.zeros_like(x),
            v=jnp.zeros_like(x),
            t=0,
        )
        for _ in range(200):
            grad = 2.0 * x  # gradient of x^2
            update, state = adam_step(grad, state, lr=0.1)
            x = x - update
        assert jnp.allclose(x, jnp.zeros(2), atol=0.1)

    def test_timestep_increments(self):
        state = AdamState(m=jnp.zeros(1), v=jnp.zeros(1), t=0)
        _, new_state = adam_step(jnp.array([1.0]), state, lr=0.01)
        assert new_state.t == 1
        _, new_state2 = adam_step(jnp.array([1.0]), new_state, lr=0.01)
        assert new_state2.t == 2

    def test_custom_betas(self):
        """Should accept custom beta1/beta2 without error."""
        state = AdamState(m=jnp.zeros(2), v=jnp.zeros(2), t=0)
        update, new_state = adam_step(
            jnp.array([1.0, -1.0]), state, lr=0.01, beta1=0.8, beta2=0.99
        )
        assert update.shape == (2,)


# ---------------------------------------------------------------------------
# line_search
# ---------------------------------------------------------------------------

class TestLineSearch:
    """Armijo backtracking line search."""

    def test_finds_reasonable_step_quadratic(self):
        """For a simple quadratic the line search should accept a positive step."""
        def obj(p):
            return float(jnp.sum(p ** 2))

        params = jnp.array([2.0, 2.0])
        direction = -2.0 * params  # steepest descent direction
        step = line_search(obj, params, direction, max_step=1.0, c1=1e-4)
        assert 0.0 < step <= 1.0

        # The step should actually decrease the objective
        new_params = params + step * direction
        assert obj(new_params) < obj(params)

    def test_respects_max_step(self):
        def obj(p):
            return float(jnp.sum(p ** 2))

        params = jnp.array([1.0])
        direction = jnp.array([-1.0])
        step = line_search(obj, params, direction, max_step=0.5)
        assert step <= 0.5

    def test_returns_small_step_for_bad_direction(self):
        """If the direction is uphill, line search should return a very small step."""
        def obj(p):
            return float(jnp.sum(p ** 2))

        params = jnp.array([1.0, 1.0])
        direction = 2.0 * params  # gradient ascent direction
        step = line_search(obj, params, direction, max_step=1.0)
        # Should shrink to a tiny step or the minimum
        assert step < 0.1


# ---------------------------------------------------------------------------
# ConvergenceMonitor
# ---------------------------------------------------------------------------

class TestConvergenceMonitor:
    def test_detects_convergence(self):
        mon = ConvergenceMonitor()
        for i in range(10):
            mon.update(100.0 - 0.001 * i, jnp.array([float(i)]))
        # Very small changes → should converge with a reasonable tolerance
        assert mon.is_converged(tol=0.01, patience=5)

    def test_not_converged_early(self):
        mon = ConvergenceMonitor()
        mon.update(100.0, jnp.array([0.0]))
        mon.update(50.0, jnp.array([1.0]))
        assert not mon.is_converged(tol=0.01, patience=5)

    def test_detects_divergence(self):
        mon = ConvergenceMonitor()
        mon.update(1.0, jnp.array([0.0]))
        mon.update(100.0, jnp.array([1.0]))
        assert mon.is_diverging(max_increase=10)

    def test_no_divergence_on_decrease(self):
        mon = ConvergenceMonitor()
        mon.update(100.0, jnp.array([0.0]))
        mon.update(50.0, jnp.array([1.0]))
        assert not mon.is_diverging(max_increase=10)

    def test_history(self):
        mon = ConvergenceMonitor()
        mon.update(10.0, jnp.array([1.0]))
        mon.update(9.0, jnp.array([2.0]))
        assert len(mon.history) == 2
        assert mon.history[0][0] == 10.0
        assert mon.history[1][0] == 9.0

    def test_patience_parameter(self):
        """Convergence requires patience consecutive small changes."""
        mon = ConvergenceMonitor()
        # 3 small changes
        for i in range(4):
            mon.update(100.0 - 0.0001 * i, jnp.array([float(i)]))
        # patience=5 means we need 5 consecutive small changes
        assert not mon.is_converged(tol=0.01, patience=5)
        # patience=3 → should be enough
        assert mon.is_converged(tol=0.01, patience=3)


# ---------------------------------------------------------------------------
# scale / unscale roundtrip
# ---------------------------------------------------------------------------

class TestScaling:
    def test_roundtrip(self):
        params = jnp.array([100.0, 0.001, -50.0])
        scales = jnp.array([100.0, 0.001, 50.0])
        normed = scale_parameters(params, scales)
        recovered = unscale_parameters(normed, scales)
        assert jnp.allclose(params, recovered, atol=1e-12)

    def test_scaled_values_near_unit(self):
        params = jnp.array([200.0, 0.002])
        scales = jnp.array([100.0, 0.001])
        normed = scale_parameters(params, scales)
        # Scaled values should be O(1)
        assert jnp.all(jnp.abs(normed) < 10.0)
