"""Tests for sensitivity equations and gradient computation."""

import jax
import jax.numpy as jnp
import pytest

from nlmixr2.sensitivity import (
    SensitivityResult,
    compute_fim,
    jax_adjoint_gradient,
    solve_with_sensitivities,
)


# ---------------------------------------------------------------------------
# Helper: 1-compartment PK model
# ---------------------------------------------------------------------------

def _rhs_1cpt(t, y, params):
    """dy/dt = -ke * y  for a single central compartment.

    params is a 1-D jnp array: [ke].
    """
    ke = params[0]
    return jnp.array([-ke * y[0]])


def _rhs_2cpt(t, y, params):
    """Two-compartment model.

    State: [central, peripheral]
    params array: [k10, k12, k21]
    """
    k10, k12, k21 = params[0], params[1], params[2]
    central = y[0]
    periph = y[1]
    dc = -k10 * central - k12 * central + k21 * periph
    dp = k12 * central - k21 * periph
    return jnp.array([dc, dp])


# ---------------------------------------------------------------------------
# Tests for solve_with_sensitivities
# ---------------------------------------------------------------------------

class TestSolveWithSensitivities:
    """Tests for the forward sensitivity solver."""

    def test_returns_sensitivity_result(self):
        params = jnp.array([0.1])
        y0 = jnp.array([100.0])
        t_eval = jnp.linspace(0.0, 24.0, 13)
        result = solve_with_sensitivities(
            _rhs_1cpt, (0.0, 24.0), y0, params, t_eval,
        )
        assert isinstance(result, SensitivityResult)

    def test_correct_shapes_1cpt(self):
        params = jnp.array([0.1])
        y0 = jnp.array([100.0])
        t_eval = jnp.linspace(0.0, 24.0, 13)
        result = solve_with_sensitivities(
            _rhs_1cpt, (0.0, 24.0), y0, params, t_eval,
        )
        n_times = len(t_eval)
        n_states = 1
        n_params = 1
        assert result.states.shape == (n_times, n_states)
        assert result.sensitivities.shape == (n_times, n_states, n_params)
        assert result.times.shape == (n_times,)

    def test_correct_shapes_2cpt(self):
        params = jnp.array([0.1, 0.05, 0.03])
        y0 = jnp.array([100.0, 0.0])
        t_eval = jnp.linspace(0.0, 48.0, 25)
        result = solve_with_sensitivities(
            _rhs_2cpt, (0.0, 48.0), y0, params, t_eval,
        )
        n_times = len(t_eval)
        assert result.states.shape == (n_times, 2)
        assert result.sensitivities.shape == (n_times, 2, 3)

    def test_sensitivities_1cpt_analytical(self):
        """For dy/dt = -ke*y, y(0)=A0:
        y(t) = A0 * exp(-ke*t)
        dy/dke = -t * A0 * exp(-ke*t)
        """
        ke_val = 0.1
        A0 = 100.0
        params = jnp.array([ke_val])
        y0 = jnp.array([A0])
        t_eval = jnp.linspace(0.5, 24.0, 48)  # skip t=0 where sensitivity is 0

        result = solve_with_sensitivities(
            _rhs_1cpt, (0.0, 24.0), y0, params, t_eval,
        )

        # Analytical sensitivity: dy/dke = -t * A0 * exp(-ke * t)
        analytical_sens = -t_eval * A0 * jnp.exp(-ke_val * t_eval)

        # Compare: result.sensitivities is (n_times, 1, 1), squeeze it
        computed = result.sensitivities[:, 0, 0]
        jnp.allclose(computed, analytical_sens, atol=1e-2, rtol=1e-2)
        # Use assert for a hard check with relaxed tolerance
        max_rel_err = jnp.max(
            jnp.abs(computed - analytical_sens) / (jnp.abs(analytical_sens) + 1e-10)
        )
        assert float(max_rel_err) < 0.05, (
            f"Max relative error {float(max_rel_err):.4f} exceeds 5%"
        )

    def test_with_dosing_events(self):
        """Ensure dosing_events parameter is accepted and shapes are correct."""
        params = jnp.array([0.1])
        y0 = jnp.array([0.0])
        t_eval = jnp.linspace(0.0, 24.0, 13)
        dosing = [{"time": 0.0, "amount": 100.0, "compartment": 0}]
        result = solve_with_sensitivities(
            _rhs_1cpt, (0.0, 24.0), y0, params, t_eval,
            dosing_events=dosing,
        )
        assert result.states.shape == (13, 1)
        assert result.sensitivities.shape == (13, 1, 1)


# ---------------------------------------------------------------------------
# Tests for compute_fim
# ---------------------------------------------------------------------------

class TestComputeFIM:
    """Tests for the Fisher Information Matrix computation."""

    def test_fim_shape(self):
        """FIM should be (n_params x n_params)."""
        params = jnp.array([0.1])
        y0 = jnp.array([100.0])
        t_eval = jnp.linspace(0.5, 24.0, 24)
        result = solve_with_sensitivities(
            _rhs_1cpt, (0.0, 24.0), y0, params, t_eval,
        )
        sigma = 1.0
        fim = compute_fim(result.sensitivities, sigma)
        n_params = 1
        assert fim.shape == (n_params, n_params)

    def test_fim_shape_multiparams(self):
        """FIM for 2-cpt model has shape (3, 3)."""
        params = jnp.array([0.1, 0.05, 0.03])
        y0 = jnp.array([100.0, 0.0])
        t_eval = jnp.linspace(0.5, 48.0, 25)
        result = solve_with_sensitivities(
            _rhs_2cpt, (0.0, 48.0), y0, params, t_eval,
        )
        fim = compute_fim(result.sensitivities, sigma=1.0)
        assert fim.shape == (3, 3)

    def test_fim_positive_semi_definite(self):
        """All eigenvalues of FIM should be >= 0."""
        params = jnp.array([0.1, 0.05, 0.03])
        y0 = jnp.array([100.0, 0.0])
        t_eval = jnp.linspace(0.5, 48.0, 50)
        result = solve_with_sensitivities(
            _rhs_2cpt, (0.0, 48.0), y0, params, t_eval,
        )
        fim = compute_fim(result.sensitivities, sigma=1.0)
        eigvals = jnp.linalg.eigvalsh(fim)
        # All eigenvalues should be non-negative (allow small numerical noise)
        assert jnp.all(eigvals >= -1e-10), (
            f"FIM has negative eigenvalue: {eigvals}"
        )

    def test_fim_symmetric(self):
        """FIM should be symmetric."""
        params = jnp.array([0.1, 0.05, 0.03])
        y0 = jnp.array([100.0, 0.0])
        t_eval = jnp.linspace(0.5, 48.0, 25)
        result = solve_with_sensitivities(
            _rhs_2cpt, (0.0, 48.0), y0, params, t_eval,
        )
        fim = compute_fim(result.sensitivities, sigma=1.0)
        assert jnp.allclose(fim, fim.T, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests for jax_adjoint_gradient
# ---------------------------------------------------------------------------

class TestAdjointGradient:
    """Tests for the adjoint-method gradient computation."""

    def _objective(self, params, rhs_func, y0, t_span, t_eval, observations):
        """Simple least-squares objective: sum((y - obs)^2)."""
        import diffrax

        term = diffrax.ODETerm(lambda t, y, args: rhs_func(t, y, args))
        solver = diffrax.Dopri5()
        stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)
        saveat = diffrax.SaveAt(ts=t_eval)
        sol = diffrax.diffeqsolve(
            term, solver,
            t0=t_span[0], t1=t_span[1],
            dt0=0.01, y0=y0, args=params,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=100_000,
        )
        pred = sol.ys[:, 0]
        return jnp.sum((pred - observations) ** 2)

    def test_gradient_is_finite(self):
        """Adjoint gradient should produce finite values."""
        ke_val = 0.1
        A0 = 100.0
        params = jnp.array([ke_val])
        y0 = jnp.array([A0])
        t_eval = jnp.linspace(0.5, 24.0, 12)
        # Generate "observations" from the model itself + small noise
        obs = A0 * jnp.exp(-ke_val * t_eval) + 0.5

        def obj_fn(p):
            return self._objective(p, _rhs_1cpt, y0, (0.0, 24.0), t_eval, obs)

        grad = jax_adjoint_gradient(obj_fn, _rhs_1cpt, params)
        assert jnp.all(jnp.isfinite(grad)), f"Non-finite gradient: {grad}"

    def test_gradient_matches_finite_differences(self):
        """Adjoint gradient should roughly match finite-difference gradient."""
        ke_val = 0.1
        A0 = 100.0
        params = jnp.array([ke_val])
        y0 = jnp.array([A0])
        t_eval = jnp.linspace(0.5, 24.0, 12)
        obs = A0 * jnp.exp(-ke_val * t_eval) + 0.5

        def obj_fn(p):
            return self._objective(p, _rhs_1cpt, y0, (0.0, 24.0), t_eval, obs)

        grad = jax_adjoint_gradient(obj_fn, _rhs_1cpt, params)

        # Finite difference
        eps = 1e-5
        fd_grad = jnp.zeros_like(params)
        for i in range(len(params)):
            p_plus = params.at[i].add(eps)
            p_minus = params.at[i].add(-eps)
            fd_grad = fd_grad.at[i].set(
                (obj_fn(p_plus) - obj_fn(p_minus)) / (2 * eps)
            )

        # Check direction and magnitude are close
        assert jnp.allclose(grad, fd_grad, rtol=1e-2, atol=1e-2), (
            f"Adjoint grad={grad}, FD grad={fd_grad}"
        )

    def test_gradient_direction_correct(self):
        """Gradient should point in direction of increasing objective."""
        ke_val = 0.1
        A0 = 100.0
        params = jnp.array([ke_val])
        y0 = jnp.array([A0])
        t_eval = jnp.linspace(0.5, 24.0, 12)
        # Observations generated at ke=0.15, so gradient at ke=0.1
        # should push ke toward 0.15 (gradient should be negative since
        # obj decreases as ke moves toward true value)
        ke_true = 0.15
        obs = A0 * jnp.exp(-ke_true * t_eval)

        def obj_fn(p):
            return self._objective(p, _rhs_1cpt, y0, (0.0, 24.0), t_eval, obs)

        grad = jax_adjoint_gradient(obj_fn, _rhs_1cpt, params)
        # At ke=0.1, the model overestimates compared to ke=0.15 data.
        # Increasing ke reduces the predicted values and should reduce
        # the objective, so gradient should be negative.
        assert float(grad[0]) < 0, (
            f"Expected negative gradient, got {float(grad[0])}"
        )
