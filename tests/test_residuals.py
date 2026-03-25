"""Tests for CWRES, NPDE, and WRES diagnostic residuals."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from nlmixr2.diagnostics import compute_cwres, compute_npde, compute_wres


# ---------------------------------------------------------------------------
# CWRES tests
# ---------------------------------------------------------------------------

class TestComputeCwres:
    def test_shape_matches_input(self):
        dv = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = jnp.array([1.1, 2.1, 3.1, 4.1, 5.1])
        ipred = jnp.array([1.05, 2.05, 3.05, 4.05, 5.05])
        sigma = 0.5
        cwres = compute_cwres(dv, pred, ipred, sigma)
        assert cwres.shape == dv.shape

    def test_zero_residuals_gives_zeros(self):
        dv = jnp.array([1.0, 2.0, 3.0])
        pred = jnp.array([1.0, 2.0, 3.0])
        ipred = jnp.array([1.0, 2.0, 3.0])
        sigma = 1.0
        cwres = compute_cwres(dv, pred, ipred, sigma)
        assert jnp.allclose(cwres, jnp.zeros(3), atol=1e-7)

    def test_sign_matches_residual_direction(self):
        # DV > IPRED => positive CWRES
        dv = jnp.array([5.0, 1.0, 3.0])
        pred = jnp.array([4.0, 2.0, 3.0])
        ipred = jnp.array([4.0, 2.0, 3.0])
        sigma = 1.0
        cwres = compute_cwres(dv, pred, ipred, sigma)
        assert float(cwres[0]) > 0.0  # DV > IPRED
        assert float(cwres[1]) < 0.0  # DV < IPRED
        assert jnp.isclose(cwres[2], 0.0, atol=1e-7)  # DV == IPRED

    def test_single_observation(self):
        dv = jnp.array([2.0])
        pred = jnp.array([1.0])
        ipred = jnp.array([1.5])
        sigma = 0.5
        cwres = compute_cwres(dv, pred, ipred, sigma)
        assert cwres.shape == (1,)
        expected = (2.0 - 1.5) / 0.5
        assert jnp.isclose(cwres[0], expected, atol=1e-6)


# ---------------------------------------------------------------------------
# NPDE tests
# ---------------------------------------------------------------------------

class TestComputeNpde:
    def test_shape_matches_observations(self):
        dv = jnp.array([1.0, 2.0, 3.0])
        simulated_dvs = jnp.ones((100, 3))  # 100 sims, 3 obs
        npde = compute_npde(dv, simulated_dvs)
        assert npde.shape == dv.shape

    def test_centered_simulations_gives_near_zero(self):
        # If observed value is at the median of simulations, NPDE ~ 0
        key = jnp.array([0, 1], dtype=jnp.uint32)  # simple PRNG key
        n_sim = 10000
        n_obs = 5
        # Simulations centered around the observed values
        dv = jnp.array([5.0, 10.0, 15.0, 20.0, 25.0])
        # Create simulations symmetrically around dv
        offsets = jnp.linspace(-2.0, 2.0, n_sim).reshape(n_sim, 1)
        simulated_dvs = dv[None, :] + offsets
        npde = compute_npde(dv, simulated_dvs)
        # Each NPDE should be approximately 0 (observed is at center)
        assert jnp.allclose(npde, 0.0, atol=0.1)

    def test_all_simulations_below_observed(self):
        # All sims < observed => fraction ~ 1 => large positive NPDE
        dv = jnp.array([10.0])
        simulated_dvs = jnp.ones((100, 1)) * 5.0  # all sims = 5 < 10
        npde = compute_npde(dv, simulated_dvs)
        assert float(npde[0]) > 2.0  # should be large positive

    def test_all_simulations_above_observed(self):
        # All sims > observed => fraction ~ 0 => large negative NPDE
        dv = jnp.array([1.0])
        simulated_dvs = jnp.ones((100, 1)) * 10.0  # all sims = 10 > 1
        npde = compute_npde(dv, simulated_dvs)
        assert float(npde[0]) < -2.0  # should be large negative

    def test_single_observation(self):
        dv = jnp.array([5.0])
        simulated_dvs = jnp.linspace(0.0, 10.0, 200).reshape(200, 1)
        npde = compute_npde(dv, simulated_dvs)
        assert npde.shape == (1,)


# ---------------------------------------------------------------------------
# WRES tests
# ---------------------------------------------------------------------------

class TestComputeWres:
    def test_wres_formula(self):
        dv = jnp.array([3.0, 5.0, 7.0])
        pred = jnp.array([2.0, 4.0, 6.0])
        sigma = 2.0
        wres = compute_wres(dv, pred, sigma)
        expected = (dv - pred) / sigma
        assert jnp.allclose(wres, expected, atol=1e-7)

    def test_shape_matches_input(self):
        dv = jnp.array([1.0, 2.0, 3.0, 4.0])
        pred = jnp.array([1.1, 2.1, 3.1, 4.1])
        sigma = 1.0
        wres = compute_wres(dv, pred, sigma)
        assert wres.shape == dv.shape

    def test_single_observation(self):
        dv = jnp.array([10.0])
        pred = jnp.array([8.0])
        sigma = 0.5
        wres = compute_wres(dv, pred, sigma)
        assert jnp.isclose(wres[0], (10.0 - 8.0) / 0.5, atol=1e-7)
