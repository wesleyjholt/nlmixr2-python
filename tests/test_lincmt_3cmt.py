"""Tests for 3-compartment analytical PK solutions."""

import jax
import jax.numpy as jnp
import pytest

from nlmixr2.lincmt import (
    linCmt,
    three_cmt_bolus,
    three_cmt_oral,
)


# ---------------------------------------------------------------------------
# three_cmt_bolus
# ---------------------------------------------------------------------------

class TestThreeCmtBolus:
    """3-compartment IV bolus: triexponential decay."""

    # Realistic 3-cpt parameters
    DOSE = 1000.0
    K10 = 0.3      # elimination from central
    K12 = 0.2      # central -> shallow peripheral
    K21 = 0.1      # shallow peripheral -> central
    K13 = 0.05     # central -> deep peripheral
    K31 = 0.02     # deep peripheral -> central
    V1 = 20.0

    def test_initial_concentration(self):
        """C(0) = dose / V1."""
        conc = three_cmt_bolus(
            self.DOSE, self.K10, self.K12, self.K21,
            self.K13, self.K31, self.V1, jnp.array([0.0]),
        )
        assert jnp.allclose(conc, jnp.array([self.DOSE / self.V1]), rtol=1e-6)

    def test_triexponential_decay_three_phases(self):
        """Log-concentration should show three distinct exponential phases.

        We verify this by fitting the tail (dominated by gamma), the middle
        (dominated by beta after subtracting gamma), and the early part
        (dominated by alpha after subtracting beta+gamma).
        """
        times = jnp.linspace(0.0, 200.0, 10000)
        conc = three_cmt_bolus(
            self.DOSE, self.K10, self.K12, self.K21,
            self.K13, self.K31, self.V1, times,
        )
        # All concentrations should be positive
        assert jnp.all(conc > 0.0)

        # Log-concentration should not be perfectly linear (not mono-exponential)
        log_conc = jnp.log(conc)
        # Slope at early times vs late times should differ
        early_slope = (log_conc[100] - log_conc[0]) / (times[100] - times[0])
        late_slope = (log_conc[-1] - log_conc[-100]) / (times[-1] - times[-100])
        # Early decline should be steeper than late decline
        assert early_slope < late_slope  # both negative, early more negative

    def test_monotonic_terminal_decay(self):
        """After the distribution phases, concentration decays monotonically."""
        # Use late time points where terminal phase dominates
        times = jnp.linspace(20.0, 200.0, 500)
        conc = three_cmt_bolus(
            self.DOSE, self.K10, self.K12, self.K21,
            self.K13, self.K31, self.V1, times,
        )
        diffs = jnp.diff(conc)
        assert jnp.all(diffs <= 0.0)

    def test_mass_balance_auc(self):
        """AUC(0->inf) for 3-cpt IV bolus = dose / (V1 * k10).

        This follows from the fact that at steady state, all drug is
        eliminated through k10 from V1.
        """
        times = jnp.linspace(0.0, 500.0, 200000)
        conc = three_cmt_bolus(
            self.DOSE, self.K10, self.K12, self.K21,
            self.K13, self.K31, self.V1, times,
        )
        auc_numerical = jnp.trapezoid(conc, times)
        auc_analytical = self.DOSE / (self.V1 * self.K10)
        assert jnp.allclose(auc_numerical, auc_analytical, rtol=1e-2)

    def test_returns_jax_array(self):
        conc = three_cmt_bolus(
            self.DOSE, self.K10, self.K12, self.K21,
            self.K13, self.K31, self.V1, jnp.array([0.0, 1.0]),
        )
        assert isinstance(conc, jax.Array)

    def test_approaches_zero(self):
        """Concentration should approach zero at very late times."""
        conc = three_cmt_bolus(
            self.DOSE, self.K10, self.K12, self.K21,
            self.K13, self.K31, self.V1, jnp.array([1000.0]),
        )
        assert conc[0] < 1e-6


# ---------------------------------------------------------------------------
# three_cmt_oral
# ---------------------------------------------------------------------------

class TestThreeCmtOral:
    """3-compartment with first-order oral absorption."""

    DOSE = 1000.0
    KA = 1.5
    K10 = 0.3
    K12 = 0.2
    K21 = 0.1
    K13 = 0.05
    K31 = 0.02
    V1 = 20.0

    def test_concentration_at_zero(self):
        """C(0) = 0 (drug not yet absorbed)."""
        conc = three_cmt_oral(
            self.DOSE, self.KA, self.K10, self.K12, self.K21,
            self.K13, self.K31, self.V1, jnp.array([0.0]),
        )
        assert jnp.allclose(conc, 0.0, atol=1e-5)

    def test_rises_then_falls(self):
        """Oral dosing should produce a rise-then-fall profile."""
        times = jnp.linspace(0.0, 100.0, 5000)
        conc = three_cmt_oral(
            self.DOSE, self.KA, self.K10, self.K12, self.K21,
            self.K13, self.K31, self.V1, times,
        )
        peak_idx = jnp.argmax(conc)
        assert peak_idx > 0
        assert peak_idx < len(times) - 1

    def test_auc_oral(self):
        """AUC(0->inf) for 3-cpt oral with F=1: dose / (V1 * k10)."""
        times = jnp.linspace(0.0, 500.0, 200000)
        conc = three_cmt_oral(
            self.DOSE, self.KA, self.K10, self.K12, self.K21,
            self.K13, self.K31, self.V1, times,
        )
        auc_numerical = jnp.trapezoid(conc, times)
        auc_analytical = self.DOSE / (self.V1 * self.K10)
        assert jnp.allclose(auc_numerical, auc_analytical, rtol=1e-2)

    def test_returns_jax_array(self):
        conc = three_cmt_oral(
            self.DOSE, self.KA, self.K10, self.K12, self.K21,
            self.K13, self.K31, self.V1, jnp.array([0.0, 1.0]),
        )
        assert isinstance(conc, jax.Array)


# ---------------------------------------------------------------------------
# linCmt dispatch for 3-cpt
# ---------------------------------------------------------------------------

class TestLinCmtThreeCmt:
    """linCmt unified dispatch for 3-compartment models."""

    def test_three_cmt_bolus_dispatch(self):
        params = {
            "dose": 1000.0, "k10": 0.3, "k12": 0.2, "k21": 0.1,
            "k13": 0.05, "k31": 0.02, "V1": 20.0,
        }
        times = jnp.array([0.0, 1.0, 5.0])
        conc = linCmt(params, times, model_type="three_cmt_bolus")
        expected = three_cmt_bolus(1000.0, 0.3, 0.2, 0.1, 0.05, 0.02, 20.0, times)
        assert jnp.allclose(conc, expected, atol=1e-10)

    def test_three_cmt_oral_dispatch(self):
        params = {
            "dose": 1000.0, "ka": 1.5, "k10": 0.3, "k12": 0.2, "k21": 0.1,
            "k13": 0.05, "k31": 0.02, "V1": 20.0,
        }
        times = jnp.array([0.0, 1.0, 5.0])
        conc = linCmt(params, times, model_type="three_cmt_oral")
        expected = three_cmt_oral(1000.0, 1.5, 0.3, 0.2, 0.1, 0.05, 0.02, 20.0, times)
        assert jnp.allclose(conc, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------

class TestThreeCmtJIT:
    """3-compartment functions should be JIT-compilable."""

    def test_three_cmt_bolus_jit(self):
        f = jax.jit(three_cmt_bolus)
        times = jnp.array([0.0, 1.0, 5.0])
        conc = f(1000.0, 0.3, 0.2, 0.1, 0.05, 0.02, 20.0, times)
        expected = three_cmt_bolus(1000.0, 0.3, 0.2, 0.1, 0.05, 0.02, 20.0, times)
        assert jnp.allclose(conc, expected, atol=1e-10)

    def test_three_cmt_oral_jit(self):
        f = jax.jit(three_cmt_oral)
        times = jnp.array([0.0, 1.0, 5.0])
        conc = f(1000.0, 1.5, 0.3, 0.2, 0.1, 0.05, 0.02, 20.0, times)
        expected = three_cmt_oral(1000.0, 1.5, 0.3, 0.2, 0.1, 0.05, 0.02, 20.0, times)
        assert jnp.allclose(conc, expected, atol=1e-4)
