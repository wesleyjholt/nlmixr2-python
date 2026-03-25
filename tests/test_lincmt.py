"""Tests for analytical (closed-form) PK compartment model solutions."""

import jax
import jax.numpy as jnp
import pytest

from nlmixr2.lincmt import (
    linCmt,
    one_cmt_bolus,
    one_cmt_infusion,
    one_cmt_oral,
    superposition,
    two_cmt_bolus,
    two_cmt_oral,
)


# ---------------------------------------------------------------------------
# one_cmt_bolus
# ---------------------------------------------------------------------------

class TestOneCmtBolus:
    """1-compartment IV bolus: C(t) = (dose/V) * exp(-ke * t)."""

    def test_initial_concentration(self):
        """C(0) = dose / V."""
        dose, ke, V = 1000.0, 0.1, 10.0
        conc = one_cmt_bolus(dose, ke, V, jnp.array([0.0]))
        assert jnp.allclose(conc, jnp.array([dose / V]), atol=1e-10)

    def test_exponential_decay(self):
        dose, ke, V = 500.0, 0.2, 20.0
        times = jnp.array([0.0, 1.0, 5.0, 10.0])
        conc = one_cmt_bolus(dose, ke, V, times)
        expected = (dose / V) * jnp.exp(-ke * times)
        assert jnp.allclose(conc, expected, atol=1e-10)

    def test_half_life(self):
        """At t = ln(2)/ke, concentration should be half the initial."""
        dose, ke, V = 100.0, 0.1, 5.0
        t_half = jnp.log(2.0) / ke
        conc = one_cmt_bolus(dose, ke, V, jnp.array([0.0, t_half]))
        assert jnp.allclose(conc[1], conc[0] / 2.0, rtol=1e-6)

    def test_returns_jax_array(self):
        conc = one_cmt_bolus(100.0, 0.1, 10.0, jnp.array([0.0, 1.0]))
        assert isinstance(conc, jax.Array)

    def test_single_time_point(self):
        conc = one_cmt_bolus(100.0, 0.1, 10.0, jnp.array([1.0]))
        assert conc.shape == (1,)

    def test_auc_bolus(self):
        """AUC(0->inf) for IV bolus = dose / (V * ke)."""
        dose, ke, V = 100.0, 0.1, 10.0
        # Numerical AUC via trapezoidal rule with dense time grid
        times = jnp.linspace(0.0, 200.0, 10000)
        conc = one_cmt_bolus(dose, ke, V, times)
        auc_numerical = jnp.trapezoid(conc, times)
        auc_analytical = dose / (V * ke)
        assert jnp.allclose(auc_numerical, auc_analytical, rtol=1e-3)


# ---------------------------------------------------------------------------
# one_cmt_oral
# ---------------------------------------------------------------------------

class TestOneCmtOral:
    """1-compartment oral: Bateman equation."""

    def test_concentration_at_zero(self):
        """C(0) = 0 (drug hasn't been absorbed yet)."""
        conc = one_cmt_oral(100.0, 1.0, 0.1, 10.0, jnp.array([0.0]))
        assert jnp.allclose(conc, 0.0, atol=1e-10)

    def test_peak_time(self):
        """tmax = ln(ka/ke) / (ka - ke)."""
        dose, ka, ke, V = 100.0, 1.5, 0.1, 10.0
        tmax_analytical = jnp.log(ka / ke) / (ka - ke)
        # Evaluate at a fine grid around tmax
        times = jnp.linspace(0.0, 50.0, 50000)
        conc = one_cmt_oral(dose, ka, ke, V, times)
        tmax_numerical = times[jnp.argmax(conc)]
        assert jnp.allclose(tmax_numerical, tmax_analytical, atol=0.01)

    def test_cmax_value(self):
        """Cmax at tmax should match analytical formula."""
        dose, ka, ke, V = 100.0, 1.5, 0.1, 10.0
        tmax = jnp.log(ka / ke) / (ka - ke)
        cmax = one_cmt_oral(dose, ka, ke, V, jnp.array([tmax]))
        expected = (dose * ka / (V * (ka - ke))) * (
            jnp.exp(-ke * tmax) - jnp.exp(-ka * tmax)
        )
        assert jnp.allclose(cmax[0], expected, rtol=1e-6)

    def test_bateman_equation(self):
        dose, ka, ke, V = 500.0, 2.0, 0.3, 25.0
        times = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0])
        conc = one_cmt_oral(dose, ka, ke, V, times)
        expected = (dose * ka / (V * (ka - ke))) * (
            jnp.exp(-ke * times) - jnp.exp(-ka * times)
        )
        assert jnp.allclose(conc, expected, atol=1e-10)

    def test_ka_near_ke(self):
        """When ka ~ ke, the model should still produce finite values (no NaN)."""
        dose, V = 100.0, 10.0
        ke = 0.5
        ka = ke + 1e-8  # very close to ke
        times = jnp.array([0.0, 1.0, 2.0, 5.0])
        conc = one_cmt_oral(dose, ka, ke, V, times)
        assert jnp.all(jnp.isfinite(conc))
        assert jnp.all(conc >= 0.0)

    def test_auc_oral(self):
        """AUC(0->inf) for oral with F=1: dose / (V * ke)."""
        dose, ka, ke, V = 100.0, 1.5, 0.1, 10.0
        times = jnp.linspace(0.0, 300.0, 50000)
        conc = one_cmt_oral(dose, ka, ke, V, times)
        auc_numerical = jnp.trapezoid(conc, times)
        auc_analytical = dose / (V * ke)
        assert jnp.allclose(auc_numerical, auc_analytical, rtol=1e-3)


# ---------------------------------------------------------------------------
# one_cmt_infusion
# ---------------------------------------------------------------------------

class TestOneCmtInfusion:
    """1-compartment zero-order infusion."""

    def test_during_infusion(self):
        """During infusion: C(t) = (Rate / (V*ke)) * (1 - exp(-ke*t))."""
        dose, ke, V, tinf = 100.0, 0.1, 10.0, 2.0
        rate = dose / tinf
        times = jnp.array([0.5, 1.0, 1.5])
        conc = one_cmt_infusion(dose, ke, V, tinf, times)
        expected = (rate / (V * ke)) * (1.0 - jnp.exp(-ke * times))
        assert jnp.allclose(conc, expected, rtol=1e-6)

    def test_end_of_infusion(self):
        """At t=tinf, concentration should match the during-infusion formula."""
        dose, ke, V, tinf = 100.0, 0.1, 10.0, 2.0
        rate = dose / tinf
        conc = one_cmt_infusion(dose, ke, V, tinf, jnp.array([tinf]))
        expected = (rate / (V * ke)) * (1.0 - jnp.exp(-ke * tinf))
        assert jnp.allclose(conc[0], expected, rtol=1e-6)

    def test_after_infusion(self):
        """After infusion, concentration decays exponentially from end-of-infusion."""
        dose, ke, V, tinf = 100.0, 0.1, 10.0, 2.0
        rate = dose / tinf
        c_end = (rate / (V * ke)) * (1.0 - jnp.exp(-ke * tinf))
        times = jnp.array([3.0, 5.0, 10.0])
        conc = one_cmt_infusion(dose, ke, V, tinf, times)
        expected = c_end * jnp.exp(-ke * (times - tinf))
        assert jnp.allclose(conc, expected, rtol=1e-6)

    def test_concentration_at_zero(self):
        conc = one_cmt_infusion(100.0, 0.1, 10.0, 2.0, jnp.array([0.0]))
        assert jnp.allclose(conc, 0.0, atol=1e-10)

    def test_auc_infusion(self):
        """AUC(0->inf) = dose / (V * ke) regardless of infusion duration."""
        dose, ke, V, tinf = 100.0, 0.1, 10.0, 2.0
        times = jnp.linspace(0.0, 200.0, 50000)
        conc = one_cmt_infusion(dose, ke, V, tinf, times)
        auc_numerical = jnp.trapezoid(conc, times)
        auc_analytical = dose / (V * ke)
        assert jnp.allclose(auc_numerical, auc_analytical, rtol=1e-3)


# ---------------------------------------------------------------------------
# two_cmt_bolus
# ---------------------------------------------------------------------------

class TestTwoCmtBolus:
    """2-compartment IV bolus: biexponential decay."""

    def test_initial_concentration(self):
        """C(0) = dose / V1."""
        dose, k10, k12, k21, V1 = 1000.0, 0.1, 0.05, 0.03, 20.0
        conc = two_cmt_bolus(dose, k10, k12, k21, V1, jnp.array([0.0]))
        assert jnp.allclose(conc, jnp.array([dose / V1]), rtol=1e-6)

    def test_biexponential_decay(self):
        """Concentration should follow biexponential: A*exp(-alpha*t) + B*exp(-beta*t)."""
        dose, k10, k12, k21, V1 = 1000.0, 0.3, 0.1, 0.05, 20.0
        times = jnp.array([0.0, 1.0, 2.0, 5.0, 10.0, 20.0])
        conc = two_cmt_bolus(dose, k10, k12, k21, V1, times)

        # Compute alpha, beta (eigenvalues of the 2-cpt system)
        a_sum = k10 + k12 + k21
        discriminant = jnp.sqrt(a_sum**2 - 4.0 * k10 * k21)
        alpha = (a_sum + discriminant) / 2.0
        beta = (a_sum - discriminant) / 2.0

        A = (dose / V1) * (alpha - k21) / (alpha - beta)
        B = (dose / V1) * (k21 - beta) / (alpha - beta)

        expected = A * jnp.exp(-alpha * times) + B * jnp.exp(-beta * times)
        assert jnp.allclose(conc, expected, rtol=1e-6)

    def test_monotonic_decay(self):
        """Central compartment concentration should eventually decay monotonically."""
        dose, k10, k12, k21, V1 = 100.0, 0.5, 0.1, 0.05, 10.0
        times = jnp.linspace(5.0, 50.0, 100)
        conc = two_cmt_bolus(dose, k10, k12, k21, V1, times)
        # After distribution phase, should be monotonically decreasing
        diffs = jnp.diff(conc)
        assert jnp.all(diffs <= 0.0)

    def test_returns_jax_array(self):
        conc = two_cmt_bolus(100.0, 0.1, 0.05, 0.03, 10.0, jnp.array([0.0, 1.0]))
        assert isinstance(conc, jax.Array)

    def test_auc_two_cmt_bolus(self):
        """AUC(0->inf) for 2-cpt IV bolus = dose / (V1 * k10)."""
        dose, k10, k12, k21, V1 = 1000.0, 0.3, 0.1, 0.05, 20.0
        times = jnp.linspace(0.0, 300.0, 100000)
        conc = two_cmt_bolus(dose, k10, k12, k21, V1, times)
        auc_numerical = jnp.trapezoid(conc, times)
        auc_analytical = dose / (V1 * k10)
        assert jnp.allclose(auc_numerical, auc_analytical, rtol=1e-2)


# ---------------------------------------------------------------------------
# two_cmt_oral
# ---------------------------------------------------------------------------

class TestTwoCmtOral:
    """2-compartment oral absorption."""

    def test_concentration_at_zero(self):
        conc = two_cmt_oral(100.0, 1.5, 0.3, 0.1, 0.05, 20.0, jnp.array([0.0]))
        assert jnp.allclose(conc, 0.0, atol=1e-5)

    def test_rises_then_falls(self):
        """Oral dosing should produce a rise-then-fall profile."""
        dose, ka, k10, k12, k21, V1 = 1000.0, 1.5, 0.3, 0.1, 0.05, 20.0
        times = jnp.linspace(0.0, 50.0, 1000)
        conc = two_cmt_oral(dose, ka, k10, k12, k21, V1, times)
        peak_idx = jnp.argmax(conc)
        # Peak should not be at the endpoints
        assert peak_idx > 0
        assert peak_idx < len(times) - 1

    def test_auc_two_cmt_oral(self):
        """AUC(0->inf) for 2-cpt oral with F=1: dose / (V1 * k10)."""
        dose, ka, k10, k12, k21, V1 = 1000.0, 1.5, 0.3, 0.1, 0.05, 20.0
        times = jnp.linspace(0.0, 300.0, 100000)
        conc = two_cmt_oral(dose, ka, k10, k12, k21, V1, times)
        auc_numerical = jnp.trapezoid(conc, times)
        auc_analytical = dose / (V1 * k10)
        assert jnp.allclose(auc_numerical, auc_analytical, rtol=1e-2)

    def test_returns_jax_array(self):
        conc = two_cmt_oral(100.0, 1.5, 0.3, 0.1, 0.05, 20.0, jnp.array([0.0, 1.0]))
        assert isinstance(conc, jax.Array)


# ---------------------------------------------------------------------------
# superposition
# ---------------------------------------------------------------------------

class TestSuperposition:
    """Multiple dose superposition via linear superposition principle."""

    def test_single_dose_matches_direct(self):
        """Superposition with one dose at t=0 should match the single-dose function."""
        dose, ke, V = 100.0, 0.1, 10.0
        eval_times = jnp.array([0.0, 1.0, 5.0, 10.0])

        def single_dose_fn(d, t):
            return one_cmt_bolus(d, ke, V, t)

        conc_super = superposition(single_dose_fn, jnp.array([dose]), jnp.array([0.0]), eval_times)
        conc_direct = one_cmt_bolus(dose, ke, V, eval_times)
        assert jnp.allclose(conc_super, conc_direct, atol=1e-10)

    def test_two_doses_manual(self):
        """Two bolus doses should sum contributions at each time point."""
        dose, ke, V = 100.0, 0.1, 10.0
        dose_times = jnp.array([0.0, 5.0])
        doses = jnp.array([dose, dose])
        eval_times = jnp.array([0.0, 5.0, 10.0])

        def single_dose_fn(d, t):
            return one_cmt_bolus(d, ke, V, t)

        conc = superposition(single_dose_fn, doses, dose_times, eval_times)

        # Manual calculation:
        # t=0:  dose1 at t=0, dose2 hasn't happened -> dose/V
        # t=5:  dose1 at t=5 + dose2 at t=0 -> (dose/V)*exp(-ke*5) + dose/V
        # t=10: dose1 at t=10 + dose2 at t=5 -> (dose/V)*exp(-ke*10) + (dose/V)*exp(-ke*5)
        c0 = dose / V
        expected = jnp.array([
            c0,
            c0 * jnp.exp(-ke * 5.0) + c0,
            c0 * jnp.exp(-ke * 10.0) + c0 * jnp.exp(-ke * 5.0),
        ])
        assert jnp.allclose(conc, expected, rtol=1e-6)

    def test_steady_state_accumulation(self):
        """Repeated dosing should lead to accumulation."""
        dose, ke, V = 100.0, 0.1, 10.0
        tau = 8.0  # dosing interval
        n_doses = 10
        dose_times = jnp.arange(n_doses) * tau
        doses = jnp.full(n_doses, dose)

        def single_dose_fn(d, t):
            return one_cmt_bolus(d, ke, V, t)

        # Trough of first dose vs last dose
        eval_first = jnp.array([tau - 0.01])
        eval_last = jnp.array([dose_times[-1] + tau - 0.01])

        conc_first = superposition(single_dose_fn, doses[:1], dose_times[:1], eval_first)
        conc_last = superposition(single_dose_fn, doses, dose_times, eval_last)
        # Last trough should be higher due to accumulation
        assert conc_last[0] > conc_first[0]


# ---------------------------------------------------------------------------
# linCmt unified interface
# ---------------------------------------------------------------------------

class TestLinCmt:
    """Unified linCmt interface."""

    def test_one_cmt_bolus(self):
        params = {"dose": 100.0, "ke": 0.1, "V": 10.0}
        times = jnp.array([0.0, 1.0, 5.0])
        conc = linCmt(params, times, model_type="one_cmt_bolus")
        expected = one_cmt_bolus(100.0, 0.1, 10.0, times)
        assert jnp.allclose(conc, expected, atol=1e-10)

    def test_one_cmt_oral(self):
        params = {"dose": 100.0, "ka": 1.5, "ke": 0.1, "V": 10.0}
        times = jnp.array([0.0, 1.0, 5.0])
        conc = linCmt(params, times, model_type="one_cmt_oral")
        expected = one_cmt_oral(100.0, 1.5, 0.1, 10.0, times)
        assert jnp.allclose(conc, expected, atol=1e-10)

    def test_one_cmt_infusion(self):
        params = {"dose": 100.0, "ke": 0.1, "V": 10.0, "tinf": 2.0}
        times = jnp.array([0.0, 1.0, 3.0])
        conc = linCmt(params, times, model_type="one_cmt_infusion")
        expected = one_cmt_infusion(100.0, 0.1, 10.0, 2.0, times)
        assert jnp.allclose(conc, expected, atol=1e-10)

    def test_two_cmt_bolus(self):
        params = {"dose": 1000.0, "k10": 0.3, "k12": 0.1, "k21": 0.05, "V1": 20.0}
        times = jnp.array([0.0, 1.0, 5.0])
        conc = linCmt(params, times, model_type="two_cmt_bolus")
        expected = two_cmt_bolus(1000.0, 0.3, 0.1, 0.05, 20.0, times)
        assert jnp.allclose(conc, expected, atol=1e-10)

    def test_two_cmt_oral(self):
        params = {"dose": 1000.0, "ka": 1.5, "k10": 0.3, "k12": 0.1, "k21": 0.05, "V1": 20.0}
        times = jnp.array([0.0, 1.0, 5.0])
        conc = linCmt(params, times, model_type="two_cmt_oral")
        expected = two_cmt_oral(1000.0, 1.5, 0.3, 0.1, 0.05, 20.0, times)
        assert jnp.allclose(conc, expected, atol=1e-10)

    def test_unknown_model_type_raises(self):
        with pytest.raises(ValueError, match="Unknown model_type"):
            linCmt({"dose": 100.0}, jnp.array([0.0]), model_type="four_cmt_bolus")


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------

class TestJITCompatibility:
    """All functions should be JIT-compilable."""

    def test_one_cmt_bolus_jit(self):
        f = jax.jit(one_cmt_bolus)
        conc = f(100.0, 0.1, 10.0, jnp.array([0.0, 1.0]))
        expected = one_cmt_bolus(100.0, 0.1, 10.0, jnp.array([0.0, 1.0]))
        assert jnp.allclose(conc, expected, atol=1e-10)

    def test_one_cmt_oral_jit(self):
        f = jax.jit(one_cmt_oral)
        conc = f(100.0, 1.5, 0.1, 10.0, jnp.array([0.0, 1.0]))
        expected = one_cmt_oral(100.0, 1.5, 0.1, 10.0, jnp.array([0.0, 1.0]))
        assert jnp.allclose(conc, expected, atol=1e-10)

    def test_one_cmt_infusion_jit(self):
        f = jax.jit(one_cmt_infusion)
        conc = f(100.0, 0.1, 10.0, 2.0, jnp.array([0.0, 1.0, 3.0]))
        expected = one_cmt_infusion(100.0, 0.1, 10.0, 2.0, jnp.array([0.0, 1.0, 3.0]))
        assert jnp.allclose(conc, expected, atol=1e-10)

    def test_two_cmt_bolus_jit(self):
        f = jax.jit(two_cmt_bolus)
        conc = f(1000.0, 0.3, 0.1, 0.05, 20.0, jnp.array([0.0, 1.0]))
        expected = two_cmt_bolus(1000.0, 0.3, 0.1, 0.05, 20.0, jnp.array([0.0, 1.0]))
        assert jnp.allclose(conc, expected, atol=1e-10)

    def test_two_cmt_oral_jit(self):
        f = jax.jit(two_cmt_oral)
        conc = f(1000.0, 1.5, 0.3, 0.1, 0.05, 20.0, jnp.array([0.0, 1.0]))
        expected = two_cmt_oral(1000.0, 1.5, 0.3, 0.1, 0.05, 20.0, jnp.array([0.0, 1.0]))
        assert jnp.allclose(conc, expected, atol=1e-4)
