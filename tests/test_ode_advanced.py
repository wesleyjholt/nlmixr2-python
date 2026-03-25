"""Tests for advanced ODE dosing features: lag time, bioavailability, transit compartments."""

import jax
import jax.numpy as jnp
import pytest

from nlmixr2.ode import solve_ode, transit_compartments


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _rhs_1cpt(t, y, params):
    """dy/dt = -k * y for a single central compartment."""
    k = params["k"]
    return jnp.array([-k * y[0]])


def _rhs_1cpt_abs(t, y, params):
    """1-compartment with first-order absorption from a depot (cpt 0 -> cpt 1).

    y[0] = depot (gut), y[1] = central
    dy0/dt = -ka * y0
    dy1/dt =  ka * y0 - k * y1
    """
    ka = params["ka"]
    k = params["k"]
    dy0 = -ka * y[0]
    dy1 = ka * y[0] - k * y[1]
    return jnp.array([dy0, dy1])


def _solve_1cpt_bolus(dose, k, t_eval, **kw):
    """Convenience wrapper for 1-cpt IV bolus."""
    return solve_ode(
        rhs=_rhs_1cpt,
        t_span=(0.0, float(t_eval[-1])),
        y0=jnp.array([0.0]),
        params={"k": k},
        t_eval=t_eval,
        dosing_events=[{"time": 0.0, "amount": dose, "compartment": 0, **kw}],
    )


# ---------------------------------------------------------------------------
# Lag time tests
# ---------------------------------------------------------------------------

class TestLagTime:

    def test_lag_delays_dose(self):
        """Concentration at t < tlag should be ~0."""
        k = 0.1
        dose = 100.0
        tlag = 5.0
        t_eval = jnp.linspace(0.0, 24.0, 200)

        result = solve_ode(
            rhs=_rhs_1cpt,
            t_span=(0.0, 24.0),
            y0=jnp.array([0.0]),
            params={"k": k},
            t_eval=t_eval,
            dosing_events=[
                {"time": 0.0, "amount": dose, "compartment": 0, "lag_time": tlag}
            ],
        )

        # Before the lag all values should be ~0
        pre_lag = result[t_eval < tlag - 0.1, 0]
        assert jnp.all(jnp.abs(pre_lag) < 1e-3), "Non-zero concentration before lag"

        # After lag, drug should appear
        post_lag = result[t_eval > tlag + 1.0, 0]
        assert jnp.max(post_lag) > 10.0, "No drug after lag time"

    def test_lag_zero_matches_original(self):
        """lag_time=0 should produce the same result as no lag_time key."""
        k = 0.1
        dose = 100.0
        t_eval = jnp.linspace(0.0, 24.0, 50)

        result_no_lag = solve_ode(
            rhs=_rhs_1cpt,
            t_span=(0.0, 24.0),
            y0=jnp.array([0.0]),
            params={"k": k},
            t_eval=t_eval,
            dosing_events=[{"time": 0.0, "amount": dose, "compartment": 0}],
        )

        result_lag0 = solve_ode(
            rhs=_rhs_1cpt,
            t_span=(0.0, 24.0),
            y0=jnp.array([0.0]),
            params={"k": k},
            t_eval=t_eval,
            dosing_events=[
                {"time": 0.0, "amount": dose, "compartment": 0, "lag_time": 0.0}
            ],
        )

        assert jnp.allclose(result_no_lag, result_lag0, atol=1e-3)


# ---------------------------------------------------------------------------
# Bioavailability tests
# ---------------------------------------------------------------------------

class TestBioavailability:

    def test_half_bioavailability(self):
        """F=0.5 should give half the concentration vs F=1.0."""
        k = 0.1
        dose = 100.0
        t_eval = jnp.linspace(0.0, 24.0, 50)

        result_f1 = solve_ode(
            rhs=_rhs_1cpt,
            t_span=(0.0, 24.0),
            y0=jnp.array([0.0]),
            params={"k": k},
            t_eval=t_eval,
            dosing_events=[
                {"time": 0.0, "amount": dose, "compartment": 0, "bioavailability": 1.0}
            ],
        )

        result_f05 = solve_ode(
            rhs=_rhs_1cpt,
            t_span=(0.0, 24.0),
            y0=jnp.array([0.0]),
            params={"k": k},
            t_eval=t_eval,
            dosing_events=[
                {"time": 0.0, "amount": dose, "compartment": 0, "bioavailability": 0.5}
            ],
        )

        # F=0.5 result should be half of F=1.0 (linear system)
        assert jnp.allclose(result_f05, result_f1 * 0.5, atol=0.5)

    def test_bioavailability_one_matches_original(self):
        """F=1.0 should match result without bioavailability key."""
        k = 0.1
        dose = 100.0
        t_eval = jnp.linspace(0.0, 24.0, 50)

        result_orig = solve_ode(
            rhs=_rhs_1cpt,
            t_span=(0.0, 24.0),
            y0=jnp.array([0.0]),
            params={"k": k},
            t_eval=t_eval,
            dosing_events=[{"time": 0.0, "amount": dose, "compartment": 0}],
        )

        result_f1 = solve_ode(
            rhs=_rhs_1cpt,
            t_span=(0.0, 24.0),
            y0=jnp.array([0.0]),
            params={"k": k},
            t_eval=t_eval,
            dosing_events=[
                {"time": 0.0, "amount": dose, "compartment": 0, "bioavailability": 1.0}
            ],
        )

        assert jnp.allclose(result_orig, result_f1, atol=1e-3)


# ---------------------------------------------------------------------------
# Combined lag + bioavailability
# ---------------------------------------------------------------------------

class TestCombinedLagBioavailability:

    def test_lag_plus_bioavailability(self):
        """Combined lag_time + bioavailability should delay AND scale."""
        k = 0.1
        dose = 100.0
        tlag = 3.0
        F = 0.5
        t_eval = jnp.linspace(0.0, 24.0, 200)

        result = solve_ode(
            rhs=_rhs_1cpt,
            t_span=(0.0, 24.0),
            y0=jnp.array([0.0]),
            params={"k": k},
            t_eval=t_eval,
            dosing_events=[
                {
                    "time": 0.0,
                    "amount": dose,
                    "compartment": 0,
                    "lag_time": tlag,
                    "bioavailability": F,
                }
            ],
        )

        # Before lag: zero
        pre_lag = result[t_eval < tlag - 0.1, 0]
        assert jnp.all(jnp.abs(pre_lag) < 1e-3)

        # After lag: peak should be ~F * dose = 50
        post_lag = result[t_eval > tlag, 0]
        assert jnp.max(post_lag) < dose * 0.6  # less than full dose
        assert jnp.max(post_lag) > dose * 0.3  # but not zero


# ---------------------------------------------------------------------------
# Transit compartment tests
# ---------------------------------------------------------------------------

class TestTransitCompartments:

    def test_transit_delays_absorption(self):
        """Transit compartments should delay the peak in the central compartment."""
        ka = 1.0
        k = 0.1
        ktr = 2.0
        dose = 100.0
        n_transit = 3
        t_eval = jnp.linspace(0.0, 24.0, 300)

        # Without transit: dose into depot (cpt 0), absorb into central (cpt 1)
        result_no_transit = solve_ode(
            rhs=_rhs_1cpt_abs,
            t_span=(0.0, 24.0),
            y0=jnp.array([0.0, 0.0]),
            params={"ka": ka, "k": k},
            t_eval=t_eval,
            dosing_events=[{"time": 0.0, "amount": dose, "compartment": 0}],
        )

        # With transit: build augmented system
        transit_fn = transit_compartments(n_transit, ktr, dose_compartment=0)

        def rhs_with_transit(t, y, params):
            # y has n_transit + 2 compartments:
            # y[0..n_transit-1] = transit compartments
            # y[n_transit] = depot
            # y[n_transit+1] = central
            ka_val = params["ka"]
            k_val = params["k"]

            dydt = jnp.zeros_like(y)
            # Transit chain
            dydt = transit_fn(t, y, dydt)
            # Depot: receives from last transit, absorbed into central
            depot = y[n_transit]
            central = y[n_transit + 1]
            dydt = dydt.at[n_transit].add(-ka_val * depot)
            dydt = dydt.at[n_transit + 1].set(ka_val * depot - k_val * central)
            return dydt

        y0_transit = jnp.zeros(n_transit + 2)
        result_transit = solve_ode(
            rhs=rhs_with_transit,
            t_span=(0.0, 24.0),
            y0=y0_transit,
            params={"ka": ka, "k": k},
            t_eval=t_eval,
            dosing_events=[{"time": 0.0, "amount": dose, "compartment": 0}],
        )

        # Central compartment index
        central_no_transit = result_no_transit[:, 1]
        central_transit = result_transit[:, n_transit + 1]

        peak_time_no_transit = t_eval[int(jnp.argmax(central_no_transit))]
        peak_time_transit = t_eval[int(jnp.argmax(central_transit))]

        # Transit compartments should delay the peak
        assert peak_time_transit > peak_time_no_transit

    def test_more_transits_later_peak(self):
        """More transit compartments should produce a later and flatter peak."""
        ka = 1.0
        k = 0.1
        ktr = 2.0
        dose = 100.0
        t_eval = jnp.linspace(0.0, 24.0, 300)

        peak_times = []
        peak_values = []

        for n_transit in [2, 5]:
            transit_fn = transit_compartments(n_transit, ktr, dose_compartment=0)

            def make_rhs(nt, tfn):
                def rhs(t, y, params):
                    ka_val = params["ka"]
                    k_val = params["k"]
                    dydt = jnp.zeros_like(y)
                    dydt = tfn(t, y, dydt)
                    depot = y[nt]
                    central = y[nt + 1]
                    dydt = dydt.at[nt].add(-ka_val * depot)
                    dydt = dydt.at[nt + 1].set(ka_val * depot - k_val * central)
                    return dydt
                return rhs

            rhs_fn = make_rhs(n_transit, transit_fn)
            y0 = jnp.zeros(n_transit + 2)
            result = solve_ode(
                rhs=rhs_fn,
                t_span=(0.0, 24.0),
                y0=y0,
                params={"ka": ka, "k": k},
                t_eval=t_eval,
                dosing_events=[{"time": 0.0, "amount": dose, "compartment": 0}],
            )

            central = result[:, n_transit + 1]
            peak_times.append(float(t_eval[int(jnp.argmax(central))]))
            peak_values.append(float(jnp.max(central)))

        # More transits -> later peak
        assert peak_times[1] > peak_times[0], (
            f"5-transit peak ({peak_times[1]}) should be later than 2-transit ({peak_times[0]})"
        )
        # More transits -> flatter (lower) peak
        assert peak_values[1] < peak_values[0], (
            f"5-transit peak ({peak_values[1]}) should be lower than 2-transit ({peak_values[0]})"
        )
