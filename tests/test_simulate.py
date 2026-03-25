"""Tests for the simulation module (rxSolve equivalent)."""

import jax.numpy as jnp
import numpy as np
import pytest

from nlmixr2.event_table import EventTable, et
from nlmixr2.omega import OmegaBlock, omega
from nlmixr2.simulate import SimulationResult, simulate, to_dataframe_dict


# ---------------------------------------------------------------------------
# Helper model functions
# ---------------------------------------------------------------------------

def _analytical_one_cmt(params, times):
    """Simple 1-cmt IV bolus: C(t) = (dose/V) * exp(-ke*t)."""
    dose = params["dose"]
    ke = params["ke"]
    V = params["V"]
    return (dose / V) * jnp.exp(-ke * times)


def _ode_rhs_one_cmt(t, y, params):
    """ODE RHS for 1-cmt model: dy/dt = -ke * y."""
    ke = params["ke"]
    return -ke * y


# ---------------------------------------------------------------------------
# Tests: SimulationResult fields
# ---------------------------------------------------------------------------

class TestSimulationResultFields:
    def test_result_has_required_fields(self):
        params = {"dose": 100.0, "ke": 0.1, "V": 10.0}
        times = jnp.array([0.0, 1.0, 2.0, 4.0, 8.0])
        ev = {"time": times, "amt": jnp.array([0.0] * 5), "evid": jnp.array([0] * 5)}

        result = simulate(_analytical_one_cmt, params, ev, n_subjects=1, seed=42)

        assert isinstance(result, SimulationResult)
        assert hasattr(result, "subjects")
        assert hasattr(result, "population_params")
        assert hasattr(result, "omega")
        assert hasattr(result, "sigma")
        assert result.population_params == params
        assert result.omega is None
        assert result.sigma == 0.0

    def test_subject_dict_has_required_keys(self):
        params = {"dose": 100.0, "ke": 0.1, "V": 10.0}
        times = jnp.array([0.0, 1.0, 2.0])
        ev = {"time": times, "amt": jnp.array([0.0] * 3), "evid": jnp.array([0] * 3)}

        result = simulate(_analytical_one_cmt, params, ev, n_subjects=1, seed=0)
        subj = result.subjects[0]

        assert "id" in subj
        assert "time" in subj
        assert "pred" in subj
        assert "ipred" in subj
        assert "dv" in subj
        assert subj["id"] == 0


# ---------------------------------------------------------------------------
# Tests: Single subject, no variability
# ---------------------------------------------------------------------------

class TestSingleSubjectNoVariability:
    def test_pred_equals_ipred_equals_dv(self):
        params = {"dose": 100.0, "ke": 0.1, "V": 10.0}
        times = jnp.array([0.0, 1.0, 2.0, 4.0, 8.0])
        ev = {"time": times, "amt": jnp.array([0.0] * 5), "evid": jnp.array([0] * 5)}

        result = simulate(_analytical_one_cmt, params, ev, n_subjects=1, seed=0)
        subj = result.subjects[0]

        np.testing.assert_allclose(subj["pred"], subj["ipred"], atol=1e-6)
        np.testing.assert_allclose(subj["pred"], subj["dv"], atol=1e-6)

    def test_analytical_predictions_correct(self):
        params = {"dose": 100.0, "ke": 0.1, "V": 10.0}
        times = jnp.array([0.0, 1.0, 2.0, 4.0, 8.0])
        ev = {"time": times, "amt": jnp.array([0.0] * 5), "evid": jnp.array([0] * 5)}

        result = simulate(_analytical_one_cmt, params, ev, n_subjects=1, seed=0)
        subj = result.subjects[0]

        expected = (100.0 / 10.0) * jnp.exp(-0.1 * times)
        np.testing.assert_allclose(subj["pred"], expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# Tests: Single subject with residual error
# ---------------------------------------------------------------------------

class TestResidualError:
    def test_dv_differs_from_pred_with_sigma(self):
        params = {"dose": 100.0, "ke": 0.1, "V": 10.0}
        times = jnp.array([0.0, 1.0, 2.0, 4.0, 8.0])
        ev = {"time": times, "amt": jnp.array([0.0] * 5), "evid": jnp.array([0] * 5)}

        result = simulate(
            _analytical_one_cmt, params, ev, n_subjects=1, sigma=1.0, seed=42
        )
        subj = result.subjects[0]

        # pred and ipred should still match (no omega)
        np.testing.assert_allclose(subj["pred"], subj["ipred"], atol=1e-6)
        # dv should differ from pred due to residual error
        assert not np.allclose(subj["dv"], subj["pred"], atol=1e-3)

    def test_sigma_stored_in_result(self):
        params = {"dose": 100.0, "ke": 0.1, "V": 10.0}
        times = jnp.array([0.0, 1.0, 2.0])
        ev = {"time": times, "amt": jnp.array([0.0] * 3), "evid": jnp.array([0] * 3)}

        result = simulate(
            _analytical_one_cmt, params, ev, n_subjects=1, sigma=0.5, seed=0
        )
        assert result.sigma == 0.5


# ---------------------------------------------------------------------------
# Tests: Multiple subjects with omega (BSV)
# ---------------------------------------------------------------------------

class TestMultipleSubjectsWithOmega:
    def test_different_ipred_per_subject(self):
        params = {"dose": 100.0, "ke": 0.1, "V": 10.0}
        times = jnp.array([0.0, 1.0, 2.0, 4.0, 8.0])
        ev = {"time": times, "amt": jnp.array([0.0] * 5), "evid": jnp.array([0] * 5)}

        om = omega({"eta.ke": 0.04, "eta.V": 0.04})

        result = simulate(
            _analytical_one_cmt, params, ev, n_subjects=5, omega=om, seed=42
        )

        assert len(result.subjects) == 5
        # ipred should differ across subjects due to BSV
        ipreds = [np.array(s["ipred"]) for s in result.subjects]
        # At least some subjects should have different ipred
        all_same = all(np.allclose(ipreds[0], ip, atol=1e-6) for ip in ipreds[1:])
        assert not all_same

    def test_pred_same_across_subjects(self):
        """Population predictions should be identical for all subjects."""
        params = {"dose": 100.0, "ke": 0.1, "V": 10.0}
        times = jnp.array([0.0, 1.0, 2.0, 4.0])
        ev = {"time": times, "amt": jnp.array([0.0] * 4), "evid": jnp.array([0] * 4)}

        om = omega({"eta.ke": 0.04, "eta.V": 0.04})

        result = simulate(
            _analytical_one_cmt, params, ev, n_subjects=3, omega=om, seed=99
        )

        for s in result.subjects:
            np.testing.assert_allclose(
                s["pred"], result.subjects[0]["pred"], atol=1e-6
            )

    def test_subject_ids_are_sequential(self):
        params = {"dose": 100.0, "ke": 0.1, "V": 10.0}
        times = jnp.array([0.0, 1.0])
        ev = {"time": times, "amt": jnp.array([0.0] * 2), "evid": jnp.array([0] * 2)}

        om = omega({"eta.ke": 0.04})

        result = simulate(
            _analytical_one_cmt, params, ev, n_subjects=4, omega=om, seed=0
        )

        ids = [s["id"] for s in result.subjects]
        assert ids == [0, 1, 2, 3]

    def test_omega_stored_in_result(self):
        params = {"dose": 100.0, "ke": 0.1, "V": 10.0}
        times = jnp.array([0.0, 1.0])
        ev = {"time": times, "amt": jnp.array([0.0] * 2), "evid": jnp.array([0] * 2)}

        om = omega({"eta.ke": 0.04})

        result = simulate(
            _analytical_one_cmt, params, ev, n_subjects=1, omega=om, seed=0
        )
        assert result.omega is om


# ---------------------------------------------------------------------------
# Tests: EventTable input
# ---------------------------------------------------------------------------

class TestEventTableInput:
    def test_with_event_table_object(self):
        params = {"dose": 100.0, "ke": 0.1, "V": 10.0}
        ev_table = (
            et()
            .add_dosing(amt=100.0, time=0.0)
            .add_sampling([0.0, 1.0, 2.0, 4.0, 8.0])
        )

        result = simulate(_analytical_one_cmt, params, ev_table, n_subjects=1, seed=0)

        assert len(result.subjects) == 1
        subj = result.subjects[0]
        assert len(subj["time"]) == 5  # 5 observation times


# ---------------------------------------------------------------------------
# Tests: Dict input
# ---------------------------------------------------------------------------

class TestDictInput:
    def test_with_dict_input(self):
        params = {"dose": 100.0, "ke": 0.1, "V": 10.0}
        times = jnp.array([0.0, 1.0, 2.0, 4.0, 8.0])
        ev = {"time": times, "amt": jnp.array([0.0] * 5), "evid": jnp.array([0] * 5)}

        result = simulate(_analytical_one_cmt, params, ev, n_subjects=1, seed=0)

        assert len(result.subjects) == 1
        subj = result.subjects[0]
        np.testing.assert_allclose(np.array(subj["time"]), np.array(times), atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: to_dataframe_dict
# ---------------------------------------------------------------------------

class TestToDataframeDict:
    def test_output_structure(self):
        params = {"dose": 100.0, "ke": 0.1, "V": 10.0}
        times = jnp.array([0.0, 1.0, 2.0])
        ev = {"time": times, "amt": jnp.array([0.0] * 3), "evid": jnp.array([0] * 3)}

        result = simulate(_analytical_one_cmt, params, ev, n_subjects=2, seed=0)
        df_dict = to_dataframe_dict(result)

        assert "id" in df_dict
        assert "time" in df_dict
        assert "pred" in df_dict
        assert "ipred" in df_dict
        assert "dv" in df_dict

        # 2 subjects x 3 time points = 6 rows
        assert len(df_dict["id"]) == 6
        assert len(df_dict["time"]) == 6
        assert len(df_dict["pred"]) == 6

    def test_subjects_concatenated_in_order(self):
        params = {"dose": 100.0, "ke": 0.1, "V": 10.0}
        times = jnp.array([0.0, 1.0])
        ev = {"time": times, "amt": jnp.array([0.0] * 2), "evid": jnp.array([0] * 2)}

        result = simulate(_analytical_one_cmt, params, ev, n_subjects=3, seed=0)
        df_dict = to_dataframe_dict(result)

        # IDs should be [0, 0, 1, 1, 2, 2]
        expected_ids = [0, 0, 1, 1, 2, 2]
        assert list(df_dict["id"]) == expected_ids


# ---------------------------------------------------------------------------
# Tests: Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_results(self):
        params = {"dose": 100.0, "ke": 0.1, "V": 10.0}
        times = jnp.array([0.0, 1.0, 2.0, 4.0])
        ev = {"time": times, "amt": jnp.array([0.0] * 4), "evid": jnp.array([0] * 4)}

        om = omega({"eta.ke": 0.04, "eta.V": 0.04})

        r1 = simulate(
            _analytical_one_cmt, params, ev, n_subjects=3, omega=om, sigma=0.5, seed=123
        )
        r2 = simulate(
            _analytical_one_cmt, params, ev, n_subjects=3, omega=om, sigma=0.5, seed=123
        )

        for s1, s2 in zip(r1.subjects, r2.subjects):
            np.testing.assert_allclose(s1["ipred"], s2["ipred"], atol=1e-6)
            np.testing.assert_allclose(s1["dv"], s2["dv"], atol=1e-6)

    def test_different_seeds_different_results(self):
        params = {"dose": 100.0, "ke": 0.1, "V": 10.0}
        times = jnp.array([0.0, 1.0, 2.0, 4.0])
        ev = {"time": times, "amt": jnp.array([0.0] * 4), "evid": jnp.array([0] * 4)}

        om = omega({"eta.ke": 0.04, "eta.V": 0.04})

        r1 = simulate(
            _analytical_one_cmt, params, ev, n_subjects=3, omega=om, sigma=0.5, seed=1
        )
        r2 = simulate(
            _analytical_one_cmt, params, ev, n_subjects=3, omega=om, sigma=0.5, seed=999
        )

        # At least one subject should have different dv
        any_different = False
        for s1, s2 in zip(r1.subjects, r2.subjects):
            if not np.allclose(s1["dv"], s2["dv"], atol=1e-6):
                any_different = True
                break
        assert any_different


# ---------------------------------------------------------------------------
# Tests: ODE model support
# ---------------------------------------------------------------------------

class TestODEModel:
    def test_ode_model_flag(self):
        """Test that ODE models can be used by passing ode_rhs + initial state."""
        params = {"ke": 0.1}
        times = jnp.array([0.0, 1.0, 2.0, 4.0, 8.0])
        ev = {"time": times, "amt": jnp.array([0.0] * 5), "evid": jnp.array([0] * 5)}

        result = simulate(
            _ode_rhs_one_cmt,
            params,
            ev,
            n_subjects=1,
            seed=0,
            y0=jnp.array([10.0]),  # dose/V = 100/10 = 10
            ode_t_span=(0.0, 10.0),
        )

        subj = result.subjects[0]
        expected = 10.0 * jnp.exp(-0.1 * times)
        np.testing.assert_allclose(subj["pred"], expected, rtol=1e-3)
