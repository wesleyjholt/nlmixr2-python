"""Tests for per-subject predictions and per-subject log-likelihood (phi)."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from nlmixr2.diagnostics import compute_per_subject_predictions, compute_phi


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_model_func(params, times):
    """Linear model: pred = A * time + B."""
    a = params["A"]
    b = params["B"]
    return a * times + b


def _make_data():
    """Two subjects, 3 observations each."""
    ids = jnp.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    times = jnp.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    dv = jnp.array([1.0, 3.0, 5.0, 2.0, 4.0, 6.0])
    return {"id": ids, "time": times, "dv": dv}


def _make_fixed_params():
    return {"A": 2.0, "B": 1.0}


def _make_etas():
    """2 subjects x 2 etas (one per param)."""
    return jnp.array([
        [0.1, -0.1],   # subject 1
        [-0.2, 0.3],   # subject 2
    ])


# ---------------------------------------------------------------------------
# compute_per_subject_predictions
# ---------------------------------------------------------------------------

class TestComputePerSubjectPredictions:
    def test_output_keys(self):
        data = _make_data()
        result = compute_per_subject_predictions(
            _simple_model_func, data, _make_fixed_params(), _make_etas(),
        )
        expected_keys = {"id", "time", "dv", "pred", "ipred", "res", "ires"}
        assert set(result.keys()) == expected_keys

    def test_pred_uses_zero_etas(self):
        data = _make_data()
        fixed = _make_fixed_params()
        etas = _make_etas()
        result = compute_per_subject_predictions(
            _simple_model_func, data, fixed, etas,
        )
        # PRED should be computed with etas=0, i.e. just A*time + B
        pred = np.asarray(result["pred"])
        times = np.asarray(data["time"])
        expected = fixed["A"] * times + fixed["B"]
        np.testing.assert_allclose(pred, expected, atol=1e-6)

    def test_ipred_uses_actual_etas(self):
        data = _make_data()
        fixed = _make_fixed_params()
        etas = _make_etas()
        result = compute_per_subject_predictions(
            _simple_model_func, data, fixed, etas,
        )
        ipred = np.asarray(result["ipred"])
        times = np.asarray(data["time"])
        ids = np.asarray(data["id"])
        etas_np = np.asarray(etas)

        # Subject 1 (id=1): A+eta[0,0]=2.1, B+eta[0,1]=0.9
        # Subject 2 (id=2): A+eta[1,0]=1.8, B+eta[1,1]=1.3
        expected = np.zeros_like(times)
        for idx in range(len(times)):
            if ids[idx] == 1.0:
                a = fixed["A"] + etas_np[0, 0]
                b = fixed["B"] + etas_np[0, 1]
            else:
                a = fixed["A"] + etas_np[1, 0]
                b = fixed["B"] + etas_np[1, 1]
            expected[idx] = a * times[idx] + b

        np.testing.assert_allclose(ipred, expected, atol=1e-6)

    def test_res_equals_dv_minus_pred(self):
        data = _make_data()
        result = compute_per_subject_predictions(
            _simple_model_func, data, _make_fixed_params(), _make_etas(),
        )
        res = np.asarray(result["res"])
        dv = np.asarray(result["dv"])
        pred = np.asarray(result["pred"])
        np.testing.assert_allclose(res, dv - pred, atol=1e-6)

    def test_ires_equals_dv_minus_ipred(self):
        data = _make_data()
        result = compute_per_subject_predictions(
            _simple_model_func, data, _make_fixed_params(), _make_etas(),
        )
        ires = np.asarray(result["ires"])
        dv = np.asarray(result["dv"])
        ipred = np.asarray(result["ipred"])
        np.testing.assert_allclose(ires, dv - ipred, atol=1e-6)

    def test_output_arrays_same_length(self):
        data = _make_data()
        result = compute_per_subject_predictions(
            _simple_model_func, data, _make_fixed_params(), _make_etas(),
        )
        n = len(data["dv"])
        for key in ("id", "time", "dv", "pred", "ipred", "res", "ires"):
            assert len(result[key]) == n, f"Length mismatch for {key}"


# ---------------------------------------------------------------------------
# compute_phi
# ---------------------------------------------------------------------------

class TestComputePhi:
    def test_returns_dict_with_subject_keys(self):
        data = _make_data()
        phi = compute_phi(
            _simple_model_func, data, _make_fixed_params(), _make_etas(), sigma=1.0,
        )
        assert isinstance(phi, dict)
        # Should have one entry per unique subject ID
        unique_ids = np.unique(np.asarray(data["id"]))
        for sid in unique_ids:
            assert float(sid) in phi or int(sid) in phi

    def test_phi_values_are_negative(self):
        data = _make_data()
        phi = compute_phi(
            _simple_model_func, data, _make_fixed_params(), _make_etas(), sigma=1.0,
        )
        for subj_id, value in phi.items():
            assert value < 0.0, f"phi for subject {subj_id} should be negative, got {value}"

    def test_phi_manual_computation(self):
        """Check phi_i against the formula:
        phi_i = -0.5 * sum_j [(dv_ij - ipred_ij)^2 / sigma^2 + log(2*pi*sigma^2)]
        """
        data = _make_data()
        fixed = _make_fixed_params()
        etas = _make_etas()
        sigma = 1.5
        phi = compute_phi(
            _simple_model_func, data, fixed, etas, sigma=sigma,
        )

        ids_np = np.asarray(data["id"])
        times_np = np.asarray(data["time"])
        dv_np = np.asarray(data["dv"])
        etas_np = np.asarray(etas)
        unique_ids = np.unique(ids_np)
        param_names = list(fixed.keys())

        for i, sid in enumerate(unique_ids):
            mask = ids_np == sid
            subj_times = times_np[mask]
            subj_dv = dv_np[mask]
            # Build individual params
            indiv = dict(fixed)
            for j, name in enumerate(param_names):
                if j < etas_np.shape[1]:
                    indiv[name] = fixed[name] + etas_np[i, j]
            ipred = np.array([indiv["A"] * t + indiv["B"] for t in subj_times])
            n_obs = len(subj_dv)
            expected_phi = -0.5 * np.sum(
                (subj_dv - ipred) ** 2 / sigma ** 2 + np.log(2 * np.pi * sigma ** 2)
            )
            # Look up by float or int key
            key = float(sid)
            assert key in phi, f"Missing key {key} in phi"
            np.testing.assert_allclose(phi[key], expected_phi, atol=1e-6)

    def test_phi_sum_relates_to_objective(self):
        """Sum of phi values should be proportional to a total log-likelihood."""
        data = _make_data()
        fixed = _make_fixed_params()
        etas = _make_etas()
        sigma = 1.0
        phi = compute_phi(
            _simple_model_func, data, fixed, etas, sigma=sigma,
        )
        total_phi = sum(phi.values())
        # total_phi is total log-likelihood; -2*total_phi is ~objective
        # It should be finite and negative
        assert math.isfinite(total_phi)
        assert total_phi < 0.0
        # -2 * total_phi should be positive (like an objective function value)
        assert -2.0 * total_phi > 0.0
