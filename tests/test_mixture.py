"""Tests for mixture (latent class) model support."""

from __future__ import annotations

import numpy as np
import pytest

from nlmixr2.mixture import (
    MixtureResult,
    MixtureSpec,
    classify_subjects,
    estimate_mixture,
    mixture_log_likelihood,
)


# ---------------------------------------------------------------------------
# MixtureSpec
# ---------------------------------------------------------------------------

class TestMixtureSpec:
    def test_creation(self):
        spec = MixtureSpec(
            n_classes=2,
            class_params={0: {"CL": 1.0}, 1: {"CL": 5.0}},
            mixing_probs=(0.6, 0.4),
        )
        assert spec.n_classes == 2
        assert spec.class_params[0] == {"CL": 1.0}
        assert spec.mixing_probs == (0.6, 0.4)

    def test_mixing_probs_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1"):
            MixtureSpec(
                n_classes=2,
                class_params={0: {}, 1: {}},
                mixing_probs=(0.3, 0.3),
            )


# ---------------------------------------------------------------------------
# mixture_log_likelihood
# ---------------------------------------------------------------------------

class TestMixtureLogLikelihood:
    def test_single_class_reduces_to_normal(self):
        """With one class, result equals normal log-likelihood."""
        rng = np.random.default_rng(42)
        dv = rng.normal(10.0, 1.0, size=50)
        pred = np.full(50, 10.0)
        sigma = 1.0

        ll_mix = mixture_log_likelihood(dv, [pred], sigma, (1.0,))

        # Manual normal log-likelihood
        resid = dv - pred
        ll_normal = -0.5 * np.sum(
            np.log(2 * np.pi * sigma) + resid ** 2 / sigma
        )
        np.testing.assert_allclose(ll_mix, ll_normal, rtol=1e-6)

    def test_two_classes(self):
        """Two-class mixture log-likelihood is larger than either class alone."""
        rng = np.random.default_rng(7)
        dv = np.concatenate([
            rng.normal(5.0, 1.0, size=30),
            rng.normal(15.0, 1.0, size=30),
        ])
        pred_a = np.full(60, 5.0)
        pred_b = np.full(60, 15.0)
        sigma = 1.0
        probs = (0.5, 0.5)

        ll_mix = mixture_log_likelihood(dv, [pred_a, pred_b], sigma, probs)

        # Each single-class likelihood should be worse
        ll_a = mixture_log_likelihood(dv, [pred_a], sigma, (1.0,))
        ll_b = mixture_log_likelihood(dv, [pred_b], sigma, (1.0,))
        assert ll_mix > ll_a
        assert ll_mix > ll_b

    def test_log_sum_exp_numerical_stability(self):
        """Large residuals should not cause overflow/nan."""
        dv = np.array([1e6])
        pred_a = np.array([0.0])
        pred_b = np.array([1e6])
        sigma = 1.0

        ll = mixture_log_likelihood(dv, [pred_a, pred_b], sigma, (0.5, 0.5))
        assert np.isfinite(ll)


# ---------------------------------------------------------------------------
# classify_subjects
# ---------------------------------------------------------------------------

def _linear_model(params, times):
    """Simple linear model: y = slope * t."""
    return params["slope"] * times


class TestClassifySubjects:
    def test_assigns_correct_class_well_separated(self):
        """Subjects with very different slopes are classified correctly."""
        rng = np.random.default_rng(99)
        n_per = 10
        times = np.linspace(0, 5, n_per)

        # Subject 0 has slope ~2, Subject 1 has slope ~10
        ids = np.concatenate([np.zeros(n_per), np.ones(n_per)])
        t_all = np.tile(times, 2)
        dv = np.concatenate([
            2.0 * times + rng.normal(0, 0.1, n_per),
            10.0 * times + rng.normal(0, 0.1, n_per),
        ])

        data = {"id": ids, "time": t_all, "dv": dv}
        params_per_class = [{"slope": 2.0}, {"slope": 10.0}]
        mixing_probs = (0.5, 0.5)
        sigma = 0.1

        result = classify_subjects(
            data, _linear_model, params_per_class, mixing_probs, sigma,
        )
        assert result[0.0][0] == 0  # subject 0 -> class 0
        assert result[1.0][0] == 1  # subject 1 -> class 1

    def test_posterior_probs_sum_to_one(self):
        """Posterior probabilities for each subject must sum to 1."""
        rng = np.random.default_rng(12)
        n_per = 8
        times = np.linspace(0, 5, n_per)
        ids = np.concatenate([np.zeros(n_per), np.ones(n_per)])
        t_all = np.tile(times, 2)
        dv = np.concatenate([
            3.0 * times + rng.normal(0, 0.5, n_per),
            7.0 * times + rng.normal(0, 0.5, n_per),
        ])
        data = {"id": ids, "time": t_all, "dv": dv}
        params_per_class = [{"slope": 3.0}, {"slope": 7.0}]

        result = classify_subjects(
            data, _linear_model, params_per_class, (0.5, 0.5), 0.5,
        )
        for subj_id, (_, posteriors) in result.items():
            np.testing.assert_allclose(sum(posteriors), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# estimate_mixture (EM algorithm)
# ---------------------------------------------------------------------------

class TestEstimateMixture:
    def test_returns_mixture_result(self):
        """estimate_mixture should return a MixtureResult."""
        rng = np.random.default_rng(0)
        n_subj = 6
        n_obs = 8
        times = np.linspace(0, 5, n_obs)
        ids = np.repeat(np.arange(n_subj, dtype=float), n_obs)
        t_all = np.tile(times, n_subj)
        slopes = np.array([2.0, 2.0, 2.0, 8.0, 8.0, 8.0])
        dv = np.concatenate([
            slopes[i] * times + rng.normal(0, 0.3, n_obs)
            for i in range(n_subj)
        ])
        data = {"id": ids, "time": t_all, "dv": dv}
        ini = {"slope": 5.0}

        result = estimate_mixture(
            _linear_model, data, ini, n_classes=2,
            control={"maxiter": 30, "tol": 1e-4, "sigma": 0.5},
        )
        assert isinstance(result, MixtureResult)
        assert len(result.params_per_class) == 2
        assert len(result.mixing_probs) == 2
        assert isinstance(result.classifications, dict)
        assert isinstance(result.objective, float)
        assert isinstance(result.n_iterations, int)
        assert isinstance(result.converged, bool)

    def test_recovers_two_classes(self):
        """EM should separate two well-separated sub-populations."""
        rng = np.random.default_rng(42)
        n_subj = 20
        n_obs = 10
        times = np.linspace(0, 5, n_obs)
        true_slopes = np.array(
            [2.0] * (n_subj // 2) + [10.0] * (n_subj // 2)
        )
        ids = np.repeat(np.arange(n_subj, dtype=float), n_obs)
        t_all = np.tile(times, n_subj)
        dv = np.concatenate([
            true_slopes[i] * times + rng.normal(0, 0.2, n_obs)
            for i in range(n_subj)
        ])
        data = {"id": ids, "time": t_all, "dv": dv}
        ini = {"slope": 5.0}

        result = estimate_mixture(
            _linear_model, data, ini, n_classes=2,
            control={"maxiter": 100, "tol": 1e-6, "sigma": 0.5},
        )

        # The two recovered slopes should be close to 2 and 10 (any order)
        slopes = sorted(
            p["slope"] for p in result.params_per_class
        )
        assert abs(slopes[0] - 2.0) < 1.0
        assert abs(slopes[1] - 10.0) < 1.0

    def test_mixing_probs_sum_to_one_in_output(self):
        """Mixing probabilities in result must sum to 1."""
        rng = np.random.default_rng(5)
        n_subj = 10
        n_obs = 6
        times = np.linspace(0, 5, n_obs)
        ids = np.repeat(np.arange(n_subj, dtype=float), n_obs)
        t_all = np.tile(times, n_subj)
        slopes = np.array([1.0] * 5 + [6.0] * 5)
        dv = np.concatenate([
            slopes[i] * times + rng.normal(0, 0.3, n_obs)
            for i in range(n_subj)
        ])
        data = {"id": ids, "time": t_all, "dv": dv}
        ini = {"slope": 3.0}

        result = estimate_mixture(
            _linear_model, data, ini, n_classes=2,
            control={"maxiter": 50, "tol": 1e-4, "sigma": 0.5},
        )
        np.testing.assert_allclose(sum(result.mixing_probs), 1.0, atol=1e-10)
