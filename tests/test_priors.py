"""Tests for prior specification for Bayesian/MAP estimation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from nlmixr2.priors import (
    HalfNormalPrior,
    InverseWishartPrior,
    LogNormalPrior,
    NormalPrior,
    Prior,
    PriorSpec,
    UniformPrior,
    compute_prior_contribution,
    map_objective,
)


# ---------------------------------------------------------------------------
# NormalPrior
# ---------------------------------------------------------------------------


class TestNormalPrior:
    def test_log_density_at_mean_is_maximum(self):
        """The log density of a normal distribution is maximised at the mean."""
        prior = NormalPrior(mean=5.0, sd=2.0)
        at_mean = prior.log_density(5.0)
        away = prior.log_density(7.0)
        assert at_mean > away

    def test_symmetry(self):
        """log N(mu-d, sd) == log N(mu+d, sd) for any d."""
        prior = NormalPrior(mean=0.0, sd=1.0)
        assert math.isclose(prior.log_density(-2.0), prior.log_density(2.0), rel_tol=1e-12)

    def test_known_value(self):
        """Standard normal at 0: log(1/sqrt(2pi)) = -0.5*log(2pi)."""
        prior = NormalPrior(mean=0.0, sd=1.0)
        expected = -0.5 * math.log(2 * math.pi)
        assert math.isclose(prior.log_density(0.0), expected, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# LogNormalPrior
# ---------------------------------------------------------------------------


class TestLogNormalPrior:
    def test_positive_support(self):
        """log-normal has -inf log density for non-positive values."""
        prior = LogNormalPrior(meanlog=0.0, sdlog=1.0)
        assert prior.log_density(0.0) == -math.inf
        assert prior.log_density(-1.0) == -math.inf

    def test_positive_value_finite(self):
        prior = LogNormalPrior(meanlog=0.0, sdlog=1.0)
        assert math.isfinite(prior.log_density(1.0))

    def test_mode_higher_than_tail(self):
        """The mode of LogNormal(0,1) is exp(-1); density there should exceed a tail value."""
        prior = LogNormalPrior(meanlog=0.0, sdlog=1.0)
        mode = math.exp(-1.0)
        assert prior.log_density(mode) > prior.log_density(10.0)


# ---------------------------------------------------------------------------
# UniformPrior
# ---------------------------------------------------------------------------


class TestUniformPrior:
    def test_within_bounds(self):
        prior = UniformPrior(lower=0.0, upper=10.0)
        expected = -math.log(10.0)
        assert math.isclose(prior.log_density(5.0), expected, rel_tol=1e-12)

    def test_outside_bounds_below(self):
        prior = UniformPrior(lower=0.0, upper=10.0)
        assert prior.log_density(-1.0) == -math.inf

    def test_outside_bounds_above(self):
        prior = UniformPrior(lower=0.0, upper=10.0)
        assert prior.log_density(11.0) == -math.inf

    def test_at_boundaries(self):
        """Boundary values should be within the support [lower, upper]."""
        prior = UniformPrior(lower=0.0, upper=10.0)
        assert math.isfinite(prior.log_density(0.0))
        assert math.isfinite(prior.log_density(10.0))


# ---------------------------------------------------------------------------
# HalfNormalPrior
# ---------------------------------------------------------------------------


class TestHalfNormalPrior:
    def test_positive_only(self):
        prior = HalfNormalPrior(sd=1.0)
        assert prior.log_density(-0.1) == -math.inf

    def test_zero_is_mode(self):
        """Half-normal mode is at 0; density there should exceed any positive value."""
        prior = HalfNormalPrior(sd=1.0)
        assert prior.log_density(0.0) > prior.log_density(2.0)

    def test_finite_positive(self):
        prior = HalfNormalPrior(sd=2.0)
        assert math.isfinite(prior.log_density(1.5))


# ---------------------------------------------------------------------------
# InverseWishartPrior
# ---------------------------------------------------------------------------


class TestInverseWishartPrior:
    def test_creation(self):
        scale = np.eye(2)
        prior = InverseWishartPrior(df=5, scale=scale)
        assert prior.df == 5
        assert np.array_equal(prior.scale, scale)

    def test_positive_definite_matrix_finite(self):
        scale = np.eye(2)
        prior = InverseWishartPrior(df=5, scale=scale)
        value = np.array([[2.0, 0.1], [0.1, 3.0]])
        ld = prior.log_density(value)
        assert math.isfinite(ld)

    def test_non_positive_definite_returns_neg_inf(self):
        scale = np.eye(2)
        prior = InverseWishartPrior(df=5, scale=scale)
        # singular matrix
        value = np.array([[1.0, 0.0], [0.0, 0.0]])
        ld = prior.log_density(value)
        assert ld == -math.inf


# ---------------------------------------------------------------------------
# PriorSpec
# ---------------------------------------------------------------------------


class TestPriorSpec:
    def test_creation(self):
        spec = PriorSpec(priors={"ka": NormalPrior(mean=1.0, sd=0.5)})
        assert "ka" in spec.priors
        assert isinstance(spec.priors["ka"], NormalPrior)

    def test_empty(self):
        spec = PriorSpec(priors={})
        assert len(spec.priors) == 0


# ---------------------------------------------------------------------------
# compute_prior_contribution
# ---------------------------------------------------------------------------


class TestComputePriorContribution:
    def test_sums_all_priors(self):
        spec = PriorSpec(
            priors={
                "A": NormalPrior(mean=10.0, sd=2.0),
                "ke": NormalPrior(mean=0.5, sd=0.1),
            }
        )
        params = {"A": 10.0, "ke": 0.5}
        total = compute_prior_contribution(spec, params)
        expected = (
            NormalPrior(mean=10.0, sd=2.0).log_density(10.0)
            + NormalPrior(mean=0.5, sd=0.1).log_density(0.5)
        )
        assert math.isclose(total, expected, rel_tol=1e-12)

    def test_missing_param_ignored(self):
        """Parameters not in the prior spec should not affect the contribution."""
        spec = PriorSpec(priors={"A": NormalPrior(mean=10.0, sd=2.0)})
        params = {"A": 10.0, "ke": 0.5}
        total = compute_prior_contribution(spec, params)
        expected = NormalPrior(mean=10.0, sd=2.0).log_density(10.0)
        assert math.isclose(total, expected, rel_tol=1e-12)

    def test_empty_spec_returns_zero(self):
        spec = PriorSpec(priors={})
        total = compute_prior_contribution(spec, {"A": 1.0})
        assert total == 0.0


# ---------------------------------------------------------------------------
# map_objective
# ---------------------------------------------------------------------------


class TestMapObjective:
    def test_adds_prior_to_base(self):
        """MAP objective = base_obj - 2 * log_prior."""
        spec = PriorSpec(priors={"A": NormalPrior(mean=10.0, sd=2.0)})
        params = {"A": 10.0}
        base = 100.0
        result = map_objective(base, spec, params)
        log_prior = NormalPrior(mean=10.0, sd=2.0).log_density(10.0)
        expected = 100.0 - 2.0 * log_prior
        assert math.isclose(result, expected, rel_tol=1e-12)

    def test_no_priors_returns_base(self):
        spec = PriorSpec(priors={})
        result = map_objective(50.0, spec, {"A": 1.0})
        assert result == 50.0

    def test_prior_penalises_away_from_mode(self):
        """Moving a parameter away from the prior mean should increase the MAP objective."""
        spec = PriorSpec(priors={"A": NormalPrior(mean=10.0, sd=2.0)})
        at_mean = map_objective(100.0, spec, {"A": 10.0})
        away = map_objective(100.0, spec, {"A": 20.0})
        assert away > at_mean


# ---------------------------------------------------------------------------
# Prior base class
# ---------------------------------------------------------------------------


class TestPriorBaseClass:
    def test_is_abstract(self):
        """Prior.log_density should raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            Prior().log_density(1.0)
