"""Tests for stepwise covariate selection (covarSearchAuto equivalent)."""

from __future__ import annotations

import pytest

from nlmixr2.api import NLMIXRFit, NLMIXRModel, IniBlock, InitValue, ModelBlock
from nlmixr2.covar_search import (
    StepResult,
    forward_addition,
    backward_elimination,
    stepwise_covariate_search,
)


def _make_fit(
    objective: float = 100.0,
    n_params: int = 3,
    n_obs: int = 50,
) -> NLMIXRFit:
    """Helper to create a minimal NLMIXRFit for testing."""
    values = {f"p{i}": InitValue(estimate=float(i)) for i in range(n_params)}
    ini_block = IniBlock(values=values)
    model_block = ModelBlock(statements=("y = p0",))
    nlmixr_model = NLMIXRModel(ini=ini_block, model=model_block)
    return NLMIXRFit(
        estimator="mock",
        n_observations=n_obs,
        columns=("time", "dv"),
        parameter_count=n_params,
        objective=objective,
        model=nlmixr_model,
        control={},
        table={},
    )


def _make_data(n: int = 50) -> dict:
    return {
        "id": list(range(1, n + 1)),
        "time": [float(i) for i in range(n)],
        "dv": [float(i) * 0.5 for i in range(n)],
        "WT": [70.0 + i * 0.1 for i in range(n)],
        "AGE": [30.0 + i * 0.2 for i in range(n)],
        "SEX": [i % 2 for i in range(n)],
    }


class TestStepResult:
    def test_fields_exist(self):
        sr = StepResult(
            covariate="WT",
            parameter="CL",
            effect="linear",
            direction="forward",
            delta_obj=-5.0,
            p_value=0.02,
            selected=True,
        )
        assert sr.covariate == "WT"
        assert sr.parameter == "CL"
        assert sr.effect == "linear"
        assert sr.direction == "forward"
        assert sr.delta_obj == -5.0
        assert sr.p_value == 0.02
        assert sr.selected is True

    def test_direction_values(self):
        sr_fwd = StepResult("WT", "CL", "linear", "forward", -5.0, 0.02, True)
        sr_bwd = StepResult("WT", "CL", "linear", "backward", 3.0, 0.1, False)
        assert sr_fwd.direction == "forward"
        assert sr_bwd.direction == "backward"


class TestForwardAddition:
    def test_returns_list_of_step_results(self):
        base_fit = _make_fit(objective=100.0)
        data = _make_data()
        results = forward_addition(
            base_fit, data, covariates=["WT", "AGE"], parameters=["p0", "p1"]
        )
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, StepResult)

    def test_results_have_forward_direction(self):
        base_fit = _make_fit(objective=100.0)
        data = _make_data()
        results = forward_addition(
            base_fit, data, covariates=["WT"], parameters=["p0"]
        )
        for r in results:
            assert r.direction == "forward"

    def test_results_sorted_by_delta_obj(self):
        base_fit = _make_fit(objective=100.0)
        data = _make_data()
        results = forward_addition(
            base_fit, data, covariates=["WT", "AGE"], parameters=["p0", "p1"]
        )
        if len(results) > 1:
            # Should be sorted ascending (most negative first = largest OFV drop)
            for i in range(len(results) - 1):
                assert results[i].delta_obj <= results[i + 1].delta_obj

    def test_no_significant_covariates(self):
        """When no covariate produces a significant drop, none should be selected."""
        # Use a fit_factory that returns fits with very similar objectives
        base_fit = _make_fit(objective=100.0)
        data = _make_data()

        def no_improvement_factory(fit, data, cov, par, eff):
            """Return a fit with barely any improvement."""
            return _make_fit(objective=99.9, n_params=fit.parameter_count + 1)

        results = forward_addition(
            base_fit, data,
            covariates=["WT", "AGE"], parameters=["p0"],
            alpha=0.05,
            fit_factory=no_improvement_factory,
        )
        selected = [r for r in results if r.selected]
        assert len(selected) == 0

    def test_one_significant_covariate(self):
        """When one covariate produces a significant drop, it should be selected."""
        base_fit = _make_fit(objective=100.0)
        data = _make_data()

        call_count = {"n": 0}

        def sig_factory(fit, data, cov, par, eff):
            """First covariate (WT) gives big improvement, AGE does not."""
            call_count["n"] += 1
            if cov == "WT":
                return _make_fit(objective=88.0, n_params=fit.parameter_count + 1)
            return _make_fit(objective=99.5, n_params=fit.parameter_count + 1)

        results = forward_addition(
            base_fit, data,
            covariates=["WT", "AGE"], parameters=["p0"],
            alpha=0.05,
            fit_factory=sig_factory,
        )
        selected = [r for r in results if r.selected]
        assert len(selected) == 1
        assert selected[0].covariate == "WT"

    def test_multiple_effects(self):
        base_fit = _make_fit(objective=100.0)
        data = _make_data()
        results = forward_addition(
            base_fit, data,
            covariates=["WT"], parameters=["p0"],
            effects=("linear", "power"),
        )
        effects_seen = {r.effect for r in results}
        assert "linear" in effects_seen
        assert "power" in effects_seen


class TestBackwardElimination:
    def test_returns_list_of_step_results(self):
        full_fit = _make_fit(objective=85.0, n_params=5)
        data = _make_data()
        results = backward_elimination(
            full_fit, data, covariates=["WT", "AGE"], parameters=["p0", "p1"]
        )
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, StepResult)

    def test_results_have_backward_direction(self):
        full_fit = _make_fit(objective=85.0, n_params=5)
        data = _make_data()
        results = backward_elimination(
            full_fit, data, covariates=["WT"], parameters=["p0"]
        )
        for r in results:
            assert r.direction == "backward"

    def test_removal_of_nonsignificant_covariate(self):
        """When removing a covariate barely increases OFV, it should be selected for removal."""
        full_fit = _make_fit(objective=85.0, n_params=5)
        data = _make_data()

        def small_increase_factory(fit, data, cov, par, eff):
            """Removing any covariate barely increases objective."""
            return _make_fit(objective=85.5, n_params=fit.parameter_count - 1)

        results = backward_elimination(
            full_fit, data,
            covariates=["WT"], parameters=["p0"],
            alpha=0.01,
            fit_factory=small_increase_factory,
        )
        selected = [r for r in results if r.selected]
        assert len(selected) == 1


class TestStepwiseCovariateSearch:
    def test_returns_list_of_step_results(self):
        base_fit = _make_fit(objective=100.0)
        data = _make_data()
        results = stepwise_covariate_search(
            base_fit, data, covariates=["WT"], parameters=["p0"]
        )
        assert isinstance(results, list)

    def test_terminates(self):
        """Stepwise search should terminate even without significant covariates."""
        base_fit = _make_fit(objective=100.0)
        data = _make_data()

        def no_change_factory(fit, data, cov, par, eff):
            return _make_fit(objective=fit.objective - 0.01,
                             n_params=fit.parameter_count + 1)

        results = stepwise_covariate_search(
            base_fit, data,
            covariates=["WT", "AGE"], parameters=["p0"],
            fit_factory=no_change_factory,
        )
        assert isinstance(results, list)

    def test_max_steps_limit(self):
        """Stepwise search should respect max_steps."""
        base_fit = _make_fit(objective=100.0)
        data = _make_data()
        step_count = {"n": 0}

        def always_sig_factory(fit, data, cov, par, eff):
            step_count["n"] += 1
            # Always return a big improvement to force many steps
            return _make_fit(objective=fit.objective - 20.0,
                             n_params=fit.parameter_count + 1)

        results = stepwise_covariate_search(
            base_fit, data,
            covariates=["WT", "AGE", "SEX"], parameters=["p0", "p1"],
            max_steps=3,
            fit_factory=always_sig_factory,
        )
        # We should have at most 3 selected steps
        selected = [r for r in results if r.selected]
        assert len(selected) <= 3

    def test_empty_covariates(self):
        """With no covariates, should return empty list."""
        base_fit = _make_fit(objective=100.0)
        data = _make_data()
        results = stepwise_covariate_search(
            base_fit, data, covariates=[], parameters=["p0"]
        )
        assert results == []

    def test_empty_parameters(self):
        """With no parameters, should return empty list."""
        base_fit = _make_fit(objective=100.0)
        data = _make_data()
        results = stepwise_covariate_search(
            base_fit, data, covariates=["WT"], parameters=[]
        )
        assert results == []
