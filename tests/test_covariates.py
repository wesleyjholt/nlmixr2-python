"""Tests for covariate support and mu-referencing — TDD style."""

from __future__ import annotations

import math

import pytest
import jax.numpy as jnp

from nlmixr2.api import ini, model, IniBlock, ModelBlock
from nlmixr2.parser import parse_model, ParsedModel
from nlmixr2.covariates import (
    CovariateModel,
    extract_covariates,
    center_covariates,
    add_covariate_to_model,
    detect_mu_referencing,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ini_block(**kwargs: float) -> IniBlock:
    return ini(kwargs)


def _model_block(*stmts: str) -> ModelBlock:
    return model(list(stmts))


# ---------------------------------------------------------------------------
# extract_covariates
# ---------------------------------------------------------------------------

class TestExtractCovariates:
    def test_finds_matching_columns(self):
        """Covariate names that appear in model statements AND data columns are found."""
        model_b = _model_block(
            "cl = tvcl * (WT / 70)",
            "cp = 100 / cl",
            "cp ~ add(add.sd)",
        )
        data_columns = ["id", "time", "dv", "WT", "AGE"]
        result = extract_covariates(model_b, data_columns)
        assert "WT" in result

    def test_finds_multiple_covariates(self):
        model_b = _model_block(
            "cl = tvcl * (WT / 70) * (AGE / 40)",
            "cp = 100 / cl",
            "cp ~ add(add.sd)",
        )
        data_columns = ["id", "time", "dv", "WT", "AGE"]
        result = extract_covariates(model_b, data_columns)
        assert set(result) == {"WT", "AGE"}

    def test_ignores_non_data_names(self):
        """Names in the model that are not data columns should not be returned."""
        model_b = _model_block(
            "cl = tvcl * exp(eta.cl)",
            "cp = 100 / cl",
            "cp ~ add(add.sd)",
        )
        data_columns = ["id", "time", "dv"]
        result = extract_covariates(model_b, data_columns)
        assert result == []

    def test_ignores_standard_columns(self):
        """Standard columns like id, time, dv should not be returned as covariates."""
        model_b = _model_block(
            "cp = 100 / V",
            "cp ~ add(add.sd)",
        )
        data_columns = ["id", "time", "dv"]
        result = extract_covariates(model_b, data_columns)
        assert result == []

    def test_ignores_math_function_names(self):
        """Math function names like exp, log should not be returned."""
        model_b = _model_block(
            "cl = exp(tvcl)",
            "cp = log(cl)",
            "cp ~ add(add.sd)",
        )
        data_columns = ["id", "time", "dv", "exp", "log"]
        result = extract_covariates(model_b, data_columns)
        assert result == []


# ---------------------------------------------------------------------------
# CovariateModel dataclass
# ---------------------------------------------------------------------------

class TestCovariateModel:
    def test_creation_with_defaults(self):
        cm = CovariateModel(
            covariates={"WT": "continuous", "SEX": "categorical"},
            centering={"WT": 70.0},
            transformations={"WT": "none", "SEX": "none"},
        )
        assert cm.covariates["WT"] == "continuous"
        assert cm.covariates["SEX"] == "categorical"
        assert cm.centering["WT"] == 70.0
        assert cm.transformations["WT"] == "none"

    def test_creation_with_log_transform(self):
        cm = CovariateModel(
            covariates={"WT": "continuous"},
            centering={"WT": 70.0},
            transformations={"WT": "log"},
        )
        assert cm.transformations["WT"] == "log"

    def test_creation_with_normalized_transform(self):
        cm = CovariateModel(
            covariates={"WT": "continuous"},
            centering={"WT": 70.0},
            transformations={"WT": "normalized"},
        )
        assert cm.transformations["WT"] == "normalized"


# ---------------------------------------------------------------------------
# center_covariates
# ---------------------------------------------------------------------------

class TestCenterCovariates:
    def test_centers_with_known_data(self):
        data = {
            "id": jnp.array([1, 2, 3]),
            "time": jnp.array([0.0, 1.0, 2.0]),
            "dv": jnp.array([10.0, 20.0, 30.0]),
            "WT": jnp.array([60.0, 70.0, 80.0]),
        }
        cm = CovariateModel(
            covariates={"WT": "continuous"},
            centering={"WT": 70.0},
            transformations={"WT": "none"},
        )
        result = center_covariates(data, cm)
        # WT_centered = WT - 70 => [-10, 0, 10]
        assert "WT_centered" in result
        expected = jnp.array([-10.0, 0.0, 10.0])
        assert jnp.allclose(result["WT_centered"], expected)

    def test_original_data_preserved(self):
        data = {
            "id": jnp.array([1, 2]),
            "WT": jnp.array([60.0, 80.0]),
        }
        cm = CovariateModel(
            covariates={"WT": "continuous"},
            centering={"WT": 70.0},
            transformations={"WT": "none"},
        )
        result = center_covariates(data, cm)
        # Original columns are preserved
        assert "id" in result
        assert "WT" in result
        assert "WT_centered" in result

    def test_log_transform_centering(self):
        data = {
            "WT": jnp.array([60.0, 70.0, 80.0]),
        }
        cm = CovariateModel(
            covariates={"WT": "continuous"},
            centering={"WT": 70.0},
            transformations={"WT": "log"},
        )
        result = center_covariates(data, cm)
        # log transform: log(WT / center) = log(WT) - log(70)
        expected = jnp.log(jnp.array([60.0, 70.0, 80.0])) - jnp.log(70.0)
        assert jnp.allclose(result["WT_centered"], expected)

    def test_normalized_transform_centering(self):
        data = {
            "WT": jnp.array([60.0, 70.0, 80.0]),
        }
        cm = CovariateModel(
            covariates={"WT": "continuous"},
            centering={"WT": 70.0},
            transformations={"WT": "normalized"},
        )
        result = center_covariates(data, cm)
        # normalized: (WT - center) / center
        expected = (jnp.array([60.0, 70.0, 80.0]) - 70.0) / 70.0
        assert jnp.allclose(result["WT_centered"], expected)

    def test_centering_skips_covariates_without_centering_value(self):
        """If a covariate has no centering value, it is not centered."""
        data = {
            "SEX": jnp.array([0, 1, 0]),
        }
        cm = CovariateModel(
            covariates={"SEX": "categorical"},
            centering={},
            transformations={"SEX": "none"},
        )
        result = center_covariates(data, cm)
        assert "SEX_centered" not in result


# ---------------------------------------------------------------------------
# add_covariate_to_model — linear effect
# ---------------------------------------------------------------------------

class TestAddCovariateLinear:
    def test_linear_effect_applied(self):
        ini_b = _ini_block(tvcl=5.0, **{"add.sd": 0.5})
        model_b = _model_block(
            "cl = tvcl",
            "cp = 100 / cl",
            "cp ~ add(add.sd)",
        )
        parsed = parse_model(model_b, ini_b)
        updated = add_covariate_to_model(
            parsed, covariate_name="WT", parameter_name="cl", effect="linear"
        )
        # The updated model should have a new parameter for the covariate effect
        assert "theta_WT_cl" in updated.parameter_names
        # Test that the output_func works with WT in params
        params = {"tvcl": 5.0, "add.sd": 0.5, "theta_WT_cl": 0.1, "WT_centered": 10.0}
        pred = updated.output_func(None, 0.0, params)
        # cl = tvcl + theta_WT_cl * WT_centered = 5.0 + 0.1 * 10.0 = 6.0
        # cp = 100 / 6.0
        assert float(pred) == pytest.approx(100.0 / 6.0)

    def test_linear_effect_at_center(self):
        """When covariate is at center (centered = 0), result is unchanged."""
        ini_b = _ini_block(tvcl=5.0, **{"add.sd": 0.5})
        model_b = _model_block(
            "cl = tvcl",
            "cp = 100 / cl",
            "cp ~ add(add.sd)",
        )
        parsed = parse_model(model_b, ini_b)
        updated = add_covariate_to_model(
            parsed, covariate_name="WT", parameter_name="cl", effect="linear"
        )
        params = {"tvcl": 5.0, "add.sd": 0.5, "theta_WT_cl": 0.1, "WT_centered": 0.0}
        pred = updated.output_func(None, 0.0, params)
        # cl = tvcl + 0.1 * 0.0 = 5.0, cp = 100 / 5.0 = 20.0
        assert float(pred) == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# add_covariate_to_model — power effect
# ---------------------------------------------------------------------------

class TestAddCovariatePower:
    def test_power_effect_applied(self):
        ini_b = _ini_block(tvcl=5.0, **{"add.sd": 0.5})
        model_b = _model_block(
            "cl = tvcl",
            "cp = 100 / cl",
            "cp ~ add(add.sd)",
        )
        parsed = parse_model(model_b, ini_b)
        updated = add_covariate_to_model(
            parsed, covariate_name="WT", parameter_name="cl", effect="power"
        )
        assert "theta_WT_cl" in updated.parameter_names
        # Power model: cl = tvcl * (WT_ratio) ** theta_WT_cl
        # WT_ratio = WT / center (passed as WT_ratio)
        params = {"tvcl": 5.0, "add.sd": 0.5, "theta_WT_cl": 0.75, "WT_ratio": 2.0}
        pred = updated.output_func(None, 0.0, params)
        # cl = 5.0 * 2.0 ** 0.75
        expected_cl = 5.0 * (2.0 ** 0.75)
        assert float(pred) == pytest.approx(100.0 / expected_cl)

    def test_power_effect_at_reference(self):
        """When ratio = 1 (at reference), cl = tvcl."""
        ini_b = _ini_block(tvcl=5.0, **{"add.sd": 0.5})
        model_b = _model_block(
            "cl = tvcl",
            "cp = 100 / cl",
            "cp ~ add(add.sd)",
        )
        parsed = parse_model(model_b, ini_b)
        updated = add_covariate_to_model(
            parsed, covariate_name="WT", parameter_name="cl", effect="power"
        )
        params = {"tvcl": 5.0, "add.sd": 0.5, "theta_WT_cl": 0.75, "WT_ratio": 1.0}
        pred = updated.output_func(None, 0.0, params)
        assert float(pred) == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# add_covariate_to_model — exponential effect
# ---------------------------------------------------------------------------

class TestAddCovariateExponential:
    def test_exponential_effect_applied(self):
        ini_b = _ini_block(tvcl=5.0, **{"add.sd": 0.5})
        model_b = _model_block(
            "cl = tvcl",
            "cp = 100 / cl",
            "cp ~ add(add.sd)",
        )
        parsed = parse_model(model_b, ini_b)
        updated = add_covariate_to_model(
            parsed, covariate_name="WT", parameter_name="cl", effect="exponential"
        )
        assert "theta_WT_cl" in updated.parameter_names
        # Exponential: cl = tvcl * exp(theta_WT_cl * WT_centered)
        params = {"tvcl": 5.0, "add.sd": 0.5, "theta_WT_cl": 0.02, "WT_centered": 10.0}
        pred = updated.output_func(None, 0.0, params)
        expected_cl = 5.0 * math.exp(0.02 * 10.0)
        assert float(pred) == pytest.approx(100.0 / expected_cl)

    def test_exponential_effect_at_center(self):
        """When centered covariate = 0, cl = tvcl."""
        ini_b = _ini_block(tvcl=5.0, **{"add.sd": 0.5})
        model_b = _model_block(
            "cl = tvcl",
            "cp = 100 / cl",
            "cp ~ add(add.sd)",
        )
        parsed = parse_model(model_b, ini_b)
        updated = add_covariate_to_model(
            parsed, covariate_name="WT", parameter_name="cl", effect="exponential"
        )
        params = {"tvcl": 5.0, "add.sd": 0.5, "theta_WT_cl": 0.02, "WT_centered": 0.0}
        pred = updated.output_func(None, 0.0, params)
        assert float(pred) == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# detect_mu_referencing
# ---------------------------------------------------------------------------

class TestDetectMuReferencing:
    def test_detects_mu_referenced_parameters(self):
        """mu_1 = log(tvcl) + eta.cl is a mu-referenced pattern."""
        statements = [
            "mu_1 = log(tvcl) + eta.cl",
            "cl = exp(mu_1)",
            "cp = 100 / cl",
            "cp ~ add(add.sd)",
        ]
        result = detect_mu_referencing(statements)
        assert "mu_1" in result

    def test_detects_multiple_mu_references(self):
        statements = [
            "mu_1 = log(tvcl) + eta.cl",
            "mu_2 = log(tvv) + eta.v",
            "cl = exp(mu_1)",
            "v = exp(mu_2)",
            "cp = 100 / cl",
            "cp ~ add(add.sd)",
        ]
        result = detect_mu_referencing(statements)
        assert "mu_1" in result
        assert "mu_2" in result

    def test_non_mu_model_returns_empty(self):
        """A model without mu-referencing should return empty list."""
        statements = [
            "cl = tvcl * exp(eta.cl)",
            "cp = 100 / cl",
            "cp ~ add(add.sd)",
        ]
        result = detect_mu_referencing(statements)
        assert result == []

    def test_mu_without_eta_not_detected(self):
        """mu_X without an eta term is not a proper mu-reference."""
        statements = [
            "mu_1 = log(tvcl)",
            "cl = exp(mu_1)",
            "cp = 100 / cl",
            "cp ~ add(add.sd)",
        ]
        result = detect_mu_referencing(statements)
        assert result == []
