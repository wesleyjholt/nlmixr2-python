"""Tests for nlmixr2 model parser — TDD style."""

from __future__ import annotations

import pytest
import jax.numpy as jnp

from nlmixr2.api import ini, model, IniBlock, ModelBlock, InitValue
from nlmixr2.parser import parse_model, ParsedModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ini_block(**kwargs: float) -> IniBlock:
    """Shortcut to build an IniBlock from keyword arguments."""
    return ini(kwargs)


def _model_block(*stmts: str) -> ModelBlock:
    """Shortcut to build a ModelBlock from statement strings."""
    return model(list(stmts))


# ---------------------------------------------------------------------------
# ParsedModel structure
# ---------------------------------------------------------------------------

class TestParsedModelStructure:
    """Verify that parse_model returns a well-formed ParsedModel."""

    def test_returns_parsed_model_instance(self):
        ini_b = _ini_block(V=10.0, ke=0.1)
        model_b = _model_block("cp = 100 / V", "cp ~ add(add.sd)")
        result = parse_model(model_b, ini_b)
        assert isinstance(result, ParsedModel)

    def test_has_required_attributes(self):
        ini_b = _ini_block(V=10.0)
        model_b = _model_block("cp = 100 / V", "cp ~ add(add.sd)")
        result = parse_model(model_b, ini_b)
        assert hasattr(result, "parameter_names")
        assert hasattr(result, "state_names")
        assert hasattr(result, "ode_func")
        assert hasattr(result, "output_func")


# ---------------------------------------------------------------------------
# Parameter name extraction
# ---------------------------------------------------------------------------

class TestParameterNames:
    def test_single_param(self):
        ini_b = _ini_block(ke=0.1)
        model_b = _model_block("cp = 100 * ke", "cp ~ add(add.sd)")
        result = parse_model(model_b, ini_b)
        assert result.parameter_names == ["ke"]

    def test_multiple_params(self):
        ini_b = _ini_block(V=10.0, ke=0.1, ka=1.5)
        model_b = _model_block("cp = 100 / V", "cp ~ add(add.sd)")
        result = parse_model(model_b, ini_b)
        assert set(result.parameter_names) == {"V", "ke", "ka"}

    def test_dotted_param_names(self):
        ini_b = _ini_block(**{"add.sd": 0.5})
        model_b = _model_block("cp = 1.0", "cp ~ add(add.sd)")
        result = parse_model(model_b, ini_b)
        assert "add.sd" in result.parameter_names


# ---------------------------------------------------------------------------
# ODE state variable detection
# ---------------------------------------------------------------------------

class TestStateDetection:
    def test_no_odes_gives_empty_states(self):
        ini_b = _ini_block(V=10.0)
        model_b = _model_block("cp = 100 / V", "cp ~ add(add.sd)")
        result = parse_model(model_b, ini_b)
        assert result.state_names == []

    def test_single_compartment_ode(self):
        ini_b = _ini_block(ke=0.1)
        model_b = _model_block(
            "d/dt(central) = -ke * central",
            "cp = central / 10",
            "cp ~ add(add.sd)",
        )
        result = parse_model(model_b, ini_b)
        assert result.state_names == ["central"]

    def test_two_compartment_ode(self):
        ini_b = _ini_block(ke=0.1, k12=0.3, k21=0.2)
        model_b = _model_block(
            "d/dt(central) = -ke * central - k12 * central + k21 * peripheral",
            "d/dt(peripheral) = k12 * central - k21 * peripheral",
            "cp = central / 10",
            "cp ~ add(add.sd)",
        )
        result = parse_model(model_b, ini_b)
        assert result.state_names == ["central", "peripheral"]

    def test_ode_func_is_none_when_no_odes(self):
        ini_b = _ini_block(V=10.0)
        model_b = _model_block("cp = 100 / V", "cp ~ add(add.sd)")
        result = parse_model(model_b, ini_b)
        assert result.ode_func is None


# ---------------------------------------------------------------------------
# Residual error model detection
# ---------------------------------------------------------------------------

class TestResidualErrorDetection:
    def test_additive_error(self):
        ini_b = _ini_block(V=10.0, **{"add.sd": 0.5})
        model_b = _model_block("cp = 100 / V", "cp ~ add(add.sd)")
        result = parse_model(model_b, ini_b)
        assert result.error_model == "additive"

    def test_proportional_error(self):
        ini_b = _ini_block(V=10.0, **{"prop.sd": 0.1})
        model_b = _model_block("cp = 100 / V", "cp ~ prop(prop.sd)")
        result = parse_model(model_b, ini_b)
        assert result.error_model == "proportional"

    def test_combined_error(self):
        ini_b = _ini_block(V=10.0, **{"add.sd": 0.5, "prop.sd": 0.1})
        model_b = _model_block("cp = 100 / V", "cp ~ add(add.sd) + prop(prop.sd)")
        result = parse_model(model_b, ini_b)
        assert result.error_model == "combined"

    def test_dv_name_captured(self):
        ini_b = _ini_block(V=10.0, **{"add.sd": 0.5})
        model_b = _model_block("conc = 100 / V", "conc ~ add(add.sd)")
        result = parse_model(model_b, ini_b)
        assert result.dv_name == "conc"


# ---------------------------------------------------------------------------
# Simple algebraic model — output_func
# ---------------------------------------------------------------------------

class TestAlgebraicOutputFunc:
    def test_simple_division(self):
        ini_b = _ini_block(V=10.0)
        model_b = _model_block("cp = 100 / V", "cp ~ add(add.sd)")
        result = parse_model(model_b, ini_b)
        params = {"V": 10.0}
        pred = result.output_func(None, 0.0, params)
        assert float(pred) == pytest.approx(10.0)

    def test_multi_param_expression(self):
        ini_b = _ini_block(CL=5.0, V=50.0)
        model_b = _model_block("ke = CL / V", "cp = 1000 * ke", "cp ~ add(add.sd)")
        result = parse_model(model_b, ini_b)
        params = {"CL": 5.0, "V": 50.0}
        pred = result.output_func(None, 0.0, params)
        # ke = 5/50 = 0.1, cp = 1000 * 0.1 = 100
        assert float(pred) == pytest.approx(100.0)

    def test_exp_in_expression(self):
        """Ensure exp() calls work (mapped to jnp.exp)."""
        ini_b = _ini_block(lV=2.0)
        model_b = _model_block("V = exp(lV)", "cp = 100 / V", "cp ~ add(add.sd)")
        result = parse_model(model_b, ini_b)
        params = {"lV": 2.0}
        pred = result.output_func(None, 0.0, params)
        import math
        assert float(pred) == pytest.approx(100.0 / math.exp(2.0))


# ---------------------------------------------------------------------------
# ODE func correctness
# ---------------------------------------------------------------------------

class TestODEFunc:
    def test_one_compartment_derivatives(self):
        ini_b = _ini_block(ke=0.1)
        model_b = _model_block(
            "d/dt(central) = -ke * central",
            "cp = central / 10",
            "cp ~ add(add.sd)",
        )
        result = parse_model(model_b, ini_b)
        assert result.ode_func is not None

        state = jnp.array([100.0])  # central = 100
        params = {"ke": 0.1}
        dydt = result.ode_func(state, 0.0, params)
        # d/dt(central) = -0.1 * 100 = -10
        assert float(dydt[0]) == pytest.approx(-10.0)

    def test_two_compartment_derivatives(self):
        ini_b = _ini_block(ke=0.1, k12=0.3, k21=0.2)
        model_b = _model_block(
            "d/dt(central) = -ke * central - k12 * central + k21 * peripheral",
            "d/dt(peripheral) = k12 * central - k21 * peripheral",
            "cp = central / 10",
            "cp ~ add(add.sd)",
        )
        result = parse_model(model_b, ini_b)
        assert result.ode_func is not None

        state = jnp.array([100.0, 50.0])  # central=100, peripheral=50
        params = {"ke": 0.1, "k12": 0.3, "k21": 0.2}
        dydt = result.ode_func(state, 0.0, params)
        # d/dt(central) = -0.1*100 - 0.3*100 + 0.2*50 = -10 - 30 + 10 = -30
        assert float(dydt[0]) == pytest.approx(-30.0)
        # d/dt(peripheral) = 0.3*100 - 0.2*50 = 30 - 10 = 20
        assert float(dydt[1]) == pytest.approx(20.0)

    def test_ode_output_func_uses_state(self):
        """output_func for ODE model should use state variables."""
        ini_b = _ini_block(ke=0.1, V=10.0)
        model_b = _model_block(
            "d/dt(central) = -ke * central",
            "cp = central / V",
            "cp ~ add(add.sd)",
        )
        result = parse_model(model_b, ini_b)
        state = jnp.array([200.0])
        params = {"ke": 0.1, "V": 10.0}
        pred = result.output_func(state, 0.0, params)
        # cp = 200 / 10 = 20
        assert float(pred) == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestParseErrors:
    def test_empty_model_block(self):
        """An empty model should not be parseable (caught by model() itself)."""
        ini_b = _ini_block(V=10.0)
        with pytest.raises(ValueError):
            model([])

    def test_no_error_spec_raises(self):
        """Model with no ~ error specification should raise."""
        ini_b = _ini_block(V=10.0)
        model_b = _model_block("cp = 100 / V")
        with pytest.raises(ValueError, match="error model"):
            parse_model(model_b, ini_b)

    def test_unparseable_statement_raises(self):
        ini_b = _ini_block(V=10.0)
        model_b = _model_block("!!! garbage !!!", "cp ~ add(add.sd)")
        with pytest.raises(ValueError):
            parse_model(model_b, ini_b)


# ---------------------------------------------------------------------------
# Math functions available in model expressions
# ---------------------------------------------------------------------------

class TestMathFunctions:
    def test_log_function(self):
        ini_b = _ini_block(V=10.0)
        model_b = _model_block("cp = log(V)", "cp ~ add(add.sd)")
        result = parse_model(model_b, ini_b)
        params = {"V": 10.0}
        pred = result.output_func(None, 0.0, params)
        import math
        assert float(pred) == pytest.approx(math.log(10.0))

    def test_sqrt_function(self):
        ini_b = _ini_block(V=16.0)
        model_b = _model_block("cp = sqrt(V)", "cp ~ add(add.sd)")
        result = parse_model(model_b, ini_b)
        params = {"V": 16.0}
        pred = result.output_func(None, 0.0, params)
        assert float(pred) == pytest.approx(4.0)
