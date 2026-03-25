"""Covariate support and mu-referencing detection for nlmixr2 models.

Provides utilities to:
- Extract covariate names from model statements by matching against data columns
- Center and transform covariate data
- Add covariate effects (linear, power, exponential) to parsed models
- Detect mu-referenced parameters for SAEM efficiency
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

import jax.numpy as jnp

from .api import ModelBlock
from .parser import (
    ParsedModel,
    _MATH_NAMESPACE,
    _classify,
    _AssignStmt,
    _ErrorStmt,
    _ODEStmt,
    _safe_compile,
)

# Names that should never be treated as covariates even if they appear in data
_STANDARD_COLUMNS = frozenset({
    "id", "time", "dv", "evid", "amt", "cmt", "rate", "ss", "ii", "addl", "mdv",
})

# Math function names excluded from covariate detection
_MATH_NAMES = frozenset(_MATH_NAMESPACE.keys()) | frozenset({"add", "prop"})

# Pattern to extract Python identifiers from an expression
_IDENT_PATTERN = re.compile(r"\b([A-Za-z_][A-Za-z0-9_.]*)\b")

# Mu-referencing patterns
_MU_DEF_PATTERN = re.compile(r"^(mu_\d+)\s*=\s*(.+)$")
_ETA_PATTERN = re.compile(r"\beta[._]\w+\b")


# ---------------------------------------------------------------------------
# extract_covariates
# ---------------------------------------------------------------------------


def extract_covariates(model_block: ModelBlock, data_columns: list[str]) -> list[str]:
    """Identify covariate names that appear in model statements and match data columns.

    Returns covariate names found in both model statements and *data_columns*,
    excluding standard columns (id, time, dv, etc.) and math function names.
    """
    all_idents: set[str] = set()
    for stmt in model_block.statements:
        all_idents.update(_IDENT_PATTERN.findall(stmt))

    candidate_columns = set(data_columns) - _STANDARD_COLUMNS
    covariates = []
    for col in data_columns:
        if col in candidate_columns and col in all_idents and col not in _MATH_NAMES:
            covariates.append(col)
    return covariates


# ---------------------------------------------------------------------------
# CovariateModel
# ---------------------------------------------------------------------------


@dataclass
class CovariateModel:
    """Specification of covariates and how they should be handled.

    Attributes
    ----------
    covariates : dict[str, str]
        Mapping of covariate name to role: ``"continuous"`` or ``"categorical"``.
    centering : dict[str, float]
        Mapping of covariate name to centering value (e.g., median weight).
    transformations : dict[str, str]
        Mapping of covariate name to transformation:
        ``"none"``, ``"log"``, or ``"normalized"``.
    """

    covariates: dict[str, str]
    centering: dict[str, float]
    transformations: dict[str, str]


# ---------------------------------------------------------------------------
# center_covariates
# ---------------------------------------------------------------------------


def center_covariates(
    data: dict[str, jnp.ndarray],
    covariate_model: CovariateModel,
) -> dict[str, jnp.ndarray]:
    """Return a new data dict with centered covariate columns added.

    For each covariate with a centering value, a ``<name>_centered`` column
    is added.  Original columns are preserved.
    """
    result = dict(data)

    for cov_name in covariate_model.covariates:
        if cov_name not in covariate_model.centering:
            continue
        if cov_name not in data:
            continue

        center = covariate_model.centering[cov_name]
        transform = covariate_model.transformations.get(cov_name, "none")
        values = data[cov_name].astype(float)

        if transform == "log":
            centered = jnp.log(values) - jnp.log(center)
        elif transform == "normalized":
            centered = (values - center) / center
        else:  # "none"
            centered = values - center

        result[f"{cov_name}_centered"] = centered

    return result


# ---------------------------------------------------------------------------
# add_covariate_to_model
# ---------------------------------------------------------------------------


def _apply_covariate_effect(
    base_value: Any,
    effect: str,
    theta_cov: Any,
    params: dict[str, Any],
    cov_name: str,
) -> Any:
    """Apply a covariate effect to a base parameter value."""
    if effect == "linear":
        return base_value + theta_cov * params.get(f"{cov_name}_centered", 0.0)
    elif effect == "power":
        return base_value * (params.get(f"{cov_name}_ratio", 1.0) ** theta_cov)
    elif effect == "exponential":
        return base_value * jnp.exp(theta_cov * params.get(f"{cov_name}_centered", 0.0))
    else:
        raise ValueError(f"Unknown covariate effect type: {effect!r}")


def _rebuild_output_func(
    parsed_model: ParsedModel,
    param_name: str,
    theta_name: str,
    cov_name: str,
    effect: str,
) -> Callable:
    """Rebuild the output_func with a covariate effect injected after *param_name*.

    This replays the parser's eval chain from the stored model statements,
    inserting the covariate adjustment immediately after the assignment
    that computes *param_name*.
    """
    statements = parsed_model._model_statements
    state_names = parsed_model.state_names
    state_idx = {name: i for i, name in enumerate(state_names)}
    dv_name = parsed_model.dv_name

    # Classify statements and compile assignment RHS expressions
    classified = [_classify(s) for s in statements]
    assign_stmts = [c for c in classified if isinstance(c, _AssignStmt)]

    # Pre-compile each assignment
    assign_codes = [
        (a.lhs, _safe_compile(a.rhs_expr, f"{a.lhs} = ..."))
        for a in assign_stmts
    ]

    _effect = effect
    _param_name = param_name
    _theta_name = theta_name
    _cov_name = cov_name
    _state_idx = dict(state_idx)
    _dv_name = dv_name

    def output_func(
        state: jnp.ndarray | None, t: float, params: dict[str, Any]
    ) -> float:
        ns: dict[str, Any] = {"__builtins__": {"__import__": __import__}}
        ns.update(_MATH_NAMESPACE)
        ns["t"] = t
        ns.update(params)

        if state is not None:
            for sn, si in _state_idx.items():
                ns[sn] = state[si]

        # Evaluate assignments in order, injecting covariate after target
        for lhs, code in assign_codes:
            ns[lhs] = eval(code, ns)  # noqa: S307
            if lhs == _param_name and _theta_name in params:
                # Apply covariate effect to the just-computed value
                base_val = ns[lhs]
                theta_val = params[_theta_name]
                ns[lhs] = _apply_covariate_effect(
                    base_val, _effect, theta_val, params, _cov_name
                )

        return ns[_dv_name]

    return output_func


def _rebuild_ode_func(
    parsed_model: ParsedModel,
    param_name: str,
    theta_name: str,
    cov_name: str,
    effect: str,
) -> Callable | None:
    """Rebuild the ODE func with covariate injection, if the model has ODEs."""
    statements = parsed_model._model_statements
    state_names = parsed_model.state_names
    if not state_names:
        return None

    state_idx = {name: i for i, name in enumerate(state_names)}
    classified = [_classify(s) for s in statements]

    ode_stmts = [c for c in classified if isinstance(c, _ODEStmt)]
    assign_stmts = [c for c in classified if isinstance(c, _AssignStmt)]

    ode_codes = [
        _safe_compile(s.rhs_expr, f"d/dt({s.state_name})")
        for s in ode_stmts
    ]
    assign_codes = [
        (a.lhs, _safe_compile(a.rhs_expr, f"{a.lhs} = ..."))
        for a in assign_stmts
    ]

    _effect = effect
    _param_name = param_name
    _theta_name = theta_name
    _cov_name = cov_name
    _state_idx = dict(state_idx)
    _n_states = len(state_names)

    def ode_func(
        state: jnp.ndarray, t: float, params: dict[str, Any]
    ) -> jnp.ndarray:
        ns: dict[str, Any] = {"__builtins__": {"__import__": __import__}}
        ns.update(_MATH_NAMESPACE)
        ns["t"] = t
        ns.update(params)

        for sn, si in _state_idx.items():
            ns[sn] = state[si]

        for lhs, code in assign_codes:
            ns[lhs] = eval(code, ns)  # noqa: S307
            if lhs == _param_name and _theta_name in params:
                base_val = ns[lhs]
                theta_val = params[_theta_name]
                ns[lhs] = _apply_covariate_effect(
                    base_val, _effect, theta_val, params, _cov_name
                )

        derivs = [eval(c, ns) for c in ode_codes]  # noqa: S307
        return jnp.array(derivs)

    return ode_func


def add_covariate_to_model(
    parsed_model: ParsedModel,
    covariate_name: str,
    parameter_name: str,
    effect: str = "linear",
) -> ParsedModel:
    """Add a covariate effect to a parsed model and return an updated model.

    A new parameter ``theta_<covariate_name>_<parameter_name>`` is introduced.
    The caller must supply the covariate value in the params dict when calling
    the output_func:

    - linear / exponential: ``<cov_name>_centered``
    - power: ``<cov_name>_ratio``

    Effect types:

    - ``"linear"``:  param = base + theta * (COV - center)
    - ``"power"``:   param = base * (COV / center) ** theta
    - ``"exponential"``: param = base * exp(theta * (COV - center))
    """
    theta_name = f"theta_{covariate_name}_{parameter_name}"
    new_param_names = list(parsed_model.parameter_names) + [theta_name]

    output_func = _rebuild_output_func(
        parsed_model, parameter_name, theta_name, covariate_name, effect
    )
    ode_func = _rebuild_ode_func(
        parsed_model, parameter_name, theta_name, covariate_name, effect
    )

    return ParsedModel(
        parameter_names=new_param_names,
        state_names=parsed_model.state_names,
        ode_func=ode_func,
        output_func=output_func,
        error_model=parsed_model.error_model,
        dv_name=parsed_model.dv_name,
        _model_statements=parsed_model._model_statements,
    )


# ---------------------------------------------------------------------------
# detect_mu_referencing
# ---------------------------------------------------------------------------


def detect_mu_referencing(model_statements: list[str]) -> list[str]:
    """Detect mu-referenced parameters in model statements.

    A mu-reference follows the pattern ``mu_N = <expr> + eta.<name>``,
    where the RHS includes an eta (random effect) term.  This pattern
    enables efficient SAEM estimation.
    """
    mu_refs = []
    for stmt in model_statements:
        m = _MU_DEF_PATTERN.match(stmt.strip())
        if m:
            mu_name = m.group(1)
            rhs = m.group(2)
            if _ETA_PATTERN.search(rhs):
                mu_refs.append(mu_name)
    return mu_refs
