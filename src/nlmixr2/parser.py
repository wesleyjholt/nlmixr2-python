"""Model parser: converts nlmixr2-style model statements into executable JAX functions.

Parses model block statements (ODE definitions, algebraic assignments,
residual error specifications) and produces callable functions that use
``jax.numpy`` operations.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any, Callable

import jax.numpy as jnp

from .api import IniBlock, ModelBlock

# ---------------------------------------------------------------------------
# Public data structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedModel:
    """Result of parsing an nlmixr2 model + ini block."""

    parameter_names: list[str]
    state_names: list[str]
    ode_func: Callable[[jnp.ndarray, float, dict[str, float]], jnp.ndarray] | None
    output_func: Callable[[jnp.ndarray | None, float, dict[str, float]], float]
    error_model: str  # "additive", "proportional", or "combined"
    dv_name: str  # name of the dependent variable (LHS of ~ statement)
    _model_statements: tuple[str, ...] = ()  # raw model statements for covariate support


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_ODE_PATTERN = re.compile(r"^d/dt\((\w+)\)\s*=\s*(.+)$")
_ASSIGN_PATTERN = re.compile(r"^(\w[\w.]*)\s*=\s*(.+)$")
_ERROR_PATTERN = re.compile(r"^(\w[\w.]*)\s*~\s*(.+)$")

# Patterns for error model RHS
_ADD_ONLY = re.compile(r"^add\((\w[\w.]*)\)$")
_PROP_ONLY = re.compile(r"^prop\((\w[\w.]*)\)$")
_COMBINED = re.compile(r"^add\((\w[\w.]*)\)\s*\+\s*prop\((\w[\w.]*)\)$")

# Math functions we inject into the eval namespace (mapped to jnp equivalents)
_MATH_NAMESPACE: dict[str, Any] = {
    "exp": jnp.exp,
    "log": jnp.log,
    "sqrt": jnp.sqrt,
    "abs": jnp.abs,
    "sin": jnp.sin,
    "cos": jnp.cos,
}


def _safe_compile(expr_str: str, description: str) -> Any:
    """Parse *expr_str* as a Python expression and compile it."""
    try:
        tree = ast.parse(expr_str.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(
            f"Cannot parse {description}: {expr_str!r}"
        ) from exc
    ast.fix_missing_locations(tree)
    return compile(tree, f"<nlmixr2:{description}>", "eval")


# ---------------------------------------------------------------------------
# Statement classifiers
# ---------------------------------------------------------------------------


@dataclass
class _ODEStmt:
    state_name: str
    rhs_expr: str


@dataclass
class _AssignStmt:
    lhs: str
    rhs_expr: str


@dataclass
class _ErrorStmt:
    dv_name: str
    error_type: str  # "additive", "proportional", "combined"
    raw: str


def _classify(stmt: str) -> _ODEStmt | _AssignStmt | _ErrorStmt:
    # Try ODE first
    m = _ODE_PATTERN.match(stmt)
    if m:
        return _ODEStmt(state_name=m.group(1), rhs_expr=m.group(2).strip())

    # Try error spec (must come before assign since ~ is distinct from =)
    m = _ERROR_PATTERN.match(stmt)
    if m:
        dv = m.group(1)
        rhs = m.group(2).strip()
        if _COMBINED.match(rhs):
            etype = "combined"
        elif _ADD_ONLY.match(rhs):
            etype = "additive"
        elif _PROP_ONLY.match(rhs):
            etype = "proportional"
        else:
            raise ValueError(f"Unrecognised residual error model: {rhs!r}")
        return _ErrorStmt(dv_name=dv, error_type=etype, raw=rhs)

    # Try algebraic assignment
    m = _ASSIGN_PATTERN.match(stmt)
    if m:
        return _AssignStmt(lhs=m.group(1), rhs_expr=m.group(2).strip())

    raise ValueError(f"Unparseable model statement: {stmt!r}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def parse_model(model_block: ModelBlock, ini_block: IniBlock) -> ParsedModel:
    """Parse a ``ModelBlock`` + ``IniBlock`` into a ``ParsedModel``."""

    parameter_names = list(ini_block.values.keys())

    ode_stmts: list[_ODEStmt] = []
    assign_stmts: list[_AssignStmt] = []
    error_stmts: list[_ErrorStmt] = []

    for stmt in model_block.statements:
        classified = _classify(stmt)
        if isinstance(classified, _ODEStmt):
            ode_stmts.append(classified)
        elif isinstance(classified, _AssignStmt):
            assign_stmts.append(classified)
        elif isinstance(classified, _ErrorStmt):
            error_stmts.append(classified)

    # Validate that we have exactly one error specification
    if len(error_stmts) == 0:
        raise ValueError(
            "Model must contain a residual error model specification "
            "(e.g. 'cp ~ add(add.sd)')"
        )
    error_stmt = error_stmts[0]

    # State names (preserving order of appearance)
    state_names = [s.state_name for s in ode_stmts]

    # Build a state-index lookup
    state_idx = {name: i for i, name in enumerate(state_names)}

    # ------------------------------------------------------------------
    # Build ode_func  (state, t, params) -> d_state
    # ------------------------------------------------------------------
    ode_func = None
    if ode_stmts:
        # Pre-compile each ODE RHS expression
        ode_codes = [
            _safe_compile(s.rhs_expr, f"d/dt({s.state_name})")
            for s in ode_stmts
        ]

        # Also compile algebraic assignments that ODE RHS may depend on
        assign_codes = [
            (a.lhs, _safe_compile(a.rhs_expr, f"{a.lhs} = ..."))
            for a in assign_stmts
        ]

        _n_states = len(state_names)
        _state_idx = dict(state_idx)
        _state_names_local = list(state_names)

        def _ode_func(
            state: jnp.ndarray, t: float, params: dict[str, float]
        ) -> jnp.ndarray:
            ns: dict[str, Any] = {"__builtins__": {"__import__": __import__}}
            ns.update(_MATH_NAMESPACE)
            ns["t"] = t
            ns.update(params)
            # Inject state variables as scalars
            for _name, _i in _state_idx.items():
                ns[_name] = state[_i]
            # Evaluate algebraic assignments in order
            for _lhs, _code in assign_codes:
                ns[_lhs] = eval(_code, ns)  # noqa: S307
            # Evaluate each ODE RHS
            derivs = []
            for _code in ode_codes:
                derivs.append(eval(_code, ns))  # noqa: S307
            return jnp.array(derivs)

        ode_func = _ode_func

    # ------------------------------------------------------------------
    # Build output_func  (state | None, t, params) -> prediction
    # ------------------------------------------------------------------

    # The DV is defined by the error statement's dv_name.
    # We need to evaluate algebraic assignments in order to compute it.
    assign_codes_out = [
        (a.lhs, _safe_compile(a.rhs_expr, f"{a.lhs} = ..."))
        for a in assign_stmts
    ]
    _dv_name = error_stmt.dv_name
    _state_idx_out = dict(state_idx)

    def _output_func(
        state: jnp.ndarray | None, t: float, params: dict[str, float]
    ) -> float:
        ns: dict[str, Any] = {"__builtins__": {"__import__": __import__}}
        ns.update(_MATH_NAMESPACE)
        ns["t"] = t
        ns.update(params)
        # Inject state variables if present
        if state is not None:
            for _name, _i in _state_idx_out.items():
                ns[_name] = state[_i]
        # Evaluate algebraic assignments
        for _lhs, _code in assign_codes_out:
            ns[_lhs] = eval(_code, ns)  # noqa: S307
        return ns[_dv_name]

    return ParsedModel(
        parameter_names=parameter_names,
        state_names=state_names,
        ode_func=ode_func,
        output_func=_output_func,
        error_model=error_stmt.error_type,
        dv_name=error_stmt.dv_name,
        _model_statements=model_block.statements,
    )
