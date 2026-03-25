"""Model update/piping utilities for modifying and refitting models."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from .api import (
    IniBlock,
    ModelBlock,
    NLMIXRFit,
    NLMIXRModel,
    _coerce_init_value,
    _validate_name,
    model as _model_block,
    nlmixr2,
)


def _extract_model(model_or_fit: NLMIXRModel | NLMIXRFit) -> NLMIXRModel:
    """Extract an NLMIXRModel from either a model or a fit object."""
    if isinstance(model_or_fit, NLMIXRFit):
        return model_or_fit.model
    if isinstance(model_or_fit, NLMIXRModel):
        return model_or_fit
    raise TypeError(
        f"Expected NLMIXRModel or NLMIXRFit, got {type(model_or_fit).__name__}"
    )


def update_ini(model_or_fit: NLMIXRModel | NLMIXRFit, **new_values: Any) -> NLMIXRModel:
    """Return a new NLMIXRModel with updated ini parameter values.

    Accepts keyword arguments where:
    - A scalar updates the estimate: ``update_ini(m, tvcl=0.5)``
    - A tuple sets (lower, estimate, upper): ``update_ini(m, tvcl=(0.1, 0.5, 10.0))``
    - New parameter names are added to the ini block.
    """
    m = _extract_model(model_or_fit)
    updated_values = dict(m.ini.values)
    for name, raw_value in new_values.items():
        _validate_name(name)
        updated_values[name] = _coerce_init_value(raw_value)
    new_ini = IniBlock(values=updated_values)
    return replace(m, ini=new_ini)


def update_model(
    model_or_fit: NLMIXRModel | NLMIXRFit,
    statements: Sequence[str] | Mapping[str, str],
) -> NLMIXRModel:
    """Return a new NLMIXRModel with the model block replaced."""
    m = _extract_model(model_or_fit)
    new_model_block = _model_block(statements)
    return replace(m, model=new_model_block)


def add_statement(
    model_or_fit: NLMIXRModel | NLMIXRFit,
    statement: str,
) -> NLMIXRModel:
    """Return a new NLMIXRModel with a statement appended to the model block."""
    m = _extract_model(model_or_fit)
    text = str(statement).strip()
    if not text:
        raise ValueError("Statement cannot be blank")
    new_statements = m.model.statements + (text,)
    new_model_block = ModelBlock(statements=new_statements)
    return replace(m, model=new_model_block)


def remove_parameter(
    model_or_fit: NLMIXRModel | NLMIXRFit,
    name: str,
) -> NLMIXRModel:
    """Return a new NLMIXRModel with the named parameter removed from ini."""
    m = _extract_model(model_or_fit)
    if name not in m.ini.values:
        raise KeyError(f"Parameter {name!r} not found in ini block")
    updated_values = {k: v for k, v in m.ini.values.items() if k != name}
    new_ini = IniBlock(values=updated_values)
    return replace(m, ini=new_ini)


def refit(
    fit: NLMIXRFit,
    data: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    est: str | None = None,
    control: Mapping[str, Any] | None = None,
) -> NLMIXRFit:
    """Re-run nlmixr2() using the model from an existing fit.

    Parameters
    ----------
    fit : NLMIXRFit
        The previous fit result whose model to reuse.
    data : optional
        New data; if None, the fit cannot be re-run (data is required).
    est : optional
        Estimator name; defaults to the estimator from the original fit.
    control : optional
        Control options; defaults to the control from the original fit.

    Returns
    -------
    NLMIXRFit
        A new fit result.
    """
    if not isinstance(fit, NLMIXRFit):
        raise TypeError(f"refit() requires an NLMIXRFit, got {type(fit).__name__}")

    estimator = est if est is not None else fit.estimator
    ctrl = control if control is not None else fit.control

    result = nlmixr2(
        fit.model,
        data=data,
        est=estimator,
        control=ctrl,
    )
    if not isinstance(result, NLMIXRFit):
        raise RuntimeError("refit() expected nlmixr2() to return an NLMIXRFit (data required)")
    return result
