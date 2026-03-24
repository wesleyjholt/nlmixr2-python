"""Core API surface for the first nlmixr2 Python slice."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from math import inf
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import jax.numpy as jnp


NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9._]*$")
RESERVED_PREFIXES = ("_", "rx_", "nlmixr_")


@dataclass(frozen=True)
class InitValue:
    estimate: float
    lower: float | None = None
    upper: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
        }


@dataclass(frozen=True)
class IniBlock:
    values: dict[str, InitValue]

    def to_dict(self) -> dict[str, dict[str, float | None]]:
        return {name: value.to_dict() for name, value in self.values.items()}


@dataclass(frozen=True)
class ModelBlock:
    statements: tuple[str, ...]


@dataclass(frozen=True)
class NLMIXRModel:
    ini: IniBlock
    model: ModelBlock
    source: str = "python"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ini": self.ini.to_dict(),
            "model": list(self.model.statements),
            "source": self.source,
        }


@dataclass(frozen=True)
class NLMIXRFit:
    estimator: str
    n_observations: int
    columns: tuple[str, ...]
    parameter_count: int
    objective: float
    model: NLMIXRModel
    control: dict[str, Any]
    table: dict[str, Any]
    save_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimator": self.estimator,
            "n_observations": self.n_observations,
            "columns": list(self.columns),
            "parameter_count": self.parameter_count,
            "objective": self.objective,
            "model": self.model.to_dict(),
            "control": self.control,
            "table": self.table,
            "save_path": self.save_path,
        }


def _validate_name(name: str) -> None:
    if not isinstance(name, str) or not NAME_PATTERN.match(name):
        raise ValueError(f"Invalid nlmixr2 parameter name: {name!r}")
    if name.startswith(RESERVED_PREFIXES):
        raise ValueError(f"Reserved nlmixr2 parameter prefix in name: {name!r}")


def _coerce_init_value(raw_value: Any) -> InitValue:
    if isinstance(raw_value, InitValue):
        return raw_value
    if isinstance(raw_value, (int, float)):
        return InitValue(estimate=float(raw_value))
    if isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes)):
        values = [float(value) for value in raw_value]
        if len(values) == 1:
            return InitValue(estimate=values[0])
        if len(values) == 2:
            lower, estimate = values
            upper = inf
        elif len(values) == 3:
            lower, estimate, upper = values
        else:
            raise ValueError("Initial values must have length 1, 2, or 3")
        if lower > estimate:
            raise ValueError("Lower bound cannot exceed the estimate")
        if upper < estimate:
            raise ValueError("Upper bound cannot be lower than the estimate")
        return InitValue(estimate=estimate, lower=lower, upper=upper)
    raise TypeError(f"Unsupported initial value specification: {raw_value!r}")


def ini(definitions: Mapping[str, Any]) -> IniBlock:
    """Create the nlmixr2 ini block from Python mappings."""

    if not definitions:
        raise ValueError("ini() requires at least one parameter definition")
    values: dict[str, InitValue] = {}
    for name, raw_value in definitions.items():
        _validate_name(name)
        values[name] = _coerce_init_value(raw_value)
    return IniBlock(values=values)


def model(statements: Sequence[str] | Mapping[str, str]) -> ModelBlock:
    """Create the nlmixr2 model block from a statement list or mapping."""

    normalized: list[str] = []
    if isinstance(statements, Mapping):
        for name, expression in statements.items():
            if not str(name).strip() or not str(expression).strip():
                raise ValueError("Model assignments must have non-empty names and expressions")
            normalized.append(f"{str(name).strip()} = {str(expression).strip()}")
    else:
        for statement in statements:
            text = str(statement).strip()
            if not text:
                raise ValueError("Model statements cannot be blank")
            normalized.append(text)
    if not normalized:
        raise ValueError("model() requires at least one statement")
    return ModelBlock(statements=tuple(normalized))


def _coerce_model(model_object: Any) -> NLMIXRModel:
    if callable(model_object):
        model_object = model_object()
    if isinstance(model_object, NLMIXRModel):
        return model_object
    if isinstance(model_object, Mapping) and {"ini", "model"} <= set(model_object):
        return NLMIXRModel(ini=model_object["ini"], model=model_object["model"])
    raise TypeError("nlmixr2() requires an NLMIXRModel or a zero-argument model factory")


def _coerce_data_column(values: Any) -> jnp.ndarray:
    array = jnp.asarray(values)
    if array.ndim != 1:
        raise ValueError("Data columns must be one-dimensional sequences")
    return array


def _coerce_data(data: Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]]) -> dict[str, jnp.ndarray]:
    if isinstance(data, Mapping):
        if not data:
            raise ValueError("Data cannot be empty")
        arrays = {name: _coerce_data_column(values) for name, values in data.items()}
    else:
        rows = list(data)
        if not rows:
            raise ValueError("Data cannot be empty")
        first_row = rows[0]
        if not isinstance(first_row, Mapping):
            raise TypeError("Record data rows must be mappings")
        columns = tuple(first_row.keys())
        expected = set(columns)
        for row in rows[1:]:
            if not isinstance(row, Mapping):
                raise TypeError("Record data rows must be mappings")
            if set(row.keys()) != expected:
                raise ValueError("All record rows must use the same columns")
        arrays = {column: _coerce_data_column([row[column] for row in rows]) for column in columns}
    lengths = {int(array.shape[0]) for array in arrays.values()}
    if not lengths or lengths == {0}:
        raise ValueError("Data cannot be empty")
    if len(lengths) != 1:
        raise ValueError("All data columns must have the same length")
    return arrays


def _mock_objective(arrays: Mapping[str, jnp.ndarray]) -> float:
    if "dv" not in arrays:
        return 0.0
    dv = arrays["dv"].astype(float)
    centered = dv - jnp.mean(dv)
    return float(jnp.mean(jnp.square(centered)))


def _load_reference_fit(path: str | Path, model_object: NLMIXRModel) -> NLMIXRFit:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    source = payload.get("source")
    run = payload.get("run")
    parameters = payload.get("parameters")
    if not isinstance(source, Mapping) or not isinstance(run, Mapping) or not isinstance(parameters, Mapping):
        raise ValueError("Malformed reference fit artifact: expected source, run, and parameters mappings")
    columns = run.get("columns")
    if not isinstance(columns, Sequence) or isinstance(columns, (str, bytes)):
        raise ValueError("Malformed reference fit artifact: run.columns must be a sequence")
    estimator = str(source.get("estimator") or "").strip()
    if not estimator:
        raise ValueError("Malformed reference fit artifact: source.estimator is required")
    try:
        n_observations = int(run["n_observations"])
        objective = float(run["objective"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Malformed reference fit artifact: invalid n_observations/objective") from exc
    return NLMIXRFit(
        estimator=estimator,
        n_observations=n_observations,
        columns=tuple(str(column) for column in columns),
        parameter_count=len(parameters),
        objective=objective,
        model=model_object,
        control={
            "reference_fit_path": str(path),
            "reference_source_version": source.get("version"),
            "reference_dataset": source.get("dataset"),
        },
        table={
            "reference_parameters": dict(parameters),
            "reference_run": dict(run),
        },
        save_path=None,
    )


def nlmixr2(
    object: Any,
    data: Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]] | None = None,
    est: str = "mock",
    control: Mapping[str, Any] | None = None,
    table: Mapping[str, Any] | None = None,
    save: bool | str | Path | None = None,
    envir: Mapping[str, Any] | None = None,
    reference_fit_path: str | Path | None = None,
    **_: Any,
) -> NLMIXRModel | NLMIXRFit:
    """Return a model spec or a mock fit summary for the first Python slice."""

    del envir
    model_object = _coerce_model(object)
    if est == "reference":
        if reference_fit_path is None:
            raise ValueError("reference_fit_path is required when est='reference'")
        return _load_reference_fit(reference_fit_path, model_object)
    if data is None:
        return model_object
    if est != "mock":
        raise NotImplementedError(f"Estimator {est!r} is not implemented yet")
    arrays = _coerce_data(data)
    fit = NLMIXRFit(
        estimator=est,
        n_observations=next(iter(arrays.values())).shape[0],
        columns=tuple(arrays.keys()),
        parameter_count=len(model_object.ini.values),
        objective=_mock_objective(arrays),
        model=model_object,
        control=dict(control or {}),
        table=dict(table or {}),
        save_path=None,
    )
    if save:
        save_path = Path("nlmixr2-fit.json") if save is True else Path(save)
        save_path.write_text(json.dumps(fit.to_dict(), indent=2) + "\n", encoding="utf-8")
        fit = replace(fit, save_path=str(save_path))
    return fit
