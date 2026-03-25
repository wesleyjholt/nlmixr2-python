"""Core API surface for the first nlmixr2 Python slice."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from math import inf
from pathlib import Path
import re
import time
from typing import Any, Mapping, Sequence

import jax.numpy as jnp

NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9._]*$")
RESERVED_PREFIXES = ("_", "rx_", "nlmixr_")


@dataclass(frozen=True)
class InitValue:
    estimate: float
    lower: float | None = None
    upper: float | None = None
    fixed: bool = False

    def to_dict(self) -> dict[str, float | None | bool]:
        return {
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "fixed": self.fixed,
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
    aic: float | None = None
    bic: float | None = None
    etas: dict[str, Any] | None = None
    predictions: dict[str, Any] | None = None
    shrinkage: dict[str, float] | None = None
    elapsed_seconds: float | None = None
    covariance_result: Any = None

    def to_dict(self) -> dict[str, Any]:
        if self.covariance_result is not None:
            param_names = list(self.model.ini.values.keys())
            se_arr = self.covariance_result.standard_errors
            rse_arr = self.covariance_result.rse
            covariance_info: dict[str, Any] | None = {
                "standard_errors": {
                    param_names[i]: float(se_arr[i])
                    for i in range(len(param_names))
                },
                "rse": {
                    param_names[i]: float(rse_arr[i])
                    for i in range(len(param_names))
                },
                "condition_number": self.covariance_result.condition_number,
            }
        else:
            covariance_info = None

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
            "aic": self.aic,
            "bic": self.bic,
            "etas": self.etas,
            "predictions": self.predictions,
            "shrinkage": self.shrinkage,
            "elapsed_seconds": self.elapsed_seconds,
            "covariance": covariance_info,
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
    if isinstance(raw_value, Mapping):
        if "estimate" not in raw_value:
            raise ValueError("Dict init value specification must include 'estimate'")
        estimate = float(raw_value["estimate"])
        lower = float(raw_value["lower"]) if "lower" in raw_value else None
        upper = float(raw_value["upper"]) if "upper" in raw_value else None
        fixed = bool(raw_value.get("fixed", False))
        if lower is not None and lower > estimate:
            raise ValueError("Lower bound cannot exceed the estimate")
        if upper is not None and upper < estimate:
            raise ValueError("Upper bound cannot be lower than the estimate")
        return InitValue(estimate=estimate, lower=lower, upper=upper, fixed=fixed)
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


def _validate_foce_data(arrays: dict[str, jnp.ndarray]) -> None:
    """Ensure data has the required columns for FOCE estimation."""
    required = {"id", "time", "dv"}
    # Check case-insensitively but require exact lowercase keys in the dict
    available = set(arrays.keys())
    missing = required - available
    if missing:
        raise ValueError(
            f"FOCE estimation requires columns {sorted(required)}, "
            f"missing: {sorted(missing)}"
        )


def _validate_saem_data(arrays: dict[str, jnp.ndarray]) -> None:
    """Ensure data has the required columns for SAEM estimation."""
    required = {"id", "time", "dv"}
    available = set(arrays.keys())
    missing = required - available
    if missing:
        raise ValueError(
            f"SAEM estimation requires columns {sorted(required)}, "
            f"missing: {sorted(missing)}"
        )


def _build_model_func(parsed_model):
    """Wrap a ParsedModel's output_func into the (params, times) -> predictions signature."""
    output_func = parsed_model.output_func
    has_ode = parsed_model.ode_func is not None

    if has_ode:
        from .ode import solve_ode

        ode_func = parsed_model.ode_func
        state_names = parsed_model.state_names
        n_states = len(state_names)

        def model_func(params, times):
            # Build ODE RHS in diffrax convention: f(t, y, args)
            def rhs(t, y, args):
                return ode_func(y, t, args)

            y0 = jnp.zeros(n_states)
            t_span = (float(times[0]), float(times[-1]))
            sol = solve_ode(rhs, t_span, y0, params, times)
            # sol shape: (len(times), n_states)
            preds = jnp.array([
                output_func(sol[i], float(times[i]), params)
                for i in range(len(times))
            ])
            return preds
    else:
        def model_func(params, times):
            preds = jnp.array([
                output_func(None, float(t), params)
                for t in times
            ])
            return preds

    return model_func


def _extract_ini_values(ini_block: IniBlock) -> dict[str, float]:
    """Extract initial parameter estimates from an IniBlock."""
    return {name: iv.estimate for name, iv in ini_block.values.items()}


def _extract_fixed_names(ini_block: IniBlock) -> set[str]:
    """Extract the set of parameter names marked as fixed."""
    return {name for name, iv in ini_block.values.items() if iv.fixed}


def _extract_bounds(ini_block: IniBlock) -> dict[str, tuple[float | None, float | None]] | None:
    """Extract parameter bounds from an IniBlock.

    Returns a dict mapping parameter names to (lower, upper) tuples,
    only for parameters that have at least one non-None bound.
    Returns None if no parameters have bounds.
    """
    bounds: dict[str, tuple[float | None, float | None]] = {}
    for name, iv in ini_block.values.items():
        if iv.lower is not None or iv.upper is not None:
            bounds[name] = (iv.lower, iv.upper)
    return bounds if bounds else None


def _build_default_omega(n_etas: int) -> jnp.ndarray:
    """Build a default diagonal omega matrix."""
    return jnp.eye(n_etas) * 0.1


def _compute_covariance_step(
    objective_wrt_fixed_fn,
    param_names: list[str],
    final_params: dict[str, float],
) -> Any:
    """Compute CovarianceResult from the objective function at final estimates.

    Returns None if the computation fails for any reason.
    """
    from .hessian import (
        CovarianceResult,
        compute_hessian,
        compute_covariance,
        compute_standard_errors,
        compute_correlation,
        compute_rse,
    )

    try:
        param_arr = jnp.array([final_params[n] for n in param_names])
        hessian = compute_hessian(objective_wrt_fixed_fn, param_arr)
        covariance = compute_covariance(hessian)
        se = compute_standard_errors(covariance)
        correlation = compute_correlation(covariance)
        rse = compute_rse(se, param_arr)
        cond = float(jnp.linalg.cond(hessian))
        return CovarianceResult(
            hessian=hessian,
            covariance=covariance,
            correlation=correlation,
            standard_errors=se,
            rse=rse,
            condition_number=cond,
        )
    except Exception:
        return None


def _run_foce(
    model_object: NLMIXRModel,
    arrays: dict[str, jnp.ndarray],
    control: Mapping[str, Any] | None,
    table: Mapping[str, Any] | None,
    save: bool | str | Path | None,
) -> NLMIXRFit:
    """Run FOCE estimation and return an NLMIXRFit."""
    from .diagnostics import compute_aic, compute_bic, compute_shrinkage, compute_per_subject_predictions, compute_phi
    from .estimators import estimate_foce as _estimate_foce
    from .parser import parse_model as _parse_model

    _validate_foce_data(arrays)

    parsed = _parse_model(model_object.model, model_object.ini)
    model_func = _build_model_func(parsed)
    ini_values = _extract_ini_values(model_object.ini)

    ctrl = dict(control or {})

    # Build omega: use control["omega"] if provided, else default diagonal
    if "omega" in ctrl:
        omega_matrix = jnp.asarray(ctrl.pop("omega"))
    else:
        # Number of etas = number of parameters (FOCE adds eta to each param)
        n_etas = len(ini_values)
        omega_matrix = _build_default_omega(n_etas)

    param_bounds = _extract_bounds(model_object.ini)
    fixed_names = _extract_fixed_names(model_object.ini)

    t0 = time.monotonic()
    result = _estimate_foce(
        model=model_func,
        data=arrays,
        ini_values=ini_values,
        omega=omega_matrix,
        control=ctrl if ctrl else None,
        bounds=param_bounds,
        fixed_names=fixed_names if fixed_names else None,
    )
    elapsed = time.monotonic() - t0

    n_obs = int(arrays["dv"].shape[0])
    n_params = len(ini_values)
    aic = compute_aic(result.objective, n_params)
    bic = compute_bic(result.objective, n_params, max(n_obs, 1))

    # Per-subject etas from estimation result
    etas_dict: dict[str, Any] = {"values": result.etas}

    # Shrinkage: 1 - var(etas) / omega_diag, per random effect
    shrinkage_arr = compute_shrinkage(result.etas, omega_matrix)
    param_names = list(ini_values.keys())
    shrinkage_dict = {
        param_names[i]: float(shrinkage_arr[i])
        for i in range(len(param_names))
    }

    # Per-subject predictions and individual log-likelihood (phi)
    preds = compute_per_subject_predictions(model_func, arrays, result.fixed_params, result.etas)
    sigma_est = float(jnp.sqrt(jnp.mean(jnp.square(preds["ires"]))))
    sigma_est = max(sigma_est, 1e-8)
    phi_dict = compute_phi(model_func, arrays, result.fixed_params, result.etas, sigma=sigma_est)

    # Covariance step: compute Hessian-based parameter uncertainty
    from .estimators import _make_objective_wrt_fixed
    obj_wrt_fixed = _make_objective_wrt_fixed(
        param_names, result.etas, omega_matrix, sigma_est, model_func, arrays,
    )
    cov_result = _compute_covariance_step(obj_wrt_fixed, param_names, result.fixed_params)

    fit = NLMIXRFit(
        estimator="foce",
        n_observations=n_obs,
        columns=tuple(arrays.keys()),
        parameter_count=n_params,
        objective=result.objective,
        model=model_object,
        control=dict(control or {}),
        table={
            **(dict(table or {})),
            "fixed_params": result.fixed_params,
            "n_iterations": result.n_iterations,
            "converged": result.converged,
            "phi": phi_dict,
        },
        save_path=None,
        aic=aic,
        bic=bic,
        etas=etas_dict,
        predictions=preds,
        shrinkage=shrinkage_dict,
        elapsed_seconds=elapsed,
        covariance_result=cov_result,
    )

    if save:
        save_path = Path("nlmixr2-fit.json") if save is True else Path(save)
        save_path.write_text(
            json.dumps(fit.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        fit = replace(fit, save_path=str(save_path))

    return fit


def _validate_focei_data(arrays: dict[str, jnp.ndarray]) -> None:
    """Ensure data has the required columns for FOCEi estimation."""
    required = {"id", "time", "dv"}
    available = set(arrays.keys())
    missing = required - available
    if missing:
        raise ValueError(
            f"FOCEi estimation requires columns {sorted(required)}, "
            f"missing: {sorted(missing)}"
        )


def _run_focei(
    model_object: NLMIXRModel,
    arrays: dict[str, jnp.ndarray],
    control: Mapping[str, Any] | None,
    table: Mapping[str, Any] | None,
    save: bool | str | Path | None,
) -> NLMIXRFit:
    """Run FOCEi estimation and return an NLMIXRFit."""
    from .diagnostics import compute_aic, compute_bic, compute_shrinkage, compute_per_subject_predictions, compute_phi
    from .estimators import estimate_focei as _estimate_focei
    from .parser import parse_model as _parse_model

    _validate_focei_data(arrays)

    parsed = _parse_model(model_object.model, model_object.ini)
    model_func = _build_model_func(parsed)
    ini_values = _extract_ini_values(model_object.ini)

    ctrl = dict(control or {})

    # Build omega: use control["omega"] if provided, else default diagonal
    if "omega" in ctrl:
        omega_matrix = jnp.asarray(ctrl.pop("omega"))
    else:
        n_etas = len(ini_values)
        omega_matrix = _build_default_omega(n_etas)

    param_bounds = _extract_bounds(model_object.ini)
    fixed_names = _extract_fixed_names(model_object.ini)

    t0 = time.monotonic()
    result = _estimate_focei(
        model=model_func,
        data=arrays,
        ini_values=ini_values,
        omega=omega_matrix,
        control=ctrl if ctrl else None,
        bounds=param_bounds,
        fixed_names=fixed_names if fixed_names else None,
    )
    elapsed = time.monotonic() - t0

    n_obs = int(arrays["dv"].shape[0])
    n_params = len(ini_values)
    aic = compute_aic(result.objective, n_params)
    bic = compute_bic(result.objective, n_params, max(n_obs, 1))

    etas_dict: dict[str, Any] = {"values": result.etas}

    shrinkage_arr = compute_shrinkage(result.etas, omega_matrix)
    param_names = list(ini_values.keys())
    shrinkage_dict = {
        param_names[i]: float(shrinkage_arr[i])
        for i in range(len(param_names))
    }

    # Per-subject predictions and individual log-likelihood (phi)
    preds = compute_per_subject_predictions(model_func, arrays, result.fixed_params, result.etas)
    sigma_est = float(jnp.sqrt(jnp.mean(jnp.square(preds["ires"]))))
    sigma_est = max(sigma_est, 1e-8)
    phi_dict = compute_phi(model_func, arrays, result.fixed_params, result.etas, sigma=sigma_est)

    # Covariance step: compute Hessian-based parameter uncertainty
    from .estimators import _make_focei_objective_wrt_fixed
    obj_wrt_fixed = _make_focei_objective_wrt_fixed(
        param_names, result.etas, omega_matrix, sigma_est, model_func, arrays,
    )
    cov_result = _compute_covariance_step(obj_wrt_fixed, param_names, result.fixed_params)

    fit = NLMIXRFit(
        estimator="focei",
        n_observations=n_obs,
        columns=tuple(arrays.keys()),
        parameter_count=n_params,
        objective=result.objective,
        model=model_object,
        control=dict(control or {}),
        table={
            **(dict(table or {})),
            "fixed_params": result.fixed_params,
            "n_iterations": result.n_iterations,
            "converged": result.converged,
            "phi": phi_dict,
        },
        save_path=None,
        aic=aic,
        bic=bic,
        etas=etas_dict,
        predictions=preds,
        shrinkage=shrinkage_dict,
        elapsed_seconds=elapsed,
        covariance_result=cov_result,
    )

    if save:
        save_path = Path("nlmixr2-fit.json") if save is True else Path(save)
        save_path.write_text(
            json.dumps(fit.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        fit = replace(fit, save_path=str(save_path))

    return fit


def _run_saem(
    model_object: NLMIXRModel,
    arrays: dict[str, jnp.ndarray],
    control: Mapping[str, Any] | None,
    table: Mapping[str, Any] | None,
    save: bool | str | Path | None,
) -> NLMIXRFit:
    """Run SAEM estimation and return an NLMIXRFit."""
    from .diagnostics import compute_aic, compute_bic, compute_shrinkage, compute_per_subject_predictions, compute_phi
    from .estimators import estimate_saem as _estimate_saem
    from .parser import parse_model as _parse_model

    _validate_saem_data(arrays)

    parsed = _parse_model(model_object.model, model_object.ini)
    model_func = _build_model_func(parsed)
    ini_values = _extract_ini_values(model_object.ini)

    ctrl = dict(control or {})

    # Build omega: use control["omega"] if provided, else default diagonal
    if "omega" in ctrl:
        omega_matrix = jnp.asarray(ctrl.pop("omega"))
    else:
        n_etas = len(ini_values)
        omega_matrix = _build_default_omega(n_etas)

    param_bounds = _extract_bounds(model_object.ini)
    fixed_names = _extract_fixed_names(model_object.ini)

    t0 = time.monotonic()
    result = _estimate_saem(
        model=model_func,
        data=arrays,
        ini_values=ini_values,
        omega=omega_matrix,
        control=ctrl if ctrl else None,
        bounds=param_bounds,
        fixed_names=fixed_names if fixed_names else None,
    )
    elapsed = time.monotonic() - t0

    n_obs = int(arrays["dv"].shape[0])
    n_params = len(ini_values)
    aic = compute_aic(result.objective, n_params)
    bic = compute_bic(result.objective, n_params, max(n_obs, 1))

    # Per-subject etas from estimation result
    etas_dict: dict[str, Any] = {"values": result.etas}

    # Shrinkage: 1 - var(etas) / omega_diag, per random effect
    shrinkage_arr = compute_shrinkage(result.etas, omega_matrix)
    param_names = list(ini_values.keys())
    shrinkage_dict = {
        param_names[i]: float(shrinkage_arr[i])
        for i in range(len(param_names))
    }

    # Per-subject predictions and individual log-likelihood (phi)
    preds = compute_per_subject_predictions(model_func, arrays, result.fixed_params, result.etas)
    sigma_est = float(jnp.sqrt(jnp.mean(jnp.square(preds["ires"]))))
    sigma_est = max(sigma_est, 1e-8)
    phi_dict = compute_phi(model_func, arrays, result.fixed_params, result.etas, sigma=sigma_est)

    # Covariance step: compute Hessian-based parameter uncertainty
    from .estimators import _make_objective_wrt_fixed
    obj_wrt_fixed = _make_objective_wrt_fixed(
        param_names, result.etas, omega_matrix, sigma_est, model_func, arrays,
    )
    cov_result = _compute_covariance_step(obj_wrt_fixed, param_names, result.fixed_params)

    fit = NLMIXRFit(
        estimator="saem",
        n_observations=n_obs,
        columns=tuple(arrays.keys()),
        parameter_count=n_params,
        objective=result.objective,
        model=model_object,
        control=dict(control or {}),
        table={
            **(dict(table or {})),
            "fixed_params": result.fixed_params,
            "n_iterations": result.n_iterations,
            "converged": result.converged,
            "phi": phi_dict,
        },
        save_path=None,
        aic=aic,
        bic=bic,
        etas=etas_dict,
        predictions=preds,
        shrinkage=shrinkage_dict,
        elapsed_seconds=elapsed,
        covariance_result=cov_result,
    )

    if save:
        save_path = Path("nlmixr2-fit.json") if save is True else Path(save)
        save_path.write_text(
            json.dumps(fit.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        fit = replace(fit, save_path=str(save_path))

    return fit


def _validate_posthoc_data(arrays: dict[str, jnp.ndarray]) -> None:
    """Ensure data has the required columns for posthoc estimation."""
    required = {"id", "time", "dv"}
    available = set(arrays.keys())
    missing = required - available
    if missing:
        raise ValueError(
            f"Posthoc estimation requires columns {sorted(required)}, "
            f"missing: {sorted(missing)}"
        )


def _validate_posthoc_control(control: Mapping[str, Any] | None) -> None:
    """Validate that posthoc control contains required keys."""
    ctrl = dict(control or {})
    if "fixed_params" not in ctrl:
        raise ValueError(
            "Posthoc estimation requires 'fixed_params' in control dict"
        )
    if "omega" not in ctrl:
        raise ValueError(
            "Posthoc estimation requires 'omega' in control dict"
        )


def _run_posthoc(
    model_object: NLMIXRModel,
    arrays: dict[str, jnp.ndarray],
    control: Mapping[str, Any] | None,
    table: Mapping[str, Any] | None,
    save: bool | str | Path | None,
) -> NLMIXRFit:
    """Run posthoc (empirical Bayes) estimation and return an NLMIXRFit."""
    from .estimators import estimate_posthoc as _estimate_posthoc
    from .parser import parse_model as _parse_model

    _validate_posthoc_data(arrays)
    _validate_posthoc_control(control)

    parsed = _parse_model(model_object.model, model_object.ini)
    model_func = _build_model_func(parsed)

    ctrl = dict(control or {})
    fixed_params = ctrl.pop("fixed_params")
    omega_matrix = jnp.asarray(ctrl.pop("omega"))
    sigma = float(ctrl.pop("sigma", 1.0))

    result = _estimate_posthoc(
        model_func=model_func,
        data=arrays,
        fixed_params=fixed_params,
        omega=omega_matrix,
        sigma=sigma,
        control=ctrl if ctrl else None,
    )

    fit = NLMIXRFit(
        estimator="posthoc",
        n_observations=int(arrays["dv"].shape[0]),
        columns=tuple(arrays.keys()),
        parameter_count=len(fixed_params),
        objective=result.objective,
        model=model_object,
        control=dict(control or {}),
        table={
            **(dict(table or {})),
            "fixed_params": result.fixed_params,
            "n_iterations": result.n_iterations,
            "converged": result.converged,
        },
        save_path=None,
    )

    if save:
        save_path = Path("nlmixr2-fit.json") if save is True else Path(save)
        save_path.write_text(
            json.dumps(fit.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        fit = replace(fit, save_path=str(save_path))

    return fit


def _validate_nlm_data(arrays: dict[str, jnp.ndarray]) -> None:
    """Ensure data has the required columns for NLM estimation."""
    required = {"id", "time", "dv"}
    available = set(arrays.keys())
    missing = required - available
    if missing:
        raise ValueError(
            f"NLM estimation requires columns {sorted(required)}, "
            f"missing: {sorted(missing)}"
        )


def _run_nlm(
    model_object: NLMIXRModel,
    arrays: dict[str, jnp.ndarray],
    control: Mapping[str, Any] | None,
    table: Mapping[str, Any] | None,
    save: bool | str | Path | None,
) -> NLMIXRFit:
    """Run Nelder-Mead (NLM) estimation and return an NLMIXRFit."""
    from .diagnostics import compute_aic, compute_bic, compute_shrinkage, compute_per_subject_predictions, compute_phi
    from .estimators import estimate_nlm as _estimate_nlm
    from .parser import parse_model as _parse_model

    _validate_nlm_data(arrays)

    parsed = _parse_model(model_object.model, model_object.ini)
    model_func = _build_model_func(parsed)
    ini_values = _extract_ini_values(model_object.ini)

    ctrl = dict(control or {})

    # Build omega: use control["omega"] if provided, else default diagonal
    if "omega" in ctrl:
        omega_matrix = jnp.asarray(ctrl.pop("omega"))
    else:
        n_etas = len(ini_values)
        omega_matrix = _build_default_omega(n_etas)

    t0 = time.monotonic()
    result = _estimate_nlm(
        model_func=model_func,
        data=arrays,
        ini_values=ini_values,
        omega=omega_matrix,
        control=ctrl if ctrl else None,
    )
    elapsed = time.monotonic() - t0

    n_obs = int(arrays["dv"].shape[0])
    n_params = len(ini_values)
    aic = compute_aic(result.objective, n_params)
    bic = compute_bic(result.objective, n_params, max(n_obs, 1))

    # Per-subject etas from estimation result
    etas_dict: dict[str, Any] = {"values": result.etas}

    # Shrinkage
    shrinkage_arr = compute_shrinkage(result.etas, omega_matrix)
    param_names = list(ini_values.keys())
    shrinkage_dict = {
        param_names[i]: float(shrinkage_arr[i])
        for i in range(len(param_names))
    }

    # Per-subject predictions and individual log-likelihood (phi)
    preds = compute_per_subject_predictions(model_func, arrays, result.fixed_params, result.etas)
    sigma_est = float(jnp.sqrt(jnp.mean(jnp.square(preds["ires"]))))
    sigma_est = max(sigma_est, 1e-8)
    phi_dict = compute_phi(model_func, arrays, result.fixed_params, result.etas, sigma=sigma_est)

    fit = NLMIXRFit(
        estimator="nlm",
        n_observations=n_obs,
        columns=tuple(arrays.keys()),
        parameter_count=n_params,
        objective=result.objective,
        model=model_object,
        control=dict(control or {}),
        table={
            **(dict(table or {})),
            "fixed_params": result.fixed_params,
            "n_iterations": result.n_iterations,
            "converged": result.converged,
            "phi": phi_dict,
        },
        save_path=None,
        aic=aic,
        bic=bic,
        etas=etas_dict,
        predictions=preds,
        shrinkage=shrinkage_dict,
        elapsed_seconds=elapsed,
    )

    if save:
        save_path = Path("nlmixr2-fit.json") if save is True else Path(save)
        save_path.write_text(
            json.dumps(fit.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        fit = replace(fit, save_path=str(save_path))

    return fit


def _validate_nlme_data(arrays: dict[str, jnp.ndarray]) -> None:
    """Ensure data has the required columns for NLME estimation."""
    required = {"id", "time", "dv"}
    available = set(arrays.keys())
    missing = required - available
    if missing:
        raise ValueError(
            f"NLME estimation requires columns {sorted(required)}, "
            f"missing: {sorted(missing)}"
        )


def _run_nlme(
    model_object: NLMIXRModel,
    arrays: dict[str, jnp.ndarray],
    control: Mapping[str, Any] | None,
    table: Mapping[str, Any] | None,
    save: bool | str | Path | None,
) -> NLMIXRFit:
    """Run NLME (linearization) estimation and return an NLMIXRFit."""
    from .diagnostics import compute_aic, compute_bic, compute_shrinkage, compute_per_subject_predictions, compute_phi
    from .estimators import estimate_nlme as _estimate_nlme
    from .parser import parse_model as _parse_model

    _validate_nlme_data(arrays)

    parsed = _parse_model(model_object.model, model_object.ini)
    model_func = _build_model_func(parsed)
    ini_values = _extract_ini_values(model_object.ini)

    ctrl = dict(control or {})

    # Build omega: use control["omega"] if provided, else default diagonal
    if "omega" in ctrl:
        omega_matrix = jnp.asarray(ctrl.pop("omega"))
    else:
        n_etas = len(ini_values)
        omega_matrix = _build_default_omega(n_etas)

    t0 = time.monotonic()
    result = _estimate_nlme(
        model_func=model_func,
        data=arrays,
        ini_values=ini_values,
        omega=omega_matrix,
        control=ctrl if ctrl else None,
    )
    elapsed = time.monotonic() - t0

    n_obs = int(arrays["dv"].shape[0])
    n_params = len(ini_values)
    aic = compute_aic(result.objective, n_params)
    bic = compute_bic(result.objective, n_params, max(n_obs, 1))

    # Per-subject etas from estimation result
    etas_dict: dict[str, Any] = {"values": result.etas}

    # Shrinkage
    shrinkage_arr = compute_shrinkage(result.etas, omega_matrix)
    param_names = list(ini_values.keys())
    shrinkage_dict = {
        param_names[i]: float(shrinkage_arr[i])
        for i in range(len(param_names))
    }

    # Per-subject predictions and individual log-likelihood (phi)
    preds = compute_per_subject_predictions(model_func, arrays, result.fixed_params, result.etas)
    sigma_est = float(jnp.sqrt(jnp.mean(jnp.square(preds["ires"]))))
    sigma_est = max(sigma_est, 1e-8)
    phi_dict = compute_phi(model_func, arrays, result.fixed_params, result.etas, sigma=sigma_est)

    fit = NLMIXRFit(
        estimator="nlme",
        n_observations=n_obs,
        columns=tuple(arrays.keys()),
        parameter_count=n_params,
        objective=result.objective,
        model=model_object,
        control=dict(control or {}),
        table={
            **(dict(table or {})),
            "fixed_params": result.fixed_params,
            "n_iterations": result.n_iterations,
            "converged": result.converged,
            "phi": phi_dict,
        },
        save_path=None,
        aic=aic,
        bic=bic,
        etas=etas_dict,
        predictions=preds,
        shrinkage=shrinkage_dict,
        elapsed_seconds=elapsed,
    )

    if save:
        save_path = Path("nlmixr2-fit.json") if save is True else Path(save)
        save_path.write_text(
            json.dumps(fit.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        fit = replace(fit, save_path=str(save_path))

    return fit


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
    arrays = _coerce_data(data)

    if est == "foce":
        return _run_foce(model_object, arrays, control, table, save)

    if est == "focei":
        return _run_focei(model_object, arrays, control, table, save)

    if est == "saem":
        return _run_saem(model_object, arrays, control, table, save)

    if est == "posthoc":
        return _run_posthoc(model_object, arrays, control, table, save)

    if est == "nlm":
        return _run_nlm(model_object, arrays, control, table, save)

    if est == "nlme":
        return _run_nlme(model_object, arrays, control, table, save)

    if est != "mock":
        raise NotImplementedError(f"Estimator {est!r} is not implemented yet")
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
