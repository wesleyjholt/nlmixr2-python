"""Automatic stepwise covariate selection (equivalent to nlmixr2extra's covarSearchAuto).

Provides forward addition, backward elimination, and full stepwise
covariate selection using likelihood ratio tests to decide significance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .api import NLMIXRFit, NLMIXRModel, IniBlock, InitValue, ModelBlock, nlmixr2
from .compare import _chi2_sf


@dataclass
class StepResult:
    """Result of a single covariate search step.

    Attributes
    ----------
    covariate : str
        Name of the covariate tested.
    parameter : str
        Name of the model parameter the covariate was applied to.
    effect : str
        Type of covariate effect (e.g. "linear", "power", "exponential").
    direction : str
        Either "forward" (addition) or "backward" (elimination).
    delta_obj : float
        Change in objective function value (negative = improvement for forward).
    p_value : float
        p-value from the likelihood ratio test (chi-squared, df=1).
    selected : bool
        Whether this covariate-parameter combo was selected in this step.
    """

    covariate: str
    parameter: str
    effect: str
    direction: str
    delta_obj: float
    p_value: float
    selected: bool


def _default_fit_factory(
    base_fit: NLMIXRFit,
    data: dict[str, Any],
    covariate: str,
    parameter: str,
    effect: str,
) -> NLMIXRFit:
    """Default factory: re-fit with the mock estimator adding a covariate parameter.

    This adds a theta for the covariate effect and re-runs the mock estimator,
    producing a slightly different objective based on the data.
    """
    model_obj = base_fit.model
    # Add one more parameter for the covariate effect
    theta_name = f"theta_{covariate}_{parameter}"
    new_values = dict(model_obj.ini.values)
    new_values[theta_name] = InitValue(estimate=0.0)
    new_ini = IniBlock(values=new_values)
    new_model = NLMIXRModel(ini=new_ini, model=model_obj.model)

    return nlmixr2(new_model, data, est="mock")


def _compute_lrt_pvalue(delta_obj: float, df: int = 1) -> float:
    """Compute p-value for a likelihood ratio test.

    For forward addition: delta_obj = new_obj - base_obj (negative if improvement).
    The LRT statistic is -delta_obj (positive when there's improvement).
    """
    statistic = -delta_obj  # positive when objective decreased
    if statistic <= 0:
        return 1.0
    return _chi2_sf(statistic, df)


def forward_addition(
    base_fit: NLMIXRFit,
    data: dict[str, Any],
    covariates: Sequence[str],
    parameters: Sequence[str],
    effects: Sequence[str] = ("linear",),
    alpha: float = 0.05,
    fit_factory: Callable | None = None,
) -> list[StepResult]:
    """Forward addition step: test adding each covariate-parameter-effect combo.

    For each combination, a model with the covariate added is fit. The combo
    with the largest OFV drop that is significant (p < alpha, LRT df=1)
    is marked as selected.

    Parameters
    ----------
    base_fit : NLMIXRFit
        The current base model fit.
    data : dict
        Dataset.
    covariates : list of str
        Covariate names to test.
    parameters : list of str
        Parameter names to test covariates on.
    effects : tuple of str
        Effect types to try (default: ("linear",)).
    alpha : float
        Significance level for the LRT (default: 0.05).
    fit_factory : callable, optional
        Function(base_fit, data, covariate, parameter, effect) -> NLMIXRFit.
        If None, uses the default mock estimator factory.

    Returns
    -------
    list of StepResult
        All tested combos, sorted by delta_obj ascending (most negative first).
    """
    if fit_factory is None:
        fit_factory = _default_fit_factory

    results: list[StepResult] = []

    for cov in covariates:
        for par in parameters:
            for eff in effects:
                new_fit = fit_factory(base_fit, data, cov, par, eff)
                delta = new_fit.objective - base_fit.objective
                p_val = _compute_lrt_pvalue(delta, df=1)
                results.append(StepResult(
                    covariate=cov,
                    parameter=par,
                    effect=eff,
                    direction="forward",
                    delta_obj=delta,
                    p_value=p_val,
                    selected=False,
                ))

    # Sort by delta_obj ascending (largest drop first)
    results.sort(key=lambda r: r.delta_obj)

    # Select the best significant combo
    for r in results:
        if r.p_value < alpha and r.delta_obj < 0:
            r.selected = True
            break  # Only select the single best

    return results


def backward_elimination(
    full_fit: NLMIXRFit,
    data: dict[str, Any],
    covariates: Sequence[str],
    parameters: Sequence[str],
    effects: Sequence[str] = ("linear",),
    alpha: float = 0.01,
    fit_factory: Callable | None = None,
) -> list[StepResult]:
    """Backward elimination step: test removing each covariate-parameter combo.

    For each covariate currently in the model, a model without it is fit.
    The covariate whose removal causes the smallest OFV increase is removed
    if the increase is not significant (p >= alpha, LRT df=1).

    Parameters
    ----------
    full_fit : NLMIXRFit
        The current full model fit.
    data : dict
        Dataset.
    covariates : list of str
        Covariate names currently in the model.
    parameters : list of str
        Parameter names the covariates are on.
    effects : tuple of str
        Effect types to consider.
    alpha : float
        Significance level for the LRT (default: 0.01).
    fit_factory : callable, optional
        Function(full_fit, data, covariate, parameter, effect) -> NLMIXRFit.
        Returns a fit with the specified covariate removed.

    Returns
    -------
    list of StepResult
        All tested removals, sorted by delta_obj ascending.
    """
    if fit_factory is None:
        fit_factory = _default_fit_factory

    results: list[StepResult] = []

    for cov in covariates:
        for par in parameters:
            for eff in effects:
                reduced_fit = fit_factory(full_fit, data, cov, par, eff)
                # delta_obj is positive when removal increases objective
                delta = reduced_fit.objective - full_fit.objective
                # For backward: statistic is the increase (delta itself)
                # p-value tests if the covariate is significant
                if delta > 0:
                    p_val = _chi2_sf(delta, 1)
                else:
                    p_val = 1.0  # removing it actually helped; definitely remove

                results.append(StepResult(
                    covariate=cov,
                    parameter=par,
                    effect=eff,
                    direction="backward",
                    delta_obj=delta,
                    p_value=p_val,
                    selected=False,
                ))

    # Sort by delta_obj ascending (smallest increase first)
    results.sort(key=lambda r: r.delta_obj)

    # Select the covariate whose removal is least harmful and not significant
    for r in results:
        if r.p_value >= alpha:
            # Removal is not significant -> safe to remove
            r.selected = True
            break

    return results


def stepwise_covariate_search(
    base_fit: NLMIXRFit,
    data: dict[str, Any],
    covariates: Sequence[str],
    parameters: Sequence[str],
    effects: Sequence[str] = ("linear",),
    forward_alpha: float = 0.05,
    backward_alpha: float = 0.01,
    max_steps: int = 10,
    fit_factory: Callable | None = None,
) -> list[StepResult]:
    """Full stepwise covariate selection alternating forward and backward steps.

    Alternates forward addition and backward elimination until no more
    changes are made or *max_steps* is reached.

    Parameters
    ----------
    base_fit : NLMIXRFit
        Starting model fit.
    data : dict
        Dataset.
    covariates : list of str
        Candidate covariate names.
    parameters : list of str
        Model parameter names to test.
    effects : tuple of str
        Effect types (default: ("linear",)).
    forward_alpha : float
        Alpha for forward addition LRT (default: 0.05).
    backward_alpha : float
        Alpha for backward elimination LRT (default: 0.01).
    max_steps : int
        Maximum number of forward+backward iterations (default: 10).
    fit_factory : callable, optional
        Custom fit factory for testing.

    Returns
    -------
    list of StepResult
        All results from all steps (both forward and backward).
    """
    if not covariates or not parameters:
        return []

    all_results: list[StepResult] = []
    current_fit = base_fit
    included_covs: set[tuple[str, str, str]] = set()  # (cov, par, eff)
    remaining_covs = list(covariates)
    steps = 0

    while steps < max_steps:
        # --- Forward addition ---
        if remaining_covs:
            fwd_results = forward_addition(
                current_fit, data,
                covariates=remaining_covs,
                parameters=parameters,
                effects=effects,
                alpha=forward_alpha,
                fit_factory=fit_factory,
            )
            all_results.extend(fwd_results)

            fwd_selected = [r for r in fwd_results if r.selected]
            if not fwd_selected:
                break  # No significant addition found, stop

            best = fwd_selected[0]
            included_covs.add((best.covariate, best.parameter, best.effect))
            # Remove the selected covariate from remaining candidates
            remaining_covs = [c for c in remaining_covs if c != best.covariate]

            # Update current fit: use the factory to get the new fit
            factory = fit_factory or _default_fit_factory
            current_fit = factory(current_fit, data, best.covariate, best.parameter, best.effect)
            steps += 1
        else:
            break

        if steps >= max_steps:
            break

        # --- Backward elimination ---
        if included_covs:
            incl_covs_list = list(included_covs)
            bwd_covariates = list({c for c, p, e in incl_covs_list})
            bwd_parameters = list({p for c, p, e in incl_covs_list})

            bwd_results = backward_elimination(
                current_fit, data,
                covariates=bwd_covariates,
                parameters=bwd_parameters,
                effects=effects,
                alpha=backward_alpha,
                fit_factory=fit_factory,
            )
            all_results.extend(bwd_results)

            bwd_selected = [r for r in bwd_results if r.selected]
            if bwd_selected:
                removed = bwd_selected[0]
                key = (removed.covariate, removed.parameter, removed.effect)
                included_covs.discard(key)
                if removed.covariate not in remaining_covs:
                    remaining_covs.append(removed.covariate)

                factory = fit_factory or _default_fit_factory
                current_fit = factory(
                    current_fit, data,
                    removed.covariate, removed.parameter, removed.effect,
                )
                steps += 1

    return all_results
