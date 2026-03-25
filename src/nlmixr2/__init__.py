"""Initial Python replication harness for nlmixr2."""

from .api import ini, model, nlmixr2
from .install import (
    nlmixr2CheckInstall,
    nlmixr2conflicts,
    nlmixr2deps,
    nlmixr2packages,
    nlmixr2update,
)
from .compare import (
    ComparisonTable,
    LRTResult,
    bootstrap_comparison,
    compare_fits,
    format_comparison,
    likelihood_ratio_test,
)
from .control import FoceiControl, MockControl, SaemControl, foceiControl, saemControl
from .covariates import (
    CovariateModel,
    add_covariate_to_model,
    center_covariates,
    detect_mu_referencing,
    extract_covariates,
)
from .data import (
    ValidatedDataset,
    get_doses,
    get_observations,
    merge_datasets,
    split_by_subject,
    validate_dataset,
)
from .diagnostics import (
    compute_aic,
    compute_bic,
    compute_condition_number,
    compute_predictions,
    compute_shrinkage,
    summarize_fit,
)
from .bootstrap import BootstrapResult, bootstrap_fit, parametric_bootstrap, resample_by_subject
from .censoring import CensoringSpec, apply_censoring, censored_normal_log_likelihood, has_censoring, m3_method
from .covar_search import StepResult, backward_elimination, forward_addition, stepwise_covariate_search
from .datasets import list_datasets, load_dataset, pheno_sd, theo_sd, warfarin
from .diagnostics import compute_cwres, compute_npde, compute_wres
from .estimators import EstimationResult, estimate_foce, estimate_focei, estimate_laplacian, estimate_nlm, estimate_nlme, estimate_posthoc, estimate_saem, laplacian_objective
from .weighting import WeightingScheme, apply_weights, inverse_variance_weights, weighted_objective
from .event_table import EventTable, et
from .hessian import CovarianceResult, compute_covariance, compute_correlation, compute_hessian, compute_rse, compute_standard_errors
from .iov import IOVSpec, apply_iov, expand_omega_with_iov, extract_occasions, sample_iov_etas
from .likelihoods import binomial_log_likelihood, negative_binomial_log_likelihood, ordinal_log_likelihood, poisson_log_likelihood, select_likelihood
from .lincmt import (
    linCmt,
    one_cmt_bolus,
    one_cmt_infusion,
    one_cmt_oral,
    superposition,
    three_cmt_bolus,
    three_cmt_oral,
    two_cmt_bolus,
    two_cmt_oral,
)
from .mixture import MixtureResult, MixtureSpec, classify_subjects, estimate_mixture, mixture_log_likelihood
from .ode import solve_ode, transit_compartments
from .omega import OmegaBlock, omega, sample_etas
from .parser import ParsedModel, parse_model
from .plots import GOFData, IndividualData, EtaCovData, TracePlotData, VPCPlotData, gof_data, individual_data, eta_vs_cov_data, traceplot_data
from .sensitivity import SensitivityResult, compute_fim, jax_adjoint_gradient, solve_with_sensitivities
from .simulate import SimulationResult, simulate, to_dataframe_dict
from .steady_state import SteadyStateResult, find_steady_state, steady_state_profile, superposition_to_ss
from .time_varying import TimeVaryingCovariate, build_covariate_function, extract_time_varying, interpolate_covariate
from .update import add_statement, refit, remove_parameter, update_ini, update_model
from .vpc import VPCResult, bin_times, compute_quantiles, vpc

__all__ = [
    # Data classes
    "BootstrapResult", "CensoringSpec", "ComparisonTable", "CovarianceResult",
    "CovariateModel", "EstimationResult", "EventTable", "EtaCovData",
    "FoceiControl", "GOFData", "IOVSpec", "IndividualData", "LRTResult",
    "MixtureResult", "MixtureSpec", "MockControl", "OmegaBlock", "ParsedModel",
    "SaemControl", "SensitivityResult", "SimulationResult", "SteadyStateResult",
    "StepResult", "TimeVaryingCovariate", "TracePlotData", "VPCPlotData",
    "VPCResult", "ValidatedDataset",
    # Functions
    "add_covariate_to_model", "add_statement", "apply_censoring", "apply_iov",
    "backward_elimination", "bin_times", "binomial_log_likelihood",
    "bootstrap_comparison", "bootstrap_fit", "build_covariate_function",
    "center_covariates", "censored_normal_log_likelihood", "classify_subjects",
    "compare_fits", "compute_aic", "compute_bic", "compute_condition_number",
    "compute_correlation", "compute_covariance", "compute_cwres", "compute_fim",
    "compute_hessian", "compute_npde", "compute_predictions", "compute_quantiles",
    "compute_rse", "compute_shrinkage", "compute_standard_errors", "compute_wres",
    "detect_mu_referencing", "et", "estimate_foce", "estimate_focei",
    "estimate_laplacian", "estimate_mixture", "estimate_nlm", "estimate_nlme", "estimate_posthoc", "estimate_saem",
    "eta_vs_cov_data", "expand_omega_with_iov", "extract_covariates",
    "extract_occasions", "extract_time_varying", "find_steady_state",
    "foceiControl", "format_comparison", "forward_addition", "get_doses",
    "get_observations", "gof_data", "has_censoring", "individual_data", "ini",
    "interpolate_covariate", "jax_adjoint_gradient", "likelihood_ratio_test",
    "linCmt", "list_datasets", "load_dataset", "m3_method", "merge_datasets",
    "mixture_log_likelihood", "model", "negative_binomial_log_likelihood",
    "nlmixr2", "nlmixr2CheckInstall", "nlmixr2conflicts", "nlmixr2deps",
    "nlmixr2packages", "nlmixr2update", "omega", "one_cmt_bolus",
    "one_cmt_infusion", "one_cmt_oral", "ordinal_log_likelihood",
    "parametric_bootstrap", "parse_model", "pheno_sd", "poisson_log_likelihood",
    "refit", "remove_parameter", "resample_by_subject", "saemControl",
    "sample_etas", "sample_iov_etas", "select_likelihood", "simulate",
    "solve_ode", "solve_with_sensitivities", "split_by_subject",
    "steady_state_profile", "stepwise_covariate_search", "summarize_fit",
    "superposition", "superposition_to_ss", "theo_sd", "three_cmt_bolus",
    "three_cmt_oral", "to_dataframe_dict", "traceplot_data",
    "transit_compartments", "two_cmt_bolus", "two_cmt_oral", "update_ini",
    "update_model", "validate_dataset", "vpc", "warfarin",
    "WeightingScheme", "apply_weights", "inverse_variance_weights",
    "laplacian_objective", "weighted_objective",
]
