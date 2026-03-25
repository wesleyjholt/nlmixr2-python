# API Reference

Complete listing of all public modules and their exported functions and classes
in the `nlmixr2` Python package.

---

## nlmixr2.api -- Core API

| Name            | Kind      | Description                                                        |
|-----------------|-----------|--------------------------------------------------------------------|
| `ini()`         | function  | Create an IniBlock from a dict of parameter names to initial values |
| `model()`       | function  | Create a ModelBlock from a list of statement strings or a dict      |
| `nlmixr2()`     | function  | Fit a model to data with a specified estimator, or return a model spec |
| `InitValue`     | dataclass | Single parameter initial value with optional bounds and fixed flag  |
| `IniBlock`      | dataclass | Collection of named InitValue entries                              |
| `ModelBlock`    | dataclass | Tuple of model statement strings                                   |
| `NLMIXRModel`   | dataclass | Combined ini + model specification                                 |
| `NLMIXRFit`     | dataclass | Result of model fitting with all diagnostics and metadata          |

---

## nlmixr2.datasets -- Example Data

| Name              | Kind     | Description                                              |
|-------------------|----------|----------------------------------------------------------|
| `theo_sd()`       | function | Theophylline single-dose PK data (12 subjects, oral)     |
| `warfarin()`      | function | Warfarin PK data (32 subjects, with wt/age/sex covariates) |
| `pheno_sd()`      | function | Phenobarbital neonatal PK data (59 subjects, IV)         |
| `list_datasets()` | function | Return list of available dataset names                   |
| `load_dataset()`  | function | Load a dataset by name string                            |

---

## nlmixr2.data -- Data Utilities

| Name                 | Kind      | Description                                         |
|----------------------|-----------|-----------------------------------------------------|
| `validate_dataset()` | function  | Validate required columns and data types             |
| `get_doses()`        | function  | Extract dosing records (evid > 0) from a dataset     |
| `get_observations()` | function  | Extract observation records (evid == 0) from a dataset |
| `split_by_subject()` | function  | Split a dataset into per-subject dicts               |
| `merge_datasets()`   | function  | Merge two column-oriented datasets                   |
| `ValidatedDataset`   | dataclass | Container for a validated dataset with metadata      |

---

## nlmixr2.estimators -- Estimation Algorithms

| Name                 | Kind      | Description                                             |
|----------------------|-----------|---------------------------------------------------------|
| `estimate_foce()`    | function  | First-Order Conditional Estimation                      |
| `estimate_focei()`   | function  | First-Order Conditional Estimation with Interaction     |
| `estimate_saem()`    | function  | Stochastic Approximation EM algorithm                   |
| `estimate_nlm()`     | function  | Nonlinear minimisation estimator                        |
| `estimate_posthoc()` | function  | Post-hoc estimation (fix thetas, estimate etas)         |
| `EstimationResult`   | dataclass | Container for estimation output (objective, params, etas) |

---

## nlmixr2.control -- Estimator Configuration

| Name             | Kind      | Description                                           |
|------------------|-----------|-------------------------------------------------------|
| `FoceiControl`   | dataclass | Control parameters for FOCE/FOCEI (maxiter, tol, lr)  |
| `SaemControl`    | dataclass | Control parameters for SAEM (n_burn, n_em, n_chains)  |
| `MockControl`    | dataclass | Control parameters for mock estimator                  |
| `foceiControl()` | function  | Factory function returning FoceiControl as dict        |
| `saemControl()`  | function  | Factory function returning SaemControl as dict         |

---

## nlmixr2.diagnostics -- Post-Fit Diagnostics

| Name                       | Kind     | Description                                            |
|----------------------------|----------|--------------------------------------------------------|
| `summarize_fit()`          | function | Formatted text summary of an NLMIXRFit                 |
| `compute_predictions()`    | function | Compute PRED, IPRED, and residuals from a model func   |
| `compute_cwres()`          | function | Conditional weighted residuals: (DV - IPRED) / sigma   |
| `compute_npde()`           | function | Normalised prediction distribution errors from sims    |
| `compute_wres()`           | function | Weighted residuals: (DV - PRED) / sigma                |
| `compute_aic()`            | function | AIC = objective + 2 * n_params                         |
| `compute_bic()`            | function | BIC = objective + n_params * ln(n_obs)                 |
| `compute_condition_number()` | function | Ratio of largest to smallest eigenvalue of Hessian    |
| `compute_shrinkage()`      | function | Eta shrinkage: 1 - var(etas) / omega_diag              |

---

## nlmixr2.compare -- Model Comparison

| Name                      | Kind      | Description                                           |
|---------------------------|-----------|-------------------------------------------------------|
| `compare_fits()`          | function  | Compare multiple fits on objective, AIC, BIC          |
| `format_comparison()`     | function  | Format a ComparisonTable as a readable string         |
| `likelihood_ratio_test()` | function  | LRT between nested models with chi-squared p-value    |
| `bootstrap_comparison()`  | function  | Bootstrap CIs for objective function differences      |
| `ComparisonTable`         | dataclass | Tabular comparison of multiple model fits             |
| `LRTResult`               | dataclass | Result of a likelihood ratio test                     |

---

## nlmixr2.covariates -- Covariate Support

| Name                        | Kind      | Description                                          |
|-----------------------------|-----------|------------------------------------------------------|
| `add_covariate_to_model()`  | function  | Add a covariate effect (linear/power/exp) to a parsed model |
| `extract_covariates()`      | function  | Identify covariate names from model + data columns   |
| `center_covariates()`       | function  | Center covariate columns by specified values         |
| `detect_mu_referencing()`   | function  | Detect mu-referenced parameters for SAEM efficiency  |
| `CovariateModel`            | dataclass | Specification of covariates, centering, transforms   |

---

## nlmixr2.covar_search -- Stepwise Covariate Selection

| Name                          | Kind      | Description                                        |
|-------------------------------|-----------|----------------------------------------------------|
| `stepwise_covariate_search()` | function  | Full forward/backward stepwise covariate selection |
| `forward_addition()`         | function  | Single forward addition step (test all combos)     |
| `backward_elimination()`     | function  | Single backward elimination step                   |
| `StepResult`                  | dataclass | Result of one covariate search step                |

---

## nlmixr2.plots -- Diagnostic Plots

| Name                | Kind      | Description                                              |
|---------------------|-----------|----------------------------------------------------------|
| `gof_data()`        | function  | Bundle arrays into a GOFData container                   |
| `individual_data()` | function  | Split arrays into per-subject IndividualData list        |
| `eta_vs_cov_data()` | function  | Bundle eta and covariate matrices for scatter panels     |
| `traceplot_data()`  | function  | Bundle optimisation trace arrays                         |
| `plot_gof()`        | function  | Render four-panel GOF figure (requires matplotlib)       |
| `plot_individual()`  | function  | Render per-subject fit panels (requires matplotlib)     |
| `plot_eta_cov()`    | function  | Scatter grid of etas vs covariates (requires matplotlib) |
| `plot_vpc()`        | function  | Render VPC plot (requires matplotlib)                    |
| `GOFData`           | dataclass | Goodness-of-fit arrays (dv, pred, ipred, res, cwres)    |
| `IndividualData`    | dataclass | Per-subject time/dv/pred/ipred arrays                    |
| `EtaCovData`        | dataclass | Paired eta and covariate arrays                          |
| `TracePlotData`     | dataclass | Iteration-indexed objective and parameter history        |
| `VPCPlotData`       | dataclass | Pre-extracted VPC arrays for plotting                    |

---

## nlmixr2.vpc -- Visual Predictive Check

| Name                 | Kind      | Description                                            |
|----------------------|-----------|--------------------------------------------------------|
| `vpc()`              | function  | Generate VPC by simulating replicates and computing quantiles |
| `bin_times()`        | function  | Compute bin centers (equal-width or equal-count)       |
| `compute_quantiles()`| function  | Quantiles across simulation replicates                 |
| `VPCResult`          | dataclass | Container for VPC observed data and simulated quantiles |

---

## nlmixr2.omega -- Random Effects Matrices

| Name              | Kind      | Description                                              |
|-------------------|-----------|----------------------------------------------------------|
| `omega()`         | function  | Construct a named omega matrix from a dict specification |
| `sample_etas()`   | function  | Sample random effects from MVN(0, omega)                 |
| `OmegaBlock`      | dataclass | Named block-diagonal variance-covariance matrix          |

---

## nlmixr2.ode -- ODE Solver

| Name                     | Kind     | Description                                           |
|--------------------------|----------|-------------------------------------------------------|
| `solve_ode()`            | function | Solve ODE system with PK dosing events (Dopri5)      |
| `transit_compartments()` | function | Generate transit compartment chain RHS function       |

---

## nlmixr2.event_table -- Event Table Builder

| Name           | Kind  | Description                                               |
|----------------|-------|-----------------------------------------------------------|
| `et()`         | function | Create a new empty EventTable                           |
| `EventTable`   | class    | NONMEM-style event dataset builder with method chaining |

`EventTable` methods: `add_dosing()`, `add_sampling()`, `repeat()`,
`expand()`, `to_dict()`, `to_arrays()`

---

## nlmixr2.parser -- Model Parser

| Name            | Kind      | Description                                            |
|-----------------|-----------|--------------------------------------------------------|
| `parse_model()` | function  | Parse ModelBlock + IniBlock into a callable ParsedModel |
| `ParsedModel`   | dataclass | Parsed model with ODE func, output func, error model   |

---

## nlmixr2.lincmt -- Linear Compartment Models

| Name                 | Kind     | Description                                       |
|----------------------|----------|---------------------------------------------------|
| `linCmt()`           | function | General linear compartment model solution         |
| `one_cmt_bolus()`    | function | One-compartment IV bolus analytical solution      |
| `one_cmt_oral()`     | function | One-compartment oral analytical solution          |
| `one_cmt_infusion()` | function | One-compartment IV infusion analytical solution   |
| `two_cmt_bolus()`    | function | Two-compartment IV bolus analytical solution      |
| `two_cmt_oral()`     | function | Two-compartment oral analytical solution          |
| `three_cmt_bolus()`  | function | Three-compartment IV bolus analytical solution    |
| `three_cmt_oral()`   | function | Three-compartment oral analytical solution        |
| `superposition()`    | function | Superposition for multiple-dose concentration     |

---

## nlmixr2.simulate -- Simulation

| Name                | Kind      | Description                                          |
|---------------------|-----------|------------------------------------------------------|
| `simulate()`        | function  | Simulate PK/PD model with BSV and RUV                |
| `to_dataframe_dict()` | function | Convert SimulationResult to column-oriented dict    |
| `SimulationResult`  | dataclass | Container for simulation output (per-subject data)   |

---

## nlmixr2.bootstrap -- Bootstrap Methods

| Name                    | Kind      | Description                                        |
|-------------------------|-----------|----------------------------------------------------|
| `bootstrap_fit()`       | function  | Non-parametric bootstrap of a model fit            |
| `parametric_bootstrap()` | function | Parametric bootstrap from estimated distributions  |
| `resample_by_subject()` | function  | Resample dataset by subject with replacement       |
| `BootstrapResult`       | dataclass | Container for bootstrap parameter distributions    |

---

## nlmixr2.hessian -- Parameter Uncertainty

| Name                        | Kind      | Description                                      |
|-----------------------------|-----------|--------------------------------------------------|
| `compute_hessian()`         | function  | Compute Hessian matrix of objective function     |
| `compute_covariance()`      | function  | Invert Hessian to get parameter covariance       |
| `compute_standard_errors()` | function  | Square root of covariance diagonal               |
| `compute_correlation()`     | function  | Correlation matrix from covariance               |
| `compute_rse()`             | function  | Relative standard errors (SE / estimate)         |
| `CovarianceResult`          | dataclass | Container for Hessian, covariance, SE, RSE       |

---

## nlmixr2.sensitivity -- Sensitivity Analysis

| Name                          | Kind      | Description                                     |
|-------------------------------|-----------|-------------------------------------------------|
| `compute_fim()`               | function  | Fisher Information Matrix computation            |
| `solve_with_sensitivities()`  | function  | ODE solve with forward sensitivity equations     |
| `jax_adjoint_gradient()`      | function  | Adjoint-method gradient computation              |
| `SensitivityResult`           | dataclass | Container for sensitivity analysis output        |

---

## nlmixr2.update -- Model Modification

| Name                  | Kind     | Description                                          |
|-----------------------|----------|------------------------------------------------------|
| `update_ini()`        | function | Update initial parameter estimates in a model        |
| `update_model()`      | function | Replace model statements                             |
| `add_statement()`     | function | Append a statement to the model block                |
| `remove_parameter()`  | function | Remove a parameter from ini and model                |
| `refit()`             | function | Refit a model from an existing fit object            |

---

## nlmixr2.censoring -- Censored Data

| Name                               | Kind      | Description                                    |
|------------------------------------|-----------|------------------------------------------------|
| `apply_censoring()`                | function  | Apply censoring rules to a dataset             |
| `has_censoring()`                  | function  | Check if a dataset contains censored records   |
| `m3_method()`                      | function  | M3 method likelihood for BLQ data              |
| `censored_normal_log_likelihood()` | function  | Log-likelihood contribution for censored obs   |
| `CensoringSpec`                    | dataclass | Specification of censoring limits and type      |

---

## nlmixr2.iov -- Inter-Occasion Variability

| Name                       | Kind      | Description                                      |
|----------------------------|-----------|--------------------------------------------------|
| `apply_iov()`              | function  | Apply IOV random effects to a model              |
| `expand_omega_with_iov()`  | function  | Expand omega matrix for IOV parameters           |
| `extract_occasions()`      | function  | Extract occasion indicators from data            |
| `sample_iov_etas()`        | function  | Sample occasion-specific random effects          |
| `IOVSpec`                  | dataclass | Specification of IOV structure                   |

---

## nlmixr2.mixture -- Mixture Models

| Name                       | Kind      | Description                                      |
|----------------------------|-----------|--------------------------------------------------|
| `estimate_mixture()`       | function  | Fit a mixture model with multiple subpopulations |
| `classify_subjects()`      | function  | Assign subjects to mixture components            |
| `mixture_log_likelihood()` | function  | Log-likelihood for mixture models                |
| `MixtureSpec`              | dataclass | Specification of mixture components              |
| `MixtureResult`            | dataclass | Container for mixture model results              |

---

## nlmixr2.likelihoods -- Non-Normal Likelihoods

| Name                                | Kind     | Description                                  |
|-------------------------------------|----------|----------------------------------------------|
| `select_likelihood()`               | function | Auto-select likelihood based on data type    |
| `binomial_log_likelihood()`         | function | Binomial log-likelihood for binary data      |
| `poisson_log_likelihood()`          | function | Poisson log-likelihood for count data        |
| `negative_binomial_log_likelihood()`| function | Negative binomial log-likelihood             |
| `ordinal_log_likelihood()`          | function | Ordinal (cumulative logit) log-likelihood    |

---

## nlmixr2.steady_state -- Steady-State Utilities

| Name                     | Kind      | Description                                       |
|--------------------------|-----------|---------------------------------------------------|
| `find_steady_state()`    | function  | Find steady-state concentrations iteratively      |
| `steady_state_profile()` | function  | Compute concentration-time profile at steady state |
| `superposition_to_ss()`  | function  | Superposition method for steady-state computation |
| `SteadyStateResult`      | dataclass | Container for steady-state results                |

---

## nlmixr2.time_varying -- Time-Varying Covariates

| Name                        | Kind      | Description                                     |
|-----------------------------|-----------|-------------------------------------------------|
| `extract_time_varying()`    | function  | Extract time-varying covariate columns from data |
| `interpolate_covariate()`   | function  | Interpolate covariate values at arbitrary times  |
| `build_covariate_function()`| function  | Build a callable covariate interpolation function |
| `TimeVaryingCovariate`      | dataclass | Specification of a time-varying covariate        |

---

## nlmixr2.install -- Installation Utilities

| Name                    | Kind     | Description                                         |
|-------------------------|----------|-----------------------------------------------------|
| `nlmixr2CheckInstall()` | function | Verify that all dependencies are installed          |
| `nlmixr2deps()`         | function | List package dependencies                           |
| `nlmixr2packages()`     | function | List installed nlmixr2-related packages             |
| `nlmixr2conflicts()`    | function | Check for conflicting package versions              |
| `nlmixr2update()`       | function | Check for available updates                         |
