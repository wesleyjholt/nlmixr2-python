# nlmixr2-python: Remaining Implementation Checklist

## Tier 1: Critical (Blocks Real Use)

### Estimation
- [x] Validate SAEM on real datasets (theo_sd, warfarin) against R reference output
- [x] Compute Hessian & standard errors via JAX autodiff on FOCE/SAEM objectives
- [x] Add covariance/correlation matrices to fit objects
- [x] Enforce parameter bounds from ini block during optimization
- [x] Implement posthoc/empirical Bayes estimation (`est="posthoc"`) for per-subject etas
- [x] Implement NLME estimator (`est="nlme"`)
- [x] Implement NLM/Nelder-Mead estimator (`est="nlm"`)
- [x] Improve FOCE/SAEM optimizer robustness (adaptive step sizes, convergence monitoring)

### ODE Solver / rxode2 Features
- [x] Transit compartment support (chain of absorption compartments)
- [x] Lag time (`tlag`) in dosing
- [x] Bioavailability modifier (`F`) per compartment
- [x] RATE/DUR modifiers from data
- [x] Steady-state solving (automatic dose repetition to SS)
- [x] ADDL/II expansion in ODE solver (currently stored but not used)
- [x] 3-compartment analytical (linCmt) solutions
- [x] Sensitivity equations for gradient-based optimization

### Model Features
- [x] Censoring/BLQ handling (CENS/LIMIT columns, censored likelihood)
- [x] Inter-occasion variability (IOV) — separate omegas per occasion
- [x] Time-varying covariate interpolation during ODE integration
- [x] Categorical/count endpoints (Poisson, binomial, ordinal)
- [x] Mixture (latent class) models
- [x] Parameter fixing (`fixed=TRUE` in ini block)
- [x] FOCEi (interaction) — distinct from FOCE

## Tier 2: Post-Estimation & Diagnostics

### Residual Metrics
- [x] CWRES (conditional weighted residuals)
- [x] NPDE (normalised prediction distribution errors)
- [x] Proper WRES using full covariance matrix

### Fit Object Enrichment
- [x] Auto-compute AIC/BIC in fit output
- [x] Per-subject eta estimates in output table
- [x] Per-subject PRED/IPRED in output table
- [x] Timing/convergence info in fit
- [x] Per-subject log-likelihood (phi)
- [x] Integrate shrinkage into standard fit summary

### Plotting / Visualization
- [x] GOF plots (PRED vs DV, IPRED vs DV, residuals vs time, QQ)
- [x] Individual subject plots (observed + predicted overlays)
- [x] ETA vs covariate scatter plots
- [x] Trace plots (parameter values during estimation)
- [x] VPC plot rendering (currently only data generation)

## Tier 3: Workflow & Utilities

### Model Operations
- [x] Model update/piping (modify and refit without rebuilding)
- [x] Model formula updates (add/remove terms)
- [x] Prior specification for Bayesian estimation

### Covariate Selection
- [x] `covarSearchAuto()` — automated stepwise covariate selection
- [x] Forward addition / backward elimination
- [x] Covariate significance testing

### Bootstrap & Uncertainty
- [x] `bootstrapFit()` — full bootstrap refitting with resampled data
- [x] Nonparametric and parametric bootstrap variants
- [x] Bootstrap confidence intervals on all parameters

### Objective Function
- [x] Multiple OFV types (FOCE, FOCEi, Laplacian)
- [x] Likelihood weighting per observation

## Tier 4: Data Handling

- [x] Full NONMEM $DATA semantics (SKIP, ACCEPT, REPLICATES)
- [x] Automatic dose reconstruction from NONMEM tables
- [x] RATE/DUR inference from AMT column
- [x] Proper ADDL/II dose expansion before fitting
- [x] Example datasets package (theo_sd, warfarin, pk.oral1comp, etc.)

## Tier 5: Validation & Testing

- [x] Generate reference artifacts: warfarin FOCE, warfarin SAEM, pk.oral1comp (requires Gautschi cluster)
- [x] End-to-end integration tests: Python fit vs R fit parameter agreement
- [x] Objective function agreement tests (within tolerance)
- [x] Standard error agreement tests (once Hessian implemented)
- [x] Convergence stress tests on difficult models
- [x] Performance benchmarks (JAX vs R fitting times)

## Tier 6: Performance & Infrastructure

- [x] JIT-compile estimation loops for speed
- [x] Vectorized (vmap) subject processing
- [x] GPU acceleration for large datasets
- [x] Parallel bootstrap execution

## Tier 7: Documentation

- [x] Tutorial: basic model fitting workflow end-to-end
- [x] Tutorial: covariate modeling
- [x] Guide: diagnostic interpretation
- [x] API reference for all public functions
- [x] Migration guide: R nlmixr2 → Python
- [x] Convergence troubleshooting guide
- [x] Examples for each estimator (FOCE, SAEM)
