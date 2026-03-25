# Diagnostics Guide

This guide covers post-fit diagnostic tools available in nlmixr2-python:
residual analysis, goodness-of-fit plots, and visual predictive checks.

## Prerequisites

```python
import nlmixr2
import jax.numpy as jnp
import numpy as np
```

For plotting you will additionally need:

```bash
pip install matplotlib scipy
```

---

## 1. Overview of Diagnostics

After fitting a model, standard pharmacometric diagnostics help assess model
adequacy:

| Diagnostic         | Purpose                                           |
|--------------------|---------------------------------------------------|
| PRED / IPRED       | Population vs. individual predictions             |
| RES / IRES         | Population vs. individual residuals               |
| WRES               | Weighted residuals                                |
| CWRES              | Conditional weighted residuals                    |
| NPDE               | Normalised prediction distribution errors         |
| Shrinkage          | How much individual estimates collapse toward pop. |
| GOF plots          | Four-panel graphical assessment                   |
| VPC                | Visual predictive check                           |

---

## 2. Residuals

### 2.1 Predictions and Residuals from a Fit

After fitting, `fit.predictions` already contains per-observation arrays:

```python
# Assume `fit` is an NLMIXRFit from nlmixr2()
preds = fit.predictions

# Available keys: id, time, dv, pred, ipred, res, ires
dv    = np.asarray(preds["dv"])
pred  = np.asarray(preds["pred"])
ipred = np.asarray(preds["ipred"])
res   = np.asarray(preds["res"])     # DV - PRED
ires  = np.asarray(preds["ires"])    # DV - IPRED
```

### 2.2 Weighted Residuals (WRES)

`compute_wres` divides population residuals by the residual standard deviation:

```python
sigma = 0.7  # residual SD (from your error model)

wres = nlmixr2.compute_wres(
    dv=jnp.array(dv),
    pred=jnp.array(pred),
    sigma=sigma,
)
# WRES_i = (DV_i - PRED_i) / sigma
```

### 2.3 Conditional Weighted Residuals (CWRES)

`compute_cwres` uses individual predictions instead of population predictions:

```python
cwres = nlmixr2.compute_cwres(
    dv=jnp.array(dv),
    pred=jnp.array(pred),     # kept for API consistency
    ipred=jnp.array(ipred),
    sigma=sigma,
)
# CWRES_i = (DV_i - IPRED_i) / sigma
```

CWRES should be approximately standard normal (mean ~ 0, SD ~ 1) if the model
is well-specified. Trends in CWRES vs. time or vs. PRED indicate model
misspecification.

### 2.4 Normalised Prediction Distribution Errors (NPDE)

NPDE requires simulated datasets. For each observation, the fraction of
simulations below the observed value is transformed to a standard-normal
quantile:

```python
# Simulate multiple replicates (shape: n_sim x n_obs)
# Each row is a set of simulated DV values across all observations
n_sim = 500
simulated_dvs = jnp.array(...)  # shape (n_sim, n_obs)

npde = nlmixr2.compute_npde(
    dv=jnp.array(dv),
    simulated_dvs=simulated_dvs,
)
# npde values should follow N(0, 1) if model is correct
```

---

## 3. Shrinkage

Eta shrinkage measures how much individual parameter estimates collapse toward
the population mean. High shrinkage (> 30%) suggests the data may not
support reliable individual-level estimation for that parameter.

```python
# From a fit object:
print("Shrinkage:", fit.shrinkage)
# e.g., {'tka': 0.12, 'tcl': 0.05, 'tv': 0.35}

# Or compute directly:
etas = jnp.array(fit.etas["values"])    # shape (n_subjects, n_etas)
omega = jnp.eye(3) * jnp.array([0.5, 0.3, 0.2])  # your omega matrix

shrinkage = nlmixr2.compute_shrinkage(etas, omega)
# shrinkage_i = 1 - var(etas[:, i]) / omega[i, i]
```

---

## 4. Goodness-of-Fit (GOF) Plots

### 4.1 Prepare GOF Data

Bundle the diagnostic arrays into a `GOFData` container:

```python
gof = nlmixr2.gof_data(
    dv=dv,
    pred=pred,
    ipred=ipred,
    res=res,
    ires=ires,
    cwres=np.asarray(cwres),
    time=np.asarray(preds["time"]),
)
```

### 4.2 Render the Four-Panel Plot

`plot_gof` produces a standard four-panel figure:

1. **DV vs PRED** -- with identity line
2. **DV vs IPRED** -- with identity line
3. **CWRES vs TIME** -- should scatter randomly around zero
4. **CWRES QQ plot** -- should follow the diagonal if normally distributed

```python
from nlmixr2.plots import plot_gof

fig = plot_gof(gof, figsize=(12, 10))
fig.savefig("gof_plots.png", dpi=150)
```

### 4.3 Individual Fit Plots

View per-subject observed data overlaid with PRED and IPRED curves:

```python
from nlmixr2.plots import plot_individual

ind_data = nlmixr2.individual_data(
    data={"id": np.asarray(preds["id"]),
          "time": np.asarray(preds["time"]),
          "dv": dv},
    pred=pred,
    ipred=ipred,
    subject_ids=np.unique(np.asarray(preds["id"])),
)

fig = plot_individual(ind_data, n_cols=4)
fig.savefig("individual_fits.png", dpi=150)
```

### 4.4 Eta vs. Covariate Plots

Scatter plots of random effects against covariates can reveal unmodeled
covariate relationships:

```python
from nlmixr2.plots import plot_eta_cov

ecd = nlmixr2.eta_vs_cov_data(
    etas=np.asarray(fit.etas["values"]),       # (n_subj, n_etas)
    covariates=np.column_stack([               # (n_subj, n_covs)
        subject_weights,
        subject_ages,
    ]),
    eta_names=["eta.ka", "eta.cl", "eta.v"],
    cov_names=["WT", "AGE"],
)

fig = plot_eta_cov(ecd)
fig.savefig("eta_vs_cov.png", dpi=150)
```

A clear trend in these plots (e.g., eta.cl vs. WT) suggests adding that
covariate to the model.

---

## 5. Visual Predictive Check (VPC)

### 5.1 What is a VPC?

A VPC compares the distribution of observed data against simulated data from
the model. For each time bin, the observed percentiles (e.g., 5th, 50th,
95th) are overlaid with prediction intervals from many simulated datasets.

### 5.2 Generate VPC Data

```python
vpc_result = nlmixr2.vpc(
    fit_or_model_func=model_func,  # callable (params, times) -> predictions
    data={"id": jnp.array(data["id"]),
          "time": jnp.array(data["time"]),
          "dv": jnp.array(data["dv"])},
    n_sim=200,                     # number of simulation replicates
    omega=omega_matrix,            # between-subject covariance
    sigma=1.0,                     # residual error variance
    seed=42,                       # for reproducibility
    pi=(0.05, 0.5, 0.95),         # prediction interval quantiles
    bin_method="time",             # "time" (equal width) or "ntile" (equal count)
    n_bins=10,
)
```

### 5.3 Inspect VPC Result

The `VPCResult` contains:

```python
# Observed data
print(vpc_result.observed.keys())      # {"time", "dv"}

# Simulated quantiles (at bin centers)
print(vpc_result.simulated_quantiles.keys())  # {"time", "lo", "median", "hi"}

# Metadata
print(vpc_result.pi)      # (0.05, 0.5, 0.95)
print(vpc_result.n_sim)   # 200
```

### 5.4 Plot the VPC

```python
from nlmixr2.plots import plot_vpc

fig = plot_vpc(vpc_result, title="Visual Predictive Check - Warfarin")
fig.savefig("vpc.png", dpi=150)
```

The plot shows:

- **Scatter points**: observed data
- **Shaded band**: 5th-95th percentile prediction interval from simulations
- **Solid line**: simulated median
- **Dashed line**: reference (observed median)

### 5.5 Binning Strategies

Control how time points are grouped:

```python
# Equal-width time bins (default)
centers = nlmixr2.bin_times(times, method="time", n_bins=10)

# Equal-count (quantile) bins
centers = nlmixr2.bin_times(times, method="ntile", n_bins=10)
```

### 5.6 Custom Quantile Computation

For advanced use, compute quantiles from your own simulation matrix:

```python
# simulated_dvs shape: (n_sim, n_bins)
quantiles = nlmixr2.compute_quantiles(
    simulated_dvs,
    quantiles=(0.05, 0.5, 0.95),
)
# Returns dict: {0.05: array, 0.5: array, 0.95: array}
```

---

## 6. Information Criteria

### 6.1 AIC and BIC

```python
aic = nlmixr2.compute_aic(objective=fit.objective, n_params=fit.parameter_count)
bic = nlmixr2.compute_bic(
    objective=fit.objective,
    n_params=fit.parameter_count,
    n_obs=fit.n_observations,
)

print(f"AIC: {aic:.4f}")
print(f"BIC: {bic:.4f}")
```

Formulas:

- AIC = objective + 2 * n_params
- BIC = objective + n_params * ln(n_obs)

### 6.2 Condition Number

The condition number of the Hessian matrix indicates parameter estimability.
Values above 1000 suggest potential issues:

```python
if fit.covariance_result is not None:
    print("Condition number:", fit.covariance_result.condition_number)
```

Or compute from a Hessian matrix directly:

```python
cond = nlmixr2.compute_condition_number(hessian_matrix)
```

---

## 7. Complete Diagnostics Workflow

```python
import nlmixr2
import numpy as np
from nlmixr2.plots import plot_gof, plot_vpc

# Assume `fit` is a completed NLMIXRFit
preds = fit.predictions

# 1. Compute CWRES
import jax.numpy as jnp
cwres = nlmixr2.compute_cwres(
    dv=jnp.array(preds["dv"]),
    pred=jnp.array(preds["pred"]),
    ipred=jnp.array(preds["ipred"]),
    sigma=0.7,
)

# 2. Build GOF data and plot
gof = nlmixr2.gof_data(
    dv=np.asarray(preds["dv"]),
    pred=np.asarray(preds["pred"]),
    ipred=np.asarray(preds["ipred"]),
    res=np.asarray(preds["res"]),
    ires=np.asarray(preds["ires"]),
    cwres=np.asarray(cwres),
    time=np.asarray(preds["time"]),
)
fig = plot_gof(gof)
fig.savefig("diagnostics_gof.png", dpi=150)

# 3. VPC
vpc_result = nlmixr2.vpc(
    fit_or_model_func=model_func,
    data={"id": jnp.array(data["id"]),
          "time": jnp.array(data["time"]),
          "dv": jnp.array(data["dv"])},
    n_sim=200,
    seed=42,
)
fig = plot_vpc(vpc_result)
fig.savefig("diagnostics_vpc.png", dpi=150)

# 4. Summary
print(nlmixr2.summarize_fit(fit))
```
