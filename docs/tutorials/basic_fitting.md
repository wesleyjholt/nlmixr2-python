# Basic Model Fitting with nlmixr2-python

This tutorial walks through a complete pharmacometric workflow: loading data,
defining a nonlinear mixed-effects model, fitting it, and inspecting the
results.

## Prerequisites

```python
import nlmixr2
```

---

## 1. Load Example Data

The built-in `theo_sd` dataset contains single-dose oral theophylline PK data
for 12 subjects with approximately 10 observations each.

```python
data = nlmixr2.theo_sd()

# data is a dict of lists with keys: id, time, dv, amt, evid, wt
print("Columns:", list(data.keys()))
print("Number of records:", len(data["id"]))
print("Unique subjects:", len(set(data["id"])))
```

You can also use `load_dataset` by name:

```python
data = nlmixr2.load_dataset("theo_sd")

# See all available datasets
print(nlmixr2.list_datasets())
# ['pheno_sd', 'theo_sd', 'warfarin']
```

---

## 2. Define the Model

An nlmixr2 model has two blocks:

- **`ini()`** -- initial parameter estimates (fixed effects, bounds, fixed flags)
- **`model()`** -- structural model statements

### 2.1 The ini() Block

`ini()` accepts a dictionary mapping parameter names to initial values. Values
can be:

- A plain number: `"tka": 0.5`
- A list of `[lower, estimate]` or `[lower, estimate, upper]`:
  `"tka": [0.01, 0.5, 5.0]`
- A dict with explicit keys:
  `"tka": {"estimate": 0.5, "lower": 0.01, "upper": 5.0, "fixed": False}`

```python
ini_block = nlmixr2.ini({
    "tka":    [0.01, 0.5, 5.0],   # absorption rate constant (log-domain)
    "tcl":    [0.001, 0.04, 1.0], # clearance
    "tv":     [0.1, 0.5, 10.0],   # volume of distribution (L/kg)
    "eta.ka": 0.6,                 # IIV on ka (variance)
    "eta.cl": 0.3,                 # IIV on cl
    "eta.v":  0.1,                 # IIV on v
    "add.sd": 0.7,                 # additive residual error SD
})
```

### 2.2 The model() Block

`model()` accepts either a list of statement strings or a dict of
`{lhs: rhs}` assignments. Statements define:

- Derived parameters (e.g., exponentiated thetas + etas)
- ODE equations using `d/dt(compartment) = ...` syntax
- The dependent variable relationship and error model

```python
model_block = nlmixr2.model([
    "ka = exp(tka + eta.ka)",
    "cl = exp(tcl + eta.cl)",
    "v  = exp(tv  + eta.v)",
    "d/dt(depot)   = -ka * depot",
    "d/dt(central) =  ka * depot - cl/v * central",
    "cp = central / v",
    "cp ~ add(add.sd)",
])
```

Alternatively, use the dict form for simple assignments:

```python
model_block = nlmixr2.model({
    "ka": "exp(tka + eta.ka)",
    "cl": "exp(tcl + eta.cl)",
    "v":  "exp(tv + eta.v)",
})
```

> **Note:** The dict form does not support ODE or error-model statements.
> Use the list form for complete models.

---

## 3. Combine into an NLMIXRModel

The `nlmixr2()` function accepts the ini and model blocks bundled in an
`NLMIXRModel`:

```python
from nlmixr2.api import NLMIXRModel

mod = NLMIXRModel(ini=ini_block, model=model_block)
```

---

## 4. Fit the Model with FOCE

Pass the model, data, and `est="foce"` to `nlmixr2()`:

```python
fit = nlmixr2.nlmixr2(mod, data, est="foce")
```

The function parses the model, sets up the ODE system, and runs the
First-Order Conditional Estimation algorithm. It returns an `NLMIXRFit`
object.

### 4.1 Control Options

Fine-tune estimation via the `control` argument or a `FoceiControl` object:

```python
from nlmixr2 import FoceiControl

ctrl = FoceiControl(
    maxiter=500,      # maximum outer iterations
    tol=1e-6,         # convergence tolerance
    lr=0.01,          # learning rate for fixed effects
    lr_eta=0.05,      # learning rate for etas
    inner_steps=10,   # inner optimisation steps per iteration
    sigma=1.0,        # residual error variance
    print_progress=True,
)

fit = nlmixr2.nlmixr2(mod, data, est="foce", control=ctrl.to_dict())
```

Or pass a plain dict:

```python
fit = nlmixr2.nlmixr2(mod, data, est="foce", control={"maxiter": 1000})
```

---

## 5. Inspect the Fit

### 5.1 Scalar Summaries

```python
print("Estimator:", fit.estimator)         # "foce"
print("Objective:", fit.objective)          # -2 log-likelihood
print("AIC:      ", fit.aic)
print("BIC:      ", fit.bic)
print("N obs:    ", fit.n_observations)
print("N params: ", fit.parameter_count)
print("Elapsed:  ", fit.elapsed_seconds, "seconds")
```

### 5.2 Individual Random Effects (Etas)

The `fit.etas` dict contains the per-subject random effects matrix:

```python
etas = fit.etas
# etas["values"] is a JAX array of shape (n_subjects, n_params)
print("Eta shape:", etas["values"].shape)
```

### 5.3 Predictions

Population (PRED) and individual (IPRED) predictions, plus residuals, are
stored in `fit.predictions`:

```python
preds = fit.predictions
# preds is a dict with keys: id, time, dv, pred, ipred, res, ires
print("PRED range:", float(preds["pred"].min()), "-", float(preds["pred"].max()))
print("IPRED range:", float(preds["ipred"].min()), "-", float(preds["ipred"].max()))
```

### 5.4 Shrinkage

Eta shrinkage per parameter is available as a dict:

```python
print("Shrinkage:", fit.shrinkage)
# e.g. {'tka': 0.12, 'tcl': 0.05, 'tv': 0.08, ...}
```

### 5.5 Covariance / Standard Errors

If the covariance step succeeds, `fit.covariance_result` contains:

```python
if fit.covariance_result is not None:
    cov = fit.covariance_result
    print("Standard errors:", cov.standard_errors)
    print("RSE (%):", cov.rse)
    print("Condition number:", cov.condition_number)
```

### 5.6 Full Parameter Estimates

```python
# Fixed-effect estimates from the ini block
for name, init_val in fit.model.ini.values.items():
    print(f"  {name}: {init_val.estimate}")

# Final estimates from the fit table
print("Final fixed params:", fit.table.get("fixed_params"))
print("Converged:", fit.table.get("converged"))
print("Iterations:", fit.table.get("n_iterations"))
```

---

## 6. Summarize the Fit

The `summarize_fit()` function produces a formatted text summary analogous to
R's `print(fit)`:

```python
summary = nlmixr2.summarize_fit(fit)
print(summary)
```

Output:

```
nlmixr2 Fit Summary
========================================
Estimator:       foce
Observations:    120
Parameters:      7

Objective:       123.4567
AIC:             137.4567
BIC:             148.2345

Parameter Estimates
----------------------------------------
  tka              0.5000
  tcl              0.0400
  tv               0.5000
  eta.ka           0.6000
  eta.cl           0.3000
  eta.v            0.1000
  add.sd           0.7000
```

You can also pass prediction data to include shrinkage in the summary:

```python
summary = nlmixr2.summarize_fit(fit, predictions={
    "shrinkage": shrinkage_array,
    "shrinkage_labels": ["eta.ka", "eta.cl", "eta.v"],
})
```

---

## 7. Switching Estimators

### 7.1 SAEM

Stochastic Approximation Expectation-Maximization:

```python
fit_saem = nlmixr2.nlmixr2(mod, data, est="saem")
```

With SAEM-specific control:

```python
from nlmixr2 import SaemControl

ctrl = SaemControl(
    n_burn=300,         # burn-in iterations
    n_em=200,           # EM iterations
    n_chains=3,         # parallel chains
)

fit_saem = nlmixr2.nlmixr2(mod, data, est="saem", control=ctrl.to_dict())
```

### 7.2 FOCEI

First-Order Conditional Estimation with Interaction:

```python
fit_focei = nlmixr2.nlmixr2(mod, data, est="focei")
```

FOCEI uses the same `FoceiControl` structure as FOCE:

```python
fit_focei = nlmixr2.nlmixr2(mod, data, est="focei", control={"maxiter": 1000})
```

### 7.3 Other Estimators

```python
# Nonlinear minimisation
fit_nlm = nlmixr2.nlmixr2(mod, data, est="nlm")

# Post-hoc estimation (fix thetas, re-estimate etas)
fit_posthoc = nlmixr2.nlmixr2(mod, data, est="posthoc")
```

---

## 8. Comparing Estimators

After fitting with multiple methods, compare them:

```python
table = nlmixr2.compare_fits(
    [fit, fit_saem, fit_focei],
    names=["FOCE", "SAEM", "FOCEI"],
)

print(nlmixr2.format_comparison(table))
# Prints a table with Objective, AIC, BIC, nPar, nObs
# and identifies the best model by AIC and BIC
```

---

## 9. Saving and Exporting

Persist the fit to a JSON file:

```python
fit = nlmixr2.nlmixr2(mod, data, est="foce", save=True)
# Writes nlmixr2-fit.json in the current directory

# Or specify a custom path:
fit = nlmixr2.nlmixr2(mod, data, est="foce", save="results/theo_foce.json")
```

Convert the fit to a plain dictionary for further processing:

```python
fit_dict = fit.to_dict()
```

---

## Complete Example

```python
import nlmixr2
from nlmixr2.api import NLMIXRModel

# 1. Load data
data = nlmixr2.theo_sd()

# 2. Define model
ini_block = nlmixr2.ini({
    "tka":    [0.01, 0.5, 5.0],
    "tcl":    [0.001, 0.04, 1.0],
    "tv":     [0.1, 0.5, 10.0],
    "eta.ka": 0.6,
    "eta.cl": 0.3,
    "eta.v":  0.1,
    "add.sd": 0.7,
})

model_block = nlmixr2.model([
    "ka = exp(tka + eta.ka)",
    "cl = exp(tcl + eta.cl)",
    "v  = exp(tv  + eta.v)",
    "d/dt(depot)   = -ka * depot",
    "d/dt(central) =  ka * depot - cl/v * central",
    "cp = central / v",
    "cp ~ add(add.sd)",
])

mod = NLMIXRModel(ini=ini_block, model=model_block)

# 3. Fit with FOCE
fit = nlmixr2.nlmixr2(mod, data, est="foce")

# 4. Inspect results
print(f"Objective: {fit.objective:.4f}")
print(f"AIC: {fit.aic:.4f}")
print(f"Etas shape: {fit.etas['values'].shape}")

# 5. Summarize
print(nlmixr2.summarize_fit(fit))

# 6. Try SAEM
fit_saem = nlmixr2.nlmixr2(mod, data, est="saem")

# 7. Compare
table = nlmixr2.compare_fits([fit, fit_saem], names=["FOCE", "SAEM"])
print(nlmixr2.format_comparison(table))
```
