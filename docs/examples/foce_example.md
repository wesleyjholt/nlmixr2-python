# FOCE Fitting Example

This example demonstrates a complete FOCE (First-Order Conditional Estimation)
workflow using the built-in Theophylline single-dose dataset.

## Overview

FOCE approximates the marginal likelihood by linearizing the model around the
conditional mode of the random effects. It alternates between:

- **Inner problem:** Optimize per-subject random effects (etas) given current
  fixed-effect estimates.
- **Outer problem:** Optimize population fixed-effect parameters given current
  etas.

Gradients for both steps are computed automatically via JAX.

## Step 1: Load the data

```python
import nlmixr2

# Load the built-in Theophylline PK dataset
data = nlmixr2.theo_sd()

# Inspect structure
print("Columns:", list(data.keys()))
print("Subjects:", len(set(data["id"])))
print("Total rows:", len(data["id"]))
```

The `theo_sd()` dataset contains columns: `id`, `time`, `dv`, `amt`, `evid`,
`wt`. It has 12 subjects with a single oral dose of Theophylline and roughly
10 observations per subject.

## Step 2: Define the model

```python
# Initial parameter estimates for a one-compartment oral PK model
ini_block = nlmixr2.ini({
    "A":  300.0,    # dose/volume scaling (approximate)
    "ke": 0.08,     # elimination rate constant (1/hr)
    "ka": 1.5,      # absorption rate constant (1/hr)
})

# Model block: algebraic one-compartment oral model
model_block = nlmixr2.model([
    "conc = A * ka / (ka - ke) * (exp(-ke * t) - exp(-ka * t))",
    "conc ~ add(add.err)",
])

# Combine into a full model specification
model_spec = nlmixr2.NLMIXRModel(ini=ini_block, model=model_block)
```

## Step 3: Fit with FOCE

```python
fit = nlmixr2.nlmixr2(
    model_spec,
    data=data,
    est="foce",
    control={
        "maxiter": 500,
        "lr": 0.01,
        "lr_eta": 0.05,
        "tol": 1e-6,
        "inner_steps": 10,
        "sigma": 1.0,
    },
)
```

### Control options explained

| Parameter | Default | Description |
|---|---|---|
| `maxiter` | 100 | Maximum outer iterations |
| `lr` | 0.01 | Learning rate for fixed-effect updates (Adam optimizer) |
| `lr_eta` | 0.01 | Learning rate for random-effect updates |
| `tol` | 1e-4 | Convergence tolerance on relative objective change |
| `inner_steps` | 10 | Number of eta-optimization steps per outer iteration |
| `sigma` | 1.0 | Residual error variance |

## Step 4: Inspect results

### Convergence status

```python
print("Converged:", fit.table["converged"])
print("Iterations:", fit.table["n_iterations"])
print("Elapsed:", f"{fit.elapsed_seconds:.1f}s")
```

### Objective function and information criteria

```python
print(f"Objective (-2LL): {fit.objective:.4f}")
print(f"AIC:              {fit.aic:.4f}")
print(f"BIC:              {fit.bic:.4f}")
```

### Fixed-effect estimates

```python
for name, value in fit.table["fixed_params"].items():
    print(f"  {name}: {value:.6f}")
```

### Random effects and shrinkage

```python
import jax.numpy as jnp

etas = fit.etas["values"]
print("Eta shape:", etas.shape)         # (n_subjects, n_params)
print("Eta means:", jnp.mean(etas, axis=0))
print("Eta SDs:", jnp.std(etas, axis=0))

print("\nShrinkage:")
for name, val in fit.shrinkage.items():
    print(f"  {name}: {val:.2%}")
```

### Covariance step (standard errors)

```python
if fit.covariance_result is not None:
    cov = fit.covariance_result
    param_names = list(fit.model.ini.values.keys())
    print("\nStandard Errors:")
    for i, name in enumerate(param_names):
        print(f"  {name}: SE={float(cov.standard_errors[i]):.6f}  "
              f"RSE={float(cov.rse[i]):.1f}%")
    print(f"\nCondition number: {cov.condition_number:.1f}")
```

### Predictions

```python
preds = fit.predictions
print("Prediction keys:", list(preds.keys()))
# Keys: id, time, dv, pred, ipred, res, ires
```

## Step 5: Summarize the fit

```python
summary = nlmixr2.summarize_fit(fit)
print(summary)
```

This prints a formatted summary table similar to R's `print(fit)`, including
the estimator name, observation count, parameter estimates, objective value,
AIC, and BIC.

## Step 6: Use parameter bounds

If parameters should be constrained to physiologically plausible ranges:

```python
ini_bounded = nlmixr2.ini({
    "A":  {"estimate": 300.0, "lower": 10.0,  "upper": 5000.0},
    "ke": {"estimate": 0.08,  "lower": 0.001, "upper": 1.0},
    "ka": {"estimate": 1.5,   "lower": 0.1,   "upper": 10.0},
})

model_bounded = nlmixr2.NLMIXRModel(ini=ini_bounded, model=model_block)

fit_bounded = nlmixr2.nlmixr2(
    model_bounded, data=data, est="foce",
    control={"maxiter": 500},
)

print("Bounded estimates:")
for name, value in fit_bounded.table["fixed_params"].items():
    print(f"  {name}: {value:.6f}")
```

## Step 7: Save and reload results

```python
# Save fit to JSON
fit_saved = nlmixr2.nlmixr2(
    model_spec, data=data, est="foce",
    control={"maxiter": 200},
    save="my_foce_fit.json",
)
print("Saved to:", fit_saved.save_path)
```

## Complete Script

```python
import nlmixr2

# Data
data = nlmixr2.theo_sd()

# Model
ini_block = nlmixr2.ini({
    "A":  300.0,
    "ke": 0.08,
    "ka": 1.5,
})
model_block = nlmixr2.model([
    "conc = A * ka / (ka - ke) * (exp(-ke * t) - exp(-ka * t))",
    "conc ~ add(add.err)",
])
model_spec = nlmixr2.NLMIXRModel(ini=ini_block, model=model_block)

# Fit
fit = nlmixr2.nlmixr2(
    model_spec, data=data, est="foce",
    control={"maxiter": 500, "lr": 0.01, "tol": 1e-6},
)

# Results
print(nlmixr2.summarize_fit(fit))
print("Converged:", fit.table["converged"])
print("Objective:", fit.objective)
for name, value in fit.table["fixed_params"].items():
    print(f"  {name} = {value:.4f}")
```
