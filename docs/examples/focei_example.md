# FOCEi Fitting Example

This example demonstrates fitting with FOCEi (FOCE with Interaction) and
compares the results to a standard FOCE fit.

## Overview

FOCEi extends FOCE by accounting for the eta-epsilon interaction -- the
dependence of residual variability on individual random effects. For each
subject, the residual variance is scaled by the Jacobian of the predictions
with respect to the etas:

```
Sigma_i = sigma * (1 + sum_k (df/deta_k)^2)
```

This is more accurate than FOCE when the within-subject variability depends on
the individual parameters (e.g., proportional error with log-normal PK
parameters).

## When to use FOCEi instead of FOCE

- The model has a **proportional** or **combined** residual error model.
- Parameters enter the model nonlinearly and the error structure depends on the
  predicted value.
- You suspect that ignoring the eta-epsilon interaction is biasing the FOCE
  estimates.

FOCEi is slower than FOCE because it computes the Jacobian of predictions with
respect to etas at each iteration using JAX automatic differentiation.

## Step 1: Define model and data

```python
import nlmixr2

data = nlmixr2.theo_sd()

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
```

## Step 2: Fit with FOCEi

```python
focei_fit = nlmixr2.nlmixr2(
    model_spec,
    data=data,
    est="focei",
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

The control options for FOCEi are identical to FOCE:

| Parameter | Default | Description |
|---|---|---|
| `maxiter` | 100 | Maximum outer iterations |
| `lr` | 0.01 | Learning rate for fixed effects |
| `lr_eta` | 0.01 | Learning rate for etas |
| `tol` | 1e-4 | Convergence tolerance |
| `inner_steps` | 10 | Eta optimization steps per outer iteration |
| `sigma` | 1.0 | Residual error variance |

## Step 3: Inspect FOCEi results

```python
print("Converged:", focei_fit.table["converged"])
print(f"Objective: {focei_fit.objective:.4f}")
print(f"AIC:       {focei_fit.aic:.4f}")

print("\nParameter estimates:")
for name, value in focei_fit.table["fixed_params"].items():
    print(f"  {name}: {value:.6f}")

print("\nShrinkage:")
for name, val in focei_fit.shrinkage.items():
    print(f"  {name}: {val:.2%}")

if focei_fit.covariance_result is not None:
    cov = focei_fit.covariance_result
    param_names = list(focei_fit.model.ini.values.keys())
    print("\nStandard errors:")
    for i, name in enumerate(param_names):
        print(f"  {name}: SE={float(cov.standard_errors[i]):.6f}  "
              f"RSE={float(cov.rse[i]):.1f}%")
    print(f"Condition number: {cov.condition_number:.1f}")
```

## Step 4: Compare with FOCE

```python
# Fit the same model with FOCE
foce_fit = nlmixr2.nlmixr2(
    model_spec,
    data=data,
    est="foce",
    control={"maxiter": 500, "lr": 0.01, "tol": 1e-6},
)

# Side-by-side comparison
print(f"{'':16s} {'FOCE':>12s} {'FOCEi':>12s}")
print("-" * 42)
print(f"{'Objective':16s} {foce_fit.objective:12.4f} {focei_fit.objective:12.4f}")
print(f"{'AIC':16s} {foce_fit.aic:12.4f} {focei_fit.aic:12.4f}")
print(f"{'BIC':16s} {foce_fit.bic:12.4f} {focei_fit.bic:12.4f}")
print(f"{'Converged':16s} {str(foce_fit.table['converged']):>12s} "
      f"{str(focei_fit.table['converged']):>12s}")
print(f"{'Iterations':16s} {foce_fit.table['n_iterations']:12d} "
      f"{focei_fit.table['n_iterations']:12d}")

# Parameter comparison
foce_params = foce_fit.table["fixed_params"]
focei_params = focei_fit.table["fixed_params"]

print(f"\n{'Parameter':16s} {'FOCE':>12s} {'FOCEi':>12s} {'Diff%':>8s}")
print("-" * 50)
for name in foce_params:
    p_foce = foce_params[name]
    p_focei = focei_params[name]
    diff_pct = abs(p_focei - p_foce) / abs(p_foce) * 100 if p_foce != 0 else 0
    print(f"{name:16s} {p_foce:12.6f} {p_focei:12.6f} {diff_pct:7.1f}%")
```

## Step 5: Use the compare_fits utility

nlmixr2-python provides a built-in comparison utility:

```python
comparison = nlmixr2.compare_fits([foce_fit, focei_fit])
print(nlmixr2.format_comparison(comparison))
```

## Interpreting FOCE vs FOCEi differences

- If the parameter estimates are very similar (< 5% difference), the
  eta-epsilon interaction is negligible and FOCE is sufficient.
- If the estimates differ substantially, the interaction matters and FOCEi
  should be preferred.
- A lower AIC/BIC from FOCEi suggests that accounting for the interaction
  improves model fit.
- For models with additive-only error, FOCE and FOCEi should give nearly
  identical results.

## Complete Script

```python
import nlmixr2

data = nlmixr2.theo_sd()

ini_block = nlmixr2.ini({"A": 300.0, "ke": 0.08, "ka": 1.5})
model_block = nlmixr2.model([
    "conc = A * ka / (ka - ke) * (exp(-ke * t) - exp(-ka * t))",
    "conc ~ add(add.err)",
])
model_spec = nlmixr2.NLMIXRModel(ini=ini_block, model=model_block)

# Fit with both estimators
foce_fit = nlmixr2.nlmixr2(model_spec, data=data, est="foce",
                             control={"maxiter": 500})
focei_fit = nlmixr2.nlmixr2(model_spec, data=data, est="focei",
                              control={"maxiter": 500})

# Compare
print(f"FOCE  objective: {foce_fit.objective:.4f}")
print(f"FOCEi objective: {focei_fit.objective:.4f}")

for name in foce_fit.table["fixed_params"]:
    print(f"  {name}: FOCE={foce_fit.table['fixed_params'][name]:.4f}  "
          f"FOCEi={focei_fit.table['fixed_params'][name]:.4f}")
```
