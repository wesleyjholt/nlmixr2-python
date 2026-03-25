# SAEM Fitting Example

This example demonstrates fitting a pharmacokinetic model using the SAEM
(Stochastic Approximation Expectation-Maximization) estimator.

## Overview

SAEM is an EM-based algorithm that is often more robust to poor starting values
than gradient-based methods like FOCE. It operates in two phases:

1. **Burn-in:** Uses a fixed step size to explore the parameter space and escape
   local optima.
2. **EM (convergence):** Uses a decreasing step size (1/k) to converge to the
   maximum likelihood estimate.

Each iteration consists of:

- **E-step:** Sample individual random effects (etas) via Metropolis-Hastings
  MCMC.
- **M-step:** Update population parameters using stochastic approximation.

## Step 1: Load data and define the model

```python
import nlmixr2

# Load the Theophylline dataset
data = nlmixr2.theo_sd()

# Define a one-compartment oral PK model
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

## Step 2: Fit with SAEM

```python
fit = nlmixr2.nlmixr2(
    model_spec,
    data=data,
    est="saem",
    control={
        "n_burn": 300,
        "n_em": 200,
        "n_mcmc": 3,
        "step_size": 1.0,
        "sigma": 1.0,
    },
)
```

### Control options explained

| Parameter | Default | Description |
|---|---|---|
| `n_burn` | 300 | Number of burn-in iterations (fixed step size) |
| `n_em` | 200 | Number of EM iterations after burn-in (decreasing step size) |
| `n_mcmc` | 3 | Number of MCMC chains per E-step |
| `step_size` | 1.0 | Base step size for stochastic approximation |
| `sigma` | 1.0 | Residual error variance |

### Choosing n_burn and n_em

- **n_burn** controls how long the algorithm explores before settling down.
  For complex models or poor starting values, use a larger burn-in
  (e.g., 500--1000).
- **n_em** controls how many convergence iterations run after burn-in.
  Typically 100--300 is sufficient for well-behaved models.
- The total number of iterations is `n_burn + n_em`.

```python
# Conservative settings for a difficult model
fit_conservative = nlmixr2.nlmixr2(
    model_spec, data=data, est="saem",
    control={"n_burn": 1000, "n_em": 500, "n_mcmc": 5},
)
```

### Choosing n_mcmc

More MCMC chains per E-step improve the quality of the random-effect samples
but increase computation time. The default of 3 is usually sufficient; increase
to 5--10 for models with many random effects or strong correlations.

## Step 3: Inspect results

```python
# Convergence
print("Converged:", fit.table["converged"])
print("Iterations:", fit.table["n_iterations"])
print(f"Elapsed: {fit.elapsed_seconds:.1f}s")

# Objective and information criteria
print(f"Objective: {fit.objective:.4f}")
print(f"AIC:       {fit.aic:.4f}")
print(f"BIC:       {fit.bic:.4f}")

# Fixed-effect estimates
print("\nParameter estimates:")
for name, value in fit.table["fixed_params"].items():
    print(f"  {name}: {value:.6f}")

# Shrinkage
print("\nShrinkage:")
for name, val in fit.shrinkage.items():
    print(f"  {name}: {val:.2%}")
```

### Covariance step

SAEM does not natively produce a Hessian, but nlmixr2-python computes the
Hessian at the final SAEM estimates using the FOCE objective function. This
gives standard errors comparable to those from a FOCE fit:

```python
if fit.covariance_result is not None:
    cov = fit.covariance_result
    param_names = list(fit.model.ini.values.keys())
    for i, name in enumerate(param_names):
        print(f"  {name}: SE={float(cov.standard_errors[i]):.6f}  "
              f"RSE={float(cov.rse[i]):.1f}%")
    print(f"Condition number: {cov.condition_number:.1f}")
```

## Step 4: Using SAEM as a starting point for FOCE

A common workflow is to run SAEM first for robustness, then refine with FOCE:

```python
# Get SAEM estimates
saem_params = fit.table["fixed_params"]

# Build a new model with SAEM estimates as starting values
refined_ini = nlmixr2.ini({name: val for name, val in saem_params.items()})
refined_model = nlmixr2.NLMIXRModel(ini=refined_ini, model=model_block)

# Refine with FOCE
foce_fit = nlmixr2.nlmixr2(
    refined_model, data=data, est="foce",
    control={"maxiter": 500, "tol": 1e-6},
)

print(f"SAEM objective: {fit.objective:.4f}")
print(f"FOCE objective: {foce_fit.objective:.4f}")
```

## Step 5: Comparing multiple SAEM runs

Because SAEM is stochastic, results vary slightly between runs. It is good
practice to run SAEM multiple times and check that the estimates are consistent:

```python
objectives = []
for trial in range(3):
    f = nlmixr2.nlmixr2(
        model_spec, data=data, est="saem",
        control={"n_burn": 300, "n_em": 200},
    )
    objectives.append(f.objective)
    print(f"Trial {trial+1}: objective={f.objective:.4f}")

print(f"Range: {max(objectives) - min(objectives):.4f}")
```

If the objectives vary widely, increase `n_burn` and `n_em` or check that the
model is identifiable.

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

# Fit with SAEM
fit = nlmixr2.nlmixr2(
    model_spec, data=data, est="saem",
    control={"n_burn": 300, "n_em": 200, "n_mcmc": 3},
)

# Results
print(nlmixr2.summarize_fit(fit))
print("Converged:", fit.table["converged"])
for name, value in fit.table["fixed_params"].items():
    print(f"  {name} = {value:.4f}")
```
