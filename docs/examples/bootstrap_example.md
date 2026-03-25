# Bootstrap Uncertainty Example

This example demonstrates how to use nonparametric and parametric bootstrap
methods to estimate parameter uncertainty in nlmixr2-python.

## Overview

Bootstrap methods resample the data (or simulate new datasets) and refit the
model many times to build an empirical distribution of parameter estimates.
This provides confidence intervals and standard errors that do not rely on
the Laplacian/Hessian approximation.

nlmixr2-python provides two bootstrap functions:

- `bootstrap_fit()` -- nonparametric bootstrap (resample subjects with
  replacement).
- `parametric_bootstrap()` -- simulate new datasets from the fitted model with
  added noise, then refit.

## Nonparametric Bootstrap

### Step 1: Obtain an initial fit

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

# Fit the original data
fit = nlmixr2.nlmixr2(
    model_spec, data=data, est="foce",
    control={"maxiter": 300},
)
```

### Step 2: Run the bootstrap

```python
boot_result = nlmixr2.bootstrap_fit(
    fit=fit,
    data=data,
    n_boot=100,     # number of bootstrap replicates
    seed=42,        # base random seed for reproducibility
    ci_level=0.95,  # confidence interval level
)
```

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `fit` | (required) | Original `NLMIXRFit` whose model/estimator/control to reuse |
| `data` | (required) | Original data (column-oriented dict with an `id` column) |
| `n_boot` | 100 | Number of bootstrap replicates |
| `seed` | 0 | Base random seed (replicate *i* uses `seed + i`) |
| `ci_level` | 0.95 | Confidence interval level |

Each replicate resamples subjects with replacement (preserving all records for
each selected subject), then refits the model with the same estimator and
control settings as the original fit.

### Step 3: Inspect bootstrap results

```python
print(f"Successful fits: {boot_result.n_success}")
print(f"Failed fits:     {boot_result.n_fail}")
```

The `parameter_summary` dictionary contains per-parameter statistics computed
from the successful bootstrap replicates:

```python
for param_name, stats in boot_result.parameter_summary.items():
    print(f"\n{param_name}:")
    print(f"  Mean:   {stats['mean']:.6f}")
    print(f"  Median: {stats['median']:.6f}")
    print(f"  SE:     {stats['se']:.6f}")
    print(f"  95% CI: [{stats['ci_lo']:.6f}, {stats['ci_hi']:.6f}]")
```

The `BootstrapResult` object also stores all successful `NLMIXRFit` objects:

```python
# Distribution of objectives across replicates
objectives = [f.objective for f in boot_result.fits]
print(f"Objective range: [{min(objectives):.2f}, {max(objectives):.2f}]")
```

### Step 4: Compare bootstrap SE with Hessian-based SE

```python
if fit.covariance_result is not None:
    param_names = list(fit.model.ini.values.keys())
    print(f"{'Parameter':16s} {'Hessian SE':>12s} {'Bootstrap SE':>14s}")
    print("-" * 44)
    for i, name in enumerate(param_names):
        hess_se = float(fit.covariance_result.standard_errors[i])
        boot_se = boot_result.parameter_summary.get(name, {}).get("se", float("nan"))
        print(f"{name:16s} {hess_se:12.6f} {boot_se:14.6f}")
```

If the bootstrap and Hessian standard errors agree, the Hessian approximation
is adequate. Large discrepancies suggest model nonlinearity or
non-regular likelihood surfaces where the Hessian-based SEs may be unreliable.

---

## Parametric Bootstrap

The parametric bootstrap simulates new datasets by adding noise to the observed
DV values and refitting:

```python
pboot_result = nlmixr2.parametric_bootstrap(
    fit=fit,
    data=data,
    n_boot=100,
    seed=42,
    omega=0.1,   # between-subject variability scale
    sigma=1.0,   # residual variability scale
)

print(f"Successful fits: {pboot_result.n_success}")
print(f"Failed fits:     {pboot_result.n_fail}")

for param_name, stats in pboot_result.parameter_summary.items():
    print(f"\n{param_name}:")
    print(f"  Mean:   {stats['mean']:.6f}")
    print(f"  SE:     {stats['se']:.6f}")
    print(f"  95% CI: [{stats['ci_lo']:.6f}, {stats['ci_hi']:.6f}]")
```

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `omega` | 0.1 | Between-subject variability scale for noise generation |
| `sigma` | 1.0 | Residual variability scale for noise generation |

Each simulated dataset is created by adding `Normal(0, sigma)` noise plus
`Normal(0, omega)` per-subject perturbation to the original DV values.

---

## Practical Considerations

### How many replicates?

- **n_boot = 100** is a reasonable minimum for exploratory work.
- **n_boot = 500--1000** gives more stable confidence intervals.
- **n_boot = 200** is a practical compromise for moderate-sized datasets.

### Failed replicates

Some replicates may fail to converge (e.g., a resampled dataset that excludes
key subjects). The `n_fail` field reports how many replicates failed. A failure
rate above 20% suggests the model is fragile and may need simplification.

### Reproducibility

The `seed` parameter ensures reproducibility. Replicate *i* uses seed
`seed + i`, so the same seed always produces the same resampled datasets.

### Using with different estimators

The bootstrap reuses the estimator and control settings from the original fit.
To bootstrap a SAEM fit, simply pass a SAEM `NLMIXRFit`:

```python
saem_fit = nlmixr2.nlmixr2(
    model_spec, data=data, est="saem",
    control={"n_burn": 300, "n_em": 200},
)

boot_saem = nlmixr2.bootstrap_fit(
    fit=saem_fit, data=data, n_boot=100, seed=42,
)
```

---

## Complete Script

```python
import nlmixr2

# Data and model
data = nlmixr2.theo_sd()
ini_block = nlmixr2.ini({"A": 300.0, "ke": 0.08, "ka": 1.5})
model_block = nlmixr2.model([
    "conc = A * ka / (ka - ke) * (exp(-ke * t) - exp(-ka * t))",
    "conc ~ add(add.err)",
])
model_spec = nlmixr2.NLMIXRModel(ini=ini_block, model=model_block)

# Original fit
fit = nlmixr2.nlmixr2(model_spec, data=data, est="foce",
                        control={"maxiter": 300})

# Nonparametric bootstrap
boot = nlmixr2.bootstrap_fit(fit=fit, data=data, n_boot=100, seed=42)

print(f"Bootstrap: {boot.n_success} successes, {boot.n_fail} failures")
for name, stats in boot.parameter_summary.items():
    print(f"  {name}: mean={stats['mean']:.4f}  "
          f"SE={stats['se']:.4f}  "
          f"95% CI=[{stats['ci_lo']:.4f}, {stats['ci_hi']:.4f}]")
```
