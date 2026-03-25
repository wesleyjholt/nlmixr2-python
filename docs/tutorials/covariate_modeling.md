# Covariate Modeling with nlmixr2-python

This tutorial demonstrates how to incorporate patient covariates (e.g., body
weight, age, sex) into a population pharmacokinetic model and use automated
covariate selection to find the best model.

## Prerequisites

```python
import nlmixr2
from nlmixr2.api import NLMIXRModel
```

---

## 1. Load the Warfarin Dataset

The warfarin dataset includes 32 subjects with covariates for weight (`wt`),
age (`age`), and sex (`sex`):

```python
data = nlmixr2.warfarin()

print("Columns:", list(data.keys()))
# ['id', 'time', 'dv', 'amt', 'evid', 'wt', 'age', 'sex']

print("Subjects:", len(set(data["id"])))
# 32
```

---

## 2. Define the Base Model (No Covariates)

Start with a standard one-compartment oral PK model without covariates:

```python
ini_base = nlmixr2.ini({
    "tka":    [0.01, 1.0, 10.0],   # absorption rate constant
    "tcl":    [0.001, 0.03, 1.0],  # clearance
    "tv":     [0.1, 8.0, 50.0],    # volume of distribution
    "eta.ka": 0.5,
    "eta.cl": 0.3,
    "eta.v":  0.2,
    "add.sd": 0.5,
})

model_base = nlmixr2.model([
    "ka = exp(tka + eta.ka)",
    "cl = exp(tcl + eta.cl)",
    "v  = exp(tv  + eta.v)",
    "d/dt(depot)   = -ka * depot",
    "d/dt(central) =  ka * depot - cl/v * central",
    "cp = central / v",
    "cp ~ add(add.sd)",
])

base_mod = NLMIXRModel(ini=ini_base, model=model_base)
```

### 2.1 Fit the Base Model

```python
base_fit = nlmixr2.nlmixr2(base_mod, data, est="foce")

print(f"Base Objective: {base_fit.objective:.4f}")
print(f"Base AIC:       {base_fit.aic:.4f}")
```

---

## 3. Detect Covariates in the Data

Use `extract_covariates` to identify which data columns could serve as
covariates. It compares column names against the model statements and excludes
standard columns (id, time, dv, amt, evid):

```python
covs = nlmixr2.extract_covariates(model_base, list(data.keys()))
print("Detected covariates:", covs)
# ['wt', 'age', 'sex']
```

---

## 4. Add a Single Covariate Manually

### 4.1 Parse the Model

`add_covariate_to_model` works on a `ParsedModel`. First, parse the base
model:

```python
parsed = nlmixr2.parse_model(model_base, ini_base)
print("Parameters:", parsed.parameter_names)
print("States:", parsed.state_names)
```

### 4.2 Add Weight on Clearance (Linear Effect)

The `add_covariate_to_model` function introduces a new theta
(`theta_wt_cl`) and modifies the model so that clearance depends linearly
on centered weight:

```python
parsed_wt = nlmixr2.add_covariate_to_model(
    parsed,
    covariate_name="wt",
    parameter_name="cl",
    effect="linear",       # param = base + theta * (WT - center)
)

print("New parameters:", parsed_wt.parameter_names)
# [..., 'theta_wt_cl']
```

### 4.3 Effect Types

Three covariate effect types are supported:

| Effect          | Formula                                        | Covariate column needed    |
|-----------------|------------------------------------------------|----------------------------|
| `"linear"`      | `param = base + theta * (COV - center)`        | `<cov>_centered`           |
| `"power"`       | `param = base * (COV / center) ** theta`       | `<cov>_ratio`              |
| `"exponential"` | `param = base * exp(theta * (COV - center))`   | `<cov>_centered`           |

### 4.4 Prepare Centered Covariate Data

Before fitting, center the covariates using `CovariateModel` and
`center_covariates`:

```python
import jax.numpy as jnp
import numpy as np

cov_model = nlmixr2.CovariateModel(
    covariates={"wt": "continuous", "age": "continuous", "sex": "categorical"},
    centering={"wt": float(np.median(data["wt"])), "age": float(np.median(data["age"]))},
    transformations={"wt": "none", "age": "none"},
)

# Convert data lists to JAX arrays for center_covariates
jax_data = {k: jnp.array(v) for k, v in data.items()}
centered_data = nlmixr2.center_covariates(jax_data, cov_model)
# Now centered_data contains "wt_centered" and "age_centered" columns
```

### 4.5 Fit the Covariate Model

Build the updated NLMIXRModel with the new theta and fit:

```python
from nlmixr2.api import IniBlock, InitValue

# Add theta_wt_cl to the ini block
new_values = dict(ini_base.values)
new_values["theta_wt_cl"] = InitValue(estimate=0.0)
new_ini = IniBlock(values=new_values)

cov_mod = NLMIXRModel(ini=new_ini, model=model_base)
fit_wt = nlmixr2.nlmixr2(cov_mod, data, est="foce")

print(f"WT model Objective: {fit_wt.objective:.4f}")
print(f"WT model AIC:       {fit_wt.aic:.4f}")
```

---

## 5. Compare Base vs. Covariate Model

### 5.1 Using compare_fits

```python
table = nlmixr2.compare_fits(
    [base_fit, fit_wt],
    names=["Base", "Base + WT~CL"],
)

print(nlmixr2.format_comparison(table))
```

Output:

```
Model            Objective           AIC           BIC   nPar    nObs
----------------------------------------------------------------------
Base              123.4567      137.4567      148.2345      7     320
Base + WT~CL      118.9012      134.9012      148.1234      8     320

Best AIC: Base + WT~CL
Best BIC: Base
```

### 5.2 Using Likelihood Ratio Test

For nested models, use `likelihood_ratio_test`:

```python
lrt = nlmixr2.likelihood_ratio_test(
    fit_full=fit_wt,      # more complex model
    fit_reduced=base_fit,  # simpler model
    df=1,                  # 1 extra parameter
)

print(f"LRT statistic: {lrt.statistic:.4f}")
print(f"p-value: {lrt.p_value:.4f}")
print(f"Significant: {lrt.significant}")
```

---

## 6. Automated Stepwise Covariate Search

### 6.1 Forward Addition

Test adding each covariate-parameter combination one at a time. The most
significant addition (lowest p-value, p < alpha) is selected:

```python
fwd_results = nlmixr2.forward_addition(
    base_fit=base_fit,
    data=data,
    covariates=["wt", "age", "sex"],
    parameters=["cl", "v", "ka"],
    effects=["linear"],
    alpha=0.05,
)

for r in fwd_results:
    marker = " <-- SELECTED" if r.selected else ""
    print(f"  {r.covariate} ~ {r.parameter} ({r.effect}): "
          f"dOFV={r.delta_obj:.3f}, p={r.p_value:.4f}{marker}")
```

### 6.2 Backward Elimination

Test removing each covariate from a full model. Covariates whose removal
does not significantly worsen the fit (p >= alpha) are candidates for removal:

```python
bwd_results = nlmixr2.backward_elimination(
    full_fit=fit_wt,
    data=data,
    covariates=["wt"],
    parameters=["cl"],
    effects=["linear"],
    alpha=0.01,
)

for r in bwd_results:
    marker = " <-- REMOVE" if r.selected else ""
    print(f"  {r.covariate} ~ {r.parameter}: "
          f"dOFV={r.delta_obj:.3f}, p={r.p_value:.4f}{marker}")
```

### 6.3 Full Stepwise Search

The `stepwise_covariate_search` function alternates forward addition and
backward elimination until no more changes are made:

```python
all_steps = nlmixr2.stepwise_covariate_search(
    base_fit=base_fit,
    data=data,
    covariates=["wt", "age", "sex"],
    parameters=["cl", "v", "ka"],
    effects=["linear"],
    forward_alpha=0.05,    # significance for adding covariates
    backward_alpha=0.01,   # significance for keeping covariates
    max_steps=10,
)

print(f"Total steps evaluated: {len(all_steps)}")

# Show selected covariates
selected = [s for s in all_steps if s.selected]
for s in selected:
    print(f"  [{s.direction}] {s.covariate} ~ {s.parameter} "
          f"({s.effect}): dOFV={s.delta_obj:.3f}")
```

### 6.4 StepResult Fields

Each `StepResult` from the search contains:

| Field       | Description                                         |
|-------------|-----------------------------------------------------|
| `covariate` | Name of the covariate tested                        |
| `parameter` | Model parameter the covariate was applied to        |
| `effect`    | Effect type (`"linear"`, `"power"`, `"exponential"`) |
| `direction` | `"forward"` or `"backward"`                         |
| `delta_obj` | Change in objective function value                  |
| `p_value`   | p-value from the likelihood ratio test (df=1)       |
| `selected`  | Whether this combo was selected in this step        |

---

## 7. Complete Covariate Modeling Workflow

```python
import nlmixr2
from nlmixr2.api import NLMIXRModel

# 1. Load data
data = nlmixr2.warfarin()

# 2. Define and fit base model
ini_base = nlmixr2.ini({
    "tka": [0.01, 1.0, 10.0],
    "tcl": [0.001, 0.03, 1.0],
    "tv":  [0.1, 8.0, 50.0],
    "eta.ka": 0.5, "eta.cl": 0.3, "eta.v": 0.2,
    "add.sd": 0.5,
})

model_base = nlmixr2.model([
    "ka = exp(tka + eta.ka)",
    "cl = exp(tcl + eta.cl)",
    "v  = exp(tv  + eta.v)",
    "d/dt(depot)   = -ka * depot",
    "d/dt(central) =  ka * depot - cl/v * central",
    "cp = central / v",
    "cp ~ add(add.sd)",
])

base_mod = NLMIXRModel(ini=ini_base, model=model_base)
base_fit = nlmixr2.nlmixr2(base_mod, data, est="foce")

# 3. Run automated stepwise covariate selection
steps = nlmixr2.stepwise_covariate_search(
    base_fit=base_fit,
    data=data,
    covariates=["wt", "age", "sex"],
    parameters=["cl", "v", "ka"],
    forward_alpha=0.05,
    backward_alpha=0.01,
)

# 4. Review results
selected = [s for s in steps if s.selected and s.direction == "forward"]
for s in selected:
    print(f"Include: {s.covariate} on {s.parameter} ({s.effect}), p={s.p_value:.4f}")
```
