# Migration Guide: R nlmixr2 to Python nlmixr2

This guide provides side-by-side comparisons for users familiar with the R
nlmixr2 ecosystem who are transitioning to the Python implementation.

---

## 1. Package Loading

### R

```r
library(nlmixr2)
library(rxode2)
library(lotri)
```

### Python

```python
import nlmixr2
```

All functionality is available from the single `nlmixr2` package. There is no
need for separate `rxode2` or `lotri` imports -- their equivalents are built
in.

---

## 2. Model Definition

### 2.1 The ini() Block

#### R

```r
one.compartment <- function() {
  ini({
    tka <- log(1.57)
    tcl <- log(0.0131)
    tv  <- log(1.05)
    eta.ka ~ 0.6
    eta.cl ~ 0.3
    eta.v  ~ 0.1
    add.sd <- 0.7
  })
  model({
    ka <- exp(tka + eta.ka)
    cl <- exp(tcl + eta.cl)
    v  <- exp(tv  + eta.v)
    d/dt(depot)   = -ka * depot
    d/dt(central) =  ka * depot - cl/v * central
    cp = central / v
    cp ~ add(add.sd)
  })
}
```

#### Python

```python
ini_block = nlmixr2.ini({
    "tka":    0.45,     # log(1.57)
    "tcl":    -4.33,    # log(0.0131)
    "tv":     0.049,    # log(1.05)
    "eta.ka": 0.6,
    "eta.cl": 0.3,
    "eta.v":  0.1,
    "add.sd": 0.7,
})
```

**Key differences:**

| Feature               | R                                   | Python                                      |
|-----------------------|-------------------------------------|----------------------------------------------|
| Syntax                | Tilde for etas, `<-` for thetas     | Plain dict; all values specified the same way |
| Bounds                | `tka <- c(0, 1, 5)`                | `"tka": [0, 1, 5]` or `"tka": {"estimate": 1, "lower": 0, "upper": 5}` |
| Fixed parameters      | `tka <- fix(1.0)`                  | `"tka": {"estimate": 1.0, "fixed": True}`   |

### 2.2 The model() Block

#### R

```r
model({
  ka <- exp(tka + eta.ka)
  cl <- exp(tcl + eta.cl)
  ...
  cp ~ add(add.sd)
})
```

#### Python

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

**Key differences:**

| Feature          | R                              | Python                                |
|------------------|--------------------------------|---------------------------------------|
| Block syntax     | `model({ ... })` with bare R   | List of statement strings             |
| Assignment op    | `<-`                           | `=`                                   |
| ODE syntax       | `d/dt(x) = ...` (same)        | `"d/dt(x) = ..."` (same, as string)  |
| Error model      | `cp ~ add(add.sd)`             | `"cp ~ add(add.sd)"` (same syntax)   |
| Dict shorthand   | Not available                  | `nlmixr2.model({"ka": "exp(tka)"})`  |

### 2.3 Combining ini + model

#### R

```r
# R uses a function that calls ini() and model() internally
fit <- nlmixr2(one.compartment, theo_sd, est = "focei")
```

#### Python

```python
from nlmixr2.api import NLMIXRModel

mod = NLMIXRModel(ini=ini_block, model=model_block)
fit = nlmixr2.nlmixr2(mod, data, est="focei")
```

---

## 3. The nlmixr2() Call

#### R

```r
fit <- nlmixr2(model_function, data, est = "focei",
               control = foceiControl(maxOuterIterations = 500))
```

#### Python

```python
fit = nlmixr2.nlmixr2(mod, data, est="focei",
                       control={"maxiter": 500})

# Or with a typed control object:
from nlmixr2 import FoceiControl
ctrl = FoceiControl(maxiter=500, tol=1e-6)
fit = nlmixr2.nlmixr2(mod, data, est="focei", control=ctrl.to_dict())
```

### Estimator Name Mapping

| R              | Python        |
|----------------|---------------|
| `"focei"`      | `"focei"`     |
| `"foce"`       | `"foce"`      |
| `"saem"`       | `"saem"`      |
| `"nlm"`        | `"nlm"`       |
| `"posthoc"`    | `"posthoc"`   |

### Control Object Mapping

| R                          | Python                  |
|----------------------------|-------------------------|
| `foceiControl()`           | `FoceiControl()`        |
| `saemControl()`            | `SaemControl()`         |
| `foceiControl(...)`        | `foceiControl(...)`     |
| `saemControl(...)`         | `saemControl(...)`      |

---

## 4. Accessing Fit Results

#### R

```r
fit$objf         # Objective function value
fit$AIC          # AIC
fit$BIC          # BIC
fit$eta          # Individual random effects
fit$PRED         # Population predictions
fit$IPRED        # Individual predictions
fit$CWRES        # Conditional weighted residuals
```

#### Python

```python
fit.objective           # Objective function value
fit.aic                 # AIC
fit.bic                 # BIC
fit.etas["values"]      # Individual random effects (JAX array)
fit.predictions["pred"] # Population predictions
fit.predictions["ipred"]# Individual predictions
fit.shrinkage           # Eta shrinkage dict
fit.covariance_result   # Standard errors, RSE, condition number
```

---

## 5. Data

### 5.1 Loading Example Datasets

#### R

```r
library(nlmixr2data)
data(theo_sd)
data(warfarin)
```

#### Python

```python
data = nlmixr2.theo_sd()          # Returns dict of lists
data = nlmixr2.warfarin()
data = nlmixr2.pheno_sd()

# Or by name:
data = nlmixr2.load_dataset("theo_sd")
print(nlmixr2.list_datasets())    # ['pheno_sd', 'theo_sd', 'warfarin']
```

### 5.2 Data Format

| Feature        | R                                   | Python                                  |
|----------------|-------------------------------------|-----------------------------------------|
| Format         | `data.frame`                        | `dict[str, list]` (column-oriented)     |
| Required cols  | `ID`, `TIME`, `DV`, `AMT`, `EVID`  | `id`, `time`, `dv`, `amt`, `evid`       |
| Case           | Case-insensitive (often uppercase)  | **Lowercase required**                  |
| Passing to fit | Pass data.frame directly            | Pass dict directly; auto-converted to JAX arrays |

### 5.3 Data Utilities

#### R

```r
# subset, merge via dplyr / base R
```

#### Python

```python
# Validation
validated = nlmixr2.validate_dataset(data)

# Split by subject
subjects = nlmixr2.split_by_subject(data)

# Get doses / observations only
doses = nlmixr2.get_doses(data)
obs   = nlmixr2.get_observations(data)

# Merge datasets
combined = nlmixr2.merge_datasets(data1, data2)
```

---

## 6. rxode2 --> solve_ode

### 6.1 ODE Solving

#### R (rxode2)

```r
mod <- rxode2({
  d/dt(depot)   = -ka * depot
  d/dt(central) =  ka * depot - ke * central
  cp = central / v
})

ev <- et(amt = 100, time = 0) %>% add.sampling(seq(0, 24, by = 0.5))
sim <- rxSolve(mod, params = list(ka = 1.5, ke = 0.08, v = 30), events = ev)
```

#### Python

```python
import jax.numpy as jnp

# Define ODE RHS: f(t, y, params) -> dy/dt
def pk_rhs(t, y, params):
    depot, central = y[0], y[1]
    ka = params["ka"]
    ke = params["ke"]
    return jnp.array([
        -ka * depot,
        ka * depot - ke * central,
    ])

# Solve
solution = nlmixr2.solve_ode(
    rhs=pk_rhs,
    t_span=(0.0, 24.0),
    y0=jnp.zeros(2),
    params={"ka": 1.5, "ke": 0.08, "v": 30.0},
    t_eval=jnp.linspace(0.0, 24.0, 49),
    dosing_events=[
        {"time": 0.0, "amount": 100.0, "compartment": 0},
    ],
)
# solution shape: (49, 2) -- concentrations in depot and central
cp = solution[:, 1] / 30.0  # central / v
```

### 6.2 Dosing Events

| R (rxode2)                        | Python                                           |
|-----------------------------------|--------------------------------------------------|
| `et(amt=100, time=0)`             | `{"time": 0.0, "amount": 100.0, "compartment": 0}` |
| `et(amt=100, dur=2)`              | `{"time": 0.0, "amount": 100.0, "compartment": 0, "duration": 2.0}` |
| `et(amt=100, ii=12, addl=6)`     | `{"time": 0.0, "amount": 100.0, "compartment": 0, "addl": 6, "ii": 12.0}` |

---

## 7. et() -- Event Table

#### R

```r
ev <- et(amt = 100, time = 0, cmt = 1) %>%
  add.sampling(c(0.5, 1, 2, 4, 8, 12, 24)) %>%
  et(amt = 100, time = 24, cmt = 1)
```

#### Python

```python
ev = (nlmixr2.et()
      .add_dosing(amt=100, time=0, cmt=1)
      .add_sampling([0.5, 1, 2, 4, 8, 12, 24])
      .add_dosing(amt=100, time=24, cmt=1))

# Convert to dict or JAX arrays
data_dict = ev.to_dict()
data_arrays = ev.to_arrays()
```

### Event Table Methods

| R                          | Python                                          |
|----------------------------|-------------------------------------------------|
| `et(amt=..., time=...)`   | `et().add_dosing(amt=..., time=...)`            |
| `add.sampling(times)`     | `.add_sampling(times)`                          |
| `et() %>% et()`           | Method chaining: `.add_dosing(...).add_dosing(...)` |
| `et(amt=100, ii=12, addl=6)` | `.add_dosing(amt=100, ii=12, addl=6)`       |
| Pipe with `%>%`           | Method chaining (each method returns new table) |
| `as.data.frame(ev)`       | `ev.to_dict()` or `ev.to_arrays()`             |
| Repeat pattern            | `ev.repeat(n=6, interval=24)`                   |
| Expand ADDL/II            | `ev.expand()`                                    |

---

## 8. lotri --> omega()

#### R (lotri)

```r
omega <- lotri({
  eta.ka ~ 0.6
  eta.cl ~ 0.3
  eta.v  ~ 0.1
})

# With covariance
omega <- lotri({
  eta.ka + eta.cl ~ c(0.6, 0.05, 0.3)
})
```

#### Python

```python
# Diagonal omega
om = nlmixr2.omega({
    "eta.ka": 0.6,
    "eta.cl": 0.3,
    "eta.v":  0.1,
})
print(om.matrix)  # 3x3 diagonal JAX array
print(om.names)   # ('eta.ka', 'eta.cl', 'eta.v')

# With off-diagonal covariance
om = nlmixr2.omega({
    "eta.ka": 0.6,
    ("eta.ka", "eta.cl"): 0.05,
    "eta.cl": 0.3,
})
```

### Sampling from Omega

#### R

```r
etas <- mvrnorm(n = 100, mu = rep(0, 3), Sigma = omega)
```

#### Python

```python
import jax

key = jax.random.PRNGKey(42)
etas = nlmixr2.sample_etas(om, n=100, key=key)
# etas shape: (100, 3)
```

---

## 9. Model Comparison

#### R

```r
# Compare fits
anova(fit1, fit2)
```

#### Python

```python
table = nlmixr2.compare_fits([fit1, fit2], names=["Model A", "Model B"])
print(nlmixr2.format_comparison(table))

# Likelihood ratio test
lrt = nlmixr2.likelihood_ratio_test(fit_full=fit2, fit_reduced=fit1, df=1)
print(f"p-value: {lrt.p_value}")
```

---

## 10. Model Update / Piping

#### R

```r
# Update ini values and refit
fit2 <- fit %>% update(tka = 0.6) %>% nlmixr2(data, est = "focei")
```

#### Python

```python
# Update initial values
updated_model = nlmixr2.update_ini(mod, {"tka": 0.6})

# Refit from a fit object
fit2 = nlmixr2.refit(fit, data, est="focei")

# Add or remove parameters
updated = nlmixr2.add_statement(mod, "ke = cl / v")
reduced = nlmixr2.remove_parameter(mod, "eta.v")
```

---

## 11. Diagnostics

#### R

```r
# GOF
plot(fit)

# VPC
vpcPlot(fit)
```

#### Python

```python
from nlmixr2.plots import plot_gof, plot_vpc

# GOF -- see diagnostics guide for full details
gof = nlmixr2.gof_data(dv, pred, ipred, res, ires, cwres, time)
fig = plot_gof(gof)

# VPC
vpc_result = nlmixr2.vpc(model_func, data, n_sim=200)
fig = plot_vpc(vpc_result)
```

---

## 12. Quick Reference Table

| Concept             | R                         | Python                              |
|---------------------|---------------------------|-------------------------------------|
| Import              | `library(nlmixr2)`       | `import nlmixr2`                    |
| Define model        | Function with `ini`/`model` | `NLMIXRModel(ini=..., model=...)`|
| Initial values      | `tka <- 0.5`             | `"tka": 0.5`                       |
| Bounds              | `tka <- c(0, 0.5, 5)`   | `"tka": [0, 0.5, 5]`               |
| Fixed param         | `tka <- fix(0.5)`        | `"tka": {"estimate": 0.5, "fixed": True}` |
| Model statements    | Bare R code in `model({})` | List of strings                  |
| Fit                 | `nlmixr2(mod, data, "focei")` | `nlmixr2.nlmixr2(mod, data, est="focei")` |
| Event table         | `et()`                    | `nlmixr2.et()`                     |
| Omega matrix        | `lotri({eta ~ 0.1})`     | `nlmixr2.omega({"eta": 0.1})`     |
| ODE solve           | `rxSolve(mod, ...)`      | `nlmixr2.solve_ode(rhs, ...)`     |
| Summary             | `print(fit)` / `summary(fit)` | `nlmixr2.summarize_fit(fit)` |
| GOF plots           | `plot(fit)`               | `plot_gof(gof_data(...))`          |
| VPC                 | `vpcPlot(fit)`            | `plot_vpc(vpc(...))`               |
| Covariate search    | `covarSearchAuto(...)`    | `nlmixr2.stepwise_covariate_search(...)` |
| Compare models      | `anova(fit1, fit2)`       | `nlmixr2.compare_fits([fit1, fit2])` |
| Bootstrap           | `bootstrapFit(fit)`       | `nlmixr2.bootstrap_fit(fit)`       |
| Simulation          | `rxSolve(mod, ...)`      | `nlmixr2.simulate(...)`            |
