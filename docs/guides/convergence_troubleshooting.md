# Convergence Troubleshooting Guide

This guide covers common convergence issues when fitting nonlinear mixed-effects
models with nlmixr2-python, and practical strategies for resolving them.

## Diagnosing Convergence Problems

### Check the convergence flag

After fitting a model, the `table` dictionary on the `NLMIXRFit` object contains
a `converged` boolean:

```python
import nlmixr2

fit = nlmixr2.nlmixr2(model_spec, data=data, est="foce", control={"maxiter": 200})

# Did the optimizer declare convergence?
print(fit.table["converged"])   # True or False
print(fit.table["n_iterations"])
```

If `converged` is `False`, the optimizer exhausted its iteration budget without
satisfying the tolerance criterion. This does not always mean the estimates are
unusable, but it warrants investigation.

### Inspect the objective function value

```python
print(fit.objective)   # approximate -2 log-likelihood
print(fit.aic)
print(fit.bic)
```

Compare the objective across runs with different starting values or control
settings. A substantially higher objective (relative to a known-good run)
suggests the optimizer is stuck in a local minimum or has not yet converged.

### Examine the etas (random effects)

```python
import jax.numpy as jnp

etas = fit.etas["values"]          # shape (n_subjects, n_etas)
print("Eta means:", jnp.mean(etas, axis=0))
print("Eta SDs:  ", jnp.std(etas, axis=0))
```

Warning signs:

- Etas that are essentially zero for all subjects indicate high shrinkage. The
  data may not support individual-level estimation for that parameter.
- Etas with very large magnitude suggest the model is misspecified or the
  starting values are far from the truth.

### Review shrinkage

```python
print(fit.shrinkage)
# e.g. {"A": 0.85, "ke": 0.12}
```

Shrinkage values close to 1.0 mean the individual estimates are collapsing
toward the population value. Moderate shrinkage (< 0.3) is normal; high
shrinkage (> 0.5) may indicate that a random effect is unnecessary or that the
data are too sparse for that parameter.

### Inspect the condition number

If a covariance step was computed, the condition number of the Hessian matrix is
available:

```python
if fit.covariance_result is not None:
    print("Condition number:", fit.covariance_result.condition_number)
```

Guidelines:

| Condition number | Interpretation |
|---|---|
| < 1,000 | Well-conditioned; estimates are reliable |
| 1,000 -- 100,000 | Moderate ill-conditioning; consider reparametrization |
| > 100,000 | Severe ill-conditioning; the model is likely overparameterized |

A very large condition number means small changes in the data produce large
changes in the estimates. See "Interpreting Condition Numbers" below for more
detail.

---

## Common Issues and Solutions

### 1. Optimizer does not converge within maxiter

**Symptoms:** `fit.table["converged"]` is `False`; `n_iterations` equals
`maxiter`.

**Solutions:**

- Increase the maximum iterations:

  ```python
  fit = nlmixr2.nlmixr2(model_spec, data=data, est="foce",
                         control={"maxiter": 1000})
  ```

- Reduce the convergence tolerance (less strict):

  ```python
  fit = nlmixr2.nlmixr2(model_spec, data=data, est="foce",
                         control={"maxiter": 500, "tol": 1e-4})
  ```

- Use the SAEM estimator, which is often more robust to poor starting values:

  ```python
  fit = nlmixr2.nlmixr2(model_spec, data=data, est="saem",
                         control={"n_burn": 500, "n_em": 300})
  ```

### 2. Poor starting values

**Symptoms:** Objective function is very large on the first iteration; etas
diverge to extreme values; NaN or Inf in the objective.

**Solutions:**

- Choose initial estimates based on known pharmacokinetic properties or
  published literature values.
- Use log-scale parametrization (see "Scaling Parameters" below).
- Start from a preliminary fit: run SAEM first (which is more forgiving of
  starting values), then use its estimates as starting values for FOCE.

  ```python
  # Phase 1: rough SAEM fit
  saem_fit = nlmixr2.nlmixr2(model_spec, data=data, est="saem",
                               control={"n_burn": 200, "n_em": 100})
  saem_params = saem_fit.table["fixed_params"]

  # Phase 2: refine with FOCE using SAEM estimates
  refined_ini = nlmixr2.ini({name: val for name, val in saem_params.items()})
  refined_model = nlmixr2.NLMIXRModel(ini=refined_ini, model=model_spec.model)
  foce_fit = nlmixr2.nlmixr2(refined_model, data=data, est="foce",
                               control={"maxiter": 500})
  ```

### 3. Learning rate issues

**Symptoms:** Objective oscillates wildly between iterations; parameters
overshoot.

**Solutions:**

- Decrease the learning rate for fixed effects and/or etas:

  ```python
  fit = nlmixr2.nlmixr2(model_spec, data=data, est="foce",
                         control={"lr": 0.001, "lr_eta": 0.01})
  ```

- Increase inner steps to give the eta optimizer more time to settle before
  updating fixed effects:

  ```python
  fit = nlmixr2.nlmixr2(model_spec, data=data, est="foce",
                         control={"inner_steps": 30})
  ```

### 4. Parameters hitting bounds

**Symptoms:** One or more estimated parameters equal their lower or upper
bound after fitting.

**Solutions:**

- Widen the bounds if they are too restrictive:

  ```python
  ini_block = nlmixr2.ini({
      "A":  {"estimate": 300.0, "lower": 10.0,  "upper": 5000.0},
      "ke": {"estimate": 0.08,  "lower": 0.001, "upper": 1.0},
  })
  ```

- Re-examine whether the model structure is appropriate. Parameters
  consistently at bounds may signal misspecification.

- Use log-scale parametrization so that the transformed parameter is
  unconstrained.

### 5. High shrinkage on a random effect

**Symptoms:** Shrinkage > 0.5 for one or more parameters; etas nearly zero.

**Solutions:**

- Remove the random effect on that parameter (fix it to zero in the omega
  matrix) and refit.
- Add more informative data (e.g., richer sampling times around the phase
  governed by that parameter).
- Consider whether a simpler model (fewer random effects) is more appropriate.

---

## Scaling Parameters

Parameters that span very different orders of magnitude can cause numerical
difficulties. A standard technique is to express PK parameters on the log
scale in the model block:

```python
model_block = nlmixr2.model([
    "lA  = A",           # A is estimated on log scale internally
    "lke = ke",
    "conc = exp(lA) * exp(-exp(lke) * t)",
    "conc ~ add(add.err)",
])
```

Alternatively, keep the model in natural units but choose initial estimates and
bounds that are on a similar numeric scale (e.g., normalize clearance to L/hr
rather than mL/min).

---

## Parameter Bounds

Bounds are specified in the `ini()` call using the three-element list syntax or
dictionary syntax:

```python
# Three-element list: [lower, estimate, upper]
ini_block = nlmixr2.ini({
    "A":  [50.0,  300.0, 1000.0],
    "ke": [0.01,  0.08,  0.5],
})

# Dictionary syntax
ini_block = nlmixr2.ini({
    "A":  {"estimate": 300.0, "lower": 50.0,  "upper": 1000.0},
    "ke": {"estimate": 0.08,  "lower": 0.01,  "upper": 0.5},
})
```

During estimation, parameters are clipped to these bounds after each
optimization step. Use bounds when:

- A parameter is physically constrained (e.g., volume must be positive).
- The optimizer is exploring unreasonable regions of parameter space.

Avoid overly tight bounds, which can artificially constrain the optimizer and
produce biased estimates.

---

## Interpreting Condition Numbers

The condition number is the ratio of the largest to smallest eigenvalue of the
Hessian matrix. It measures how sensitive the objective function surface is to
different directions in parameter space.

**What a large condition number means:**

- The objective function is very flat in some directions (parameters are poorly
  identified) and very curved in others.
- Standard errors for the poorly identified parameters will be large.
- Small perturbations in the data can shift the estimates substantially.

**What to do about it:**

1. Fix parameters that are not identifiable (mark them as `fixed=True`).
2. Simplify the model -- drop random effects or reduce the number of
   compartments.
3. Reparametrize to reduce correlation between parameters.
4. Check whether additional data (e.g., IV + oral, or richer sampling) would
   improve identifiability.

---

## When to Switch Estimators

### FOCE (First-Order Conditional Estimation)

- **Strengths:** Fast per iteration; well-understood approximation; produces a
  clean Hessian for standard error computation.
- **Weaknesses:** Sensitive to starting values; may fail on highly nonlinear
  models or when the Laplacian approximation is poor.
- **Use when:** You have reasonable starting values and a relatively standard
  PK/PD model.

### FOCEi (FOCE with Interaction)

- **Strengths:** Accounts for the correlation between random effects and
  residual error (eta-epsilon interaction). More accurate than FOCE for models
  where within-subject variability depends on the individual parameters.
- **Weaknesses:** Slower than FOCE because it computes the Jacobian of
  predictions with respect to etas at each iteration.
- **Use when:** The residual error model may depend on individual parameters
  (e.g., proportional error with log-normal PK parameters).

### SAEM (Stochastic Approximation EM)

- **Strengths:** Globally more robust; less sensitive to starting values; does
  not require the Laplacian approximation.
- **Weaknesses:** Stochastic -- results vary slightly across runs; requires
  tuning of burn-in and EM iteration counts; no direct Hessian from the
  algorithm itself.
- **Use when:** FOCE fails to converge, or when the model is highly nonlinear
  with complex random-effect structures.

### NLM (Nelder-Mead)

- **Strengths:** Derivative-free; works on non-smooth objective surfaces.
- **Weaknesses:** Slow convergence in high dimensions; no gradient information.
- **Use when:** Gradient-based methods fail due to discontinuities or
  numerical issues in the objective function.

### Recommended workflow

1. Start with **FOCE** using literature-based starting values.
2. If FOCE does not converge, try **SAEM** to get rough estimates.
3. Use the SAEM estimates as starting values for a second **FOCE** or
   **FOCEi** run.
4. Compare objective values across estimators to confirm consistency.

---

## When to Simplify the Model

Consider simplifying if:

- Multiple parameters hit their bounds simultaneously.
- The condition number exceeds 100,000.
- Shrinkage exceeds 0.5 for more than one random effect.
- The optimizer converges to different solutions from different starting values
  (multiple local minima).
- Adding a parameter does not meaningfully improve AIC/BIC.

Common simplifications:

- Reduce the number of compartments (e.g., 2-compartment to 1-compartment).
- Remove random effects on parameters that show high shrinkage.
- Fix interindividual variability terms that are estimated near zero.
- Use an additive-only or proportional-only error model instead of combined.

---

## Quick Reference Checklist

1. Check `fit.table["converged"]` -- did the optimizer converge?
2. Compare `fit.objective` across runs -- is it consistent?
3. Inspect `fit.shrinkage` -- is any parameter's shrinkage > 0.5?
4. Review `fit.covariance_result.condition_number` -- is it < 100,000?
5. Look at the etas -- are they reasonable in magnitude and spread?
6. Check `fit.covariance_result.rse` -- are relative standard errors < 50%?
7. Try different starting values -- do you get the same solution?
8. If problems persist, switch estimators or simplify the model.
