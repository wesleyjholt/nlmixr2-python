"""Analytical (closed-form) solutions for standard PK compartment models.

Equivalent to rxode2's ``linCmt()`` functionality.  These closed-form
solutions are much faster than ODE solving for standard PK models.

All functions use ``jax.numpy`` and are JIT-compatible.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# 1-compartment models
# ---------------------------------------------------------------------------

def one_cmt_bolus(
    dose: float,
    ke: float,
    V: float,
    times: jnp.ndarray,
) -> jnp.ndarray:
    """1-compartment IV bolus: C(t) = (dose / V) * exp(-ke * t).

    Parameters
    ----------
    dose : float
        Bolus dose amount.
    ke : float
        Elimination rate constant (1/time).
    V : float
        Volume of distribution.
    times : jnp.ndarray
        Time points at which to evaluate concentration.

    Returns
    -------
    jnp.ndarray
        Concentration at each time point.
    """
    return (dose / V) * jnp.exp(-ke * times)


def one_cmt_oral(
    dose: float,
    ka: float,
    ke: float,
    V: float,
    times: jnp.ndarray,
) -> jnp.ndarray:
    """1-compartment first-order oral absorption (Bateman equation).

    C(t) = (dose * ka / (V * (ka - ke))) * (exp(-ke * t) - exp(-ka * t))

    When ka is very close to ke, uses L'Hopital's limit form to avoid
    division by zero: C(t) = (dose / V) * ka * t * exp(-ke * t).

    Parameters
    ----------
    dose : float
        Oral dose amount.
    ka : float
        Absorption rate constant (1/time).
    ke : float
        Elimination rate constant (1/time).
    V : float
        Volume of distribution.
    times : jnp.ndarray
        Time points at which to evaluate concentration.

    Returns
    -------
    jnp.ndarray
        Concentration at each time point.
    """
    diff = ka - ke
    # Standard Bateman equation
    conc_normal = (dose * ka / (V * diff)) * (
        jnp.exp(-ke * times) - jnp.exp(-ka * times)
    )
    # L'Hopital limit when ka -> ke: C(t) = (dose/V) * ka * t * exp(-ke * t)
    conc_limit = (dose / V) * ka * times * jnp.exp(-ke * times)
    return jnp.where(jnp.abs(diff) < 1e-6, conc_limit, conc_normal)


def one_cmt_infusion(
    dose: float,
    ke: float,
    V: float,
    tinf: float,
    times: jnp.ndarray,
) -> jnp.ndarray:
    """1-compartment zero-order infusion.

    During infusion (t <= tinf):
        C(t) = (Rate / (V * ke)) * (1 - exp(-ke * t))
    After infusion (t > tinf):
        C(t) = C(tinf) * exp(-ke * (t - tinf))

    where Rate = dose / tinf.

    Parameters
    ----------
    dose : float
        Total infused dose.
    ke : float
        Elimination rate constant (1/time).
    V : float
        Volume of distribution.
    tinf : float
        Infusion duration.
    times : jnp.ndarray
        Time points at which to evaluate concentration.

    Returns
    -------
    jnp.ndarray
        Concentration at each time point.
    """
    rate = dose / tinf
    c_during = (rate / (V * ke)) * (1.0 - jnp.exp(-ke * times))
    c_at_end = (rate / (V * ke)) * (1.0 - jnp.exp(-ke * tinf))
    c_after = c_at_end * jnp.exp(-ke * (times - tinf))
    return jnp.where(times <= tinf, c_during, c_after)


# ---------------------------------------------------------------------------
# 2-compartment models
# ---------------------------------------------------------------------------

def two_cmt_bolus(
    dose: float,
    k10: float,
    k12: float,
    k21: float,
    V1: float,
    times: jnp.ndarray,
) -> jnp.ndarray:
    """2-compartment IV bolus (central compartment concentration).

    C(t) = A * exp(-alpha * t) + B * exp(-beta * t)

    where alpha, beta are the eigenvalues of the 2-compartment system.

    Parameters
    ----------
    dose : float
        Bolus dose into central compartment.
    k10 : float
        Elimination rate constant from central compartment.
    k12 : float
        Transfer rate constant central -> peripheral.
    k21 : float
        Transfer rate constant peripheral -> central.
    V1 : float
        Volume of central compartment.
    times : jnp.ndarray
        Time points at which to evaluate concentration.

    Returns
    -------
    jnp.ndarray
        Central compartment concentration at each time point.
    """
    a_sum = k10 + k12 + k21
    discriminant = jnp.sqrt(a_sum**2 - 4.0 * k10 * k21)
    alpha = (a_sum + discriminant) / 2.0
    beta = (a_sum - discriminant) / 2.0

    c0 = dose / V1
    A = c0 * (alpha - k21) / (alpha - beta)
    B = c0 * (k21 - beta) / (alpha - beta)

    return A * jnp.exp(-alpha * times) + B * jnp.exp(-beta * times)


def two_cmt_oral(
    dose: float,
    ka: float,
    k10: float,
    k12: float,
    k21: float,
    V1: float,
    times: jnp.ndarray,
) -> jnp.ndarray:
    """2-compartment with first-order oral absorption (central compartment).

    C(t) = D * ka / V1 * [ A'*exp(-alpha*t) + B'*exp(-beta*t) + C'*exp(-ka*t) ]

    Parameters
    ----------
    dose : float
        Oral dose amount.
    ka : float
        Absorption rate constant.
    k10 : float
        Elimination rate constant from central compartment.
    k12 : float
        Transfer rate constant central -> peripheral.
    k21 : float
        Transfer rate constant peripheral -> central.
    V1 : float
        Volume of central compartment.
    times : jnp.ndarray
        Time points at which to evaluate concentration.

    Returns
    -------
    jnp.ndarray
        Central compartment concentration at each time point.
    """
    a_sum = k10 + k12 + k21
    discriminant = jnp.sqrt(a_sum**2 - 4.0 * k10 * k21)
    alpha = (a_sum + discriminant) / 2.0
    beta = (a_sum - discriminant) / 2.0

    coeff = dose * ka / V1

    A = (k21 - alpha) / ((ka - alpha) * (beta - alpha))
    B = (k21 - beta) / ((ka - beta) * (alpha - beta))
    C = (k21 - ka) / ((alpha - ka) * (beta - ka))

    return coeff * (
        A * jnp.exp(-alpha * times)
        + B * jnp.exp(-beta * times)
        + C * jnp.exp(-ka * times)
    )


# ---------------------------------------------------------------------------
# 3-compartment models
# ---------------------------------------------------------------------------

def _three_cmt_eigenvalues(
    k10: float, k12: float, k21: float, k13: float, k31: float,
):
    """Compute eigenvalues (alpha > beta > gamma) of the 3-compartment system.

    The rate matrix is::

        [[ -(k10+k12+k13),  k21,  k31 ],
         [  k12,           -k21,    0  ],
         [  k13,              0, -k31  ]]

    The eigenvalues are the three positive roots of the characteristic
    polynomial  lambda^3 - a*lambda^2 + b*lambda - c = 0, solved via
    the trigonometric method for three real roots (always the case for
    compartment models with positive rate constants).
    """
    a1 = k10 + k12 + k13
    # Coefficients of the characteristic polynomial (for positive lambda):
    # lambda^3 - a*lambda^2 + b*lambda - c = 0
    a = a1 + k21 + k31
    b = a1 * k21 + a1 * k31 + k21 * k31 - k12 * k21 - k13 * k31
    c = k10 * k21 * k31

    # Cardano/trigonometric solution for three real roots
    # Depressed cubic t^3 + pt + q = 0  via  lambda = t + a/3
    p = b - a**2 / 3.0
    q = -2.0 * a**3 / 27.0 + a * b / 3.0 - c
    # Use the cos-based formula for three real roots
    r = jnp.sqrt(jnp.clip(-p / 3.0, 0.0, None))
    cos_arg = jnp.clip(-q / (2.0 * r**3 + 1e-30), -1.0, 1.0)
    phi = jnp.arccos(cos_arg)

    root1 = 2.0 * r * jnp.cos(phi / 3.0) + a / 3.0
    root2 = 2.0 * r * jnp.cos((phi + 2.0 * jnp.pi) / 3.0) + a / 3.0
    root3 = 2.0 * r * jnp.cos((phi + 4.0 * jnp.pi) / 3.0) + a / 3.0

    # Sort: alpha > beta > gamma
    roots = jnp.array([root1, root2, root3])
    roots = jnp.sort(roots)[::-1]  # descending
    alpha = roots[0]
    beta = roots[1]
    gamma = roots[2]
    return alpha, beta, gamma


def three_cmt_bolus(
    dose: float,
    k10: float,
    k12: float,
    k21: float,
    k13: float,
    k31: float,
    V1: float,
    times: jnp.ndarray,
) -> jnp.ndarray:
    """3-compartment IV bolus (central compartment concentration).

    C(t) = A * exp(-alpha * t) + B * exp(-beta * t) + C * exp(-gamma * t)

    where alpha, beta, gamma are the eigenvalues of the 3-compartment system.

    Parameters
    ----------
    dose : float
        Bolus dose into central compartment.
    k10 : float
        Elimination rate constant from central compartment.
    k12 : float
        Transfer rate constant central -> shallow peripheral.
    k21 : float
        Transfer rate constant shallow peripheral -> central.
    k13 : float
        Transfer rate constant central -> deep peripheral.
    k31 : float
        Transfer rate constant deep peripheral -> central.
    V1 : float
        Volume of central compartment.
    times : jnp.ndarray
        Time points at which to evaluate concentration.

    Returns
    -------
    jnp.ndarray
        Central compartment concentration at each time point.
    """
    alpha, beta, gamma = _three_cmt_eigenvalues(k10, k12, k21, k13, k31)
    c0 = dose / V1

    # Macro-coefficients from the eigenvalue decomposition.
    # A_i = c0 * (alpha_i - k21) * (alpha_i - k31) / product_j!=i (alpha_i - alpha_j)
    A = c0 * (alpha - k21) * (alpha - k31) / ((alpha - beta) * (alpha - gamma))
    B = c0 * (beta - k21) * (beta - k31) / ((beta - alpha) * (beta - gamma))
    C = c0 * (gamma - k21) * (gamma - k31) / ((gamma - alpha) * (gamma - beta))

    return A * jnp.exp(-alpha * times) + B * jnp.exp(-beta * times) + C * jnp.exp(-gamma * times)


def three_cmt_oral(
    dose: float,
    ka: float,
    k10: float,
    k12: float,
    k21: float,
    k13: float,
    k31: float,
    V1: float,
    times: jnp.ndarray,
) -> jnp.ndarray:
    """3-compartment with first-order oral absorption (central compartment).

    C(t) = D*ka/V1 * [ A'*exp(-alpha*t) + B'*exp(-beta*t)
                        + C'*exp(-gamma*t) + D'*exp(-ka*t) ]

    Parameters
    ----------
    dose : float
        Oral dose amount.
    ka : float
        Absorption rate constant.
    k10 : float
        Elimination rate constant from central compartment.
    k12 : float
        Transfer rate constant central -> shallow peripheral.
    k21 : float
        Transfer rate constant shallow peripheral -> central.
    k13 : float
        Transfer rate constant central -> deep peripheral.
    k31 : float
        Transfer rate constant deep peripheral -> central.
    V1 : float
        Volume of central compartment.
    times : jnp.ndarray
        Time points at which to evaluate concentration.

    Returns
    -------
    jnp.ndarray
        Central compartment concentration at each time point.
    """
    alpha, beta, gamma = _three_cmt_eigenvalues(k10, k12, k21, k13, k31)
    coeff = dose * ka / V1

    # Each coefficient includes the residue from the absorption term
    A = (k21 - alpha) * (k31 - alpha) / ((ka - alpha) * (beta - alpha) * (gamma - alpha))
    B = (k21 - beta) * (k31 - beta) / ((ka - beta) * (alpha - beta) * (gamma - beta))
    C = (k21 - gamma) * (k31 - gamma) / ((ka - gamma) * (alpha - gamma) * (beta - gamma))
    D = (k21 - ka) * (k31 - ka) / ((alpha - ka) * (beta - ka) * (gamma - ka))

    return coeff * (
        A * jnp.exp(-alpha * times)
        + B * jnp.exp(-beta * times)
        + C * jnp.exp(-gamma * times)
        + D * jnp.exp(-ka * times)
    )


# ---------------------------------------------------------------------------
# Superposition
# ---------------------------------------------------------------------------

def superposition(
    single_dose_func: Callable,
    doses: jnp.ndarray,
    dose_times: jnp.ndarray,
    eval_times: jnp.ndarray,
) -> jnp.ndarray:
    """Multiple dose superposition via the linear superposition principle.

    For each dose, shifts the evaluation times relative to that dose time,
    evaluates the single-dose response, and sums contributions.  Times
    before a given dose contribute zero.

    Parameters
    ----------
    single_dose_func : callable
        ``f(dose, times) -> concentrations`` for a single dose at t=0.
    doses : jnp.ndarray
        Array of dose amounts, shape ``(n_doses,)``.
    dose_times : jnp.ndarray
        Array of dose administration times, shape ``(n_doses,)``.
    eval_times : jnp.ndarray
        Time points at which to evaluate total concentration.

    Returns
    -------
    jnp.ndarray
        Total concentration at each evaluation time point.
    """
    total = jnp.zeros_like(eval_times)
    for i in range(len(doses)):
        dt = eval_times - dose_times[i]
        # Only count contributions for times >= dose time
        contrib = single_dose_func(doses[i], jnp.maximum(dt, 0.0))
        contrib = jnp.where(dt >= 0.0, contrib, 0.0)
        total = total + contrib
    return total


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

def linCmt(
    params: Dict[str, Any],
    times: jnp.ndarray,
    model_type: str = "one_cmt_oral",
) -> jnp.ndarray:
    """Unified interface for analytical PK compartment models.

    Parameters
    ----------
    params : dict
        Model parameters.  Required keys depend on *model_type*:
        - ``one_cmt_bolus``: dose, ke, V
        - ``one_cmt_oral``: dose, ka, ke, V
        - ``one_cmt_infusion``: dose, ke, V, tinf
        - ``two_cmt_bolus``: dose, k10, k12, k21, V1
        - ``two_cmt_oral``: dose, ka, k10, k12, k21, V1
        - ``three_cmt_bolus``: dose, k10, k12, k21, k13, k31, V1
        - ``three_cmt_oral``: dose, ka, k10, k12, k21, k13, k31, V1
    times : jnp.ndarray
        Evaluation time points.
    model_type : str
        One of the supported model types listed above.

    Returns
    -------
    jnp.ndarray
        Concentration at each time point.
    """
    if model_type == "one_cmt_bolus":
        return one_cmt_bolus(params["dose"], params["ke"], params["V"], times)
    elif model_type == "one_cmt_oral":
        return one_cmt_oral(
            params["dose"], params["ka"], params["ke"], params["V"], times
        )
    elif model_type == "one_cmt_infusion":
        return one_cmt_infusion(
            params["dose"], params["ke"], params["V"], params["tinf"], times
        )
    elif model_type == "two_cmt_bolus":
        return two_cmt_bolus(
            params["dose"],
            params["k10"],
            params["k12"],
            params["k21"],
            params["V1"],
            times,
        )
    elif model_type == "two_cmt_oral":
        return two_cmt_oral(
            params["dose"],
            params["ka"],
            params["k10"],
            params["k12"],
            params["k21"],
            params["V1"],
            times,
        )
    elif model_type == "three_cmt_bolus":
        return three_cmt_bolus(
            params["dose"],
            params["k10"],
            params["k12"],
            params["k21"],
            params["k13"],
            params["k31"],
            params["V1"],
            times,
        )
    elif model_type == "three_cmt_oral":
        return three_cmt_oral(
            params["dose"],
            params["ka"],
            params["k10"],
            params["k12"],
            params["k21"],
            params["k13"],
            params["k31"],
            params["V1"],
            times,
        )
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Supported: one_cmt_bolus, "
            "one_cmt_oral, one_cmt_infusion, two_cmt_bolus, two_cmt_oral, "
            "three_cmt_bolus, three_cmt_oral"
        )
