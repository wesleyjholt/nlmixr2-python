"""JIT compilation and vectorized (vmap) subject processing utilities.

Provides helpers for accelerating pharmacometric model evaluation using
JAX's JIT compiler and automatic vectorization (``vmap``).
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, Dict, List

import jax
import jax.numpy as jnp
import numpy as np


def jit_model_func(model_func: Callable) -> Callable:
    """Return a JIT-compiled version of *model_func*.

    Parameters
    ----------
    model_func : callable
        A JAX-compatible function (e.g. ``one_cmt_bolus``).

    Returns
    -------
    callable
        JIT-compiled version of the function.
    """
    return jax.jit(model_func)


def vmap_over_subjects(
    model_func: Callable,
    all_params: jnp.ndarray,
    times_per_subject: jnp.ndarray,
) -> jnp.ndarray:
    """Vectorized predictions for all subjects at once using ``jax.vmap``.

    Parameters
    ----------
    model_func : callable
        ``(params_vector, times) -> predictions`` for a single subject.
        *params_vector* is a 1-D array of that subject's parameters and
        *times* is a 1-D array of observation times.
    all_params : jnp.ndarray, shape (n_subjects, n_params)
        Stacked parameter vectors, one row per subject.
    times_per_subject : jnp.ndarray, shape (n_subjects, n_times)
        Time arrays, one row per subject (all rows must have the same
        length so the array is rectangular).

    Returns
    -------
    jnp.ndarray, shape (n_subjects, n_times)
        Predicted concentrations for every subject.
    """
    vmapped = jax.vmap(model_func, in_axes=(0, 0))
    return vmapped(all_params, times_per_subject)


def batch_objective(
    objective_fn: Callable,
    data_batches: List[Any],
) -> jnp.ndarray:
    """Sum of per-batch objective values (simple data-parallelism helper).

    Parameters
    ----------
    objective_fn : callable
        ``(batch) -> scalar`` objective for a single data batch.
    data_batches : list
        List of data batches to evaluate.

    Returns
    -------
    jnp.ndarray
        Scalar sum of all per-batch objective values.
    """
    if not data_batches:
        return jnp.array(0.0)
    total = jnp.array(0.0)
    for batch in data_batches:
        total = total + objective_fn(batch)
    return total


def ensure_jit_compatible(func: Callable) -> Callable:
    """Decorator that wraps a function to be JIT-friendly.

    Currently this serves as documentation / marker that the wrapped
    function uses only JAX-compatible operations.  It converts the
    function through ``jax.jit`` so that any Python-side tracing
    issues are caught early, and uses ``functools.wraps`` to preserve
    the original metadata.

    Parameters
    ----------
    func : callable
        A function that should only use JAX-compatible operations.

    Returns
    -------
    callable
        A JIT-compiled version of *func* with preserved metadata.
    """
    jitted = jax.jit(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return jitted(*args, **kwargs)

    return wrapper


def benchmark(
    func: Callable,
    *args: Any,
    n_runs: int = 10,
) -> Dict[str, float]:
    """Benchmark a function over multiple runs.

    Parameters
    ----------
    func : callable
        The function to time.
    *args
        Positional arguments forwarded to *func*.
    n_runs : int
        Number of repetitions (default 10).

    Returns
    -------
    dict
        ``{"mean_time": float, "std_time": float, "min_time": float}``
        where times are in seconds.
    """
    timings: list[float] = []
    for _ in range(n_runs):
        # Block until computation completes (important for async JAX dispatch)
        t0 = time.perf_counter()
        result = func(*args)
        # Force evaluation by blocking on the result
        if isinstance(result, jnp.ndarray):
            result.block_until_ready()
        t1 = time.perf_counter()
        timings.append(t1 - t0)

    arr = np.array(timings)
    return {
        "mean_time": float(np.mean(arr)),
        "std_time": float(np.std(arr)),
        "min_time": float(np.min(arr)),
    }
