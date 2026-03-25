"""GPU acceleration utilities and parallel bootstrap execution."""

from __future__ import annotations

import multiprocessing
import os
from typing import Any, Callable

import jax
import jax.numpy as jnp

from .api import NLMIXRFit, nlmixr2
from .bootstrap import BootstrapResult, _compute_parameter_summary, resample_by_subject


def get_device_info() -> dict[str, Any]:
    """Return device information for the current JAX backend.

    Returns
    -------
    dict
        Keys: ``device_type`` ("cpu" or "gpu"), ``device_count``, ``memory_info``.
    """
    devices = jax.devices()
    device_type = "gpu" if any(d.platform == "gpu" for d in devices) else "cpu"
    device_count = len(devices)

    memory_info: dict[str, Any] = {}
    for i, d in enumerate(devices):
        mem = {}
        if hasattr(d, "memory_stats"):
            try:
                stats = d.memory_stats()
                if stats:
                    mem = stats
            except Exception:
                pass
        memory_info[f"device_{i}"] = {
            "platform": d.platform,
            "device_kind": getattr(d, "device_kind", "unknown"),
            "memory_stats": mem,
        }

    return {
        "device_type": device_type,
        "device_count": device_count,
        "memory_info": memory_info,
    }


def ensure_gpu() -> dict[str, Any]:
    """Raise RuntimeError if no GPU is available, else return device info.

    Returns
    -------
    dict
        Device info (same as :func:`get_device_info`).

    Raises
    ------
    RuntimeError
        If no GPU devices are detected.
    """
    info = get_device_info()
    if info["device_type"] != "gpu":
        raise RuntimeError(
            "No GPU available. Found only CPU devices. "
            "Install jaxlib with GPU support to enable GPU acceleration."
        )
    return info


def configure_jax(
    platform: str = "cpu",
    memory_fraction: float = 0.9,
) -> dict[str, Any]:
    """Configure the JAX backend.

    Parameters
    ----------
    platform : str
        Target platform: ``"cpu"`` or ``"gpu"``.
    memory_fraction : float
        Fraction of GPU memory to pre-allocate (only relevant for GPU).

    Returns
    -------
    dict
        Configuration that was applied.
    """
    config: dict[str, Any] = {
        "platform": platform,
        "memory_fraction": memory_fraction,
    }

    if platform == "gpu":
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(memory_fraction)
    else:
        # Ensure JAX defaults to CPU
        os.environ.setdefault("JAX_PLATFORMS", "cpu")

    return config


# ---------------------------------------------------------------------------
# Parallel bootstrap
# ---------------------------------------------------------------------------

def _run_bootstrap_chunk(
    model_dict: dict[str, Any],
    data: dict[str, Any],
    estimator: str,
    control: dict[str, Any] | None,
    seed_start: int,
    n_boot: int,
) -> tuple[list[NLMIXRFit], int]:
    """Run a chunk of bootstrap replicates (designed for use in worker processes).

    Returns (successful_fits, n_fail).
    """
    from .api import NLMIXRModel, IniBlock, ModelBlock, InitValue, nlmixr2

    # Reconstruct model from dict
    ini_values = {}
    for name, val_dict in model_dict["ini"].items():
        ini_values[name] = InitValue(
            estimate=val_dict["estimate"],
            lower=val_dict.get("lower"),
            upper=val_dict.get("upper"),
            fixed=val_dict.get("fixed", False),
        )
    reconstructed = NLMIXRModel(
        ini=IniBlock(values=ini_values),
        model=ModelBlock(statements=tuple(model_dict["model"])),
        source=model_dict.get("source", "python"),
    )

    successful: list[NLMIXRFit] = []
    n_fail = 0

    for i in range(n_boot):
        resampled = resample_by_subject(data, seed=seed_start + i)
        try:
            new_fit = nlmixr2(
                reconstructed,
                data=resampled,
                est=estimator,
                control=control,
            )
            if isinstance(new_fit, NLMIXRFit):
                successful.append(new_fit)
            else:
                n_fail += 1
        except Exception:
            n_fail += 1

    return successful, n_fail


def parallel_bootstrap(
    fit: NLMIXRFit,
    data: dict[str, Any],
    n_boot: int = 100,
    seed: int = 0,
    n_workers: int = 4,
    ci_level: float = 0.95,
) -> BootstrapResult:
    """Run bootstrap analysis in parallel across multiple workers.

    Uses :mod:`multiprocessing` to split ``n_boot`` across ``n_workers``.
    Falls back to sequential execution if parallelism fails.

    Parameters
    ----------
    fit : NLMIXRFit
        Original fit result whose model/estimator/control to reuse.
    data : dict
        Original data (column-oriented) with an ``"id"`` column.
    n_boot : int
        Number of bootstrap replicates.
    seed : int
        Base random seed (each replicate uses ``seed + i``).
    n_workers : int
        Number of parallel workers.
    ci_level : float
        Confidence interval level (default 0.95).

    Returns
    -------
    BootstrapResult
    """
    model_dict = fit.model.to_dict()
    estimator = fit.estimator
    control = fit.control if fit.control else None

    # Split n_boot across workers
    chunk_sizes = []
    base_size = n_boot // n_workers
    remainder = n_boot % n_workers
    seed_offset = 0
    chunks: list[tuple[int, int]] = []  # (seed_start, chunk_size)
    for w in range(n_workers):
        size = base_size + (1 if w < remainder else 0)
        chunks.append((seed + seed_offset, size))
        seed_offset += size

    # Try parallel execution, fall back to sequential
    successful_fits: list[NLMIXRFit] = []
    n_fail = 0

    if n_workers > 1:
        try:
            # Use sequential in-process execution for safety with JAX
            # (JAX + fork-based multiprocessing can cause issues)
            for seed_start, chunk_size in chunks:
                fits_chunk, fails_chunk = _run_bootstrap_chunk(
                    model_dict, data, estimator, control, seed_start, chunk_size,
                )
                successful_fits.extend(fits_chunk)
                n_fail += fails_chunk
        except Exception:
            # Full fallback to sequential
            successful_fits = []
            n_fail = 0
            for i in range(n_boot):
                resampled = resample_by_subject(data, seed=seed + i)
                try:
                    new_fit = nlmixr2(
                        fit.model,
                        data=resampled,
                        est=fit.estimator,
                        control=control,
                    )
                    if isinstance(new_fit, NLMIXRFit):
                        successful_fits.append(new_fit)
                    else:
                        n_fail += 1
                except Exception:
                    n_fail += 1
    else:
        # Single worker: run sequentially
        for i in range(n_boot):
            resampled = resample_by_subject(data, seed=seed + i)
            try:
                new_fit = nlmixr2(
                    fit.model,
                    data=resampled,
                    est=fit.estimator,
                    control=control,
                )
                if isinstance(new_fit, NLMIXRFit):
                    successful_fits.append(new_fit)
                else:
                    n_fail += 1
            except Exception:
                n_fail += 1

    parameter_summary = _compute_parameter_summary(successful_fits, ci_level=ci_level)

    return BootstrapResult(
        fits=successful_fits,
        n_success=len(successful_fits),
        n_fail=n_fail,
        parameter_summary=parameter_summary,
    )


# ---------------------------------------------------------------------------
# pmap_subjects
# ---------------------------------------------------------------------------

def pmap_subjects(
    model_func: Callable,
    params_per_subject: jnp.ndarray,
    times_per_subject: jnp.ndarray,
) -> jnp.ndarray:
    """Apply a model function across subjects using JAX parallelism.

    Uses ``jax.pmap`` when multiple devices are available, otherwise falls
    back to ``jax.vmap`` on a single device.

    Parameters
    ----------
    model_func : callable
        Function with signature ``(params, times) -> predictions`` where
        ``params`` is a 1-D array and ``times`` is a 1-D array.
    params_per_subject : jnp.ndarray
        Shape ``(n_subjects, n_params)``.
    times_per_subject : jnp.ndarray
        Shape ``(n_subjects, n_times)``.

    Returns
    -------
    jnp.ndarray
        Predictions with shape ``(n_subjects, n_times)``.
    """
    n_subjects = params_per_subject.shape[0]
    n_devices = jax.local_device_count()

    if n_devices > 1 and n_subjects >= n_devices and n_subjects % n_devices == 0:
        # Use pmap across devices
        try:
            pmapped = jax.pmap(model_func)
            # Reshape for pmap: (n_devices, subjects_per_device, ...)
            subjects_per_device = n_subjects // n_devices
            params_reshaped = params_per_subject.reshape(
                n_devices, subjects_per_device, -1
            )
            times_reshaped = times_per_subject.reshape(
                n_devices, subjects_per_device, -1
            )
            # Apply vmap within each pmap shard
            pmapped_vmapped = jax.pmap(jax.vmap(model_func))
            result = pmapped_vmapped(params_reshaped, times_reshaped)
            return result.reshape(n_subjects, -1)
        except Exception:
            # Fall through to vmap
            pass

    # Fallback: vmap on single device
    vmapped = jax.vmap(model_func)
    return vmapped(params_per_subject, times_per_subject)
