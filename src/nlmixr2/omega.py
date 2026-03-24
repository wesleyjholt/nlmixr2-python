"""Omega (random-effects) matrix construction and utilities.

Provides the Python equivalent of R's ``lotri`` package used by nlmixr2
for specifying between-subject variability (BSV) via block-diagonal
variance-covariance matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class OmegaBlock:
    """A named block-diagonal variance-covariance matrix.

    Attributes
    ----------
    matrix : jax.Array
        Symmetric positive-definite covariance matrix.
    names : tuple[str, ...]
        Parameter names corresponding to rows/columns.
    """

    matrix: jax.Array
    names: tuple[str, ...]


def omega(spec: dict[str | tuple[str, str], float]) -> OmegaBlock:
    """Construct an omega matrix from a dictionary specification.

    Parameters
    ----------
    spec
        Keys that are plain strings define diagonal (variance) elements.
        Keys that are 2-tuples of strings define off-diagonal (covariance)
        elements.  Example::

            omega({
                "eta.ka": 0.1,
                ("eta.ka", "eta.cl"): 0.05,
                "eta.cl": 0.2,
            })

    Returns
    -------
    OmegaBlock
        The resulting named covariance matrix.

    Raises
    ------
    ValueError
        If diagonal elements are not strictly positive, off-diagonal keys
        reference unknown parameters, or the resulting matrix is not positive
        semi-definite.
    """
    # Separate diagonal and off-diagonal entries, preserving insertion order.
    diag_names: list[str] = []
    diag_values: dict[str, float] = {}
    off_diag: list[tuple[str, str, float]] = []

    for key, value in spec.items():
        if isinstance(key, str):
            diag_names.append(key)
            diag_values[key] = value
        else:
            off_diag.append((key[0], key[1], value))

    # Validate diagonal > 0
    for name, val in diag_values.items():
        if val <= 0:
            raise ValueError(
                f"Diagonal element '{name}' must be strictly positive, got {val}"
            )

    # Build index lookup
    name_to_idx = {n: i for i, n in enumerate(diag_names)}
    n = len(diag_names)

    # Initialise matrix
    mat = jnp.zeros((n, n))
    for name, val in diag_values.items():
        i = name_to_idx[name]
        mat = mat.at[i, i].set(val)

    # Fill off-diagonal
    for n1, n2, val in off_diag:
        for ref in (n1, n2):
            if ref not in name_to_idx:
                raise ValueError(
                    f"Off-diagonal key references unknown parameter '{ref}'"
                )
        i, j = name_to_idx[n1], name_to_idx[n2]
        mat = mat.at[i, j].set(val)
        mat = mat.at[j, i].set(val)

    # Validate PSD via eigenvalues
    eigenvalues = jnp.linalg.eigvalsh(mat)
    if jnp.any(eigenvalues < -1e-10):
        raise ValueError(
            "Omega matrix is not positive semi-definite. "
            f"Smallest eigenvalue: {float(jnp.min(eigenvalues)):.6g}"
        )

    return OmegaBlock(matrix=mat, names=tuple(diag_names))


def block_diagonal(blocks: Sequence[OmegaBlock]) -> OmegaBlock:
    """Combine multiple OmegaBlocks into a larger block-diagonal matrix.

    Parameters
    ----------
    blocks
        Sequence of OmegaBlock instances.

    Returns
    -------
    OmegaBlock
        Combined block-diagonal matrix with concatenated parameter names.

    Raises
    ------
    ValueError
        If any parameter name appears in more than one block.
    """
    # Check for duplicate names
    all_names: list[str] = []
    for b in blocks:
        for name in b.names:
            if name in all_names:
                raise ValueError(f"Duplicate parameter name '{name}' across blocks")
            all_names.append(name)

    total = sum(b.matrix.shape[0] for b in blocks)
    mat = jnp.zeros((total, total))

    offset = 0
    for b in blocks:
        size = b.matrix.shape[0]
        mat = mat.at[offset : offset + size, offset : offset + size].set(b.matrix)
        offset += size

    return OmegaBlock(matrix=mat, names=tuple(all_names))


def cholesky_factor(ob: OmegaBlock) -> jax.Array:
    """Return the lower-triangular Cholesky factor L such that L @ L.T == ob.matrix.

    Parameters
    ----------
    ob
        An OmegaBlock with a positive-definite covariance matrix.

    Returns
    -------
    jax.Array
        Lower-triangular Cholesky factor.
    """
    return jnp.linalg.cholesky(ob.matrix)


def sample_etas(
    ob: OmegaBlock,
    n: int,
    key: jax.Array,
) -> jax.Array:
    """Sample random effects from a multivariate normal with mean zero.

    Parameters
    ----------
    ob
        OmegaBlock defining the covariance structure.
    n
        Number of samples (subjects).
    key
        JAX PRNG key.

    Returns
    -------
    jax.Array
        Array of shape ``(n, p)`` where *p* is the number of parameters.
    """
    L = cholesky_factor(ob)
    p = ob.matrix.shape[0]
    z = jax.random.normal(key, shape=(n, p))
    return z @ L.T
