"""Inter-occasion variability (IOV) support.

Provides utilities for modelling separate random effects per occasion
within a subject, complementing the between-subject variability (BSV)
represented by the standard omega matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import jax
import jax.numpy as jnp

from nlmixr2.omega import OmegaBlock, block_diagonal, cholesky_factor


@dataclass(frozen=True)
class IOVSpec:
    """Specification for inter-occasion variability.

    Attributes
    ----------
    occasion_column : str
        Column name in the data that identifies occasions (e.g. ``"occ"``).
    omega_iov : OmegaBlock
        IOV covariance matrix for the random effects that vary across
        occasions within a subject.
    parameter_names : tuple[str, ...]
        Names of the model parameters affected by IOV.
    """

    occasion_column: str
    omega_iov: OmegaBlock
    parameter_names: tuple[str, ...]


def extract_occasions(
    data: Dict[str, jax.Array],
    occasion_column: str,
) -> Dict[int, List[int]]:
    """Extract the unique occasions per subject from a dataset.

    Parameters
    ----------
    data
        Must contain ``"id"`` and the column named by *occasion_column*.
    occasion_column
        Key in *data* that holds integer occasion indicators.

    Returns
    -------
    dict
        Mapping from subject id (int) to a sorted list of unique occasion
        indices observed for that subject.
    """
    ids = data["id"]
    occs = data[occasion_column]

    result: Dict[int, List[int]] = {}
    unique_ids = jnp.unique(ids)
    for uid in unique_ids:
        uid_int = int(uid)
        mask = ids == uid
        subj_occs = occs[mask]
        unique_occs = sorted(int(o) for o in jnp.unique(subj_occs))
        result[uid_int] = unique_occs

    return result


def sample_iov_etas(
    iov_spec: IOVSpec,
    n_subjects: int,
    n_occasions: int,
    seed: int,
) -> jax.Array:
    """Sample IOV random effects for every subject and occasion.

    Parameters
    ----------
    iov_spec
        IOV specification containing the covariance structure.
    n_subjects
        Number of subjects.
    n_occasions
        Number of occasions per subject.
    seed
        PRNG seed for reproducibility.

    Returns
    -------
    jax.Array
        Array of shape ``(n_subjects, n_occasions, n_iov_params)``.
    """
    key = jax.random.PRNGKey(seed)
    L = cholesky_factor(iov_spec.omega_iov)
    p = iov_spec.omega_iov.matrix.shape[0]

    z = jax.random.normal(key, shape=(n_subjects, n_occasions, p))
    # Matrix multiply each (n_occasions, p) slice by L^T → correlated samples
    return jnp.einsum("ijk,lk->ijl", z, L)


def apply_iov(
    params: Dict[str, float],
    bsv_etas: jax.Array,
    iov_etas: jax.Array,
    bsv_names: Sequence[str],
    iov_names: Sequence[str],
) -> Dict[str, float]:
    """Apply both BSV and IOV random effects to population parameters.

    Parameters
    ----------
    params
        Population (fixed-effect) parameter values.
    bsv_etas
        BSV random effects for this subject, shape ``(n_bsv,)``.
    iov_etas
        IOV random effects for this subject at a specific occasion,
        shape ``(n_iov,)``.
    bsv_names
        Ordered parameter names corresponding to *bsv_etas*.
    iov_names
        Ordered parameter names corresponding to *iov_etas*.

    Returns
    -------
    dict
        Individual parameter values adjusted for both BSV and IOV.
    """
    result = dict(params)

    # Add BSV
    bsv_name_set = set(bsv_names)
    for j, name in enumerate(bsv_names):
        if name in result:
            result[name] = result[name] + float(bsv_etas[j])

    # Add IOV
    for j, name in enumerate(iov_names):
        if name in result:
            result[name] = result[name] + float(iov_etas[j])

    return result


def expand_omega_with_iov(
    omega_bsv: OmegaBlock,
    iov_spec: IOVSpec,
    n_occasions: int,
) -> OmegaBlock:
    """Build a combined block-diagonal omega with BSV and IOV blocks.

    The resulting matrix has the BSV block in the upper-left corner,
    followed by *n_occasions* copies of the IOV covariance block along
    the diagonal.

    Parameters
    ----------
    omega_bsv
        Between-subject variability covariance block.
    iov_spec
        IOV specification.
    n_occasions
        Number of occasions.  Each occasion contributes its own
        independent copy of the IOV covariance block.

    Returns
    -------
    OmegaBlock
        Combined block-diagonal covariance matrix.
    """
    # Build IOV blocks with occasion-indexed names
    iov_blocks: list[OmegaBlock] = []
    for occ in range(n_occasions):
        names = tuple(
            f"{name}.occ{occ}" for name in iov_spec.omega_iov.names
        )
        iov_blocks.append(OmegaBlock(matrix=iov_spec.omega_iov.matrix, names=names))

    return block_diagonal([omega_bsv] + iov_blocks)
