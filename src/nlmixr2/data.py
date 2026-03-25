"""NONMEM-style pharmacometric dataset validation and handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import jax.numpy as jnp
import numpy as np

REQUIRED_COLUMNS = {"id", "time", "dv"}
OPTIONAL_COLUMNS = {
    "amt", "evid", "cmt", "mdv", "cens", "limit",
    "wt", "bw", "age", "sex", "rate", "dur", "ii", "addl",
}
ALL_KNOWN_COLUMNS = REQUIRED_COLUMNS | OPTIONAL_COLUMNS


@dataclass(frozen=True)
class ValidatedDataset:
    """A validated NONMEM-style dataset with columns stored as JAX arrays."""

    columns: dict[str, jnp.ndarray]
    n_subjects: int
    n_observations: int
    n_doses: int
    subject_ids: tuple
    column_names: tuple


def _to_dict_of_lists(data: Union[dict, list]) -> dict[str, list]:
    """Convert input data to a dict of lists with lowercase keys."""
    if isinstance(data, list):
        # List of record dicts
        if len(data) == 0:
            raise ValueError("Empty dataset")
        keys = {k.lower() for row in data for k in row.keys()}
        result = {k: [] for k in keys}
        for row in data:
            row_lower = {k.lower(): v for k, v in row.items()}
            for k in keys:
                result[k].append(row_lower.get(k, 0.0))
        return result
    elif isinstance(data, dict):
        return {k.lower(): list(v) for k, v in data.items()}
    else:
        raise TypeError(f"Expected dict or list, got {type(data)}")


def _compute_counts(evid_arr: np.ndarray | None, mdv_arr: np.ndarray | None, n_records: int):
    """Compute n_observations and n_doses from evid and mdv arrays."""
    if evid_arr is not None:
        is_obs = evid_arr == 0
        is_dose = evid_arr == 1
    else:
        is_obs = np.ones(n_records, dtype=bool)
        is_dose = np.zeros(n_records, dtype=bool)

    # MDV=1 means the observation should be ignored
    if mdv_arr is not None:
        is_obs = is_obs & (mdv_arr == 0)

    return int(np.sum(is_obs)), int(np.sum(is_dose))


def validate_dataset(data: Union[dict, list]) -> ValidatedDataset:
    """Validate and normalize a NONMEM-style dataset.

    Parameters
    ----------
    data : dict of arrays/lists, or list of record dicts

    Returns
    -------
    ValidatedDataset
    """
    col_dict = _to_dict_of_lists(data)

    # Check required columns
    for req in REQUIRED_COLUMNS:
        if req not in col_dict:
            raise ValueError(f"Missing required column: {req.upper()}")

    # Convert to numpy for validation, then to JAX
    np_cols = {}
    for k, v in col_dict.items():
        np_cols[k] = np.array(v, dtype=np.float64)

    # Validate no NaN in ID and TIME
    if np.any(np.isnan(np_cols["id"])):
        raise ValueError("NaN values found in ID column")
    if np.any(np.isnan(np_cols["time"])):
        raise ValueError("NaN values found in TIME column")

    n_records = len(np_cols["id"])

    # Sort by ID then TIME
    sort_idx = np.lexsort((np_cols["time"], np_cols["id"]))
    for k in np_cols:
        np_cols[k] = np_cols[k][sort_idx]

    # Compute counts
    evid_arr = np_cols.get("evid")
    mdv_arr = np_cols.get("mdv")
    n_obs, n_doses = _compute_counts(evid_arr, mdv_arr, n_records)

    # Unique subject IDs (sorted)
    unique_ids = tuple(sorted(set(np_cols["id"].tolist())))
    # Convert float IDs to int if they are whole numbers
    unique_ids = tuple(int(x) if x == int(x) else x for x in unique_ids)

    # Convert to JAX arrays
    jax_cols = {k: jnp.array(v) for k, v in np_cols.items()}

    column_names = tuple(sorted(jax_cols.keys()))

    return ValidatedDataset(
        columns=jax_cols,
        n_subjects=len(unique_ids),
        n_observations=n_obs,
        n_doses=n_doses,
        subject_ids=unique_ids,
        column_names=column_names,
    )


def _filter_rows(ds: ValidatedDataset, mask: np.ndarray) -> ValidatedDataset:
    """Create a new ValidatedDataset with only the rows where mask is True."""
    np_cols = {k: np.array(v)[mask] for k, v in ds.columns.items()}
    n_records = int(np.sum(mask))

    evid_arr = np_cols.get("evid")
    mdv_arr = np_cols.get("mdv")
    n_obs, n_doses = _compute_counts(evid_arr, mdv_arr, n_records)

    unique_ids = tuple(sorted(set(np_cols["id"].tolist())))
    unique_ids = tuple(int(x) if x == int(x) else x for x in unique_ids)

    jax_cols = {k: jnp.array(v) for k, v in np_cols.items()}

    return ValidatedDataset(
        columns=jax_cols,
        n_subjects=len(unique_ids),
        n_observations=n_obs,
        n_doses=n_doses,
        subject_ids=unique_ids,
        column_names=ds.column_names,
    )


def split_by_subject(dataset: ValidatedDataset) -> dict:
    """Split a ValidatedDataset into per-subject datasets."""
    ids = np.array(dataset.columns["id"])
    result = {}
    for sid in dataset.subject_ids:
        mask = ids == sid
        result[sid] = _filter_rows(dataset, mask)
    return result


def get_observations(dataset: ValidatedDataset) -> ValidatedDataset:
    """Return only observation records (evid==0 or missing, and mdv!=1)."""
    ids = np.array(dataset.columns["id"])
    n = len(ids)

    if "evid" in dataset.columns:
        evid = np.array(dataset.columns["evid"])
        mask = evid == 0
    else:
        mask = np.ones(n, dtype=bool)

    if "mdv" in dataset.columns:
        mdv = np.array(dataset.columns["mdv"])
        mask = mask & (mdv == 0)

    return _filter_rows(dataset, mask)


def get_doses(dataset: ValidatedDataset) -> ValidatedDataset:
    """Return only dosing records (evid==1)."""
    if "evid" not in dataset.columns:
        # No EVID column => no doses, return empty
        mask = np.zeros(len(np.array(dataset.columns["id"])), dtype=bool)
    else:
        evid = np.array(dataset.columns["evid"])
        mask = evid == 1

    return _filter_rows(dataset, mask)


def merge_datasets(datasets: list[ValidatedDataset]) -> ValidatedDataset:
    """Merge multiple ValidatedDatasets into one.

    All datasets must have the same columns.
    """
    if not datasets:
        raise ValueError("No datasets to merge")
    if len(datasets) == 1:
        return datasets[0]

    # Collect all column names (union)
    all_cols = set()
    for ds in datasets:
        all_cols.update(ds.column_names)

    # Concatenate
    merged = {}
    for col in all_cols:
        arrays = []
        for ds in datasets:
            if col in ds.columns:
                arrays.append(np.array(ds.columns[col]))
            else:
                arrays.append(np.zeros(len(np.array(ds.columns["id"]))))
        merged[col] = np.concatenate(arrays).tolist()

    return validate_dataset(merged)


# ---------------------------------------------------------------------------
# NONMEM $DATA-style filtering: SKIP, ACCEPT, record selection
# ---------------------------------------------------------------------------


def _eval_predicate(predicate, value) -> bool:
    """Evaluate a single predicate against a value.

    - callable: call it
    - tuple (lo, hi): inclusive range check
    - other: equality
    """
    if callable(predicate):
        return bool(predicate(value))
    elif isinstance(predicate, tuple) and len(predicate) == 2:
        lo, hi = predicate
        return lo <= value <= hi
    else:
        return value == predicate


def filter_dataset(
    data: dict[str, list],
    accept: dict | None = None,
    skip: dict | None = None,
) -> dict[str, list]:
    """Filter a dict-of-lists dataset using NONMEM ACCEPT/SKIP semantics.

    Parameters
    ----------
    data : dict[str, list]
        Dataset as column-name -> list-of-values.
    accept : dict of {column: predicate}, optional
        Keep rows where **all** predicates are True.
    skip : dict of {column: predicate}, optional
        Remove rows where **any** predicate is True.

    Returns
    -------
    dict[str, list]
        Filtered copy of the dataset.
    """
    if accept is None and skip is None:
        return {k: list(v) for k, v in data.items()}

    keys = list(data.keys())
    n = len(data[keys[0]])
    keep = [True] * n

    # Accept: row must satisfy ALL predicates
    if accept is not None:
        for col, pred in accept.items():
            col_l = col.lower()
            vals = data.get(col_l, data.get(col))
            if vals is None:
                raise KeyError(f"Column '{col}' not found in dataset")
            for i in range(n):
                if keep[i]:
                    if not _eval_predicate(pred, vals[i]):
                        keep[i] = False

    # Skip: row is removed if ANY predicate matches
    if skip is not None:
        for col, pred in skip.items():
            col_l = col.lower()
            vals = data.get(col_l, data.get(col))
            if vals is None:
                raise KeyError(f"Column '{col}' not found in dataset")
            for i in range(n):
                if keep[i]:
                    if _eval_predicate(pred, vals[i]):
                        keep[i] = False

    return {
        k: [v[i] for i in range(n) if keep[i]]
        for k, v in data.items()
    }


def expand_doses(data: dict[str, list]) -> dict[str, list]:
    """Expand ADDL/II into individual dose records.

    Each row with ADDL > 0 and II > 0 generates *addl* additional dose
    records spaced by II time units.  The original row is kept (with
    ADDL set to 0), and all generated rows copy AMT, EVID, CMT, etc.

    Parameters
    ----------
    data : dict[str, list]

    Returns
    -------
    dict[str, list]
        Dataset with additional dose rows inserted.
    """
    keys = list(data.keys())
    n = len(data[keys[0]])

    addl_vals = data.get("addl")
    ii_vals = data.get("ii")

    # Nothing to expand if columns are missing
    if addl_vals is None or ii_vals is None:
        return {k: list(v) for k, v in data.items()}

    rows: list[dict] = []
    for i in range(n):
        row = {k: data[k][i] for k in keys}
        addl = int(addl_vals[i])
        ii = ii_vals[i]

        # Emit the original row with ADDL zeroed
        base_row = dict(row)
        base_row["addl"] = 0
        rows.append(base_row)

        if addl > 0 and ii > 0:
            for j in range(1, addl + 1):
                new_row = dict(row)
                new_row["time"] = row["time"] + j * ii
                new_row["addl"] = 0
                new_row["ii"] = 0.0
                # Keep dv as-is (typically 0 for dose records)
                rows.append(new_row)

    # Rebuild dict-of-lists
    out: dict[str, list] = {k: [] for k in keys}
    for row in rows:
        for k in keys:
            out[k].append(row[k])
    return out


def infer_evid(data: dict[str, list]) -> dict[str, list]:
    """Infer EVID from the AMT column when EVID is absent.

    If AMT > 0 and there is no EVID column, set EVID=1 for those rows
    and EVID=0 otherwise.  If EVID already exists, the data is returned
    unchanged (as a shallow copy).

    Parameters
    ----------
    data : dict[str, list]

    Returns
    -------
    dict[str, list]
    """
    out = {k: list(v) for k, v in data.items()}

    if "evid" in out:
        return out

    if "amt" not in out:
        return out

    out["evid"] = [1 if a > 0 else 0 for a in out["amt"]]
    return out


def reconstruct_doses(data: dict[str, list]) -> dict[str, list]:
    """Reconstruct missing RATE or DUR from AMT and the other.

    - If RATE present but DUR missing: DUR = AMT / RATE (where RATE > 0)
    - If DUR present but RATE missing: RATE = AMT / DUR (where DUR > 0)

    Parameters
    ----------
    data : dict[str, list]

    Returns
    -------
    dict[str, list]
    """
    out = {k: list(v) for k, v in data.items()}

    has_rate = "rate" in out
    has_dur = "dur" in out
    has_amt = "amt" in out

    if not has_amt:
        return out

    n = len(out["amt"])

    if has_rate and not has_dur:
        out["dur"] = [
            (out["amt"][i] / out["rate"][i]) if out["rate"][i] > 0 else 0.0
            for i in range(n)
        ]
    elif has_dur and not has_rate:
        out["rate"] = [
            (out["amt"][i] / out["dur"][i]) if out["dur"][i] > 0 else 0.0
            for i in range(n)
        ]

    return out
