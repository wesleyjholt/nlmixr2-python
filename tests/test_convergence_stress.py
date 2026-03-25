"""Convergence stress tests: difficult starting values, small datasets, edge cases."""

from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

from nlmixr2 import ini, model, nlmixr2, theo_sd
from nlmixr2.api import NLMIXRFit, NLMIXRModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _theo_model_with_inits(A=10.0, ke=0.1, V=1.0):
    """Simple exponential decay model with customizable starting values."""
    return NLMIXRModel(
        ini=ini({"A": A, "ke": ke, "V": V}),
        model=model([
            "cp = A * exp(-ke * t)",
            "cp ~ add(V)",
        ]),
    )


def _theo_obs_data():
    """Load theo_sd and return only observation rows (evid==0)."""
    raw = theo_sd()
    out = {k: [] for k in ["id", "time", "dv"]}
    for i in range(len(raw["id"])):
        if raw["evid"][i] == 0:
            out["id"].append(raw["id"][i])
            out["time"].append(raw["time"][i])
            out["dv"].append(raw["dv"][i])
    return out


def _subset_subjects(data, subject_ids):
    """Return a subset of data keeping only the given subject IDs."""
    keep = set(subject_ids)
    out = {k: [] for k in data}
    for i, sid in enumerate(data["id"]):
        if sid in keep:
            for k in data:
                out[k].append(data[k][i])
    return out


def _single_obs_per_subject(data):
    """Keep only the first observation per subject."""
    seen_obs = set()
    out = {k: [] for k in data}
    for i in range(len(data["id"])):
        sid = data["id"][i]
        if sid not in seen_obs:
            seen_obs.add(sid)
            for k in data:
                out[k].append(data[k][i])
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConvergenceStress:
    def test_foce_difficult_starting_values(self):
        """FOCE with starting values far from truth should still improve or converge."""
        # True params roughly: A~10, ke~0.1, V~1
        # Start far away: A=100, ke=2.0, V=0.01
        bad_model = _theo_model_with_inits(A=100.0, ke=2.0, V=0.01)
        data = _theo_obs_data()

        fit = nlmixr2(bad_model, data=data, est="foce", control={"maxiter": 50})

        assert isinstance(fit, NLMIXRFit)
        # At minimum, the pipeline should not crash and produce a finite objective
        assert math.isfinite(fit.objective)
        fp = fit.table["fixed_params"]
        # Parameters should still be positive (basic sanity)
        assert fp["A"] > 0
        assert fp["ke"] > 0
        assert fp["V"] > 0

    def test_saem_small_dataset(self):
        """SAEM with only 2 subjects should not crash."""
        data = _subset_subjects(_theo_obs_data(), {1, 2})
        mdl = _theo_model_with_inits()

        fit = nlmixr2(mdl, data=data, est="saem", control={"maxiter": 20})

        assert isinstance(fit, NLMIXRFit)
        assert math.isfinite(fit.objective)

    def test_foce_single_observation_per_subject(self):
        """FOCE with one observation per subject: edge case that should not crash."""
        data = _single_obs_per_subject(_theo_obs_data())
        mdl = _theo_model_with_inits()

        fit = nlmixr2(mdl, data=data, est="foce", control={"maxiter": 30})

        assert isinstance(fit, NLMIXRFit)
        assert math.isfinite(fit.objective)
