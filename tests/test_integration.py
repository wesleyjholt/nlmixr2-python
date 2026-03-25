"""Integration tests: validate SAEM, FOCE, FOCEi, NLM on built-in datasets."""

from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

from nlmixr2 import ini, model, nlmixr2, theo_sd
from nlmixr2.api import NLMIXRFit, NLMIXRModel
from nlmixr2.diagnostics import summarize_fit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _theo_model():
    """Simple exponential decay model: cp = A * exp(-ke * t).

    Parameters: A (amplitude), ke (elimination rate), V (additive error scale).
    """
    return NLMIXRModel(
        ini=ini({"A": 10.0, "ke": 0.1, "V": 1.0}),
        model=model([
            "cp = A * exp(-ke * t)",
            "cp ~ add(V)",
        ]),
    )


def _subset_subjects(data, subject_ids):
    """Return a subset of data keeping only the given subject IDs."""
    keep = set(subject_ids)
    out = {k: [] for k in data}
    for i, sid in enumerate(data["id"]):
        if sid in keep:
            for k in data:
                out[k].append(data[k][i])
    return out


def _theo_obs_data(n_subjects=4):
    """Load theo_sd, keep first n_subjects, return only obs rows (evid==0)."""
    raw = theo_sd()
    # Determine which subject IDs to keep
    all_ids = sorted(set(raw["id"]))
    keep_ids = set(all_ids[:n_subjects])
    out = {k: [] for k in ["id", "time", "dv"]}
    for i in range(len(raw["id"])):
        if raw["id"][i] in keep_ids and raw["evid"][i] == 0:
            out["id"].append(raw["id"][i])
            out["time"].append(raw["time"][i])
            out["dv"].append(raw["dv"][i])
    return out


def _fit_theo(est, maxiter=20, n_subjects=4, **extra_control):
    """Run a fit on theo_sd observation data with the given estimator."""
    ctrl = {"maxiter": maxiter, **extra_control}
    return nlmixr2(_theo_model(), data=_theo_obs_data(n_subjects), est=est, control=ctrl)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_foce_theo_sd(self):
        """FOCE on theo_sd: objective is finite, parameters are positive."""
        fit = _fit_theo("foce")
        assert isinstance(fit, NLMIXRFit)
        assert math.isfinite(fit.objective)

        fp = fit.table["fixed_params"]
        assert fp["A"] > 0, f"A should be positive, got {fp['A']}"
        assert fp["ke"] > 0, f"ke should be positive, got {fp['ke']}"
        assert fp["V"] > 0, f"V should be positive, got {fp['V']}"

    def test_saem_theo_sd(self):
        """SAEM on theo_sd: converges and gives finite objective."""
        fit = _fit_theo("saem")
        assert isinstance(fit, NLMIXRFit)
        assert math.isfinite(fit.objective)

        fp = fit.table["fixed_params"]
        assert fp["A"] > 0
        assert fp["ke"] > 0
        assert fp["V"] > 0

    def test_focei_theo_sd(self):
        """FOCEi on theo_sd: objective is finite, parameters are positive."""
        fit = _fit_theo("focei")
        assert isinstance(fit, NLMIXRFit)
        assert math.isfinite(fit.objective)

        fp = fit.table["fixed_params"]
        assert fp["A"] > 0
        assert fp["ke"] > 0
        assert fp["V"] > 0

    def test_foce_vs_saem_agreement(self):
        """FOCE and SAEM objectives on theo_sd should be in the same ballpark."""
        foce_fit = _fit_theo("foce")
        saem_fit = _fit_theo("saem")

        foce_obj = foce_fit.objective
        saem_obj = saem_fit.objective

        assert math.isfinite(foce_obj)
        assert math.isfinite(saem_obj)

        # Within 50% of each other (they use different approximations)
        mean_obj = (abs(foce_obj) + abs(saem_obj)) / 2.0
        if mean_obj > 0:
            relative_diff = abs(foce_obj - saem_obj) / mean_obj
            assert relative_diff < 0.5, (
                f"FOCE ({foce_obj:.2f}) and SAEM ({saem_obj:.2f}) objectives "
                f"differ by {relative_diff:.0%}, expected < 50%"
            )

    def test_nlm_theo_sd(self):
        """NLM on theo_sd: produces a fit with finite objective."""
        fit = _fit_theo("nlm", maxiter=20, n_subjects=3)
        assert isinstance(fit, NLMIXRFit)
        assert math.isfinite(fit.objective)

        fp = fit.table["fixed_params"]
        assert fp["A"] > 0
        assert fp["ke"] > 0
        assert fp["V"] > 0

    def test_posthoc_after_foce(self):
        """Run FOCE, then posthoc to get individual etas."""
        foce_fit = _fit_theo("foce", maxiter=20)
        assert isinstance(foce_fit, NLMIXRFit)

        # Build posthoc control from the FOCE result
        n_params = foce_fit.parameter_count
        posthoc_ctrl = {
            "fixed_params": foce_fit.table["fixed_params"],
            "omega": jnp.eye(n_params) * 0.1,
            "sigma": 1.0,
        }
        posthoc_fit = nlmixr2(
            _theo_model(), data=_theo_obs_data(4), est="posthoc", control=posthoc_ctrl,
        )
        assert isinstance(posthoc_fit, NLMIXRFit)
        assert posthoc_fit.estimator == "posthoc"
        assert math.isfinite(posthoc_fit.objective)

    def test_fit_summary_complete(self):
        """FOCE fit summary contains expected sections."""
        fit = _fit_theo("foce", maxiter=20)
        summary = summarize_fit(fit)

        assert isinstance(summary, str)
        assert "nlmixr2 Fit Summary" in summary
        assert "Estimator" in summary
        assert "foce" in summary
        assert "Objective" in summary
        assert "AIC" in summary
        assert "BIC" in summary
        assert "Parameter Estimates" in summary
        # Check parameter names appear
        assert "ke" in summary
