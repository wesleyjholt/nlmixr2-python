"""Tests for model update/piping utilities."""

from __future__ import annotations

import pytest

from nlmixr2.api import (
    IniBlock,
    InitValue,
    ModelBlock,
    NLMIXRFit,
    NLMIXRModel,
    ini,
    model,
    nlmixr2,
)
from nlmixr2.update import (
    add_statement,
    refit,
    remove_parameter,
    update_ini,
    update_model,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model() -> NLMIXRModel:
    return NLMIXRModel(
        ini=ini({"tvcl": 1.0, "tvv": 10.0}),
        model=model(["cl = tvcl", "v = tvv"]),
    )


def _make_fit() -> NLMIXRFit:
    m = _make_model()
    data = {"id": [1, 1], "time": [0.0, 1.0], "dv": [0.5, 0.3]}
    return nlmixr2(m, data=data, est="mock")


# ---------------------------------------------------------------------------
# update_ini
# ---------------------------------------------------------------------------

class TestUpdateIni:
    def test_changes_parameter_value(self):
        m = _make_model()
        updated = update_ini(m, tvcl=0.5)
        assert updated.ini.values["tvcl"].estimate == 0.5
        # Other parameter unchanged
        assert updated.ini.values["tvv"].estimate == 10.0

    def test_adds_new_parameter(self):
        m = _make_model()
        updated = update_ini(m, tvka=2.0)
        assert "tvka" in updated.ini.values
        assert updated.ini.values["tvka"].estimate == 2.0
        # Existing parameters still present
        assert "tvcl" in updated.ini.values
        assert "tvv" in updated.ini.values

    def test_with_bounds_tuple(self):
        m = _make_model()
        updated = update_ini(m, tvcl=(0.1, 0.5, 10.0))
        iv = updated.ini.values["tvcl"]
        assert iv.lower == 0.1
        assert iv.estimate == 0.5
        assert iv.upper == 10.0

    def test_accepts_fit(self):
        fit = _make_fit()
        updated = update_ini(fit, tvcl=0.5)
        assert isinstance(updated, NLMIXRModel)
        assert updated.ini.values["tvcl"].estimate == 0.5

    def test_original_model_unchanged(self):
        m = _make_model()
        original_estimate = m.ini.values["tvcl"].estimate
        _ = update_ini(m, tvcl=99.0)
        assert m.ini.values["tvcl"].estimate == original_estimate


# ---------------------------------------------------------------------------
# update_model
# ---------------------------------------------------------------------------

class TestUpdateModel:
    def test_replaces_statements(self):
        m = _make_model()
        updated = update_model(m, ["cl = tvcl * exp(eta.cl)", "v = tvv"])
        assert updated.model.statements == (
            "cl = tvcl * exp(eta.cl)",
            "v = tvv",
        )

    def test_accepts_fit(self):
        fit = _make_fit()
        updated = update_model(fit, ["cl = tvcl"])
        assert isinstance(updated, NLMIXRModel)
        assert updated.model.statements == ("cl = tvcl",)

    def test_original_model_unchanged(self):
        m = _make_model()
        original_stmts = m.model.statements
        _ = update_model(m, ["cl = tvcl * 2"])
        assert m.model.statements == original_stmts


# ---------------------------------------------------------------------------
# add_statement
# ---------------------------------------------------------------------------

class TestAddStatement:
    def test_appends_statement(self):
        m = _make_model()
        updated = add_statement(m, "ka = tvka")
        assert updated.model.statements[-1] == "ka = tvka"
        # Previous statements still present
        assert updated.model.statements[0] == "cl = tvcl"
        assert updated.model.statements[1] == "v = tvv"

    def test_accepts_fit(self):
        fit = _make_fit()
        updated = add_statement(fit, "ka = tvka")
        assert isinstance(updated, NLMIXRModel)
        assert "ka = tvka" in updated.model.statements


# ---------------------------------------------------------------------------
# remove_parameter
# ---------------------------------------------------------------------------

class TestRemoveParameter:
    def test_removes_from_ini(self):
        m = _make_model()
        updated = remove_parameter(m, "tvv")
        assert "tvv" not in updated.ini.values
        assert "tvcl" in updated.ini.values

    def test_accepts_fit(self):
        fit = _make_fit()
        updated = remove_parameter(fit, "tvv")
        assert isinstance(updated, NLMIXRModel)
        assert "tvv" not in updated.ini.values

    def test_original_model_unchanged(self):
        m = _make_model()
        _ = remove_parameter(m, "tvv")
        assert "tvv" in m.ini.values

    def test_raises_on_missing_parameter(self):
        m = _make_model()
        with pytest.raises(KeyError):
            remove_parameter(m, "nonexistent")


# ---------------------------------------------------------------------------
# refit
# ---------------------------------------------------------------------------

class TestRefit:
    def test_returns_new_fit(self):
        fit = _make_fit()
        data = {"id": [1, 1, 1], "time": [0.0, 1.0, 2.0], "dv": [0.5, 0.3, 0.1]}
        new_fit = refit(fit, data=data)
        assert isinstance(new_fit, NLMIXRFit)
        # New fit has the new observation count
        assert new_fit.n_observations == 3

    def test_with_different_estimator(self):
        fit = _make_fit()
        data = {"id": [1, 1], "time": [0.0, 1.0], "dv": [0.5, 0.3]}
        new_fit = refit(fit, data=data, est="mock")
        assert isinstance(new_fit, NLMIXRFit)
        assert new_fit.estimator == "mock"

    def test_preserves_estimator_when_not_specified(self):
        fit = _make_fit()
        data = {"id": [1, 1], "time": [0.0, 1.0], "dv": [0.5, 0.3]}
        new_fit = refit(fit, data=data)
        assert new_fit.estimator == fit.estimator

    def test_refit_with_control(self):
        fit = _make_fit()
        data = {"id": [1, 1], "time": [0.0, 1.0], "dv": [0.5, 0.3]}
        new_fit = refit(fit, data=data, control={"maxeval": 100})
        assert isinstance(new_fit, NLMIXRFit)
