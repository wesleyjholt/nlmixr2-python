from __future__ import annotations

import json
from pathlib import Path

import pytest

from nlmixr2 import ini, model, nlmixr2


def test_ini_accepts_scalar_and_bounded_values():
    block = ini({"lCl": 1.6, "prop.err": (0.0, 0.2, 1.0), "eta.cl": (0.1,)})

    assert block.values["lCl"].estimate == pytest.approx(1.6)
    assert block.values["prop.err"].lower == pytest.approx(0.0)
    assert block.values["prop.err"].upper == pytest.approx(1.0)
    assert block.values["eta.cl"].estimate == pytest.approx(0.1)


def test_ini_rejects_reserved_names():
    with pytest.raises(ValueError, match="Reserved nlmixr2 parameter prefix"):
        ini({"rx_bad": 1.0})


def test_ini_rejects_invalid_bound_lengths():
    with pytest.raises(ValueError, match="length 1, 2, or 3"):
        ini({"theta": (0.0, 1.0, 2.0, 3.0)})


def test_ini_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="Lower bound"):
        ini({"theta": (2.0, 1.0, 3.0)})


def test_model_accepts_statement_sequences():
    block = model(["kel = exp(tkel)", "cp = centr / v"])

    assert block.statements == ("kel = exp(tkel)", "cp = centr / v")


def test_model_accepts_mapping_assignments():
    block = model({"kel": "exp(tkel)", "cp": "centr / v"})

    assert block.statements == ("kel = exp(tkel)", "cp = centr / v")


def test_model_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one statement"):
        model([])


def test_model_rejects_blank_statements():
    with pytest.raises(ValueError, match="cannot be blank"):
        model(["   "])


def _toy_model():
    return {
        "ini": ini({"tka": 0.45, "tcl": 1.2, "tv": 3.1}),
        "model": model(["ka = exp(tka)", "cl = exp(tcl)", "v = exp(tv)"]),
    }


def test_nlmixr2_returns_model_when_no_data_is_supplied():
    result = nlmixr2(_toy_model)

    assert result.source == "python"
    assert set(result.ini.values) == {"tka", "tcl", "tv"}


def test_nlmixr2_builds_mock_fit_from_columnar_data():
    fit = nlmixr2(
        _toy_model,
        data={"ID": [1, 1, 2], "TIME": [0.0, 1.0, 0.0], "dv": [10.0, 12.0, 11.0]},
        control={"max_eval": 5},
        table={"keep": "dv"},
    )

    assert fit.estimator == "mock"
    assert fit.n_observations == 3
    assert fit.parameter_count == 3
    assert fit.objective == pytest.approx(2.0 / 3.0)
    assert fit.control["max_eval"] == 5


def test_nlmixr2_accepts_record_data_and_writes_json(tmp_path):
    save_path = tmp_path / "fit.json"
    fit = nlmixr2(
        _toy_model,
        data=[
            {"ID": 1, "TIME": 0.0, "dv": 9.0},
            {"ID": 1, "TIME": 1.0, "dv": 10.0},
        ],
        save=save_path,
    )

    payload = json.loads(save_path.read_text(encoding="utf-8"))
    assert fit.save_path == str(save_path)
    assert payload["n_observations"] == 2
    assert payload["columns"] == ["ID", "TIME", "dv"]


def test_nlmixr2_rejects_unsupported_estimators():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nlmixr2(_toy_model, data={"ID": [1], "dv": [1.0]}, est="focei")


def test_nlmixr2_loads_reference_fit_artifact(tmp_path):
    artifact = {
        "artifact_version": 1,
        "source": {
            "tool": "nlmixr2",
            "version": "5.0.0",
            "estimator": "focei",
            "dataset": "theo_sd",
        },
        "run": {
            "n_observations": 144,
            "columns": ["ID", "TIME", "DV", "AMT", "EVID", "CMT", "WT"],
            "objective": 116.812,
            "aic": 373.4118,
            "bic": 393.5914,
            "log_likelihood": -179.7059,
        },
        "parameters": {
            "tka": {"estimate": 0.461},
            "tcl": {"estimate": 1.01},
            "tv": {"estimate": 3.46},
            "add.sd": {"estimate": 0.692},
        },
    }
    path = tmp_path / "reference-fit.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    fit = nlmixr2(_toy_model, est="reference", reference_fit_path=path)

    assert fit.estimator == "focei"
    assert fit.n_observations == 144
    assert fit.objective == pytest.approx(116.812)
    assert fit.columns == ("ID", "TIME", "DV", "AMT", "EVID", "CMT", "WT")
    assert fit.table["reference_parameters"]["tcl"]["estimate"] == pytest.approx(1.01)
    assert fit.control["reference_fit_path"] == str(path)


def test_nlmixr2_requires_reference_fit_path_for_reference_estimator():
    with pytest.raises(ValueError, match="reference_fit_path"):
        nlmixr2(_toy_model, est="reference")


def test_nlmixr2_rejects_malformed_reference_artifacts(tmp_path):
    path = tmp_path / "bad-reference-fit.json"
    path.write_text(json.dumps({"source": {"estimator": "focei"}}), encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed reference fit artifact"):
        nlmixr2(_toy_model, est="reference", reference_fit_path=path)


def test_nlmixr2_loads_real_reference_fit_fixture():
    path = Path(__file__).with_name("fixtures") / "reference-theophylline-fit.json"

    fit = nlmixr2(_toy_model, est="reference", reference_fit_path=path)

    assert fit.estimator == "focei"
    assert fit.n_observations == 144
    assert fit.parameter_count == 4
    assert fit.objective == pytest.approx(117.003350569348)
    assert fit.table["reference_parameters"]["tka"]["estimate"] == pytest.approx(0.4750782090457)
