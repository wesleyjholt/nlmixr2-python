from __future__ import annotations

import json
from math import fsum
from pathlib import Path

import pytest

from nlmixr2 import ini, model, nlmixr2


def _toy_model():
    return {
        "ini": ini({"tka": 0.45, "tcl": 1.2, "tv": 3.1}),
        "model": model(["ka = exp(tka)", "cl = exp(tcl)", "v = exp(tv)"]),
    }


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ({}, "Data cannot be empty"),
        ({"ID": [], "dv": []}, "Data cannot be empty"),
    ],
)
def test_nlmixr2_rejects_empty_columnar_data(data, match):
    with pytest.raises(ValueError, match=match):
        nlmixr2(_toy_model, data=data)


def test_nlmixr2_rejects_scalar_column_values():
    with pytest.raises(ValueError, match="one-dimensional sequences"):
        nlmixr2(_toy_model, data={"ID": 1, "dv": 1.0})


def test_nlmixr2_rejects_inconsistent_record_keys():
    rows = [
        {"ID": 1, "TIME": 0.0, "dv": 9.0},
        {"ID": 1, "dv": 10.0, "AMT": 100.0},
    ]

    with pytest.raises(ValueError, match="same columns"):
        nlmixr2(_toy_model, data=rows)


def test_nlmixr2_rejects_non_mapping_record_rows():
    with pytest.raises(TypeError, match="must be mappings"):
        nlmixr2(_toy_model, data=[("ID", 1), ("dv", 9.0)])


def test_nlmixr2_save_round_trip_preserves_fit_types(tmp_path):
    save_path = tmp_path / "fit.json"

    fit = nlmixr2(
        _toy_model,
        data={"ID": [1, 1, 2], "TIME": [0.0, 1.0, 0.0], "dv": [10.0, 12.0, 11.0]},
        save=save_path,
    )

    payload = json.loads(save_path.read_text(encoding="utf-8"))

    assert fit.columns == ("ID", "TIME", "dv")
    assert isinstance(fit.columns, tuple)
    assert payload["columns"] == ["ID", "TIME", "dv"]


def test_nlmixr2_large_columnar_mock_fit_is_stable():
    dv = [float((idx % 17) - 8) for idx in range(5000)]
    mean = fsum(dv) / len(dv)
    expected = fsum((value - mean) ** 2 for value in dv) / len(dv)

    fit = nlmixr2(
        _toy_model,
        data={
            "ID": [(idx // 10) + 1 for idx in range(5000)],
            "TIME": [float(idx % 10) for idx in range(5000)],
            "dv": dv,
        },
    )

    assert fit.n_observations == 5000
    assert fit.objective == pytest.approx(expected)


def test_nlmixr2_large_record_data_matches_columnar_fit():
    columnar = {
        "ID": [(idx // 8) + 1 for idx in range(2048)],
        "TIME": [float(idx % 8) for idx in range(2048)],
        "dv": [float((idx % 13) + 0.25) for idx in range(2048)],
    }
    rows = [
        {"ID": row_id, "TIME": time, "dv": dv}
        for row_id, time, dv in zip(columnar["ID"], columnar["TIME"], columnar["dv"])
    ]

    columnar_fit = nlmixr2(_toy_model, data=columnar)
    record_fit = nlmixr2(_toy_model, data=rows)

    assert record_fit.n_observations == columnar_fit.n_observations
    assert record_fit.columns == columnar_fit.columns
    assert record_fit.objective == pytest.approx(columnar_fit.objective)


def test_reference_fit_loading_is_stable_across_repeated_reads():
    path = Path(__file__).with_name("fixtures") / "reference-theophylline-fit.json"

    fits = [nlmixr2(_toy_model, est="reference", reference_fit_path=path) for _ in range(25)]

    assert {fit.estimator for fit in fits} == {"focei"}
    assert {fit.n_observations for fit in fits} == {144}
    assert {fit.parameter_count for fit in fits} == {4}
    for fit in fits:
        assert fit.objective == pytest.approx(117.003350569348)
