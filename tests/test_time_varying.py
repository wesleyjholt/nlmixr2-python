"""Tests for time-varying covariate support — TDD style."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from nlmixr2.time_varying import (
    TimeVaryingCovariate,
    interpolate_covariate,
    build_covariate_function,
    extract_time_varying,
)


# ---------------------------------------------------------------------------
# interpolate_covariate — LOCF
# ---------------------------------------------------------------------------


class TestLOCFInterpolation:
    def test_exact_time_points(self):
        """LOCF returns exact value when t matches a time point."""
        tvc = TimeVaryingCovariate(
            name="wt",
            times=jnp.array([0.0, 1.0, 2.0]),
            values=jnp.array([70.0, 75.0, 80.0]),
            method="locf",
        )
        assert float(interpolate_covariate(tvc, 0.0)) == pytest.approx(70.0)
        assert float(interpolate_covariate(tvc, 1.0)) == pytest.approx(75.0)
        assert float(interpolate_covariate(tvc, 2.0)) == pytest.approx(80.0)

    def test_between_time_points(self):
        """LOCF carries forward the most recent value."""
        tvc = TimeVaryingCovariate(
            name="wt",
            times=jnp.array([0.0, 1.0, 2.0]),
            values=jnp.array([70.0, 75.0, 80.0]),
            method="locf",
        )
        # Between 0 and 1 => carry forward 70
        assert float(interpolate_covariate(tvc, 0.5)) == pytest.approx(70.0)
        # Between 1 and 2 => carry forward 75
        assert float(interpolate_covariate(tvc, 1.5)) == pytest.approx(75.0)
        # After 2 => carry forward 80
        assert float(interpolate_covariate(tvc, 3.0)) == pytest.approx(80.0)

    def test_before_first_time_point(self):
        """LOCF before the first time point returns the first value."""
        tvc = TimeVaryingCovariate(
            name="wt",
            times=jnp.array([1.0, 2.0, 3.0]),
            values=jnp.array([70.0, 75.0, 80.0]),
            method="locf",
        )
        assert float(interpolate_covariate(tvc, 0.0)) == pytest.approx(70.0)
        assert float(interpolate_covariate(tvc, -1.0)) == pytest.approx(70.0)


# ---------------------------------------------------------------------------
# interpolate_covariate — linear
# ---------------------------------------------------------------------------


class TestLinearInterpolation:
    def test_midpoint(self):
        """Linear interpolation at the midpoint of two values."""
        tvc = TimeVaryingCovariate(
            name="wt",
            times=jnp.array([0.0, 2.0]),
            values=jnp.array([70.0, 80.0]),
            method="linear",
        )
        assert float(interpolate_covariate(tvc, 1.0)) == pytest.approx(75.0)

    def test_exact_points(self):
        """Linear interpolation returns exact values at knots."""
        tvc = TimeVaryingCovariate(
            name="wt",
            times=jnp.array([0.0, 1.0, 3.0]),
            values=jnp.array([70.0, 75.0, 85.0]),
            method="linear",
        )
        assert float(interpolate_covariate(tvc, 0.0)) == pytest.approx(70.0)
        assert float(interpolate_covariate(tvc, 1.0)) == pytest.approx(75.0)
        assert float(interpolate_covariate(tvc, 3.0)) == pytest.approx(85.0)

    def test_extrapolation(self):
        """Linear extrapolation clamps to boundary values."""
        tvc = TimeVaryingCovariate(
            name="wt",
            times=jnp.array([1.0, 3.0]),
            values=jnp.array([70.0, 80.0]),
            method="linear",
        )
        # Before first time => clamp to first value
        assert float(interpolate_covariate(tvc, 0.0)) == pytest.approx(70.0)
        # After last time => clamp to last value
        assert float(interpolate_covariate(tvc, 5.0)) == pytest.approx(80.0)


# ---------------------------------------------------------------------------
# build_covariate_function
# ---------------------------------------------------------------------------


class TestBuildCovariateFunction:
    def test_multiple_covariates(self):
        """build_covariate_function returns a callable giving a dict of values."""
        tvc_wt = TimeVaryingCovariate(
            name="wt",
            times=jnp.array([0.0, 2.0]),
            values=jnp.array([70.0, 80.0]),
            method="locf",
        )
        tvc_alb = TimeVaryingCovariate(
            name="alb",
            times=jnp.array([0.0, 1.0]),
            values=jnp.array([4.0, 3.5]),
            method="linear",
        )
        cov_fn = build_covariate_function([tvc_wt, tvc_alb])

        result = cov_fn(0.5)
        assert float(result["wt"]) == pytest.approx(70.0)
        assert float(result["alb"]) == pytest.approx(3.75)

        result2 = cov_fn(2.0)
        assert float(result2["wt"]) == pytest.approx(80.0)
        assert float(result2["alb"]) == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# extract_time_varying
# ---------------------------------------------------------------------------


class TestExtractTimeVarying:
    def test_detects_changing_covariate(self):
        """A covariate that changes over time within a subject is extracted."""
        data = {
            "id": np.array([1, 1, 1, 2, 2]),
            "time": np.array([0.0, 1.0, 2.0, 0.0, 1.0]),
            "wt": np.array([70.0, 75.0, 80.0, 60.0, 60.0]),
        }
        result = extract_time_varying(data, ["wt"], id_column="id")

        # Subject 1 has time-varying wt
        assert 1 in result
        assert len(result[1]) == 1
        tvc = result[1][0]
        assert tvc.name == "wt"
        np.testing.assert_array_equal(tvc.times, [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(tvc.values, [70.0, 75.0, 80.0])

    def test_static_covariate_not_extracted(self):
        """A covariate that is constant within a subject is NOT extracted."""
        data = {
            "id": np.array([1, 1, 1]),
            "time": np.array([0.0, 1.0, 2.0]),
            "wt": np.array([70.0, 70.0, 70.0]),
        }
        result = extract_time_varying(data, ["wt"], id_column="id")

        # Subject 1 has no time-varying covariates
        assert 1 not in result or len(result[1]) == 0
