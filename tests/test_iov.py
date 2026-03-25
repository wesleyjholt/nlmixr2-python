"""Tests for inter-occasion variability (IOV) support."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nlmixr2.omega import OmegaBlock, omega
from nlmixr2.iov import (
    IOVSpec,
    apply_iov,
    expand_omega_with_iov,
    extract_occasions,
    sample_iov_etas,
)


# ---------------------------------------------------------------------------
# IOVSpec creation
# ---------------------------------------------------------------------------

class TestIOVSpec:
    def test_creation(self):
        omega_iov = omega({"eta.ka": 0.1, "eta.cl": 0.2})
        spec = IOVSpec(
            occasion_column="occ",
            omega_iov=omega_iov,
            parameter_names=("ka", "cl"),
        )
        assert spec.occasion_column == "occ"
        assert spec.omega_iov is omega_iov
        assert spec.parameter_names == ("ka", "cl")

    def test_frozen(self):
        omega_iov = omega({"eta.ka": 0.1})
        spec = IOVSpec(
            occasion_column="occ",
            omega_iov=omega_iov,
            parameter_names=("ka",),
        )
        with pytest.raises(AttributeError):
            spec.occasion_column = "other"


# ---------------------------------------------------------------------------
# extract_occasions
# ---------------------------------------------------------------------------

class TestExtractOccasions:
    def test_two_occasions_per_subject(self):
        data = {
            "id": jnp.array([1, 1, 1, 1, 2, 2, 2, 2]),
            "occ": jnp.array([0, 0, 1, 1, 0, 0, 1, 1]),
        }
        result = extract_occasions(data, "occ")
        # Subject 1 and 2 each have occasions [0, 1]
        assert set(result.keys()) == {1, 2}
        assert result[1] == [0, 1]
        assert result[2] == [0, 1]

    def test_unequal_occasions(self):
        data = {
            "id": jnp.array([1, 1, 1, 2, 2]),
            "occ": jnp.array([0, 1, 2, 0, 1]),
        }
        result = extract_occasions(data, "occ")
        assert result[1] == [0, 1, 2]
        assert result[2] == [0, 1]

    def test_single_occasion(self):
        data = {
            "id": jnp.array([1, 1, 2, 2]),
            "occ": jnp.array([0, 0, 0, 0]),
        }
        result = extract_occasions(data, "occ")
        assert result[1] == [0]
        assert result[2] == [0]


# ---------------------------------------------------------------------------
# sample_iov_etas
# ---------------------------------------------------------------------------

class TestSampleIOVEtas:
    def test_shape(self):
        omega_iov = omega({"eta.ka": 0.1, "eta.cl": 0.2})
        spec = IOVSpec(
            occasion_column="occ",
            omega_iov=omega_iov,
            parameter_names=("ka", "cl"),
        )
        result = sample_iov_etas(spec, n_subjects=5, n_occasions=3, seed=42)
        assert result.shape == (5, 3, 2)

    def test_single_param(self):
        omega_iov = omega({"eta.ka": 0.1})
        spec = IOVSpec(
            occasion_column="occ",
            omega_iov=omega_iov,
            parameter_names=("ka",),
        )
        result = sample_iov_etas(spec, n_subjects=2, n_occasions=4, seed=0)
        assert result.shape == (2, 4, 1)

    def test_reproducibility_with_same_seed(self):
        omega_iov = omega({"eta.ka": 0.1, "eta.cl": 0.2})
        spec = IOVSpec(
            occasion_column="occ",
            omega_iov=omega_iov,
            parameter_names=("ka", "cl"),
        )
        r1 = sample_iov_etas(spec, n_subjects=3, n_occasions=2, seed=99)
        r2 = sample_iov_etas(spec, n_subjects=3, n_occasions=2, seed=99)
        assert jnp.allclose(r1, r2)

    def test_different_seeds_differ(self):
        omega_iov = omega({"eta.ka": 0.1})
        spec = IOVSpec(
            occasion_column="occ",
            omega_iov=omega_iov,
            parameter_names=("ka",),
        )
        r1 = sample_iov_etas(spec, n_subjects=3, n_occasions=2, seed=0)
        r2 = sample_iov_etas(spec, n_subjects=3, n_occasions=2, seed=1)
        assert not jnp.allclose(r1, r2)


# ---------------------------------------------------------------------------
# apply_iov
# ---------------------------------------------------------------------------

class TestApplyIOV:
    def test_adds_both_bsv_and_iov(self):
        params = {"ka": 1.0, "cl": 2.0, "v": 10.0}
        bsv_etas = jnp.array([0.1, 0.2])  # for ka, cl
        iov_etas = jnp.array([0.05, -0.03])  # for ka, cl at this occasion
        bsv_names = ("ka", "cl")
        iov_names = ("ka", "cl")

        result = apply_iov(params, bsv_etas, iov_etas, bsv_names, iov_names)
        # ka = 1.0 + 0.1 (BSV) + 0.05 (IOV) = 1.15
        assert float(result["ka"]) == pytest.approx(1.15)
        # cl = 2.0 + 0.2 (BSV) + (-0.03) (IOV) = 2.17
        assert float(result["cl"]) == pytest.approx(2.17)
        # v is unaffected
        assert float(result["v"]) == pytest.approx(10.0)

    def test_iov_subset_of_bsv(self):
        """IOV may affect only a subset of the BSV parameters."""
        params = {"ka": 1.0, "cl": 2.0}
        bsv_etas = jnp.array([0.1, 0.2])  # for ka, cl
        iov_etas = jnp.array([0.05])  # IOV only for ka
        bsv_names = ("ka", "cl")
        iov_names = ("ka",)

        result = apply_iov(params, bsv_etas, iov_etas, bsv_names, iov_names)
        assert float(result["ka"]) == pytest.approx(1.15)
        # cl gets BSV but no IOV
        assert float(result["cl"]) == pytest.approx(2.2)

    def test_iov_etas_differ_across_occasions_bsv_constant(self):
        """IOV etas should differ across occasions while BSV stays the same."""
        omega_iov = omega({"eta.ka": 0.1})
        spec = IOVSpec(
            occasion_column="occ",
            omega_iov=omega_iov,
            parameter_names=("ka",),
        )
        iov_samples = sample_iov_etas(spec, n_subjects=1, n_occasions=3, seed=42)
        # shape: (1, 3, 1)
        bsv_eta = jnp.array([0.5])  # constant BSV for subject
        params = {"ka": 1.0}

        results = []
        for occ in range(3):
            r = apply_iov(
                params, bsv_eta, iov_samples[0, occ, :],
                bsv_names=("ka",), iov_names=("ka",),
            )
            results.append(float(r["ka"]))

        # BSV contribution is constant (0.5), IOV differs per occasion
        # So results should differ from each other
        assert not (results[0] == pytest.approx(results[1]))
        assert not (results[0] == pytest.approx(results[2]))

        # But BSV component is the same: result - iov = params + bsv for all
        for occ in range(3):
            bsv_component = float(params["ka"]) + float(bsv_eta[0])
            iov_component = float(iov_samples[0, occ, 0])
            assert results[occ] == pytest.approx(bsv_component + iov_component)


# ---------------------------------------------------------------------------
# expand_omega_with_iov
# ---------------------------------------------------------------------------

class TestExpandOmegaWithIOV:
    def test_block_structure(self):
        omega_bsv = omega({"eta.ka": 0.1, "eta.cl": 0.2})
        omega_iov = omega({"eta.iov.ka": 0.05})
        iov_spec = IOVSpec(
            occasion_column="occ",
            omega_iov=omega_iov,
            parameter_names=("ka",),
        )
        combined = expand_omega_with_iov(omega_bsv, iov_spec, n_occasions=2)
        # BSV: 2x2, IOV: 1x1 repeated 2 occasions = total 4x4
        assert combined.matrix.shape == (4, 4)
        # BSV block in top-left
        assert float(combined.matrix[0, 0]) == pytest.approx(0.1)
        assert float(combined.matrix[1, 1]) == pytest.approx(0.2)
        # IOV blocks on diagonal
        assert float(combined.matrix[2, 2]) == pytest.approx(0.05)
        assert float(combined.matrix[3, 3]) == pytest.approx(0.05)
        # Cross-block zeros
        assert float(combined.matrix[0, 2]) == pytest.approx(0.0)
        assert float(combined.matrix[2, 0]) == pytest.approx(0.0)
        assert float(combined.matrix[2, 3]) == pytest.approx(0.0)

    def test_multi_param_iov(self):
        omega_bsv = omega({"eta.ka": 0.1})
        omega_iov = omega({
            "eta.iov.ka": 0.05,
            "eta.iov.cl": 0.03,
        })
        iov_spec = IOVSpec(
            occasion_column="occ",
            omega_iov=omega_iov,
            parameter_names=("ka", "cl"),
        )
        combined = expand_omega_with_iov(omega_bsv, iov_spec, n_occasions=2)
        # BSV: 1x1, IOV: 2x2 repeated 2 times = total 1 + 2*2 = 5x5
        assert combined.matrix.shape == (5, 5)

    def test_single_occasion_reduces_to_bsv_plus_one_iov_block(self):
        omega_bsv = omega({"eta.ka": 0.1})
        omega_iov = omega({"eta.iov.ka": 0.05})
        iov_spec = IOVSpec(
            occasion_column="occ",
            omega_iov=omega_iov,
            parameter_names=("ka",),
        )
        combined = expand_omega_with_iov(omega_bsv, iov_spec, n_occasions=1)
        # BSV: 1x1, IOV: 1x1 * 1 occasion = total 2x2
        assert combined.matrix.shape == (2, 2)
        assert float(combined.matrix[0, 0]) == pytest.approx(0.1)
        assert float(combined.matrix[1, 1]) == pytest.approx(0.05)
        assert float(combined.matrix[0, 1]) == pytest.approx(0.0)

    def test_names_include_occasion_index(self):
        omega_bsv = omega({"eta.ka": 0.1})
        omega_iov = omega({"eta.iov.ka": 0.05})
        iov_spec = IOVSpec(
            occasion_column="occ",
            omega_iov=omega_iov,
            parameter_names=("ka",),
        )
        combined = expand_omega_with_iov(omega_bsv, iov_spec, n_occasions=2)
        assert combined.names[0] == "eta.ka"
        # IOV names should include occasion index
        assert "occ0" in combined.names[1] or "0" in combined.names[1]
        assert "occ1" in combined.names[2] or "1" in combined.names[2]
