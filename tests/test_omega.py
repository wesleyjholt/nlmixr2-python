"""Tests for the omega (random-effects) matrix module."""

import jax
import jax.numpy as jnp
import pytest

from nlmixr2.omega import OmegaBlock, block_diagonal, cholesky_factor, omega, sample_etas


# ---------------------------------------------------------------------------
# omega() – diagonal matrix creation
# ---------------------------------------------------------------------------

class TestOmegaDiagonal:
    def test_simple_diagonal(self):
        ob = omega({"eta.ka": 0.1, "eta.cl": 0.2})
        assert isinstance(ob, OmegaBlock)
        expected = jnp.array([[0.1, 0.0], [0.0, 0.2]])
        assert jnp.allclose(ob.matrix, expected)

    def test_parameter_names_preserved(self):
        ob = omega({"eta.ka": 0.1, "eta.cl": 0.2, "eta.v": 0.3})
        assert ob.names == ("eta.ka", "eta.cl", "eta.v")

    def test_single_parameter(self):
        ob = omega({"eta.ka": 0.5})
        assert ob.matrix.shape == (1, 1)
        assert float(ob.matrix[0, 0]) == pytest.approx(0.5)

    def test_matrix_is_jax_array(self):
        ob = omega({"eta.ka": 0.1})
        assert isinstance(ob.matrix, jax.Array)


# ---------------------------------------------------------------------------
# omega() – off-diagonal (covariance) elements
# ---------------------------------------------------------------------------

class TestOmegaOffDiagonal:
    def test_with_covariance(self):
        ob = omega({
            "eta.ka": 0.1,
            ("eta.ka", "eta.cl"): 0.05,
            "eta.cl": 0.2,
        })
        expected = jnp.array([[0.1, 0.05], [0.05, 0.2]])
        assert jnp.allclose(ob.matrix, expected)

    def test_symmetric(self):
        ob = omega({
            "eta.ka": 0.1,
            ("eta.ka", "eta.cl"): 0.03,
            "eta.cl": 0.2,
        })
        assert jnp.allclose(ob.matrix, ob.matrix.T)

    def test_three_params_with_off_diag(self):
        ob = omega({
            "eta.ka": 0.1,
            ("eta.ka", "eta.cl"): 0.02,
            "eta.cl": 0.2,
            "eta.v": 0.3,
        })
        assert ob.matrix.shape == (3, 3)
        assert float(ob.matrix[0, 1]) == pytest.approx(0.02)
        assert float(ob.matrix[1, 0]) == pytest.approx(0.02)
        # eta.v has no off-diagonal entries
        assert float(ob.matrix[0, 2]) == pytest.approx(0.0)
        assert float(ob.matrix[2, 0]) == pytest.approx(0.0)

    def test_multiple_off_diag(self):
        ob = omega({
            "eta.ka": 0.10,
            ("eta.ka", "eta.cl"): 0.02,
            ("eta.ka", "eta.v"): 0.01,
            "eta.cl": 0.20,
            ("eta.cl", "eta.v"): 0.03,
            "eta.v": 0.30,
        })
        assert ob.matrix.shape == (3, 3)
        assert float(ob.matrix[0, 1]) == pytest.approx(0.02)
        assert float(ob.matrix[0, 2]) == pytest.approx(0.01)
        assert float(ob.matrix[1, 2]) == pytest.approx(0.03)
        assert jnp.allclose(ob.matrix, ob.matrix.T)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestOmegaValidation:
    def test_diagonal_must_be_positive(self):
        with pytest.raises(ValueError, match="positive"):
            omega({"eta.ka": -0.1})

    def test_zero_diagonal_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            omega({"eta.ka": 0.0})

    def test_non_psd_rejected(self):
        # covariance larger than geometric mean of variances → not PSD
        with pytest.raises(ValueError, match="positive semi-definite"):
            omega({
                "eta.ka": 0.01,
                ("eta.ka", "eta.cl"): 0.5,
                "eta.cl": 0.01,
            })

    def test_off_diag_references_unknown_param(self):
        with pytest.raises(ValueError, match="unknown"):
            omega({
                "eta.ka": 0.1,
                ("eta.ka", "eta.MISSING"): 0.05,
                "eta.cl": 0.2,
            })


# ---------------------------------------------------------------------------
# block_diagonal()
# ---------------------------------------------------------------------------

class TestBlockDiagonal:
    def test_combine_two_blocks(self):
        b1 = omega({"eta.ka": 0.1})
        b2 = omega({"eta.cl": 0.2, "eta.v": 0.3})
        combined = block_diagonal([b1, b2])
        assert combined.matrix.shape == (3, 3)
        assert combined.names == ("eta.ka", "eta.cl", "eta.v")
        assert float(combined.matrix[0, 0]) == pytest.approx(0.1)
        assert float(combined.matrix[1, 1]) == pytest.approx(0.2)
        assert float(combined.matrix[2, 2]) == pytest.approx(0.3)
        # off-diagonal between blocks is zero
        assert float(combined.matrix[0, 1]) == pytest.approx(0.0)
        assert float(combined.matrix[0, 2]) == pytest.approx(0.0)

    def test_single_block_passthrough(self):
        b = omega({"eta.ka": 0.1, "eta.cl": 0.2})
        combined = block_diagonal([b])
        assert jnp.allclose(combined.matrix, b.matrix)
        assert combined.names == b.names

    def test_preserves_within_block_covariance(self):
        b1 = omega({
            "eta.ka": 0.1,
            ("eta.ka", "eta.cl"): 0.05,
            "eta.cl": 0.2,
        })
        b2 = omega({"eta.v": 0.3})
        combined = block_diagonal([b1, b2])
        assert float(combined.matrix[0, 1]) == pytest.approx(0.05)
        assert float(combined.matrix[1, 0]) == pytest.approx(0.05)
        assert float(combined.matrix[0, 2]) == pytest.approx(0.0)

    def test_duplicate_names_rejected(self):
        b1 = omega({"eta.ka": 0.1})
        b2 = omega({"eta.ka": 0.2})
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            block_diagonal([b1, b2])


# ---------------------------------------------------------------------------
# cholesky_factor()
# ---------------------------------------------------------------------------

class TestCholeskyFactor:
    def test_diagonal_cholesky(self):
        ob = omega({"eta.ka": 0.04, "eta.cl": 0.09})
        L = cholesky_factor(ob)
        assert L.shape == (2, 2)
        # For diagonal, cholesky is sqrt on diagonal
        assert float(L[0, 0]) == pytest.approx(0.2)
        assert float(L[1, 1]) == pytest.approx(0.3)
        assert float(L[0, 1]) == pytest.approx(0.0)

    def test_cholesky_reconstructs_original(self):
        ob = omega({
            "eta.ka": 0.10,
            ("eta.ka", "eta.cl"): 0.02,
            "eta.cl": 0.20,
        })
        L = cholesky_factor(ob)
        reconstructed = L @ L.T
        assert jnp.allclose(reconstructed, ob.matrix, atol=1e-6)

    def test_cholesky_is_lower_triangular(self):
        ob = omega({
            "eta.ka": 0.10,
            ("eta.ka", "eta.cl"): 0.02,
            "eta.cl": 0.20,
        })
        L = cholesky_factor(ob)
        assert jnp.allclose(L, jnp.tril(L))


# ---------------------------------------------------------------------------
# sample_etas()
# ---------------------------------------------------------------------------

class TestSampleEtas:
    def test_output_shape(self):
        ob = omega({"eta.ka": 0.1, "eta.cl": 0.2})
        samples = sample_etas(ob, n=100, key=jax.random.PRNGKey(0))
        assert samples.shape == (100, 2)

    def test_zero_mean(self):
        ob = omega({"eta.ka": 1.0, "eta.cl": 1.0})
        samples = sample_etas(ob, n=50_000, key=jax.random.PRNGKey(42))
        means = jnp.mean(samples, axis=0)
        assert jnp.allclose(means, jnp.zeros(2), atol=0.05)

    def test_approximate_covariance(self):
        ob = omega({
            "eta.ka": 0.10,
            ("eta.ka", "eta.cl"): 0.02,
            "eta.cl": 0.20,
        })
        samples = sample_etas(ob, n=100_000, key=jax.random.PRNGKey(7))
        empirical_cov = jnp.cov(samples.T)
        assert jnp.allclose(empirical_cov, ob.matrix, atol=0.01)

    def test_single_sample(self):
        ob = omega({"eta.ka": 0.1})
        samples = sample_etas(ob, n=1, key=jax.random.PRNGKey(0))
        assert samples.shape == (1, 1)

    def test_reproducible_with_same_key(self):
        ob = omega({"eta.ka": 0.1, "eta.cl": 0.2})
        s1 = sample_etas(ob, n=10, key=jax.random.PRNGKey(0))
        s2 = sample_etas(ob, n=10, key=jax.random.PRNGKey(0))
        assert jnp.allclose(s1, s2)


# ---------------------------------------------------------------------------
# OmegaBlock dataclass attributes
# ---------------------------------------------------------------------------

class TestOmegaBlockAttributes:
    def test_names_is_tuple(self):
        ob = omega({"eta.ka": 0.1, "eta.cl": 0.2})
        assert isinstance(ob.names, tuple)

    def test_size_property(self):
        ob = omega({"eta.ka": 0.1, "eta.cl": 0.2, "eta.v": 0.3})
        assert ob.matrix.shape == (3, 3)
        assert len(ob.names) == 3
