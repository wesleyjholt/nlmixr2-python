"""Tests for nlmixr2.plots – data-preparation and optional rendering."""

from __future__ import annotations

import numpy as np
import pytest

from nlmixr2.plots import (
    GOFData,
    IndividualData,
    EtaCovData,
    TracePlotData,
    VPCPlotData,
    gof_data,
    individual_data,
    eta_vs_cov_data,
    traceplot_data,
)


# ---------------------------------------------------------------------------
# gof_data
# ---------------------------------------------------------------------------

class TestGofData:
    def test_returns_gofdata(self):
        n = 20
        dv = np.random.default_rng(0).normal(10, 1, n)
        pred = dv + np.random.default_rng(1).normal(0, 0.5, n)
        ipred = dv + np.random.default_rng(2).normal(0, 0.3, n)
        res = dv - pred
        ires = dv - ipred
        cwres = res / res.std()
        time = np.linspace(0, 24, n)

        result = gof_data(dv, pred, ipred, res, ires, cwres, time)

        assert isinstance(result, GOFData)
        np.testing.assert_array_equal(result.dv, dv)
        np.testing.assert_array_equal(result.pred, pred)
        np.testing.assert_array_equal(result.ipred, ipred)
        np.testing.assert_array_equal(result.res, res)
        np.testing.assert_array_equal(result.ires, ires)
        np.testing.assert_array_equal(result.cwres, cwres)
        np.testing.assert_array_equal(result.time, time)

    def test_single_observation(self):
        """Edge case: single data point."""
        result = gof_data(
            dv=np.array([5.0]),
            pred=np.array([4.8]),
            ipred=np.array([5.1]),
            res=np.array([0.2]),
            ires=np.array([-0.1]),
            cwres=np.array([0.15]),
            time=np.array([0.0]),
        )
        assert isinstance(result, GOFData)
        assert len(result.dv) == 1


# ---------------------------------------------------------------------------
# individual_data
# ---------------------------------------------------------------------------

class TestIndividualData:
    def _make_inputs(self, n_subjects=4, n_obs_per=5):
        rng = np.random.default_rng(42)
        ids = np.repeat(np.arange(1, n_subjects + 1), n_obs_per)
        time = np.tile(np.linspace(0, 24, n_obs_per), n_subjects)
        dv = rng.normal(10, 1, len(ids))
        pred = dv + rng.normal(0, 0.5, len(ids))
        ipred = dv + rng.normal(0, 0.3, len(ids))
        data = {"id": ids, "time": time, "dv": dv}
        return data, pred, ipred, ids

    def test_returns_list_of_correct_length(self):
        data, pred, ipred, ids = self._make_inputs(n_subjects=4)
        subject_ids = np.unique(ids)
        result = individual_data(data, pred, ipred, subject_ids)

        assert isinstance(result, list)
        assert len(result) == 4
        assert all(isinstance(r, IndividualData) for r in result)

    def test_per_subject_arrays(self):
        data, pred, ipred, ids = self._make_inputs(n_subjects=3, n_obs_per=6)
        subject_ids = np.unique(ids)
        result = individual_data(data, pred, ipred, subject_ids)

        for item in result:
            assert len(item.time) == 6
            assert len(item.dv) == 6
            assert len(item.pred) == 6
            assert len(item.ipred) == 6

    def test_single_subject(self):
        """Edge case: only one subject."""
        data, pred, ipred, ids = self._make_inputs(n_subjects=1, n_obs_per=3)
        subject_ids = np.unique(ids)
        result = individual_data(data, pred, ipred, subject_ids)
        assert len(result) == 1

    def test_single_timepoint(self):
        """Edge case: one observation per subject."""
        data, pred, ipred, ids = self._make_inputs(n_subjects=2, n_obs_per=1)
        subject_ids = np.unique(ids)
        result = individual_data(data, pred, ipred, subject_ids)
        assert len(result) == 2
        assert len(result[0].time) == 1


# ---------------------------------------------------------------------------
# eta_vs_cov_data
# ---------------------------------------------------------------------------

class TestEtaCovData:
    def test_basic(self):
        rng = np.random.default_rng(7)
        n = 10
        etas = rng.normal(0, 1, (n, 2))
        covariates = rng.normal(50, 10, (n, 3))
        eta_names = ["eta.Ka", "eta.CL"]
        cov_names = ["WT", "AGE", "SEX"]

        result = eta_vs_cov_data(etas, covariates, eta_names, cov_names)

        assert isinstance(result, EtaCovData)
        np.testing.assert_array_equal(result.etas, etas)
        np.testing.assert_array_equal(result.covariates, covariates)
        assert result.eta_names == eta_names
        assert result.cov_names == cov_names

    def test_single_eta_single_cov(self):
        etas = np.array([[0.1], [0.2]])
        covariates = np.array([[70.0], [80.0]])
        result = eta_vs_cov_data(etas, covariates, ["eta.CL"], ["WT"])
        assert result.etas.shape == (2, 1)
        assert result.covariates.shape == (2, 1)


# ---------------------------------------------------------------------------
# traceplot_data
# ---------------------------------------------------------------------------

class TestTracePlotData:
    def test_fields(self):
        objectives = np.array([100.0, 90.0, 85.0, 82.0])
        param_history = {
            "CL": np.array([1.0, 1.1, 1.15, 1.12]),
            "V": np.array([20.0, 19.5, 19.8, 19.7]),
        }
        result = traceplot_data(objectives, param_history)

        assert isinstance(result, TracePlotData)
        np.testing.assert_array_equal(result.objectives, objectives)
        assert list(result.param_history.keys()) == ["CL", "V"]
        np.testing.assert_array_equal(result.param_history["CL"], param_history["CL"])

    def test_single_iteration(self):
        result = traceplot_data(
            objectives=np.array([50.0]),
            param_history={"CL": np.array([1.0])},
        )
        assert len(result.objectives) == 1


# ---------------------------------------------------------------------------
# Matplotlib rendering (skip if not installed)
# ---------------------------------------------------------------------------

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
class TestPlotGof:
    def _make_gof(self, n=50):
        rng = np.random.default_rng(0)
        dv = rng.normal(10, 1, n)
        pred = dv + rng.normal(0, 0.5, n)
        ipred = dv + rng.normal(0, 0.3, n)
        res = dv - pred
        ires = dv - ipred
        cwres = rng.normal(0, 1, n)
        time = np.linspace(0, 24, n)
        return gof_data(dv, pred, ipred, res, ires, cwres, time)

    def test_returns_figure(self):
        from nlmixr2.plots import plot_gof
        fig = plot_gof(self._make_gof())
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_figsize(self):
        from nlmixr2.plots import plot_gof
        fig = plot_gof(self._make_gof(), figsize=(8, 6))
        assert isinstance(fig, Figure)
        plt.close(fig)


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
class TestPlotIndividual:
    def test_returns_figure(self):
        from nlmixr2.plots import plot_individual
        rng = np.random.default_rng(42)
        n_sub, n_obs = 4, 5
        ids = np.repeat(np.arange(1, n_sub + 1), n_obs)
        time = np.tile(np.linspace(0, 24, n_obs), n_sub)
        dv = rng.normal(10, 1, len(ids))
        pred = dv + 0.5
        ipred = dv + 0.2
        data = {"id": ids, "time": time, "dv": dv}
        ind_data = individual_data(data, pred, ipred, np.unique(ids))
        fig = plot_individual(ind_data)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_single_subject(self):
        from nlmixr2.plots import plot_individual
        data = {"id": np.array([1, 1, 1]), "time": np.array([0, 1, 2]), "dv": np.array([5, 6, 7])}
        pred = np.array([5.1, 5.9, 7.1])
        ipred = np.array([5.0, 6.0, 7.0])
        ind_data = individual_data(data, pred, ipred, np.array([1]))
        fig = plot_individual(ind_data)
        assert isinstance(fig, Figure)
        plt.close(fig)


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
class TestPlotEtaCov:
    def test_returns_figure(self):
        from nlmixr2.plots import plot_eta_cov
        rng = np.random.default_rng(7)
        ecd = eta_vs_cov_data(
            rng.normal(0, 1, (10, 2)),
            rng.normal(50, 10, (10, 2)),
            ["eta.Ka", "eta.CL"],
            ["WT", "AGE"],
        )
        fig = plot_eta_cov(ecd)
        assert isinstance(fig, Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# VPCPlotData
# ---------------------------------------------------------------------------

class TestVPCPlotData:
    def test_creation(self):
        t = np.linspace(0, 24, 10)
        dv = np.random.default_rng(0).normal(5, 1, 10)
        vpd = VPCPlotData(
            observed_time=t,
            observed_dv=dv,
            sim_time=t,
            sim_lo=dv - 1,
            sim_median=dv,
            sim_hi=dv + 1,
        )
        assert isinstance(vpd, VPCPlotData)
        np.testing.assert_array_equal(vpd.observed_time, t)
        np.testing.assert_array_equal(vpd.observed_dv, dv)
        np.testing.assert_array_equal(vpd.sim_median, dv)
        assert len(vpd.sim_lo) == 10
        assert len(vpd.sim_hi) == 10

    def test_single_point(self):
        vpd = VPCPlotData(
            observed_time=np.array([0.0]),
            observed_dv=np.array([5.0]),
            sim_time=np.array([0.0]),
            sim_lo=np.array([4.0]),
            sim_median=np.array([5.0]),
            sim_hi=np.array([6.0]),
        )
        assert len(vpd.sim_time) == 1


# ---------------------------------------------------------------------------
# plot_vpc (skip if matplotlib not installed)
# ---------------------------------------------------------------------------

def _make_vpc_result():
    """Create a minimal VPCResult-like object for testing."""
    from nlmixr2.vpc import VPCResult

    rng = np.random.default_rng(99)
    n_obs = 30
    n_bins = 8
    times = np.linspace(0, 24, n_obs)
    dv = rng.normal(10, 1, n_obs)
    bin_centers = np.linspace(0, 24, n_bins)
    median_vals = np.full(n_bins, 10.0)

    return VPCResult(
        observed={"time": times, "dv": dv},
        simulated_quantiles={
            "time": bin_centers,
            "lo": median_vals - 2.0,
            "median": median_vals,
            "hi": median_vals + 2.0,
        },
        pi=(0.05, 0.5, 0.95),
        n_sim=200,
    )


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
class TestPlotVpc:
    def test_returns_figure(self):
        from nlmixr2.plots import plot_vpc
        vpc_result = _make_vpc_result()
        fig = plot_vpc(vpc_result)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_title_and_figsize(self):
        from nlmixr2.plots import plot_vpc
        vpc_result = _make_vpc_result()
        fig = plot_vpc(vpc_result, title="My VPC", figsize=(8, 5))
        assert isinstance(fig, Figure)
        # Verify the title propagated
        ax = fig.axes[0]
        assert ax.get_title() == "My VPC"
        plt.close(fig)

    def test_minimal_vpc_result(self):
        """VPC result with a single bin / single observation."""
        from nlmixr2.plots import plot_vpc
        from nlmixr2.vpc import VPCResult

        vpc_result = VPCResult(
            observed={"time": np.array([1.0]), "dv": np.array([5.0])},
            simulated_quantiles={
                "time": np.array([1.0]),
                "lo": np.array([4.0]),
                "median": np.array([5.0]),
                "hi": np.array([6.0]),
            },
            pi=(0.05, 0.5, 0.95),
            n_sim=10,
        )
        fig = plot_vpc(vpc_result)
        assert isinstance(fig, Figure)
        plt.close(fig)
