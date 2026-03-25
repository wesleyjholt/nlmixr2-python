"""Pharmacometric diagnostic plot data and optional matplotlib rendering.

All ``*_data`` functions return plain dataclasses containing NumPy arrays,
so they work without matplotlib.  The ``plot_*`` convenience functions
require matplotlib and will raise ``ImportError`` if it is absent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class GOFData:
    """Goodness-of-fit arrays needed for standard GOF diagnostic plots."""

    dv: np.ndarray
    pred: np.ndarray
    ipred: np.ndarray
    res: np.ndarray
    ires: np.ndarray
    cwres: np.ndarray
    time: np.ndarray


@dataclass
class IndividualData:
    """Per-subject arrays for individual-fit plots."""

    subject_id: Any
    time: np.ndarray
    dv: np.ndarray
    pred: np.ndarray
    ipred: np.ndarray


@dataclass
class EtaCovData:
    """Paired eta / covariate arrays for scatter-panel plots."""

    etas: np.ndarray
    covariates: np.ndarray
    eta_names: list[str]
    cov_names: list[str]


@dataclass
class TracePlotData:
    """Iteration-indexed objective and parameter history."""

    objectives: np.ndarray
    param_history: dict[str, np.ndarray]


@dataclass
class VPCPlotData:
    """Pre-extracted VPC arrays for plotting (usable without matplotlib)."""

    observed_time: np.ndarray
    observed_dv: np.ndarray
    sim_time: np.ndarray
    sim_lo: np.ndarray
    sim_median: np.ndarray
    sim_hi: np.ndarray


# ---------------------------------------------------------------------------
# Data-preparation functions
# ---------------------------------------------------------------------------

def gof_data(
    dv: np.ndarray,
    pred: np.ndarray,
    ipred: np.ndarray,
    res: np.ndarray,
    ires: np.ndarray,
    cwres: np.ndarray,
    time: np.ndarray,
) -> GOFData:
    """Bundle arrays into a :class:`GOFData` container."""
    return GOFData(
        dv=np.asarray(dv),
        pred=np.asarray(pred),
        ipred=np.asarray(ipred),
        res=np.asarray(res),
        ires=np.asarray(ires),
        cwres=np.asarray(cwres),
        time=np.asarray(time),
    )


def individual_data(
    data: dict[str, np.ndarray],
    pred: np.ndarray,
    ipred: np.ndarray,
    subject_ids: np.ndarray,
) -> list[IndividualData]:
    """Split population-level arrays into per-subject :class:`IndividualData`.

    Parameters
    ----------
    data : dict
        Must contain ``"id"``, ``"time"``, and ``"dv"`` arrays (all same length).
    pred, ipred : np.ndarray
        Population / individual predictions aligned with ``data``.
    subject_ids : np.ndarray
        Unique subject identifiers to iterate over.
    """
    ids = np.asarray(data["id"])
    time_arr = np.asarray(data["time"])
    dv_arr = np.asarray(data["dv"])
    pred_arr = np.asarray(pred)
    ipred_arr = np.asarray(ipred)

    results: list[IndividualData] = []
    for sid in subject_ids:
        mask = ids == sid
        results.append(
            IndividualData(
                subject_id=sid,
                time=time_arr[mask],
                dv=dv_arr[mask],
                pred=pred_arr[mask],
                ipred=ipred_arr[mask],
            )
        )
    return results


def eta_vs_cov_data(
    etas: np.ndarray,
    covariates: np.ndarray,
    eta_names: list[str],
    cov_names: list[str],
) -> EtaCovData:
    """Bundle eta and covariate matrices into :class:`EtaCovData`."""
    return EtaCovData(
        etas=np.asarray(etas),
        covariates=np.asarray(covariates),
        eta_names=list(eta_names),
        cov_names=list(cov_names),
    )


def traceplot_data(
    objectives: np.ndarray,
    param_history: dict[str, np.ndarray],
) -> TracePlotData:
    """Bundle optimisation trace arrays into :class:`TracePlotData`."""
    return TracePlotData(
        objectives=np.asarray(objectives),
        param_history={k: np.asarray(v) for k, v in param_history.items()},
    )


# ---------------------------------------------------------------------------
# Optional matplotlib rendering
# ---------------------------------------------------------------------------

try:
    import matplotlib  # noqa: E402
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from scipy import stats as _sp_stats

    _HAS_MPL = True
except ImportError:  # pragma: no cover
    _HAS_MPL = False


def _require_mpl() -> None:
    if not _HAS_MPL:
        raise ImportError(
            "matplotlib (and scipy) are required for plot rendering.  "
            "Install them with:  pip install matplotlib scipy"
        )


def plot_gof(gof: GOFData, figsize: tuple[int, int] = (12, 10)) -> "Figure":
    """Render a four-panel goodness-of-fit figure.

    Panels
    ------
    1. DV vs PRED  (with identity line)
    2. DV vs IPRED (with identity line)
    3. CWRES vs TIME
    4. CWRES QQ plot
    """
    _require_mpl()

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Panel 1: DV vs PRED
    ax = axes[0, 0]
    ax.scatter(gof.pred, gof.dv, s=12, alpha=0.6)
    lims = _combined_lims(gof.pred, gof.dv)
    ax.plot(lims, lims, "k--", linewidth=0.8)
    ax.set_xlabel("PRED")
    ax.set_ylabel("DV")
    ax.set_title("DV vs PRED")

    # Panel 2: DV vs IPRED
    ax = axes[0, 1]
    ax.scatter(gof.ipred, gof.dv, s=12, alpha=0.6)
    lims = _combined_lims(gof.ipred, gof.dv)
    ax.plot(lims, lims, "k--", linewidth=0.8)
    ax.set_xlabel("IPRED")
    ax.set_ylabel("DV")
    ax.set_title("DV vs IPRED")

    # Panel 3: CWRES vs TIME
    ax = axes[1, 0]
    ax.scatter(gof.time, gof.cwres, s=12, alpha=0.6)
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xlabel("TIME")
    ax.set_ylabel("CWRES")
    ax.set_title("CWRES vs TIME")

    # Panel 4: CWRES QQ
    ax = axes[1, 1]
    _sp_stats.probplot(np.asarray(gof.cwres, dtype=float), plot=ax)
    ax.set_title("CWRES QQ Plot")

    fig.tight_layout()
    return fig


def plot_individual(
    ind_data: list[IndividualData],
    n_cols: int = 3,
    figsize_per_panel: tuple[float, float] = (4.0, 3.0),
) -> "Figure":
    """Render individual-fit panels (DV, PRED, IPRED vs TIME)."""
    _require_mpl()

    n = len(ind_data)
    n_cols = min(n_cols, n)
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False,
    )

    for idx, item in enumerate(ind_data):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        ax.plot(item.time, item.dv, "o", label="DV", markersize=4)
        ax.plot(item.time, item.pred, "-", label="PRED", linewidth=1)
        ax.plot(item.time, item.ipred, "--", label="IPRED", linewidth=1)
        ax.set_title(f"Subject {item.subject_id}")
        ax.set_xlabel("TIME")
        ax.set_ylabel("Concentration")
        if idx == 0:
            ax.legend(fontsize="small")

    # Hide unused axes
    for idx in range(n, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)

    fig.tight_layout()
    return fig


def plot_eta_cov(ecd: EtaCovData, figsize_per_panel: tuple[float, float] = (3.5, 3.0)) -> "Figure":
    """Scatter-panel grid of each eta vs each covariate."""
    _require_mpl()

    n_eta = len(ecd.eta_names)
    n_cov = len(ecd.cov_names)
    fig, axes = plt.subplots(
        n_eta,
        n_cov,
        figsize=(figsize_per_panel[0] * n_cov, figsize_per_panel[1] * n_eta),
        squeeze=False,
    )

    for i, eta_name in enumerate(ecd.eta_names):
        for j, cov_name in enumerate(ecd.cov_names):
            ax = axes[i][j]
            ax.scatter(ecd.covariates[:, j], ecd.etas[:, i], s=12, alpha=0.6)
            ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
            ax.set_xlabel(cov_name)
            ax.set_ylabel(eta_name)

    fig.tight_layout()
    return fig


def plot_vpc(
    vpc_result: Any,
    title: str = "Visual Predictive Check",
    figsize: tuple[int, int] = (10, 6),
) -> "Figure":
    """Render a Visual Predictive Check plot.

    Parameters
    ----------
    vpc_result
        A :class:`~nlmixr2.vpc.VPCResult` instance (or any object with
        ``observed`` and ``simulated_quantiles`` dict attributes).
    title
        Plot title.
    figsize
        Figure dimensions ``(width, height)`` in inches.

    Returns
    -------
    Figure
        A matplotlib Figure with observed data, simulated median, and
        prediction-interval shaded band.
    """
    _require_mpl()

    obs = vpc_result.observed
    sim = vpc_result.simulated_quantiles

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Observed data as scatter points
    ax.scatter(obs["time"], obs["dv"], s=14, alpha=0.4, color="C0", label="Observed", zorder=3)

    # Simulated prediction interval band
    ax.fill_between(
        sim["time"], sim["lo"], sim["hi"],
        alpha=0.25, color="C1", label="Prediction interval",
    )

    # Simulated median line
    ax.plot(sim["time"], sim["median"], color="C1", linewidth=1.5, label="Simulated median")

    # Unity / reference line (y = x identity through the data range)
    all_vals = np.concatenate([obs["dv"], sim["lo"], sim["hi"]])
    ref_lo, ref_hi = float(np.min(all_vals)), float(np.max(all_vals))
    ax.axhline(np.median(obs["dv"]), color="k", linewidth=0.8, linestyle="--", label="Reference")

    ax.set_xlabel("Time")
    ax.set_ylabel("DV / Concentration")
    ax.set_title(title)
    ax.legend(fontsize="small")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _combined_lims(a: np.ndarray, b: np.ndarray) -> list[float]:
    """Return ``[min, max]`` spanning both arrays (for identity lines)."""
    lo = float(min(np.min(a), np.min(b)))
    hi = float(max(np.max(a), np.max(b)))
    return [lo, hi]
