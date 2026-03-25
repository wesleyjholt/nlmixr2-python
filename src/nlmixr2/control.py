"""Structured control objects for nlmixr2 estimators.

Provides validated dataclass-based configuration equivalent to R nlmixr2's
``foceiControl()``, ``saemControl()``, etc.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields


# ---------------------------------------------------------------------------
# Base mixin
# ---------------------------------------------------------------------------

class _ControlMixin:
    """Shared helpers for all control dataclasses."""

    def to_dict(self) -> dict:
        """Return a plain dict of all control fields."""
        return asdict(self)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# FoceiControl
# ---------------------------------------------------------------------------

@dataclass
class FoceiControl(_ControlMixin):
    """Control parameters for the FOCE-I estimator.

    Attributes
    ----------
    maxiter : int
        Maximum number of outer iterations.
    tol : float
        Convergence tolerance on relative objective change.
    lr : float
        Learning rate for fixed-effect parameters.
    lr_eta : float
        Learning rate for random-effect (eta) updates.
    inner_steps : int
        Number of inner optimisation steps per outer iteration.
    sigma : float
        Residual error variance.
    print_progress : bool
        Whether to print iteration progress.
    """

    maxiter: int = 500
    tol: float = 1e-6
    lr: float = 0.01
    lr_eta: float = 0.05
    inner_steps: int = 10
    sigma: float = 1.0
    print_progress: bool = False

    def __post_init__(self) -> None:
        if self.maxiter <= 0:
            raise ValueError("maxiter must be positive")
        if self.tol < 0:
            raise ValueError("tol must be non-negative")
        if self.lr < 0:
            raise ValueError("lr must be non-negative")
        if self.lr_eta < 0:
            raise ValueError("lr_eta must be non-negative")
        if self.inner_steps <= 0:
            raise ValueError("inner_steps must be positive")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")


# ---------------------------------------------------------------------------
# SaemControl
# ---------------------------------------------------------------------------

@dataclass
class SaemControl(_ControlMixin):
    """Control parameters for the SAEM estimator.

    Attributes
    ----------
    n_burn : int
        Number of burn-in iterations.
    n_em : int
        Number of EM iterations after burn-in.
    n_mcmc : int
        Number of MCMC chains per E-step.
    step_size : float
        Step-size for stochastic approximation.
    sigma : float
        Residual error variance.
    print_progress : bool
        Whether to print iteration progress.
    """

    n_burn: int = 300
    n_em: int = 200
    n_mcmc: int = 3
    step_size: float = 1.0
    sigma: float = 1.0
    print_progress: bool = False

    def __post_init__(self) -> None:
        if self.n_burn < 0:
            raise ValueError("n_burn must be non-negative")
        if self.n_em < 0:
            raise ValueError("n_em must be non-negative")
        if self.n_mcmc <= 0:
            raise ValueError("n_mcmc must be positive")
        if self.step_size < 0:
            raise ValueError("step_size must be non-negative")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")


# ---------------------------------------------------------------------------
# MockControl
# ---------------------------------------------------------------------------

@dataclass
class MockControl(_ControlMixin):
    """Minimal control object for the mock estimator."""

    pass


# ---------------------------------------------------------------------------
# Factory functions (mirroring R's foceiControl(), saemControl())
# ---------------------------------------------------------------------------

def foceiControl(**kwargs) -> FoceiControl:
    """Create a :class:`FoceiControl` with validated parameters.

    Raises ``TypeError`` for unknown keyword arguments (handled by the
    dataclass constructor).
    """
    return FoceiControl(**kwargs)


def saemControl(**kwargs) -> SaemControl:
    """Create a :class:`SaemControl` with validated parameters.

    Raises ``TypeError`` for unknown keyword arguments (handled by the
    dataclass constructor).
    """
    return SaemControl(**kwargs)
