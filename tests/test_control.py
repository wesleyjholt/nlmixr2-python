"""Tests for estimator control objects (TDD – written before implementation)."""

import importlib
import sys
from pathlib import Path

import pytest

# Import control module directly to avoid circular-import issues in __init__.py.
_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

# Temporarily remove nlmixr2 from sys.modules so we can import the sub-module
# without triggering __init__.py's circular imports.
_saved = sys.modules.pop("nlmixr2", None)
_control = importlib.import_module("nlmixr2.control")
if _saved is not None:
    sys.modules["nlmixr2"] = _saved

FoceiControl = _control.FoceiControl
MockControl = _control.MockControl
SaemControl = _control.SaemControl
foceiControl = _control.foceiControl
saemControl = _control.saemControl


# ── FoceiControl defaults ───────────────────────────────────────────

class TestFoceiControlDefaults:
    def test_maxiter_default(self):
        c = FoceiControl()
        assert c.maxiter == 500

    def test_tol_default(self):
        c = FoceiControl()
        assert c.tol == 1e-6

    def test_lr_default(self):
        c = FoceiControl()
        assert c.lr == 0.01

    def test_lr_eta_default(self):
        c = FoceiControl()
        assert c.lr_eta == 0.05

    def test_inner_steps_default(self):
        c = FoceiControl()
        assert c.inner_steps == 10

    def test_sigma_default(self):
        c = FoceiControl()
        assert c.sigma == 1.0

    def test_print_progress_default(self):
        c = FoceiControl()
        assert c.print_progress is False


# ── FoceiControl custom values ──────────────────────────────────────

class TestFoceiControlCustom:
    def test_custom_maxiter(self):
        c = FoceiControl(maxiter=1000)
        assert c.maxiter == 1000

    def test_custom_tol(self):
        c = FoceiControl(tol=1e-8)
        assert c.tol == 1e-8

    def test_custom_lr(self):
        c = FoceiControl(lr=0.1)
        assert c.lr == 0.1

    def test_custom_lr_eta(self):
        c = FoceiControl(lr_eta=0.001)
        assert c.lr_eta == 0.001

    def test_custom_inner_steps(self):
        c = FoceiControl(inner_steps=50)
        assert c.inner_steps == 50

    def test_custom_sigma(self):
        c = FoceiControl(sigma=2.5)
        assert c.sigma == 2.5

    def test_custom_print_progress(self):
        c = FoceiControl(print_progress=True)
        assert c.print_progress is True


# ── FoceiControl validation ─────────────────────────────────────────

class TestFoceiControlValidation:
    def test_negative_maxiter_raises(self):
        with pytest.raises(ValueError, match="maxiter"):
            FoceiControl(maxiter=-1)

    def test_zero_maxiter_raises(self):
        with pytest.raises(ValueError, match="maxiter"):
            FoceiControl(maxiter=0)

    def test_negative_tol_raises(self):
        with pytest.raises(ValueError, match="tol"):
            FoceiControl(tol=-0.001)

    def test_zero_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            FoceiControl(sigma=0.0)

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            FoceiControl(sigma=-1.0)

    def test_negative_lr_raises(self):
        with pytest.raises(ValueError, match="lr"):
            FoceiControl(lr=-0.01)

    def test_negative_lr_eta_raises(self):
        with pytest.raises(ValueError, match="lr_eta"):
            FoceiControl(lr_eta=-0.01)

    def test_zero_inner_steps_raises(self):
        with pytest.raises(ValueError, match="inner_steps"):
            FoceiControl(inner_steps=0)

    def test_negative_inner_steps_raises(self):
        with pytest.raises(ValueError, match="inner_steps"):
            FoceiControl(inner_steps=-5)


# ── FoceiControl to_dict / roundtrip ────────────────────────────────

class TestFoceiControlToDict:
    def test_to_dict_returns_dict(self):
        c = FoceiControl()
        d = c.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_keys(self):
        c = FoceiControl()
        d = c.to_dict()
        expected_keys = {
            "maxiter", "tol", "lr", "lr_eta",
            "inner_steps", "sigma", "print_progress",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values_match(self):
        c = FoceiControl(maxiter=200, sigma=3.0)
        d = c.to_dict()
        assert d["maxiter"] == 200
        assert d["sigma"] == 3.0

    def test_roundtrip(self):
        original = FoceiControl(maxiter=250, lr=0.05, sigma=2.0)
        d = original.to_dict()
        restored = FoceiControl(**d)
        assert restored.to_dict() == d


# ── foceiControl factory ────────────────────────────────────────────

class TestFoceiControlFactory:
    def test_returns_focei_control(self):
        c = foceiControl()
        assert isinstance(c, FoceiControl)

    def test_passes_kwargs(self):
        c = foceiControl(maxiter=999, sigma=0.5)
        assert c.maxiter == 999
        assert c.sigma == 0.5

    def test_validation_via_factory(self):
        with pytest.raises(ValueError, match="maxiter"):
            foceiControl(maxiter=-10)

    def test_unknown_kwarg_raises(self):
        with pytest.raises(TypeError):
            foceiControl(bogus=42)


# ── SaemControl defaults ───────────────────────────────────────────

class TestSaemControlDefaults:
    def test_n_burn_default(self):
        c = SaemControl()
        assert c.n_burn == 300

    def test_n_em_default(self):
        c = SaemControl()
        assert c.n_em == 200

    def test_n_mcmc_default(self):
        c = SaemControl()
        assert c.n_mcmc == 3

    def test_step_size_default(self):
        c = SaemControl()
        assert c.step_size == 1.0

    def test_sigma_default(self):
        c = SaemControl()
        assert c.sigma == 1.0

    def test_print_progress_default(self):
        c = SaemControl()
        assert c.print_progress is False


# ── SaemControl custom values ──────────────────────────────────────

class TestSaemControlCustom:
    def test_custom_n_burn(self):
        c = SaemControl(n_burn=500)
        assert c.n_burn == 500

    def test_custom_n_em(self):
        c = SaemControl(n_em=400)
        assert c.n_em == 400

    def test_custom_n_mcmc(self):
        c = SaemControl(n_mcmc=5)
        assert c.n_mcmc == 5

    def test_custom_step_size(self):
        c = SaemControl(step_size=0.5)
        assert c.step_size == 0.5

    def test_custom_sigma(self):
        c = SaemControl(sigma=2.0)
        assert c.sigma == 2.0


# ── SaemControl validation ─────────────────────────────────────────

class TestSaemControlValidation:
    def test_negative_n_burn_raises(self):
        with pytest.raises(ValueError, match="n_burn"):
            SaemControl(n_burn=-1)

    def test_negative_n_em_raises(self):
        with pytest.raises(ValueError, match="n_em"):
            SaemControl(n_em=-1)

    def test_zero_n_mcmc_raises(self):
        with pytest.raises(ValueError, match="n_mcmc"):
            SaemControl(n_mcmc=0)

    def test_negative_step_size_raises(self):
        with pytest.raises(ValueError, match="step_size"):
            SaemControl(step_size=-0.1)

    def test_zero_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            SaemControl(sigma=0.0)

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            SaemControl(sigma=-1.0)


# ── SaemControl to_dict / roundtrip ────────────────────────────────

class TestSaemControlToDict:
    def test_to_dict_keys(self):
        c = SaemControl()
        d = c.to_dict()
        expected = {"n_burn", "n_em", "n_mcmc", "step_size", "sigma", "print_progress"}
        assert set(d.keys()) == expected

    def test_roundtrip(self):
        original = SaemControl(n_burn=100, n_em=50, sigma=0.8)
        d = original.to_dict()
        restored = SaemControl(**d)
        assert restored.to_dict() == d


# ── saemControl factory ─────────────────────────────────────────────

class TestSaemControlFactory:
    def test_returns_saem_control(self):
        c = saemControl()
        assert isinstance(c, SaemControl)

    def test_passes_kwargs(self):
        c = saemControl(n_burn=100, sigma=0.5)
        assert c.n_burn == 100
        assert c.sigma == 0.5

    def test_validation_via_factory(self):
        with pytest.raises(ValueError, match="sigma"):
            saemControl(sigma=-1.0)

    def test_unknown_kwarg_raises(self):
        with pytest.raises(TypeError):
            saemControl(bogus=42)


# ── MockControl ─────────────────────────────────────────────────────

class TestMockControl:
    def test_creation(self):
        c = MockControl()
        assert c is not None

    def test_to_dict(self):
        c = MockControl()
        d = c.to_dict()
        assert isinstance(d, dict)
