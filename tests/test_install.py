from __future__ import annotations

import nlmixr2.install as install
from nlmixr2 import (
    nlmixr2CheckInstall,
    nlmixr2conflicts,
    nlmixr2deps,
    nlmixr2packages,
    nlmixr2update,
)


def test_nlmixr2packages_returns_documented_ecosystem_names():
    packages = nlmixr2packages()

    assert packages[:5] == ("cli", "crayon", "dplyr", "purrr", "rstudioapi")
    assert packages[-1] == "nlmixr2"


def test_nlmixr2packages_can_exclude_nlmixr2_itself():
    packages = nlmixr2packages(include_self=False)

    assert "nlmixr2" not in packages
    assert "nlmixr2plot" in packages


def test_nlmixr2deps_returns_direct_dependencies():
    deps = nlmixr2deps()

    assert deps[:3] == ("dparser", "lotri", "rxode2ll")
    assert deps[-1] == "nlmixr2"


def test_nlmixr2deps_recursive_superset_includes_ecosystem_packages():
    deps = nlmixr2deps(recursive=True)

    assert "dparser" in deps
    assert "cli" in deps
    assert deps.index("dparser") < deps.index("cli")


def test_nlmixr2CheckInstall_reports_success(monkeypatch):
    monkeypatch.setattr(install, "_python_version", lambda name: f"{name}-ok")
    monkeypatch.setattr(install, "_reference_runtime_version", lambda: "5.0.0")

    result = nlmixr2CheckInstall()

    assert result["ok"] is True
    assert result["missing_python"] == ()
    assert result["reference_runtime"]["version"] == "5.0.0"
    assert result["ecosystem_packages"][-1] == "nlmixr2"


def test_nlmixr2CheckInstall_reports_missing_packages(monkeypatch):
    monkeypatch.setattr(install, "_python_version", lambda name: None if name == "jax" else "ok")
    monkeypatch.setattr(install, "_reference_runtime_version", lambda: None)

    result = nlmixr2CheckInstall()

    assert result["ok"] is False
    assert result["missing_python"] == ("jax",)
    assert result["reference_runtime"]["available"] is False


def test_nlmixr2conflicts_detects_reserved_prefixes_and_duplicates():
    result = nlmixr2conflicts(["theta", "rx_eta", "theta", "nlmixr_bad"])

    assert result["conflicts"] == ("rx_eta", "nlmixr_bad")
    assert result["duplicates"] == ("theta",)


def test_nlmixr2conflicts_is_empty_for_clean_names():
    result = nlmixr2conflicts(["theta", "eta.cl", "prop.err"])

    assert result["conflicts"] == ()
    assert result["duplicates"] == ()


def test_nlmixr2update_returns_a_dry_run_command():
    result = nlmixr2update(extras=("dev",))

    assert result["executed"] is False
    assert result["command"][-1] == "nlmixr2-python[dev]"


def test_nlmixr2update_executes_subprocess_when_requested(monkeypatch):
    calls: list[list[str]] = []

    class CompletedProcess:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_run(command, check, capture_output, text):
        calls.append(command)
        assert check is False
        assert capture_output is True
        assert text is True
        return CompletedProcess()

    monkeypatch.setattr(install.subprocess, "run", fake_run)

    result = nlmixr2update(dry_run=False, python="python-test")

    assert result["executed"] is True
    assert result["returncode"] == 0
    assert calls == [["python-test", "-m", "pip", "install", "--upgrade", "nlmixr2-python"]]
