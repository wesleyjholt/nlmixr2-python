"""Installation and environment helpers that mirror the R package surface."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import subprocess
from typing import Sequence


NLMIXR2_PACKAGES = (
    "cli",
    "crayon",
    "dplyr",
    "purrr",
    "rstudioapi",
    "nlmixr2est",
    "nlmixr2extra",
    "rxode2",
    "lotri",
    "nlmixr2plot",
    "tibble",
    "magrittr",
    "nlmixr2",
)

NLMIXR2_DIRECT_DEPS = (
    "dparser",
    "lotri",
    "rxode2ll",
    "rxode2parse",
    "rxode2random",
    "rxode2et",
    "rxode2",
    "nlmixr2data",
    "nlmixr2est",
    "nlmixr2extra",
    "nlmixr2plot",
    "nlmixr2",
)

PYTHON_RUNTIME_DEPS = ("jax", "jaxlib", "numpy")


def nlmixr2deps(recursive: bool = False) -> tuple[str, ...]:
    """Return nlmixr2 dependency package names."""

    if not recursive:
        return NLMIXR2_DIRECT_DEPS
    combined: list[str] = list(NLMIXR2_DIRECT_DEPS)
    for package in NLMIXR2_PACKAGES:
        if package not in combined:
            combined.append(package)
    return tuple(combined)


def nlmixr2packages(include_self: bool = True) -> tuple[str, ...]:
    """Return the documented nlmixr2 ecosystem package names."""

    packages = NLMIXR2_PACKAGES
    if not include_self:
        packages = tuple(package for package in packages if package != "nlmixr2")
    return packages


def _python_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _reference_runtime_version(
    env_prefix: str | Path = "/scratch/gautschi/holtw/codex-envs/nlmixr2-ref",
) -> str | None:
    rscript = Path(env_prefix) / "bin" / "Rscript"
    if not rscript.is_file():
        return None
    completed = subprocess.run(
        [str(rscript), "-e", 'cat(as.character(utils::packageVersion("nlmixr2")))' ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    token = completed.stdout.strip()
    return token or None


def nlmixr2CheckInstall(packages: Sequence[str] | None = None) -> dict[str, object]:
    """Check the local Python runtime and optional Gautschi reference runtime."""

    requested = tuple(packages or PYTHON_RUNTIME_DEPS)
    resolved = {package: _python_version(package) for package in requested}
    missing = tuple(name for name, package_version in resolved.items() if package_version is None)
    reference_version = _reference_runtime_version()
    return {
        "ok": not missing,
        "python_packages": resolved,
        "missing_python": missing,
        "reference_runtime": {
            "available": reference_version is not None,
            "version": reference_version,
        },
        "ecosystem_packages": nlmixr2packages(),
    }


def nlmixr2conflicts(names: Sequence[str] | None = None) -> dict[str, object]:
    """Report name conflicts against the nlmixr2 reserved prefixes."""

    reserved_prefixes = ("_", "rx_", "nlmixr_")
    checked = tuple(names or ())
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in checked:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    conflicts = tuple(
        name
        for name in checked
        if isinstance(name, str) and name.startswith(reserved_prefixes)
    )
    return {
        "reserved_prefixes": reserved_prefixes,
        "conflicts": conflicts,
        "duplicates": tuple(duplicates),
    }


def nlmixr2update(
    package: str = "nlmixr2-python",
    dry_run: bool = True,
    extras: Sequence[str] = (),
    python: str = "python3",
) -> dict[str, object]:
    """Build or execute a package update command."""

    target = package
    if extras:
        target = f"{package}[{','.join(extras)}]"
    command = [python, "-m", "pip", "install", "--upgrade", target]
    if dry_run:
        return {"command": command, "executed": False}
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    return {
        "command": command,
        "executed": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
