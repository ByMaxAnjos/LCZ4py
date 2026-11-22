"""QA harness: import-safety for ``LCZ4py.general`` and ``LCZ4py.local``.

Two layers of verification:

1. **In-process** — every module under ``LCZ4py.general``, ``LCZ4py.local`` and
   ``LCZ4py._internal`` must import, and every name advertised in
   ``general.__all__`` / ``local.__all__`` must resolve.  No network is used.

2. **Subprocess with optional dependencies blocked** — a fresh interpreter gets
   a ``sys.meta_path`` finder that makes a set of optional third-party
   distributions look uninstalled, and then imports ``LCZ4py.general`` and
   ``LCZ4py.local``.  This is the actual contract we care about: a user who has
   only the core requirements must still be able to import the package.

   The blocker works at the *loader* level (``find_spec`` returns a spec whose
   loader raises ``ModuleNotFoundError`` on ``create_module``).  A finder that
   raised directly from ``find_spec`` would break availability probes that do
   not catch the exception — e.g. xarray's ``module_available("dask")`` calls
   ``importlib.util.find_spec`` — and report false failures.  With this design
   a real ``import dask`` raises ``ModuleNotFoundError`` exactly as it would if
   dask were absent, while ``find_spec`` probing stays harmless.

   Each blocked run is a fresh interpreter because module-level ``try: import``
   guards are evaluated once per process and cannot be re-tested after the
   modules are already in ``sys.modules``.

Run with::

    python -m pytest tests/test_imports_general_local.py -q
"""

from __future__ import annotations

import importlib
import os
import pkgutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

SCANNED_PACKAGES = ("LCZ4py.general", "LCZ4py.local", "LCZ4py._internal")
PUBLIC_PACKAGES = ("LCZ4py.general", "LCZ4py.local")

# Optional dependencies that must not be required merely to import the package.
# Keep this list in sync with requirements.txt (and the blocked list in the QA
# brief).  ``xarray`` is included deliberately: it is listed as a soft/optional
# requirement for the UCP path, so ``LCZ4py.general`` should degrade gracefully.
OPTIONAL_BLOCKED = (
    "pystac_client",
    "planetary_computer",
    "h5netcdf",
    "h5py",
    "pykrige",
    "sklearn",
    "duckdb",
    "geoarrow",
    "exactextract",
    "datashader",
    "holoviews",
    "colorcet",
    "contextily",
    "xarray",
    "dask",
    "cdsapi",
)

# Subsets used for faster, more precise diagnostics when the combined run fails.
OPTIONAL_GROUPS = {
    "network_stac": ("pystac_client", "planetary_computer", "cdsapi", "contextily"),
    "accel_geo": ("pykrige", "sklearn", "duckdb", "geoarrow", "exactextract"),
    "io_and_plot_extras": (
        "h5netcdf",
        "h5py",
        "datashader",
        "holoviews",
        "colorcet",
    ),
    "array_stack": ("xarray", "dask"),
}

SUBPROCESS_TIMEOUT_S = 240


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def _enumerate_modules(package: str) -> list[str]:
    """Import *package* and return every submodule name below it."""
    pkg = importlib.import_module(package)
    names = [package]
    names.extend(
        info.name for info in pkgutil.walk_packages(pkg.__path__, package + ".")
    )
    return names


def _enumerate_all() -> list[str]:
    return [m for pkg in SCANNED_PACKAGES for m in _enumerate_modules(pkg)]


ALL_MODULES = _enumerate_all()


# ---------------------------------------------------------------------------
# 1. In-process import coverage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module_name", ALL_MODULES)
def test_module_imports(module_name: str) -> None:
    """Every module (including private ``_internal`` helpers) imports cleanly."""
    module = importlib.import_module(module_name)
    assert module is not None
    assert module.__name__ == module_name


@pytest.mark.parametrize("package_name", PUBLIC_PACKAGES)
def test_public_api_names_resolve(package_name: str) -> None:
    """Every name in the package's ``__all__`` resolves via getattr."""
    package = importlib.import_module(package_name)
    exported = getattr(package, "__all__", None)
    assert exported, f"{package_name}.__all__ is missing or empty"

    missing = [name for name in exported if not hasattr(package, name)]
    assert not missing, (
        f"{package_name}.__all__ advertises {len(missing)} unresolved name(s): "
        f"{missing}"
    )


@pytest.mark.parametrize("package_name", PUBLIC_PACKAGES)
def test_public_api_all_has_no_duplicates(package_name: str) -> None:
    package = importlib.import_module(package_name)
    exported = list(package.__all__)
    duplicates = sorted({name for name in exported if exported.count(name) > 1})
    assert not duplicates, f"{package_name}.__all__ has duplicates: {duplicates}"


def test_top_level_package_reexports_both_groups() -> None:
    """``import LCZ4py`` must expose the union of both groups' public names."""
    lcz = importlib.import_module("LCZ4py")
    general = importlib.import_module("LCZ4py.general")
    local = importlib.import_module("LCZ4py.local")

    expected = set(general.__all__) | set(local.__all__)
    missing = sorted(name for name in expected if not hasattr(lcz, name))
    assert not missing, f"LCZ4py is missing {len(missing)} re-export(s): {missing}"


# ---------------------------------------------------------------------------
# 2. Optional-dependency isolation (fresh subprocess per scenario)
# ---------------------------------------------------------------------------

_BLOCKER_SNIPPET = textwrap.dedent(
    """
    import importlib.machinery
    import sys

    BLOCKED = {blocked!r}


    class _MissingLoader:
        \"\"\"Loader that fails exactly like an uninstalled distribution.\"\"\"

        def create_module(self, spec):
            raise ModuleNotFoundError(
                "No module named {{!r}}".format(spec.name), name=spec.name
            )

        def exec_module(self, module):
            name = module.__spec__.name
            raise ModuleNotFoundError(
                "No module named {{!r}}".format(name), name=name
            )


    class _BlockOptionalDeps:
        \"\"\"Make the blocked top-level names look uninstalled.\"\"\"

        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in BLOCKED:
                return importlib.machinery.ModuleSpec(fullname, _MissingLoader())
            return None


    sys.meta_path.insert(0, _BlockOptionalDeps())

    # Sanity check: the block really does raise ModuleNotFoundError, and only
    # for the configured top-level names.
    for _name in sorted(BLOCKED):
        try:
            __import__(_name)
        except ModuleNotFoundError as _exc:
            assert _exc.name.split(".")[0] == _name, (_name, _exc.name)
        else:  # pragma: no cover - only hit if the blocker is broken
            raise AssertionError("expected {{!r}} to be blocked".format(_name))
    """
)


def _run_blocked_import(blocked: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    """Import LCZ4py.general + LCZ4py.local in a subprocess with *blocked* hidden."""
    script = _BLOCKER_SNIPPET.format(blocked=tuple(sorted(blocked)))
    script += textwrap.dedent(
        """
        import LCZ4py.general as general
        import LCZ4py.local as local

        assert general.__file__.startswith({repo!r}), general.__file__
        print("IMPORT_OK", len(general.__all__), len(local.__all__))
        """
    ).format(repo=str(REPO_ROOT))

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env.pop("PYTHONWARNINGS", None)

    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_S,
    )


def _assert_blocked_import_ok(blocked: tuple[str, ...], label: str) -> None:
    proc = _run_blocked_import(blocked)
    if proc.returncode != 0:
        pytest.fail(
            f"Import failed with optional dependencies blocked ({label}: "
            f"{', '.join(blocked)}).\n"
            f"--- last lines of stderr ---\n"
            + "\n".join(proc.stderr.strip().splitlines()[-18:])
        )
    assert "IMPORT_OK" in proc.stdout, proc.stdout


@pytest.mark.parametrize(
    "group,blocked",
    list(OPTIONAL_GROUPS.items()),
    ids=list(OPTIONAL_GROUPS),
)
def test_import_without_optional_group(
    group: str, blocked: tuple[str, ...]
) -> None:
    """Each optional dependency family can be absent without breaking import."""
    _assert_blocked_import_ok(blocked, group)


def test_import_without_all_optional_deps() -> None:
    """The combined worst case: every optional dependency is unavailable."""
    _assert_blocked_import_ok(OPTIONAL_BLOCKED, "all optional deps")
