"""Integrity checks for the tutorial notebooks.

These tests do **not** execute notebooks (they depend on downloaded data and
network services). They guarantee that every shipped ``.ipynb`` is valid JSON
in the notebook format and that every code cell is syntactically valid Python
after stripping IPython magics/shell escapes.
"""

from __future__ import annotations

import ast
import glob
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted(
    glob.glob(str(ROOT / "notebooks" / "**" / "*.ipynb"), recursive=True)
    + glob.glob(str(ROOT / "*.ipynb"))
)


def _strip_magics(source: str) -> str:
    lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("%", "!")):
            lines.append("pass")
        else:
            lines.append(line)
    return "\n".join(lines)


def test_notebooks_are_discovered() -> None:
    assert NOTEBOOKS, "no tutorial notebooks found"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: Path(p).name)
def test_notebook_is_valid(path: str) -> None:
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    assert nb.cells, f"{path} has no cells"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: Path(p).name)
def test_code_cells_parse(path: str) -> None:
    nb = nbformat.read(path, as_version=4)
    failures = []
    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        source = _strip_magics(cell.source)
        if not source.strip():
            continue
        try:
            ast.parse(source)
        except SyntaxError as exc:  # pragma: no cover - reported via assertion
            failures.append(f"cell {index}: {exc.msg} (line {exc.lineno})")
    assert not failures, f"{path} has syntax errors:\n" + "\n".join(failures)
