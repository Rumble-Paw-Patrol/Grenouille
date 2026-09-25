"""Notebooks d'exploration (`notebooks/`) : committés sans sorties — les lecteurs audio
embarquent le son des enregistrements, données de l'ONF et de Biophonia, qui ne vont jamais sur
GitHub — et leur code se compile."""

import ast
import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))


def _code_cells(path: Path) -> list[dict]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_the_exploration_notebooks_exist():
    assert [p.name[:2] for p in NOTEBOOKS] == ["01", "02", "03"]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_are_committed_without_outputs(path):
    for cell in _code_cells(path):
        assert cell["outputs"] == [] and cell["execution_count"] is None, (
            f"{path.name} : vider les sorties (Clear All Outputs) avant de committer"
        )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_code_compiles(path):
    for cell in _code_cells(path):
        ast.parse("".join(cell["source"]))
