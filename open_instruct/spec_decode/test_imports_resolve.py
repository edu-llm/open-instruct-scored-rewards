"""Check that this package's imports resolve, without importing vLLM.

WHY THIS FILE EXISTS. ``conftest.py`` skips ``test_olmoe_eagle3.py`` and
``test_metrics_vllm.py`` when vLLM is absent, which is every laptop that is not Linux. So a
name moved between modules in this package is invisible locally and surfaces as a pytest
collection error on a GPU node -- which is exactly what happened on 2026-08-08: registration
helpers moved from ``olmoe_eagle3`` to ``registration``, the test kept importing them from the
old module, and ``PYTEST_EXIT=2`` came back from two paid runs.

The fix is not "remember to update the imports". It is to make the failure reachable from a
machine with no GPU: every ``from open_instruct.spec_decode.X import a, b`` in this package is
resolved against the *parsed source* of module X. No module is imported, nothing touches CUDA,
and this runs anywhere Python does.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

PACKAGE_DIR = pathlib.Path(__file__).parent
PACKAGE_PREFIX = "open_instruct.spec_decode"


def module_source_path(module: str) -> pathlib.Path | None:
    """Map ``open_instruct.spec_decode.foo`` to its file, or None if outside this package."""
    if not module.startswith(f"{PACKAGE_PREFIX}."):
        return None
    tail = module[len(PACKAGE_PREFIX) + 1 :]
    candidate = PACKAGE_DIR / f"{tail.replace('.', '/')}.py"
    return candidate if candidate.exists() else None


def top_level_names(tree: ast.Module) -> set[str]:
    """Names a module defines at module level: assignments, defs, classes, and its imports."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            names.update((alias.asname or alias.name.split(".")[0]) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update((alias.asname or alias.name) for alias in node.names)
    return names


def intra_package_imports() -> list[tuple[str, str, str]]:
    """Every (importing file, target module, imported name) inside this package."""
    found: list[tuple[str, str, str]] = []
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            if module_source_path(node.module) is None:
                continue
            for alias in node.names:
                found.append((path.name, node.module, alias.name))
    return found


IMPORTS = intra_package_imports()


def test_there_are_intra_package_imports_to_check():
    # Guards the guard: if the collection above silently found nothing, every test below would
    # pass vacuously and this file would be worse than useless.
    assert IMPORTS, "no intra-package imports discovered; has the layout changed?"


@pytest.mark.parametrize(("source_file", "module", "name"), IMPORTS, ids=lambda v: str(v))
def test_imported_name_exists_in_target_module(source_file: str, module: str, name: str):
    target = module_source_path(module)
    assert target is not None, f"{source_file} imports from {module}, which has no source file"
    defined = top_level_names(ast.parse(target.read_text(encoding="utf-8"), filename=str(target)))
    assert name in defined, (
        f"{source_file} imports {name!r} from {module}, which does not define it. "
        f"Was it moved? {target.name} defines: {sorted(defined)}"
    )
