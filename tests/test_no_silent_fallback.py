"""
Guards against the reintroduction of silent-fallback patterns.

This project's escalation gate once read:

    similarity_threshold = getattr(sr, "SIMILARITY_THRESHOLD", 0.35)

If that import ever degraded, the human-escalation gate would silently revert
to a dead MiniLM-era threshold -- a silent failure in the one gate the entire
thesis rests on. The same class of problem produced three separate
stale-artifact bugs.

These are static checks over the source tree: fast, no model loading, and
they fail the build the moment the pattern reappears.
"""

from __future__ import annotations

import ast
import os
import re

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

# Thresholds whose silent default would change a live gating decision.
GUARDED_NAMES = {
    "SIMILARITY_THRESHOLD",
    "CASCADE_CONFIDENCE_THRESHOLD",
    "FILING_CONFIDENCE_THRESHOLD",
}


def _getattr_defaults(tree):
    """Yield (lineno, attr_name) for getattr(obj, "THRESHOLD", <number>).

    Uses the AST rather than a text scan so that prose in docstrings and
    comments describing this anti-pattern is not itself flagged.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name)
                and node.func.id == "getattr"):
            continue
        if len(node.args) < 3:
            continue

        name_arg, default_arg = node.args[1], node.args[2]
        if not (isinstance(name_arg, ast.Constant)
                and isinstance(name_arg.value, str)):
            continue
        if name_arg.value not in GUARDED_NAMES:
            continue
        if isinstance(default_arg, ast.Constant) and isinstance(
            default_arg.value, (int, float)
        ):
            yield node.lineno, name_arg.value


def _python_files():
    for root, _dirs, files in os.walk(SRC_DIR):
        if "__pycache__" in root:
            continue
        for name in files:
            if name.endswith(".py"):
                yield os.path.join(root, name)


def _read(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def test_no_getattr_threshold_defaults():
    """A calibrated threshold must never have a hardcoded fallback."""
    offenders = []
    for path in _python_files():
        try:
            tree = ast.parse(_read(path), filename=path)
        except SyntaxError:  # pragma: no cover - not our concern here
            continue
        rel = os.path.relpath(path, PROJECT_ROOT)
        for lineno, name in _getattr_defaults(tree):
            offenders.append(f"{rel}:{lineno}: getattr(..., '{name}', <num>)")

    assert not offenders, (
        "Silent threshold fallback reintroduced -- import and fail loudly "
        "instead:\n" + "\n".join(offenders)
    )


def test_agent_package_declares_no_local_thresholds():
    """Every calibrated constant lives in config.py, nowhere else."""
    agent_dir = os.path.join(SRC_DIR, "agent")
    pattern = re.compile(
        r"^\s*(SIMILARITY_THRESHOLD|CASCADE_CONFIDENCE_THRESHOLD"
        r"|FILING_CONFIDENCE_THRESHOLD)\s*=",
        re.MULTILINE,
    )
    offenders = []
    for name in sorted(os.listdir(agent_dir)):
        if not name.endswith(".py") or name == "config.py":
            continue
        path = os.path.join(agent_dir, name)
        if pattern.search(_read(path)):
            offenders.append(f"src/agent/{name}")

    assert not offenders, (
        "Threshold redeclared outside config.py: " + ", ".join(offenders)
    )


@pytest.mark.parametrize("module_name", [
    "config", "errors", "schemas", "logging_setup",
    "artifacts", "classifier", "retriever", "resolver", "pipeline",
    # conformal.py was added in Phase 1 and never listed here, so the
    # import-side-effect guard had a hole in it.
    "conformal",
    # Phase 3B.
    "agents", "orchestrator",
    # Phase 4A.
    "drift",
])
def test_agent_modules_import_without_side_effects(module_name):
    """Importing the library must not load models, read .env, or print.

    streamlit_app.py cannot be imported headlessly because it calls
    st.set_page_config() at import time -- which is exactly why the
    adversarial test grew its own copy of the cascade. The agent package must
    not repeat that.
    """
    import importlib

    module = importlib.import_module(f"src.agent.{module_name}")
    assert module is not None
