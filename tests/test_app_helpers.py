"""Tests for app.py's model-resolution helpers (no Streamlit required).

app.py imports streamlit at module scope, so the helpers are loaded from
source here rather than by importing the module.
"""

import ast
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent / "app.py"


@pytest.fixture(scope="module")
def helpers():
    """Execute only the helper functions and constants defined in app.py."""
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    wanted = {"resolve_ngram_model_path", "describe_artefact", "NGRAM_MODEL_ENV"}
    kept = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in wanted)
        or (
            isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) in wanted for t in node.targets)
        )
        or (isinstance(node, ast.Import) and all(a.name == "os" for a in node.names))
        or (
            isinstance(node, ast.ImportFrom)
            and node.module in ("pathlib", "__future__")
        )
    ]
    namespace = {}
    exec(compile(ast.Module(body=kept, type_ignores=[]), str(APP), "exec"), namespace)
    return namespace


def test_env_variable_supplies_a_default_model(helpers):
    resolve = helpers["resolve_ngram_model_path"]
    env = {helpers["NGRAM_MODEL_ENV"]: "/models/real.json"}
    assert str(resolve("", env)) == "/models/real.json"


def test_sidebar_input_overrides_the_environment(helpers):
    resolve = helpers["resolve_ngram_model_path"]
    env = {helpers["NGRAM_MODEL_ENV"]: "/models/env.json"}
    assert str(resolve("/models/typed.json", env)) == "/models/typed.json"


def test_no_model_configured_means_bootstrap(helpers):
    resolve = helpers["resolve_ngram_model_path"]
    assert resolve("", {}) is None
    assert resolve("   ", {}) is None


def test_describe_artefact_summarises_corpus_licence_and_metrics(helpers):
    lines = helpers["describe_artefact"](
        {
            "model_kind": "factorised",
            "orders": {"pitch": 3},
            "corpus": {
                "source": "manifest",
                "datasets": [{"name": "my-set", "license": "CC0 1.0"}],
            },
            "metrics": {"test": {"notes": 96, "bits_per_note": 3.9, "pitch_bits_per_note": 3.1}},
        }
    )
    joined = "\n".join(lines)
    assert "my-set" in joined
    assert "CC0 1.0" in joined
    assert "factorised" in joined
    assert "3.90 bits/note" in joined and "96 notes" in joined


def test_describe_artefact_labels_synthetic_corpora(helpers):
    lines = helpers["describe_artefact"]({"corpus": {"source": "synthetic-bootstrap"}})
    assert any("synthetic bootstrap" in line for line in lines)


def test_describe_artefact_tolerates_missing_metrics(helpers):
    # An artefact with no metrics still describes what it can, without raising.
    lines = helpers["describe_artefact"]({})
    assert not any("bits/note" in line for line in lines)
