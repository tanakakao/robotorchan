"""Documentation coverage contract for public model exports."""

from __future__ import annotations

import json
from pathlib import Path

import robotorchan.models as models

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "model_coverage.json"


def _load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_public_models_have_documentation_coverage() -> None:
    """Every public model must be represented by the coverage manifest."""
    manifest = _load_manifest()
    excluded = set(manifest["excluded_public_exports"])
    public_models = set(models.__all__) - excluded
    covered_models = set(manifest["models"])

    assert covered_models == public_models


def test_coverage_references_existing_files() -> None:
    """Coverage entries must point to durable guide, theory, and example files."""
    manifest = _load_manifest()

    assert manifest["schema_version"] == 3
    assert manifest["source_of_truth"] == "robotorchan.models.registry.MODEL_REGISTRY"
    assert set(manifest["coverage_contract"]) == {"guide", "theory", "notebook"}

    for model_name, coverage in manifest["models"].items():
        assert set(coverage) == {"guide", "theory", "notebook"}, model_name
        for kind, relative_path in coverage.items():
            path = ROOT / relative_path
            assert path.is_file(), f"{model_name}: missing {kind}: {relative_path}"


def test_exclusions_are_real_public_non_model_exports() -> None:
    """Explicit exclusions must stay visible instead of silently drifting."""
    manifest = _load_manifest()
    excluded = set(manifest["excluded_public_exports"])

    assert excluded <= set(models.__all__)
    assert excluded == {"UnsupportedModelOperationError"}


def test_coverage_uses_family_guides_instead_of_legacy_model_index() -> None:
    """Coverage should resolve to the reorganized model-family guides."""
    manifest = _load_manifest()

    for model_name, coverage in manifest["models"].items():
        guide = coverage["guide"]
        assert guide.startswith("docs/models/"), model_name
        assert guide != "docs/models.md", model_name


def test_notebook_coverage_uses_registered_example_notebooks() -> None:
    """Representative examples must stay inside the documented notebook collection."""
    manifest = _load_manifest()
    notebook_root = "examples/notebooks/"

    for model_name, coverage in manifest["models"].items():
        notebook = coverage["notebook"]
        assert notebook.startswith(notebook_root), model_name
        assert notebook.endswith(".ipynb"), model_name
