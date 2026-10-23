"""Static API inventory contracts; model construction is intentionally avoided."""

import json
import runpy
from pathlib import Path

from robotorchan.models import SingleTaskGP
from robotorchan.models.base import FittableModelMixin
from robotorchan.models.classification.registry import CLASSIFICATION_MODEL_REGISTRY
from robotorchan.models.non_gp.random_forest import RandomForestSurrogate
from robotorchan.models.registry import MODEL_REGISTRY

_AUDIT = runpy.run_path(
    str(Path(__file__).resolve().parents[3] / "scripts" / "audit_model_api.py"),
)
_method_info = _AUDIT["_method_info"]
build_inventory = _AUDIT["build_inventory"]
classify_training_api = _AUDIT["classify_training_api"]
find_contract_gaps = _AUDIT["find_contract_gaps"]
find_missing_documentation_paths = _AUDIT["find_missing_documentation_paths"]
find_test_references = _AUDIT["find_test_references"]
find_upstream_constructor_calls = _AUDIT["find_upstream_constructor_calls"]


def test_inventory_covers_registered_models_without_instantiation() -> None:
    inventory = build_inventory()

    assert set(inventory["regression_models"]) == set(MODEL_REGISTRY)
    assert set(inventory["classification_models"]) == set(CLASSIFICATION_MODEL_REGISTRY)
    assert "static_signature_inventory" in inventory["scope"]
    json.dumps(inventory)


def test_inventory_reports_training_interfaces_without_claiming_runtime_support() -> None:
    inventory = build_inventory()
    exact = inventory["regression_models"]["SingleTaskGP"]
    forest = inventory["regression_models"]["RandomForestSurrogate"]

    assert exact["methods"]["make_mll"]["status"] == "declared"
    assert forest["methods"]["fit"]["status"] == "declared"
    assert forest["supports_fit_flag"] is True
    assert exact["methods"]["posterior"]["status"] == "declared"


def test_abstract_fit_placeholder_is_not_reported_as_implementation() -> None:
    assert _method_info(FittableModelMixin, "fit")["status"] == "placeholder"
    assert _method_info(RandomForestSurrogate, "fit")["status"] == "declared"
    assert _method_info(SingleTaskGP, "fit")["status"] == "declared"


def test_training_api_classification_separates_exact_and_non_gp() -> None:
    inventory = build_inventory()
    exact = inventory["regression_models"]["SingleTaskGP"]
    forest = inventory["regression_models"]["RandomForestSurrogate"]
    assert exact["training_api"]["category"] == "fit_declared"
    assert forest["training_api"]["category"] == "fit_declared"
    assert all(
        row["training_api"]["verification"] == "static_only"
        for row in inventory["regression_models"].values()
    )


def test_static_contract_gap_report_detects_false_capability_claims() -> None:
    row = build_inventory()["regression_models"]["RandomForestSurrogate"]
    example = {
        "regression_models": {
            "Example": {
                **row,
                "supports_fit_flag": True,
                "methods": {**row["methods"], "fit": {"status": "placeholder"}},
            },
        },
        "classification_models": {},
    }
    gaps = find_contract_gaps(example)
    assert gaps == [
        {
            "model": "Example",
            "issue": "supports_fit_true_without_concrete_method",
            "observed": "placeholder",
        }
    ]


def test_all_registered_classifiers_declare_prediction_methods() -> None:
    inventory = build_inventory()
    for row in inventory["classification_models"].values():
        assert row["methods"]["predict_proba"]["status"] == "declared"
        assert row["methods"]["predict_class"]["status"] == "declared"


def test_existing_fit_without_capability_flag_is_not_marked_missing() -> None:
    inventory = build_inventory()
    classifiers = inventory["classification_models"]
    concrete = [
        (name, row)
        for name, row in classifiers.items()
        if row["methods"]["fit"]["status"] == "declared" and row["supports_fit_flag"] is not True
    ]
    assert concrete
    for name, row in concrete:
        assert row["training_api"]["category"] == "fit_declared"
        assert any(
            issue["model"] == name and issue["issue"] == "fit_declared_without_supports_fit_true"
            for issue in inventory["static_contract_gaps"]
        )


def test_inventory_reports_upstream_ownership_without_runtime_claims() -> None:
    inventory = build_inventory()
    exact = inventory["regression_models"]["SingleTaskGP"]
    mixed = inventory["regression_models"]["MixedSingleTaskGP"]
    forest = inventory["regression_models"]["RandomForestSurrogate"]

    assert exact["implementation_module"].startswith("robotorchan.")
    assert any(base.endswith(".SingleTaskGP") for base in exact["upstream_bases"])
    assert any(base.endswith(".MixedSingleTaskGP") for base in mixed["upstream_bases"])
    assert forest["provenance_verification"] == "static_mro_only"
    assert forest["upstream_bases"] == [] or all(
        not base.startswith("sklearn.") for base in forest["upstream_bases"]
    )


def test_inventory_preserves_registry_documentation_provenance() -> None:
    inventory = build_inventory()
    regression = inventory["regression_models"]["SingleTaskGP"]
    classification = inventory["classification_models"]["binary.standard"]

    assert regression["documentation"]["guide"] == "docs/models/standard.md"
    assert regression["documentation"]["theory"] == "docs/theory/02_gaussian_process.md"
    assert regression["documentation"]["notebook"].endswith("01_single_task_gp.ipynb")
    assert regression["documentation"]["verification"] == "registry_metadata_only"
    assert classification["documentation"] == {
        "verification": "not_recorded_in_classification_registry"
    }


def test_all_registered_models_expose_static_provenance_fields() -> None:
    inventory = build_inventory()
    for family in ("regression_models", "classification_models"):
        for row in inventory[family].values():
            assert row["implementation_module"]
            assert row["class"].startswith(f"{row['implementation_module']}.")
            assert row["provenance_verification"] == "static_mro_only"
            assert isinstance(row["upstream_bases"], list)
            assert all(
                base.startswith(("botorch.", "gpytorch.", "sklearn."))
                for base in row["upstream_bases"]
            )
            assert "verification" in row["documentation"]

    for row in inventory["regression_models"].values():
        assert row["documentation"]["verification"] == "registry_metadata_only"
        assert all(row["documentation"][field] for field in ("guide", "theory", "notebook"))

    for row in inventory["classification_models"].values():
        assert row["documentation"]["verification"] == ("not_recorded_in_classification_registry")


def test_registry_documentation_paths_exist_in_repository() -> None:
    root = Path(__file__).resolve().parents[3]
    missing = find_missing_documentation_paths(build_inventory(), repository_root=root)
    assert missing == []


def test_documentation_path_check_reports_missing_file(tmp_path: Path) -> None:
    inventory = build_inventory()
    missing = find_missing_documentation_paths(inventory, repository_root=tmp_path)
    assert len(missing) == 3 * len(MODEL_REGISTRY)
    assert {item["kind"] for item in missing} == {"guide", "theory", "notebook"}


def test_test_reference_inventory_covers_all_registered_models() -> None:
    inventory = build_inventory()
    references = find_test_references(
        inventory, repository_root=Path(__file__).resolve().parents[3]
    )
    assert set(references) == set(MODEL_REGISTRY) | set(CLASSIFICATION_MODEL_REGISTRY)
    assert all(row["verification"] == "syntactic_reference_only" for row in references.values())
    assert all(row["test_files"] == sorted(set(row["test_files"])) for row in references.values())
    assert any(
        path.endswith("test_mixed_single_task_gp.py")
        for path in references["MixedSingleTaskGP"]["test_files"]
    )


def test_test_reference_inventory_ignores_strings_and_comments(tmp_path: Path) -> None:
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    (test_dir / "test_example.py").write_text(
        '# SingleTaskGP\\ntext = "SingleTaskGP"\\n', encoding="utf-8"
    )
    references = find_test_references(build_inventory(), repository_root=tmp_path)
    assert references["SingleTaskGP"]["test_files"] == []


def test_upstream_call_scan_covers_all_registered_models() -> None:
    inventory = build_inventory()
    calls = find_upstream_constructor_calls(
        inventory, repository_root=Path(__file__).resolve().parents[3]
    )
    assert set(calls) == set(MODEL_REGISTRY) | set(CLASSIFICATION_MODEL_REGISTRY)
    assert all(row["verification"] == "static_direct_import_call_only" for row in calls.values())
    assert all(
        row["upstream_calls"] == sorted(set(row["upstream_calls"])) for row in calls.values()
    )
    assert (
        "sklearn.ensemble.RandomForestRegressor" in calls["RandomForestSurrogate"]["upstream_calls"]
    )
    assert "gpytorch.likelihoods.BernoulliLikelihood" in calls["binary.standard"]["upstream_calls"]


def test_upstream_call_scan_ignores_strings_and_unrelated_calls(tmp_path: Path) -> None:
    inventory = {
        "regression_models": {
            "Example": {"class": "example.Example"},
        },
        "classification_models": {},
    }
    source = tmp_path / "src" / "example.py"
    source.parent.mkdir()
    source.write_text(
        "from sklearn.ensemble import RandomForestRegressor\n"
        "class Example:\n"
        "    def run(self):\n"
        '        message = "RandomForestRegressor()"\n'
        "        return len(message)\n",
        encoding="utf-8",
    )
    calls = find_upstream_constructor_calls(inventory, repository_root=tmp_path)
    assert calls["Example"]["upstream_calls"] == []
