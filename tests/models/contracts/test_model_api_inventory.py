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
    assert _method_info(SingleTaskGP, "fit")["status"] == "absent"


def test_training_api_classification_separates_exact_and_non_gp() -> None:
    inventory = build_inventory()
    exact = inventory["regression_models"]["SingleTaskGP"]
    forest = inventory["regression_models"]["RandomForestSurrogate"]
    assert exact["training_api"]["category"] == "mll_training_fit_candidate"
    assert forest["training_api"]["category"] == "fit_declared"
    assert all(\n        row["training_api"]["verification"] == "static_only"\n        for row in inventory["regression_models"].values()\n    )


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
    assert gaps == [{
        "model": "Example",
        "issue": "supports_fit_true_without_concrete_method",
        "observed": "placeholder",
    }]


def test_all_registered_classifiers_declare_prediction_methods() -> None:
    inventory = build_inventory()
    for row in inventory["classification_models"].values():
        assert row["methods"]["predict_proba"]["status"] == "declared"
        assert row["methods"]["predict_class"]["status"] == "declared"
