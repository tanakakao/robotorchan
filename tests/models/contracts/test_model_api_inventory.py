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
