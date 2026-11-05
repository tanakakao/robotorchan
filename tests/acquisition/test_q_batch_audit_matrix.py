"""Contract tests for scoped q-batch evidence in the existing audit matrix."""

from scripts.audit_acquisition_optimizer_matrix import build_inventory, build_summary


def test_q_batch_optimizer_contracts_cover_dispatch_registry() -> None:
    report = build_inventory()
    contracts = report["q_batch_optimizer_contracts"]
    assert set(contracts) == set(report["optimizers"])
    for entry in contracts.values():
        assert entry["q_values"] == [1, 2, 4]
        assert entry["joint_synthetic_acquisition"]["evidence"]
        assert not entry["registered_acquisition_pairs_verified_q_gt_1"]


def test_mixed_botorch_is_not_mislabeled_joint() -> None:
    contracts = build_inventory()["q_batch_optimizer_contracts"]
    assert contracts["botorch"]["mixed_q"]["status"] == "sequential_backend"
    assert contracts["cmaes"]["joint_synthetic_acquisition"]["status"] == "solver_mock_only"
    assert contracts["cmaes"]["mixed_q"]["status"] == "solver_optional_not_guaranteed"


def test_interpoint_evidence_is_scoped_to_tested_backends() -> None:
    contracts = build_inventory()["q_batch_optimizer_contracts"]
    for name in ("de", "ga"):
        assert contracts[name]["mixed_interpoint_inequality"]["status"] == "runtime_test_present"
        assert contracts[name]["mixed_interpoint_inequality"]["evidence"]
    for name in set(contracts) - {"de", "ga"}:
        assert contracts[name]["mixed_interpoint_inequality"]["status"] == "unverified"


def test_summary_preserves_q_batch_contracts() -> None:
    report = build_inventory()
    summary = build_summary(report)
    assert summary["q_batch_optimizer_contracts"] == report["q_batch_optimizer_contracts"]
    assert summary["schema_version"] == report["schema_version"]
