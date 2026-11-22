"""Jupyter Notebook の実行確認用スクリプト。

通常の CI では比較的軽量な Notebook のみ実行し、Fully Bayesian SAAS や
構造化出力のように実行時間が長くなりやすい Notebook は除外する。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOK_DIR = Path(__file__).resolve().parent / "notebooks"

DEFAULT_NOTEBOOKS = [
    "01_single_task_gp.ipynb",
    "02_mixed_single_task_gp.ipynb",
    "03_multi_fidelity_gp.ipynb",
    "04_multitask_gp.ipynb",
    "05_model_list_gp.ipynb",
    "06_variational_gp.ipynb",
    "08_map_saas_and_additive_gp.ipynb",
    "10_hierarchical_gp.ipynb",
    "11_heterogeneous_multitask_gp.ipynb",
    "12_contextual_gp.ipynb",
    "13_robust_gp.ipynb",
    "14_robust_observation_models.ipynb",
    "15_noise_models.ipynb",
    "16_uncertain_input_gp.ipynb",
    "17_nonstationary_gp.ipynb",
    "18_reduced_gp.ipynb",
    "19_neural_reduction_gp.ipynb",
    "20_reduced_multitask_gp.ipynb",
    "21_mixed_reduced_gp.ipynb",
    "24_pairwise_gp.ipynb",
    "26_classification_active_learning.ipynb",
    "28_classification_robust_uncertain.ipynb",
    "29_classification_probability_layers.ipynb",
    "30_multiclass_classification_active_learning.ipynb",
    "31_ordinal_classification_active_learning.ipynb",
    "32_standard_acquisition.ipynb",
    "33_multiobjective_acquisition.ipynb",
    "34_active_learning_acquisition.ipynb",
    "35_regression_active_learning.ipynb",
    "36_objective_posterior_transform.ipynb",
    "37_posterior_sampling.ipynb",
    "38_acquisition_initialization.ipynb",
    "39_acquisition_optimization.ipynb",
    "40_batch_async_fantasization.ipynb",
    "41_outcome_constrained_bo.ipynb",
    "42_nonlinear_candidate_constraints.ipynb",
    "43_heterogeneous_constrained_bo.ipynb",
    "44_sequential_bo_workflow.ipynb",
    "45_end_to_end_workflows.ipynb",
    "47_turbo_trust_region.ipynb",
    "48_robust_input_perturbation.ipynb",
    "58_causal_bayesian_optimization.ipynb",
    "61_heterogeneous_classification_constraints.ipynb",
]

SLOW_NOTEBOOKS = [
    "07_saas_gp.ipynb",
    "09_structured_output_gp.ipynb",
]

EXCLUDED_NOTEBOOKS = [
    "22_expressive_surrogate_gp.ipynb",
    "23_non_gp_surrogates.ipynb",
    "25_mixed_preference_bo.ipynb",
    "27_binary_classification_extensions.ipynb",
    "46_study_ask_tell_mixed_q_batch.ipynb",
    "49_multifidelity_cost_aware.ipynb",
    "50_safe_bo_usage_guide.ipynb",
    "51_safe_bo_mixed_q_batch.ipynb",
    "52_transfer_bayesian_optimization.ipynb",
    "53_combinatorial_bayesian_optimization.ipynb",
    "54_high_dimensional_bo_benchmark.ipynb",
    "55_heterogeneous_bo_benchmark.ipynb",
    "56_turbo_multi_benchmark.ipynb",
    "57_safe_bo_benchmark.ipynb",
    "59_causal_bo_benchmark.ipynb",
    "60_contextual_bayesian_optimization.ipynb",
]


def validate_notebook_manifest() -> None:
    """Notebook ファイルが実行区分へ明示的に分類されていることを確認する。"""
    discovered = {path.name for path in NOTEBOOK_DIR.glob("*.ipynb")}
    prefixes = [name.split("_", 1)[0] for name in discovered]
    expected = [f"{number:02d}" for number in range(1, len(discovered) + 1)]
    if sorted(prefixes) != expected:
        raise ValueError("Notebook の番号は 01 から重複・欠番なく連続させてください。")
    groups = {
        "default": set(DEFAULT_NOTEBOOKS),
        "slow": set(SLOW_NOTEBOOKS),
        "excluded": set(EXCLUDED_NOTEBOOKS),
    }

    duplicates = (
        (groups["default"] & groups["slow"])
        | (groups["default"] & groups["excluded"])
        | (groups["slow"] & groups["excluded"])
    )
    if duplicates:
        raise ValueError(f"Notebook の実行区分が重複しています: {sorted(duplicates)}")

    declared = set().union(*groups.values())
    ci_group_by_name = {name: group_name for group_name, names in groups.items() for name in names}
    metadata_group = {"default": "default", "slow": "slow", "manual": "excluded"}
    for name in sorted(discovered):
        notebook = nbformat.read(NOTEBOOK_DIR / name, as_version=4)
        metadata = notebook.metadata.get("robotorchan")
        if metadata is None or "ci" not in metadata:
            continue
        declared_ci = metadata["ci"]
        if declared_ci not in metadata_group:
            raise ValueError(f"Notebook metadata ci が不正です: {name}: {declared_ci!r}")
        expected_group = metadata_group[declared_ci]
        actual_group = ci_group_by_name.get(name)
        if actual_group != expected_group:
            raise ValueError(
                "Notebook metadata ci と manifest の実行区分が一致しません: "
                f"{name}: metadata={declared_ci!r}, manifest={actual_group!r}"
            )

    undeclared = discovered - declared
    missing = declared - discovered
    if undeclared or missing:
        raise ValueError(
            "Notebook manifest が実ファイルと一致しません: "
            f"undeclared={sorted(undeclared)}, missing={sorted(missing)}"
        )


def execute_notebook(path: Path, timeout: int) -> None:
    """Notebook を上から順に実行し、例外があればそのまま失敗させる。"""
    print(f"[notebook] 実行開始: {path.name}", flush=True)
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_DIR)}},
    )
    client.execute()
    print(f"[notebook] 実行完了: {path.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="robotorchan Notebook を実行確認します。")
    parser.add_argument(
        "--include-slow",
        action="store_true",
        help="通常 CI では除外する重い Notebook も実行します。",
    )
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="通常 CI 対象外の Notebook も実行します。",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="1セルあたりのタイムアウト秒数。既定値は 300 秒です。",
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=DEFAULT_NOTEBOOKS + SLOW_NOTEBOOKS + EXCLUDED_NOTEBOOKS,
        help="指定したNotebookのみ実行します。複数回指定できます。",
    )
    args = parser.parse_args()

    validate_notebook_manifest()

    notebook_names = list(DEFAULT_NOTEBOOKS)
    if args.include_slow:
        notebook_names.extend(SLOW_NOTEBOOKS)
    if args.include_excluded:
        notebook_names.extend(EXCLUDED_NOTEBOOKS)
    if args.only:
        notebook_names = list(dict.fromkeys(args.only))

    missing = [name for name in notebook_names if not (NOTEBOOK_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"Notebook が見つかりません: {missing}")

    for name in notebook_names:
        execute_notebook(NOTEBOOK_DIR / name, timeout=args.timeout)

    print(f"[notebook] {len(notebook_names)} 本の実行確認が完了しました。", flush=True)


if __name__ == "__main__":
    main()
