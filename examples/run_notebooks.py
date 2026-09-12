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
    "07_pairwise_gp.ipynb",
    "09_map_saas_and_additive_gp.ipynb",
    "10_robust_gp.ipynb",
    "12_hierarchical_gp.ipynb",
    "13_heterogeneous_multitask_gp.ipynb",
    "14_contextual_gp.ipynb",
]

SLOW_NOTEBOOKS = [
    "08_saas_gp.ipynb",
    "11_structured_output_gp.ipynb",
]


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
        "--timeout",
        type=int,
        default=300,
        help="1セルあたりのタイムアウト秒数。既定値は 300 秒です。",
    )
    args = parser.parse_args()

    notebook_names = list(DEFAULT_NOTEBOOKS)
    if args.include_slow:
        notebook_names.extend(SLOW_NOTEBOOKS)

    missing = [name for name in notebook_names if not (NOTEBOOK_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"Notebook が見つかりません: {missing}")

    for name in notebook_names:
        execute_notebook(NOTEBOOK_DIR / name, timeout=args.timeout)

    print(f"[notebook] {len(notebook_names)} 本の実行確認が完了しました。", flush=True)


if __name__ == "__main__":
    main()
