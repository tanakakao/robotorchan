# Phase 1 — リンク・CI・検証参照の依存関係監査（初版）

> Baseline: `main` `9de071c899dcce5fa2c7c3c0083281d529a3b077`。既存5ファイルは変更しない。静的な参照抽出と指定ファイルの直接読取に基づく。実行検証や全パスの存在確認は未実施。

## 既存5ファイルのMarkdownリンク（重複除外）

| Source | Distinct link targets |
| --- | --- |
| `README.md` | `models.md`, `../../theory/README.md`, `verification.md#phase-7-final-exit-reconciliation`, `acquisition.md`, `verification.md`, `optimization.md`, `README.md` |
| `models.md` | `README.md`, `../../theory/README.md`, `verification.md`, `README.md#reader-navigation-phase-10`, `../../../tests/models/standard/test_phase7_kronecker_covariance_reference.py` |
| `acquisition.md` | `README.md`, `../../theory/README.md`, `verification.md`, `README.md#reader-navigation-phase-10` |
| `optimization.md` | `README.md`, `../../theory/README.md`, `verification.md`, `README.md#reader-navigation-phase-10` |
| `verification.md` | `README.md`, `models.md`, `acquisition.md`, `optimization.md`, `../../theory/README.md` |

**移行時の高リスクアンカー**: `README.md#reader-navigation-phase-10` と `verification.md#phase-7-final-exit-reconciliation`。旧ファイルの削除・改題・見出し変更で壊れるため、旧ファイル保持または互換アンカーを確保する。

## CIで直接確認した依存

`.github/workflows/documentation-links.yml` はPRとmain pushで以下を実行する。

1. `python -m unittest discover -s tests/docs -p test_docs_links.py`
2. `python scripts/check_docs_links.py`
3. `python scripts/check_notebook_links.py`
4. `python scripts/check_docs_references.py`
5. `python -m unittest discover -s tests/docs -p test_theory_implementation_mapping.py`
6. `python scripts/check_theory_implementation_mapping.py`

`scripts/check_theory_implementation_mapping.py` は **既存5ファイルの存在を必須**とし、README→models/acquisition/optimization、verification→各subject、各subject→README/verification/theory の **特定のMarkdownリンク文字列**を要求する。新しい章を追加するだけではこのチェックは失敗しないが、旧ファイルを移動・削除・リンクを書換えると失敗する。Phase 2以降はチェック契約の意図を維持しながら新構造へ拡張する必要がある。

`tests/docs/test_theory_implementation_mapping.py` は、現行ナビゲーションが空エラーであること、README欠損の検出、subject→verificationの欠落検出をassertしている。変更時にこれらの否定テストを消してはいけない。

## 本文中の実装・テスト・Notebook参照（静的抽出の例）

| Source | 確認された参照例 | 保全上の意味 |
| --- | --- | --- |
| `README.md` | `src/robotorchan/models/registry.py`, `src/robotorchan/acquisition/registry.py`, `scripts/generate_model_coverage.py`, `docs/model_coverage.json` | registryと生成成果物のsource of truth |
| `models.md` | `tests/models/standard/test_mixed_single_task_gp.py`, `tests/models/standard/test_phase7_kronecker_covariance_reference.py`, `tests/models/classification/robust/test_label_noise.py` | model family別のassertion根拠 |
| `acquisition.md` | `tests/acquisition/test_phase5_native_integration.py`, `tests/acquisition/test_epig.py`, `examples/notebooks/28_regression_active_learning.ipynb` | acquisitionの実装・テスト・実行例 |
| `optimization.md` | `tests/optim/one_shot/test_mixed.py`, `tests/optim/embedding/test_baxus_reference_schedule.py`, `tests/optim/trust_region/test_turbo_strategy.py` | one-shot、embedding、stateful strategy |
| `verification.md` | `tests/models/classification/test_phase7_binary_probit_reference.py`, `tests/acquisition/test_phase8_bald_probability_variance_reference.py`, `tests/optim/trust_region/test_phase9_turbo_geometry_reference.py` | 数値参照のproperty-specific証拠 |

`README.md` のインラインコード形式のpath候補は18件、`models.md` は158件、`acquisition.md` は42件、`optimization.md` は36件、`verification.md` は21件（各ファイル内の重複除外、正規表現による抽出）。**存在確認済み件数ではない**。 `models/...` のような省略パス、`::Class` 指定、`{a,b}.py` の集合表記は、そのままファイルパスとして検査しない。

## 外部からの参照について

GitHubコード検索で `check_theory_implementation_mapping` は `README.md`、`tests/docs/test_theory_implementation_mapping.py`、`.github/workflows/documentation-links.yml` に見つかった。 `28_regression_active_learning` は `examples/run_notebooks.py` と `acquisition.md` に見つかった。検索結果は**網羅的な被参照リストではない**ため、全リポジトリのtreeと全文走査が必要。

## 移行ガードレール

- 旧5ファイルと既存のアンカーは、新ガイドへの移行期間中維持する。
- 新しい章の相対リンクは必ず新しいファイル位置を基準に解決する。
- `docs/model_coverage.json` と生成スクリプトの所有関係を変更しない。
- 新ガイドのCIチェック追加は、現行の否定テストとリンク検査を保持してから行う。
- 数値テスト名の記載は「存在・実行・独立参照による妥当性」をそれぞれ区別する。

## 未完了

- [ ] 全repo treeに対する被参照・実装/テスト/Notebookパスの存在確認。
- [ ] 全リンクのアンカー解決（`check_docs_references.py` 実行）。
- [ ] 全174見出しの本文レベルレビュー。
- [ ] 現行CIの実行結果確認と移行後の比較。
- [ ] Phase 1の完了判定。


## 追加確認：重要参照先の実在性（個別取得）

GitHubの作業ブランチ上で、以下の10ファイルを直接取得できた。**ファイルの存在は確認済み**だが、テストの実行成功や記述内容の数学的正当性は未確認。

| Category | Path | Result |
| --- | --- | --- |
| Model numerical test | `tests/models/standard/test_phase7_kronecker_covariance_reference.py` | 存在確認 |
| Classification numerical test | `tests/models/classification/test_phase7_binary_probit_reference.py` | 存在確認 |
| Acquisition numerical test | `tests/acquisition/test_phase8_bald_probability_variance_reference.py` | 存在確認 |
| Optimization numerical test | `tests/optim/trust_region/test_phase9_turbo_geometry_reference.py` | 存在確認 |
| Notebook | `examples/notebooks/28_regression_active_learning.ipynb` | 存在確認 |
| Model registry | `src/robotorchan/models/registry.py` | 存在確認 |
| Acquisition registry | `src/robotorchan/acquisition/registry.py` | 存在確認 |
| Optimizer dispatch | `src/robotorchan/optim/dispatch.py` | 存在確認 |
| Reference checker | `scripts/check_docs_references.py` | 存在確認 |
| Theory chapter | `docs/theory/24_classification.md` | 存在確認 |

### 次のレビューで優先する点

1. `models.md` のfamily別見出しに含まれる **技術的な主張** を章別本文とAppendixに二重取りこぼしなく割り当てる。
2. `verification.md` の各数値テストについて、テスト存在・実行結果・独立参照・検証限界を別列で管理する。
3. `README.md` と各subjectファイルのリンク先を新章へ移す際、既存のナビゲーション契約を維持する。

## 追加の参照先確認（第2バッチ）

既存5ファイルの本文から、特定形式の`src/robotorchan/`、`tests/`、`examples/notebooks/`、`scripts/`、`docs/`の明示パスを抽出した結果、重複除外で**178候補**を識別した。省略パス・波括弧展開・相対リンク・外部からの被参照はこの件数に含まれない。178件すべての存在確認は未完了。

追加で以下6件をGitHub作業ブランチから直接取得し、すべて存在を確認した。

| Path | Status |
| --- | --- |
| `docs/models/classification.md` | 取得成功 |
| `docs/optimization/epig.md` | 取得成功 |
| `docs/theory/acquisition/classification-active-learning.md` | 取得成功 |
| `scripts/audit_model_api.py` | 取得成功 |
| `scripts/audit_acquisition_api.py` | 取得成功 |
| `scripts/generate_model_coverage.py` | 取得成功 |

前回の個別取得10件と合わせて、**個別取得による確認は計16件**。実行結果や全参照の網羅性は未確認。以後は新規監査ファイルを増やさず、同じ台帳を更新する。
