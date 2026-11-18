# robotorchan examples

このディレクトリには `robotorchan.models` の代表的なモデルファミリーと利用パターンの実行可能な使用例を収録します。すべての public class に1対1で Notebook を作るのではなく、同じ理論・学習契約を共有する派生モデルは概念単位でまとめます。

中心となるサンプルは `examples/notebooks/` 以下の Jupyter Notebook です。モデルの選び方は [`docs/models.md`](../docs/models.md)、具体的な使い方は本ディレクトリ、実装の正しさは `tests/` を参照してください。

ドキュメント全体では、[プロジェクトREADME](../README.md) → [Getting Started](../docs/README.md) →
[モデル選択](../docs/getting_started/model_selection.md) → [Model guides](../docs/models/README.md) →
[理論確認](../docs/theory/README.md) → **実行例（現在地）**、という導線を想定しています。モデルの背景を確認したい場合は対応する理論章へ戻り、実装詳細が必要な場合は [architecture](../docs-internal/architecture/architecture.md) を参照してください。

## インストール

Notebook 用依存関係を含めてインストールします。

```bash
pip install -e ".[examples]"
```

Fully Bayesian SAAS の例も実行する場合は次を使用します。

```bash
pip install -e ".[examples,fully-bayesian]"
```

## 単体実行できるPython例

[`unified_api_workflow.py`](unified_api_workflow.py) では、2出力Exact GPのfit、
BoTorch標準Sampler / Objective / qSimpleRegretとrobotorchanのSobol Optimizerを接続します。

```bash
python examples/unified_api_workflow.py
```

契約の詳細は [Unified API workflow](../docs/optimization/unified-api-workflow.md) にまとめています。

## ベンチマークとの使い分け

モデル・獲得関数の学習例は以下の Notebook を参照してください。複数 seed の公平な実測比較や保存済み結果の再生成には [Benchmark Quickstart](../docs/benchmarks/quickstart.md) を使用します。Notebook の1回の実行結果だけでは性能の優位性を判断できません。

## Notebook の方針

Notebook の標準構成・metadata・Tier A/B/C の要件は、
[`notebook-authoring-standard.md`](../docs-internal/maintenance/notebook-authoring-standard.md) を
source of truth とします。CI の default / slow / specialist 実行方針は
[`notebook-ci.md`](../docs-internal/maintenance/notebook-ci.md) を参照してください。

各 Notebook は単体で理解でき、上から順番に実行できる構成にします。説明文と見出しは原則として日本語とし、モデル名・API名・Bayesian Optimization / posterior / likelihood など、コードと対応付けた方が分かりやすい技術用語は英語表記を残します。

基本構成は次のとおりです。

1. このモデルを使う場面
2. Import と再現性設定
3. 合成データ
4. モデル構築
5. robotorchan 共通 API
6. モデル学習
7. Posterior 予測
8. 可視化
9. Bayesian Optimization の例（適用できる場合）
10. 使い所と注意点

### 共通 API

supervised wrapper では、対応している場合に次の robotorchan 共通 API を確認します。

```python
model.raw_train_X
model.raw_train_Y
model.raw_train_Yvar
model.raw_data
model.raw_data_names
model.supports_mll
model.make_mll()
```

特殊なモデルでは、無理に supervised API に合わせず、そのモデル固有の raw-data interface を示します。例えば `PairwiseGP` は pairwise comparison を扱い、Fully Bayesian SAAS は exact-MML の `make_mll()` を意図的に提供しません。

## データ

実データがモデル理解に不可欠でない限り、小さな合成データを使用します。これにより再現性、実行速度、ネットワーク非依存性、理解しやすさ、将来の CI 実行可能性を保ちます。

主要なロジックを隠す共通 helper module は避け、多少の重複であれば Notebook 内に小さな関数を定義します。

## 可視化

軽量な可視化には Matplotlib を使用します。通常の回帰例では、可能な範囲で学習観測値、posterior mean、不確実性、BO の候補点を可視化します。

## 現在の coverage

Notebook は標準 GP から Robust / Noise / Input uncertainty / Nonstationary、linear / neural reduction、Reduced MultiTask、Mixed reduction までを概念単位でカバーします。1 public class = 1 Notebook にはせず、同じ統計的仮定や学習契約を共有するモデルは同じ Notebook で比較します。

## Notebook 一覧

理論の基礎からモデル構築、獲得関数、最適化、応用へ進める順に01〜57を採番しています。CIの実行順・実行区分は `run_notebooks.py` で管理します。

### GPの基礎

| Notebook | 主なモデル | 状態 |
|---|---|---|
| [`01_single_task_gp.ipynb`](notebooks/01_single_task_gp.ipynb) | `SingleTaskGP` | 利用可能 |
| [`02_mixed_single_task_gp.ipynb`](notebooks/02_mixed_single_task_gp.ipynb) | `MixedSingleTaskGP` | 利用可能 |
| [`03_multi_fidelity_gp.ipynb`](notebooks/03_multi_fidelity_gp.ipynb) | `SingleTaskMultiFidelityGP` | 利用可能 |
| [`04_multitask_gp.ipynb`](notebooks/04_multitask_gp.ipynb) | `MultiTaskGP`, `KroneckerMultiTaskGP` | 利用可能 |
| [`05_model_list_gp.ipynb`](notebooks/05_model_list_gp.ipynb) | `ModelListGP` | 利用可能 |
| [`06_variational_gp.ipynb`](notebooks/06_variational_gp.ipynb) | `SingleTaskVariationalGP` | 利用可能 |

### 発展的なGPモデル

| Notebook | 主なモデル | 状態 |
|---|---|---|
| [`07_saas_gp.ipynb`](notebooks/07_saas_gp.ipynb) | `SaasFullyBayesianSingleTaskGP`, `SaasFullyBayesianMultiTaskGP` | 利用可能 / 通常CI対象外 |
| [`08_map_saas_and_additive_gp.ipynb`](notebooks/08_map_saas_and_additive_gp.ipynb) | `AdditiveMapSaasSingleTaskGP`, `EnsembleMapSaasSingleTaskGP`, `OrthogonalAdditiveGP` | 利用可能 |
| [`09_structured_output_gp.ipynb`](notebooks/09_structured_output_gp.ipynb) | `HigherOrderGP`, `LatentKroneckerGP` | 利用可能 / 通常CI対象外 |
| [`10_hierarchical_gp.ipynb`](notebooks/10_hierarchical_gp.ipynb) | `HierarchicalConditionalKernelGP`, `HierarchicalConditionalKernelMultiTaskGP` | 利用可能 |
| [`11_heterogeneous_multitask_gp.ipynb`](notebooks/11_heterogeneous_multitask_gp.ipynb) | `HeterogeneousMTGP` | 利用可能 |
| [`12_contextual_gp.ipynb`](notebooks/12_contextual_gp.ipynb) | `SACGP`, `LCEAGP`, `LCEMGP` | 利用可能 |
| [`13_robust_gp.ipynb`](notebooks/13_robust_gp.ipynb) | `RobustRelevancePursuitSingleTaskGP` | 利用可能 |
| [`14_robust_observation_models.ipynb`](notebooks/14_robust_observation_models.ipynb) | Student-t, Contaminated | 利用可能 |
| [`15_noise_models.ipynb`](notebooks/15_noise_models.ipynb) | Replicate Noise | 利用可能 |
| [`16_uncertain_input_gp.ipynb`](notebooks/16_uncertain_input_gp.ipynb) | Uncertain Input / Categorical | 利用可能 |
| [`17_nonstationary_gp.ipynb`](notebooks/17_nonstationary_gp.ipynb) | Nonstationary GP | 利用可能 |

### 次元削減・表現力の高いモデル

| Notebook | 主なモデル | 状態 |
|---|---|---|
| [`18_reduced_gp.ipynb`](notebooks/18_reduced_gp.ipynb) | PCA / PLS / Random Projection | 利用可能 |
| [`19_neural_reduction_gp.ipynb`](notebooks/19_neural_reduction_gp.ipynb) | AE / VAE / Supervised AE | 利用可能 |
| [`20_reduced_multitask_gp.ipynb`](notebooks/20_reduced_multitask_gp.ipynb) | Reduced MultiTask | 利用可能 |
| [`21_mixed_reduced_gp.ipynb`](notebooks/21_mixed_reduced_gp.ipynb) | Mixed PCA / PLS | 利用可能 |
| [`22_expressive_surrogate_gp.ipynb`](notebooks/22_expressive_surrogate_gp.ipynb) | DKL / I-BNN / Spectral Mixture / DeepGP | 利用可能 / 解説中心 |
| [`23_non_gp_surrogates.ipynb`](notebooks/23_non_gp_surrogates.ipynb) | Random Forest / Extra Trees / Gradient Boosting / NGBoost | 利用可能 |

### 選好学習

| Notebook | 主なモデル | 状態 |
|---|---|---|
| [`24_pairwise_gp.ipynb`](notebooks/24_pairwise_gp.ipynb) | `PairwiseGP` | 利用可能 |
| [`25_mixed_preference_bo.ipynb`](notebooks/25_mixed_preference_bo.ipynb) | Mixed Preference BO | 利用可能 |

### 分類・分類Active Learning

| Notebook | 主なモデル | 状態 |
|---|---|---|
| [`26_classification_active_learning.ipynb`](notebooks/26_classification_active_learning.ipynb) | Binary GP Classification / AL | 利用可能 |
| [`27_binary_classification_extensions.ipynb`](notebooks/27_binary_classification_extensions.ipynb) | 拡張Binary GP / Mixed / q-batch Active Learning | 利用可能 |
| [`28_classification_robust_uncertain.ipynb`](notebooks/28_classification_robust_uncertain.ipynb) | Robust / Uncertain-input Classification | 利用可能 |
| [`29_classification_probability_layers.ipynb`](notebooks/29_classification_probability_layers.ipynb) | Ensemble / Calibration / Reliability / Conformal | 利用可能 |
| [`30_multiclass_classification_active_learning.ipynb`](notebooks/30_multiclass_classification_active_learning.ipynb) | Multiclass GP分類 / Active Learning | 利用可能 |
| [`31_ordinal_classification_active_learning.ipynb`](notebooks/31_ordinal_classification_active_learning.ipynb) | 順序GP分類 / Mixed / q-batch / Ask-and-Tell | 利用可能 |

### 獲得関数・Posterior・Sampling

| Notebook | 主なモデル | 状態 |
|---|---|---|
| [`32_standard_acquisition.ipynb`](notebooks/32_standard_acquisition.ipynb) | Standard BO acquisition | 利用可能 |
| [`33_multiobjective_acquisition.ipynb`](notebooks/33_multiobjective_acquisition.ipynb) | Multi-objective acquisition | 利用可能 |
| [`34_active_learning_acquisition.ipynb`](notebooks/34_active_learning_acquisition.ipynb) | Regression Active Learning | 利用可能 |
| [`35_regression_active_learning.ipynb`](notebooks/35_regression_active_learning.ipynb) | Regression Active Learning | 利用可能 |
| [`36_objective_posterior_transform.ipynb`](notebooks/36_objective_posterior_transform.ipynb) | Objective / PosteriorTransform | 利用可能 |
| [`37_posterior_sampling.ipynb`](notebooks/37_posterior_sampling.ipynb) | Posterior / Sampling | 利用可能 |

### 候補最適化・制約

| Notebook | 主なモデル | 状態 |
|---|---|---|
| [`38_acquisition_initialization.ipynb`](notebooks/38_acquisition_initialization.ipynb) | Acquisition Initialization | 利用可能 |
| [`39_acquisition_optimization.ipynb`](notebooks/39_acquisition_optimization.ipynb) | Acquisition Optimization | 利用可能 |
| [`40_batch_async_fantasization.ipynb`](notebooks/40_batch_async_fantasization.ipynb) | Batch / Async / Fantasization | 利用可能 |
| [`41_outcome_constrained_bo.ipynb`](notebooks/41_outcome_constrained_bo.ipynb) | Outcome-constrained BO | 利用可能 |
| [`42_nonlinear_candidate_constraints.ipynb`](notebooks/42_nonlinear_candidate_constraints.ipynb) | Nonlinear candidate constraints | 利用可能 |
| [`43_heterogeneous_constrained_bo.ipynb`](notebooks/43_heterogeneous_constrained_bo.ipynb) | Heterogeneous constrained BO | 利用可能 |

### BO実行フロー・Study

| Notebook | 主なモデル | 状態 |
|---|---|---|
| [`44_sequential_bo_workflow.ipynb`](notebooks/44_sequential_bo_workflow.ipynb) | Sequential BO workflow | 利用可能 |
| [`45_end_to_end_workflows.ipynb`](notebooks/45_end_to_end_workflows.ipynb) | Sequential / Batch / Async E2E | 利用可能 |
| [`46_study_ask_tell_mixed_q_batch.ipynb`](notebooks/46_study_ask_tell_mixed_q_batch.ipynb) | Study / Ask-and-Tell | 利用可能 |

### 高度なBayesian Optimization

| Notebook | 主なモデル | 状態 |
|---|---|---|
| [`47_turbo_trust_region.ipynb`](notebooks/47_turbo_trust_region.ipynb) | TuRBO / Trust Region | 利用可能 |
| [`48_robust_input_perturbation.ipynb`](notebooks/48_robust_input_perturbation.ipynb) | Robust BO / Input Perturbation | 利用可能 |
| [`49_multifidelity_cost_aware.ipynb`](notebooks/49_multifidelity_cost_aware.ipynb) | Multi-Fidelity / Cost-aware BO | 利用可能 |
| [`50_safe_bo_usage_guide.ipynb`](notebooks/50_safe_bo_usage_guide.ipynb) | Safe BO 日本語利用ガイド | 利用可能 |
| [`51_safe_bo_mixed_q_batch.ipynb`](notebooks/51_safe_bo_mixed_q_batch.ipynb) | Safe BO Mixed / q-batch | 利用可能 |
| [`52_transfer_bayesian_optimization.ipynb`](notebooks/52_transfer_bayesian_optimization.ipynb) | Transfer BO | 利用可能 |
| [`53_combinatorial_bayesian_optimization.ipynb`](notebooks/53_combinatorial_bayesian_optimization.ipynb) | Combinatorial BO | 利用可能 |

### ベンチマーク

| Notebook | 主なモデル | 状態 |
|---|---|---|
| [`54_high_dimensional_bo_benchmark.ipynb`](notebooks/54_high_dimensional_bo_benchmark.ipynb) | 高次元BO benchmark | 利用可能 / benchmark解説 |
| [`55_heterogeneous_bo_benchmark.ipynb`](notebooks/55_heterogeneous_bo_benchmark.ipynb) | Heterogeneous BO benchmark | 利用可能 |
| [`56_turbo_multi_benchmark.ipynb`](notebooks/56_turbo_multi_benchmark.ipynb) | TuRBO-m benchmark | 利用可能 |
| [`57_safe_bo_benchmark.ipynb`](notebooks/57_safe_bo_benchmark.ipynb) | Safe BO benchmark | 利用可能 |


## Benchmark

再現可能な性能比較・探索戦略比較は `benchmarks/` に配置します。Notebook は利用方法や結果の読み方を説明する役割とし、benchmark本体のロジックをNotebookへ重複実装しません。高次元入力モデル・高次元MultiTask・acquisition optimization・sequential BO・batch BO の実行コードは `benchmarks/`、その契約テストは `tests/benchmarks/` に集約します。

高次元MultiTaskの実行方法は [`docs/models/high_dimensional_multitask.md`](../docs/models/high_dimensional_multitask.md)、モデル選択は [`docs/models/high_dimensional_model_selection.md`](../docs/models/high_dimensional_model_selection.md) を参照してください。

## Notebook の一括実行

通常のCI対象となる軽量Notebookは次のコマンドで一括実行できます。

```bash
python examples/run_notebooks.py
```

1セルあたりのタイムアウトを変更する場合は `--timeout` を指定します。

```bash
python examples/run_notebooks.py --timeout 600
```

通常CIから除外している `07_saas_gp.ipynb` と `09_structured_output_gp.ipynb` も含めて実行する場合は、必要な optional dependency をインストールしてから `--include-slow` を付けます。

```bash
pip install -e ".[examples,fully-bayesian]"
python examples/run_notebooks.py --include-slow --timeout 900
```

## Notebook CI

GitHub Actions の `notebooks` job は Python 3.11 / CPU 環境で
`examples/run_notebooks.py` の `DEFAULT_NOTEBOOKS` を実行します。
実行対象の正本は同スクリプトのmanifestです。番号の一覧をここに重複管理しません。

`SLOW_NOTEBOOKS` の `07_saas_gp.ipynb` と
`09_structured_output_gp.ipynb` は通常の `notebooks` job から外れますが、
専用の `notebook-saas` / `notebook-structured` job で実行します。
`EXCLUDED_NOTEBOOKS` は通常の一括実行対象外で、個別のCI jobで実行するものと
手動実行を想定するものが含まれます。特定のNotebookのCI実行状況は
`.github/workflows/ci.yml` と関連workflowを確認してください。

`--include-slow` は `SLOW_NOTEBOOKS` のみ追加します。
`EXCLUDED_NOTEBOOKS` も一括実行する場合は `--include-excluded` を指定できます。
特定の1件を確認する場合は `--only ファイル名.ipynb` を使用してください。

CIでは `MPLBACKEND=Agg` を使用し、GUIを必要とせずMatplotlibの描画セルを実行します。

## 再現性

各 Notebook の冒頭付近で固定 seed を設定します。

```python
import torch

torch.manual_seed(0)
```

モデル固有の理由がない限り GP の例では `torch.double` を優先します。

## 実行時間

通常の Notebook は日常的に実行できる程度の軽さを目標とします。Fully Bayesian NUTS は例外で、ドキュメント用には sampling 設定を小さくします。計算負荷が高いNotebookは通常CIとは分離します。

## ドキュメント・Notebook・テストの役割

- [`docs/models.md`](../docs/models.md): どのモデルを選ぶか、なぜ選ぶかを説明します。
- `examples/notebooks/`: 各モデルをどのように使うかを実行可能な形で示します。
- `tests/`: 実装挙動と互換性を検証します。

Notebook は利用方法を示すドキュメントであり、unit test の代替ではありません。


## TuRBO

`examples/turbo.py` は `TuRBOStrategy` の最小 stateful loop です。モデルfit、acquisition
生成、local candidate最適化、評価、`update_state()`、必要時の`restart()`という責務分離を
そのまま示します。Mixed / MultiFidelity / constraints / robust utility の詳細は
`docs/optimization/turbo.md` を参照してください。

## Ordinal GP の比較ベンチマーク

[順序分類のベンチマーク](benchmarks/ordinal_active_learning.py)は、
連続入力とMixed入力に対して、Random・Predictive Entropy・BALDで
順序ラベルを追加する順番と保留データの指標を比較します。

```bash
python examples/benchmarks/ordinal_active_learning.py --seeds 3 --budget 5 --epochs 8
```

同じseed内の比較手法は初期ラベル、候補プール、保留データ、ラベル予算を共有します。
Accuracy・NLL・Brier score・ordinal MAEをCSVへ記録します。
これは合成データの再現可能な`q=1`比較であり、実環境での優劣を保証しません。
モデルの数理・利用APIは
[Ordinal GP 理論](../docs/theory/24_classification.md)、
[Ordinal GP Guide](../docs/guide/24_classification.md)と
[31番Notebook](notebooks/31_ordinal_classification_active_learning.ipynb)を参照してください。
