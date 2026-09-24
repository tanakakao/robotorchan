# robotorchan examples

このディレクトリには `robotorchan.models` の代表的なモデルファミリーと利用パターンの実行可能な使用例を収録します。すべての public class に1対1で Notebook を作るのではなく、同じ理論・学習契約を共有する派生モデルは概念単位でまとめます。

中心となるサンプルは `examples/notebooks/` 以下の Jupyter Notebook です。モデルの選び方は [`docs/models.md`](../docs/models.md)、具体的な使い方は本ディレクトリ、実装の正しさは `tests/` を参照してください。

ドキュメント全体では、[プロジェクトREADME](../README.md) → [Getting Started](../docs/README.md) →
[モデル選択](../docs/getting_started/model_selection.md) → [Model guides](../docs/models/README.md) →
[理論確認](../docs/theory/README.md) → **実行例（現在地）**、という導線を想定しています。モデルの背景を確認したい場合は対応する理論章へ戻り、実装詳細が必要な場合は [architecture](../docs/development/architecture.md) を参照してください。

## インストール

Notebook 用依存関係を含めてインストールします。

```bash
pip install -e ".[examples]"
```

Fully Bayesian SAAS の例も実行する場合は次を使用します。

```bash
pip install -e ".[examples,fully-bayesian]"
```

## Notebook の方針

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

| Notebook | 主なモデル | 状態 |
|---|---|---|
| [`01_single_task_gp.ipynb`](notebooks/01_single_task_gp.ipynb) | `SingleTaskGP` | 利用可能 |
| [`02_mixed_single_task_gp.ipynb`](notebooks/02_mixed_single_task_gp.ipynb) | `MixedSingleTaskGP` | 利用可能 |
| [`03_multi_fidelity_gp.ipynb`](notebooks/03_multi_fidelity_gp.ipynb) | `SingleTaskMultiFidelityGP` | 利用可能 |
| [`04_multitask_gp.ipynb`](notebooks/04_multitask_gp.ipynb) | `MultiTaskGP`, `KroneckerMultiTaskGP` | 利用可能 |
| [`05_model_list_gp.ipynb`](notebooks/05_model_list_gp.ipynb) | `ModelListGP` | 利用可能 |
| [`06_variational_gp.ipynb`](notebooks/06_variational_gp.ipynb) | `SingleTaskVariationalGP` | 利用可能 |
| [`07_pairwise_gp.ipynb`](notebooks/07_pairwise_gp.ipynb) | `PairwiseGP` | 利用可能 |
| [`08_saas_gp.ipynb`](notebooks/08_saas_gp.ipynb) | `SaasFullyBayesianSingleTaskGP`, `SaasFullyBayesianMultiTaskGP` | 利用可能 / 通常CI対象外 |
| [`09_map_saas_and_additive_gp.ipynb`](notebooks/09_map_saas_and_additive_gp.ipynb) | `AdditiveMapSaasSingleTaskGP`, `EnsembleMapSaasSingleTaskGP`, `OrthogonalAdditiveGP` | 利用可能 |
| [`10_robust_gp.ipynb`](notebooks/10_robust_gp.ipynb) | `RobustRelevancePursuitSingleTaskGP` | 利用可能 |
| [`11_structured_output_gp.ipynb`](notebooks/11_structured_output_gp.ipynb) | `HigherOrderGP`, `LatentKroneckerGP` | 利用可能 / 通常CI対象外 |
| [`12_hierarchical_gp.ipynb`](notebooks/12_hierarchical_gp.ipynb) | `HierarchicalConditionalKernelGP`, `HierarchicalConditionalKernelMultiTaskGP` | 利用可能 |
| [`13_heterogeneous_multitask_gp.ipynb`](notebooks/13_heterogeneous_multitask_gp.ipynb) | `HeterogeneousMTGP` | 利用可能 |
| [`14_contextual_gp.ipynb`](notebooks/14_contextual_gp.ipynb) | `SACGP`, `LCEAGP`, `LCEMGP` | 利用可能 |
| [`15_robust_observation_models.ipynb`](notebooks/15_robust_observation_models.ipynb) | Student-t, Contaminated | 利用可能 |
| [`16_noise_models.ipynb`](notebooks/16_noise_models.ipynb) | Replicate Noise | 利用可能 |
| [`17_uncertain_input_gp.ipynb`](notebooks/17_uncertain_input_gp.ipynb) | Uncertain Input / Categorical | 利用可能 |
| [`18_nonstationary_gp.ipynb`](notebooks/18_nonstationary_gp.ipynb) | Nonstationary GP | 利用可能 |
| [`19_reduced_gp.ipynb`](notebooks/19_reduced_gp.ipynb) | PCA / PLS / Random Projection | 利用可能 |
| [`20_neural_reduction_gp.ipynb`](notebooks/20_neural_reduction_gp.ipynb) | AE / VAE / Supervised AE | 利用可能 |
| [`21_reduced_multitask_gp.ipynb`](notebooks/21_reduced_multitask_gp.ipynb) | Reduced MultiTask | 利用可能 |
| [`22_mixed_reduced_gp.ipynb`](notebooks/22_mixed_reduced_gp.ipynb) | Mixed PCA / PLS | 利用可能 |
| [`23_high_dimensional_bo_benchmark.ipynb`](notebooks/23_high_dimensional_bo_benchmark.ipynb) | 高次元BO benchmark | 利用可能 / benchmark解説 |
| [`24_expressive_surrogate_gp.ipynb`](notebooks/24_expressive_surrogate_gp.ipynb) | DKL / I-BNN / Spectral Mixture / DeepGP | 利用可能 / 解説中心 |
| [`25_non_gp_surrogates.ipynb`](notebooks/25_non_gp_surrogates.ipynb) | Random Forest / Extra Trees / Gradient Boosting / NGBoost | 利用可能 |

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

通常CIから除外している `08_saas_gp.ipynb` と `11_structured_output_gp.ipynb` も含めて実行する場合は、必要な optional dependency をインストールしてから `--include-slow` を付けます。

```bash
pip install -e ".[examples,fully-bayesian]"
python examples/run_notebooks.py --include-slow --timeout 900
```

## Notebook CI

GitHub Actions の `notebooks` job では Python 3.11 / CPU 環境を使用し、次の Notebook を実行します。

```text
01, 02, 03, 04, 05, 06, 07,
09, 10,
12, 13, 14, 15, 16, 17, 18,
19, 20, 21, 22
```

以下は通常の Pull Request CI から除外します。

- `08_saas_gp.ipynb`: NUTS / MCMC を使うため実行時間の変動が大きい
- `11_structured_output_gp.ipynb`: HOGP / LatentKroneckerGP の学習が比較的重く、通常の軽量CIから分離した方が安定する

`08` のwrapper自体は既存の `fully-bayesian-extra` job でテストします。重いNotebookも必要に応じて `examples/run_notebooks.py --include-slow` で明示的に実行できます。

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
