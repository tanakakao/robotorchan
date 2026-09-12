# robotorchan examples

このディレクトリには `robotorchan.models` で公開しているモデルの実行可能な使用例を収録します。

中心となるサンプルは `examples/notebooks/` 以下の Jupyter Notebook です。モデルの選び方は [`docs/models.md`](../docs/models.md)、具体的な使い方は本ディレクトリ、実装の正しさは `tests/` を参照してください。

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
| [`08_saas_gp.ipynb`](notebooks/08_saas_gp.ipynb) | `SaasFullyBayesianSingleTaskGP`, `SaasFullyBayesianMultiTaskGP` | 利用可能 |
| [`09_map_saas_and_additive_gp.ipynb`](notebooks/09_map_saas_and_additive_gp.ipynb) | `AdditiveMapSaasSingleTaskGP`, `EnsembleMapSaasSingleTaskGP`, `OrthogonalAdditiveGP` | 利用可能 |
| [`10_robust_gp.ipynb`](notebooks/10_robust_gp.ipynb) | `RobustRelevancePursuitSingleTaskGP` | 利用可能 |
| [`11_structured_output_gp.ipynb`](notebooks/11_structured_output_gp.ipynb) | `HigherOrderGP`, `LatentKroneckerGP` | 利用可能 |
| [`12_hierarchical_gp.ipynb`](notebooks/12_hierarchical_gp.ipynb) | `HierarchicalConditionalKernelGP`, `HierarchicalConditionalKernelMultiTaskGP` | 利用可能 |
| [`13_heterogeneous_multitask_gp.ipynb`](notebooks/13_heterogeneous_multitask_gp.ipynb) | `HeterogeneousMTGP` | 利用可能 |
| [`14_contextual_gp.ipynb`](notebooks/14_contextual_gp.ipynb) | `SACGP`, `LCEAGP`, `LCEMGP` | 利用可能 |

## 再現性

各 Notebook の冒頭付近で固定 seed を設定します。

```python
import torch

torch.manual_seed(0)
```

モデル固有の理由がない限り GP の例では `torch.double` を優先します。

## 実行時間

通常の Notebook は日常的に実行できる程度の軽さを目標とします。Fully Bayesian NUTS は例外で、ドキュメント用には sampling 設定を小さくします。Notebook CI を追加する際は、重い MCMC Notebook を通常の軽量実行経路から分離します。

## ドキュメント・Notebook・テストの役割

- [`docs/models.md`](../docs/models.md): どのモデルを選ぶか、なぜ選ぶかを説明します。
- `examples/notebooks/`: 各モデルをどのように使うかを実行可能な形で示します。
- `tests/`: 実装挙動と互換性を検証します。

Notebook は利用方法を示すドキュメントであり、unit test の代替ではありません。
