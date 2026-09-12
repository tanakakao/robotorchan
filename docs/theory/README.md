# robotorchan 理論ガイド

このディレクトリでは、`robotorchan` を使う上で必要となるベイズ最適化（Bayesian Optimization; BO）と Gaussian Process（GP）の理論を、実装と対応付けながら説明します。

`docs/models.md` が「どのモデルを選ぶか」を中心にした実務ガイドであるのに対し、本ディレクトリは「なぜそのモデルが使えるのか」「数式上は何をしているのか」を理解するための理論ガイドです。

## 推奨する読み方

初めてベイズ最適化を学ぶ場合は、次の順で読むことを推奨します。

1. [ベイズ最適化とは](01_bayesian_optimization.md)
2. [Gaussian Process](02_gaussian_process.md)
3. [Kernel](03_kernel.md)
4. [Acquisition Function](04_acquisition_function.md)
5. [Mixed Variables](05_mixed_variables.md)
6. [Multi-Fidelity](06_multi_fidelity.md)
7. [Multi-task / Multi-output](07_multitask_multioutput.md)
8. [High-dimensional GP](08_high_dimensional_gp.md)
9. [Variational GP](09_variational_gp.md)
10. [Preference Learning](10_preference_learning.md)
11. [Structured Output](11_structured_output.md)
12. Hierarchical / Contextual GP
13. Model Selection

## このガイドの構成方針

各章は、可能な限り次の順番で説明します。

1. **直感** — 何を解決するための考え方か
2. **数式** — 最低限必要な定式化
3. **モデル化上の意味** — 仮定を変えると何が変わるか
4. **ベイズ最適化との関係** — 次点選択にどう影響するか
5. **robotorchan との対応** — 実装上どのモデル・APIに対応するか
6. **使い所と注意点** — 実務で選ぶ際の判断基準

## 理論と実装の対応

| 理論テーマ | 主な robotorchan モデル |
|---|---|
| 標準 Gaussian Process | `SingleTaskGP` |
| 連続 + カテゴリ変数 | `MixedSingleTaskGP` |
| Multi-Fidelity | `SingleTaskMultiFidelityGP` |
| Multi-task | `MultiTaskGP`, `KroneckerMultiTaskGP` |
| 独立 Multi-output | `ModelListGP` |
| 大規模データ / Sparse GP | `SingleTaskVariationalGP` |
| Preference Learning | `PairwiseGP` |
| 高次元 BO | `SaasFullyBayesianSingleTaskGP`, `SaasFullyBayesianMultiTaskGP` |
| MAP-SAAS / Additive GP | `AdditiveMapSaasSingleTaskGP`, `EnsembleMapSaasSingleTaskGP`, `OrthogonalAdditiveGP` |
| Robust GP | `RobustRelevancePursuitSingleTaskGP` |
| Structured Output | `HigherOrderGP`, `LatentKroneckerGP` |
| Hierarchical search space | `HierarchicalConditionalKernelGP`, `HierarchicalConditionalKernelMultiTaskGP` |
| Heterogeneous Multi-task | `HeterogeneousMTGP` |
| Contextual GP | `SACGP`, `LCEAGP`, `LCEMGP` |

## 関連ドキュメント

- [モデル概要・使い所](../models.md)
- [実装設計](../architecture.md)
- [Notebook 一覧](../../examples/README.md)
- [実行例](../../examples/notebooks/)

## 今後追加する章

以下は今後の理論章として順次追加します。

- `12_hierarchical_contextual_gp.md`
- `13_model_selection.md`
