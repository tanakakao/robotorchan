# robotorchan 理論ガイド

このディレクトリでは、`robotorchan` を使う上で必要となるベイズ最適化（Bayesian Optimization; BO）と Gaussian Process（GP）の理論を、実装と対応付けながら説明します。

`docs/models.md` が「どのモデルを選ぶか」を中心にした実務ガイドであるのに対し、本ディレクトリは「なぜそのモデルが使えるのか」「数式上は何をしているのか」を理解するための理論ガイドです。

## ドキュメント内での位置付け

実務上の入口は [モデル概要・使い所ガイド](../models.md) です。そこで候補モデルを絞り、この理論ガイドで統計的仮定や数式を確認し、最後に [Notebook一覧](../../examples/README.md) から対応するコードを実行する流れを推奨します。

[プロジェクトREADME](../../README.md) → [モデル選択](../models.md) → **理論確認（現在地）** → [実行例](../../examples/README.md)

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
12. [Hierarchical / Contextual GP](12_hierarchical_contextual_gp.md)
13. [Model Selection](13_model_selection.md)
14. [Robust Gaussian Process](14_robust_gaussian_process.md)
15. [Heteroskedastic / Replicate Noise](15_heteroskedastic_noise.md)
16. [Uncertain-input GP](16_uncertain_input_gp.md)
17. [Nonstationary GP](17_nonstationary_gp.md)
18. [Dimensionality Reduction GP](18_dimensionality_reduction_gp.md)
19. [Neural Representation Learning for GP](19_neural_reduction_gp.md)
20. [High-dimensional Search Strategies](20_high_dimensional_search.md)
21. [Advanced High-dimensional GP Models](21_advanced_high_dimensional_models.md)
22. [Expressive GP Models](22_expressive_gp.md)

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
| Robust / heavy-tailed observation | `RobustRelevancePursuitSingleTaskGP`, `StudentTSingleTaskGP`, `ContaminatedSingleTaskGP` |
| Heteroskedastic / replicate noise | `HeteroskedasticSingleTaskGP`, `JointHeteroskedasticSingleTaskGP`, `ReplicateNoiseSingleTaskGP` |
| Input uncertainty | `UncertainInputSingleTaskGP`, `UncertainCategoricalSingleTaskGP` |
| Nonstationarity | `NonstationarySingleTaskGP` |
| Input/output reduction | `PCAGP`, `PLSGP`, `OutputPCAGP`, `OutputPLSGP` |
| Neural reduction | `AutoEncoderGP`, `VAEGP`, `JointEncoderGP`, `JointVAEGP` |
| High-dimensional search | REMBO, HeSBO, ALEBO, TuRBO, BAxUS |
| Structured Output | `HigherOrderGP`, `LatentKroneckerGP` |
| Hierarchical search space | `HierarchicalConditionalKernelGP`, `HierarchicalConditionalKernelMultiTaskGP` |
| Heterogeneous Multi-task | `HeterogeneousMTGP` |
| Contextual GP | `SACGP`, `LCEAGP`, `LCEMGP` |
| Expressive GP | `JointEncoderGP`, `SingleTaskDeepGP`, `InfiniteWidthBNNGP`, `SpectralMixtureGP` |

## 関連ドキュメント

- [モデル概要・使い所](../models.md)
- [実装設計](../development/architecture.md)
- [Notebook 一覧](../../examples/README.md)
- [実行例](../../examples/notebooks/)

## Model guide との責務分離

`docs/models/` は「どのモデルを、どの条件で使うか」を扱い、public class、入力形式、学習契約、制約、Mixed / MultiTask 対応、Notebook への導線を記載します。

`docs/theory/` は「なぜそのモデルが成立するか」を扱い、確率モデル、kernel / likelihood、構造仮定、推論、獲得関数との関係を説明します。class ごとの API 一覧や同じ使用手順を theory 側へ重複させません。

新しいモデルが既存の統計的仮定を共有する場合は既存 theory chapter へ接続し、新しい class が増えたという理由だけで theory chapter を複製しません。一方、新しい likelihood、kernel、inference、search geometry など独立した理論仮定を導入する場合は、既存章への追記または新章を追加します。
