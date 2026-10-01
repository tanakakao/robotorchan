# robotorchan 理論ガイド

このディレクトリでは、`robotorchan` を使う上で必要となるベイズ最適化（Bayesian Optimization; BO）と
Gaussian Process（GP）の理論を、実装と対応付けながら説明します。

`docs/models.md` が「どのモデルを選ぶか」を中心にした実務ガイドであるのに対し、
本ディレクトリは「なぜそのモデルが使えるのか」「数式上は何をしているのか」を理解するための
理論ガイドです。

## ドキュメント内での位置付け

実務上の入口は [モデル概要・使い所ガイド](../models.md) です。そこで候補モデルを絞り、
この理論ガイドで統計的仮定や数式を確認します。最後に
[Notebook一覧](../../examples/README.md) から対応するコードを実行する流れを推奨します。

[プロジェクトREADME](../../README.md) → [モデル選択](../models.md)
→ **理論確認（現在地）** → [実行例](../../examples/README.md)

## 推奨する読み方

初めてベイズ最適化を学ぶ場合は、次の順で読むことを推奨します。

1. [ベイズ最適化とは](01_bayesian_optimization.md)
2. [Gaussian Process](02_gaussian_process.md)
3. [Kernel](03_kernel.md)
4. [Acquisition Function](04_acquisition_function.md)
   - [Acquisition Function Theory](acquisition/README.md) — 01〜12の詳細理論
   - [Acquisition Selection Guide](acquisition/12_selection_guide.md) — 問題設定から獲得関数を選ぶ入口
   - [Optimization guides](../optimization/README.md) — API・利用方法
   - [Acquisition integration status](../optimization/acquisition-integration.md) — 現在の対応範囲と制約
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
23. [Non-GP Surrogates](23_non_gp_surrogates.md)

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
| MAP-SAAS / Additive GP | `AdditiveMapSaasSingleTaskGP`, `EnsembleMapSaasSingleTaskGP`,<br>`OrthogonalAdditiveGP` |
| Robust / heavy-tailed observation | `RobustRelevancePursuitSingleTaskGP`, `StudentTSingleTaskGP`,<br>`ContaminatedSingleTaskGP` |
| Heteroskedastic / replicate noise | `HeteroskedasticSingleTaskGP`, `JointHeteroskedasticSingleTaskGP`,<br>`ReplicateNoiseSingleTaskGP` |
| Input uncertainty | `UncertainInputSingleTaskGP`, `UncertainCategoricalSingleTaskGP` |
| Nonstationarity | `NonstationarySingleTaskGP` |
| Input/output reduction | `PCAGP`, `PLSGP`, `OutputPCAGP`, `OutputPLSGP` |
| Neural reduction | `AutoEncoderGP`, `VAEGP`, `JointEncoderGP`, `JointVAEGP` |
| High-dimensional search | REMBO, HeSBO, ALEBO, TuRBO, BAxUS |
| Structured Output | `HigherOrderGP`, `LatentKroneckerGP` |
| Hierarchical search space | `HierarchicalConditionalKernelGP`,<br>`HierarchicalConditionalKernelMultiTaskGP` |
| Heterogeneous Multi-task | `HeterogeneousMTGP` |
| Contextual GP | `SACGP`, `LCEAGP`, `LCEMGP` |
| Expressive GP | `JointEncoderGP`, `SingleTaskDeepGP`, `InfiniteWidthBNNGP`, `SpectralMixtureGP` |
| Non-GP empirical ensemble | `RandomForestSurrogate`, `ExtraTreesSurrogate`,<br>`GradientBoostingSurrogate`, `HistGradientBoostingSurrogate` |
| Distributional non-GP | `NGBoostSurrogate` |

## 関連ドキュメント

- [モデル概要・使い所](../models.md)
- [実装設計](../development/architecture.md)
- [Notebook 一覧](../../examples/README.md)
- [実行例](../../examples/notebooks/)

## Model guide との責務分離

`docs/models/` は「どのモデルを、どの条件で使うか」を扱い、public class、入力形式、学習契約、
制約、Mixed / MultiTask 対応、Notebook への導線を記載します。

`docs/theory/` は「なぜそのモデルが成立するか」を扱い、確率モデル、kernel / likelihood、
構造仮定、推論、獲得関数との関係を説明します。class ごとの API 一覧や同じ使用手順を
 theory 側へ重複させません。

新しいモデルが既存の統計的仮定を共有する場合は既存 theory chapter へ接続し、
新しい class が増えたという理由だけで theory chapter を複製しません。
一方、新しい likelihood、kernel、inference、search geometry など独立した理論仮定を
導入する場合は、既存章への追記または新章を追加します。


## 理論体系の全体像

各章は単なるモデル一覧ではなく、次の責務ごとに分けています。

| Layer | 理論テーマ | 主な章 |
| --- | --- | --- |
| BO foundation | surrogate と acquisition による逐次意思決定 | 01, 04 |
| GP foundation | posterior、kernel、likelihood | 02, 03 |
| Input structure | Mixed、MultiFidelity、階層・context | 05, 06, 12 |
| Output structure | MultiTask / Multi-output、structured output | 07, 11 |
| Inference | exact / variational / fully Bayesian | 02, 09, 21 |
| Robustness | heavy tail、noise、input uncertainty、nonstationarity | 14-17 |
| Representation | linear / neural dimensionality reduction | 18, 19 |
| Search geometry | embedding、trust region、adaptive subspace | 20 |
| High-dimensional prior | SAAS、MAP-SAAS、additive structure | 21 |
| Expressiveness | DKL、DeepGP、NNGP、spectral mixture | 22 |
| Non-GP posterior | tree ensemble、boosting、NGBoost | 23 |
| Selection | problem assumptions と model capability の対応 | 13 |

この分類では、1つの model が複数 layer に関係することがあります。

例えば `JointEncoderGP` は neural representation と expressive kernel geometry の両方に関係します。
重複した class description を各章へコピーするのではなく、それぞれの章で異なる理論責務を説明します。

## Model・Posterior・Search を分離する

robotorchan の理論では、次の3層を明示的に分離します。

```text
Surrogate model
 -> predictive posterior / samples
 -> acquisition function
 -> candidate-search strategy
```

例えば、

- `ALEBOGP` は surrogate geometry
- `ALEBOStrategy` は candidate-search geometry
- `JointEncoderGP` は model-internal representation
- REMBO / HeSBO / BAxUS は search-space representation
- `RandomForestSurrogate` は empirical posterior model
- `TreeEnsembleSearchStrategy` は gradient-free candidate optimization

を担当します。

「高次元対応」「Mixed対応」「BoTorch互換」というラベルだけで、これらの責務を混同しません。

## Inference と posterior semantics

同じ `posterior(X)` interface を持っていても statistical meaning は異なります。

| Family | 主な inference / predictive semantics |
| --- | --- |
| Exact GP | Gaussian posterior from exact GP inference |
| Variational GP | variational approximation |
| Fully Bayesian SAAS | MCMC-integrated GP posterior |
| DeepGP | stochastic variational / Monte Carlo posterior |
| Tree ensemble | empirical member distribution |
| Bootstrap boosting | empirical distribution of complete fitted models |
| NGBoost | conditional predictive distribution |

BoTorch-compatible interface は、これらが同じ確率モデルであることを意味しません。

acquisition compatibility は posterior requirements、sampling support、multi-output、
fantasization などの capability と合わせて判断します。

## Mixed 対応は実装方式まで確認する

`MixedXxxGP` という名前だけでは categorical treatment は決まりません。

現在の model families には、

- native categorical covariance
- model-owned one-hot encoding
- continuous-only representation + categorical passthrough
- learnable categorical embedding
- numeric category validation only

という複数方式があります。

例えば Mixed SAAS / MAP-SAAS と Mixed OAK は内部 one-hot route を持ち、
Mixed Joint neural GP は continuous representation と categorical covariance を分離します。

一方 non-GP tree surrogate の `cat_dims` validation は、sklearn backend が native categorical
split semantics を持つことを意味しません。

各章では model name だけでなく、この内部 semantics まで記述します。

## 高次元対応の分類

高次元対応も1種類ではありません。

```text
sparse original-feature relevance
 -> SAAS / MAP-SAAS

additive decomposition
 -> OrthogonalAdditiveGP

data-driven representation
 -> PCA / PLS / AE / VAE / joint neural models

fixed or learned search embedding
 -> REMBO / HeSBO / ALEBO

local trust region
 -> TuRBO

adaptive search subspace
 -> BAxUS
```

model-side dimensionality reduction と candidate-search-space reduction を同一視しません。

## Robustness の分類

robustness も原因ごとに分けます。

| 問題 | 主な理論章 |
| --- | --- |
| heavy-tailed / contaminated observation | 14 |
| input relevance / sparse correction | 14 |
| heteroskedastic observation noise | 15 |
| replicate-based noise estimation | 15 |
| uncertain input | 16 |
| nonstationary function geometry | 17 |

外れ値、noise、input uncertainty、nonstationarity はすべて予測 uncertainty を増やし得ますが、
同じ statistical mechanism ではありません。

## Acquisition theory との接続

Acquisition Function の概要は
[Acquisition Function](04_acquisition_function.md)
に置き、詳細理論は
[Acquisition Function Theory](acquisition/README.md)
へ分離します。

model chapter 側では acquisition formula を重複して網羅せず、

- posterior requirement
- sampleability
- joint covariance requirement
- multi-output requirement
- fantasization requirement
- differentiability / search strategy

の観点から接続可能性を説明します。

## 実装 coverage を確認する基準

理論ドキュメント監査では、各 model family について少なくとも次を確認します。

1. public class / strategy が理論章に対応しているか
2. raw input と internal representation の関係が説明されているか
3. fitting contract が実装と一致するか
4. posterior semantics が説明されているか
5. Mixed / MultiTask / Kronecker / MultiFidelity の対応範囲を過大評価していないか
6. acquisition compatibility の前提が説明されているか
7. search-space strategy と surrogate responsibility を混同していないか
8. unsupported combination が暗黙に supported と読めないか
9. model-specific limitations が guide と矛盾していないか
10. canonical literature へ辿れるか

新規 model を追加した場合も、この基準で theory coverage を確認します。

## 参考文献の方針

理論上重要な claim は、可能な限り原論文、標準的 textbook、または upstream method paper へ
接続します。

参考文献は次の優先順位で選びます。

1. method の原論文
2. canonical extension / inference paper
3. Gaussian Process / Bayesian optimization の標準的 textbook
4. upstream implementation が依拠する method paper

robotorchan 固有の API behavior は論文ではなく実装との対応として記述します。

01-13章は初期から段階的に詳細化されたため、14-23章と比べて参考文献 section の形式が
統一されていません。内容上の citation source と実装対応を再監査し、後続 phase で
chapter-level reference section を統一します。

## 章を追加する判断

新しい public class が増えただけでは theory chapter を増やしません。

新章を検討するのは、

- 新しい probabilistic model
- 新しい likelihood / noise model
- 新しい kernel geometry
- 新しい inference method
- 新しい representation assumption
- 新しい candidate-search geometry

のように独立した理論仮定が追加された場合です。

既存理論の Mixed / MultiTask / Kronecker variant は、原則として既存章へ実装上の差分を追記します。

## 現在の再監査状況

理論ドキュメントは実装拡張に合わせて段階的に再監査しています。

特に14-23章では、

- robustness / noise / uncertain input
- nonstationarity
- dimensionality reduction
- neural representation
- high-dimensional search
- SAAS / additive / ALEBO
- expressive GP
- non-GP surrogate

について、数式、実装契約、BoTorch integration、参考文献を再確認しています。

このREADMEは各章のsource of truthではなく、理論体系全体のnavigationと責務境界を示すindexです。
具体的なmodel behaviorは各chapterとcurrent implementationを基準にします。
