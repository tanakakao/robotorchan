# Robust / Noise / Uncertainty model guide

このドキュメントは robotorchan の robust modeling に関する恒久的な入口です。
「robust」という名前で異なる問題を一括りにせず、通常の GP から**何の仮定が変わるか**で
surrogate と BO-layer の機能を選びます。

モデル選択全体は [models.md](models.md)、個別の数理・API 契約は
[models/](models/) を参照してください。モデル選択の詳細は [models/robust_noise.md](models/robust_noise.md) を参照してください。

## 問題設定から選ぶ

| 問題 | 主なモデル / 層 | 変わるもの |
| --- | --- | --- |
| 少数の gross outlier | `RobustRelevancePursuitSingleTaskGP` | sparse correction / inference |
| residual 全体が heavy-tailed | `StudentTSingleTaskGP` | observation likelihood |
| nominal + gross-error mixture | `ContaminatedSingleTaskGP` | observation likelihood |
| noise variance が入力依存 | `HeteroskedasticSingleTaskGP` | observation-noise process |
| response と noise を joint に学習 | `JointHeteroskedasticSingleTaskGP` | joint variational inference |
| 同一条件の反復測定がある | `ReplicateNoiseSingleTaskGP` | replicate-derived fixed noise |
| latent response の滑らかさが場所で変化 | `NonstationarySingleTaskGP` | covariance |
| 観測された連続 training input が不確か | `UncertainInputSingleTaskGP` | training-input covariance |
| 観測カテゴリが確率的に不確か | `UncertainCategoricalSingleTaskGP` | expected categorical covariance |
| candidate の実現値がずれる | scenario / perturbation layer | candidate evaluation |
| 環境因子 `w` が変動する | scenario layer | candidate evaluation |
| CVaR / worst-case / SN などを最適化 | risk aggregation layer | decision objective |

## Observation robustness

### Sparse outlier

`RobustRelevancePursuitSingleTaskGP` は少数の観測に sparse な補正が必要という仮定です。
残差分布全体が heavy-tailed という Student-t の仮定とは異なります。

### Heavy-tailed residual

`StudentTSingleTaskGP` は Student-t likelihood により大きな残差を Gaussian likelihood より
許容します。詳細は [Student-t GP](models/student_t_gp.md) を参照してください。

### Contamination mixture

`ContaminatedSingleTaskGP` は nominal observation と分散の大きい gross-error observation の
mixture として観測を表現します。詳細は
[Contaminated GP](models/contaminated_gp.md) を参照してください。

## Observation noise

`HeteroskedasticSingleTaskGP` は response GP と入力依存 noise の推定を反復する実用的な
baseline です。`JointHeteroskedasticSingleTaskGP` は response と log-noise の latent process を
joint objective で学習します。

反復測定がある場合は、`ReplicateNoiseSingleTaskGP` が同一 design condition 内の empirical
variance から group mean の observation variance を構成します。

詳細:
[heteroskedastic feasibility](models/heteroskedastic_gp_feasibility.md)、
[joint heteroskedastic GP](models/joint_heteroskedastic_gp.md)、
[replicate-noise GP](models/replicate_noise_gp.md)。

## Input uncertainty

`UncertainInputSingleTaskGP` は観測済み training input の連続座標に Gaussian uncertainty が
ある場合のモデルです。candidate-time perturbation とは異なります。

`UncertainCategoricalSingleTaskGP` は有限カテゴリ上の probability vector を使って
categorical covariance を周辺化します。整数カテゴリコードへ連続 jitter を加えるモデルでは
ありません。

詳細:
[uncertain-input GP](models/uncertain_input_gp.md)、
[full-covariance uncertain-input GP](models/full_covariance_uncertain_input_gp.md)、
[uncertain categorical GP](models/uncertain_categorical_gp.md)。

## Nonstationarity

`NonstationarySingleTaskGP` は latent response の局所 smoothness / lengthscale が入力位置で
変化する場合に使います。入力位置によって**観測ノイズ**が変わる heteroskedastic GP とは
別の仮定です。詳細は [Nonstationary GP](models/nonstationary_gp.md) を参照してください。

## Mixed variables

Mixed 版が必要なのは、categorical design variable が covariance または inference に
実際に影響する場合です。現在の主な対応は次の通りです。

- `MixedRobustRelevancePursuitSingleTaskGP`
- `MixedStudentTSingleTaskGP`
- `MixedContaminatedSingleTaskGP`
- `MixedHeteroskedasticSingleTaskGP`
- `MixedJointHeteroskedasticSingleTaskGP`
- `MixedReplicateNoiseSingleTaskGP`
- `MixedNonstationarySingleTaskGP`
- `MixedUncertainInputSingleTaskGP`

`UncertainCategoricalSingleTaskGP` は categorical uncertainty 自体を直接モデル化するため、
単純な Mixed counterpart は設けません。

Mixed 対応の現在値は [models/robust_noise.md](models/robust_noise.md) と実装を正とします。

## Compositional robust BO

candidate-time uncertainty、environmental scenario、risk aggregation は surrogate class を
増やさず composable な BO-layer として扱います。

```text
surrogate model
    × uncertainty / scenario generator
    × risk aggregation
    -> acquisition
```

物理的な tolerance や process uncertainty は原則として raw design space で scenario を
生成し、その後 PCA / PLS / neural reducer へ渡します。カテゴリ次元や long-format
MultiTask の task feature を暗黙に jitter してはいけません。

SAAS も original-coordinate surrogate なので scenario/risk layer を外側から合成します。
ALEBO は generic reducer ではなく embedded search strategy であるため、robust composition は
strategy geometry を考慮して扱います。

候補側 uncertainty / scenario と surrogate の責務を分離するこの構成を公開契約とします。

## Cross-product model を作らない基準

次のような名前の組合せだけを理由に model class を追加しません。

- `RobustPCAGP`
- `StudentTPCAGP`
- `StudentTHeteroskedasticGP`
- `UncertainInputPLSGP`
- `RobustALEBOGP`

新しい class は likelihood、kernel、posterior、inference のいずれかに新しい統計的仮定が
必要な場合に限ります。Mixed / reduced / MultiTask / robust の全組合せを機械的に作る設計には
しません。

## 関連ガイド

- [Robust / Noise models](models/robust_noise.md)
- [Input uncertainty](models/uncertain_input.md)
- [Robust GP theory](theory/14_robust_gaussian_process.md)
- [Heteroskedastic noise theory](theory/15_heteroskedastic_noise.md)
- [Uncertain-input GP theory](theory/16_uncertain_input_gp.md)
- [Nonstationary GP theory](theory/17_nonstationary_gp.md)

開発フェーズ番号や closeout 記録ではなく、現在の実装と上記の恒久ドキュメントを仕様の正とします。
