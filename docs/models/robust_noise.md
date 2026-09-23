# Robust / Noise / Nonstationary models

予測誤差が大きいという理由だけで同じ robust model を選ばないでください。外れ値、heavy tail、入力依存 noise、反復測定、latent process の非定常性は異なる生成機構です。

## Sparse outliers

`RobustRelevancePursuitSingleTaskGP` と `MixedRobustRelevancePursuitSingleTaskGP` は、少数の異常観測を relevance pursuit で扱います。long-format の複数タスクには `RobustRelevancePursuitMultiTaskGP`、Mixed入力を含む場合は `MixedRobustRelevancePursuitMultiTaskGP` を使用します。task feature は task covariance が扱い、Mixed版の `cat_dims` には含めません。

## Heavy-tailed residuals

`StudentTSingleTaskGP` と `MixedStudentTSingleTaskGP` は Student-t observation model を使い、大きな residual に Gaussian likelihood より頑健です。long-format 複数タスクには `StudentTMultiTaskGP`、Mixed入力には `MixedStudentTMultiTaskGP` を使用します。複数タスク版は data covariance と task covariance の積による ICM-style covariance を variational GP に適用します。

詳細: [Student-t GP](student_t_gp.md)

## Contaminated observations

`ContaminatedSingleTaskGP` と `MixedContaminatedSingleTaskGP` は、通常観測と汚染観測の mixture としてモデル化します。複数タスクには `ContaminatedMultiTaskGP`、Mixed入力には `MixedContaminatedMultiTaskGP` を使用し、task covariance を保持したまま contamination mixture loss を適用します。heavy tail を一様に仮定する Student-t と生成仮定が異なります。

詳細: [Contaminated GP](contaminated_gp.md)

## Heteroskedastic noise

`HeteroskedasticSingleTaskGP` / `MixedHeteroskedasticSingleTaskGP` は入力依存 noise を扱います。long-format の複数タスクには `HeteroskedasticMultiTaskGP`、Mixed入力を含む場合は `MixedHeteroskedasticMultiTaskGP` を使用します。task feature は noise GP を含む両方の MultiTaskGP で task covariance として扱い、`cat_dims` には含めません。`JointHeteroskedasticSingleTaskGP` / Mixed版は response と log-noise の潜在過程を joint に扱います。

詳細: [Joint heteroskedastic](joint_heteroskedastic_gp.md)

## Replicate-derived noise

`ReplicateNoiseSingleTaskGP` と Mixed版は同一入力の反復測定を利用して group mean と variance-of-the-mean を構成します。反復測定がない heteroskedastic problem の代替ではありません。

詳細: [Replicate noise](replicate_noise_gp.md)

## Nonstationary latent process

`NonstationarySingleTaskGP` / `MixedNonstationarySingleTaskGP`、および long-format 複数タスク向け `NonstationaryMultiTaskGP` / `MixedNonstationaryMultiTaskGP` は入力位置によって latent function の局所 smoothness が変わる場合を扱います。観測 variance が変わる heteroskedastic noise とは区別します。

詳細: [Nonstationary GP](nonstationary_gp.md)

### Kronecker / Multi-Fidelity robust variants

`NonstationaryKroneckerMultiTaskGP` は、同じ入力点で全タスクを観測する block-design Kronecker 構造と nonstationary latent process を組み合わせます。long-format の `NonstationaryMultiTaskGP` とは観測設計が異なります。

Multi-Fidelity では、`ReplicateNoiseMultiFidelityGP` と `HeteroskedasticMultiFidelityGP` が public model です。前者は反復測定から variance-of-the-mean を構成し、後者は fidelity coordinate を保持する noise process を学習します。両者とも代表的な native Multi-Fidelity KG workflow を検証しています。

Relevance Pursuit × Kronecker / Multi-Fidelity は、既存 single-task likelihood mixin をそのまま再利用できないため専用設計が必要です。実装不能という意味ではなく、専用 likelihood / outlier structure の設計価値と保守コストを評価してから prototype する候補です。

Notebook: [Robust GP](../../examples/notebooks/10_robust_gp.ipynb)、[Robust observation](../../examples/notebooks/15_robust_observation_models.ipynb)、[Noise models](../../examples/notebooks/16_noise_models.ipynb)、[Nonstationary](../../examples/notebooks/18_nonstationary_gp.ipynb)  
Theory: [Robust GP](../theory/14_robust_gaussian_process.md)、[Heteroskedastic noise](../theory/15_heteroskedastic_noise.md)、[Nonstationary GP](../theory/17_nonstationary_gp.md)


## Current Kronecker heteroskedastic boundary

`HeteroskedasticKroneckerMultiTaskGP` and its Mixed prototype are intentionally not public models.
The current prototypes can learn an auxiliary task-specific log-noise process, but that predicted
noise is not yet coupled back into the response observation likelihood. They therefore must not be
used as if the response posterior already represented input-dependent observation noise.

The public block-design robust path is currently `NonstationaryKroneckerMultiTaskGP` for latent
nonstationarity. Heteroskedastic Kronecker will return to the public API only after a shaped
task-specific observation-noise path is implemented and runtime-validated.
