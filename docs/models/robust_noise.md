# Robust / Noise / Nonstationary models

予測誤差が大きいという理由だけで同じ robust model を選ばないでください。外れ値、heavy tail、入力依存 noise、反復測定、latent process の非定常性は異なる生成機構です。

## Sparse outliers

`RobustRelevancePursuitSingleTaskGP` と `MixedRobustRelevancePursuitSingleTaskGP` は、少数の異常観測を relevance pursuit で扱います。long-format の複数タスクには `RobustRelevancePursuitMultiTaskGP`、Mixed入力を含む場合は `MixedRobustRelevancePursuitMultiTaskGP` を使用します。task feature は task covariance が扱い、Mixed版の `cat_dims` には含めません。

## Heavy-tailed residuals

`StudentTSingleTaskGP` と `MixedStudentTSingleTaskGP` は Student-t observation model を使い、大きな residual に Gaussian likelihood より頑健です。

詳細: [Student-t GP](student_t_gp.md)

## Contaminated observations

`ContaminatedSingleTaskGP` と `MixedContaminatedSingleTaskGP` は、通常観測と汚染観測の mixture としてモデル化します。heavy tail を一様に仮定する Student-t と生成仮定が異なります。

詳細: [Contaminated GP](contaminated_gp.md)

## Heteroskedastic noise

`HeteroskedasticSingleTaskGP` / `MixedHeteroskedasticSingleTaskGP` は入力依存 noise を扱います。`JointHeteroskedasticSingleTaskGP` / Mixed版は response と log-noise の潜在過程を joint に扱います。

詳細: [Joint heteroskedastic](joint_heteroskedastic_gp.md)

## Replicate-derived noise

`ReplicateNoiseSingleTaskGP` と Mixed版は同一入力の反復測定を利用して group mean と variance-of-the-mean を構成します。反復測定がない heteroskedastic problem の代替ではありません。

詳細: [Replicate noise](replicate_noise_gp.md)

## Nonstationary latent process

`NonstationarySingleTaskGP` と `MixedNonstationarySingleTaskGP` は入力位置によって latent function の局所 smoothness が変わる場合を扱います。観測 variance が変わる heteroskedastic noise とは区別します。

詳細: [Nonstationary GP](nonstationary_gp.md)

Notebook: [Robust GP](../../examples/notebooks/10_robust_gp.ipynb)、[Robust observation](../../examples/notebooks/15_robust_observation_models.ipynb)、[Noise models](../../examples/notebooks/16_noise_models.ipynb)、[Nonstationary](../../examples/notebooks/18_nonstationary_gp.ipynb)  
Theory: [Robust GP](../theory/14_robust_gaussian_process.md)、[Heteroskedastic noise](../theory/15_heteroskedastic_noise.md)、[Nonstationary GP](../theory/17_nonstationary_gp.md)
