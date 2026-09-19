# Reduced GP family

Reduced GP は高次元入力・出力を低次元表現へ写像して GP を構築するファミリーです。reducer の再学習、逆変換、Mixed入力、Multi-taskとの組合せはモデルごとに契約が異なります。

## Classical reduction

基本クラスは `ReducedGP` です。代表的な named wrapper として `PCAGP`、`PLSGP`、`RandomProjectionGP`、出力側の `OutputPCAGP`、`OutputPLSGP` があります。

Mixed入力には `MixedReducedGP`、`MixedPCAGP`、`MixedPLSGP`、`MixedRandomProjectionGP` を使用します。カテゴリ列を連続 reducer に混ぜないことが重要です。

## Neural reduction

非線形表現には `AutoEncoderGP`、`VAEGP`、`SupervisedAutoEncoderGP`、`SupervisedVAEGP`、`JointEncoderGP`、`HybridAutoEncoderGP`、`JointVAEGP` があります。

Mixed版として `MixedAutoEncoderGP`、`MixedVAEGP`、`MixedSupervisedAutoEncoderGP`、`MixedSupervisedVAEGP`、`MixedJointEncoderGP`、`MixedHybridAutoEncoderGP`、`MixedJointVAEGP` を提供します。

supervised / joint 系は、単に事前学習済み encoder を固定して使うモデルと同じではありません。representation と GP objective の学習関係を確認してください。

## Multi-task reduction

共通基盤として `ReducedMultiTaskGP` と `ReducedKroneckerMultiTaskGP`、Mixed版として `MixedReducedMultiTaskGP` と `MixedReducedKroneckerMultiTaskGP` があります。

named wrappers は PCA / PLS / Random Projection / AutoEncoder / VAE / SupervisedAutoEncoder / SupervisedVAE / JointEncoder / HybridAutoEncoder / JointVAE と、通常 MultiTask / Kronecker MultiTask の組合せを提供します。モデル名は reducer と task covariance の両方を表します。

## 選択上の注意

次元削減は情報を捨てる操作です。入力分散をよく説明する方向が目的関数に重要とは限らないため、PCAだけでなく supervised reduction や PLS が妥当な場合があります。Neural reducer は柔軟ですが、小標本では representation learning 自体の不確かさと過学習を考慮する必要があります。

Notebook: [Classical reduction](../../examples/notebooks/19_reduced_gp.ipynb)、[Neural reduction](../../examples/notebooks/20_neural_reduction_gp.ipynb)、[Reduced multi-task](../../examples/notebooks/21_reduced_multitask_gp.ipynb)、[Mixed reduced](../../examples/notebooks/22_mixed_reduced_gp.ipynb)  
Theory: [Dimensionality reduction](../theory/18_dimensionality_reduction_gp.md)、[Neural reduction](../theory/19_neural_reduction_gp.md)
