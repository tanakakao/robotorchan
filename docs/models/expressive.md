# Expressive GP family

表現力を高めたsurrogateとして、robotorchanではDKL、DeepGP、Infinite-width BNN GP、
Spectral Mixture GPを扱います。これらは同じ問題を別名で実装したものではなく、
「何を学習可能な表現とするか」「posteriorをどう構成するか」が異なります。

## モデル選択

`JointEncoderGP` はneural feature extractorとExact GPをjoint trainingするDKLです。
入力に有用な非線形表現が存在すると考える場合に候補になります。Mixed入力には
`MixedJointEncoderGP` を使用し、カテゴリ変数を連続encoderへそのまま通しません。

複数タスクでは、long-formatの `JointEncoderMultiTaskGP` とblock-designの
`JointEncoderKroneckerMultiTaskGP` が既にpublic APIです。前者はtask featureをencoderへ
入れず、data featureだけを潜在表現へ写像してtask identityを再結合します。後者はtask identityが
`train_Y` の出力列にあるため、`train_X` 全体をencoderへ通します。どちらもencoderは
GPの学習目的からjoint trainingされ、`posterior()` には元入力空間のXを渡します。

一方、`MixedReducedMultiTaskGP` はfitted reducerを使うreduction基盤であり、
joint-training DKLのMixed MultiTask実装ではありません。Mixed × MultiTask DKLには `MixedJointEncoderMultiTaskGP` を使用します。continuous data featureだけを
encoderで潜在表現へ写像し、categorical featureとtask identityは変換せずGPへ渡します。

`SingleTaskDeepGP` は確率的なGP階層を使います。DKLより推論が重く、
Monte Carlo posteriorを使うため、階層的な確率表現が必要な場合に比較対象とします。
`MultiTaskDeepGP` はlong-formatのtask featureを明示的に受け取り、data featureを標準化しつつtask identityを学習可能なembeddingとしてDeepGPへ結合します。`posterior()` と `training_loss()` には元のlong-format入力を渡します。Mixed single-task入力には `MixedSingleTaskDeepGP` を使用します。連続featureだけを標準化し、categorical featureは学習可能なembeddingへ変換してDeepGPへ結合します。カテゴリ値を連続量として補間しないことを明示的な契約としています。Mixed × MultiTask には `MixedMultiTaskDeepGP` を使用します。連続 feature とカテゴリ feature を分離し、task identity を構造列として扱います。

`InfiniteWidthBNNGP` は無限幅ReLU networkに対応するNNGP kernelをExact GPとして
利用します。neural-network由来のpriorを使いつつ、Exact GPの学習・posterior contractを
維持したい場合に候補になります。long-format複数タスクには `InfiniteWidthBNNMultiTaskGP` を使用し、task featureをNNGP data kernelから除外してtask covarianceで扱います。Mixed single-task入力には `MixedInfiniteWidthBNNGP`、Mixed × MultiTask には `MixedInfiniteWidthBNNMultiTaskGP` を使用します。

`SpectralMixtureGP` は周期、準周期、複数周波数を持つ定常関数に適しています。
`num_mixtures` と初期化への感度があるため、標準kernelとの比較を推奨します。long-format複数タスクには `SpectralMixtureMultiTaskGP`、Mixed single-task入力には `MixedSpectralMixtureGP`、Mixed × MultiTask には `MixedSpectralMixtureMultiTaskGP` を使用します。

## 学習とBoTorch

DKL、Infinite-width BNN GP、Spectral Mixture GPはExact MLLを使います。
DeepGPはvariational objectiveを用いるため学習方法が異なります。

いずれも候補点は元入力空間で与えます。対応するMC acquisitionと
`optimize_acqf` の統合範囲は [BoTorch統合契約](expressive_botorch_integration.md) を
参照してください。

## 関連資料

Theory: [Expressive GP models](../theory/22_expressive_gp.md)  
Notebook: [Expressive surrogate GP](../../examples/notebooks/24_expressive_surrogate_gp.ipynb)  
Benchmark: [Predictive benchmark](../benchmarks/expressive_predictive.md)


### SpectralMixtureMultiTaskGP

`SpectralMixtureMultiTaskGP` はlong-format複数タスク向けです。task featureはspectral-mixture data kernelから除外し、BoTorchのtask covarianceでタスク間相関を扱います。周期・準周期・複数周波数構造をタスク間で共有しつつ、元のlong-format入力空間でposterior/acquisitionを利用できます。


### Mixed expressive kernels

`MixedInfiniteWidthBNNGP` は連続特徴にinfinite-width ReLU NNGP kernel、カテゴリ特徴にrobotorchan共通categorical kernelを使い、加法項とinteraction項を組み合わせます。

`MixedSpectralMixtureGP` は連続特徴にspectral-mixture kernel、カテゴリ特徴に共通categorical kernelを使います。spectral parametersの初期化には連続特徴だけを使用し、カテゴリIDを連続値として補間しません。
