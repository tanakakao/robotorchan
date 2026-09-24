# Multi-task / Multi-output

Multi-task と Multi-output は「出力が複数」という見た目だけで選ばず、出力間で何を共有したいかと観測設計から選びます。

## Long-format task model

`MultiTaskGP` と `MixedMultiTaskGP` は task feature を含む long-format 表現でタスク間相関を学習します。タスクごとに異なる入力位置で観測される場合にも適用できます。

## Block-design model

`KroneckerMultiTaskGP` と `MixedKroneckerMultiTaskGP` は、同じ入力点で全タスクを観測する block design に向きます。

```text
train_X: [n, d]
train_Y: [n, m]
```

## Kronecker extensions

Kronecker 系は long-format MultiTask の別名ではありません。block design を前提として、task covariance と入力側の構造を組み合わせます。

現在は `MixedKroneckerMultiTaskGP` に加え、PCA / PLS / Random Projection / neural encoder などの高次元 Kronecker variants と `NonstationaryKroneckerMultiTaskGP` を公開しています。新しい組合せは「MultiTask版があるから」という理由では追加せず、block-designで利用価値があり、posterior / sampling / acquisition contract を検証できる場合に実装します。

Robust Relevance Pursuitなど既存likelihoodとの単純合成が成立しない候補についても直ちに除外せず、専用設計の規模と実務価値を比較して Implement / Prototype / Hold / Reject を判断します。

### Kronecker extension responsibility

すべての公開Kronecker variantで維持する基本契約は

```text
train_X: [..., n, d]
train_Y: [..., n, m]
task identity: train_Y の出力軸
```

です。基本共分散を

\[
K_f = K_{data} \otimes K_{task}
\]

と書くと、各extensionが変更する責務は次のように整理できます。

| extension | 変更する要素 | task factor |
|---|---|---|
| Mixed | `K_data` のcontinuous/categorical構造 | 維持 |
| PCA / PLS / Random Projection | `K_data` へ入る連続表現 | 維持 |
| AE / VAE / joint encoder | `K_data` へ入る学習表現 | 維持 |
| Nonstationary | `K_data` の定常性仮定 | 維持 |
| Spectral Mixture | `K_data` を周波数混合kernelへ変更 | 維持 |
| Infinite-width BNN | `K_data` をReLU NNGP kernelへ変更 | 維持 |

task identityを`train_X`へ埋め込んでlong-formatへ変換することは、Kronecker extensionの
実装方法ではありません。

Heteroskedastic / Replicate Noise / RRP / Student-t / Contaminatedは観測モデル側にも
専用設計が必要なため、現時点ではpublic Kronecker modelではありません。SAAS Kroneckerと
DeepGP KroneckerもHoldであり、公開済みモデルとして扱いません。

## Independent outputs

`ModelListGP` は複数の GP をまとめる構成です。複数目的 BO だから task covariance が必要とは限りません。目的ごとに独立 surrogate を置く場合に自然です。

## Heterogeneous tasks

`HeterogeneousMTGP` と `MixedHeterogeneousMTGP` は、タスクごとに利用可能な特徴量構造が異なる場合を扱います。`feature_indices` により各タスクの特徴を共通 full feature space に対応付けます。

## Fully Bayesian multi-task

`SaasFullyBayesianMultiTaskGP` と `MixedSaasFullyBayesianMultiTaskGP` は、高次元性と task sharing を同時に扱う SAAS 系です。NUTS を使うため通常の exact MLL fitting とは異なります。

## Reduced multi-task

次元削減と multi-task を組み合わせる `ReducedMultiTaskGP` / `ReducedKroneckerMultiTaskGP` 系、および PCA / PLS / Random Projection / AE / VAE / supervised / joint 系の named wrappers は [Reduced models](reduced.md) で整理します。

Notebook: [Multi-task](../../examples/notebooks/04_multitask_gp.ipynb)、[Heterogeneous](../../examples/notebooks/13_heterogeneous_multitask_gp.ipynb)、[Reduced multi-task](../../examples/notebooks/21_reduced_multitask_gp.ipynb)  
Theory: [Multi-task / Multi-output](../theory/07_multitask_multioutput.md)
