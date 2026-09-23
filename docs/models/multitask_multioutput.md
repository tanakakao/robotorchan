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
