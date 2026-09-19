# Dimensionality Reduction Gaussian Process

## 直感

入力次元 D が大きくても目的関数が低次元構造に依存するなら、

```text
x in R^D -> z = r(x) in R^d -> GP(z),  d << D
```

として surrogate の modeling dimension を下げられます。重要なのは「GPの入力を減らすこと」と
「獲得関数を低次元空間で探索すること」は別問題だという点です。

## PCA

PCA は X の分散をよく説明する直交方向を選びます。Yを見ないため安定した baseline ですが、
入力分散が小さくても目的に重要な方向を落とす可能性があります。

robotorchan: `PCAGP`, `PCAMultiTaskGP`, `PCAKroneckerMultiTaskGP`。

## PLS

PLS は X と Y の関係を利用して latent component を学習します。教師ありのため小標本では
component 数と過学習に注意します。

robotorchan: `PLSGP`, `PLSMultiTaskGP`, `PLSKroneckerMultiTaskGP`。

## Random Projection

Random Projection はデータから射影を学習せず、安価な固定 embedding を使います。
複雑な reducer の価値を測る対照モデルとしても有用です。

## Output reduction

高次元 Y に対しては `OutputPCAGP` / `OutputPLSGP` が

```text
Y -> latent Y -> GP -> posterior sample -> original Y
```

を行います。PCA component 自体を本来の objective とみなす必要はなく、posterior を元 Y 空間へ
戻してから objective / constraint を評価します。

## MultiTask

long-format MultiTask では task feature を reducer に入れません。Kronecker 形式では task identity
が Y 列にあるため X 全体を削減できます。Mixed input では continuous feature だけを削減し、
category と task identity を保持します。

## Reducer lifecycle

BO iteration ごとに reducer を再 fit すると latent coordinate system が変わります。そのため
frozen reduction では基底を固定し、更新が必要なら raw training data からモデルを再構築します。

## Search-space reduction との違い

PCA/PLS GP の posterior は original X を受け、内部で reduction します。したがって通常の
`optimize_acqf` は依然 D 次元を探索します。REMBO / ALEBO / BAxUS などは
[High-dimensional search](20_high_dimensional_search.md) の問題です。

実装詳細は [high-dimensional inputs](../models/high_dimensional_inputs.md) と
[high-dimensional outputs](../models/high_dimensional_outputs.md) を参照してください。
