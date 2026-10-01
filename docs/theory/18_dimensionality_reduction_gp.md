# Dimensionality Reduction Gaussian Process

## 直感

入力次元 `D` が大きくても、目的関数が実質的に低次元構造へ依存するなら、

```text
x in R^D
  -> z = r_x(x) in R^d
  -> GP(z),  d << D
```

として surrogate の modeling dimension を下げられます。

また output dimension `M` が大きい場合には、

```text
Y in R^M
  -> h = r_y(Y) in R^m
  -> GP(h)
  -> reconstructed posterior in R^M
```

という output reduction も可能です。

重要なのは、**surrogate modeling dimension の削減**と
**acquisition optimization の search-space dimension の削減**は別問題だという点です。

## 1. なぜ reduction を使うか

高次元 GP では、有限データから各方向の lengthscale や相互作用を識別することが難しくなります。
reduction は

- effective dimension を下げる
- irrelevant / redundant direction を圧縮する
- GP covariance estimation を簡単にする
- high-dimensional output の posterior を扱いやすくする

ために使えます。

一方、reducer が目的に重要な方向を捨てれば、その情報は downstream GP では復元できません。
したがって reduction 自体も modeling assumption です。

## 2. PCA input reduction

PCA は centered input matrix の分散を最大化する直交方向を求めます。

```text
X_centered = X - mean(X)
z = X_centered W_PCA
```

上位 `d` component を残すことで `D -> d` に削減します。

PCA は `Y` を使わない unsupervised reduction なので、安定した baseline になりやすい一方、
入力分散が小さくても objective に重要な方向を落とす可能性があります。

robotorchan:

- `PCAGP`
- `PCAMultiTaskGP`
- `PCAKroneckerMultiTaskGP`

## 3. PLS input reduction

PLS は input variation だけでなく outcome との関係を利用して latent component を構成します。

概念的には

```text
z = X W_PLS
```

ですが、`W_PLS` は `X` と `Y` の共分散構造を使って学習されます。

PCA より objective-aware ですが、同じ training outcomes を reducer と GP の両方に使うため、
小標本では component 数や validation に注意が必要です。

robotorchan:

- `PLSGP`
- `PLSMultiTaskGP`
- `PLSKroneckerMultiTaskGP`

Long-format MultiTask では task identity を reducer へ入れず、各 row の outcome を使って
data feature の supervised reduction を学習します。

Kronecker MultiTask では shared `X` と multi-output `Y` を使い、全 task outcome と関連する
input direction を学習します。

## 4. Random Projection

Random Projection はデータから projection matrix を学習せず、固定されたランダム写像

```text
z = X R
```

を使います。

Johnson-Lindenstrauss 型の考え方では、十分な latent dimension があれば有限点集合の距離を
近似的に保存できます。

robotorchan:

- `RandomProjectionGP`
- `RandomProjectionMultiTaskGP`
- `RandomProjectionKroneckerMultiTaskGP`

PCA / PLS と異なり projection fit が outcome に依存せず、再現性は `random_state` で管理します。
複雑な learned reducer が本当に必要かを比較する baseline としても有用です。

## 5. Frozen reducer lifecycle

robotorchan の reduced GP では reducer lifecycle が重要です。

未fit reducer を渡した場合、model construction 時に一度 fit します。既にfit済みの reducer を
渡した場合は、その latent coordinate system を保持して `transform` だけを適用します。

```text
raw X
  -> frozen reducer
  -> reduced X
  -> GP
```

BO iteration ごとに reducer を無条件で再fitすると latent coordinates が変わり、既存 GP state
や candidate geometry の意味も変わります。

そのため reducer を更新したい場合は、raw training data を source of truth としてモデル全体を
再構築するのが基本です。

## 6. Original-space API と internal reduced space

`ReducedGP` 系では public API は original input space を維持します。

training 時には

```text
train_X_original -> reducer -> train_X_reduced
```

posterior 時には

```text
X_original -> reducer.transform -> GP.posterior
```

となります。

実装によっては既に reduced dimension の input も受け付けますが、BO workflow では raw /
original-space contract を維持する方が search-space constraint や feature semantics を
失いにくくなります。

`condition_on_observations` でも同じ reducer を使い、追加観測を既存 latent coordinate system
へ写像します。

## 7. Input transform との順序

reduction と BoTorch `input_transform` は同じものではありません。

robotorchan の reduced GP では概念的に

```text
raw X
  -> reduction
  -> BoTorch input_transform
  -> GP covariance
```

という順序です。

例えば raw-space Normalize と reduced-space Normalize では意味が異なります。
reducer と `input_transform` を併用するときは、どの空間に transform が作用するかを意識する
必要があります。

## 8. Output reduction

高次元 outcome では output-side reducer を使えます。

robotorchan の `OutputPCAGP` / `OutputPLSGP` は training outcomes を latent output space へ
変換し、GP posterior を original output space へ復元します。

```text
Y_original
  -> output reducer
  -> Y_latent
  -> GP posterior
  -> restored posterior
  -> Y_original space
```

重要なのは、PCA / PLS component 自体を本来の objective とみなす必要がないことです。
public posterior を元の `Y` 空間へ戻した後で objective / constraint を評価できます。

これは multi-objective BO や structured outcome で特に重要です。

## 9. Input + output reduction

`ReducedGP` の plumbing は input reducer と output reducer を独立に扱えるため、

```text
X_original -> X_latent -> GP <- Y_latent <- Y_original
```

という構成も可能です。

ただし input information loss と output reconstruction error の両方が surrogate error に
寄与するため、片側だけの reduction より診断項目が増えます。

## 10. MultiTask と task identity

Long-format MultiTask では input row に task feature が含まれます。

```text
[x_1, ..., x_D, task]
```

task identity を PCA / PLS の連続 feature として圧縮すると task semantics が壊れるため、
`ReducedMultiTaskGP` は task feature を reducer から除外します。

```text
data X -> reducer -> z
task -> preserved

final GP input = [z, task]
```

reduced space では task feature の列位置が変わるため、model は original task feature と
reduced task feature の両方を管理します。

Kronecker MultiTask では task identity は `Y` の列構造にあるため、shared design `X`
全体を input reducer に渡せます。

## 11. Mixed input

mixed-variable reduction では categorical code をそのまま PCA / PLS の連続量として扱わない
ことが重要です。

robotorchan の mixed reduced model は continuous features を reduction し、categorical features
を保持して GP 側の mixed covariance と統合する設計を取ります。

概念的には

```text
continuous X -> reducer -> z_cont
categorical X -----------> c

GP input = [z_cont, c]
```

です。

この責務分離により、カテゴリ番号の大小や間隔を Euclidean geometry と誤解することを避けます。

## 12. Neural reduction との境界

AutoEncoder、VAE、Supervised AutoEncoder、Supervised VAE も同じ reduced-GP family に
属しますが、線形 PCA / PLS / Random Projection とは学習目的と lifecycle が複雑になります。

これらの理論は
[Neural Reduction GP](19_neural_reduction_gp.md) で詳しく扱います。

本章では reduction framework、linear reducer、MultiTask / Mixed / output reduction の共通原理を
中心にします。

## 13. Search-space reduction との違い

PCA / PLS GP の posterior が internal reduced space を使っていても、通常の
`optimize_acqf` が自動的に低次元探索になるわけではありません。

public acquisition が original `X` を受けるなら、

```text
optimize over x in R^D
  -> acquisition(x)
  -> model reduces x internally
```

となり、optimizer 自体は依然 `D` 次元を探索します。

REMBO / ALEBO / BAxUS のように optimizer が低次元 search coordinates を直接探索する方式は
別の問題です。

詳細は [High-dimensional search](20_high_dimensional_search.md) を参照してください。

## 14. Bayesian optimization での使い分け

### PCA

- unsupervised baseline が欲しい
- outcome leakage を避けたい
- input covariance structure が意味を持つ

### PLS

- outcome と関連する低次元方向を抽出したい
- supervised reduction の追加自由度を許容できる

### Random Projection

- 非常に安価な fixed embedding が欲しい
- learned reduction と比較する baseline が欲しい
- projection fit の不安定性を避けたい

reduction の選択は GP fit だけでなく、candidate ranking が original-space objective に対して
安定しているかで評価する必要があります。

## 15. 実装との対応

現在の linear reduction family で重要な契約は次です。

- public training / posterior API は original-space data を基本とする
- reducer は model construction 時に fit され、その後 frozen
- pre-fitted reducer は再fitせず再利用可能
- posterior / condition-on-observations でも同じ reducer を使う
- Long-format MultiTask は task feature を reduction しない
- Kronecker MultiTask は shared design `X` を reduction する
- Mixed input は continuous reduction と categorical identity を分離する
- output reducer は posterior を original outcome space へ復元する
- model-space reduction と optimizer-space reduction は別機能

実装詳細は [high-dimensional inputs](../models/high_dimensional_inputs.md) と
[high-dimensional outputs](../models/high_dimensional_outputs.md) を参照してください。

## 参考文献

1. Jolliffe, I. T. and Cadima, J. (2016).
   Principal component analysis: A review and recent developments.
   *Philosophical Transactions of the Royal Society A*, 374.
2. Wold, S., Sjöström, M., and Eriksson, L. (2001).
   PLS-regression: A basic tool of chemometrics.
   *Chemometrics and Intelligent Laboratory Systems*, 58(2).
3. Johnson, W. B. and Lindenstrauss, J. (1984).
   Extensions of Lipschitz mappings into a Hilbert space.
   *Contemporary Mathematics*, 26.
4. Bingham, E. and Mannila, H. (2001).
   Random projection in dimensionality reduction: Applications to image and
   text data.
   *Proceedings of the Seventh ACM SIGKDD International Conference on
   Knowledge Discovery and Data Mining*.
5. Tripathy, R., Bilionis, I., and Gonzalez, M. (2016).
   Gaussian processes with built-in dimensionality reduction:
   Applications to high-dimensional uncertainty propagation.
   *Journal of Computational Physics*, 321.
6. Rasmussen, C. E. and Williams, C. K. I. (2006).
   *Gaussian Processes for Machine Learning*. MIT Press.

## 関連章

- [High-dimensional GP](08_high_dimensional_gp.md)
- [Neural Reduction GP](19_neural_reduction_gp.md)
- [High-dimensional Search](20_high_dimensional_search.md)
- [MultiTask / Multi-output](07_multitask_multioutput.md)
