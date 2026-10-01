# Nonstationary Gaussian Process

## 直感

stationary kernel は、入力空間のどこにいても同じ距離・方向の関係を同じ covariance rule で
評価します。材料・製造プロセスでは、安定領域では滑らかでも、相境界や遷移領域では目的関数が
急激に変化することがあります。

nonstationary GP は latent function の局所的な smoothness を入力位置に依存させ、このような
空間的に変化する関数構造を表現します。

robotorchan では Gibbs covariance による input-dependent lengthscale を採用しています。

## 1. Stationarity

stationary kernel は一般に

```text
k(x, x') = k(x - x')
```

と書けます。RBF や stationary Matérn kernel では、lengthscale `ell` は入力空間全体で
共有されます。

同じ `ell` で全領域を説明しようとすると、

- smooth region に合わせれば急変領域を過度に平滑化する
- rapid region に合わせれば smooth region で過度に短い相関を仮定する

という妥協が起こり得ます。

## 2. Input-dependent lengthscale

nonstationary covariance では

```text
ell = ell(x)
```

として局所 lengthscale を入力位置に依存させます。

robotorchan の `GibbsKernel` は各入力次元について正の local lengthscale を

```text
ell(x) = softplus(a + B x) + ell_floor
```

として表現します。

`a` は `lengthscale_intercept`、`B` は `lengthscale_slope` に対応します。
`softplus` と正の `lengthscale_floor` により lengthscale の正値性を保ちます。

これは任意の neural lengthscale field ではなく、**affine map を softplus で正値化した
比較的制約された nonstationary parameterization** です。

## 3. Gibbs covariance

1次元の Gibbs covariance の基本形は

```text
k(x, x')
  = sqrt(
      2 ell(x) ell(x')
      / (ell(x)^2 + ell(x')^2)
    )
    * exp(
      -(x - x')^2
      / (ell(x)^2 + ell(x')^2)
    )
```

です。

多次元では各次元の prefactor を積で組み合わせ、距離項を各次元について加算します。
robotorchan の実装もこの構造を使います。

局所 lengthscale が一定なら stationary RBF 型 covariance に近い構造へ戻ります。
一方、`ell(x)` が場所によって変われば、同じ Euclidean distance でも場所に応じて
covariance が変わります。

## 4. Single-task implementation

`NonstationarySingleTaskGP` は exact `SingleTaskGP` の covariance を

```text
ScaleKernel(GibbsKernel)
```

へ置き換えます。

したがって observation model や exact-GP training contract を独自に作り直すのではなく、
主な変更点は data covariance です。

`local_lengthscale(X)` により、学習された局所 lengthscale を入力点ごとに取得できます。
これは nonstationarity を診断する上でも重要です。

## 5. Mixed variables

`MixedNonstationarySingleTaskGP` では nonstationarity を continuous dimensions にだけ
適用します。

categorical feature を連続値として Gibbs lengthscale field に入力するのではなく、
robotorchan の mixed covariance の中で

- continuous branch: Gibbs covariance
- categorical branch: categorical covariance
- interaction branch: continuous × categorical

を構成します。

実装には continuous covariance の additive branch と interaction branch があるため、
`local_lengthscale(X)` は2組の local lengthscale を返します。

これは「カテゴリごとに独立した GP」を意味するのではなく、mixed covariance の中で
continuous nonstationarity と categorical similarity を共存させる設計です。

## 6. MultiTask

現在の実装は single-task だけでなく MultiTask にも対応します。

### Long-format MultiTask

`NonstationaryMultiTaskGP` では task feature を data feature から分離し、Gibbs covariance は
data covariance branch に適用します。

```text
K((x, t), (x', t'))
  = K_data_nonstationary(x, x') * K_task(t, t')
```

という MultiTask GP の構造を保ちます。

task identity は local smoothness を計算する通常の連続 feature として扱いません。

### Kronecker MultiTask

`NonstationaryKroneckerMultiTaskGP` は block-design の Kronecker MultiTask GP で
data covariance を Gibbs kernel に置き換えます。

shared design matrix `X` 上で nonstationary data covariance を学び、task covariance は
Kronecker MultiTask の構造として保持します。

### Mixed MultiTask

`MixedNonstationaryMultiTaskGP` は

- continuous data dimensions
- categorical data dimensions
- task feature

を分離します。

continuous data dimensions に Gibbs covariance、categorical data dimensions に mixed
categorical covariance、task feature に task covariance を割り当てます。

## 7. Heteroskedasticity との違い

nonstationarity と heteroskedasticity は異なる統計構造です。

```text
nonstationary:
  k = k(x, x')
  ell = ell(x)

heteroskedastic:
  y = f(x) + epsilon
  Var[epsilon | x] = sigma^2(x)
```

前者は latent function の covariance / smoothness、後者は observation noise variance の
変化です。

急変領域でデータが散らばると見た目だけでは区別しにくいため、noise model と covariance
model のどちらで説明しているかを明確にする必要があります。

詳細は [Heteroskedastic / Replicate Noise](15_heteroskedastic_noise.md) を参照してください。

## 8. Input warping との違い

nonstationarity を表現する別の方法として、入力を非線形変換して stationary kernel を適用する
input warping があります。

```text
x -> w(x) -> stationary kernel
```

Gibbs kernel は座標そのものを変換するのではなく、local metric / lengthscale を入力依存に
します。

どちらも effective geometry を変えますが、parameterization と解釈が異なります。

## 9. Bayesian optimization との関係

stationary GP が急変領域を過度に平滑化すると、

- posterior mean が境界を平均化する
- posterior variance が局所構造を適切に表さない
- acquisition が promising region を見落とす

可能性があります。

nonstationary surrogate はこれを改善し得ますが、自由度も増えます。特に小標本 BO では、
nonstationarity を識別する情報が不足すると lengthscale field 自体が不安定になり得ます。

そのため `local_lengthscale(X)` の診断、複数初期値での fit、stationary baseline との比較が
有用です。

## 10. 現在の parameterization の範囲

robotorchan の Gibbs kernel は実用上意図的に限定された構造です。

- local lengthscale は feature ごとに存在
- lengthscale field は affine + softplus
- `lengthscale_floor > 0`
- arbitrary deep network による metric field ではない
- exact GP の covariance module として利用
- output noise の nonstationarity は別モデル

したがって「一般的な nonstationary GP の全方式」を実装しているわけではありません。
現在の理論章は、実装されている Gibbs-kernel family を中心に説明します。

## 11. 実装との対応

対応クラスは次です。

- `GibbsKernel`
- `NonstationarySingleTaskGP`
- `MixedNonstationarySingleTaskGP`
- `NonstationaryMultiTaskGP`
- `MixedNonstationaryMultiTaskGP`
- `NonstationaryKroneckerMultiTaskGP`

重要な契約は次です。

- `lengthscale_floor` は正
- local lengthscale は `softplus(a + Bx) + floor`
- SingleTask / Kronecker は1組の local lengthscale を返す
- Mixed は additive / interaction continuous branch の2組を返す
- task feature は data smoothness と分離する
- categorical feature は continuous Gibbs geometry に混ぜない
- heteroskedastic noise は別の modeling axis

詳細は [Nonstationary GP](../models/nonstationary_gp.md) を参照してください。

## 参考文献

1. Gibbs, M. N. (1997).
   *Bayesian Gaussian Processes for Regression and Classification*.
   PhD thesis, University of Cambridge.
2. Paciorek, C. J. and Schervish, M. J. (2004).
   Nonstationary covariance functions for Gaussian process regression.
   *Advances in Neural Information Processing Systems 16*.
3. Heinonen, M., Mannerström, H., Rousu, J., Kaski, S., and Lähdesmäki, H. (2016).
   Non-stationary Gaussian process regression with Hamiltonian Monte Carlo.
   *Proceedings of the 19th International Conference on Artificial Intelligence
   and Statistics*.
4. Rasmussen, C. E. and Williams, C. K. I. (2006).
   *Gaussian Processes for Machine Learning*. MIT Press.

## 関連章

- [Gaussian Process](02_gaussian_process.md)
- [Kernel](03_kernel.md)
- [Heteroskedastic / Replicate Noise](15_heteroskedastic_noise.md)
- [High-dimensional GP](08_high_dimensional_gp.md)
