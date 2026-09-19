# Uncertain-input Gaussian Process

## 直感

通常の GP は入力 `x` が正確に観測されていると仮定します。しかし配合量、温度、位置、
加工条件などには設定値と実現値のずれがあり得ます。uncertain-input GP は output noise ではなく
**入力座標そのものの不確かさ**を扱います。

## 連続入力

観測入力を

```text
x_obs = x_true + delta
delta ~ Normal(0, Sigma_x)
```

と考えると、kernel は一点同士の距離ではなく入力分布間で平均化された covariance を考える
必要があります。

`UncertainInputSingleTaskGP` はこの training-input uncertainty を covariance に反映します。
full covariance が必要な場合も、入力次元間の相関を無視しない形で扱います。

## Categorical uncertainty

カテゴリ `c` に対して「コード 0.1 のずれ」のような連続 perturbation は意味を持ちません。
代わりに

```text
P(c = category_k) = p_k
```

という有限カテゴリ上の確率として表現し、categorical covariance を期待値として周辺化します。

robotorchan: `UncertainCategoricalSingleTaskGP`。

## Mixed input

`MixedUncertainInputSingleTaskGP` では continuous uncertainty と nominal categorical feature を
分離します。continuous dimensions だけを Gaussian uncertainty の対象にし、category identity は
categorical kernel 側で保持します。

## Training-input uncertainty と candidate perturbation

両者は区別します。

- training-input uncertainty: 過去に観測した X 自体が不確か
- candidate perturbation: 提案した X が実験時にずれる
- environmental scenario: 制御不能な w が変動する

後二者は surrogate class の直積ではなく scenario / risk layer として合成するのが基本です。

## BO との関係

入力不確かさを posterior に反映するだけでなく、実際の目的が expected performance、CVaR、
worst-case performance なら acquisition 側にも scenario aggregation が必要です。

実装仕様は [uncertain-input GP](../models/uncertain_input_gp.md)、
[uncertain categorical GP](../models/uncertain_categorical_gp.md) を参照してください。
