# Heteroskedastic / Replicate Noise

## 直感

標準 GP はしばしば全入力で同じ観測ノイズ分散を仮定します。しかし実験条件によって測定精度が
変わる場合や、同一条件を反復測定できる場合には、その情報を明示的に使えます。

## Homoskedastic と heteroskedastic

homoskedastic model は

```text
y_i = f(x_i) + epsilon_i
epsilon_i ~ Normal(0, sigma^2)
```

です。heteroskedastic model では

```text
epsilon_i ~ Normal(0, sigma^2(x_i))
```

となり、noise variance が入力依存になります。

## Heteroskedastic GP

`HeteroskedasticSingleTaskGP` は response GP と入力依存 noise estimate を扱う実用的な
構成です。`JointHeteroskedasticSingleTaskGP` は response と log-noise の latent process を
joint objective で推論します。後者は coupling を直接扱える一方、推論も複雑になります。

Mixed counterpart では categorical covariance を保持し、continuous と categorical の構造を
壊さず noise process を扱います。

## Replicate noise

同じ design condition `x` を複数回測定できるなら、反復群

```text
y_(x,1), ..., y_(x,r)
```

から sample variance を推定できます。group mean を学習対象にするとき、その平均値の
observation variance は replicate 数も反映します。

`ReplicateNoiseSingleTaskGP` はこの情報を fixed-noise GP に接続します。
`MixedReplicateNoiseSingleTaskGP` はカテゴリを含む design condition の同一性を保持します。

## Nonstationarity との違い

heteroskedasticity は「観測のばらつき」が場所で変わる問題です。latent function 自体の
smoothness が場所で変わる問題は [Nonstationary GP](17_nonstationary_gp.md) です。

## BO との関係

高ノイズ領域では epistemic uncertainty と observation noise を混同しないことが重要です。
replicate を追加するか、新しい design point を探索するかという実験設計にも影響します。

対応モデルの仕様は [Robust / Noise model guide](../models/robust_noise.md) と
[Joint heteroskedastic GP](../models/joint_heteroskedastic_gp.md) を参照してください。
