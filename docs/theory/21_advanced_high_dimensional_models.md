# Advanced High-dimensional GP Models

## 直感

高次元問題への対応は projection だけではありません。「元特徴のごく一部だけが効く」
「関数が加法的」「低次元線形 subspace に依存する」など、構造仮定を prior / kernel に
組み込む方法があります。

## ARD と sparsity

ARD kernel は次元ごとの lengthscale `ell_j` を持ちます。大きな lengthscale の次元は関数変化へ
与える影響が小さいと解釈できます。ただし通常の ARD だけでは高次元小標本で識別が難しくなります。

## SAAS

SAAS は inverse lengthscale に強い shrinkage prior を置き、多くの次元をほぼ無効化しつつ
少数の active dimension を残す考え方です。

robotorchan:
`SaasFullyBayesianSingleTaskGP`, `SaasFullyBayesianMultiTaskGP` と Mixed counterparts。

SAAS は PCA のように latent coordinate へ射影しません。「元特徴の sparsity」を仮定します。

## MAP-SAAS

fully Bayesian inference は MCMC cost が大きいため、MAP approximation を使う選択肢があります。
`AdditiveMapSaasSingleTaskGP` / `EnsembleMapSaasSingleTaskGP` は高次元での計算量と
sparsity modeling のトレードオフを狙います。

## Additive structure

加法 GP は概念的に

```text
f(x) = f_1(x_S1) + ... + f_K(x_SK)
```

として低次元 component の和で高次元関数を表します。`OrthogonalAdditiveGP` はこの系統です。
active dimension sparsity と additive decomposition は異なる仮定です。

## Relevance pursuit

Relevance pursuit は sparse な relevance / correction 構造を選択し、高次元または外れ値を含む問題で不要な自由度を抑える考え方です。robotorchan の `RobustRelevancePursuitSingleTaskGP` は observation robustness の文脈で利用するため、SAAS の「入力次元 sparsity」と同一視しません。前者は観測側の sparse correction、後者は入力 lengthscale prior による active dimension sparsity です。

## ALEBOGP

`ALEBOGP` は ALEBO search embedding 上で projection に整合する Mahalanobis geometry を
学習する専用 surrogate です。一般的な data-driven dimensionality reducer ではありません。

## ALEBO の geometry

ALEBO は ambient space から低次元線形 subspace へ探索を制約します。surrogate の Mahalanobis geometry と candidate reconstruction が同じ embedding を共有することが重要です。単に GP の入力へ任意の projection を入れる Reduced GP とは目的が異なります。

## どの仮定を置くか

- 少数の original feature が効く: SAAS / MAP-SAAS
- 低次元 component の和: additive GP
- 観測 X から latent representation を学ぶ: PCA/PLS/AE/VAE
- objective が低次元線形 search subspace に依存: ALEBO
- local region を絞って探索: TuRBO
- intrinsic dimension を探索中に拡張: BAxUS

高次元という理由だけで複数の構造仮定を同時に重ねず、予測性能・計算量・BO trajectory を
同じ条件で比較します。

関連: [High-dimensional GP](08_high_dimensional_gp.md)、
[High-dimensional search](20_high_dimensional_search.md)。
