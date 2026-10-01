# Heteroskedastic / Replicate Noise

## 直感

標準 GP 回帰では、観測ノイズをすべての入力で同じ分散とすることがよくあります。

```text
y_i = f(x_i) + epsilon_i
epsilon_i ~ Normal(0, sigma^2)
```

しかし実験条件、装置状態、測定レンジ、fidelity などによって測定精度が変わる場合、
`sigma^2` を一定とする仮定は適切ではありません。

robotorchan では noise の扱いを少なくとも次の3種類に分けます。

- 入力依存の未知 noise variance を学習する heteroskedastic GP
- response と noise process を joint に学習する joint heteroskedastic GP
- replicate から観測分散を直接推定する replicate-noise GP

## 1. Aleatoric uncertainty と epistemic uncertainty

GP posterior の uncertainty を解釈するとき、次の区別が重要です。

- epistemic uncertainty: latent function を十分に知らないことによる不確かさ
- aleatoric uncertainty: 観測機構そのもののランダムなばらつき

heteroskedasticity は主に後者が入力 `x` に依存する状況です。

```text
y_i = f(x_i) + epsilon_i
epsilon_i ~ Normal(0, sigma^2(x_i))
```

高ノイズ領域で観測が散らばっていても、それだけで latent function の epistemic uncertainty
が大きいとは限りません。この区別は Bayesian optimization で「新しい場所を探索するか」
「同じ場所を再測定するか」を考える際にも重要です。

## 2. Heteroskedastic GP

一般的な heteroskedastic GP では、response process `f(x)` に加えて noise variance を表す
latent process を導入します。正値性を保つため、典型的には log variance をモデル化します。

```text
f ~ GP(m_f, k_f)
g ~ GP(m_g, k_g)

y_i | f_i, g_i ~ Normal(f_i, exp(g_i))
```

これにより noise level が入力空間で滑らかに変化する構造を表現できます。

robotorchan の `HeteroskedasticSingleTaskGP` は response GP と入力依存 noise estimate を
扱う構成です。実装詳細では、noise model がどの quantity を学習し、response model へ
どのように接続されるかをコードとテストで確認してください。

Mixed counterpart は categorical covariance を保持し、continuous feature と category identity
を同じ Euclidean geometry に押し込みません。

## 3. Joint heteroskedastic inference

response と noise を別々の段階で推定すると、noise estimate の不確かさや両者の coupling を
十分に反映できない場合があります。

`JointHeteroskedasticSingleTaskGP` は response と log-noise の latent process を joint
objective で学習する構成です。概念的には

```text
q(f, g)
E_q(f,g)[log p(y | f, g)] - KL terms
```

のように、response と noise process を同じ推論問題の中で扱います。

利点は noise process の uncertainty を明示的に扱いやすいことです。一方で latent process が
増えるため、標準 exact GP より推論・最適化が複雑になります。

モデル固有の training contract は
[Joint heteroskedastic GP](../models/joint_heteroskedastic_gp.md) を参照してください。

## 4. Replicate noise

同一 design condition `x` を `r` 回測定できる場合、

```text
y_(x,1), ..., y_(x,r)
```

からその条件における観測分散を直接推定できます。

sample mean を

```text
y_bar(x) = (1 / r) * sum_j y_(x,j)
```

sample variance を

```text
s^2(x) = 1 / (r - 1) * sum_j (y_(x,j) - y_bar(x))^2
```

とすると、独立 replicate を仮定した mean の variance は

```text
Var[y_bar(x)] ~= s^2(x) / r
```

です。

robotorchan の replicate-noise 実装は exact duplicate の design row を group 化し、
group mean と unbiased sample variance を計算します。GP に渡す `train_Yvar` は
replicate variance そのものではなく **mean variance** であり、

```text
train_Yvar = max(s^2(x) / r, noise_floor)
```

という関係を持ちます。

これは理論と実装の対応上、特に重要です。

## 5. Replicate の成立条件

replicate と呼ぶためには「同じ条件」が何を意味するかを明確にする必要があります。

robotorchan の aggregation は raw input row の exact duplicate を同一条件として扱います。
したがって連続値に丸め誤差がある場合、意味的には同一条件でも別 group になる可能性があります。

また各 design condition には最低2 replicate が必要です。1点だけでは unbiased sample variance
を推定できないためです。

`noise_floor` はゼロ分散や数値的に極端に小さい分散をそのまま fixed noise として渡すことを
避けるために使います。

## 6. Mixed replicate noise

`MixedReplicateNoiseSingleTaskGP` では、continuous / categorical を含む raw row の同一性を
保ったまま replicate aggregation を行います。

カテゴリ列について「近いカテゴリ」という連続距離を導入するのではなく、同じカテゴリ値を持つ
design condition の replicate として扱います。aggregation 後の surrogate でも native mixed
covariance を保持します。

## 7. Multi-fidelity replicate noise

robotorchan には `ReplicateNoiseMultiFidelityGP` もあります。ここでは fidelity feature も
raw design row の一部なので、

```text
(x, s_low) != (x, s_high)
```

として別の replicate group になります。

これは重要な意味を持ちます。異なる fidelity の観測値を「同じ条件の反復測定」として
平均してはいけません。replicate は同一 fidelity 内の観測ばらつきを推定し、
multi-fidelity covariance は fidelity 間の関係を別途モデル化します。

## 8. Heteroskedastic model と replicate model の違い

両者は「noise が一定ではない」問題を扱いますが、情報源が異なります。

| 方式 | noise 情報 | 主な仮定 |
| --- | --- | --- |
| Homoskedastic GP | 全体から1つ | 一定分散 |
| Heteroskedastic GP | 入力から学習 | noise field が構造を持つ |
| Joint heteroskedastic | response と同時推論 | latent noise process |
| Replicate noise | 同一点の反復 | replicate が利用可能 |

replicate が十分にあるなら、局所的な observation variance を直接推定できる利点があります。
一方、replicate のない新しい領域の noise level を一般化したい場合は heteroskedastic noise
model が必要になります。

## 9. Robust GP との違い

heteroskedasticity は「分散が場所によって変わる」問題です。heavy-tailed error や
gross outlier と同じではありません。

例えば高分散だが Gaussian な領域は heteroskedastic model の対象です。一方、通常は安定して
いるが稀に極端な異常値が出るなら Student-t や contamination model の方が生成機構に近い
可能性があります。

詳細は [Robust Gaussian Process](14_robust_gaussian_process.md) を参照してください。

## 10. Nonstationarity との違い

heteroskedasticity は observation variance の変化です。

```text
sigma^2 = sigma^2(x)
```

一方、nonstationary GP は latent function の covariance structure 自体が場所によって変わる
問題です。例えば lengthscale が `x` に依存する状況です。

```text
ell = ell(x)
```

両者は同時に存在し得ますが、同じ概念ではありません。

詳細は [Nonstationary GP](17_nonstationary_gp.md) を参照してください。

## 11. Bayesian optimization との関係

noise modeling は acquisition の解釈に直接影響します。

- latent posterior uncertainty が高い領域
- observation noise が高い領域
- replicate により noise estimate を改善できる領域

を区別できることが重要です。

Noisy Expected Improvement のような noisy-observation 向け acquisition を使う場合でも、
surrogate 側の observation model が適切であることは別途必要です。acquisition の
「noisy」という名前だけで heteroskedasticity が自動的に学習されるわけではありません。

## 12. 実装との対応

理論章で保持すべき実装契約は次です。

- heteroskedasticity は output noise の入力依存性
- joint model は response / noise process を同じ推論問題で扱う
- replicate aggregation は exact duplicate raw rows に基づく
- replicate variance は unbiased sample variance
- GP に渡す mean variance は `replicate_variance / replicate_count`
- mean variance には `noise_floor` を適用
- 各 replicate group は2観測以上を要求
- Mixed 版は categorical identity を保持
- MultiFidelity 版は fidelity を replicate identity の一部として保持

対応モデルの仕様は [Robust / Noise model guide](../models/robust_noise.md)、
[Replicate noise GP](../models/replicate_noise_gp.md)、
[Joint heteroskedastic GP](../models/joint_heteroskedastic_gp.md) を参照してください。

## 参考文献

1. Goldberg, P. W., Williams, C. K. I., and Bishop, C. M. (1998).
   Regression with input-dependent noise: A Gaussian process treatment.
   *Advances in Neural Information Processing Systems 10*.
2. Kersting, K., Plagemann, C., Pfaff, P., and Burgard, W. (2007).
   Most likely heteroscedastic Gaussian process regression.
   *Proceedings of the 24th International Conference on Machine Learning*.
3. Lázaro-Gredilla, M. and Titsias, M. K. (2011).
   Variational heteroscedastic Gaussian process regression.
   *Proceedings of the 28th International Conference on Machine Learning*.
4. Binois, M., Gramacy, R. B., and Ludkovski, M. (2018).
   Practical heteroskedastic Gaussian process modeling for large simulation
   experiments. *Journal of Computational and Graphical Statistics*, 27(4).
5. Rasmussen, C. E. and Williams, C. K. I. (2006).
   *Gaussian Processes for Machine Learning*. MIT Press.

## 関連章

- [Gaussian Process](02_gaussian_process.md)
- [Robust Gaussian Process](14_robust_gaussian_process.md)
- [Uncertain-input Gaussian Process](16_uncertain_input_gp.md)
- [Multi-fidelity](06_multi_fidelity.md)
