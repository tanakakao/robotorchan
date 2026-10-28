# Heteroskedastic / Replicate Noise

[GP章](02_gaussian_process.md)の潜在事後分布と観測ノイズを前提とします。
本章では、潜在応答 $f$、対数ノイズ分散 $g$、同一条件の反復数 $r$ を用いて、分散の情報源を区別します。

## 直感

標準 GP 回帰では、観測ノイズをすべての入力で同じ分散とすることがよくあります。

```text
y_i = f(x_i) + epsilon_i
epsilon_i ~ Normal(0, sigma^2)
```

しかし実験条件、装置状態、測定レンジ、fidelity などによって測定精度が変わる場合、
`sigma^2` を一定とする仮定は適切ではありません。

- 入力依存の未知 noise variance を学習する heteroskedastic GP
- response と noise process を joint に学習する joint heteroskedastic GP
- replicate から観測分散を直接推定する replicate-noise GP

### 最初に3つのvarianceを分ける

この章で最も重要なのは、次の量を同じ「GPの分散」と呼ばないことです。

~~~text
Var[f(x) | D]
  latent functionをまだ知らない不確かさ

sigma^2(x)
  1回の観測に入るobservation noise variance

Var[y_bar(x)]
  replicate平均を観測値として使うときのmean variance
~~~
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

~~~text
response GPをfit
  ↓
training residualを計算
  ↓
log(residual^2)をnoise GPへfit
  ↓
予測noise varianceをresponse likelihoodへ反映
  ↓
反復
~~~
この反復推定は一つの近似法です。学習残差はノイズの実現値そのものではなく、
応答推定の誤差も含みます。また零残差の対数は定義できず、補正や下限が必要です。
混合入力ではカテゴリの意味を保つ共分散を、忠実度がある場合はノイズにも忠実度依存性を考えます。

## 3. Joint heteroskedastic inference

response と noise を別々の段階で推定すると、noise estimate の不確かさや両者の coupling を
十分に反映できない場合があります。

同時推論では応答と対数ノイズの潜在過程を一つの推論問題として学習します。概念的には

```text
q(f, g)
E_q(f,g)[log p(y | f, g)] - KL terms
```

のように、response と noise process を同じ推論問題の中で扱います。

対数ノイズの近似事後分布が正規分布 $q(g(x))=\mathcal N(m_g,v_g)$ なら、
正規分布の積率母関数から、平均ノイズ分散は単純な `exp(E[g])` ではなく

~~~text
E[exp(g)] = exp(E[g] + Var[g] / 2)
~~~

を使ってobservation varianceへ戻します。したがってnoise-process uncertaintyも
posterior-mean noise varianceに反映されます。

例えば $m_g=0,v_g=1$ なら、平均ノイズ分散は $e^{1/2}\simeq1.649$ です。
$e^{m_g}=1$ は対数正規分布の中央値であり、平均とは異なります。

新しい観測が $y_*=f_*+\epsilon_*$ で、$f_*,g_*$ を与えた下で
$\mathbb E[\epsilon_*\mid f_*,g_*]=0$、$\operatorname{Var}(\epsilon_*\mid f_*,g_*)=e^{g_*}$ とすると、全分散の公式より

$$
\operatorname{Var}(y_*\mid D)=\operatorname{Var}(f_*\mid D)+\mathbb E[e^{g_*}\mid D].
$$

応答とノイズ過程の事後相関があっても、この条件付き零平均の仮定の下では交差項は消えます。
ただし、同時事後分布は一般に非正規なので、対数正規の積率の式を使う箇所は近似分布への仮定です。
潜在過程が増えるため、通常の正規尤度GPより推論・最適化は複雑になります。

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

```text
既知の観測分散 = max(s^2(x) / r, noise_floor)
```

という関係を持ちます。

ここで下限値を加える操作は数値的な正則化であり、不偏な標本分散そのものとは区別します。
また、反復から推定した分散を固定して使うことは、その推定誤差まで事後分布で積分することとは異なります。
例えば $r=4$、標本分散 $s^2=0.36$ なら、平均の分散推定値は0.09、標準誤差は0.3です。
個々の観測の標準偏差0.6をそのまま平均の誤差へ使うと、反復による精度の改善を表せません。

## 5. Replicate の成立条件

replicate と呼ぶためには「同じ条件」が何を意味するかを明確にする必要があります。

また各 design condition には最低2 replicate が必要です。1点だけでは unbiased sample variance
を推定できないためです。

ノイズ分散の下限は数値的な安定化に使えますが、反復数が少ないと分散推定自体にも大きな誤差が残ります。
また、同一条件でも装置ドリフトなどで反復が相関していれば $s^2/r$ の式はそのまま使えません。
各反復の分散を $\sigma^2$、異なる反復間の相関を $\rho$ とすると、共分散を足し合わせることで

$$
\operatorname{Var}(\bar y)=\frac{\sigma^2}{r}\{1+(r-1)\rho\}
$$

を得ます。例えば $r=4,\rho=0.5$ なら $0.625\sigma^2$ で、独立な場合の $0.25\sigma^2$ より大きくなります。

## 6. Mixed replicate noise

混合入力では、連続・カテゴリを含む設計条件が同じかを定義してから反復観測を集約します。

カテゴリ列について「近いカテゴリ」という連続距離を導入するのではなく、同じカテゴリ値を持つ
設計条件の反復として扱います。集約後の共分散にも、カテゴリの統計的な意味を保持します。

## 7. Multi-fidelity replicate noise

```text
(x, s_low) != (x, s_high)
```

として別の replicate group になります。

これは重要な意味を持ちます。異なる fidelity の観測値を「同じ条件の反復測定」として
平均してはいけません。replicate は同一 fidelity 内の観測ばらつきを推定し、
multi-fidelity covariance は fidelity 間の関係を別途モデル化します。

### Known noise と learned noise も区別する

`既知の観測分散` を外部から与えられる場合は、そのvarianceを既知として使うfixed-noise modelingです。
これは未知のnoise surfaceをresidualから推定するheteroskedastic GPとは異なります。

~~~text
known 既知の観測分散
  → noiseを観測・推定済みとしてresponse GPへ渡す

heteroskedastic GP
  → Xからnoise process自体を学習する

replicate GP
  → replicateから各group meanのvarianceを推定して既知の観測分散へ渡す
~~~

replicate modelは局所varianceをデータから構成しますが、それだけではreplicateのない
新しいXに対するnoise surfaceを学習するモデルではありません。

## 8. Heteroskedastic model と replicate model の違い

両者は「noise が一定ではない」問題を扱いますが、情報源が異なります。

| 方式 | noise 情報 | 主な仮定 |
| --- | --- | --- |
| Homoskedastic GP | 全体から1つ | 一定分散 |
| Iterative heteroskedastic | residualから学習 | smoothなlog-noise field |
| Joint heteroskedastic | responseと同時推論 | 2つのlatent variational process |
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

## 12. Bayesian optimization との関係

noise modeling は acquisition の解釈に直接影響します。

- latent posterior uncertainty が高い領域
- observation noise が高い領域
- replicate により noise estimate を改善できる領域

を区別できることが重要です。

Noisy Expected Improvement のような noisy-observation 向け acquisition を使う場合でも、
surrogate 側の observation model が適切であることは別途必要です。acquisition の
「noisy」という名前だけで heteroskedasticity が自動的に学習されるわけではありません。

## 14. この章で覚えておくこと

潜在関数の予測分散、1回の測定のノイズ分散、反復平均の分散は異なる量です。
既知とする分散、残差から推定するノイズ過程、反復測定で推定する局所分散は情報源も異なります。
反復平均を応答として使う場合は、独立性を確認したうえで平均の分散を扱います。

交互推定と同時変分推論では、ノイズ推定の不確実性の扱いが変わります。
カテゴリ・タスク・忠実度の意味をノイズ過程でも保持し、
ノイズに対応した獲得関数を選ぶこととノイズモデル自体を学ぶことを区別します。

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
