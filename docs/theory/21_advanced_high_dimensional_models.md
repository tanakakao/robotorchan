# Advanced High-dimensional Gaussian Process Models

[高次元GP章](08_high_dimensional_gp.md)の疎性・加法性と、[高次元探索章](20_high_dimensional_search.md)の座標系を前提とします。
本章では、同じ「高次元対応」でも異なる仮定と推論になる理由を比較します。

## 直感

高次元問題への対応は projection だけではありません。

```text
high-dimensional X
 -> projection
```

以外にも、

- 元特徴のごく一部だけが効く
- 関数が低次元 component の和で表せる
- 特定の low-dimensional linear subspace に目的関数が存在する
- 観測の一部だけが異常である

といった構造仮定を prior / kernel / likelihood に組み込めます。

この章では「高次元」という共通ラベルの下にある異なる statistical assumptions を分離します。

### 最初に「高次元へ効く理由」を分ける

同じhigh-dimensional modelでも、次元の難しさを減らす仮定は同じではありません。

| 構造仮定 | 何を簡単にするか | 代表例 |
| --- | --- | --- |
| axis-aligned sparsity | 多くの元特徴を弱くする | SAAS / MAP-SAAS |
| additive structure | 高次相互作用を制限する | OAK |
| linear subspace | ambient space内の低次元方向を使う | ALEBO |
| learned representation | 観測Xからlatent表現を作る | PCA / PLS / AE / VAE |
| local search geometry | candidate探索範囲を局所化する | TuRBO |
| adaptive search subspace | 探索部分空間を段階的に拡張する | BAxUS |

これらは「高次元」という症状に対する異なるstatistical / search assumptionsです。
複数を同じ意味のdimension reductionとして扱いません。

## 1. ARD と dimension relevance

ARD kernel は dimension ごとの lengthscale を持ちます。

```text
k(x, x')
  = sigma^2 exp(
      -1/2 sum_j
      (x_j - x'_j)^2 / ell_j^2
    )
```

大きな `ell_j` は、そのdimensionの変化に対して、**そのkernelと他のhyperparameterを固定した条件では**
関数が比較的鈍感であることを意味します。lengthscaleだけから因果的重要度を結論づけることは
できません。

しかし通常の ARD では高次元・小標本になるほど多数の lengthscale を安定に推定しにくくなります。

そこで lengthscale / inverse lengthscale に sparsity-inducing prior を置く考え方が SAAS です。

## 2. SAAS

SAAS は多くの input dimension の effect を強く shrink し、少数の active dimensions を
残すことを狙います。

概念的には inverse lengthscale

```text
rho_j = 1 / ell_j
```

の多くを零付近へ縮める考え方です。ただし原著の半Cauchy事前分布を直接置く対象は逆二乗長さ尺度 $1/\ell_j^2$ であり、
$1/\ell_j$ への事前とは密度が異なります。定義と変数変換は[8.9節](08_high_dimensional_gp.md#89-saas-prior)を参照してください。

これは PCA のように

```text
x -> W^T x
```

と latent coordinate を作る方法ではありません。

元のinput dimensionsを保持したまま、そのrelevanceにsparse priorを置きます。

したがってSAASが想定する低有効次元性は主として**元座標軸に沿ったsparsity**です。
任意に回転したdense linear subspaceを直接学習するALEBOやprojection modelとは仮定が異なります。

## 3. Fully Bayesian SAAS

完全ベイズ推論では、ハイパーパラメータ $\theta$ の事後分布について
$p(f_*\mid D)=\int p(f_*\mid D,\theta)p(\theta\mid D)\,d\theta$ と予測します。
MCMCはこの積分を標本平均で近似する方法であり、重要な座標を一組に確定する操作ではありません。
対照的に、周辺尤度最大化やMAPでは、選んだパラメータの下での条件付き予測を使います。
MCMC標本数・混合・収束に由来する誤差と、事前分布の構造による偏りは区別します。

## 4. Fully Bayesian MultiTask SAAS

SAAS fully Bayesian マルチタスク GP は long-format task-feature representation を使います。

```text
X = [data features, task feature]
```

task structure と data-feature sparsity は別の役割です。

## 5. Mixed Fully Bayesian SAAS

カテゴリがある場合、連続変数だけへの疎性と、符号化後の座標への疎性は異なる仮定です。
例えばOne-hot符号化をすると一つの元カテゴリが複数列になるため、
列ごとの長さ尺度を元変数一つの重要度とそのまま同一視できません。
連続入力へのワーピングを用いる場合も、カテゴリの識別子に連続的な距離や順序を導入しないよう分けます。

## 6. MAP-SAAS

MAP型の方法は、疎性を誘導する事前分布を用いながらハイパーパラメータを点推定します。
異なる縮小尺度の下でのカーネルを加える構成と、複数のモデルの予測を混合する構成は別です。
例えば $K=K_1+K_2$ という一つのGPは、$p(f)=\tfrac12p_1(f)+\tfrac12p_2(f)$ という混合分布と一致しません。
前者は潜在過程の和、後者はモデルの選択に対する分布です。
また、複数のMAP推定を等重みで統合しても、それだけでハイパーパラメータの事後積分にはなりません。
この「加算」を、入力ごとの主効果へ分解する加法カーネルと混同しないことが重要です。

## 7. Mixed MAP-SAAS

One-hot符号化したカテゴリへMAP-SAAS型のカーネルを使う構成と、
カテゴリ専用の共分散を連続カーネルと組み合わせる構成は異なります。
同じ混合入力でも、符号化と事前分布が定める情報共有の仮定を確認します。

## 8. Additive structure

高次元関数が少数 active dimensions に依存するという仮定と、関数が component の和であるという
仮定は別です。

additive GP は概念的に

```text
f(x)
  = sum_k f_k(x_{S_k})
```

とします。

各 component が低次元なら、ambient dimension が大きくても statistical complexity を抑えられます。

## 9. Orthogonal Additive GP

一次の主効果に加えて二次相互作用を含む構成では、
$f(x)=c+\sum_j f_j(x_j)+\sum_{j<k}f_{jk}(x_j,x_k)$ と表します。
これは任意次数の相互作用を自由に表現する仮定ではありません。

直交化の一つの構成は、基準確率測度 $P$ に関する関数の積分を零に条件付けるものです。
$a(s)=\int k(s,u)dP(u)$、$c=\iint k(u,v)dP(u)dP(v)>0$ とすれば、条件付き共分散は

$$
k_\perp(s,t)=k(s,t)-\frac{a(s)a(t)}{c}
$$

です。GPの線形汎関数 $\int f(u)dP(u)$ を零に条件付ける公式から得られ、
$\int k_\perp(s,t)dP(t)=0$ となります。直交性は基準測度を指定して初めて意味を持ちます。

orthogonalization により component decomposition の解釈性を改善することが目的の1つです。

この model は SAAS のような active-dimension shrinkage model ではありません。

```text
SAAS:
  sparse relevance across original dimensions

OAK:
  additive decomposition across components
```

という違いがあります。

## 10. Mixed Orthogonal Additive GP

連続区間 $[0,1]$ の一様測度を使う場合、上の積分はGauss–Legendre求積などで近似できます。
一方、カテゴリの確率測度 $P(c_j)=p_j$ なら、$a(s)=\sum_jp_jk(s,c_j)$、
$c=\sum_{i,j}p_ip_jk(c_i,c_j)$ と離散和で定義します。
連続カーネルをカテゴリカーネルへ置き換えるだけでは、直交化の測度まで整合したことにはなりません。

One-hot列を連続区間の座標として使う構成では、成分の解釈は符号化後の列に関するものです。
有効なOne-hotベクトルは単体の頂点上にあるため、列ごとの独立な一様分布とは異なります。
元カテゴリに対する効果を解釈したい場合は、この基準測度の違いを考慮する必要があります。
具体的な符号化・入力範囲の制約は[実装ガイド](../guide/21_advanced_high_dimensional_models.md)で扱います。

## 11. Relevance pursuit は別の sparsity

Robust Relevance Pursuit 単一タスク GP も sparse structure を使いますが、SAASとは対象が違います。

```text
SAAS
 -> sparse input relevance

Robust relevance pursuit
 -> sparse observation-side correction / robustness
```

です。

どちらも sparsity という語を使えますが、統計的意味は同一ではありません。

詳細は [Robust Gaussian Process](14_robust_gaussian_process.md) を参照してください。

## 12. ALEBO の共分散モデル

ALEBO は objective が ambient space 内の low-dimensional linear subspace に依存するという
仮定を利用します。

ALEBO の共分散モデル は ALEBO search embedding と整合する Mahalanobis geometry を持つ専用 surrogate です。

```text
k(z, z')
  = sigma^2 exp(
      -1/2 (z - z')^T M (z - z')
    )
```

ここで $M\in\mathbb R^{d\times d}$ は対称半正定値の計量です。
$M=B^\top B$ と書けば、距離は $\|B(z-z')\|^2$ となり、線形写像後のRBFなので妥当なカーネルです。
任意の対称行列を使えるわけではありません。負の固有値がある方向では、
離れた二点の共分散が対角の分散を上回り、共分散行列の半正定値性を壊し得ます。

これは training data から PCA reducer を fit する 次元削減 GP とは別の考え方です。

## 13. ALEBO metric uncertainty

計量パラメータを $\theta$ とし、事後モードの周りで負対数事後密度のHessianを用いると、
正定値の曲率がある場合はLaplace近似で計量の不確実性を表せます。
平坦な方向や識別不能性があれば、この近似にも正則化や診断が必要です。

S個の計量標本それぞれについて平均 $\mu_s$、共分散 $\Sigma_s$ の正規予測を得ると、混合分布の共分散は

$$
\bar\Sigma=\frac1S\sum_s(\Sigma_s+\mu_s\mu_s^\top)-\bar\mu\bar\mu^\top,
\qquad \bar\mu=\frac1S\sum_s\mu_s.
$$

この平均・共分散に一致する正規分布へ置き換える方法がモーメント整合です。
対角分散だけでなく候補間の共分散も保持できますが、多峰性や高次モーメントは保持しません。
例えば平均 $-1,1$、分散0.25の等重み混合と、平均0・分散1.25の正規分布は、
平均と分散は同じでも同じ確率分布ではありません。非線形効用の期待値が変わり得ます。

## 14. ALEBO の共分散モデル と ALEBO 探索

```text
ALEBO の共分散モデル
 -> embedded-space surrogate geometry

ALEBO 探索
 -> feasible target-space candidate search
```

です。

両者は協調できますが同一オブジェクトではありません。

search strategy 側は original box に対応する feasible polytope を扱い、model 側は embedded
coordinates の covariance geometry を扱います。

詳細は [High-dimensional Search](20_high_dimensional_search.md) を参照してください。

## 15. Projection model との違い

PCA / PLS / AE / VAE は observed training inputs から representation を構築します。

SAAS / additive GP / ALEBO はそれぞれ異なる構造仮定です。

| Family | 主な仮定 | 元特徴をlatentへ変換 |
| --- | --- | --- |
| SAAS | 少数の元特徴がactive | No |
| MAP-SAAS | sparse relevanceのMAP近似 | No |
| OAK | additive component structure | No |
| PCA / PLS | low-dimensional linear representation | Yes |
| AE / VAE | nonlinear representation | Yes |
| ALEBO | low-dimensional linear search subspace | Search側 |
| TuRBO | optimumをlocal regionで探索可能 | No |
| BAxUS | useful subspaceを段階的に拡張可能 | Search側 |

「高次元対応」という理由だけで同じ family とみなさないことが重要です。

## 16. Mixed model の注意

高次元 model の Mixed 対応には複数方式があります。

- native categorical covariance
- One-hot符号化した列へのカーネル
- continuous-only reduction + categorical passthrough
- search strategy 側で discrete dimensions を構造化

どの方式を使うかは underlying model の数学的構造によります。

したがって混合入力という分類だけでは、同じカテゴリ間共分散や同じ成分解釈が得られるとは限りません。

## 17. 問題構造との対応

| 観測・仮定 | model / strategy側の考え方 |
| --- | --- |
| 少数のoriginal axesが主に効く | SAAS / MAP-SAAS |
| low-order additive decompositionが妥当 | OAK |
| observed Xからlinear representationを学ぶ | PCA / PLS |
| observed Xからnonlinear representationを学ぶ | AE / VAE |
| low-dimensional linear search subspaceを仮定 | ALEBO |
| original spaceの局所探索へ集中 | TuRBO |
| search subspace dimensionを探索中に拡張 | BAxUS |

これは性能順位ではありません。高次元だからといって複数の強い仮定を無条件に重ねると、
model misspecificationを増やす可能性があります。

predictive performance、posterior calibration、計算量、BO trajectoryを同じ条件で比較します。

## 19. この章で覚えておくこと

高次元への対策には、元座標の疎性、低次相互作用、低次元部分空間、表現学習、局所探索という異なる仮定があります。
SAASの疎性とOAKの加法分解、完全ベイズ推論の積分とMAPの点推定を区別します。
One-hot符号化では成分の解釈が列単位になり、元のカテゴリの効果や離散測度と同じとは限りません。

埋め込み空間のMahalanobis計量は半正定値性を満たす必要があります。
計量の不確実性を積分する場合も、混合分布をそのまま扱う方法と、平均・共分散を合わせた正規近似は別です。
予測モデルの幾何と、可行な候補を探す戦略は、それぞれの座標と仮定を確認して組み合わせます。

## 参考文献

1. Eriksson, D. and Jankowiak, M. (2021).
   High-dimensional Bayesian optimization with sparse axis-aligned subspaces.
   *Proceedings of the Thirty-Seventh Conference on Uncertainty in Artificial Intelligence*.
2. Hvarfner, C., Hellsten, E. O., and Nardi, L. (2024).
   Vanilla Bayesian optimization performs great in high dimensions.
   *International Conference on Machine Learning*.
3. Lu, X., Boukouvalas, A., and Hensman, J. (2022).
   Additive Gaussian processes revisited.
   *International Conference on Machine Learning*.
4. Letham, B., Calandra, R., Rai, A., and Bakshy, E. (2020).
   Re-examining linear embeddings for high-dimensional Bayesian optimization.
   *Advances in Neural Information Processing Systems*.
5. Rasmussen, C. E. and Williams, C. K. I. (2006).
   *Gaussian Processes for Machine Learning*. MIT Press.

## 関連章

- [High-dimensional GP](08_high_dimensional_gp.md)
- [Dimensionality Reduction GP](18_dimensionality_reduction_gp.md)
- [Neural Representation GP](19_neural_reduction_gp.md)
- [High-dimensional Search](20_high_dimensional_search.md)
- [Robust Gaussian Process](14_robust_gaussian_process.md)
