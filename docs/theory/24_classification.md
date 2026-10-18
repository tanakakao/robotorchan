# ガウス過程による分類

## 1. 問題設定

GP分類はラベル自体にガウス観測ノイズを仮定せず、
実数値の潜在関数を通じて離散ラベルをモデル化します。
二値観測 \(y_i\in\{0,1\}\) に対して、次を仮定します。

$$
f\sim\mathcal{GP}(m,k),\qquad
y_i\mid f_i\sim\operatorname{Bernoulli}(\pi(f_i)),
\quad f_i=f(x_i).
$$

リンク関数 \(\pi:\mathbb R\to(0,1)\) にはlogistic
\(\pi(f)=1/(1+\exp(-f))\) やprobit \(\pi(f)=\Phi(f)\) を利用できます。
潜在値の事前分布は多変量正規分布ですが、条件付き観測分布はBernoulli分布です。
潜在値の事後分布は一般にガウス分布ではありません。
予測ラベル分布はBernoulli分布であり、ガウス分布ではありません。

## 2. 事後推論

\(f=(f_1,\ldots,f_n)\) とデータ
\(\mathcal D=\{(x_i,y_i)\}_{i=1}^n\) に対し、Bayesの定理から次を得ます。

$$
p(f\mid\mathcal D)
=\frac{p(f)\prod_{i=1}^n p(y_i\mid f_i)}
{\int p(f')\prod_{i=1}^n p(y_i\mid f_i')\,df'}.
$$

Bernoulli尤度はガウス事前分布に共役ではないため、
正規化積分を一般に解析的には計算できません。
代表的な近似法にはLaplace近似、期待値伝播、変分推論があり、
近似仮定と数値的性質が異なります。

### 2.1 疎変分推論

誘導点 \(Z\) 上の誘導変数を \(u=f(Z)\) とし、
変分分布 \(q(u)\) を導入すると、潜在分布の近似は次のとおりです。

$$
q(f)=\int p(f\mid u)q(u)\,du.
$$

標準的なEvidence Lower Bound（ELBO）は次のとおりです。

$
\log p(y\mid X)
\geq
\mathcal L
=
\sum_{i=1}^n
\mathbb E_{q(f_i)}[\log p(y_i\mid f_i)]
-
\operatorname{KL}\!\left(q(u)\,\|\,p(u)\right).
$$

この下界はJensenの不等式から導かれます。
上式は $q(f,u)=p(f\mid u)q(u)$ という変分族と、
条件付き独立な観測尤度を仮定しています。
尤度の期待値には数値積分やMonte Carlo推定が必要な場合があります。
ミニバッチ学習ではデータ適合項を全データ数に合わせてスケーリングします。
ELBOはGP回帰で用いる厳密なガウス周辺対数尤度とは異なります。

## 3. 潜在不確実性と予測確率

潜在事後分布の近似 \(q(f_*\mid\mathcal D)\) に対して、
事後予測確率は次のとおりです。

$$
\bar p_*(x)
=
p(y_*=1\mid x,\mathcal D)
\approx
\int \pi(f_*)q(f_*\mid\mathcal D)\,df_*.
$$

**一般に \(\bar p_*(x)\neq\pi(\mathbb E_q[f_*])\) です。**
潜在平均だけにリンク関数を適用すると、潜在不確実性を無視します。

次の3種類の不確実性は区別する必要があります。

1. **潜在不確実性：** 潜在関数の尺度で測る \(\operatorname{Var}_q[f_*]\)。
2. **確率の不確実性：** Bernoulli成功確率の事後変動
   \(\operatorname{Var}_q[\pi(f_*)]\)。
3. **予測ラベルの不確実性：**
   \(\operatorname{Var}[y_*\mid x,\mathcal D]=\bar p_*(1-\bar p_*)\)。
   成功確率が既知でも残るラベルの偶然性を含みます。

全分散の法則により、この区別は次のように表せます。

$$
\underbrace{\bar p_*(1-\bar p_*)}_{\text{predictive label variance}}
=
\underbrace{\mathbb E_q[\pi(f_*)(1-\pi(f_*))]}_{\text{conditional label variance}}
+
\underbrace{\operatorname{Var}_q[\pi(f_*)]}_{\text{probability uncertainty}}.
$$

probitリンクで周辺分布が \(q(f_*)=\mathcal N(\mu_*,s_*^2)\) のとき、
予測確率は次の解析式を持ちます。

$$
\bar p_*=\Phi\!\left(\frac{\mu_*}{\sqrt{1+s_*^2}}\right).
$$

logisticリンクでは、近似なしに同じ恒等式は成立しません。

## 4. 分類の不確実性と能動学習

\(C\) クラスの予測エントロピーは次のとおりです。

$$
H[y_*\mid x,\mathcal D]
=
-\sum_{c=1}^{C}\bar p_c(x)\log\bar p_c(x).
$$

これは将来のラベルの不確実性を測り、潜在モデルの不確実性だけを表すものではありません。
BALDはラベルと潜在関数の相互情報量を利用します。

$$
I(y_*;f_*\mid x,\mathcal D)
=
H[y_*\mid x,\mathcal D]
-
\mathbb E_{q(f_*)}\!\left[H[y_*\mid f_*]\right].
$$

予測エントロピーが大きくても情報利得が大きいとは限りません。
潜在関数がよく分かっていても、ラベルに本質的なノイズが残り得るためです。
確率分散基準は \(\operatorname{Var}_q[\pi(f_*)]\) を用い、
境界基準は指定した判定閾値付近の不確実性を重視します。
それぞれエントロピーや分散サンプリングとは統計的な目的が異なります。

これらはGP事後分布そのものの性質ではなく、*候補選択の目的*です。
獲得関数の理論は[能動学習](acquisition/09_active_learning.md)と
[分類能動学習](acquisition/classification-active-learning.md)を参照してください。

## 5. 多クラス分類

\(C>2\) の場合、潜在関数のベクトル
\(f_1(x),\ldots,f_C(x)\) とsoftmaxリンクを持つカテゴリ尤度を使う方法があります。

$$
p(y=c\mid f(x))
=
\frac{\exp(f_c(x))}{\sum_{j=1}^{C}\exp(f_j(x))}.
$$

潜在関数は事前に独立でも、多出力共分散を通じて相関していても構いません。
予測クラス確率は潜在関数の**同時**事後分布についてsoftmaxを積分します。

$$
p(y=c\mid x,\mathcal D)
=
\mathbb E_{p(f(x)\mid\mathcal D)}
\left[
\frac{\exp(f_c(x))}{\sum_j\exp(f_j(x))}
\right].
$$

softmax尤度は全潜在ロジットに同じ定数を加えても不変であり、
識別可能性に注意が必要です。多クラス推論と意思決定は、
一般に単一の二値閾値に還元できません。

## 6. モデルの仮定と限界

- リンク関数と潜在共分散がクラス確率を決めます。柔軟なカーネルでも校正は保証されません。
- クラス不均衡、非対称な誤分類コスト、データ分布の変化は閾値や評価にも影響します。
- 二値ラベルへのガウス尤度回帰はBernoulli GP分類と等価ではありません。
- Brier scoreや対数損失は確率予測を評価します。正解率だけでは校正を評価できません。
- 潜在分散、確率分散、予測エントロピーは尺度と意味が異なり、交換できません。

### 6.1 分類閾値と最適化制約の意味

二値分類で $\bar p(x)=P(y=1\mid x,\mathcal D)$ とすると、
予測クラスは閾値 $\tau$ を使って
$\hat y(x)=\mathbf 1[\bar p(x)\geq\tau]$ と定義できます。
$\tau=0.5$ は対称な誤分類コストの下で自然な選択ですが、
コストやクラス事前確率、意思決定の目的が変われば最適な閾値も変わります。

制約付きベイズ最適化で「合格確率が少なくとも $\eta$」という要件を
課す場合、条件は $\bar p(x)\geq\eta$ です。
これは潜在平均 $\mathbb E[f(x)]\geq0$ と一般に等価ではありません。
また、予測クラスの閾値と最適化上の要求確率は異なる概念です。

### 6.2 不均衡・ラベルノイズ・校正

クラス不均衡がある場合、全体の正解率だけでは少数クラスの性能を
判断できません。確率予測の評価には対数損失、Brier score、
校正曲線などを併用し、必要に応じてクラス別の指標も確認します。

観測ラベルが誤って記録される場合、潜在関数の不確実性だけでなく、
ラベル生成過程そのものをモデル化する必要があります。
たとえば真のクラス $z\in\{0,1\}$ に対して対称な反転確率
$\epsilon\in[0,1/2)$ を仮定すると、

$
P(y=1\mid f)=\epsilon+(1-2\epsilon)\pi(f).
$

この式は通常のBernoulliリンクとは異なる観測過程を表します。
観測ラベルのノイズと予測クラスの不確実性を混同しないことが重要です。

### 6.3 多クラス潜在表現の識別性

softmaxは全クラスの潜在値に同じ定数を加えても変化しません。
そのため、クラス確率は識別可能でも潜在ロジットの絶対的な
基準は識別できない場合があります。基準クラスを固定する方法や
適切な事前分布を用いる方法が考えられます。

多クラスの予測確率は潜在関数の**同時分布**を通じて計算されるため、
各クラスの潜在平均にsoftmaxを適用するだけでは、一般に
事後不確実性を正しく反映できません。

### 6.4 高次元分類とモデル構造の合成

高次元分類では、入力次元削減、疎な長さ尺度事前分布、ランダム埋め込み、
ニューラル特徴表現を潜在GPの入力側に組み合わせられます。
これらは入力の表現や潜在共分散を変える仮定であり、
Bernoulli尤度と予測クラス確率の定義を自動的に変えるものではありません。

一方、確率校正、適合性予測、アンサンブル、不確実性診断は、
予測分布をどのように利用・評価するかという別の問題です。
それぞれを組み合わせる際は、潜在関数の不確実性と
クラス確率の不確実性を混同しないことが重要です。

マルチフィデリティ、タスク、カテゴリなどの構造列は、
入力次元削減の対象となる連続設計座標とは統計的な意味が異なります。
単一の射影に無条件で含めると、根拠のない距離を導入する可能性があります。

## 7. 参考文献

- Rasmussen, C. E., & Williams, C. K. I. (2006).
  *Gaussian Processes for Machine Learning*, Chapter 3.
- Hensman, J., Matthews, A. G. de G., & Ghahramani, Z. (2015).
  *Scalable Variational Gaussian Process Classification*. AISTATS.
- Gal, Y., Islam, R., & Ghahramani, Z. (2017).
  *Deep Bayesian Active Learning with Image Data*. ICML.
- Eriksson, D., & Jankowiak, M. (2021).
  *High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces*. UAI.
- Houlsby, N., Huszár, F., Ghahramani, Z., & Lengyel, M. (2011).
  *Bayesian Active Learning for Classification and Preference Learning*.

## 8. 関連理論

- [Variational GP](09_variational_gp.md)
- [Mixed Variables](05_mixed_variables.md)
- [Multi-task / Multi-output](07_multitask_multioutput.md)
- [High-dimensional GP](08_high_dimensional_gp.md)
- [Dimensionality Reduction GP](18_dimensionality_reduction_gp.md)
- [Expressive GP Models](22_expressive_gp.md)
- [Classification Active Learning](acquisition/classification-active-learning.md)

ライブラリ固有のモデル群、予測API、対応する組合せについては
[分類モデルガイド](../models/classification.md)を参照してください。
