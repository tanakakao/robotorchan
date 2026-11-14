# 13. 分類のActive Learning

分類のActive Learning（能動学習）は、**ラベルがまだ分からない候補の中から、次に実験する価値が高い入力を選ぶ**方法です。製造条件と合否（0/1）を例にすれば、すべての条件を測定する代わりに、すでに観測したラベルから学習して次の条件を選びます。

~~~text
観測済みデータ (X, y)
  → 分類モデルを学習
  → 未観測候補の確率・不確実性を予測
  → 獲得関数を最大化して候補を選択
  → 実際に測定してラベルを取得
  → データを更新し、必要なら繰り返す
~~~

**クラス1の確率が高い点を選ぶこと**と、**学習に役立つ点を選ぶこと**は同じではありません。潜在関数、クラス確率、観測ラベルの違いは[分類GPの理論](../24_classification.md)を参照してください。

## 13.1 記号と不確実性の整理

観測データを $\mathcal D$、潜在値を $f(x)$、潜在値が与えられたときの成功確率を $p_f(x)=\pi(f(x))$ とおきます。近似事後分布を $q(f\mid\mathcal D)$ とすると、モデルの予測確率は

$$
\bar p(x)=\mathbb E_{q(f(x)\mid\mathcal D)}[p_f(x)].
$$

| 対象 | 例となる量 | 意味 |
| --- | --- | --- |
| 潜在関数 | $\operatorname{Var}_q[f(x)]$ | 連続的な潜在値の推定誤差 |
| 成功確率 | $\operatorname{Var}_q[p_f(x)]$ | 成功確率そのものの推定の曖昧さ |
| 次のラベル | $H[y\mid x,\mathcal D]$ | 0/1のどちらが出るかの曖昧さ |

成功確率がほぼ確実に0.5と分かっていても、次のラベルは不確かです。モデルの知識不足（epistemic）と、ラベルに内在するランダムさ（aleatoric）を混同しないでください。

## 13.2 Predictive Entropy：予測ラベルの曖昧さ

$C$ クラスで各クラスの予測確率を $\bar p_c$ とすると、

$$
a_{\mathrm{entropy}}(x)
=-\sum_{c=1}^{C}\bar p_c(x)\log\bar p_c(x).
$$

二値なら $p=\bar p(x)$ とおいて

$$
h(p)=-p\log p-(1-p)\log(1-p).
$$

ここでは自然対数を用い、$0\log0=0$ とします。$p=0.5$ で最大 $\log 2\simeq0.693$、$p=0.1$ で約0.325、$p=0$ または1で0です。

**直感：** 予測が五分五分の点を優先します。ただし、ラベル固有の偶然性だけでも高いEntropyになります。そのため情報利得と同一ではありません。

公開API：PredictiveEntropy。単一点版は現在 $q=1$ を要求します。

## 13.3 Margin Uncertainty：クラス確率の接戦

予測確率の上位2クラスを $p_{(1)}\ge p_{(2)}$ とすると、現行実装が最大化するスコアは

$$
a_{\mathrm{margin}}(x)
=1-\bigl[p_{(1)}(x)-p_{(2)}(x)\bigr].
$$

二値分類では上位2クラスが $p,1-p$ なので、

$$
a_{\mathrm{margin}}(x)=1-|2p-1|.
$$

$p=0.5$ なら1、$p=0.9$ なら0.2です。二値の単一点ではEntropyと同じ順位を与えますが、**値の尺度は異なります**。多クラスでは上位2クラス以外の確率を直接評価しないため、Entropyと同じ順位になるとは限りません。

公開API：MarginUncertainty。

## 13.4 Probability Variance：成功確率そのものの事後分散

潜在事後サンプルをクラス確率へ変換したときのクラスごとの分散を

$$
v_c(x)=\operatorname{Var}_{q(f(x)\mid\mathcal D)}
[P(y=c\mid f(x))]
$$

とします。現行実装はクラスごとの分散の平均

$$
a_{\mathrm{probvar}}(x)=\frac{1}{C}\sum_{c=1}^{C}v_c(x)
$$

を使用します。二値分類では2つのクラス確率が $p$ と $1-p$ なので、それぞれの分散は等しくなります。

成功確率がほぼ確実に0.5なら確率の事後分散はほぼ0です。一方、成功確率が0.1か0.9かをモデルが決めかねていれば、平均確率は0.5でも分散は大きくなります。

**注意：** これはラベルのBernoulli分散 $\bar p(1-\bar p)$ とは異なります。

公開API：ProbabilityVariance。num_samples引数でMonte Carloのサンプル数を設定できます。

## 13.5 BALD：ラベルから得られるモデル情報

BALD（Bayesian Active Learning by Disagreement）は、予測ラベル $y$ と潜在値 $f$ の相互情報量を考えます。

$$
a_{\mathrm{BALD}}(x)
=I(y;f\mid x,\mathcal D)
=H[y\mid x,\mathcal D]
-\mathbb E_{q(f(x)\mid\mathcal D)}[H(y\mid f(x))].
$$

第1項は平均した予測分布のEntropy、第2項は潜在値を仮に知っていた場合にも残るラベルのEntropyです。差は、ラベルを観測することにより潜在関数について学べる情報の尺度です。

### EntropyとBALDの具体例

平均予測確率が両方とも0.5になる、概念的な極限例を考えます。

| 候補 | 成功確率についてのモデルの状態 | Entropy | BALD |
| --- | --- | --- | --- |
| A | 成功確率が常に0.5だとほぼ確信 | $\log 2$ | ほぼ0 |
| B | 成功確率がほぼ0かほぼ1か判断できない | $\log 2$ | ほぼ$\log 2$ |

Aではラベルは予測しづらくてもモデルの知識はすでに十分です。Bでは新しいラベルを観測することで、相反するモデルの可能性を区別できることがあります。

**Entropy最大とBALD最大は一致するとは限りません。** ただし、どの問題でもBALDが優れているという主張ではありません。

公開API：BALD。実装はモデルのmutual_information契約を用い、num_samplesで推定精度と計算負荷を調整します。

## 13.6 Latent Straddle：潜在境界付近を探索する

二値GPの潜在値の平均を $\mu_f(x)$、標準偏差を $\sigma_f(x)$ とします。現行実装は

$$
a_{\mathrm{straddle}}(x)
=\beta\sigma_f(x)-|\mu_f(x)|,\qquad\beta\ge0
$$

を最大化します。平均がゼロに近いほど潜在境界へ近く、標準偏差が大きいほど未学習の可能性があります。

例えば $\beta=1$ とすると、点Aが $(\mu,\sigma)=(0,0.1)$ のときスコアは0.1、点Bが $(0.4,1.0)$ のとき0.6なので、点Bが優先されます。

**潜在ゼロ境界は任意の確率閾値による意思決定境界と同義ではありません。** この指標は現行実装では二値分類用です。スコアが負でも最大化して比較します。

公開API：LatentStraddle。betaの既定値は1.96です。

## 13.7 目的に応じた選択

| 目的 | 獲得関数 | 対象 |
| --- | --- | --- |
| 次のラベルが曖昧な点を知る | PredictiveEntropy | 平均予測確率 |
| クラスの接戦を知る | MarginUncertainty | 上位2クラスの確率差 |
| 成功確率の推定を改善したい | ProbabilityVariance | 確率の事後分散 |
| 追加ラベルから情報を得たい | BALD | 相互情報量 |
| 潜在ゼロ境界を調べたい | LatentStraddle | 潜在平均と標準偏差 |

予測したクラス1確率そのものを最大化したい場合、または可行確率を安全制約として使いたい場合は、別の最適化目的です。Active Learningの「学習価値」と混同しないでください。

## 13.8 q-batch：同時に複数の点を提案する場合

$q=2$ は次に提案する**2点の候補**を表し、2クラス分類の意味ではありません。

### 周辺スコア集約型

qPredictiveEntropy、qMarginUncertainty、qProbabilityVariance、qLatentStraddleは、各候補点のスコアを

$$
a_{\mathrm{marginal}}(x_{1:q})
=\frac{1}{q}\sum_{i=1}^{q}a(x_i)
$$

で平均します。この計算は**同時ラベル分布を評価しません**。近接した候補が互いに情報を重複させる場合にも、直接の重複ペナルティはありません。

### qBALD：同時情報利得

qBALDはラベルベクトル $Y_{1:q}$ と潜在ベクトル $F_{1:q}$ の相互情報量を使用します。

$$
I(Y_{1:q};F_{1:q}\mid\mathcal D)
=H(Y_{1:q}\mid\mathcal D)
-\mathbb E_{q(F_{1:q}\mid\mathcal D)}
 [H(Y_{1:q}\mid F_{1:q})].
$$

実装は同じ潜在事後サンプルから得た確率を用いて、二値ラベルの $2^q$ 通りの組合せを列挙します。$q=2$ なら (0,0)、(0,1)、(1,0)、(1,1) の4通りです。指数的な計算量のため、現行実装は**二値かつ $1\le q\le8$**に制限しています。

qBALDは点間の依存を考慮できますが、選択候補が必ず一定以上離れることまでは保証しません。単一点版のPredictiveEntropyなどは $q=1$ 専用であり、q-batch版とは区別します。

## 13.9 Ensemble、校正、信頼性

複数の分類器のクラス確率を重み付き平均してensemble予測分布を作れます。メンバーごとのEntropyと、平均予測分布のEntropyとの差を計算して不一致を測ることもできます。

ただし、任意の経験ensembleの不一致を、厳密に校正されたBayesian事後不確実性と同一視しないでください。異種モデルには共通の潜在関数がない場合もあり、LatentStraddleをそのまま適用できません。

Temperature Scaling等の確率校正は予測確率を変えるため、EntropyやMarginなどにも影響し得ます。一方、校正だけでは潜在GPの事後分布は更新されません。校正の品質は学習から独立したデータを使い、NLL、Brier Score、ECEなどで確認します。

## 13.10 robotorchanの公開APIとNotebook

- [分類獲得関数の実装](../../../src/robotorchan/acquisition/active_learning/classification.py)
- [39番：分類GPとActive Learning](../../../examples/notebooks/39_classification_active_learning.ipynb)
- [47番：拡張モデルとq-batch](../../../examples/notebooks/47_binary_classification_extensions.ipynb)
- [分類モデル一覧](../../models/classification.md)
- [二値分類の対応表](../../models/binary_classification_extension_matrix.md)

~~~python
from robotorchan.acquisition import BALD, PredictiveEntropy

entropy_acq = PredictiveEntropy(model)
bald_acq = BALD(model, num_samples=64)
~~~

上記のmodelは、必要な分類予測APIを持つ学習済みモデルを表します。単一点用獲得関数への候補形状は通常 batch_shape × 1 × d です。Mixedや各種制約との組合せについては、名前から全対応を推測せず実装・テストと対応表を確認してください。

### 理解度チェック

1. 予測確率が0.5なら、モデルの知識も必ず不足していますか？ → **いいえ**。成功確率が既知でもラベルは偶然に揺れます。
2. EntropyとBALDの違いは？ → **BALDはラベルを得たときの潜在モデルの情報利得を評価**します。
3. qPredictiveEntropyはラベルベクトルの同時Entropyですか？ → **いいえ**。周辺Entropyの平均です。
4. qBALDは何を追加で考えますか？ → **候補の同時予測分布と相互情報量**です。

### 参考文献

- Houlsby, N., Huszár, F., Ghahramani, Z., & Lengyel, M. (2011).
  *Bayesian Active Learning for Classification and Preference Learning*.
- Gal, Y., Islam, R., & Ghahramani, Z. (2017).
  *Deep Bayesian Active Learning with Image Data*. ICML.
