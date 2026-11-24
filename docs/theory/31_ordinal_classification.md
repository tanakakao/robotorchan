# 31. 順序Gaussian Process分類（Ordinal GP）

**順序付きクラス**（不良・要確認・良好など）のためのGP分類を扱います。クラス番号を連続値として回帰するのではなく、1つの潜在GPと閾値を用いるOrdered-probitモデルを導入します。前提となる[二値GP分類](24_classification.md)と、比較対象となる[マルチクラスGP分類](30_multiclass_classification.md)も参照してください。

## 順序GP分類：モデルと応用

### 1. なぜ通常のマルチクラス分類と区別するのか

製品の品質を「不良（0）・要確認（1）・良好（2）」と評価する場合、
クラスには **0 < 1 < 2 という順序**があります。一方、材質A・B・Cの分類には
通常そのような順序がありません。順序回帰は前者の順序情報を利用します。

| 観点 | 二値分類 | 通常のマルチクラスGP | 順序GP |
| --- | --- | --- | --- |
| クラス | 2種類 | 3種類以上、順序なし | 3種類以上、順序あり |
| 潜在関数 | 通常1個 | クラスごと | **1個** |
| 観測モデル | Bernoulli | Softmax / Categorical | Ordered probit / Categorical |
| クラス間の関係 | 合否の二択 | クラスごとの相対比較 | 共通尺度上の閾値で区切る |
| 解釈 | 成功しやすさ | 各クラスの傾向 | 潜在的な品質・重症度など |

順序があっても、ラベルの差が等間隔とは限りません。
たとえば「0と1の差」と「1と2の差」が同じ大きさとは仮定しません。
ラベルを単純な連続値とみなすGP回帰とは、この点で異なります。

### 2. Ordered-probitの確率モデル

一つの潜在関数にGP事前分布を置きます。

$$
f \sim \mathcal{GP}(m,k), \qquad
b_0=-\infty < b_1 < \cdots < b_{C-1} < b_C=+\infty.
$$

各入力で、潜在スコアに独立な標準正規ノイズを足した値が
どの区間へ入るかで観測クラスが決まる、と考えます。

$$
z(x)=f(x)+\epsilon,\quad \epsilon\sim\mathcal N(0,1),
\qquad y=c\iff b_c < z(x)\le b_{c+1}.
$$

したがって、$\Phi$ を標準正規分布の累積分布関数とすると

$$
P(y=c\mid f(x))
=\Phi(b_{c+1}-f(x))-\Phi(b_c-f(x)),
\qquad c=0,\ldots,C-1.
$$

最初のクラスでは $\Phi(b_0-f)=0$、
最後のクラスでは $\Phi(b_C-f)=1$ と読みます。
これにより各クラス確率は非負で、総和は1になります。

**直感：** $f(x)$ が大きいほど、観測値が高いクラスの区間に入る
確率が増えます。つまり潜在関数は「高い順序ラベルになりやすさ」を表し、
クラス確率そのものではありません。

3クラスで閾値が $b_1=0,\ b_2=0.7$ のとき、
$f(x)=-1$ ならクラス0に入る確率は $\Phi(1)\approx0.841$ です。
$f(x)=1$ ならクラス2に入る確率は
$1-\Phi(-0.3)\approx0.618$ です。
これは**潜在値が既知の場合の条件付き確率**であり、学習後の予測確率とは別です。

### 3. 閾値の単調性と識別可能性

閾値が入れ替わると中間クラスの確率が負になるため、
学習中も $b_1<\cdots<b_{C-1}$ を守る必要があります。

一つのOrdered-probitのパラメータ化では $b_1=0$ に固定し、
残りは正の増分 $\delta_j$ を累積して構築します。

$$
b_1=0,\quad
\delta_j=\mathrm{softplus}(a_j)+10^{-4}>0,\quad
b_{j+1}=\sum_{i=1}^{j}\delta_i.
$$

これは閾値の順序を機械的に保証し、潜在位置と閾値の
同時平行移動による曖昧さを抑えるパラメータ化です。
最初の閾値を0に置くのは**尺度の原点を決める約束**であり、
「品質ゼロ」など実世界の基準が0であることは意味しません。
probitノイズの分散は1に固定して尺度を定めています。

### 4. 学習：変分GPと非ガウス尤度

観測ラベルに対する尤度はOrdered-probitのCategorical分布です。
Gaussian尤度とGPの共役性は使えないため、
誘導点を持つ変分近似 $q(f)$ を学習します。

$$
\mathcal L =
\sum_{i=1}^{n}\mathbb E_{q(f_i)}
  [\log P(y_i\mid f_i)]
-\mathrm{KL}(q(u)\|p(u)).
$$

第一項は観測した順序ラベルをどれだけ説明できるか、
第二項は誘導変数の近似分布が事前分布から離れすぎないようにする項です。
期待対数尤度の計算には一変量Gauss–Hermite求積などによる数値積分を使用できます。

閾値とGPパラメータはデータから推定するため、
少数例・極端なクラス不均衡・閾値をまたがない学習データでは
適切な順序境界を推定できない場合があります。

### 5. 予測：潜在平均を閾値へ代入するだけでは不十分

予測位置での変分潜在事後が
$q(f_*)=\mathcal N(\mu_*,s_*^2)$ と近似されるとします。

$$
P(y_*=c\mid x_*,\mathcal D)
\approx \mathbb E_{q(f_*)}
[\Phi(b_{c+1}-f_*)-\Phi(b_c-f_*)].
$$

標準正規ノイズとGaussianな潜在事後を足すと、
分散が $1+s_*^2$ のGaussianになるので、

$$
\boxed{
P(y_*=c\mid x_*,\mathcal D)
=
\Phi\!\left(\frac{b_{c+1}-\mu_*}{\sqrt{1+s_*^2}}\right)
-\Phi\!\left(\frac{b_c-\mu_*}{\sqrt{1+s_*^2}}\right)
}
$$

と周辺予測確率を解析的に計算できます（この変分Gaussian近似の下で）。
潜在不確実性 $s_*^2$ が大きい場合に、
潜在平均 $\mu_*$ だけを使う確率とは異なります。

潜在GPの事後分布は、クラス確率そのものではありません。
予測クラス確率は $C$ 成分のベクトルで、
これが最大となるクラスを予測ラベルとして選択できます。
潜在事後からの同時サンプルをOrdered-probitへ通すと、
**条件付きクラス確率のサンプル**が得られます。
平均予測確率と確率サンプルを同じものとして扱いません。


### 5.1 予測確率の解析式を導く

導出の鍵は、潜在関数の不確実性と観測時のprobitノイズを**足し合わせる**ことです。学習後の近似潜在事後が \(f_*\mid\mathcal D\sim\mathcal N(\mu_*,s_*^2)\)、新しい観測ノイズが \(\epsilon_*\sim\mathcal N(0,1)\) で、これらが独立なら、

$$
z_*=f_*+\epsilon_*\mid\mathcal D
\sim\mathcal N(\mu_*,s_*^2+1).
$$

したがって閾値 \(b\) を下回る確率は

$$
P(z_*\le b\mid\mathcal D)
=
\Phi\left(\frac{b-\mu_*}{\sqrt{1+s_*^2}}\right).
$$

クラス \(c\) は \(b_c<z_*\le b_{c+1}\) で定義されるため、上式の**累積確率の差**がクラス確率になります。潜在事後の分散 \(s_*^2\) を無視して \(\Phi(b-\mu_*)\) と計算すると、確率が過度に確信的になる場合があります。

**近似の境界：** この式は潜在事後をGaussianとする近似の下で解析的です。学習により得た閾値・カーネルパラメータを固定しているので、それらのパラメータの事後不確実性を全て周辺化した厳密Bayesian予測という意味ではありません。

### 5.2 順序クラス集合の確率

品質ラベルが \(0,1,2\) で、クラス1以上を許容する場合は、許容集合 \(S=\{1,2\}\) の確率が

$$
P(C_*\in\{1,2\}\mid x_*,\mathcal D)
=
1-\Phi\left(\frac{b_1-\mu_*}{\sqrt{1+s_*^2}}\right)
$$

と書けます。これはクラス0の補集合に当たります。より一般に上位クラス集合 \(\{k,\ldots,C-1\}\) は同様の一つの累積確率で計算できます。

一方、許容クラスが連続した上位区間でない場合は、個々のクラスの予測確率を足して求めます。クラス集合の指定は業務上の許容条件を表し、順序モデルが自動的に決めるものではありません。回帰目的との合成は[第32章](32_heterogeneous_classification_constraints.md)を参照してください。

### 6. Active Learning：どのラベルを追加すると有益か

測定前の入力 $x$ で、以下の獲得関数を使えます。

- **予測エントロピー**：予測ラベルの曖昧さ
  $H[y\mid x,\mathcal D]=-\sum_c p_c\log p_c$ を測ります。
- **マージン不確実性**：上位2クラスの確率差が小さい点を重視します。
- **クラス確率の事後分散**：潜在事後から得るクラス確率の変動を測ります。
- **BALD**：予測エントロピーから潜在関数を固定した場合の
  条件付きエントロピーの期待値を引きます。

同じクラス確率の曖昧さでも、観測しても解消しにくい偶然性と、
学習により減らせる不確実性は異なります。EntropyとBALDが
同じ候補を選ぶとは限りません。

複数候補 $q>1$ の場合、
周辺エントロピー・マージン・確率分散のバッチ版は各候補の周辺スコアを平均します。
そのため候補間の重複情報を直接は考慮しません。
一方、同時バッチBALDは潜在関数の**相関した同時サンプル**から
同時ラベルベクトルの情報量をモンテカルロ推定します。
ラベルベクトルの列挙数は $C^q$ と指数的に増えます。
そのため実装では計算資源に応じた上限を設ける場合があります。

### 7. Mixed入力と順序ラベルの違い

Mixed入力を扱う順序GPは、**入力側**にカテゴリ列がある場合のモデルです。
入力のカテゴリ値（例えば設備A・B・C）には順序が不要です。
それに対して**出力側**のラベル（例えば正常・注意・異常）には順序があります。
両者を混同しないでください。

モデル化の段階ではカテゴリ入力列をカーネルへ知らせます。
候補生成の段階では、探索空間へ許されるカテゴリ値を指定します。
これらは異なる責務であり、自動的には連動しません。
入力への既知の線形制約は、
予測クラスが安全である確率の制約と同じではありません。

### 8. 読み方・評価上の注意

正解クラスに割り当てた確率をみるNLL、
全クラスの確率をみるBrier score、Accuracyに加え、
順序を無視しないMAE（クラス番号の絶対誤差）などを併用できます。
ただし、クラス番号に等間隔の意味がない場合は
MAEの解釈に注意してください。保留データとクラス別の誤りを確認し、
同じラベル予算でRandom Samplingなどと比較します。

数式から実装への対応は
[順序GP Guide](../guide/31_ordinal_classification.md)を、
学習・予測・候補提案・Mixed入力・2ラウンドの更新は
[31番 Ordinal GP Notebook](../../examples/notebooks/31_ordinal_classification_active_learning.ipynb)
を参照してください。

## 参考文献・関連資料

- [第24章：二値GP分類](24_classification.md)
- [第30章：マルチクラスGP分類](30_multiclass_classification.md)
- [順序分類の実装ガイド](../guide/31_ordinal_classification.md)
- [31番Notebook：順序GPとActive Learning](../../examples/notebooks/31_ordinal_classification_active_learning.ipynb)
- [61番Notebook：回帰×分類制約付きBO](../../examples/notebooks/61_heterogeneous_classification_constraints.ipynb)
- Chu, W., & Ghahramani, Z. (2005). *Gaussian Processes for Ordinal Regression*. Journal of Machine Learning Research 6, 1019–1041.
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*, Chapter 3.


## 高次元入力と順序プロビット

高次元入力 $x\in\mathbb R^D$ に固定した次元削減
$T:\mathbb R^D\to\mathbb R^d$ を適用し、単一の潜在関数
$f(T(x))$ をGPでモデル化します。潜在事後を
$\mathcal N(\mu(T(x)),s^2(T(x)))$ と近似すると、
閾値 $b_c$ による順序クラス確率は

$$
p(y=c\mid x,\mathcal D)=
\Phi\left(\frac{b_{c+1}-\mu(T(x))}{\sqrt{1+s^2(T(x))}}\right)
-\Phi\left(\frac{b_c-\mu(T(x))}{\sqrt{1+s^2(T(x))}}\right).
$$

PCAは入力分散を重視し、PLSは順序ラベルの数値符号化と入力の関係を
利用します。順序カテゴリの番号を等間隔と仮定したPLS変換が、
実際のカテゴリ間隔を反映するとは限りません。
Random Projectionは教師信号を用いずに縮約します。

ALEBOを用いる場合、GPへの入力は元の $x$ ではなく低次元探索座標 $z$ です。
実験条件 $x$ への写像はALEBO探索戦略が担います。
Mahalanobisカーネルの採用は順序プロビット尤度や閾値の定義を変更しません。

縮約後の識別性能、NLL、Brier score、順序ラベル誤差を保留データで比較します。
縮約器を評価データまで含めて学習するとデータ漏洩になるため、
訓練データでのみfitしてください。

**参考文献**：Chu & Ghahramani (2005), *Gaussian Processes for Ordinal
Regression*; Letham et al. (2020), *Re-examining Linear Embeddings for
High-Dimensional Bayesian Optimization*.


## SAAS型疎ARD事前分布による高次元順序分類

順序分類ではクラスごとにGPを置く代わりに、単一の潜在関数
$f(x)$ と順序閾値を用います。高次元入力については、
ARDカーネルの各逆長さ尺度 $\rho_j=1/\ell_j$ に
Half-Cauchy事前分布を置いて多数の特徴の寄与を小さくする方法を利用できます。

$$
\rho_j\sim\mathrm{HalfCauchy}(\tau).
$$

縮小事前分布は**入力の関連性**に作用します。
クラス間の順序や閾値を変更するものではありません。
順序プロビットのクラス確率は、従来どおり潜在Gaussian事後を
閾値尤度で積分して計算します。

SAAS型の順序分類モデル と MAP-SAAS型の順序分類モデル は、
二値分類で利用する既存の疎ARDカーネル生成器を再利用します。
学習は既存の順序分類用変分学習を用い、完全Bayesian SAASの
MCMCによるハイパーパラメータ積分は行いません。
この違いを明示した上で、通常の順序GPと保留データのNLL、
Brier score、順序ラベル誤差を比較することが重要です。

**参考文献**：Chu & Ghahramani (2005), *Gaussian Processes for Ordinal
Regression*; Eriksson & Jankowiak (2021), *High-Dimensional Bayesian
Optimization with Sparse Axis-Aligned Subspaces*.


## タスク間相関を持つ順序GP分類

設備や工程ごとに品質段階（不良・要確認・良好など）を予測する場合、
単一の順序潜在関数をタスク $t$ にも依存させることができます。
ICMカーネルは

$$
k((x,t),(x',t'))=k_x(x,x')B_{tt'}
$$

と定義し、タスク間の共分散 $B$ を学習します。
順序分類で利用する潜在GPの事後分布はこのカーネルから得られ、
予測ラベル確率は従来の順序プロビット閾値で計算します。

この構成では**全タスクが順序クラスの意味と閾値を共有**します。
同じクラス番号が設備ごとに異なる意味を持つ場合や、
タスク固有の閾値が必要な場合は別のモデル化が必要です。
また、Block形式は各入力で全タスクのラベルが揃う実験に適し、
欠測を含む観測にはLong形式が適します。

タスク共分散がKronecker積として表現可能であることと、
分類推論で厳密なKronecker専用アルゴリズムを使用することは同義ではありません。

**参考文献**：Chu & Ghahramani (2005), *Gaussian Processes for Ordinal
Regression*; Bonilla, Chai & Williams (2008), *Multi-task Gaussian Process Prediction*.


## Multi-Fidelity分類：測定条件の精度を入力として表す

実験のシミュレーション精度や測定回数など、忠実度（fidelity）$s$ を入力の一部として扱います。
通常の設計変数 $x$ と分けて、分類用の潜在GPに次の積共分散を使用します。

$$
k((x,s),(x',s')) = \sigma^2 k_x(x,x')k_s(s,s').
$$

$k_s$ にはBoTorchが提供する Exponential Decay カーネル
（反復忠実度）またはDownsamplingカーネル（データ忠実度）を使います。
これにより、異なる忠実度の観測が同じ潜在関数の推定に寄与します。

順序分類では単一の潜在関数にこの共分散を使い、順序閾値とProbit尤度で各クラスの確率を計算します。

Mixed入力ではカテゴリ変数にカテゴリ用共分散を、連続設計変数に連続共分散を使い、
忠実度の入力列はカテゴリ変数と混同せずに分離します。

忠実度が高くなるほど観測誤差が必ず小さくなるとは仮定しません。
また、これは**忠実度を含む分類サロゲートモデル**であり、
評価コストを使った候補選択やMulti-Fidelity Knowledge Gradientまでを
自動的に実行するものではありません。
忠実度のスケールと高忠実度での保留データ予測性能を別途確認する必要があります。

**参考文献**：Wu et al. (2020), *Practical Multi-fidelity Bayesian Optimization for Hyperparameter Tuning*;
Kandasamy et al. (2017), *Multi-fidelity Bayesian Optimisation with Continuous Approximations*.


## Deep Gaussian Processによる順序分類

通常のGPでは、観測データを条件付けた潜在関数の事後分布をガウス分布として扱うことができます。一方、Deep GPでは複数の確率的なGP写像を合成するため、最終的な潜在関数の分布は一般にガウス分布ではありません。

単一の深層潜在関数 $f(x)$ と順序閾値 $b_1<\cdots<b_{C-1}$ を使用します。順序確率は $p(y=c\mid f)=\Phi(b_{c+1}-f)-\Phi(b_c-f)$（端点は無限大）で求めます。学習ではOrdered Probit尤度に対する変分ELBOを用います。

予測時のクラス確率は、潜在事後のサンプル $f^{(s)}(x)$ から

$$
\hat p(y=c\mid\mathcal D,x)=
\frac{1}{S}\sum_{s=1}^S p(y=c\mid f^{(s)}(x))
$$

として推定します。ここで $S$ はモンテカルロサンプル数です。潜在事後平均にリンク関数を適用するだけでは、一般にこの期待値と一致しません。

クラス確率の分散やBALDも同じサンプルから推定できますが、サンプル数による誤差があります。今回のモデルは既存の 既存の深層ガウス過程の確率的階層を再利用し、ガウス事後を前提とする通常のExact GP獲得関数と同一視しません。

**参考文献**：Damianou and Lawrence (2013), *Deep Gaussian Processes*; Salimbeni and Deisenroth (2017), *Doubly Stochastic Variational Inference for Deep Gaussian Processes*; Chu and Ghahramani (2005), *Gaussian Processes for Ordinal Regression*.


## 無限幅ニューラルネットワーク由来のGPカーネル（NNGP）

無限幅の全結合ReLUニューラルネットワークを考えると、適切な重みの事前分布と幅の極限の下で、その関数の事前分布はガウス過程になります。これがNeural Network Gaussian Process（NNGP）です。有限幅のニューラルネットワークを学習するという意味ではありません。また、複数の確率的なGP層を組み合わせるDeep GPとも異なります。

初期共分散の例として、設計変数 $x,z$ に対し

$$
K^{(0)}(x,z)=\sigma_b^2+\frac{\sigma_w^2}{d}x^\top z
$$

を用います。ReLUの各層で、角度 $\theta$ と両入力の分散から解析的なReLU共分散写像を繰り返して $K^{(\ell+1)}$ を得ます。分類には、この共分散を潜在GPの事前分布として与えます。

順序分類では単一の潜在GPにNNGP型共分散を与え、Ordered Probit尤度と閾値を用いて順序クラス確率を推定します。

Mixed入力では、連続設計変数のNNGP型共分散と、カテゴリ変数用の共分散を組み合わせます。カテゴリコードを無条件に連続変数と見なすものではありません。

NNGPカーネルを使用しても分類尤度は非ガウスであるため、分類の事後計算は変分近似です。NNGPの無限幅極限という理論と、有限の学習データから行う近似推論は区別してください。

**参考文献：** Lee et al. (2018), *Deep Neural Networks as Gaussian Processes*; Matthews et al. (2018), *Gaussian Process Behaviour in Wide Deep Neural Networks*; Williams (1997), *Computing with Infinite Networks*.


## Contextual・Hierarchical GP分類

実験条件・設備・製造ラインなどの**コンテキスト**が変わると、同じ設計変数でも分類確率が変わることがあります。データが少ないコンテキストにも学習結果を共有できるよう、共通の潜在GP成分とコンテキスト固有の偏差成分を加法的にモデル化します。

設計変数 $x$ とコンテキスト識別子 $c$ に対して、

$$
k((x,c),(x',c'))=k_0(x,x')+k_1(x,x')k_c(c,c')
$$

を考えます。$k_c$ は識別子の関連性を表すIndexKernelであり、数値IDを連続量の距離として解釈するものではありません。共通項 $k_0$ が複数コンテキスト間の統計的共有を担い、相互作用項 $k_1 k_c$ がコンテキスト別の偏差を表します。

さらに、グループ $g$ とその下位サブグループ $s$ がある場合は

$$
k((x,g,s),(x',g',s')) =
k_0(x,x')+k_g(x,x')k_G(g,g')+k_s(x,x')k_S(s,s')
$$

とし、全体・グループ・サブグループの3層を表現します。ここでは**各サブグループIDが一つのグループにのみ属する**ことを前提とし、入力で検証します。これは厳密な因果的階層の推定ではなく、階層を反映したGP共分散の指定です。

順序分類では、単一の潜在GPにこの共分散構造を与え、順序閾値を持つProbit尤度でクラス確率を推定します。 分類尤度は非ガウスであるため、潜在GPの学習は変分近似です。

既知のコンテキスト下で候補点を選ぶことと、未観測のコンテキストに汎化させることは区別してください。新しいIDを用いる際の扱いはモデル仕様によって制限されます。

**参考文献：** Bonilla, Chai and Williams (2008), *Multi-task Gaussian Process Prediction*; Álvarez, Rosasco and Lawrence (2012), *Kernels for Vector-Valued Functions: a Review*; Gelman et al. (2013), *Bayesian Data Analysis*（階層モデリングの背景）。


## 観測ラベルノイズを考慮した分類

ラベルの誤記録がある場合、潜在的な**真のクラス** $y$ と、学習データに記録された**観測ラベル** $\tilde y$ を区別します。誤記録を表す遷移行列 $T$ を

$$
T_{ij}=P(\tilde y=j\mid y=i),\qquad
T_{ij}\geq 0,\quad\sum_j T_{ij}=1
$$

と定義します。各行は真のクラスごとに観測ラベルへ移る条件付き確率を表すため、行和が1です。クラス $i$ の真の予測確率を $p_i(x)$ とすると、観測ラベルの予測確率は

$$
P(\tilde y=j\mid x,\mathcal D)
=\sum_{i=0}^{C-1}p_i(x)T_{ij}
$$

です。対称ノイズ率 $\epsilon$ なら、対角要素は $1-\epsilon$、非対角要素は $\epsilon/(C-1)$ です。より一般には非対称な行列を指定できます。

**学習の際に遷移行列を使うことが重要です。** 推論後の確率に行列を掛けるだけでは、ノイズの影響を含む学習にはなりません。モデルの潜在関数から得たクラス確率 $p_i(x;f)$ に対し、観測ラベルの対数尤度

$$
\log P(\tilde y=j\mid f,x)
=\log\left(\sum_i p_i(x;f)T_{ij}\right)
$$

を使い、変分ELBOを最大化します。

真の順序クラス確率は単一潜在GPとOrdered Probitの閾値から定義し、観測ラベルの確率に遷移行列を掛けてGauss–Hermite求積で期待対数尤度を近似します。

遷移行列が正しく指定されているかどうかにモデルの解釈は依存します。ノイズ率の識別・推定は別問題であり、固定の既知遷移行列を指定しただけで未知の系統的なラベル誤りを解決できるわけではありません。データ分布や評価条件に応じてノイズ機構の妥当性を検討してください。

**参考文献**：Frénay and Verleysen (2014), *Classification in the Presence of Label Noise: A Survey*; Natarajan et al. (2013), *Learning with Noisy Labels*; Patrini et al. (2017), *Making Deep Neural Networks Robust to Label Noise: A Loss Correction Approach*.


## 入力不確実性：候補点の摂動とカテゴリ分布

分類モデルへの入力 $x$ が完全には制御できない場合、設計した候補値 $x$ と実際に評価される入力 $\tilde x$ を分けて考えます。本節は**候補入力の不確実性**を対象とし、学習時の説明変数に含まれる測定誤差を修正する仕組みとは異なります。

連続入力で、$\tilde x=x+\varepsilon$、$\varepsilon\sim\mathcal N(0,\Sigma_x)$ と仮定します。データ $\mathcal D$ から学習したモデルのクラス予測確率を $p(y=c\mid z,\mathcal D)$ とすると、

$$
p_{\mathrm{uncertain}}(y=c\mid x,\mathcal D)
=\int p(y=c\mid x+\varepsilon,\mathcal D)
p(\varepsilon)\,d\varepsilon
\approx\frac{1}{M}\sum_{m=1}^M
p(y=c\mid x+\varepsilon_m,\mathcal D)
$$

によって不確実性を周辺化します。**GP潜在関数の平均を先に入力摂動で平均してから尤度に入れる方法ではありません。** 非線形な分類リンクを経由した確率を積分する必要があります。共分散 $\Sigma_x$ は標準偏差（各軸独立）または半正定値の共分散行列として指定できます。

カテゴリ値が不確かな場合、カテゴリの有限集合 $\{a_k\}$ と確率 $w_k$ が既知であれば、

$$
p_{\mathrm{uncertain}}(y=c\mid x,\mathcal D)
=\sum_{k}w_k\,
p(y=c\mid x_{\mathrm{cat}}=a_k,\mathcal D)
$$

を厳密な**有限和**で計算できます。カテゴリIDは連続的なガウス摂動の対象にせず、既存のMixedカーネルのまま扱います。

単一潜在GPのOrdered Probitにより計算した各順序クラスの確率を、入力不確実性について周辺化します。 対象が入力不確実性込みのクラスエントロピーである場合、周辺化後の確率からエントロピーを計算します。通常の候補入力に対して評価した獲得関数をそのまま呼び出しても、入力不確実性は自動的には考慮されません。

**参考文献**：McHutchon and Rasmussen (2011), *Gaussian Process Training with Input Noise*; Nogueira et al. (2016), *Bayesian Optimization with Unknown Constraints*（候補評価の不確実性とは異なる関連領域）; Bogunovic et al. (2018), *Adversarially Robust Optimization with Gaussian Processes*（ロバスト目的の別の定式化）。


## Ensemble・非GP分類：予測確率とモデル間の不確実性

GP以外の教師あり分類器（Random Forest、Extra Trees、Gradient Boosting、Histogram Gradient Boosting）は、分類の入力とラベルから直接クラス確率を学習します。これらはガウス過程の潜在関数のPosteriorを持ちません。単一の推定器が返すクラス確率のエントロピーと、モデルパラメータについての**認識論的不確実性**は区別する必要があります。実装では単一の非GP推定器に対して虚偽の `latent_posterior()`、`probability_posterior()` やBALD値を生成しません。

予測確率のアンサンブルでは、モデル $m=1,\ldots,M$ のクラス確率 $p_m(y=c\mid x)$ を重み $w_m$（$w_m\geq0$、$\sum_m w_m=1$）で平均します。

$$
\bar p_c(x)=\sum_{m=1}^{M}w_m p_m(y=c\mid x)
$$

モデル間の確率分散は

$$
V_c(x)=\sum_m w_m\bigl(p_m(y=c\mid x)-\bar p_c(x)\bigr)^2
$$

です。ラベルの予測エントロピー $H(\bar p)$ と、個別モデルのエントロピーの加重平均との差

$$
H(\bar p)-\sum_m w_m H(p_m)
$$

は**アンサンブルメンバー間の意見の不一致**を測ります。これはメンバーを確率的に選ぶという近似のもとでの相互情報量であり、必ずしも真のベイズPosteriorのBALDとは一致しません。GPと非GPの混合アンサンブルでは、単一の共通潜在GPを定義していません。

順序ラベルを使う非GP分類器もクラス確率を返しますが、これらのツリーモデルは通常のカテゴリ分類として学習するため、Ordered Probitの閾値構造や単調な順序制約を学習するモデルではありません。 クラス確率が真に校正されているかは別途検証が必要です。

**参考文献：** Breiman (2001), *Random Forests*; Geurts, Ernst and Wehenkel (2006), *Extremely randomized trees*; Lakshminarayanan, Pritzel and Blundell (2017), *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*; Dietterich (2000), *Ensemble Methods in Machine Learning*.


## 確率校正、Conformal予測集合、意思決定とActive Learning

GP分類の予測確率は、モデルの尤度・潜在GP事後を通じた確率ですが、その値が頻度論的に校正されているとは限りません。確率ベクトル $p(x)$ に温度 $T>0$ を適用する場合、

$$
p_c^{(T)}(x)=\frac{p_c(x)^{1/T}}{\sum_j p_j(x)^{1/T}}
$$

とします。$T$ は学習に使わなかった検証データで負の対数尤度を最小化して推定します。GPの潜在Posteriorは変更しませんが、クラス確率のエントロピーと確率サンプルの分布は変化します。

Split Conformalでは、独立に保留した校正集合の真のクラス $y_i$ に対し、非適合度 $s_i=1-p_{y_i}(x_i)$ を計算します。校正データ数を $n$、要求する誤り率を $\alpha$ として、

$$
\widehat{q}=\mathrm{Quantile}_{\lceil(n+1)(1-\alpha)\rceil/n}(s_1,\ldots,s_n)
$$

を有限標本補正付きで選び、予測集合を

$$
\Gamma(x)=\{c:1-p_c(x)\leq\widehat{q}\}
$$

と定義します。基礎モデルの学習や温度校正に使っていないデータでConformal校正を行い、校正例と将来のテスト例が交換可能（exchangeable）であれば、標準的なsplit conformalの**周辺被覆率**が保証されます。特定クラスごとの被覆率、条件付き被覆率、分布シフト下の保証ではありません。

誤分類コストを $C_{a,y}$（行：採用する行動 $a$、列：真のクラス $y$）と定義します。条件付き期待損失およびBayes決定は

$$
R(a\mid x)=\sum_y C_{a,y}p_y(x),\qquad
a^\star(x)=\arg\min_a R(a\mid x)
$$

です。確率の最大クラスを選ぶことと最小期待損失の判断は一般に異なります。

順序分類では、絶対距離のコストなら確率分布の中央値に対応する決定、二乗距離なら事後平均に最も近いクラスを選ぶ決定になります。

Active Learningは目的別に使い分けます。PredictiveEntropyは予測ラベルの不確実性、BALDは認識論的な不一致、qPredictiveEntropyはバッチ内の**周辺エントロピー平均**を評価します。Bayes決定の最小期待リスクが高い点をプールから選ぶことは可能ですが、これはリスクの高い点を調べるヒューリスティクスであり、観測後の期待リスク減少（value of information）を計算するものではありません。またConformal集合の大きさは直接的な情報利得ではありません。

**参考文献：** Guo et al. (2017), *On Calibration of Modern Neural Networks*; Vovk, Gammerman and Shafer (2005), *Algorithmic Learning in a Random World*; Angelopoulos and Bates (2023), *Conformal Prediction: A Gentle Introduction*; Houlsby et al. (2011), *Bayesian Active Learning for Classification and Preference Learning*; Berger (1985), *Statistical Decision Theory and Bayesian Analysis*.
