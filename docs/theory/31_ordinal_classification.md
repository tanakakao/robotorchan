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
\delta_j=\operatorname{softplus}(a_j)+10^{-4}>0,\quad
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
-\operatorname{KL}(q(u)\|p(u)).
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
