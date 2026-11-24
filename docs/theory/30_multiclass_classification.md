# 30. マルチクラスGaussian Process分類

3クラス以上の**順序を仮定しない**分類を扱います。二値分類との違い、softmax尤度、変分推論、クラス確率、Active Learning、q-batchの情報量と評価を段階的に解説します。二値GP分類の基礎は[第24章](24_classification.md)、順序情報を使うモデルは[第31章](31_ordinal_classification.md)を参照してください。

## 1. 多クラスの確率モデル

$C>2$ の場合、潜在関数のベクトル
$f_0(x),\ldots,f_{C-1}(x)$ とsoftmaxリンクを持つカテゴリ尤度を使う方法があります。

$$
p(y=c\mid f(x))
=
\frac{\exp(f_c(x))}{\sum_{j=0}^{C-1}\exp(f_j(x))}.
$$

潜在関数は事前に独立でも、多出力共分散を通じて相関していても構いません。
予測クラス確率は潜在関数の**同時**事後分布についてsoftmaxを積分します。

$$
p(y=c\mid x,\mathcal D)
=
\mathbb E_{p(f(x)\mid\mathcal D)}
\left[
\frac{\exp(f_c(x))}{\sum_{j=0}^{C-1}\exp(f_j(x))}
\right].
$$

softmax尤度は全潜在ロジットに同じ定数を加えても不変であり、
識別可能性に注意が必要です。多クラス推論と意思決定は、
一般に単一の二値閾値に還元できません。

## 2. Softmaxの識別可能性

softmaxは全クラスの潜在値に同じ定数を加えても変化しません。
そのため、クラス確率は識別可能でも潜在ロジットの絶対的な
基準は識別できない場合があります。基準クラスを固定する方法や
適切な事前分布を用いる方法が考えられます。

多クラスの予測確率は潜在関数の**同時分布**を通じて計算されるため、
各クラスの潜在平均にsoftmaxを適用するだけでは、一般に
事後不確実性を正しく反映できません。


### 2.1 非共役な分類を変分推論で扱う理由

入力とラベルのデータを \(\mathcal D=\{(x_i,y_i)\}_{i=1}^{n}\)、潜在関数のクラス別ベクトルを \(\boldsymbol f(x)\) とします。GaussianなGP事前分布とCategorical（softmax）尤度の積から得られる真の事後分布は、一般にGaussian分布に戻りません。したがって、通常のGP回帰と同じ閉形式の条件付きGaussian事後分布だけでは分類ラベルを扱えません。

誘導変数 \(\boldsymbol u\) に変分分布 \(q(\boldsymbol u)\) を置き、これが誘導する \(\boldsymbol f\) の周辺分布 \(q(\boldsymbol f)\) を使います。典型的な変分目的は

$$
\mathcal L
=
\sum_{i=1}^{n}
\mathbb E_{q(\boldsymbol f(x_i))}
\left[
\log P(y_i\mid \boldsymbol f(x_i))
\right]
-
\operatorname{KL}\bigl(q(\boldsymbol u)\,\|\,p(\boldsymbol u)\bigr).
$$

前項は観測ラベルへの対数確率を高くし、後項は変分分布がGP事前分布から過度に離れることを抑えます。softmaxの非線形性によって期待対数尤度は一般に解析積分できず、モンテカルロ法などで近似します。

**注意：** クラスごとの潜在GPを独立に置いても、softmax後のクラス確率は互いに独立ではありません。全クラスの確率が必ず1に合計されるためです。潜在GPの依存構造はモデルの仮定であり、すべてのマルチクラスGPが独立な潜在GPを使うという意味ではありません。

### 2.2 予測確率は事後期待値である

学習後の予測確率は

$$
p_c(x)=
\int
\frac{\exp(f_c(x))}
{\sum_{j=0}^{C-1}\exp(f_j(x))}
q(\boldsymbol f(x))\,d\boldsymbol f(x).
$$

潜在事後サンプル \(\boldsymbol f^{(s)}(x)\sim q(\boldsymbol f(x))\) を \(S\) 個得られるなら、

$$
\widehat p_c(x)
=
\frac{1}{S}\sum_{s=1}^{S}
\frac{\exp(f_c^{(s)}(x))}
{\sum_{j=0}^{C-1}\exp(f_j^{(s)}(x))}
$$

と近似します。ここで、\(\operatorname{softmax}(\mathbb E[\boldsymbol f])\) と \(\mathbb E[\operatorname{softmax}(\boldsymbol f)]\) は一般に一致しません。モンテカルロサンプルを増やすと積分の近似誤差は低減し得ますが、学習データ不足やモデルの仮定違反から生じる誤差まで解消されるわけではありません。

### 2.3 予測EntropyとBALDの違いを式で確認する

予測確率 \(p_c(x)\) を用いる予測エントロピーは

$$
H(C\mid x,\mathcal D)
=-\sum_{c=0}^{C-1}p_c(x)\log p_c(x).
$$

これは**次に得るラベルの不確実性全体**を表します。これに対してBALDの情報利得は

$$
I(C;\boldsymbol f\mid x,\mathcal D)
=
H(C\mid x,\mathcal D)
-
\mathbb E_{q(\boldsymbol f(x))}
\left[
H(C\mid x,\boldsymbol f(x))
\right].
$$

右辺の第2項は潜在関数の値を仮に知っても残るラベルの偶然性です。したがって、Entropyが高くても、モデルを学ぶ余地が小さければBALDは低くなり得ます。多数候補の**同時**情報利得では、候補間の潜在GPの相関も扱う必要があります。候補ごとのEntropyの平均は、この同時情報利得とは異なります。

## 3. 予測・学習・Active Learningへの拡張

前節までの式では、クラスラベルを `0, …, C−1`、潜在関数を \(f_0(x),\ldots,f_{C-1}(x)\) として統一しました。Softmaxによるクラス確率の定義、変分ELBO、予測時の事後積分、EntropyとBALDの相違は第1・2節で導出しています。

以下では、それらの確率をどのように解釈し、候補を選び、予測精度や確率校正を評価するかを具体例で説明します。

### 例：3クラスの確率をどう読むか

入力条件 `x` に対し、予測確率が
`p=(0.7, 0.2, 0.1)` だったとします。
この値は、次に観測するラベルがクラス0・1・2になる予測確率です。
潜在GPの平均が `(0.7, 0.2, 0.1)` という意味ではありません。

潜在値 `f_c(x)` は任意の実数であり、
softmaxを通して0〜1の確率へ変換されます。
潜在GPが不確かな場合、異なる潜在サンプルごとに異なる確率が得られます。
それらを平均したものが予測確率のモンテカルロ近似です。

### 例：予測エントロピーとBALDが異なる理由

予測エントロピーは「次のラベルが何になるか分からない度合い」です。
たとえば複数クラスの確率がほぼ同じなら大きくなります。

一方、BALDは「追加観測により潜在モデルについて何が分かるか」を
測る情報量です。数式では次のように表せます。

$$
I(y;f\mid x,D)
=H(y\mid x,D)-E_{f\mid D}[H(y\mid f,x)].
$$

右辺の第1項は予測ラベルのエントロピー、
第2項は潜在関数が分かっている場合にも残るラベルの曖昧さです。
第2項を差し引くことで、観測しても解消しにくい曖昧さと
モデルの不確実性を区別します。

### q-batchと計算量

1点ずつ選ぶ `q=1` と、複数点を同時に選ぶ `q>1` では
情報量の扱いが異なります。
同じような候補を2点選んでも、得られる情報が単純に2倍になるとは限りません。
joint batch BALDはラベルベクトルの同時分布を使って評価します。

ただし `C` クラス、`q` 候補ならラベルベクトルは `C**q` 通りです。
3クラス・2候補は9通り、3クラス・5候補は243通りです。
具体的な実装では列挙数に上限を設ける場合があります。
これは数学的限界ではなく、計算資源とのトレードオフです。

### 分類モデルの検証で注意すること

- Accuracyは最大確率のクラスが正しい割合を測りますが、確率の校正は測れません。
- NLLは正解クラスへ割り当てた確率の良し悪しを評価します。
- Brier scoreは予測確率と正解のone-hotベクトルの差を評価します。
- Active Learningの優劣は、同じラベル予算での性能推移を複数シードで比較する必要があります。

これらの指標は異なる性質を測るため、1つだけで性能を判断しないでください。


### 二値分類から何が変わるのか

二値分類は `y ∈ {0,1}` で、潜在関数 `f(x)` をリンク関数に通して
Bernoulli分布の成功確率を表します。マルチクラス分類は
`y ∈ {0,…,C−1}` で、各クラスに対応する潜在関数
`f_0(x),…,f_{C−1}(x)` からCategorical分布を作ります。

| 観点 | 二値分類 | マルチクラス分類 |
| --- | --- | --- |
| 観測ラベル | 0または1 | 0からC−1の整数 |
| 潜在関数 | 通常は1つ | クラスごとに潜在GPを置く設計では1クラスに1つ |
| リンク関数 | Sigmoid等 | Softmax |
| 尤度 | Bernoulli | Categorical（SoftmaxLikelihood） |
| 出力確率 | 1つの成功確率で十分 | C個の確率の合計が1 |
| 不確実性 | 最大エントロピーはlog 2 | 最大エントロピーはlog C |

クラスが3つのとき、潜在値が `(2, 1, 0)` なら、
softmaxは概ね `(0.665, 0.245, 0.090)` です。
潜在値は確率ではなく、全クラスの**相対的な差**が重要です。
すべての潜在値に同じ定数を足してもsoftmaxの結果は変わりません。

### なぜ変分推論とモンテカルロサンプリングが必要なのか

GPは潜在関数に事前分布を置きます。しかしCategorical尤度とGP事前分布の
組合せでは、観測後の潜在関数の事後分布を一般に解析的に計算できません。
そのため、誘導点を使う変分近似によって潜在GPの分布を学習します。

クラスごとに独立した潜在GPを置くモデルでは、
変分GPを用い、クラスの潜在サンプルをsoftmaxに通した
対数尤度とKL正則化項を使って学習します。
**独立した潜在GP**とは、クラスごとの潜在過程の近似構造についての
仮定であり、softmax後のクラス確率が独立という意味ではありません。
クラス確率は合計1の制約で結び付いています。

予測時に求めたいのは
次の事後期待値です。

$$
p(y=c\mid x,D)=\mathbb{E}_{f\mid D}[\operatorname{softmax}(f(x))_c].
$$
softmaxは非線形なので一般に
期待値のsoftmaxとsoftmaxの期待値は一致しません。
予測確率は潜在GPのサンプルを複数回取り、
softmax後の確率を平均して近似します。
モンテカルロサンプル数を増やすとモンテカルロ誤差は通常小さくなりますが、
モデル自体の誤差が消えるわけではありません。

### 予測が曖昧な点と、学習によって改善できる点

3クラスの予測確率が `(1/3, 1/3, 1/3)` に近いと、
予測エントロピーは最大の `log 3` に近づきます。
ただし、その曖昧さが観測ノイズによるものなら、
ラベルを追加しても必ずしも改善しません。

BALDは
予測ラベルのエントロピーから、潜在関数が既知のときの
条件付きエントロピーの事後期待値を引いて、
モデルの潜在関数に関する情報を得られる候補を優先します。
したがってEntropyとBALDで提案順位が違っても不具合ではありません。
少数データ、強いノイズ、クラス不均衡では特に違いを確認してください。

### Active Learningの実験サイクルと検証

1. ラベル付きデータでモデルを学習する。
2. 未測定候補の獲得関数値を計算し、候補を提案する。
3. 提案した条件で実験・検査を行い、初めてラベルを取得する。
4. 取得したラベルを学習データへ追加し、再学習する。
5. 同じラベル予算でRandom Sampling等と比較する。

評価用データはこのサイクルに混ぜず、最後まで独立に保ちます。
Accuracyだけでなく、NLL・Brier score・クラス別Recallも確認してください。
実行できる2ラウンドの例は
[30番Notebook](../../examples/notebooks/30_multiclass_classification_active_learning.ipynb)
を参照してください。




### 3.1 確率校正とモデル評価の具体式

保留データ \((x_i,y_i)\) に対して、モデルがクラス \(c\) の確率 \(p_{ic}\) を予測したとします。Accuracyは最大確率クラスの一致率ですが、予測確率そのものの妥当性までは測れません。

多クラス負の対数尤度（NLL）は

$$
\operatorname{NLL}
=-\frac1n\sum_{i=1}^{n}\log p_{i,y_i}
$$

であり、真のクラスへ極端に小さい確率を与えた予測を強く罰します。多クラスBrier scoreは

$$
\operatorname{Brier}
=\frac1n\sum_{i=1}^{n}\sum_{c=0}^{C-1}
\left(p_{ic}-\boldsymbol1[y_i=c]\right)^2
$$

のように定義できます（\(C\) で正規化する流儀もあるため、比較時は定義を明記します）。特に少数クラスでは、全体のAccuracyに加えてクラス別Recallや混同行列を確認してください。

## 4. 関連資料と参考文献

- [第24章：二値GP分類](24_classification.md)
- [第31章：順序GP分類](31_ordinal_classification.md)
- [マルチクラス分類の実装ガイド](../guide/30_multiclass_classification.md)
- [30番Notebook：マルチクラス分類とActive Learning](../../examples/notebooks/30_multiclass_classification_active_learning.ipynb)
- [61番Notebook：回帰×分類制約付きBO](../../examples/notebooks/61_heterogeneous_classification_constraints.ipynb)
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*, Chapter 3.
- Hensman, J., Matthews, A. G. de G., & Ghahramani, Z. (2015). *Scalable Variational Gaussian Process Classification*. AISTATS.
- Houlsby, N., Huszár, F., Ghahramani, Z., & Lengyel, M. (2011). *Bayesian Active Learning for Classification and Preference Learning*.


## 高次元入力：固定次元削減とALEBO

入力 $x\in\mathbb R^D$ が高次元であると、GPの距離尺度の推定は難しくなります。
固定した変換 $T:\mathbb R^D\to\mathbb R^d$（$d<D$）を用いる方法では、
クラス $c$ の潜在関数を $f_c(T(x))$ とモデル化します。PCAは入力の分散、
PLSは入力とラベルを数値化した応答との共分散、Random Projectionはランダムな
線形写像を利用します。**PLSで整数クラス番号を連続量として扱うと、クラス番号の
割り当てに依存する**ため、名義尺度の多クラス分類では注意が必要です。
クラス番号の置換で結果が変わる可能性があり、PCAやランダム射影との比較が重要です。

マルチクラス分類の確率は、低次元化しても
$$
p(y=c\mid x,\mathcal D)
=\mathbb E_{f\mid\mathcal D}
\left[\frac{\exp f_c(T(x))}{\sum_{j=0}^{C-1}\exp f_j(T(x))}\right]
$$
であり、潜在GPの平均値にsoftmaxを適用するだけでは一般に一致しません。

ALEBOは固定の次元削減モデルとは逆向きの発想で、低次元探索座標
$z\in\mathbb R^d$ から元の探索空間へ候補を写像します。
ALEBO分類モデルが受け取るのは **$z$** であり、元の $x$ ではありません。
Mahalanobisカーネルは埋め込み座標での幾何を表します。
分類モデルが探索空間への逆写像や実験制約を自動で処理するわけではありません。

固定変換が重要な識別情報を捨てると、GPの学習を改善しても失われた情報は
復元できません。独立した評価データでNLL・Brier score・クラス別Recallを比較し、
元の入力次元、縮約次元、変換の学習データ、乱数シードを記録します。

**参考文献**：Binois, Ginsbourger & Roustant (2018), *On the Choice of the
Low-Dimensional Domain for Global Optimization*; Letham et al. (2020),
*Re-examining Linear Embeddings for High-Dimensional Bayesian Optimization*.


## SAAS型疎ARD事前分布による高次元マルチクラス分類

高次元入力 $x\in\mathbb R^D$ で全特徴が同じ程度に重要とは限りません。
ARD Matérnカーネルは各入力次元に長さ尺度 $\ell_j$ を持ち、
$\ell_j$ が大きい次元では予測がその特徴に対して変化しにくくなります。
SAAS型モデルでは逆長さ尺度 $\rho_j=1/\ell_j$ に原点近くを好む
Half-Cauchy事前分布を置き、多くの入力次元の効果を小さくすることを促します。

$$
\rho_j\sim\mathrm{HalfCauchy}(\tau),\qquad j=1,\ldots,D.
$$

ここで $\tau$ は固定する縮小スケールです。
本実装は既存の二値分類のSAAS型／MAP-SAAS型カーネル生成器を再利用し、
**各クラスの潜在変分GPに独立したカーネル**を配置します。
Softmax分類確率は各潜在GPの事後サンプルをSoftmaxへ通して求めます。
固定したスケールのHalf-Cauchy事前分布を使用することは、
グローバル・ローカル尺度を階層的に積分する完全Bayesian SAASと同義ではありません。
また、MAP-SAAS型のマルチクラス分類モデル は既存MAP-SAAS型のカーネル構成を
再利用した変分分類モデルであり、完全Bayesian MCMC推論を実行しません。

実験では重要な変数の識別が改善するか、保留データでのNLL、
Brier score、クラス別Recallを比較してください。少数データでは特徴の
選択が不安定になる可能性があります。

**参考文献**：Eriksson & Jankowiak (2021), *High-Dimensional Bayesian
Optimization with Sparse Axis-Aligned Subspaces*;
Rasmussen & Williams (2006), *Gaussian Processes for Machine Learning*.


## タスク間相関を持つマルチクラスGP

複数の設備や製造ラインなどで、同一のクラス集合を予測するとします。
単純にタスクごとに独立した分類器を作ると、他タスクの観測情報が共有されません。
ICM（intrinsic coregionalization model）は、入力 $x$ とタスクID $t$ の組に対して

$$
k((x,t),(x',t'))=k_x(x,x')B_{tt'}
$$

という積カーネルを使用します。ここで $B$ は半正定値のタスク共分散行列です。
各クラスの潜在関数にはタスク共分散を持つGPを置きますが、
クラスラベルの確率は**タスクを固定した潜在値のsoftmax**を周辺化して得ます。
タスク相関とクラス間のsoftmax依存は異なる概念です。

Long形式では、1行に通常の入力特徴と整数のタスクIDを含めます。
Block形式では、各入力条件について全タスクの観測ラベルを並べます。
Block形式が成立するのは同じ入力を各タスクで観測した場合です。
欠損タスクがあるならLong形式を使います。

タスク共分散を持つICMカーネルはKronecker形の共分散を表せますが、
**分類用の変分推論がKronecker専用の高速線形代数を用いているとは限りません**。
実装の推論アルゴリズムとカーネル構造を区別してください。

**参考文献**：Bonilla, Chai & Williams (2008), *Multi-task Gaussian Process Prediction*;
Álvarez, Rosasco & Lawrence (2012), *Kernels for Vector-Valued Functions: A Review*.


## Multi-Fidelity分類：測定条件の精度を入力として表す

実験のシミュレーション精度や測定回数など、忠実度（fidelity）$s$ を入力の一部として扱います。
通常の設計変数 $x$ と分けて、分類用の潜在GPに次の積共分散を使用します。

$$
k((x,s),(x',s')) = \sigma^2 k_x(x,x')k_s(s,s').
$$

$k_s$ にはBoTorchが提供する Exponential Decay カーネル
（反復忠実度）またはDownsamplingカーネル（データ忠実度）を使います。
これにより、異なる忠実度の観測が同じ潜在関数の推定に寄与します。

マルチクラスでは各クラスの潜在関数にこの共分散を使い、クラス確率はSoftmaxで統合します。

Mixed入力ではカテゴリ変数にカテゴリ用共分散を、連続設計変数に連続共分散を使い、
忠実度の入力列はカテゴリ変数と混同せずに分離します。

忠実度が高くなるほど観測誤差が必ず小さくなるとは仮定しません。
また、これは**忠実度を含む分類サロゲートモデル**であり、
評価コストを使った候補選択やMulti-Fidelity Knowledge Gradientまでを
自動的に実行するものではありません。
忠実度のスケールと高忠実度での保留データ予測性能を別途確認する必要があります。

**参考文献**：Wu et al. (2020), *Practical Multi-fidelity Bayesian Optimization for Hyperparameter Tuning*;
Kandasamy et al. (2017), *Multi-fidelity Bayesian Optimisation with Continuous Approximations*.


## Deep Gaussian Processによるマルチクラス分類

通常のGPでは、観測データを条件付けた潜在関数の事後分布をガウス分布として扱うことができます。一方、Deep GPでは複数の確率的なGP写像を合成するため、最終的な潜在関数の分布は一般にガウス分布ではありません。

クラスごとに独立した深層潜在関数 $f_c(x)$ を持ち、各層の確率的な写像を積み重ねます。ラベル $y$ の条件付き分布は $p(y=c\mid f_1,\ldots,f_C)=\exp(f_c)/\sum_j\exp(f_j)$ と表します。各層の変分分布のKLペナルティを含む目的関数を用い、Softmaxの対数尤度の期待値をサンプリングで近似します。

予測時のクラス確率は、潜在事後のサンプル $f^{(s)}(x)$ から

$$
\hat p(y=c\mid\mathcal D,x)=
\frac{1}{S}\sum_{s=1}^S p(y=c\mid f^{(s)}(x))
$$

として推定します。ここで $S$ はモンテカルロサンプル数です。潜在事後平均にリンク関数を適用するだけでは、一般にこの期待値と一致しません。

クラス確率の分散やBALDも同じサンプルから推定できますが、サンプル数による誤差があります。今回のモデルは既存の 既存の深層ガウス過程の確率的階層を再利用し、ガウス事後を前提とする通常のExact GP獲得関数と同一視しません。

**参考文献**：Damianou and Lawrence (2013), *Deep Gaussian Processes*; Salimbeni and Deisenroth (2017), *Doubly Stochastic Variational Inference for Deep Gaussian Processes*; Hensman et al. (2015), *Scalable Variational Gaussian Process Classification*.


## 無限幅ニューラルネットワーク由来のGPカーネル（NNGP）

無限幅の全結合ReLUニューラルネットワークを考えると、適切な重みの事前分布と幅の極限の下で、その関数の事前分布はガウス過程になります。これがNeural Network Gaussian Process（NNGP）です。有限幅のニューラルネットワークを学習するという意味ではありません。また、複数の確率的なGP層を組み合わせるDeep GPとも異なります。

初期共分散の例として、設計変数 $x,z$ に対し

$$
K^{(0)}(x,z)=\sigma_b^2+\frac{\sigma_w^2}{d}x^\top z
$$

を用います。ReLUの各層で、角度 $\theta$ と両入力の分散から解析的なReLU共分散写像を繰り返して $K^{(\ell+1)}$ を得ます。分類には、この共分散を潜在GPの事前分布として与えます。

マルチクラス分類ではクラス別潜在GPに共通のNNGP型共分散を与え、Softmax尤度と変分推論からクラス確率を推定します。

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

マルチクラス分類では、クラス別の潜在GPにこの共分散構造を与え、Softmaxを介してクラス確率を推定します。 分類尤度は非ガウスであるため、潜在GPの学習は変分近似です。

既知のコンテキスト下で候補点を選ぶことと、未観測のコンテキストに汎化させることは区別してください。新しいIDを用いる際の扱いはモデル仕様によって制限されます。

**参考文献：** Bonilla, Chai and Williams (2008), *Multi-task Gaussian Process Prediction*; Álvarez, Rosasco and Lawrence (2012), *Kernels for Vector-Valued Functions: a Review*; Gelman et al. (2013), *Bayesian Data Analysis*（階層モデリングの背景）。
