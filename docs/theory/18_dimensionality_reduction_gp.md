# ガウス過程モデルの次元削減

## 1. モデル入力次元と探索次元

元の入力を \(x\in\mathbb R^D\)、次元削減後の表現を
\(z=T(x)\in\mathbb R^d\)（\(d<D\)）とします。
削減入力を用いる代理モデルでは次を仮定します。

$$
f(x)\approx g(T(x)),\qquad g\sim\mathcal{GP}(m_z,k_z).
$$

これは*モデル入力の次元*を変更しますが、候補を元空間から選ぶ必要がある場合、
獲得関数の最適化次元が自動的に下がるわけではありません。

一方、探索空間の埋め込みでは低次元の決定変数 \(z\) を最適化し、
実行可能な元入力 \(x=S(z)\) に写像します。二つの変換は逆写像とは限りません。

## 2. 主成分分析（PCA）

中心化された観測行列 \(X_c\in\mathbb R^{n\times D}\) に対して、
PCAは標本分散が大きくなる直交方向を選びます。
\(V_d\in\mathbb R^{D\times d}\) を上位の右特異ベクトルとすると、表現は次のとおりです。

$$
z=V_d^\top(x-\bar x).
$$

PCAは教師なし手法であり、低分散方向でも応答の予測に重要な場合があります。
したがって主成分数は、寄与率だけではなくモデル選択の対象です。

## 3. 部分的最小二乗法（PLS）

PLSは入力と出力の関係を用いて潜在方向を構成します。
基本的には、射影した説明変数と応答の共分散が大きい方向を選びます。
PCAとは異なる教師あり手法であり、入力分散が小さくても予測に重要な方向を保持できます。

PLSは目的変数を利用するため、成分数の選択を含む射影の学習は
交差検証の各学習分割内で実施する必要があります。
これを守らないと目的変数の情報漏洩が生じます。

## 4. ランダム射影

ランダムな線形写像 \(R\in\mathbb R^{d\times D}\) は次の表現を与えます。

$$
z=Rx.
$$

適切なランダム射影は、射影次元が十分に大きければ、*有限集合*上の点間距離を
高い確率で近似的に保存します。これはJohnson–Lindenstrauss型の保証ですが、
特定の低次元射影が応答関数やその最適解を保存することは保証しません。

## 5. 誘導される共分散と識別可能性

固定された変換 \(T\) に対して、削減入力GPは次の共分散を誘導します。

$$
k_x(x,x')=k_z(T(x),T(x')).
$$

\(k_z\) が妥当なカーネルなら、この共分散は正半定値です。
ただし \(T(x)=T(x')\) のとき、代理モデルは両入力を区別できません。
捨てられた方向に沿う応答の変動は、別途モデル化しなければ表現できません。

変換 \(T\) をGPと同時学習すると、異なる表現とカーネルパラメータが
同じ予測関数を与えるという識別不能性が追加されます。

## 6. 出力次元削減と不確実性の復元

\(y\in\mathbb R^M\) の線形な出力表現を次のように仮定します。

$$
y\approx \mu_y+A z_y,\qquad A\in\mathbb R^{M\times r}.
$$

削減空間の予測分布の平均を \(\mu_z\)、共分散を \(\Sigma_z\) とすると、
復元された出力のモーメントは次のようになります。

$$
\mathbb E[y]\approx\mu_y+A\mu_z,
\qquad
\operatorname{Cov}(y)\approx A\Sigma_zA^\top.
$$

独立な残差モデルの共分散 \(\Sigma_\epsilon\) がある場合、復元共分散に加算します。
この項がないと、捨てた出力方向の不確実性が欠落します。
非線形デコーダでは一般にサンプリングまたは近似的なモーメント伝播が必要です。

下流の多目的獲得関数では、潜在座標を物理目的値と混同せず、
元の出力が持つ意味を保持する必要があります。

## 7. 混合・マルチタスク・構造化入力

カテゴリ識別子、タスク番号、忠実度、文脈ラベルは統計的意味が異なります。
数値コードを通常のユークリッド射影に含めると、根拠のない距離が生じます。

混合入力では、連続設計変数だけを削減する方法が考えられます。

$$
T(x_{\mathrm{cont}},x_{\mathrm{cat}})
=
\big(T_c(x_{\mathrm{cont}}),x_{\mathrm{cat}}\big).
$$

得られたカーネルは削減連続空間とカテゴリ共分散を組み合わせられます。
タスクや忠実度の座標は、区別のないPCA射影に混ぜず、
それぞれの共分散構造でモデル化できます。

## 8. 統計的・最適化上の限界

- 次元削減は帰納的バイアスであり、応答に重要な方向を捨てる可能性があります。
- 情報漏洩を防ぐため、教師あり・教師なしの削減とも各学習分割内で再学習します。
- モデル入力の次元削減は、実行可能な探索空間の次元削減を保証しません。
- 非可逆な射影では、元空間への候補復元が一意に決まりません。
- 捨てた出力方向を無視すると予測不確実性を過小評価する可能性があります。
- 射影次元と代理モデルのハイパーパラメータは合わせて評価します。

## 9. 入力削減と探索空間削減の違い

入力削減 $z=T(x)$ は代理モデルが利用する特徴表現を変えますが、
候補を元の入力空間 $\mathcal X$ から選ぶ必要がある場合、
獲得関数の最適化変数は引き続き $x$ です。

一方、低次元の探索変数 $z$ を直接最適化するには、
元空間への写像 $S(z)\in\mathcal X_{\mathrm{feasible}}$ が必要です。
一般のPCAやPLSは、逆変換した点が箱型制約・整数制約・組成制約を
満たすことを保証しません。したがって、**予測に適した射影**と
**実行可能な候補を生成できる埋め込み**は区別して設計します。

## 10. 出力削減における不確実性と目的関数

多出力応答 $y$ を低次元潜在変数 $z_y$ に圧縮した場合、
潜在空間の予測分散が小さくても、捨てた出力方向の誤差が
小さいとは限りません。線形復元なら、潜在共分散に加えて
必要に応じて残差共分散を評価します。

多目的最適化では、元の物理量の目的方向や制約境界を
潜在成分に置き換えるだけでは一般に等価になりません。
元空間の目的・制約を復元した予測分布で評価するか、
潜在空間の目的が元の問題と等価になる条件を明示する必要があります。

## 11. 入力変換と不確実性伝播の整合性

線形入力射影 $z=T(x)$ を用いる場合、$T$ の推定誤差を無視して
固定変換として扱うGPは、射影の不確実性を予測分散に含めません。
特に少数データで教師あり射影を学習するときは、
射影の変動とGPの予測不確実性を区別する必要があります。

また、出力の線形復元 $y\approx\mu_y+A z_y+\epsilon$ において、
$z_y$ と残差 $\epsilon$ が独立である場合には、

$
\operatorname{Cov}(y)
\approx A\Sigma_zA^\top+\Sigma_\epsilon.
$

独立性が成立しない場合は交差共分散項も必要です。
この仮定を明示せずに残差分散だけを加算すると、
不確実性を過大または過小に評価する可能性があります。

## 参考文献

1. Jolliffe, I. T., & Cadima, J. (2016). Principal component analysis:
   A review and recent developments. *Philosophical Transactions of
   the Royal Society A*, 374.
2. Wold, S., Sjöström, M., & Eriksson, L. (2001). PLS-regression:
   A basic tool of chemometrics. *Chemometrics and Intelligent
   Laboratory Systems*, 58(2), 109–130.
3. Johnson, W. B., & Lindenstrauss, J. (1984). Extensions of
   Lipschitz mappings into a Hilbert space. *Contemporary Mathematics*, 26.
4. Constantine, P. G., Dow, E., & Wang, Q. (2014). Active subspace
   methods in theory and practice: Applications to kriging surfaces.
   *SIAM Journal on Scientific Computing*, 36.
5. Rasmussen, C. E., & Williams, C. K. I. (2006).
   *Gaussian Processes for Machine Learning*. MIT Press.

## 関連文書

- [Gaussian Process](02_gaussian_process.md)
- [High-dimensional GP](08_high_dimensional_gp.md)
- [Neural reduction GP](19_neural_reduction_gp.md)
- [High-dimensional search](20_high_dimensional_search.md)
- [MultiTask / MultiOutput](07_multitask_multioutput.md)
- [Input-reduction model guide](../models/high_dimensional_inputs.md)
- [Output-reduction model guide](../models/high_dimensional_outputs.md)
