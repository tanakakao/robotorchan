# 32. 異種モデルによる分類制約付きベイズ最適化

連続的な性能指標は回帰モデルで予測し、品質や安全性のような離散ラベルは分類モデルで予測する問題を考えます。本章では、**単目的回帰 × 分類制約** と **多目的回帰 × 分類制約**を統一的に扱います。分類器は二値・マルチクラス・順序分類のいずれでも構いませんが、クラス確率の定義と解釈は異なります。

前提：[第24章（二値分類）](24_classification.md)、[第30章（マルチクラス分類）](30_multiclass_classification.md)、[第31章（順序分類）](31_ordinal_classification.md)。実験コードは[Notebook 61](../../examples/notebooks/61_heterogeneous_classification_constraints.ipynb)にあります。

## 1. 何を最大化し、何を許容するのか

実験条件を \(x\in\mathcal X\)、観測する連続目的値を \(f(x)\)、品質クラスを \(C(x)\in\{0,\ldots,K-1\}\) とします。分類用データ \(\mathcal D_C\) による各クラスの**予測確率**を

$$
p_k(x)=P(C(x)=k\mid x,\mathcal D_C),
\qquad \sum_{k=0}^{K-1}p_k(x)=1
$$

と定義します。許容するクラスの集合 \(S\subseteq\{0,\ldots,K-1\}\) に対し、

$$
p_S(x)=P(C(x)\in S\mid x,\mathcal D_C)=\sum_{k\in S}p_k(x)
$$

が候補点の周辺実行可能確率（Probability of Feasibility, PoF）です。

例えばクラス0=不良、1=要確認、2=良好で、予測確率が \((0.15,0.25,0.60)\) なら、許容集合 \(S=\{1,2\}\) に対する PoF は \(0.85\) です。マルチクラスGPではSoftmaxによる確率、順序GPではOrdered-probitによる確率を使いますが、**クラス集合に属する確率を足す式**は共通です。

順序モデルでも集合 \(S\) は利用者が業務仕様に基づいて指定します。「順序があるからクラス1以上を自動的に許容する」わけではありません。

## 2. 単目的：回帰GPの期待改善量

回帰側の学習データを \(\mathcal D_f\) とし、最大化する目的の既知最良値を \(f_{\mathrm{best}}\) とします。候補バッチ \(X=(x_1,\ldots,x_q)\) に対して

$$
I_q(X)=\max\left\{0,\ \max_{1\le i\le q}f(x_i)-f_{\mathrm{best}}\right\},
$$

$$
qEI(X)=\mathbb E_{p(f(X)\mid\mathcal D_f)}[I_q(X)]
$$

を定義します。\(q>1\) では回帰GPの**候補間を含む同時事後分布**から期待値を評価します。目的関数の出力値と分類確率は異なる単位なので、単純に確率だけを最大化する方法とは目的が異なります。

分類PoFで重み付けする簡便な獲得スコアは

$$
A_{\mathrm{single}}(X)
=
qEI(X)\,w(X),
\qquad
w(X)=\prod_{i=1}^{q}p_S(x_i).
$$

単一候補 \(q=1\) で、改善量と実行可能事象の事後独立を仮定できるなら、この積は期待実行可能改善の因数分解として解釈できます。しかし、**複数候補の積は一般の制約付きqEIと同一ではありません**。例えばバッチ内の1点だけ可行な場合にも改善が起こることがありますが、上式の積は全候補が可行な事象を重視するためです。

既知ラベルが観測されている場合、例の \(f_{\mathrm{best}}\) には通常、**観測された可行点の最良値**を使います。不適合な実験の最高性能を可行な基準点として扱うと、改善量の比較対象が変わります。

## 3. 多目的：Pareto集合と期待超体積改善

\(M\ge2\) 個の連続目的を同時最大化するとき、目的ベクトルを \(\boldsymbol f(x)=(f_1(x),\ldots,f_M(x))\) とします。目的間にトレードオフがあるため、一般には唯一の最良値ではなく、他の候補に全目的で劣らない**Pareto非劣解集合**を考えます。

参照点 \(\boldsymbol r\) は目的空間で「これ以上の改善を測る基準」となる点です。既知の可行観測の目的ベクトル集合を \(\mathcal P\)、参照点に対する支配超体積を \(HV_{\boldsymbol r}(\mathcal P)\) と書くと、

$$
\Delta HV(X)
=
HV_{\boldsymbol r}\bigl(\mathcal P\cup\{\boldsymbol f(x_1),\ldots,\boldsymbol f(x_q)\}\bigr)
-
HV_{\boldsymbol r}(\mathcal P),
$$

$$
qEHVI(X)
=
\mathbb E_{p(\boldsymbol f(X)\mid\mathcal D_f)}[\Delta HV(X)]
$$

が**期待超体積改善**です。2目的の場合は、参照点から可行Pareto前線までの「目的空間で支配される面積」をどれだけ増やせるかと解釈できます。

回帰の多目的獲得スコアに分類PoFを重み付けすると、

$$
A_{\mathrm{multi}}(X)=qEHVI(X)\,w(X)
$$

を得ます。**2つの連続目的は同じ回帰モデルの結合事後分布**から得る必要があります。分類器は別のモデルとして構成できますが、回帰と分類の依存や相関は、この積だけでは表現されません。

既知の観測品質ラベルがある場合は、\(\mathcal P\) を**既知の可行観測だけ**から構成します。ラベルが不適合な観測をPareto参照集合へ入れると、可行な改善量を過小評価しかねません。参照点は全目的の向き・尺度と整合させて選びます。

## 4. 確率積の仮定とq-batchの注意

複数の分類制約 \(j=1,\ldots,J\) があるとき、各候補のPoFを \(p_{S_j}^{(j)}(x_i)\) とすると、分類側の重み付けを

$$
w_{\mathrm{product}}(X)
=
\prod_{i=1}^{q}\prod_{j=1}^{J}p_{S_j}^{(j)}(x_i)
$$

とできます。これは以下の**二種類の独立性を近似的に置く**ことに対応します。

- 候補点間の分類ラベルの実行可能事象の独立性
- 別々の分類制約間の実行可能事象の独立性

通常、同じ分類GPが近傍の入力を予測する場合、候補間の確率的依存は無視できません。また分類確率はモデルの近似事後分布から得られるため、**校正誤差や分布外入力**にも注意が必要です。

候補ごとのPoFが \(0.8,0.6\) なら、積は \(0.48\) です。別の集約として \(\min(0.8,0.6)=0.6\) を使うこともできますが、**最小値は同時実行可能確率ではなく**、積より通常大きくなる別のスコアです。最小値の方が統計的に保守的という意味ではありません。

## 5. ソフト重み付けと確率下限の制約は違う

「許容クラスになる確率が80%以上」を確実に候補条件へ課すなら、

$$
p_S(x)\ge0.8
$$

という**候補に対する確率制約**を別途指定する必要があります。\(A_{\mathrm{single}}\) や \(A_{\mathrm{multi}}\) を最大化しても、その不等式を満たすことは保証されません。

さらに、予測PoFが80%以上でも、実際に80%以上の頻度で許容品質になると保証されるわけではありません。予測確率の信頼性は、保留データでのNLL、Brier score、信頼性曲線等を使って検証する必要があります。特に安全上重要な用途では、ソフトPoF重み付けを安全保証と見なしてはいけません。

## 6. 実験・実装に移るときの整理

1. **連続目的の数**：1目的ならqEI、2目的以上ならqEHVI等を検討する
2. **分類器の選択**：ラベルに順序がないならマルチクラス、順序があるならOrdinal GPを検討する
3. **可行クラス集合**：クラス番号の意味を文書化し、業務仕様に基づいて選ぶ
4. **学習**：回帰モデルと分類モデルを**ともに学習**する。異種モデルの結合分布を自動的に得るわけではない
5. **改善の基準**：可行な観測点の最良値／可行Pareto集合を使用する
6. **候補の最適化**：合成された獲得スコアを探索領域で最大化し、次に実験する \(x\) を得る
7. **再学習**：候補の測定後に連続目的値と品質ラベルを観測し、それぞれの学習データへ加える

この章は獲得スコアの数学的前提を述べたもので、実験実行時の**厳密な同時可行確率**・**閾値制約**・**非同期fantasization**を自動的に正当化するものではありません。

## 参考文献

1. Jones, D. R., Schonlau, M., & Welch, W. J. (1998). *Efficient Global Optimization of Expensive Black-Box Functions*. Journal of Global Optimization.
2. Gardner, J. R., Kusner, M. J., Xu, Z. E., Weinberger, K. Q., & Cunningham, J. P. (2014). *Bayesian Optimization with Inequality Constraints*. ICML, PMLR 32.
3. Balandat, M., et al. (2020). *BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization*. NeurIPS.
4. Daulton, S., Balandat, M., & Bakshy, E. (2020). *Differentiable Expected Hypervolume Improvement for Parallel Multi-Objective Bayesian Optimization*. NeurIPS.
5. Hernández-Lobato, J. M., et al. (2016). *Predictive Entropy Search for Bayesian Optimization with Unknown Constraints*. ICML, PMLR 48.

**文献上の手法と本章の近似の区別**：上記の論文がすべてPoFの周辺確率積をそのまま用いるという意味ではありません。本章のqEI/qEHVI×PoFの合成と、その独立性・閾値非保証は明示的に区別しています。
