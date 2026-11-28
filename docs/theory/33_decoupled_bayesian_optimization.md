# 33. Decoupled Bayesian Optimization

## 1. なぜ評価対象も選ぶのか

材料設計では、同じ組成でも強度試験、導電率測定、品質判定の費用や所要時間は異なる。
通常の coupled BO は設計点を選ぶたびに全項目を測定する。
Decoupled BO は設計点と測定項目を同時に選び、限られた予算を意思決定に有用な観測へ配分する。

入力を \(x\in\mathcal X\subseteq\mathbb R^d\)、出力数を \(m\) とする。
評価マスク \(a\in\{0,1\}^m\setminus\{0\}\) の第 \(j\) 成分が1なら出力 \(j\) を測定する。
許容マスク集合 \(\mathcal A\) は実験装置や測定工程によって決まる。
全項目を測定する場合は \(\mathcal A=\{\mathbf 1\}\) である。

## 2. 部分観測と事後分布

各出力の観測集合は異なってよい。

$$
D_j=\{(x_i,y_{ij}):a_{ij}=1\},\qquad
y_{ij}=f_j(x_i)+\epsilon_{ij}.
$$

独立GP事前と独立観測雑音を仮定すれば、

$$
p(f_1,\ldots,f_m\mid D)=\prod_{j=1}^m p(f_j\mid D_j).
$$

未測定値はゼロでも平均値でもなく、未知の確率変数である。
独立モデルでは出力間の直接的な情報伝播はないが、共通の効用を通して測定対象の選択が結合する。
相関を持つ多出力GPでは、出力間共分散を介して別出力の観測からも学習できる。
この場合、独立GPの積分解は成立しない。

## 3. 意思決定と Knowledge Gradient

最終的に選ぶ設計を \(z\)、効用を \(u(f(z))\) とする。
有限の意思決定集合 \(Z\) における現在の価値は

$$
V(D)=\max_{z\in Z}\mathbb E[u(f(z))\mid D].
$$

測定候補 \((x,a)\) の未知観測を \(Y_a(x)\) とすると、

$$
\alpha_{\mathrm{KG}}(x,a)=
\mathbb E_{Y_a(x)\mid D}[V(D\cup\{(x,a,Y_a)\})]-V(D).
$$

観測前に最良と思う設計と、観測後に選び直す設計の期待効用差である。
固定された意思決定集合と正確な事後分布の下では、期待値と最大化の関係から KG は非負になる。
有限 fantasy、数値最適化、近似事後分布では負の推定値も生じうる。

線形効用 \(u(f)=w^\top f\) では事後平均の重み付き和で終端価値を評価できる。
異なる物理単位を持つ特性は、重みを決める前に尺度と意思決定上の価値を検討する。

## 4. 制約と分類出力

未知制約を \(g_j(z)\leq 0\) とし、不合格時の効用を \(u_0\) とする。
目的と制約が独立な場合、期待効用は

$$
u_0+P(\text{feasible}\mid D,z)
(\mathbb E[u(f(z))\mid D]-u_0).
$$

独立なGaussian制約では合格確率は各制約の正規累積分布関数の積になる。
相関がある場合には同時確率を用いる必要がある。
二値分類では合格クラス確率、多クラス分類では許容クラスの確率和、
順序分類では許容順位の確率和を使う。
ラベルの整数値をそのまま連続回帰の観測とみなす必要はない。
Particle KG は潜在状態の粒子を観測尤度で再重み付けし、観測後の期待効用を近似する。
有限粒子による退化や尤度の設定が精度に影響する。

## 5. 情報獲得と多目的最適化

EPIG は測定候補の観測と、関心のある将来の予測対象 \(Y_*\) の相互情報量を使う。

$$
I(Y_a;Y_*\mid D)=H(Y_*\mid D)-
\mathbb E_{Y_a\mid D}H(Y_*\mid D,Y_a).
$$

Gaussian のスカラー予測では条件付き分散を用いて

$$
I(Y_a;Y_*\mid D)=\frac12\log
\frac{\mathrm{Var}(Y_*\mid D)}
{\mathrm{Var}(Y_*\mid D,Y_a)}
$$

と書ける。独立出力ごとの情報量を重み付き集計する方法は、
一般の相関した多出力の同時相互情報量とは区別する。
EPIG は予測の改善を目的とし、KG は意思決定効用の改善を目的とする。

多目的では、最大化方向へ揃えた目的ベクトルと参照点 \(r\) に対して
非劣解集合の支配領域の体積 Hypervolume を定義する。
HVKG は観測後に選ぶ Pareto 推奨集合の期待 Hypervolume の改善を評価する。
固定重みKGとは異なり、単一の重み付き目的に限定しない。

## 6. コスト、バッチ、非同期

加算コストなら \(c(x,a)=\sum_j a_j c_j(x)\) である。
共通前処理や装置切替がある場合はマスク全体のコスト関数を使う。
残予算 \(B\) の下で、非負の獲得関数に対する一段階の費用効率は

$$
\max_{x,a:c(x,a)\le B}\frac{\alpha(x,a)}{c(x,a)}.
$$

これは有限予算全体の動的計画法の最適解を保証しない。
バッチでは \(X\in\mathbb R^{q\times d}\)、\(A\in\{0,1\}^{q\times m}\) を選び、
同時観測の情報重複とバッチ全体のコストを考慮する。
逐次貪欲法は同時最適化の近似である。

非同期では未完了の観測を既知データに混ぜず、pending の点とマスクを条件に fantasy を生成する。
未完了分の予算を予約し、実測コストで精算することで過剰な発注を防ぐ。

## 7. Multi-Fidelity と比較実験

fidelity \(s\) を入力に加える場合、低忠実度の観測から目標忠実度 \(s_*\) の意思決定を改善する。
測定項目と忠実度は異なる選択軸である。
低忠実度で高い値を得ても、それを目標忠実度の最適値と解釈しない。

比較実験では初期観測を共有し、その費用も総予算に含める。
反復数ではなく累積評価費用で比較し、真値で評価した推奨性能と学習に使う観測を区別する。
複数seedの軌跡を単位として再標本化すれば、軌跡内の依存を保持した点ごとの信頼区間を得られる。

## 参考文献

- Frazier, Powell and Dayanik (2009), The Knowledge-Gradient Policy for Correlated Normal Beliefs, INFORMS Journal on Computing.
- Hernández-Lobato et al. (2016), Predictive Entropy Search for Multi-objective Bayesian Optimization with Constraints, arXiv:1609.01051.
- Gelbart, Snoek and Adams (2014), Bayesian Optimization with Unknown Constraints, UAI.
- Daulton et al. (2023), Hypervolume Knowledge Gradient: Multi-Objective Bayesian Optimization with Partial Information, ICML, PMLR 202.
- Bickford Smith et al. (2023), Prediction-Oriented Bayesian Active Learning, AISTATS, PMLR 206.
- Wu and Frazier (2017), Continuous-Fidelity Bayesian Optimization with Knowledge Gradient, NeurIPS workshop.

