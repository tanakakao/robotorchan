# 29. Contextual Bayesian Optimization（文脈付きベイズ最適化）

## 1. 何を最適化するのか

通常のベイズ最適化では、実験条件 $x$ を選び、未知の目的関数 $f(x)$ を最大化します。Contextual BOでは、実験時に観測できるが**自分では操作できない**条件 $c$（外気温、材料ロット、設備状態など）が加わります。目的関数は $f(x,c)$ です。

- **意思決定変数 $x$**：実験者が選ぶ設定値
- **Context $c$**：実験前に観測される外部条件
- **観測値 $y$**：実験後に得られる品質など

目的は単一の設定値ではなく、Contextから設定値への**方策（policy）** $\pi(c)$ を学ぶことです。

$$
\pi^*(c) \in \arg\max_{x \in \mathcal X} f(x,c).
$$

Contextを固定して次の実験を提案する場合、獲得関数 $a(x,c)$ を用いて

$$
x_{t+1} \in \arg\max_{x \in \mathcal X} a(x,c_{t+1})
$$

を解きます。Contextを固定するのは**探索時だけ**であり、学習時には $(x,c)$ を一緒にGPへ入力します。

## 2. GPと獲得関数

観測データは $\mathcal D_t=\{((x_i,c_i),y_i)\}_{i=1}^t$ です。GPは、未観測点での予測平均 $\mu_t(x,c)$ と予測分散 $\sigma_t^2(x,c)$ を提供します。獲得関数は予測値と不確実性を用いて次の実験を選びます。既知のContextを固定した獲得関数の最適化は、Context自体の最適化とは異なります。

たとえば $f(x,c)=1-(x-c)^2$ なら、最適な設定値は $x=c$ です。常に $x=0.5$ とする固定設定より、Contextに応じて設定値を変える方策が理論的に有利です。ただし、**この例の理論最適方策は学習済みBOの実績ではありません**。

## 3. Context分布と方策の評価

Contextが有限集合 $c_1,\ldots,c_m$ 上の確率 $p_1,\ldots,p_m$ で現れるなら、方策の期待目的値は

$$
V(\pi)=\sum_{j=1}^{m}p_j f(\pi(c_j),c_j),\qquad
p_j\ge 0,\quad \sum_j p_j=1.
$$

既知の結果を重み付き平均すれば正確な期待値が得られます。Contextを確率 $p_j$ で繰り返しサンプリングする場合は

$$
\widehat V_N(\pi)=\frac{1}{N}\sum_{k=1}^{N}
f(\pi(C_k),C_k)
$$

で推定できます。複数方策に同じContext標本を使えば、比較のサンプリング変動を抑えられる場合があります。ここで扱う評価関数は**与えられた目的値の集約**であり、GP事後分布から方策を推定したり不確実性区間を返したりするものではありません。

## 4. 同じContextのq-batchと異なるContext

同じContextで $q>1$ の候補を同時提案する場合、同じ $c$ を全候補へ固定してjoint acquisitionを最適化できます。一方、異なるContextの候補をまとめて要求する場合、各Contextについて独立した $q=1$ 最適化を行う設計もあります。後者は**異なるContext間のjoint q-batch最適化ではありません**。

## 5. 通常のBO・因果BOとの違い

通常のBOは固定された入力 $x$ の最適化が中心です。Contextual BOは観測された $c$ に応じて $x$ を変えます。因果BOでは変数への**介入**とその因果効果を明示的に扱います。Contextを観測して条件付けるだけでは因果効果は識別できません。

## 6. 評価上の注意

学習中に観測された最良値は、未知Context上での方策価値を保証しません。性能比較では、同じ評価予算、同じContext系列、複数の乱数シードに加えて、可能なら独立した評価用Context集合を用意してください。

実装との対応は[Guide](../guide/29_contextual_bayesian_optimization.md)、ベンチマークの解釈は[Phase 7資料](../../benchmarks/CONTEXTUAL_POLICY.md)を参照してください。


## 7. 問題設定を厳密に書く

意思決定空間を $\mathcal X\subseteq\mathbb R^{d_x}$、Context空間を
$\mathcal C\subseteq\mathbb R^{d_c}$ とします。時刻 $t$ に、実験者は
**意思決定前に** Context $C_t$ を観測し、その後 $X_t\in\mathcal X$
を選択して

$$
Y_t=f(X_t,C_t)+\varepsilon_t
$$

を観測します。$\varepsilon_t$ は観測雑音です。標準的なGP回帰の例では
$\varepsilon_t\mid X_t,C_t\sim\mathcal N(0,\sigma_n^2)$ と仮定しますが、
この仮定はあらゆる実験に成り立つわけではありません。

Context $C_t$ が分布 $P_C$ に従い、目的関数 $f$ が未知であるとします。
方策 $\pi:\mathcal C\to\mathcal X$ はContextを受け取り意思決定を返します。
実行可能集合がContextに依存する場合は
$\pi(c)\in\mathcal X(c)$ が必要です。分布が既知で、$f$ が積分可能なら

$$
V(\pi)=\mathbb E_{C\sim P_C}[f(\pi(C),C)]
$$

が期待方策価値です。点ごとの最適方策が存在し、可測に選択できるなら、
$\pi^*(c)\in\arg\max_{x\in\mathcal X(c)}f(x,c)$ は期待方策価値も最大化します。
ただし、**最大値が必ず達成されるとは限りません**。例えばコンパクトな
実行可能集合と連続な目的関数は、最大値の存在を保証する代表的な十分条件です。

また、Contextが意思決定の結果として変化する場合、単純な外生Contextの
定式化では不十分です。この章では実験前にContextが観測され、
意思決定によってその値が変化しない状況を扱います。

## 8. GP事後分布の導出

入力を $z=(x,c)$ とまとめ、事前分布を

$$
f\sim\mathcal{GP}(m,k)
$$

と置きます。観測入力 $Z=(z_1,\ldots,z_t)$、観測ベクトル
$y=(y_1,\ldots,y_t)^\top$ とし、独立なガウス観測雑音を仮定します。
$K_{ZZ}$ を $(K_{ZZ})_{ij}=k(z_i,z_j)$、
$k_{Zz}$ を $(k_{Zz})_i=k(z_i,z)$ とすると、

$$
\begin{aligned}
\mu_t(z)
&=m(z)+k_{zZ}(K_{ZZ}+\sigma_n^2 I)^{-1}(y-m_Z),\\
s_t^2(z)
&=k(z,z)-k_{zZ}(K_{ZZ}+\sigma_n^2 I)^{-1}k_{Zz}.
\end{aligned}
$$

ここで $s_t^2(z)$ は**潜在関数 $f(z)$ の事後分散**です。
新しい雑音付き観測 $Y$ の予測分散は、独立な同分散雑音の仮定のもとで
$s_t^2(z)+\sigma_n^2$ になります。両者を混同しないでください。
数値計算では逆行列を明示的に作らず、通常はCholesky分解などを使います。

### 獲得関数：Expected Improvementの例

最大化問題で、改善基準を $b$ とし、
潜在関数の事後分布が $\mathcal N(\mu_t(z),s_t^2(z))$ なら、
改善量 $I(z)=\max(f(z)-b,0)$ の期待値は

$$
\operatorname{EI}_t(z)
=(\mu_t(z)-b)\Phi(u)+s_t(z)\phi(u),
\qquad
u=\frac{\mu_t(z)-b}{s_t(z)}
$$

です（$s_t(z)>0$）。$\Phi$ と $\phi$ は標準正規分布の累積分布関数と
確率密度関数です。$s_t(z)=0$ なら
$\operatorname{EI}_t(z)=\max(\mu_t(z)-b,0)$ と定義できます。
既知のContext $c_t$ に対して最適化するのは
$\operatorname{EI}_t(x,c_t)$ の **$x$ のみ**です。
実装は数値安定性のためにLogEI系の獲得関数を利用できます。
ただしノイズあり・バッチの場合は獲得関数の定義や改善基準が異なり、
上式をそのまま適用できるとは限りません。

## 9. 小さな計算例：方策の期待値とregret

$C$ が $0$ と $1$ をそれぞれ確率 $1/2$ でとり、
$f(x,c)=1-(x-c)^2$、$\mathcal X=[0,1]$ とします。

- 固定方策 $\pi_{\mathrm{fixed}}(c)=0.5$ では、両Contextで $f=0.75$ なので
  $V(\pi_{\mathrm{fixed}})=0.75$ です。
- Context依存方策 $\pi^*(c)=c$ では、両Contextで $f=1$ なので
  $V(\pi^*)=1$ です。

この場合、期待方策regretは

$$
R(\pi)=V(\pi^*)-V(\pi)
$$

と定義でき、固定方策のregretは $0.25$ です。
ただし $f$ が未知の場合、$\pi^*$ 自体は通常わかりません。
**学習中の最良観測値**と**未知Contextでの方策regret**は別の指標です。

## 10. 推定値と不確実性の違い

有限Context集合で $r_j=f(\pi(c_j),c_j)$ が**既知**なら、
$\sum_j p_jr_j$ はその離散分布についての正確な期待値です。
一方、$C_1,\ldots,C_N$ を独立同分布でサンプリングし、
$r(C)$ の分散が有限なら、

$$
\mathbb E[\widehat V_N]=V(\pi),\qquad
\operatorname{Var}(\widehat V_N)
=\frac{\operatorname{Var}(r(C))}{N}.
$$

これらは **Contextのサンプリング誤差** に関する式であり、
GPの事後不確実性、観測雑音、Context分布推定誤差は別途扱う必要があります。
同じContext標本を使った2方策の差の推定では、
差 $r_A(C)-r_B(C)$ の分散が重要です。共通乱数の使用が
常に分散削減を保証するわけではありません。

## 参考文献

1. Krause, A. and Ong, C. S. (2011). *Contextual Gaussian Process Bandit Optimization*. Advances in Neural Information Processing Systems 24. Contextを観測してから意思決定するGP banditの代表的研究。
2. Srinivas, N., Krause, A., Kakade, S. M., and Seeger, M. (2010). *Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design*. ICML. GP-UCBと累積regretの理論。
3. Rasmussen, C. E. and Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press, Chapter 2. GP事後平均・分散の導出。https://gaussianprocess.org/gpml/
4. Jones, D. R., Schonlau, M., and Welch, W. J. (1998). *Efficient Global Optimization of Expensive Black-Box Functions*. Journal of Global Optimization 13, 455–492. Expected Improvementを用いたEGO。
5. Ament, S., Daulton, S., Eriksson, D., Balandat, M., and Bakshy, E. (2023). *Unexpected Improvements to Expected Improvement for Bayesian Optimization*. NeurIPS. LogEIの数値安定性に関する研究。
6. Balandat, M. et al. (2020). *BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization*. NeurIPS. BoTorchのモジュラーなモデル・獲得関数・最適化構成。

参考文献は理論の背景であり、robotorchanの各APIの実装検証を直接保証するものではありません。
