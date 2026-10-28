# Candidate Constraint Optimization

[獲得関数の制約章](../acquisition/08_constraints.md)を前提とします。
本章では既知の候補制約を扱い、不等式は $g_j(X)\ge0$、等式は $h_k(X)=0$ の符号規約を使います。
未知応答の制約でよく使う逆向きの符号と混同しません。

## 5.1 Candidate constraint と outcome constraint

candidate/input-space constraint は評価前から既知の feasibility を定義します。

例えば inequality

$$
g(x)\ge 0
$$

や equality

$$
h(x)=0
$$

です。

これは未知の black-box outcome $c(x)$ を surrogate で model 化し、
feasibility probability などを acquisition utility に組み込む
constrained Bayesian optimization とは別レイヤーです。

## 5.2 Linear inequality

係数と閾値による intra-point inequality は

$$
\sum_j a_j x_j \ge b
$$

を表します。

## 5.3 Linear equality

equality は

$$
\sum_j a_j x_j=b
$$

です。

floating-point computation では derivative-free evaluation 時に
tolerance を使って equality violation を判定する必要があります。

許容誤差を含む violation measure は

$$
v_{eq}(x)
=
\max\left(|h(x)|-\varepsilon,0\right)
$$

という形で equality tolerance を扱います。

## 5.4 Intra-point と inter-point

q-batch では constraint が1候補内だけに閉じるとは限りません。

intra-point:

$$
g(x_i)\ge0
$$

inter-point:

$$
g(x_1,\ldots,x_q)\ge0
$$

を区別します。

例えば candidate 間の最小距離や、batch 全体のresource budgetは
inter-point constraintになり得ます。

## 5.5 Nonlinear constraint

nonlinear inequality は関数 $g$ が

$$
g(X)\ge0
$$

を満たす領域として表現します。

概念上、intra-point constraint は各 candidate に閉じ、
inter-point constraint は joint $[q,d]$ batch 全体に依存します。

gradient-based optimizer では制約関数の candidate coordinates に関する
微分可能性が必要になる場合があります。一方、derivative-free optimizer は
制約値の評価だけを利用する方式もあります。

滑らかな問題では、正則性条件の下で局所解が満たす必要条件をKKT条件で表せます。
最大化を最小化 $-\alpha$ に直し、Lagrange関数を

$$
\mathcal L(X,\lambda,\nu)=-\alpha(X)-\sum_j\lambda_jg_j(X)+\sum_k\nu_kh_k(X)
$$

とすると、停留条件 $\nabla_X\mathcal L=0$、可行性、$\lambda_j\ge0$、
相補性 $\lambda_jg_j(X)=0$ を課します。境界で効いている制約が勾配と釣り合う形です。
非凸な獲得関数では、これらを満たすことだけで大域最適性は保証されません。
箱型境界も制約として含めます。

## 5.6 Linear と nonlinear の同時適用

linear inequality、linear equality、nonlinear inequality を同時に課すと、
許容探索領域は各制約の実行可能集合の共通部分になります。
例えば、次の符号規約を採用できます。

$$
\sum_j a_j x_j \ge b,
\qquad
\sum_j a_j x_j = b,
\qquad
g(X) \ge 0
$$

です。

box bounds や固定された入力座標も同じ実行可能集合の定義に含められます。
joint q-batch では、各候補に対する制約と候補集合全体に対する制約を
同時に考える必要があります。

## 5.7 Violation measure

derivative-free backend では異なるconstraintを共通の非負 violationへ変換できます。

inequalityでは

$$
v_{ineq}(x)=\max(-g(x),0)
$$

equalityでは

$$
v_{eq}(x)=\max(|h(x)|-\varepsilon,0)
$$

とし、複数constraintの violation を加算します。

$$
V(x)=\sum_j v_j(x)
$$

可行候補は、定義した等式許容誤差の範囲で $V(x)=0$ となる点です。
異なる単位の制約違反を足す場合は、尺度や重みを定義する必要があります。
同じ制約を単位換算しただけでも、生の違反量の順位やペナルティの強さは変わり得ます。

## 5.8 Native enforcement

native enforcement では optimizer 自身へ constraint を渡します。

非線形制約を局所法で扱う場合、実行可能な初期点や制約に対応した
初期化手法が重要になることがあります。

## 5.9 Penalty method

penalty semantics では constrained problem を

$$
\max_x\;\alpha(x)-\lambda V(x)
$$

のような unconstrained score に変換します。

$\lambda$ が小さすぎれば infeasible solution を好む可能性があり、
大きすぎれば feasible boundary 近傍の数値geometryを悪化させる場合があります。

有限の penalty weight では厳密な実行可能性が保証されるとは限りません。

例えば $[0,1]$ で $\alpha(x)=x$、可行条件を $x=0$ とし、二乗違反のペナルティを使うと、
目的は $x-\lambda x^2$ です。$\lambda\ge0.5$ では最大点が $1/(2\lambda)>0$ となり、有限の係数では厳密な可行解になりません。
一方、絶対値型のペナルティには適切な条件で有限係数でも正確になる場合があります。
ペナルティの形と仮定を定めずに、係数を大きくすれば十分と考えることはできません。

## 5.10 Feasibility-first

feasibility-firstでは objective value と violation を単純に同じscaleで加算しません。

Deb-style orderingでは、

1. feasible は infeasible より優先
2. feasible 同士は acquisition value で比較
3. infeasible 同士は violation が小さい方を優先

という ordering を利用できます。

この順序は目的値と制約違反量の単位を直接混合しない利点があります。

## 5.11 Repair / projection / rejection

constraint handling の一般的な別方式として、

- repair: infeasible candidate を feasible candidate へ修正
- projection: feasible set への射影を計算
- rejection: infeasible proposal を捨てて再生成

があります。

repair は一般に元の提案分布を変え、projection は境界に候補を集中させる場合があります。
rejection は feasible region が小さい場合に効率が低下します。

## 5.12 Mixed variable と constraint

mixed space では integer / categorical の許容集合と制約条件を同時に満たす必要があります。
連続提案 $z$ を修復写像 $R$ により $x=R(z)$ とする場合、制約は
実際に評価する $x$ に対して判定します。
カテゴリ値によって feasible region が変化する場合もあります。

## 5.13 Structured features と constraint

candidate constraint は、実際に探索する入力空間の座標に対して定義します。
モデル内部の task / output 構造と、候補入力に含まれる task / fidelity 座標は区別します。
構造的な変数を候補座標として含める場合、その変数への制約も同じ座標系で定義します。

## 5.14 Coordinate transform と constraint

$x=T(z)$ で探索する場合、元空間の制約 $g(x)\ge 0$ は

$$
g(T(z))\ge 0
$$

へ写像する必要があります。座標変換だけを適用して制約を変換しない場合、
元の constrained optimization problem と異なる問題になります。

## 5.15 Sequential と inter-point

inter-point constraint は q-batch 全体に依存します。
そのため joint optimization と整合しますが、1点ずつ選ぶ greedy sequential
optimization では制約をそのまま分解できない場合があります。

例えば候補プールが $\{0,0.4,1\}$、二候補間の距離を0.8以上とすると、可行な組は0と1です。
最初に0.4を選ぶと、それ自体は点ごとの境界内でも次に追加できる候補がありません。
逐次構成では、現在点の可行性だけでなく、残りの候補を加えて集合制約を満たせるかも問題になります。

## 5.16 Robust / input perturbation と candidate constraint

nominal candidate $x$ が $g(x)\ge 0$ を満たしても、摂動 $w$ を加えた
$x+w$ が必ず実行可能とは限りません。

すべての摂動に対する robust feasibility は

$$
\inf_{w\in\mathcal W} g(x+w)\ge 0
$$

であり、chance constraint は

$$
\Pr(g(x+W)\ge 0)\ge 1-\delta
$$

で表せます。これらは nominal candidate feasibility より強い、または異なる条件です。

## 5.17 Differentiability of nonlinear constraints

gradient-based constrained optimization では、制約関数の微分可能性や
導関数の数値的安定性が重要です。
一方、derivative-free methods は制約値のみを利用する方式もあります。

## 5.18 Constraint method selection

制約の種類、実行可能集合の幾何、可微分性、変数型、joint q-batch の有無に応じて
native enforcement、penalty、feasibility-first、repair、projection を選びます。
特定ライブラリの対応状況は [Optimization guides](../../optimization/README.md) を参照してください。

## この章で覚えておくこと

候補制約は未知応答の制約とは異なり、評価前に可行集合を定めます。
制約の符号・許容誤差・単位を明示し、修復や変換後の実候補で可行性を確認します。
点間制約がある場合は、各点が個別に可行であるだけでは十分ではありません。

## 参考文献

1. Deb, K. (2000).
   *An Efficient Constraint Handling Method for Genetic Algorithms*.
   Computer Methods in Applied Mechanics and Engineering, 186, 311-338.
2. Lampinen, J. (2002).
   *A Constraint Handling Approach for the Differential Evolution Algorithm*.
   Proceedings of the Congress on Evolutionary Computation.
3. Nocedal, J., & Wright, S. J. (2006).
   *Numerical Optimization*. Springer.
