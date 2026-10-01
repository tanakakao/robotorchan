# Candidate Constraint Optimization

## 5.1 Candidate constraint と outcome constraint

candidate/input-space constraint は評価前から既知の feasibility を定義します。

例えば inequality

\[
g(x)\ge 0
\]

や equality

\[
h(x)=0
\]

です。

これは未知の black-box outcome \(c(x)\) を surrogate で model 化し、
feasibility probability などを acquisition utility に組み込む
constrained Bayesian optimization とは別レイヤーです。

## 5.2 Linear inequality

robotorchan は BoTorch と同じ tuple contract

```text
(indices, coefficients, rhs)
```

を使います。

intra-point inequality は

\[
\sum_j a_j x_j \ge b
\]

を表します。

## 5.3 Linear equality

equality は

\[
\sum_j a_j x_j=b
\]

です。

floating-point computation では derivative-free evaluation 時に
tolerance を使って equality violation を判定する必要があります。

現在の共通 violation evaluator は

\[
v_{eq}(x)
=
\max\left(|h(x)|-\varepsilon,0\right)
\]

という形で equality tolerance を扱います。

## 5.4 Intra-point と inter-point

q-batch では constraint が1候補内だけに閉じるとは限りません。

intra-point:

\[
g(x_i)\ge0
\]

inter-point:

\[
g(x_1,\ldots,x_q)\ge0
\]

を区別します。

例えば candidate 間の最小距離や、batch 全体のresource budgetは
inter-point constraintになり得ます。

## 5.5 Nonlinear constraint

nonlinear inequality は callable \(g\) が

\[
g(X)\ge0
\]

を返す contract として表現します。

intra-point callable は candidate ごと、
inter-point callable は joint \([q,d]\) batch 全体を受け取ります。

gradient-based optimizer では callable が candidate coordinates に対して
微分可能である必要があります。
derivative-free backend ではこの勾配条件は不要です。

## 5.6 Violation measure

derivative-free backend では異なるconstraintを共通の非負 violationへ変換できます。

inequalityでは

\[
v_{ineq}(x)=\max(-g(x),0)
\]

equalityでは

\[
v_{eq}(x)=\max(|h(x)|-\varepsilon,0)
\]

とし、複数constraintの violation を加算します。

\[
V(x)=\sum_j v_j(x)
\]

feasible candidate は \(V(x)=0\) です。

## 5.7 Native enforcement

native enforcement では optimizer 自身へ constraint を渡します。

現在の BoTorch backend は linear inequality / equality と nonlinear inequality を
upstream optimizer contract へ直接渡します。

nonlinear constraint では feasible initial conditions または
対応する initial-condition generator が必要です。

## 5.8 Penalty method

penalty semantics では constrained problem を

\[
\max_x\;\alpha(x)-\lambda V(x)
\]

のような unconstrained score に変換します。

\(\lambda\) が小さすぎれば infeasible solution を好む可能性があり、
大きすぎれば feasible boundary 近傍の数値geometryを悪化させる場合があります。

現在の PSO や一部 hybrid / CMA-ES route はこの family に位置付けられます。

## 5.9 Feasibility-first

feasibility-firstでは objective value と violation を単純に同じscaleで加算しません。

Deb-style orderingでは、

1. feasible は infeasible より優先
2. feasible 同士は acquisition value で比較
3. infeasible 同士は violation が小さい方を優先

という ordering を利用できます。

robotorchan の GA 系共通評価はこの考え方を明示的に実装しています。

## 5.10 Repair / projection / rejection

constraint handling の一般的な別方式として、

- repair: infeasible candidate を feasible candidate へ修正
- projection: feasible set への射影を計算
- rejection: infeasible proposal を捨てて再生成

があります。

capability metadata に semantics が存在しても、
各 backend が現在すべての方式を実装していることを意味しません。

## 5.11 Mixed variable と constraint

mixed space では constraint evaluation の前に
integer / categorical feasibility を保証する必要があります。

continuous proposal \(z\) を integer repair して \(x=R(z)\) とする場合、
constraint は実際に評価される \(x\) に対して判定すべきです。

現在の DE / PSO なども structured repair 後の candidate に対して
acquisition / constraint evaluation を行います。

## 5.12 Coordinate transform と constraint

public-space constraint を latent / embedded coordinateへそのまま渡すことはできません。

\[
x=T(z)
\]

で探索するなら、public constraint \(g(x)\ge0\) は

\[
g(T(z))\ge0
\]

として正しく写像する必要があります。

robotorchan は exact mapping を実装していない embedding / latent strategy では
public-space `CandidateConstraints` を明示的に拒否します。

これは保守的なAPI制限ではなく、異なる座標系で別のconstraintを
誤って解くことを防ぐ correctness contract です。

## 5.13 Sequential と inter-point

inter-point constraint は q-batch 全体を同時に見る必要があります。

そのため joint optimization と自然に整合しますが、
1点ずつ選ぶ greedy sequential optimization では同じcontractを
そのまま適用できない場合があります。

現在の BoTorch contract でも inter-point nonlinear constraint と
greedy sequential optimization の組合せには制限があります。

## 5.14 Runtime capability

最終的な利用可否は

```text
constraint type
 -> optimizer capability
 -> enforcement semantics
 -> variable-space compatibility
 -> initialization compatibility
```

の順に確認します。

「nonlinear constraint対応」という1つのbooleanだけで判断しません。

## 参考文献

1. Deb, K. (2000).
   *An Efficient Constraint Handling Method for Genetic Algorithms*.
   Computer Methods in Applied Mechanics and Engineering, 186, 311-338.
2. Lampinen, J. (2002).
   *A Constraint Handling Approach for the Differential Evolution Algorithm*.
   Proceedings of the Congress on Evolutionary Computation.
3. Nocedal, J., & Wright, S. J. (2006).
   *Numerical Optimization*. Springer.
4. BoTorch documentation. *Optimization* and candidate-constraint API.
