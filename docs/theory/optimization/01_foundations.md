# Acquisition Optimization Foundations

## 1.1 問題設定

acquisition function \(\alpha\) を構築した後、BO は内部問題として

\[
X^* \in \arg\max_{X\in\mathcal X^q}\alpha(X)
\]

を解きます。\(q\) は同時に提案する candidate 数です。

真の black-box objective の最適化が BO の外側の問題であるのに対し、
acquisition optimization は surrogate posterior から作られた utility を最大化する
**inner optimization problem** です。

## 1.2 Acquisition と optimizer は別

acquisition は候補の utility を定義し、optimizer はその landscape を探索します。
同じ acquisition でも optimizer、initialization、constraint handling により
得られる candidate は変わり得ます。

acquisition の理論上の定義と、その最大値を数値的に十分よく見つけられるかは別問題です。
inner optimization が不十分なら、理論上よい acquisition を使っても
Bayes decision rule の近似品質が低下します。

## 1.3 なぜ難しいか

acquisition landscape は一般に非凸で、複数の局所解を持ち得ます。
batch BO では decision variable が \(q\times d\) に増えます。

MC acquisition では posterior sample を通した sample-average objective を最適化するため、
posterior sampling と candidate gradient の両方が数値計算へ関係します。

そのため multi-start local optimization、sampling、population method、
mixed / discrete optimization、trust-region、hybrid method を使い分けます。

## 1.4 Joint q-batch

batch acquisition では

\[
\max_{X_1,\ldots,X_q}\alpha(X_1,\ldots,X_q)
\]

を joint に解けます。各 \(X_i\) を独立に最大化する問題ではありません。

candidate 間の相関や重複の価値が acquisition に入る場合、
joint optimization は \(qd\)-dimensional な coupled problem になります。

## 1.5 Sequential greedy

joint optimization の代わりに、candidate を1点ずつ選択し、
既選択点を pending / fantasy として次の acquisition に反映する方法があります。

Wilson et al. は代表的 acquisition に対して、
batch maximization と greedy construction の関係を理論的に整理しています。

joint と sequential は単なる API option の違いではなく、
解く inner optimization problem の構造が異なります。

## 1.6 Sample Average Approximation

BoTorch 系の MC acquisition では、base samples を固定した
sample-average approximation を candidate input に関して最適化できます。

概念的には

\[
\hat\alpha_N(X)
=
\frac{1}{N}\sum_{i=1}^{N}
u\!\left(f^{(i)}(X)\right)
\]

を最適化します。

reparameterized posterior sample と automatic differentiation を組み合わせることで、
MC acquisition でも candidate gradient を利用できる場合があります。

## 1.7 Capability

optimizer の適用可能性は continuous / integer / categorical、
gradient requirement、q-batch、sequential、fixed features、
candidate constraints、GPU などで決まります。

robotorchan はこれらを optimizer capability として実装側で明示します。

## 1.8 Constraint semantics

candidate constraint の support は boolean だけでは十分ではありません。

robotorchan では native、penalty、feasibility-first、repair、projection、
rejection を異なる enforcement semantics として区別します。

同じ feasible set を対象としても、探索 trajectory や境界付近の挙動は同一ではありません。

## 1.9 実装との対応

現在の scalar dispatcher には BoTorch、Torch optimizer、random、Sobol、
DE、CMA-ES、GA、PSO、hybrid があります。

Mixed GA、BoTorch mixed、NSGA-II など specialist backend は
public primitive / capability を持ちますが、scalar dispatcher 名とは別です。

## 参考文献

1. Wilson, J. T., Hutter, F., & Deisenroth, M. P. (2018).
   *Maximizing Acquisition Functions for Bayesian Optimization*. NeurIPS 31.
2. Balandat, M., Karrer, B., Jiang, D. R., Daulton, S., Letham, B.,
   Wilson, A. G., & Bakshy, E. (2020).
   *BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization*.
   NeurIPS 33.
3. BoTorch documentation. *Optimization*.
