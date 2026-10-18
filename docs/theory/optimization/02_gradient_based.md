# Gradient-based Acquisition Optimization

## 2.1 基本原理

微分可能な acquisition \(\alpha(X)\) に対して

\[
\nabla_X\alpha(X)
\]

を利用して局所最適化します。

box constraint がある場合は \(X\in[\ell,u]\) を満たす範囲で最大化します。

## 2.2 Candidate gradient と model gradient

ここで必要なのは model parameter \(\theta\) に対する
\(\nabla_\theta \mathcal L\) ではなく、candidate input に対する
\(\nabla_X\alpha(X)\) です。

model fitting に Adam を使うことと、
candidate optimization に Adam を使うことは別の optimization problem です。

## 2.3 Reparameterized MC gradient

posterior sample を

\[
Y(X,\epsilon)=\mu(X)+L(X)\epsilon
\]

のように reparameterize できる場合、固定した base sample \(\epsilon\) に対して
acquisition estimator を candidate input で微分できます。

これは MC acquisition を gradient-based に最適化する重要な基盤です。

sampler が stochastic に毎 step 変化するなど objective 自体が noisy な設定では、
deterministic quasi-second-order method が適切とは限りません。

## 2.4 Multi-start

acquisition は非凸なので、複数 initial conditions

\[
X_0^{(1)},\ldots,X_0^{(r)}
\]

から局所探索し、得られた局所解の中から最大 acquisition value を持つものを採用します。

restart 数を増やしても global optimum の保証にはなりませんが、
異なる basin を探索する機会は増えます。

## 2.5 Quasi-Newton と制約付き局所最適化

L-BFGS-B は box constraint を扱う準 Newton 法の代表例です。
SLSQP は微分可能な等式・不等式制約を扱える逐次二次計画法です。
どちらも一般の非凸 acquisition に対する大域最適性を保証しません。

## 2.6 First-order iterative optimization

Adam、AdamW、SGD などの first-order method は candidate input を反復更新します。
ただし box / nonlinear constraint を自動的に満たすわけではなく、
射影・パラメータ化・制約付き更新などの追加処理が必要になる場合があります。

実際に利用できる solver と constraint support は
[Optimization guides](../../optimization/README.md) を参照してください。

## 2.7 Gradient が使えることと適していること

autograd graph が構築できるだけでは十分ではありません。

- acquisition が広い領域で flat
- gradient がほぼゼロ
- categorical rounding が入る
- posterior が empirical tree ensemble
- stochastic sampler により objective が noisy
- discontinuous candidate repair が入る

などでは gradient-based search が不利になり得ます。

## 2.8 Restart batch と q-batch

複数 restart の acquisition evaluation は並列化できます。
これは複数 BO candidate を意味する q-batch とは別の batch dimension です。

```text
restart batch: 複数の初期値を並列に最適化
q-batch:       1つの acquisition decision が複数候補を含む
```

両者を混同しません。

## 参考文献

1. Wilson, J. T., Hutter, F., & Deisenroth, M. P. (2018).
   *Maximizing Acquisition Functions for Bayesian Optimization*. NeurIPS 31.
2. Balandat, M., Karrer, B., Jiang, D. R., Daulton, S., Letham, B.,
   Wilson, A. G., & Bakshy, E. (2020).
   *BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization*.
   NeurIPS 33.
3. BoTorch documentation. *Optimization*.
4. BoTorch documentation. *Acquisition function optimization with torch.optim*.
