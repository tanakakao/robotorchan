# Gradient-based Acquisition Optimization

[基礎章](01_foundations.md)と、微分・行列の勾配の基礎を前提とします。
以下では最大化の勾配方向を用い、候補行列は必要に応じて $qd$ 次元のベクトルとして扱います。

## 2.1 基本原理

微分可能な acquisition $\alpha(X)$ に対して

$$
\nabla_X\alpha(X)
$$

を利用して局所最適化します。

box constraint がある場合は $X\in[\ell,u]$ を満たす範囲で最大化します。

例えば閉じた凸集合 $\mathcal F$ へのユークリッド射影 $\Pi_{\mathcal F}$ を使えば、
射影勾配上昇は

$$
X_{k+1}=\Pi_{\mathcal F}(X_k+\eta\nabla\alpha(X_k)),\qquad \eta>0
$$

です。勾配がL-Lipschitz連続で、$0<\eta\le1/L$、現在点が可行なら、
射影の最適性と滑らかさから
$\alpha(X_{k+1})-\alpha(X_k)\ge(1/\eta-L/2)\|X_{k+1}-X_k\|^2$ を得ます。
これは一段の改善であり、非凸な効用の大域解へ到達する保証ではありません。

境界での停止条件には通常の勾配零ではなく、射影勾配
$G_\eta(X)=(\Pi_{\mathcal F}(X+\eta\nabla\alpha(X))-X)/\eta$ を使えます。
例えば $\alpha(x)=x$ を $[0,1]$ で最大化すると、最適点1でも通常の勾配は1ですが、射影勾配は零です。

## 2.2 Candidate gradient と model gradient

ここで必要なのは model parameter $\theta$ に対する
$\nabla_\theta \mathcal L$ ではなく、candidate input に対する
$\nabla_X\alpha(X)$ です。

model fitting に Adam を使うことと、
candidate optimization に Adam を使うことは別の optimization problem です。

## 2.3 Reparameterized MC gradient

posterior sample を

$$
Y(X,\epsilon)=\mu(X)+L(X)\epsilon
$$

のように reparameterize できる場合、固定した base sample $\epsilon$ に対して
acquisition estimator を candidate input で微分できます。

ここで $\epsilon\sim\mathcal N(0,I)$ は候補に依存しない標準正規の基底乱数、$L(X)L(X)^\top$ は予測共分散です。
有限な固定標本の目的を微分することと、期待値と微分を交換できることは別です。
後者には、例えば導関数を積分可能な関数で支配できるなどの正則性が必要です。

特に硬い指示関数では注意が必要です。$Z\sim\mathcal N(0,1)$ に対する
$\mathbb E[\mathbf1(Z\le x)]=\Phi(x)$ の微分は $\phi(x)>0$ ですが、
標本ごとの指示関数の微分はほとんど至る所で零です。
自動微分の経路があるというだけで、真の獲得効用の勾配が正しく得られるわけではありません。

sampler が stochastic に毎 step 変化するなど objective 自体が noisy な設定では、
deterministic quasi-second-order method が適切とは限りません。

## 2.4 Multi-start

acquisition は非凸なので、複数 initial conditions

$$
X_0^{(1)},\ldots,X_0^{(r)}
$$

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

## この章で覚えておくこと

勾配法は候補座標に関する局所的な情報を使います。境界、非滑らかな操作、標本近似を含む場合は、
勾配の意味と停止条件を確認します。複数初期値や準Newton法を使っても、局所探索の限界は残ります。

## 参考文献

1. Wilson, J. T., Hutter, F., & Deisenroth, M. P. (2018).
   *Maximizing Acquisition Functions for Bayesian Optimization*. NeurIPS 31.
2. Balandat, M., Karrer, B., Jiang, D. R., Daulton, S., Letham, B.,
   Wilson, A. G., & Bakshy, E. (2020).
   *BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization*.
   NeurIPS 33.
3. BoTorch documentation. *Optimization*.
4. BoTorch documentation. *Acquisition function optimization with torch.optim*.
