# Acquisition Optimization Foundations

[獲得関数の基礎](../acquisition/01_foundations.md)の期待効用を前提とします。
入力次元を $d$、候補数を $q$ とし、候補全体 $X\in\mathbb R^{q\times d}$ を決定変数として考えます。
本章では、何を最大化するかと、どの精度で数値的に解くかを分けます。

## 1.1 問題設定

acquisition function $\alpha$ を構築した後、BO は内部問題として

$$
X^* \in \arg\max_{X\in\mathcal X^q}\alpha(X)
$$

を解きます。$q$ は同時に提案する candidate 数です。

真の black-box objective の最適化が BO の外側の問題であるのに対し、
acquisition optimization は surrogate posterior から作られた utility を最大化する
**inner optimization problem** です。

制約がある場合は、可行集合 $\mathcal F\subseteq\mathcal X^q$ 上で最大化します。
例えば非空でコンパクトな $\mathcal F$ と連続な $\alpha$ なら最大値は達成されますが、最大点は一つとは限りません。
獲得関数が有限な値を持つことも前提です。対数効用の $-\infty$ や非有限値を扱う場合は別に定義します。

## 1.2 Acquisition と optimizer は別

acquisition は候補の utility を定義し、optimizer はその landscape を探索します。
同じ acquisition でも optimizer、initialization、constraint handling により
得られる candidate は変わり得ます。

acquisition の理論上の定義と、その最大値を数値的に十分よく見つけられるかは別問題です。
inner optimization が不十分なら、理論上よい acquisition を使っても
Bayes decision rule の近似品質が低下します。

誤差には、獲得関数の標本近似、数値最適化、探索領域の制限によるものがあります。
探索する部分領域を $\mathcal R\subseteq\mathcal F$、そこでの最大値を $\alpha^*_{\mathcal R}$ とし、
$\sup_{X\in\mathcal R}|\hat\alpha(X)-\alpha(X)|\le\epsilon$、
$\hat X\in\mathcal R$ が $\hat\alpha(\hat X)\ge\sup_{X\in\mathcal R}\hat\alpha(X)-\delta_{opt}$ を満たすとします。
ここで $\epsilon,\delta_{opt}\ge0$、$\alpha^*_{\mathcal F}$ は元の可行集合での最大値です。このとき

$$
\alpha^*_{\mathcal F}-\alpha(\hat X)
\le (\alpha^*_{\mathcal F}-\alpha^*_{\mathcal R})+2\epsilon+\delta_{opt}.
$$

近似値との二つの差を足し引きすることで得られる上界です。
局所領域内の探索を高精度にしても、領域外の良い候補を取り逃がす項は消えません。
また、これは獲得効用の損失であり、真のブラックボックス関数の性能を直接保証する式ではありません。

## 1.3 なぜ難しいか

acquisition landscape は一般に非凸で、複数の局所解を持ち得ます。
batch BO では decision variable が $q\times d$ に増えます。

MC acquisition では posterior sample を通した sample-average objective を最適化するため、
posterior sampling と candidate gradient の両方が数値計算へ関係します。

そのため multi-start local optimization、sampling、population method、
mixed / discrete optimization、trust-region、hybrid method を使い分けます。

## 1.4 Joint q-batch

batch acquisition では

$$
\max_{X_1,\ldots,X_q}\alpha(X_1,\ldots,X_q)
$$

を joint に解けます。各 $X_i$ を独立に最大化する問題ではありません。

candidate 間の相関や重複の価値が acquisition に入る場合、
joint optimization は $qd$-dimensional な coupled problem になります。

## 1.5 Sequential greedy

joint optimization の代わりに、candidate を1点ずつ選択し、
既選択点の未知の結果や集合への追加価値を次の効用に反映する方法があります。
既選択点を観測済みとして扱うことや、予測平均を真の観測値として追加することとは区別します。

Wilson et al. は代表的 acquisition に対して、
batch maximization と greedy construction の関係を理論的に整理しています。

## 1.6 Sample Average Approximation

概念的には

$$
\hat\alpha_N(X)
=
\frac{1}{N}\sum_{i=1}^{N}
u\!\left(f^{(i)}(X)\right)
$$

を最適化します。

reparameterized posterior sample と automatic differentiation を組み合わせることで、
MC acquisition でも candidate gradient を利用できる場合があります。

ここで $N$ は標本数、$u$ は効用です。最適化の間に基底乱数を固定すれば、
同じ候補には同じ標本近似を評価でき、通常の数値最適化で扱える目的関数になります。
ただし有限標本の最大化は真の期待効用の最大化と同じではなく、固定標本へ過度に適合する場合もあります。

## 1.7 適用条件

optimizer の適用可能性は continuous / integer / categorical、
gradient requirement、q-batch、sequential、fixed features、
candidate constraints、GPU などで決まります。

適用可能性の評価では、数学的な可微分性、変数型、制約の性質、batch の結合構造を個別に確認します。

## 1.8 Constraint semantics

candidate constraint の support は boolean だけでは十分ではありません。

制約を厳密に満たす探索（native / feasibility-first / projection / rejection）と、
違反量を目的関数へ組み込む penalty、候補を修復する repair は異なる方式です。
penalty 法では有限の重みだけで厳密な実行可能性が保証されるとは限りません。

同じ feasible set を対象としても、探索 trajectory や境界付近の挙動は同一ではありません。

## 1.9 探索原理の分類

局所勾配法、random / quasi-random sampling、進化計算、mixed-variable search、
trust-region search、hybrid optimization は異なる探索原理です。
search geometry と数値 optimizer の選択は分離して考えます。

## この章で覚えておくこと

獲得関数の最適化は、実験対象そのものの最適化とは別の内側の問題です。
標本近似、数値探索、領域制限の誤差を分け、候補数qと探索の計算量を区別して解釈します。

## 参考文献

1. Wilson, J. T., Hutter, F., & Deisenroth, M. P. (2018).
   *Maximizing Acquisition Functions for Bayesian Optimization*. NeurIPS 31.
2. Balandat, M., Karrer, B., Jiang, D. R., Daulton, S., Letham, B.,
   Wilson, A. G., & Bakshy, E. (2020).
   *BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization*.
   NeurIPS 33.
3. BoTorch documentation. *Optimization*.
