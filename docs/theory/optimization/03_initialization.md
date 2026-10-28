# Acquisition Optimization Initialization

[勾配法章](02_gradient_based.md)の多初期値探索を前提とします。
本章では、初期候補の被覆、初期値の選択、可行な探索の開始点を分けて考えます。

## 3.1 Initialization の役割

multi-start local optimization では initial conditions

$$
X_0^{(1)},\ldots,X_0^{(r)}
$$

が探索する basin を決めます。

acquisition landscape が非凸なら、局所 optimizer が同じでも
初期値によって異なる局所解へ到達します。

## 3.2 Raw samples

典型的な流れは次の通りです。

1. search space から多数の raw candidates を生成
2. acquisition value を評価
3. 有望な候補を restart initial conditions として選択
4. 各 restart から local optimization

raw candidate 数は初期候補 pool の広さ、
restart 数は局所探索を開始する初期点の数に対応します。

## 3.3 Random と quasi-random

raw candidates は独立一様乱数だけでなく、
Sobol sequence などの quasi-random design から生成できます。

low-discrepancy design は有限個の点で search space を
より均等に覆うことを狙います。

良い局所解へ収束する領域に入る確率が一回の独立な初期値で $p$ なら、
N個の独立な初期値が少なくとも一つ入る確率は $1-(1-p)^N$ です。
例えば $p=0.01,N=100$ なら約0.634であり、初期値を増やしても有限回で必ず捉えるわけではありません。
この式は独立抽出の仮定に依存するので、Sobol点へそのまま適用しません。
また、生の候補が良い領域に入ることと、選抜した局所初期値がそこに残ることも異なります。

## 3.4 Acquisition-weighted initialization

raw candidates を単純に上位だけ選ぶとは限りません。

acquisition value を使って有望な initial conditions を
確率的に選ぶ heuristic があります。

non-negative acquisition では正の acquisition value を標準化し、
その値に応じて sampling probability を変える heuristic があります。

高い acquisition value を優先しながら、
restartが同じ領域へ偏ることを緩和する狙いがあります。
例えば有限な評価値 $a_i$ と温度 $T>0$ に対し、
$p_i=\exp((a_i-\max_j a_j)/T)/\sum_k\exp((a_k-\max_j a_j)/T)$ で選ぶ方法を考えられます。
最大値を引いても確率は変わらず、指数関数のあふれを抑えられます。
Tが小さいと高評価へ集中し、大きいと一様へ近づきます。全値が等しければ一様です。
ただし、評価値の重み付けだけで空間的な多様性が保証されるわけではありません。

## 3.5 Flat acquisition

EI 系などでは search space の広い領域で acquisition value と gradient が
ほぼゼロになる場合があります。

そのような点を無作為に initial condition とすると、
gradient-based local search がほとんど移動できない可能性があります。

## 3.6 One-shot initialization

Knowledge Gradient 系では candidate $X$ に加えて
auxiliary fantasy variables $X'$ を含む augmented problem

$$
\max_{X,X'}\tilde\alpha(X,X')
$$

を解きます。

通常の q-candidate initialization とは geometry が異なるため、
fantasy variables も含めた専用 initialization が重要です。

この場合、通常の候補変数と補助変数の両方に対して初期値を構成します。

## 3.7 Constraint と initialization

candidate constraint がある場合、initial conditions も
optimizer が扱える feasible / compatible な形である必要があります。

nonlinear constraint や mixed variable では、
continuous unconstrained problem と同じ initialization を
機械的に流用できるとは限りません。

例えば $x_1+x_2=1$ を単位正方形の中で満たす点は線分上にあり、
連続一様乱数をそのまま引いて厳密な等式へ一致する確率は零です。
棄却だけに頼らず、$x=(z,1-z),\ z\in[0,1]$ のようなパラメータ化や射影などで初期点を構成します。
不等式・等式・離散性を同時に課す場合も、実際に探索する点での可行性を確認します。

## 3.8 Search coordinates

high-dimensional embedding、trust region、hybrid optimization では、
initialization を行う座標系や探索領域自体が変わることがあります。

```text
public input space
latent / embedded space
trust-region local space
```

どの座標系で initial conditions を生成するかを明示する必要があります。

## この章で覚えておくこと

初期化は探索の出発点と局所解の多様性を左右します。
広く候補を生成すること、評価値で選抜すること、可行性を満たすことは別の役割です。
拡張問題や変換後の座標では、その問題で実際に最適化する変数に対して初期値を用意します。

## 参考文献

1. Wilson, J. T., Hutter, F., & Deisenroth, M. P. (2018).
   *Maximizing Acquisition Functions for Bayesian Optimization*. NeurIPS 31.
2. Balandat, M., Karrer, B., Jiang, D. R., Daulton, S., Letham, B.,
   Wilson, A. G., & Bakshy, E. (2020).
   *BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization*.
   NeurIPS 33.
3. BoTorch documentation. *Optimization*.
4. BoTorch documentation. *The one-shot Knowledge Gradient acquisition function*.
