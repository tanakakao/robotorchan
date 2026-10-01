# Acquisition Optimization Initialization

## 3.1 Initialization の役割

multi-start local optimization では initial conditions

\[
X_0^{(1)},\ldots,X_0^{(r)}
\]

が探索する basin を決めます。

acquisition landscape が非凸なら、局所 optimizer が同じでも
初期値によって異なる局所解へ到達します。

## 3.2 Raw samples

典型的な流れは次の通りです。

1. search space から多数の raw candidates を生成
2. acquisition value を評価
3. 有望な候補を restart initial conditions として選択
4. 各 restart から local optimization

`raw_samples` は candidate pool の広さ、
`num_restarts` は local optimization を開始する basin の数に対応します。

## 3.3 Random と quasi-random

raw candidates は独立一様乱数だけでなく、
Sobol sequence などの quasi-random design から生成できます。

low-discrepancy design は有限個の点で search space を
より均等に覆うことを狙います。

## 3.4 Acquisition-weighted initialization

raw candidates を単純に上位だけ選ぶとは限りません。

BoTorch の initialization heuristic には acquisition value を使って
有望な initial conditions を確率的に選ぶ方式があります。

non-negative acquisition では正の acquisition value を標準化し、
その値に応じて sampling probability を変える heuristic があります。

高い acquisition value を優先しながら、
restart が同一 basin に集中しすぎることを避ける狙いがあります。

## 3.5 Flat acquisition

EI 系などでは search space の広い領域で acquisition value と gradient が
ほぼゼロになる場合があります。

そのような点を無作為に initial condition とすると、
gradient-based local search がほとんど移動できない可能性があります。

## 3.6 One-shot initialization

Knowledge Gradient 系では candidate \(X\) に加えて
auxiliary fantasy variables \(X'\) を含む augmented problem

\[
\max_{X,X'}\tilde\alpha(X,X')
\]

を解きます。

通常の q-candidate initialization とは geometry が異なるため、
fantasy variables も含めた専用 initialization が重要です。

BoTorch には KG 系のための
`gen_one_shot_kg_initial_conditions` が用意されています。

## 3.7 Constraint と initialization

candidate constraint がある場合、initial conditions も
optimizer が扱える feasible / compatible な形である必要があります。

nonlinear constraint や mixed variable では、
continuous unconstrained problem と同じ initialization を
機械的に流用できるとは限りません。

## 3.8 Search coordinates

high-dimensional embedding、trust region、hybrid optimization では、
initialization を行う座標系や探索領域自体が変わることがあります。

```text
public input space
latent / embedded space
trust-region local space
```

どの座標系で initial conditions を生成するかを明示する必要があります。

## 3.9 robotorchan の実装境界

現在の対応範囲は
[Acquisition initialization](../../optimization/initialization.md) と
[Initialization compatibility](../../optimization/initialization-compatibility.md)
を source of truth とします。

理論上利用可能な heuristic が、すべての backend で公開されているとは限りません。

## 参考文献

1. Wilson, J. T., Hutter, F., & Deisenroth, M. P. (2018).
   *Maximizing Acquisition Functions for Bayesian Optimization*. NeurIPS 31.
2. Balandat, M., Karrer, B., Jiang, D. R., Daulton, S., Letham, B.,
   Wilson, A. G., & Bakshy, E. (2020).
   *BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization*.
   NeurIPS 33.
3. BoTorch documentation. *Optimization*.
4. BoTorch documentation. *The one-shot Knowledge Gradient acquisition function*.
