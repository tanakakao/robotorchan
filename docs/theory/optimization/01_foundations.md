# Acquisition Optimization Foundations

## 1.1 問題設定

acquisition function \(\alpha\) を構築した後、BO は通常
\[
X^* \in \arg\max_{X\in\mathcal X^q}\alpha(X)
\]
を解きます。\(q\) は同時に提案する candidate 数です。

## 1.2 Acquisition と optimizer は別

acquisition は候補の utility を定義し、optimizer はその landscape を探索します。
同じ acquisition でも optimizer、initialization、constraint handling により
得られる candidate は変わり得ます。

## 1.3 非凸最適化

acquisition landscape は一般に非凸です。
そのため multi-start local optimization、sampling、population method、
trust region、hybrid method などを問題構造に応じて使い分けます。

## 1.4 Capability

適用可能性は continuous / integer / categorical、gradient requirement、
q-batch、sequential、fixed features、constraints、GPU などで決まります。

## 1.5 Constraint semantics

native enforcement、penalty、feasibility-first、repair、projection、
rejection は異なる探索 semantics です。
単純な「constraint supported」だけでは挙動を表せません。
