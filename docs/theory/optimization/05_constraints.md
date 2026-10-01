# Candidate Constraint Optimization

## 5.1 Candidate constraint

candidate/input-space constraint は
\[
g(X)\ge 0,\qquad h(X)=0
\]
のように評価前から既知の feasibility を定義します。
未知の outcome constraint を acquisition に組み込む constrained BO とは別です。

## 5.2 種類

linear inequality、linear equality、nonlinear inequality、
q-batch 内の点を結ぶ inter-point nonlinear constraint を区別します。

## 5.3 Enforcement

robotorchan は native、penalty、feasibility-first、repair、projection、
rejection という enforcement semantics を区別できます。

BoTorch backend は native、DE / GA 系では feasibility-first、
CMA-ES / PSO / hybrid では penalty を利用する構成があります。
