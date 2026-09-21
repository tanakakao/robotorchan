# 12. Acquisition Selection Guide

## 12.1 手法名ではなく意思決定問題から選ぶ

Acquisition function は「一般に最も優れた一つ」を選ぶものではありません。

まず、

1. 最終的に何を知りたい / 良くしたいか
2. observation noise があるか
3. 一度に何点評価するか
4. objective は単一か複数か
5. constraints があるか
6. fidelity / evaluation cost が異なるか
7. target distribution や boundary が重要か

を定義し、その decision problem に対応する acquisition family を選びます。

この章は各 acquisition の優劣ランキングではなく、**problem semantics と acquisition semantics を対応付けるためのガイド**です。

## 12.2 最初の分岐: Optimization か Learning か

最初に最終目的を分けます。

```text
何をしたいか?
│
├─ objective を最大化 / 最小化したい
│    → Bayesian Optimization
│
├─ 関数・prediction を正確に学びたい
│    → Active Learning / Experimental Design
│
└─ threshold boundary を学びたい
     → Level-set Estimation
```

この分岐を飛ばして acquisition 名から選ぶと、posterior variance が大きいだけの点を optimization で選んだり、EI を function learning に使ったりするような目的不整合が起こります。

## 12.3 単目的・低ノイズ・逐次 BO

単一 objective、比較的低 noise、1点ずつ評価する標準 BO では、improvement-based acquisition が自然な基準になります。

代表例:

- LogExpectedImprovement
- LogProbabilityOfImprovement
- Upper Confidence Bound

実務上は numerical stability のため LogEI 系を第一に検討できます。

```text
目的
    incumbent を改善する点を探す

候補
    LogEI

より「改善確率」を直接重視
    LogPI

mean + uncertainty の明示的 trade-off
    UCB
```

詳細は [Improvement](02_improvement.md) と [Confidence Bound](03_confidence_bound.md) を参照してください。

## 12.4 Noise が無視できない BO

observation noise がある場合、observed best を真の incumbent とみなすことが危険になります。

この場合は noisy acquisition を検討します。

```text
single objective
    qLogNEI / noisy improvement family
```

baseline points の latent values に関する uncertainty を含めて candidate utility を評価します。

「q」が付いていても q=1 で使える acquisition があります。q は単に parallelism の有無ではなく joint Monte Carlo formulation と関係します。

詳細は [Batch / Noisy](04_batch_noisy.md) を参照してください。

## 12.5 Batch BO

一度に q 点を並列評価するなら、pointwise acquisition の上位 q 点を選ぶより joint q-acquisition を検討します。

```text
noise が小さい
    qLogEI / qUCB など

noise がある
    qLogNEI など
```

重要なのは、

\[
\alpha(x_1)+\cdots+\alpha(x_q)
\]

ではなく、

\[
\alpha(x_1,\ldots,x_q)
\]

として候補間 correlation と redundancy を扱うことです。

## 12.6 非同期 BO

複数 worker があり、既に評価中の pending points がある場合は、それらを無視して次候補を選ぶと重複探索が起こり得ます。

```text
completed data
    +
pending points
    ↓
next acquisition optimization
```

pending semantics を扱える acquisition / optimization path を利用します。

batch と asynchronous は同じではありません。

- batch: 同時に複数候補を決める
- async: 一部候補の結果がまだ返っていない状態で追加候補を決める

## 12.7 探索を明示的に制御したい

exploration strength を明示的に扱いたい場合は UCB family が理解しやすい選択肢です。

\[
\mu(x)+\sqrt{\beta}\sigma(x)
\]

のように mean と uncertainty の寄与を直接確認できます。

ただし \(\beta\) の convention は実装ごとに確認します。

UCB が常に EI より探索的という固定的なランキングではなく、posterior と parameterization に依存します。

## 12.8 Optimum について情報を得たい

immediate improvement より、optimum に関する information gain を重視するなら information-theoretic BO を検討します。

```text
optimum value f* について学ぶ
    → MES

batch / scalable information criterion
    → GIBBON 系

optimizer location x* について学ぶ
    → PES 系の理論
```

「uncertainty が大きい点」と「optimum について情報を持つ点」は同じではありません。

詳細は [Information-theoretic Acquisition](05_information_theoretic.md) を参照してください。

## 12.9 将来の意思決定価値を評価したい

candidate 自身の improvement ではなく、観測後の最終 decision がどれだけ改善するかを評価したいなら KG family を検討します。

```text
one-step value of information
    → qKnowledgeGradient

複数段階の adaptive decision
    → qMultiStepLookahead
```

計算量は増えますが、remaining budget や terminal decision が重要な問題では理論的に自然です。

詳細は [Lookahead Acquisition](06_lookahead.md) を参照してください。

## 12.10 Multi-objective BO

複数 objective に trade-off があるなら Pareto structure を扱います。

```text
hypervolume improvement を直接評価
    → qLogEHVI

noise を考慮
    → qLogNEHVI

scalarization で Pareto front を探索
    → qLogNParEGO

将来の multi-objective decision value
    → HVKG
```

reference point、目的数、noise、計算量によって選択肢が変わります。

詳細は [Multi-objective Acquisition](07_multiobjective.md) を参照してください。

## 12.11 EHVI と NParEGO の選択

両者を「どちらが常に優れるか」で選びません。

### Hypervolume family

- hypervolume が最終評価指標と整合する
- reference point を意味のある形で設定できる
- objective 数が扱いやすい

場合に自然です。

### Scalarization family

- scalarized subproblems を繰り返したい
- many-objective で hypervolume computation が重い
- preference / weight sampling と整合する

場合に有力です。

problem semantics と computational structure の両方から判断します。

## 12.12 Constraints がある

未知 outcome constraint がある場合は objective utility と feasibility を組み合わせます。

```text
objective
    何を改善したいか

constraint posterior
    feasible かどうか

constrained acquisition
    feasible improvement / utility
```

一方、既知の input constraint は acquisition optimizer の feasible domain として扱える場合があります。

```text
known input constraint
    → candidate search space の制約

unknown outcome constraint
    → posterior-aware constrained acquisition
```

詳細は [Constrained Acquisition](08_constraints.md) を参照してください。

## 12.13 Safety guarantee が必要

constraint-aware BO と safe BO は同じではありません。

制約違反が一度でも許容できない問題では、通常の feasibility-weighted acquisition だけで安全性が保証されるとは限りません。

```text
制約違反も観測として許容可能
    → constrained BO

高確率で安全領域から出ないこと自体が要求
    → safe optimization framework を検討
```

安全性が要求仕様なら、acquisition の名前ではなく保証条件から手法を選びます。

## 12.14 関数全体を学びたい

optimization ではなく regression function を広く学びたいなら uncertainty-based AL を検討します。

```text
candidate 自身の uncertainty
    → PosteriorVariance / PosteriorStd

target region 全体の uncertainty
    → qNIPV / integrated variance reduction
```

PosteriorVariance は単純で解釈しやすい一方、candidate を観測した結果ほかの場所の uncertainty がどう減るかは直接評価しません。

## 12.15 Deployment distribution 上の prediction を学びたい

予測精度が必要な target distribution / target set が明確なら EPIG のような target-aware information acquisition が自然です。

```text
candidate が不確実
    → variance / entropy sampling

candidate を知ると
target prediction が改善する
    → EPIG
```

候補 pool と deployment distribution が異なる場合、この区別は特に重要です。

詳細は [Active Learning Acquisition](09_active_learning.md) を参照してください。

## 12.16 Threshold boundary を学びたい

\[
f(x)=t
\]

の境界が目的なら level-set acquisition を使います。

```text
established boundary criterion
    → Straddle

randomized exploration coefficient
    → RandomizedStraddle

variance × target proximity heuristic
    → BoundaryVariance
```

BoundaryVariance は robotorchan-specific heuristic であり、literature named method として扱いません。

詳細は [Level-set Acquisition](10_level_set.md) を参照してください。

## 12.17 Constraint boundary だけを学びたい場合

未知 constraint \(c(x)\le0\) の boundary を知りたいだけなら、問題は level-set estimation に近づきます。

しかし objective optimization も必要なら constrained BO です。

```text
c(x)=0 の位置を知りたい
    → level-set problem

c(x)<=0 の中で f(x) を最大化したい
    → constrained BO
```

「境界がある」という表面的な共通点だけで acquisition を選びません。

## 12.18 Fidelity と evaluation cost が異なる

複数 fidelity を選べる場合は、target-fidelity decision に対する value と cost を考えます。

```text
target decision の value of information
    → MF-KG

cost
    → cost-aware utility

candidate fidelity
    → acquisition optimization の decision variable
```

低 fidelity が安いだけでは選択理由になりません。target fidelity へ情報 transfer できることが必要です。

詳細は [Multi-Fidelity and Cost-aware Acquisition](11_multifidelity_cost_aware.md) を参照してください。

## 12.19 Multi-Fidelity model を使えば十分か

十分とは限りません。

```text
MF surrogate
    fidelity間の関係を予測

MF acquisition
    どの fidelity を次に評価する価値があるか判断
```

通常 acquisition で fidelity を単なる design variable として探索すると、target fidelity や cost semantics が十分に反映されない場合があります。

## 12.20 Mixed variables と acquisition

continuous / integer / categorical variables が混在していても、acquisition criterion 自体と acquisition optimizer は分けて考えます。

例えば EI を使うかどうかは utility の問題であり、categorical variables をどう探索するかは candidate optimization の問題です。

```text
acquisition
    candidate の価値を定義

acquisition optimizer
    mixed domain でその価値を最大化
```

Mixed model があることと Mixed 専用 acquisition が必要であることも同義ではありません。

## 12.21 High-dimensional input と acquisition

高次元問題でも同じ責務分離が重要です。

PCA、PLS、AE、ALEBO、SAAS、TuRBO などは surrogate representation や search strategy に関係します。

一方 EI、UCB、MES などは candidate utility を定義します。

```text
high-dimensional model / representation
    posterior をどう作るか

acquisition
    posterior から candidate value をどう作るか

search strategy
    高次元空間で candidate をどう探すか
```

「高次元だから専用 acquisition が必須」とは限りません。

## 12.22 Multi-task と acquisition

Multi-task surrogate を使っていても、acquisition が何を意思決定するかを別途定義します。

例えば、

- target task は固定で x だけ選ぶ
- task 自体も次の evaluation choice にする
- 複数 task の joint utility を評価する

では問題が異なります。

MultiTaskGP を使っていることだけで multi-task experimental design が自動的に定義されるわけではありません。

## 12.23 Model capability と acquisition capability を混同しない

robotorchan には Mixed、MultiTask、Multi-Fidelity、高次元、ロバスト、表現力の高い surrogate models があります。

しかし、

```text
model が posterior を返せる
```

ことと

```text
特定 acquisition の semantics が
その posterior structure に定義されている
```

ことは別です。

特に、

- multi-output
- structured output
- ensemble posterior
- q > 1

では reduction / joint utility semantics が必要です。

unsupported structure を generic tensor operation で無理に通すより、意味が定義されるまで明示的に拒否する方が安全です。

## 12.24 Robust optimization

入力摂動や環境変数に対して robust objective を最適化する場合、まず「何を robust utility とするか」を定義します。

例えば、

\[
\mathbb E_w[f(x,w)]
\]

や worst-case / risk measure などです。

その robust objective を posterior samples から構成した上で acquisition を評価します。

したがって robustness は acquisition 名だけで決まるものではなく、objective transformation / environmental integration と acquisition の組合せとして考えます。

## 12.25 Candidate set が有限の場合

finite candidate pool なら continuous acquisition optimization は不要な場合があります。

全候補について score を計算して argmax を取れます。

posterior sampling / Thompson sampling も finite-pool selection と相性が良い方法です。

一方、q > 1 で diversity / joint information が必要なら単純な top-q ranking では不十分な場合があります。

## 12.26 Thompson Sampling

posterior function sample

\[
\tilde f\sim p(f\mid\mathcal D)
\]

を生成し、その sample 上の optimum を選ぶ Thompson Sampling は、明示的な deterministic acquisition formula とは異なる candidate-selection mechanism です。

finite pool では特に自然に利用できます。

multi-output では「何を最大化する posterior sample なのか」を objective で明示する必要があります。

## 12.27 Numerical stability

理論的に同じ utility family でも、数値的に安定した formulation を優先することがあります。

典型例:

```text
EI
    → LogEI

qEI
    → qLogEI

qNEI
    → qLogNEI

qEHVI
    → qLogEHVI

qNEHVI
    → qLogNEHVI
```

Log formulation は問題の目的を別物にするためではなく、極小 improvement probability などで acquisition optimization を安定させるために使われます。

## 12.28 Computational budget

高価なのは black-box evaluation だけとは限りません。

MES、KG、multi-step lookahead、high-q MOBO などでは acquisition computation 自体が大きくなる場合があります。

選択時には、

\[
\text{experiment cost}
\quad\text{vs}\quad
\text{acquisition computation cost}
\]

も考えます。

black-box evaluation が数日かかるなら重い acquisition が合理的でも、millisecond-level simulator では acquisition overhead が支配的になる可能性があります。

## 12.29 推奨する選択手順

実務では次の順番で整理すると acquisition family を選びやすくなります。

### Step 1: 最終目的

```text
optimization?
function learning?
boundary learning?
```

### Step 2: output structure

```text
single objective?
multi-objective?
constraints?
target task?
```

### Step 3: observation process

```text
noise?
batch?
pending evaluations?
finite pool?
```

### Step 4: evaluation structure

```text
same cost?
multiple fidelities?
different costs?
```

### Step 5: decision horizon

```text
myopic improvement?
information gain?
terminal decision value?
multi-step planning?
```

### Step 6: computational structure

```text
continuous / mixed?
high dimensional?
large q?
many objectives?
```

### Step 7: implementation contract

最後に現在の robotorchan / BoTorch integration と posterior compatibility を確認します。

## 12.30 Problem-to-family quick map

| Problem setting | Acquisition family to inspect first | Main question |
| --- | --- | --- |
| Single-objective, low-noise BO | LogEI / UCB | improvement or confidence bound |
| Noisy BO | qLogNEI | latent improvement under noise |
| Parallel BO | q-acquisition | joint batch utility |
| Optimum information | MES / GIBBON | information about optimum |
| Terminal decision value | KG | value of information |
| Multi-step planning | qMultiStepLookahead | adaptive future decisions |
| Multi-objective | qLogEHVI / qLogNEHVI / qLogNParEGO | Pareto utility |
| Unknown constraints | constrained acquisition | feasible utility |
| Function learning | PosteriorVariance / qNIPV | predictive uncertainty |
| Target-aware AL | EPIG | information about target predictions |
| Boundary learning | Straddle family | uncertainty near threshold |
| Multi-fidelity | MF-KG + cost-aware utility | target-fidelity value per cost |

この表は開始点であり、ランキングではありません。

## 12.31 robotorchan のドキュメント層

手法を選んだ後は、目的に応じて参照先を分けます。

```text
docs/theory/acquisition/
    WHY
    理論・数式・関係性

docs/optimization/
    HOW
    API・使用例・現在の適用範囲

docs/optimization/acquisition-status.md
    STATUS
    現在の integration / limitation

src/robotorchan/acquisition/
    IMPLEMENTATION
```

Theory に記載されていることは、robotorchan がすべて独自実装していることを意味しません。

BoTorch native で十分な標準 acquisition は再実装せず、robotorchan-specific acquisition が必要な領域だけを補完します。

## 12.32 まとめ

Acquisition selection の最も重要な原則は、

```text
手法名から問題を選ばない。
問題から utility を定義し、
その utility に対応する acquisition を選ぶ。
```

ことです。

同じ posterior model でも、最終目的が変われば適切な acquisition は変わります。

また model、objective、constraint、acquisition、acquisition optimizer、search strategy はそれぞれ異なる責務です。

この分離を保つことで、robotorchan の多様な surrogate models と BoTorch の acquisition ecosystem を組み合わせても、問題設定の意味を失わずに設計できます。
