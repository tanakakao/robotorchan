# Hybrid Acquisition Optimization

## 11.1 目的

Hybrid acquisition optimization は、1つの numerical optimizer に全探索責務を持たせず、
異なる性質を持つ探索段階を接続します。

代表的な hybrid optimization は

    global derivative-free search
      -> promising joint q-batch
      -> BoTorch local refinement

という **global-to-local** 構成です。

これは複数 optimizer の多数決や、候補を単純に比較する ensemble ではありません。
前段の出力を後段の initial condition として渡す、明示的な二段最適化です。

## 11.2 なぜ global-to-local か

acquisition function は smooth であっても、多峰性を持つことがあります。

gradient-based local optimization は局所的な refinement に適していますが、
initial condition が悪ければ別の basin を探索できません。

一方、Differential Evolution、CMA-ES、Genetic Algorithm などの
derivative-free / population method は広い領域を探索できますが、
高精度な局所収束では追加評価を必要とする場合があります。

Hybrid method は

    global exploration capability
    + local refinement capability

を段階的に組み合わせます。

## 11.3 Global-to-local composition

二段階探索では広域探索で得た候補を局所探索の初期値とします。
各段階は異なる数値最適化手法を利用できますが、目的関数と探索変数の意味を保つ必要があります。

## 11.4 Joint q-batch hand-off

global stage は acquisition を joint q-batch として最適化し、

    X_global in R^(q x d)

を返します。

local stage には

    batch_initial_conditions = X_global.unsqueeze(0)

として渡します。

つまり global stage の上位 q 個の independent point を後から束ねるのではありません。

q-batch acquisition の場合、global stage から local stage まで
`[q, d]` を1つの joint decision として維持する必要があります。

## 11.5 Initialization としての global search

通常の multi-start BoTorch optimization では、raw samples などから initial conditions を選びます。

Hybrid backend では global optimizer 自体が initial-condition generator の役割を持ちます。

    global optimizer
      -> one promising q-batch
      -> explicit batch_initial_conditions
      -> local optimization

そのため local stage では `raw_samples=None` です。

global search と local initialization は独立した処理ではなく、
global candidate がそのまま local start になります。

## 11.6 Local restart diversity

同じ初期点を複製しても探索 basin の多様性は増えません。
複数 restart を使う場合は、異なる global candidates を初期値として渡すことが有効です。

## 11.7 Global optimizer と gradient

global stage は candidate gradient を必要としない optimizer を使えます。

しかし local stage が gradient-based optimization を使う場合、
Hybrid 全体としては local refinement に必要な differentiability を失ってはいけません。

したがって

    global stage does not require gradients

と

    hybrid pipeline does not require gradients

は同じ意味ではありません。

局所段階に勾配が必要な場合は、
この local-stage requirement を表します。

## 11.8 Constraint semantics は stage-dependent

candidate constraints は global stage と local stage の両方へ渡されます。

ただし enforcement method は同一ではありません。

| Global stage | Global constraint handling | Local stage |
| --- | --- | --- |
| DE | feasibility-first | BoTorch native |
| CMA-ES | penalty | BoTorch native |
| Mixed GA | feasibility-first | BoTorch mixed native |
| custom callable | callable の contract に依存 | BoTorch native / mixed native |

したがって Hybrid の制約処理を単一の「penalty method」と理解するのは不正確です。

最終 candidate は local stage の constraint semantics も満たす必要がありますが、
global stage で infeasible landscape をどう探索するかは選択した optimizer に依存します。

## 11.9 Nonlinear constraints

nonlinear candidate constraints が存在する場合、local BoTorch stage では

    constrained local optimization

が default として設定されます。

制約付き局所最適化では初期点の実行可能性と数値安定性に注意が必要です。

global stage 側の nonlinear constraint handling は選択した global optimizer の方式に従います。

したがって同じ constraint object を共有しても、数値的な enforcement mechanism は
stage 間で同一とは限りません。

## 11.10 Fixed features

fixed features は global stage と local stage の両方で一貫して維持する必要があります。
local stage にも引き継がれます。

Mixed local stage では common fixed features と
`mixed_fixed_features_list` の各 discrete configuration を merge します。

同じ dimension に矛盾する値が指定された場合は error になります。

ただし段階間で固定座標が異なると、
`optimizer="hybrid"` と `fixed_features` の組合せは拒否されます。

## 11.11 Mixed-variable Hybrid

Mixed-variable hybrid では構造化された変数空間を扱います。

integer / categorical dimension が存在する場合、global stage と local stage の双方で
structured variable semantics を保持する必要があります。

現在の構成では global optimizer として Mixed GA を利用でき、
local stage は BoTorch mixed optimization を使えます。

local stage に渡す `mixed_fixed_features_list` は、各 entry が
すべての structured dimension を固定していなければなりません。

これにより local refinement 中に categorical / integer coordinate を
continuous variable として誤って動かすことを防ぎます。

## 11.12 Mixed hand-off の意味

continuous hybrid では global candidate 全体が local initial condition になります。

Mixed local optimization ではさらに

    discrete assignment
      + continuous local refinement

という構造があります。

したがって Mixed Hybrid の hand-off は単に Tensor を渡すだけでなく、

- global candidate
- variable-space semantics
- legal discrete assignments
- common fixed features

を一致させる必要があります。

## 11.13 数値最適化手法の合成可能性

global / local の組合せには、変数型、制約、座標系、勾配の要件が整合することが必要です。
理論上の合成可能性と特定ライブラリの対応範囲は区別します。

## 11.14 Stage-dependent properties

Hybrid の性質は構成要素に依存するため、単一の optimizer 名だけでは特徴を決定できません。
たとえば全体の可微分性要件は、各段階で必要な情報の和集合で評価します。

## 11.15 Custom global search

任意の広域探索法を初期候補生成に使う場合、局所法が期待する目的関数、変数型、制約、
探索座標系と整合する必要があります。

## 11.16 Computational environment

Hybrid の計算コストは global search の評価回数と local refinement の費用に依存します。
各 stage の計算デバイスが異なる場合、データ転送のコストも考慮する必要があります。
特定の実行環境への対応状況は数学的な hybrid formulation とは別の問題です。

## 11.17 Hybrid と sequential optimization

joint hybrid q-batch optimization と sequential greedy selection は異なる操作です。
前者は q 個の候補を一括最適化し、後者は既選択点を条件として次の点を選びます。
両者を組み合わせることも可能ですが、pending state の扱いを明確にする必要があります。

## 11.18 Hybrid と one-shot acquisition

one-shot acquisition は auxiliary variables を含む augmented q-batch を最適化します。
Hybrid で扱うには、各段階で augmented variables の意味と制約を維持し、
最適化後に実候補だけを抽出する必要があります。

## 11.19 Hybrid と Trust Region

Hybrid optimization は global-to-local な optimizer composition を表します。
一方、TuRBO は探索領域の geometry と state adaptation を表します。
両者を組み合わせる場合は局所領域、状態更新、restart と初期化の整合が必要です。

## 11.20 Hybrid と high-dimensional embedding

REMBO、HeSBO、ALEBO、BAxUS では探索座標が元空間と異なる場合があります。
Hybrid を reduced search space で用いる場合、global / local の各段階で
座標写像と制約の意味を一貫させる必要があります。

## 11.21 何を組み合わせているのか

Hybrid method を理解するときは、少なくとも次の4層を分けます。

| Layer | 例 |
| --- | --- |
| Search domain | global box, trust region, embedded polytope |
| Global numerical search | DE, CMA-ES, Mixed GA |
| Hand-off | explicit batch initial condition |
| Local numerical search | BoTorch continuous / mixed optimization |

この分離により「global exploration」と「acquisition exploration」を混同しにくくなります。

global optimizer が広く候補を探しても、acquisition function 自体の
exploration-exploitation trade-off が変わるわけではありません。

## 11.22 Failure modes

Hybrid optimization では各 stage 単独にはない failure mode があります。

- global stage が悪い basin だけを返す
- global stage と local stage で constraint semantics がずれる
- fixed / discrete coordinate が hand-off で変化する
- q-batch shape を point-wise candidate と誤認する
- local stage が必要とする gradient が取得できない
- coordinate system が stage 間で一致しない
- 同じ global candidate を複製して multi-start と誤認する

そのため「optimizerを2つ使う」ことより、stage間 contract を維持することが重要です。

## 11.23 理論上の整合条件

- joint q-batch の意味を段階間で保つ
- 異なる初期点を使う場合は basin の多様性を意識する
- 制約の符号規約と実行可能集合を段階間で一致させる
- fixed / discrete coordinate を hand-off で変更しない
- local optimizer に必要な勾配が得られることを確認する
- reduced coordinates と元空間の写像を整合させる

実装固有の制約と対応状況は [Optimization guides](../../optimization/README.md) を参照してください。

## 11.24 関連章

initial-condition semantics は
[Initialization](03_initialization.md) を参照してください。

Mixed / discrete optimization は
[Mixed and Discrete Optimization](04_mixed_discrete.md) を参照してください。

candidate constraints は
[Candidate Constraints](05_constraints.md) を参照してください。

derivative-free global optimizer は
[Derivative-free Optimization](06_derivative_free.md) と
[Evolutionary Optimization](07_evolutionary.md) を参照してください。

batch / one-shot semantics は
[Batch and One-shot Optimization](08_batch_one_shot.md) を参照してください。

Trust Region は [Trust-region Optimization](10_trust_region.md) を参照してください。

## 参考文献

1. Jones, D. R., Schonlau, M., & Welch, W. J. (1998).
   *Efficient Global Optimization of Expensive Black-Box Functions*.
   Journal of Global Optimization, 13, 455-492.
2. Storn, R., & Price, K. (1997).
   *Differential Evolution - A Simple and Efficient Heuristic for Global Optimization
   over Continuous Spaces*.
   Journal of Global Optimization, 11, 341-359.
3. Hansen, N., & Ostermeier, A. (2001).
   *Completely Derandomized Self-Adaptation in Evolution Strategies*.
   Evolutionary Computation, 9(2), 159-195.
4. Wilson, J. T., Hutter, F., & Deisenroth, M. P. (2018).
   *Maximizing Acquisition Functions for Bayesian Optimization*.
   Advances in Neural Information Processing Systems 31.
5. BoTorch documentation for acquisition optimization and mixed optimization.
