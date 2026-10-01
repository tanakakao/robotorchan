# Hybrid Acquisition Optimization

## 11.1 目的

Hybrid acquisition optimization は、1つの numerical optimizer に全探索責務を持たせず、
異なる性質を持つ探索段階を接続します。

robotorchan の現在の hybrid backend は

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

## 11.3 robotorchan の現在の構成

`optimize_acqf_hybrid()` の global stage は現在、

- Differential Evolution
- CMA-ES
- Mixed Genetic Algorithm
- compatible custom callable

を受け取れます。

local stage は BoTorch backend です。

continuous problem では `optimize_acqf_botorch()`、
structured mixed problem では `optimize_acqf_mixed_botorch()` を使います。

したがって「Hybrid」という独立した数値アルゴリズムがあるのではなく、
global optimizer と BoTorch local optimizer の orchestration です。

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

## 11.6 現在の local restart contract

現在の実装では `num_restarts=1` のみを受け付けます。

同じ global candidate を複製して複数 restart としても、
initialization diversity は増えません。

そのため

    num_restarts > 1

は明示的に拒否されます。

将来複数 restart を提供するなら、異なる global seeds / basins から
distinct initial conditions を生成する必要があります。

## 11.7 Global optimizer と gradient

global stage は candidate gradient を必要としない optimizer を使えます。

しかし local stage は BoTorch の gradient-based optimization を使うため、
Hybrid 全体としては local refinement に必要な differentiability を失ってはいけません。

したがって

    global stage does not require gradients

と

    hybrid pipeline does not require gradients

は同じ意味ではありません。

現在の capability descriptor が `requires_grad=True` とするのは、
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

    batch_limit = 1

が default として設定されます。

これは constrained local optimization の runtime contract に合わせるためです。

global stage 側の nonlinear constraint handling は選択した global optimizer の方式に従います。

したがって同じ constraint object を共有しても、数値的な enforcement mechanism は
stage 間で同一とは限りません。

## 11.10 Fixed features

直接 `optimize_acqf_hybrid()` を使う場合、`fixed_features` は global stage に渡され、
local stage にも引き継がれます。

Mixed local stage では common fixed features と
`mixed_fixed_features_list` の各 discrete configuration を merge します。

同じ dimension に矛盾する値が指定された場合は error になります。

ただし後述する named dispatcher では、現在の public capability validation により
`optimizer="hybrid"` と `fixed_features` の組合せは拒否されます。

## 11.11 Mixed-variable Hybrid

direct hybrid backend は `MixedVariableSpace` を受け取れます。

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

## 11.13 Direct backend と named dispatcher

robotorchan には2つの利用層があります。

### Direct backend

    optimize_acqf_hybrid(...)

現在の direct backend は、

- `fixed_features`
- `variable_space`
- `mixed_fixed_features_list`

を扱う実装経路を持ちます。

### Named dispatcher

    optimize_acqf(..., optimizer="hybrid")

named dispatcher は最初に `HYBRID_OPTIMIZER_CAPABILITIES` で要求を検証します。

現在この descriptor は

- `mixed=False`
- `integer=False`
- `categorical=False`
- `fixed_features=False`

なので、direct backend に実装があっても dispatcher 経由では
Mixed / fixed-feature request が事前に拒否されます。

これは現在の **API surface 差** であり、両経路を同一 capability とみなしてはいけません。

## 11.14 Capability metadata の限界

Hybrid optimizer の capability は、選択した global stage に依存します。

たとえば constraint handling は

    DE       -> feasibility-first
    CMA-ES   -> penalty
    Mixed GA -> feasibility-first

です。

そのため1つの static `OptimizerCapabilities` だけでは、
すべての hybrid composition を完全には表現できません。

現在の `HYBRID_OPTIMIZER_CAPABILITIES` は named dispatcher の validation contract として
扱うべきであり、direct backend が内部的に持つ全機能の一覧ではありません。

## 11.15 Custom global optimizer

`global_optimizer` には compatible callable を渡せます。

必要な基本 contract は

    optimizer(
        acq_function,
        bounds,
        q,
        **global_options,
    ) -> (candidates, acquisition_value)

という形です。

しかし callable を受け付けることは、任意の optimizer が安全に組み合わせられることを
意味しません。

constraints、fixed features、q-batch、variable space、dtype/device など、
hybrid backend が渡す引数と semantics を global callable が正しく処理する必要があります。

## 11.16 CPU / device boundary

現在の hybrid orchestration は numerical optimizer layer では CPU-oriented です。

特に SciPy Differential Evolution や CMA-ES を global stage に使う場合、
Tensor-native GPU population optimization と同じ device-native pipeline ではありません。

したがって surrogate posterior がGPUで計算可能であることと、
hybrid numerical optimization 全体がGPU-nativeであることは区別する必要があります。

## 11.17 Hybrid と sequential optimization

現在の `HYBRID_OPTIMIZER_CAPABILITIES` は `sequential=False` です。

Hybrid backend は joint q-batch を global-to-local に refine する設計であり、
q=1 を繰り返して `X_pending` を更新する sequential orchestration を
backend 内には持ちません。

したがって

    joint hybrid q-batch

と

    sequential greedy batch generation

は別の構造です。

## 11.18 Hybrid と one-shot acquisition

one-shot acquisition は auxiliary variables を含む augmented q-batch を最適化します。

通常の q-batch hand-offだけを前提にした Hybrid backendへ one-shot acquisition を渡す場合、
augmented candidate semantics、initial-condition shape、candidate extraction が
正しく維持される必要があります。

現在の Hybrid backendは one-shot 専用 initializer / extraction path を所有しません。

したがって one-shot acquisition については、
専用の BoTorch / robotorchan one-shot optimization contract を優先します。

## 11.19 Hybrid と Trust Region

Hybrid optimization と Trust Region は異なる軸です。

Hybrid は

    global optimizer -> local optimizer

という optimizer composition を表します。

TuRBO は

    global domain -> adaptive local domain

という search geometry / state adaptation を表します。

理論上は組み合わせ可能でも、bounds、state update、restart、initialization、
constraint semantics を統合した専用 contract が必要です。

現在の hybrid backend が TuRBO state を自動管理するわけではありません。

## 11.20 Hybrid と high-dimensional embedding

REMBO、HeSBO、ALEBO、BAxUS では optimization coordinate が
original/public space と異なる場合があります。

Hybrid optimization を reduced search space で使うには、

    global search coordinate
    -> local refinement coordinate
    -> public candidate coordinate

の mapping を両 stage で一致させる必要があります。

現在の generic hybrid backend は arbitrary embedding の coordinate conversion を
自動構築しません。

したがって search embedding と Hybrid を自由合成できるとはみなしません。

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

## 11.23 実装上の contract

現在の実装に対応する主要 contract は次です。

- global stage は DE / CMA-ES / Mixed GA / compatible callable
- local stage は BoTorch continuous または mixed optimization
- global candidate は explicit `batch_initial_conditions` として渡す
- joint q-batch semantics を stage 間で維持する
- local `raw_samples` は使わない
- local restart は現在1つだけ
- constraints は両 stage に渡すが enforcement mechanism は stage-dependent
- nonlinear constraints では local `batch_limit=1` を default にする
- direct backend は Mixed / fixed-feature 実装経路を持つ
- named dispatcher の Hybrid capability は現在 Mixed / fixed features を拒否する
- custom global optimizer は明示的な compatibility contract を負う
- Hybrid 自体は sequential / one-shot / TuRBO / embedding orchestration を所有しない
- current numerical orchestration は GPU-native とはみなさない

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
