# Acquisition Optimizer Selection Guide

## 12.1 この章の目的

acquisition optimizer は名前から選ぶのではなく、問題と runtime contract を分解して選びます。

確認する軸は主に、

    variable type
    -> candidate constraints
    -> gradient availability
    -> q semantics
    -> search geometry
    -> runtime / device

です。

理論的に利用可能な手法と、現在の robotorchan API が公開する組合せは同じではありません。
最終的な source of truth は実装と capability metadata です。

## 12.2 まず変数型を確認する

continuous-only problem と integer / categorical を含む問題では探索空間が異なります。

| Backend | Continuous | Integer | Categorical | Mixed |
| --- | --- | --- | --- | --- |
| BoTorch named | Yes | No | No | No |
| BoTorch mixed direct | Yes | Yes | Yes | Yes |
| Torch Adam/AdamW/SGD | Yes | No | No | No |
| Random / Sobol | Yes | Yes | Yes | Yes |
| DE | Yes | Yes | No | Yes |
| CMA-ES | Yes | No | No | No |
| GA | Yes | Yes | Yes* | Yes* |
| PSO | Yes | Yes | No | Yes |
| Hybrid named | Yes | No | No | No |
| Hybrid direct | Yes | implementation pathあり | implementation pathあり | implementation pathあり |

`GA` の categorical / mixed capability は `MixedVariableSpace` が与えられた場合に
Mixed GA backendへdispatchされる経路を含みます。

Hybrid direct と named dispatcher の差は
[Hybrid Optimization](11_hybrid.md) を参照してください。

## 12.3 Smooth continuous acquisition

candidate gradient が利用でき、continuous domain であれば、
BoTorch multi-start optimization は標準的な構成です。

robotorchan の named `botorch` backend は

- continuous
- q-batch
- sequential
- fixed features
- linear / nonlinear candidate constraints

を扱います。

ただし `MixedVariableSpace` を named `optimizer="botorch"` に直接渡す経路はありません。
Mixed problem では explicit BoTorch mixed backend など、structured-space対応経路を使います。

## 12.4 Torch local optimizer

`torch_adam`、`torch_adamw`、`torch_sgd` は
Tensor-native gradient optimization です。

現在の capability は

- continuous
- q-batch
- fixed features
- GPU-native
- batch evaluation

ですが、candidate constraints と sequential dispatch は公開していません。

「gradient-based」であることだけを理由に BoTorch backend と同じ制約機能を
持つとはみなしません。

## 12.5 Sampling

Random / Sobol search は gradient を必要とせず、Tensor-native に多数候補を評価できます。

現在の capability metadata では

- continuous / integer / categorical / mixed
- q-batch
- fixed features
- GPU
- batch evaluation

を持ちます。

一方、一般の `CandidateConstraints` は capability 上 unsupported です。

sampling は baseline、initial exploration、gradient-free search として有用ですが、
constraint-aware optimizer と同一ではありません。

## 12.6 Differential Evolution

DE は gradient-free な population search です。

現在の capability は

- continuous
- integer
- mixed
- q-batch
- fixed features
- linear / nonlinear / inter-point constraints

です。

categorical capability は持ちません。

constraint handling は現在 feasibility-first です。
SciPy orchestrationを含むため、numerical optimizer layer は native GPU backend ではありません。

## 12.7 CMA-ES

CMA-ES は continuous derivative-free optimization を提供します。

現在の capability は

- continuous
- q-batch
- linear / nonlinear / inter-point constraints

です。

integer、categorical、Mixed、fixed features は capability 上公開していません。

constraint handling は penalty です。
optional dependency と CPU-oriented orchestration も runtime selection に影響します。

## 12.8 Genetic Algorithm

continuous GA は gradient-free population search です。

通常の GA capability は

- continuous
- q-batch
- fixed features
- GPU
- batch evaluation
- candidate constraints

を持ちます。

`optimizer="ga"` に mixed `MixedVariableSpace` を渡すと、
dispatcher は Mixed GA capability と backend を選択します。

Mixed GA は integer / categorical / mixed variable を扱い、
constraint handling は feasibility-first です。

## 12.9 Particle Swarm Optimization

PSO は gradient-free population search です。

現在の capability は

- continuous
- integer
- mixed
- q-batch
- fixed features
- GPU
- batch evaluation
- candidate constraints

です。

categorical capability は公開していません。
constraint handling は penalty です。

## 12.10 Hybrid

Hybrid backend は

    derivative-free global search
      -> explicit initial condition
      -> BoTorch local refinement

です。

named dispatcher の current capability は continuous q-batch と constraints を中心とし、
Mixed / fixed features / sequential は公開していません。

direct `optimize_acqf_hybrid()` は Mixed / fixed-feature 実装経路を持つため、
named capability と direct backend capability を混同しないことが重要です。

## 12.11 NSGA-II

NSGA-II は scalar acquisition optimizer registry の named backend ではなく、
specialist optimizer です。

現在の capability metadata は

- continuous
- gradient-free
- GPU
- batch evaluation

で、通常の `q_batch` capability は false です。

NSGA-II の population / Pareto semantics を、
scalar acquisition の joint q-batch semantics と同一視しません。

## 12.12 Constraint handling は boolean ではない

constraint support は「対応 / 非対応」だけでは不十分です。

現在の主要 semantics は次です。

| Backend | Linear inequality | Linear equality | Nonlinear | Inter-point nonlinear |
| --- | --- | --- | --- | --- |
| BoTorch | native | native | native | native |
| BoTorch mixed | native | native | native | unsupported |
| DE | feasibility-first | feasibility-first | feasibility-first | feasibility-first |
| CMA-ES | penalty | penalty | penalty | penalty |
| GA | feasibility-first | feasibility-first | feasibility-first | feasibility-first |
| Mixed GA | feasibility-first | feasibility-first | feasibility-first | feasibility-first |
| PSO | penalty | penalty | penalty | penalty |
| Hybrid named metadata | penalty | penalty | penalty | penalty |
| Torch / Random / Sobol / NSGA-II | unsupported | unsupported | unsupported | unsupported |

Hybrid の実際のglobal-stage semanticsは選択した global optimizer に依存し、
local stage は BoTorch native constraint handling です。

## 12.13 Native / feasibility-first / penalty

`native` は numerical optimizer 自身の constrained optimizationへ委譲します。

`feasibility-first` は feasible candidate を infeasible candidate より優先し、
infeasible 同士では violation を使って比較します。

`penalty` は constraint violation を search objective に加えます。

同じ constraint class を受け取れても、探索挙動は異なります。
したがって capability table の boolean だけで backend を交換可能とはみなしません。

## 12.14 Inter-point constraints

inter-point constraint は1 candidate rowだけでは判定できず、
joint q-batch 全体を必要とします。

BoTorch continuous、DE、CMA-ES、GA、Mixed GA、PSOなどには
対応経路がありますが、BoTorch mixed capability は
inter-point nonlinear constraint を公開していません。

また TuRBO Thompson sampling の finite-pool filtering も
inter-point constraints を扱いません。

q-batch support と inter-point constraint support は別の capability です。

## 12.15 Joint q-batch

多くの scalar acquisition backend は `q_batch=True` です。

これは

    X in R^(q x d)

を1つの decision variable として acquisition を最適化できることを表します。

「候補をq個返せる」ことだけを意味しません。

Random Search でも population method でも、joint acquisition の場合は
`[q, d]` 単位の評価を維持する必要があります。

## 12.16 Sequential optimization

sequential optimization は q-batch support とは別です。

named capability で sequential を公開している標準経路は BoTorch backend です。

sequential greedy generation では、選択済み候補を pending context として次の q=1 optimizationへ
反映する必要があります。

単に q=1 optimizer を q 回呼ぶだけでは、acquisition state が更新されなければ
同じ意味になりません。

## 12.17 One-shot acquisition

qKG、multi-step lookaheadなどの one-shot acquisition は、
実 candidate に auxiliary variables を加えた augmented batch を最適化します。

そのため optimizer が通常の q-batch に対応していても、
one-shot acquisition に自動対応するとは限りません。

確認する必要があるのは

- augmented q の生成
- one-shot initial conditions
- constraint index semantics
- actual candidate extraction
- Mixed assignment semantics

です。

one-shot では専用 optimization path を優先します。

## 12.18 Fixed features

fixed features は一部 coordinate を既知値に固定する contract です。

現在の named capabilityでは BoTorch、Torch、Random/Sobol、DE、GA、PSO が対応します。

CMA-ES と Hybrid named capability は対応を公開していません。

ただし Hybrid direct backend には fixed-feature path があるため、
利用する API surface を確認する必要があります。

## 12.19 GPU と batch evaluation

GPU capability は surrogate がGPUに置けるかではなく、
numerical optimizer が device-native に動くかを表します。

現在 Tensor-native として扱われる主な backend は

- Torch optimizer
- Random / Sobol
- GA / Mixed GA
- PSO
- NSGA-II

です。

SciPy DE、CMA-ES、Hybrid orchestrationは numerical optimizer layer ではCPU-orientedです。

## 12.20 High-dimensional search

高次元問題では optimizer backend だけでなく search coordinates を検討します。

robotorchan には

- LatentSpace
- REMBO
- HeSBO
- ALEBO
- BAxUS

などの search strategy があります。

これらは「DEかBoTorchか」と同じ選択軸ではありません。

    search strategy
      -> search coordinates / domain
      -> optimizer backend

という別レイヤーです。

## 12.21 Trust Region

TuRBO は optimizer backend の名前ではなく、stateful search geometry です。

current incumbent 周辺の trust region を適応させ、
その local bounds 内で acquisition optimization または Thompson sampling を行います。

したがって高次元・局所探索を検討するときは

    surrogate
    acquisition
    trust-region strategy
    numerical optimizer

を分けて考えます。

## 12.22 Mixed-variable selection

Mixed problem では、少なくとも次を確認します。

1. variable type を optimizer が保持するか
2. categorical variable を連続値として誤って補間しないか
3. fixed-feature enumeration が必要か
4. candidate constraints が structured variable と整合するか
5. q-batchでも discrete assignment が維持されるか

現在の主な経路は explicit BoTorch mixed、Mixed GA、
および対応範囲内のDE / PSOです。

Random / Sobolにも mixed capabilityがありますが、
general candidate constraintsはunsupportedです。

## 12.23 MultiFidelity selection

MultiFidelity model を使うことと、optimizer backend の選択は別です。

fidelity coordinate を通常の design variable と同じように局所化すると、
探索意図が変わる場合があります。

TuRBO MultiFidelity path では design dimensionsだけをlocalizeし、
fidelity dimensionsはglobal rangeに残します。

また qMFKG など one-shot acquisitionでは augmented optimization contract も確認します。

## 12.24 Multi-objective selection

multi-objective BOでも、acquisition output自体がscalarなら通常の scalar optimizerを
利用できる場合があります。

一方、NSGA-IIのように vector-valued numerical objective / Pareto populationを扱う手法は
別の optimization semantics です。

したがって

    multi-objective BO
      != always multi-objective numerical optimizer

です。

acquisitionのoutput shapeとoptimizerのobjective contractを確認します。

## 12.25 Search strategy と backend を分ける

robotorchan の optimization stack は概念的に

    acquisition
      -> search strategy
      -> optimizer backend
      -> candidate

と分けられます。

REMBO、BAxUS、TuRBOは search strategy 側、
BoTorch、DE、GA、PSOなどは numerical backend 側です。

Hybridは複数backendをstageとして接続する orchestration です。

この区別を保つと、機能の組合せ可否を名前だけで判断することを避けられます。

## 12.26 実装ベースの選択フロー

実際には次の順序で絞り込みます。

    1. continuous / integer / categorical / mixed
    2. CandidateConstraints の種類
    3. joint q / sequential / one-shot
    4. gradient availability
    5. fixed features
    6. high-dimensional / trust-region geometry
    7. GPU / batch evaluation
    8. named dispatcher か specialist direct API か
    9. capability metadata と runtime validation を確認

この順序なら「理論的に使えそうだが現在のAPIでは拒否される」組合せを早めに検出できます。

## 12.27 Capability metadata は runtime contract

`OptimizerCapabilities` は説明用ラベルだけではありません。

named dispatcher は

- variable-space support
- fixed features
- sequential
- constraint support

を optimization 開始前に検証します。

unsupported combination は黙って近似するのではなく、原則として明示的に拒否します。

この fail-fast behavior は optimizer selection の一部です。

## 12.28 Specialist API

すべての optimizer が `OPTIMIZER_CAPABILITIES` registry に入るわけではありません。

registry は scalar `robotorchan.optim.optimize_acqf()` が受け取る named backend を対象にします。

BoTorch mixed、Mixed GA、NSGA-IIなどには specialist capability descriptor / direct API があり、
named registry にないこと自体は「未実装」を意味しません。

## 12.29 Compatibility matrix の読み方

表で Yes でも、次の条件まで自動的に保証するわけではありません。

- arbitrary one-shot acquisition
- arbitrary search embedding
- arbitrary custom constraints
- arbitrary custom global optimizer
- cross-device bitwise reproducibility
- search strategyとの任意合成

capability はその backend が公開している contract の範囲で解釈します。

## 12.30 最終確認

optimizerを変更するときは、少なくとも

    variable semantics
    q semantics
    constraint semantics
    initialization semantics
    coordinate system
    dtype / device
    API surface

が維持されるか確認します。

同じ acquisition value を最大化する backend でも、
これらの contract が異なれば完全なdrop-in replacementではありません。

## 12.31 関連ドキュメント

理論の詳細はこの optimization theory subtree の各章を参照してください。

runtime-facingな最新情報は

- [Optimization guides](../../optimization/README.md)
- [Constraint handling metadata](../../optimization/constraint-handling-metadata.md)
- [Initialization compatibility](../../optimization/initialization-compatibility.md)
- [Optimizer runtime contract](../../optimization/runtime-contract.md)

を確認してください。

## 参考文献

1. Wilson, J. T., Hutter, F., & Deisenroth, M. P. (2018).
   *Maximizing Acquisition Functions for Bayesian Optimization*.
   Advances in Neural Information Processing Systems 31.
2. Storn, R., & Price, K. (1997).
   *Differential Evolution - A Simple and Efficient Heuristic for Global Optimization
   over Continuous Spaces*.
   Journal of Global Optimization, 11, 341-359.
3. Hansen, N., & Ostermeier, A. (2001).
   *Completely Derandomized Self-Adaptation in Evolution Strategies*.
   Evolutionary Computation, 9(2), 159-195.
4. Eriksson, D., Pearce, M., Gardner, J., Turner, R. D., & Poloczek, M. (2019).
   *Scalable Global Optimization via Local Bayesian Optimization*.
   Advances in Neural Information Processing Systems 32.
5. BoTorch documentation for acquisition optimization, mixed optimization,
   one-shot acquisition functions, and high-dimensional Bayesian optimization.
