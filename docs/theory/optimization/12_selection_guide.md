# Acquisition Optimizer Selection Guide

## 12.1 この章の目的

acquisition optimizer は手法名だけでなく、最適化問題の構造に基づいて選びます。

確認する軸は主に、

    variable type
    -> candidate constraints
    -> gradient availability
    -> q semantics
    -> search geometry
    -> evaluation cost

です。

理論上の適用可能性と、個別ライブラリの実装上の対応状況は区別します。

## 12.2 まず変数型を確認する

連続変数は勾配法や連続探索に適し、整数・カテゴリ変数にはその離散構造を保つ探索が必要です。
unordered categorical に数値的な距離を仮定しないことが重要です。

## 12.3 Smooth continuous acquisition

滑らかな連続 acquisition では multi-start gradient-based optimization が有力です。
ただし非凸性があるため、局所解から大域最適性は保証されません。

## 12.4 First-order local optimizer

SGD、Adam、AdamW などの一次法は candidate gradient を用います。
制約を満たすためには射影・変数変換などの追加設計が必要になる場合があります。

## 12.5 Sampling

Random / Sobol sampling は勾配を必要とせず、探索の基準手法として有用です。
高次元では被覆に必要な候補数が増えることに注意します。

## 12.6 Differential Evolution

DE は個体間差分に基づく population search です。
整数変数には repair を組み込めますが、unordered categorical には type-aware な扱いが必要です。

## 12.7 CMA-ES

CMA-ES は共分散を適応させる連続空間の分布ベース探索です。
制約や離散変数を扱う場合は、別途適切な処理が必要です。

## 12.8 Genetic Algorithm

GA は selection、crossover、mutation により探索します。
Mixed GA では変数型ごとに演算子を変更することでカテゴリ構造を維持できます。

## 12.9 Particle Swarm Optimization

PSO は個体と群れの最良位置を使って探索を更新します。
連続空間以外へ適用する場合は位置更新や repair の意味を定義する必要があります。

## 12.10 Hybrid

Hybrid optimization は広域探索から局所 refinement へ初期点を引き渡します。
各段階の制約・座標系・勾配要件を整合させる必要があります。

## 12.11 NSGA-II

NSGA-II は vector-valued objective の Pareto set を近似する手法です。
多目的 BO の scalar acquisition を最適化する場合に必須というわけではありません。

## 12.12 Constraint handling は boolean ではない

native enforcement、feasibility-first、penalty、projection、repair、rejection は
異なる数学的な制約処理方式です。実行可能性の保証と探索挙動も異なります。

## 12.13 Native / feasibility-first / penalty

`native` は numerical optimizer 自身の constrained optimizationへ委譲します。

`feasibility-first` は feasible candidate を infeasible candidate より優先し、
infeasible 同士では violation を使って比較します。

`penalty` は constraint violation を search objective に加えます。

同じ constraint class を受け取れても、探索挙動は異なります。
したがって制約処理方式の違いを無視して探索手法を交換可能とはみなしません。

## 12.14 Inter-point constraints

inter-point constraint は q-batch 全体の候補間関係を定義します。
単一点ずつの greedy optimization に分解できるとは限りません。

## 12.15 Joint q-batch

多くの scalar acquisition backend は `q_batch=True` です。

これは

    X in R^(q x d)

を1つの decision variable として acquisition を最適化できることを表します。

「候補をq個返せる」ことだけを意味しません。

Random Search でも population method でも、joint acquisition の場合は
`[q, d]` 単位の評価を維持する必要があります。

## 12.16 Sequential optimization

sequential greedy generation では既選択候補を後続の獲得関数に反映します。
単に同じ単一点最適化を繰り返すだけでは同等になりません。

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

fixed features は探索空間の一部座標を定数に固定する操作です。
整数・カテゴリ変数では固定値が許容集合に属する必要があります。

## 12.19 計算環境と評価コスト

acquisition の評価コストと数値 optimizer の評価回数を区別します。
GPU 上で surrogate を評価できても、探索手法全体が GPU-native とは限りません。

## 12.20 High-dimensional search

高次元問題では optimizer backend だけでなく search coordinates を検討します。

高次元探索には

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

Random / Sobol による mixed-variable sampling では、
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

最適化の構造は概念的に

    acquisition
      -> search strategy
      -> optimizer backend
      -> candidate

と分けられます。

REMBO、BAxUS、TuRBOは search strategy 側、
BoTorch、DE、GA、PSOなどは numerical backend 側です。

Hybridは複数backendをstageとして接続する orchestration です。

この区別を保つと、機能の組合せ可否を名前だけで判断することを避けられます。

## 12.26 理論に基づく選択フロー

1. 連続・整数・カテゴリ変数の構造を確認する
2. 制約の種類と実行可能集合を確認する
3. joint q / sequential / one-shot を区別する
4. candidate gradient の利用可能性を確認する
5. fixed features と探索座標系を確認する
6. 評価コストと計算予算を考慮する

## 12.27 理論と実装の対応

数学的に適用可能な手法が、特定の実装ですべて利用可能とは限りません。
実際の対応状況は [Optimization guides](../../optimization/README.md) を参照してください。

## 12.28 特殊な最適化構造

Mixed、one-shot、vector-valued objective、trust-region search は通常の
continuous scalar optimization と異なる構造を持ちます。

## 12.29 適用可能性の限界

ある探索手法が変数型に適合しても、任意の制約、one-shot、embedding、
並列化との組合せが数学的・数値的に適切とは限りません。

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
