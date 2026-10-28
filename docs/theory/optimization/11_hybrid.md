# Hybrid Acquisition Optimization

## 11.1 目的

Hybrid acquisition optimization は、1つの numerical optimizer に全探索責務を持たせず、
異なる性質を持つ探索段階を接続します。

代表的な構成は、広域探索で得た候補を初期値として局所探索へ渡す**global-to-local**方式です。
ここでglobalは広い領域を探索する役割を指し、大域最適性の証明を意味しません。

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

局所段階でも、この行列全体を一つの初期解として扱います。
複数の初期解を使うなら、q×dの行列を初期解ごとに用意します。

つまり global stage の上位 q 個の independent point を後から束ねるのではありません。

q-batch acquisition の場合、global stage から local stage まで
`[q, d]` を1つの joint decision として維持する必要があります。

## 11.5 Initialization としての global search

二段階法では、広域探索が局所法の初期値生成を兼ねます。
局所段階のために別の無関係な候補を生成するのではなく、得られた有望なq点組から改善を始めます。
一組だけを渡す設計と、異なる複数の組を渡して再始動する設計は区別します。
どちらの場合も、局所法が要求する可行性や勾配の条件を引き渡し時に満たす必要があります。

## 11.6 Local restart diversity

同じ初期点を複製しても探索 basin の多様性は増えません。
複数restartを使う場合は、異なる広域候補を初期値として渡すことが有効です。

同じ可行集合 $\mathcal F$、同じ固定された効用 $\hat\alpha$ を使い、可行な初期解を $X_0$、
局所法の可行な返り値を $X_L$ とします。両者を再比較して良い方を残せば
$\hat\alpha(X_{return})\ge\hat\alpha(X_0)$ です。
これが二段階法で初期解を保存する理由です。局所法が必ず改善するという仮定は不要です。
ただしMC標本を途中で変えた場合や、最後に修復して点を動かした場合は同じ比較にはなりません。
修復後の点で効用と制約を再評価し、標本近似上の改善を真の期待効用の改善と同一視しません。

## 11.7 Global optimizer と gradient

global stage は candidate gradient を必要としない optimizer を使えます。

しかし local stage が gradient-based optimization を使う場合、
Hybrid 全体としては local refinement に必要な differentiability を失ってはいけません。

したがって

    global stage does not require gradients

と

    hybrid pipeline does not require gradients

は同じ意味ではありません。

したがって、広域段階で不連続な修復を使えても、局所段階で微分する目的関数と座標変換には必要な正則性が求められます。

## 11.8 Constraint semantics は stage-dependent

candidate constraints は global stage と local stage の両方へ渡されます。

ただし enforcement method は同一ではありません。

| Global stage | Global constraint handling | Local stage |
| --- | --- | --- |
| ペナルティで探索 | 違反候補が残り得る | 可行初期点を要するなら先に修復・選別する |
| 可行性優先で探索 | 可行点があれば優先する | 引き渡す点の残差を再確認する |
| 可行な変数変換で探索 | 変換が保つ制約に限定される | 元の変数と制約へ整合させる |

これは制約処理の組合せ例であり、各手法に一意の方式が決まるわけではありません。
したがって Hybrid の制約処理を単一の「penalty method」と理解するのは不正確です。

最終 candidate は local stage の constraint semantics も満たす必要がありますが、
global stage で infeasible landscape をどう探索するかは選択した optimizer に依存します。

## 11.9 Nonlinear constraints

制約付き局所最適化では初期点の実行可能性と数値安定性に注意が必要です。

global stage 側の nonlinear constraint handling は選択した global optimizer の方式に従います。

したがって同じ constraint object を共有しても、数値的な enforcement mechanism は
stage 間で同一とは限りません。

## 11.10 Fixed features

固定座標は両段階で同じ値を維持します。
例えば共通の固定条件 $x_j=a$ と、ある離散割当の条件 $x_j=c_j$ があるなら、
$a\ne c_j$ の枝の可行集合は空です。後から片方を上書きして解決しません。
固定値が元の許容集合に属することも必要です。
段階間で固定条件を変えると、異なる最適化問題を解くことになります。

## 11.11 Mixed-variable Hybrid

Mixed-variable hybrid では構造化された変数空間を扱います。

integer / categorical dimension が存在する場合、global stage と local stage の双方で
structured variable semantics を保持する必要があります。

連続局所法を各離散割当に適用する構成なら、枝ごとに整数・カテゴリ座標を固定し、
連続座標だけを微分して改善します。全変数を一度連続化して後から丸める方法とは異なります。

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
両者を組み合わせる場合は局所領域、状態更新、restartと初期化の整合が必要です。
この場合の広域段階も信頼領域内を探索します。globalという名称だけで、元の全領域を探索したとはいえません。

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
| Hand-off | q点組を保った初期解の引き渡し |
| Local numerical search | 勾配法・制約付き局所法による改善 |

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

## この章で覚えておくこと

二段階探索では、広域段階の有望解を局所段階へ引き継ぎます。
同じ効用・可行集合・座標の意味を保ち、各段階で必要な微分可能性と可行な初期値を用意します。
初期解を保存して比較することで、同じ標本近似上での不要な悪化を避けられます。

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
