# Hybrid optimization：獲得関数最適化の実装ガイド

## この章の問い

global gradient-free探索の候補をgradient-based local refinementの初期点として渡す段階的最適化。

数理的な正本は[理論の対応章](../../theory/optimization/11_hybrid.md)とする。本章は、そこで定義される問題の**候補の座標系、内部処理、
委譲と設計上の制限**を学ぶための日本語案内である。

## 適用条件と失敗の解釈

joint q×dを保持し、global候補の独立top-qへの置換をしない。非微分posteriorならlocal refinementは不適切。

## Differential EvolutionからBoTorch局所探索へ接続する

Hybrid backendはDEなどのglobal候補をBoTorch局所最適化の初期値へ渡す。内部handoffでは`[q,d]`から`[1,q,d]`へ次元が追加され、
局所探索後には実験候補`[q,d]`を返す。

## 補足となる実装上の設計

#### 目的：実装上の補足

global derivative-free search -> promising joint q-batch -> BoTorch local refinement

#### Initialization としての global search：実装上の補足

通常の multi-start BoTorch optimization では、raw samples などから initial conditions を選びます。

#### Constraint semantics は stage-dependent：実装上の補足

| Global stage | Global constraint handling | Local stage |
| --- | --- | --- |
| DE | feasibility-first | BoTorch native |
| CMA-ES | penalty | BoTorch native |
| Mixed GA | feasibility-first | BoTorch mixed native |
| custom callable | callable の contract に依存 | BoTorch native / mixed native |

#### Nonlinear constraints：実装上の補足

nonlinear candidate constraintsがある場合、局所BoTorch段階には可行な初期条件または対応するinitializerが必要となる。
大域段階の候補を無検査で局所段階へ引き継いでよいとは限らない。

#### Mixed-variable Hybrid：実装上の補足

現在の構成では global optimizer として Mixed GA を利用でき、local stage は BoTorch mixed optimization を使えます。

#### 何を組み合わせているのか：実装上の補足

| Layer | 例 |
| --- | --- |
| Search domain | global box, trust region, embedded polytope |
| Global numerical search | DE, CMA-ES, Mixed GA |
| Hand-off | explicit batch initial condition |
| Local numerical search | BoTorch continuous / mixed optimization |

## 関連する実装資料

非線形候補制約がある場合、現在の実装は局所段階の`batch_limit`を未指定なら1に設定する。大域段階と局所段階で制約の扱いは異なるため、最終候補の可行性を確認する。
詳細は[Hybrid利用ガイド](../../optimization/hybrid-optimizer.md)を参照する。

## globalからlocalへのhandoff

[hybrid.py](../../../src/robotorchan/optim/backends/hybrid.py)の
`optimize_acqf_hybrid()`はglobal optimizerを解決し、
その返却候補`[q,d]`へrestart軸を加えた`[1,q,d]`を局所BoTorchの初期値へ渡します。同じglobal解の複製でrestart数を増やさないため、
現在は複数restartを拒否します。

変数型と制約は両段階へ接続しますが、処理方法はglobal backendとlocal backendで異なります。handoffと引数検査はrobotorchanの実装であり、
局所更新則はBoTorchへの委譲です。global / localの順序だけで大域収束を保証する設計ではありません。
