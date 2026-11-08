# Candidate constraints：座標・数理契約・backendの制限

## CandidateとOutcomeの制約を分離する

[理論の対応章](../../theory/optimization/05_constraints.md)を数理的な正本とする。実験実施前に既知の`X`だけで判定できるcandidate-space制
約は`robotorchan.optim.constraints.contracts.CandidateConstraints`へ渡す。一方、未知の観測`Y`に対する制約や分類に基づく実行可能確率は、
[Constrained Acquisition](../acquisition/08_constraints.md)のobjective/composition側で扱う。

## 線形・非線形制約の数式と呼び出し規約

- **linear inequality**
  契約：`(indices, coefficients, rhs)`
  可行条件とshape：`sum(X[indices] * coefficients) >= rhs`。BoTorch形式を保持

- **linear equality**
  契約：同形式のtuple
  可行条件とshape：線形式が`rhs`と等しい。数値許容誤差に注意

- **nonlinear equality**
  契約：`(callable, is_intrapoint)` と `nonlinear_equality_tolerance`
  可行条件：`abs(h(X)) <= epsilon`。内部で2つの非線形不等式へ変換

- **nonlinear inequality**
  契約：`(callable, is_intrapoint)`
  可行条件とshape：callableが返す**scalar Tensor >= 0**

- **intrapoint nonlinear**
  契約：`is_intrapoint=True`
  可行条件とshape：callableへの入力は1候補`[d]`

- **interpoint nonlinear**
  契約：`is_intrapoint=False`
  可行条件とshape：callableへの入力はjoint候補`[q,d]`

線形tupleのindicesが1次元なら1点内の制約、2次元ならq座標とd座標を含むinter-point制約である。
`contracts.py`の`CandidateConstraints.__post_init__`は形式を検査し、
`validation.py::validate_candidate_constraints`はboundsとqが分かった段階で範囲・sequentialの制限を確認する。

`q > 1`でinter-point linear/nonlinear制約を指定した場合、`sequential=True`はサポートされず、**joint q-batch探索**が必要となる。
独立にq点を逐次提案すると相互制約が壊れるためである。

## optimizer dispatchの責務

`src/robotorchan/optim/dispatch.py::optimize_acqf`は`bounds`、`q`、`constraints`、`fixed_features`、
`variable_space`を受け取り、`validate_candidate_constraints`と各backendのcapability checksを行う。一般の候補は`[q,d]`で、
戻り値は`(candidate, acquisition_value)`。acquisition valueをblack-boxの新観測`Y`として使ってはならない。

公開dispatcherのnamed backendは`botorch`、`torch_adam`、`torch_adamw`、`torch_sgd`、`random`、`sobol`、`de`、
`cmaes`、`ga`、`pso`、`hybrid`。Mixed BoTorch、one-shot、NSGA-II、
TuRBOなどは同じ`optimizer=`文字列で選ぶ12番目以降のbackendではない。
`optimizer="botorch"`に`MixedVariableSpace.is_mixed`を直接渡すことも拒否されるため、
明示的なmixed backendまたは対応optimizerを使う。

capability descriptorの`constraint_handling`はnative、feasibility-first、penalty、unsupportedなどの戦略を区別する。
* *penalty方式が返した候補をhard-feasibleと断定できない**。制約型・backend別に候補値を検算する。

## 非線形の初期点と探索座標

`validate_nonlinear_initial_conditions`は明示的な`batch_initial_conditions`に対し、`[num_restarts,q,d]`形、
bounds、device・dtype、有限値とcallableのscalar返値を調べる。モデルとboundsを同じdevice/dtypeへ揃え、
gradient backendであればcallableの微分可能性を確認する。inter-pointの条件を持つ初期点は個別候補だけで可否判定しない。

埋め込み・潜在探索は通常、公開Xとは別の内部座標Zで最適化する。元のXへのlinear constraintを無変換でZへ渡すと**別の数学的制約**になるため、
`reject_unmapped_candidate_constraints`は未対応の座標写像を`NotImplementedError`として拒否する。
明示的な逆写像に基づく適切な合成なしに可行性を保証しない。

## Backendごとに異なる可行性の数学的意味

$$
\max_{X\in\mathcal X^q} A(X),\quad
g_j(X)\ge0,\quad h_k(X)=0.
$$

| 経路 | 処理契約 | 制限・数学的意味 |
| --- | --- | --- |
| Continuous BoTorch | NATIVE | 線形・非線形制約をBoTorchへ渡す。inter-point nonlinearも対応するが、feasible初期点が必要 |
| Native mixed BoTorch | NATIVE | fixed_features_listで混合値を列挙。inter-point nonlinearは拒否 |
| Adam / AdamW / SGD | UNSUPPORTED | CandidateConstraintsが非空なら明示拒否 |
| Random / Sobol | FEASIBILITY_FILTER | サンプリングしたjoint候補を制約でフィルタし、可行候補の中から最大値を選ぶ。可行候補が得られなければValueError |
| DE / GA / Mixed GA | FEASIBILITY_FIRST | 制約違反量を用いる順位付け。常に可行点を返す保証ではない |
| CMA-ES / PSO | PENALTY | 探索中に違反量を評価する。返却候補の可行性はbackend実装で別途確認する | 
| Hybrid | GLOBAL→LOCAL | global候補を検証し、local候補が制約違反または獲得関数値を改善しない場合はglobal候補を維持する |
| NSGA-II | 別API | vector objective用であり、scalar獲得関数のcandidate制約dispatchに含まれない |

FEASIBILITY_FILTERはサンプリングした候補から可行なものだけを残す方式である。FEASIBILITY_FIRSTは可行性を順位に反映する方式であり、PENALTYは概念的には次のようにacquisitionと違反量をトレードオフさせる。

$$
A_{\mathrm{penalty}}(X)=A(X)-\lambda V(X).
$$

これは個々のbackendの正確な更新則や係数を再定義した式ではない。具体的な違反量の集約や候補選択はそれぞれのソースで確認する。

### BoTorch形式の入力制約の例

入力に対する残差と、未知出力Yの可行確率は別の量である。非線形制約はcallableの戻り値が0以上で可行、intrapointなら入力[d]、interpointなら[q,d]とする。
gradient backendでは微分可能性が必要となる。BoTorch nativeは非線形制約でfeasibleなbatch_initial_conditionsまたはic_generatorを
要求し、native mixedはinter-point nonlinearを明示拒否する。

## 関連する実装資料

1. Deb, K. (2000). *An Efficient Constraint Handling Method for Genetic Algorithms*. Computer
   Methods in Applied Mechanics and Engineering, 186, 311-338.
2. Lampinen, J. (2002). *A Constraint Handling Approach for the Differential Evolution Algorithm*.
   Proceedings of the Congress on Evolutionary Computation.
3. Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization*. Springer.
4. BoTorch documentation. *Optimization* and candidate-constraint API.

## 制約残差と候補順位の計算

[evaluation.py](../../../src/robotorchan/optim/constraints/evaluation.py)の
`candidate_constraint_violation()`は線形・非線形の違反量を集計します。`feasibility_first_ranks()`は可行候補を違反候補より優先し、
可行群では獲得値、違反群では違反量を順位へ反映します。Theoryのfeasibility-first方針を、数値scoreだけへの罰則加算とは分けて実装しています。

[contracts.py](../../../src/robotorchan/optim/constraints/contracts.py)の
`CandidateConstraints`はimmutableなデータ表現であり、探索更新は行いません。BoTorchのnative制約へ委譲する経路と、
ローカルの違反量で順位付けする経路を同じhard-feasibility保証として扱わないことが重要です。

## 実行例とGuideの対応

このGuideは制約の数学的意味とbackendへの委譲・制限を説明します。実行可能なコード、
候補の確認、可視化は既存Notebookに集約しています。

| 確認したい内容 | Notebook | 対応する節 |
| --- | --- | --- |
| 線形制約・固定変数・候補間制約とjoint q-batch | [31_acquisition_optimization](../../../examples/notebooks/31_acquisition_optimization.ipynb) | 5〜8 |
| 非線形不等式・実行可能領域の可視化 | [38_nonlinear_candidate_constraints](../../../examples/notebooks/38_nonlinear_candidate_constraints.ipynb) | 3〜4 |
| 非線形等式・許容誤差内の候補提案 | [38_nonlinear_candidate_constraints](../../../examples/notebooks/38_nonlinear_candidate_constraints.ipynb) | 4a |
| 非線形候補間制約とjoint q-batch | [38_nonlinear_candidate_constraints](../../../examples/notebooks/38_nonlinear_candidate_constraints.ipynb) | 5〜6 |

Mixed変数を含む候補間制約はbackendごとに対応が異なります。Mixed GA / DEには
公開dispatcherを経由するq-batch制約テストがありますが、Mixed Hybridのq>1における
候補間線形制約はローカル段階で維持できないため明示的に拒否します。
対応状況をすべてのoptimizerへ一般化せず、利用するbackendの制限を確認してください。
