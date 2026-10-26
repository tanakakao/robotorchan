# Candidate constraints：座標・数理契約・backendの制限

## 1. CandidateとOutcomeの制約を分離する

[理論の対応章](../../../../theory/optimization/05_constraints.md)を数理的な正本とする。実験実施前に既知の`X`だけで判定できるcandidate-space制約は`robotorchan.optim.constraints.contracts.CandidateConstraints`へ渡す。一方、未知の観測`Y`に対する制約や分類に基づく実行可能確率は、[Constrained Acquisition](../acquisition/08_constraints.md)のobjective/composition側で扱う。

## 2. 線形・非線形制約の数式と呼び出し規約

| 制約 | 契約 | 可行条件とshape |
| --- | --- | --- |
| linear inequality | `(indices, coefficients, rhs)` | `sum(X[indices] * coefficients) >= rhs`。BoTorch形式を保持 |
| linear equality | 同形式のtuple | 線形式が`rhs`と等しい。数値許容誤差に注意 |
| nonlinear inequality | `(callable, is_intrapoint)` | callableが返す**scalar Tensor >= 0** |
| intrapoint nonlinear | `is_intrapoint=True` | callableへの入力は1候補`[d]` |
| interpoint nonlinear | `is_intrapoint=False` | callableへの入力はjoint候補`[q,d]` |

線形tupleのindicesが1次元なら1点内の制約、2次元ならq座標とd座標を含むinter-point制約である。`contracts.py`の`CandidateConstraints.__post_init__`は形式を検査し、`validation.py::validate_candidate_constraints`はboundsとqが分かった段階で範囲・sequentialの制限を確認する。

`q > 1`でinter-point linear/nonlinear制約を指定した場合、`sequential=True`はサポートされず、**joint q-batch探索**が必要となる。独立にq点を逐次提案すると相互制約が壊れるためである。

## 3. optimizer dispatchの責務

`src/robotorchan/optim/dispatch.py::optimize_acqf`は`bounds`、`q`、`constraints`、`fixed_features`、`variable_space`を受け取り、`validate_candidate_constraints`と各backendのcapability checksを行う。一般の候補は`[q,d]`で、戻り値は`(candidate, acquisition_value)`。acquisition valueをblack-boxの新観測`Y`として使ってはならない。

公開dispatcherのnamed backendは`botorch`、`torch_adam`、`torch_adamw`、`torch_sgd`、`random`、`sobol`、`de`、`cmaes`、`ga`、`pso`、`hybrid`。Mixed BoTorch、one-shot、NSGA-II、TuRBOなどは同じ`optimizer=`文字列で選ぶ12番目以降のbackendではない。`optimizer="botorch"`に`MixedVariableSpace.is_mixed`を直接渡すことも拒否されるため、明示的なmixed backendまたは対応optimizerを使う。

capability descriptorの`constraint_handling`はnative、feasibility-first、penalty、unsupportedなどの戦略を区別する。**penalty方式が返した候補をhard-feasibleと断定できない**。制約型・backend別に候補値を検算する。

## 4. 非線形の初期点と探索座標

`validate_nonlinear_initial_conditions`は明示的な`batch_initial_conditions`に対し、`[num_restarts,q,d]`形、bounds、device・dtype、有限値とcallableのscalar返値を調べる。モデルとboundsを同じdevice/dtypeへ揃え、gradient backendであればcallableの微分可能性を確認する。inter-pointの条件を持つ初期点は個別候補だけで可否判定しない。

埋め込み・潜在探索は通常、公開Xとは別の内部座標Zで最適化する。元のXへのlinear constraintを無変換でZへ渡すと**別の数学的制約**になるため、`reject_unmapped_candidate_constraints`は未対応の座標写像を`NotImplementedError`として拒否する。明示的な逆写像に基づく適切な合成なしに可行性を保証しない。

## 5. テストによる検証と未検証の境界

[旧Optimization Mapping](../../optimization.md)で示される`tests/optim/contracts/test_optimizer_dispatch.py`はdispatch引数・不正組合せの拒否等を検査する。`tests/optim/integration/test_nonlinear_candidate_constraints_e2e.py`は選ばれた問題におけるcandidate-level feasibilityを1e-6許容誤差で確認する。この証拠は一部の経路の具体的なcandidateの可行性であり、全backend・すべてのconstraint・大域最適性を示さない。

[Optimization verification索引](../verification/optimization.md)と[旧Verification Mapping](../../verification.md)は数値参照とその制限を保持する。旧Mappingの各区間が検収されるまでは原文を削除しない。

## 6. Backendごとに異なる可行性の数学的意味

旧Optimization Mappingの固定区間246–257の制約処理表と、src/robotorchan/optim/capabilities.py、backends/botorch.py、backends/torch.py、optim/one_shot/mixed.pyを対応付ける。候補空間の制約は獲得関数値とは別に評価し、backendごとに異なる方法で探索に反映する。

$$
\max_{X\in\mathcal X^q} A(X),\quad
g_j(X)\ge0,\quad h_k(X)=0.
$$

| 経路 | 処理契約 | 制限・数学的意味 |
| --- | --- | --- |
| Continuous BoTorch | NATIVE | 線形・非線形制約をBoTorchへ渡す。inter-point nonlinearも対応するが、feasible初期点が必要 |
| Native mixed BoTorch | NATIVE | fixed_features_listで混合値を列挙。inter-point nonlinearは拒否 |
| Adam / AdamW / SGD | UNSUPPORTED | CandidateConstraintsが非空なら明示拒否 |
| Random / Sobol | UNSUPPORTED | 候補のサンプリング・argmaxだけでは制約の可行性を保証しない |
| DE / GA / Mixed GA | FEASIBILITY_FIRST | 制約違反量を用いる順位付け。常に可行点を返す保証ではない |
| CMA-ES / PSO / Hybrid | PENALTY | acquisitionスコアから制約違反分を減点する近似。有限ペナルティでは違反が残り得る |
| NSGA-II | 別API | vector objective用であり、scalar獲得関数のcandidate制約dispatchに含まれない |

FEASIBILITY_FIRSTは可行性を順位に反映する方式であり、PENALTYは概念的には次のようにacquisitionと違反量をトレードオフさせる。

$$
A_{\mathrm{penalty}}(X)=A(X)-\lambda V(X).
$$

これは個々のbackendの正確な更新則や係数を再定義した式ではない。具体的な違反量の集約や候補選択はそれぞれのソースで確認する。

### 6.1 BoTorch形式の入力制約の例

~~~python
import torch
from robotorchan.optim.constraints.contracts import CandidateConstraints

constraints = CandidateConstraints(
    inequality_constraints=(
        (
            torch.tensor([0], dtype=torch.long),
            torch.tensor([1.0], dtype=torch.double),
            0.3,
        ),
    ),
)
~~~

これは入力xの第0列についてx[0] >= 0.3を意味し、未知出力Yの可行確率ではない。非線形制約はcallableの戻り値が0以上で可行、intrapointなら入力[d]、interpointなら[q,d]とする。gradient backendでは微分可能性が必要となる。BoTorch nativeは非線形制約でfeasibleなbatch_initial_conditionsまたはic_generatorを要求し、native mixedはinter-point nonlinearを明示拒否する。

### 6.2 実装根拠と未検証事項

宣言の根拠はsrc/robotorchan/optim/capabilities.py、符号・shape契約はconstraints/contracts.py、委譲と例外条件はbackends/botorch.pyおよびbackends/torch.py。tests/optim/integration/test_nonlinear_candidate_constraints_e2e.pyは選択した条件の可行候補を検証するが、全backendと任意の制約の大域最適性を保証しない。この改訂では独立数値テストを再実行していない。

**移行判定：partial / review required。** Phase 3 routingは変更せず、旧Mapping174区間のverified判定を昇格しない。
