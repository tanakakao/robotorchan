# 非線形候補制約

入力候補に対する非線形制約の利用ガイドです。[理論](../theory/optimization/05_constraints.md)に数学的な背景を示します。

## 候補制約と出力制約

| 層 | 空間 | 実行可能の符号 | 所有者 |
| --- | --- | --- | --- |
| 候補制約 | 提案入力 X | `g(X) >= 0` | 獲得関数 optimizer |
| 出力制約 | モデル出力・sample Y | `c(Y) <= 0` | 制約付き獲得関数 |

`CandidateConstraints` は前者のみを表します。未知の実行可能性をモデル化せず、制約付き獲得関数や多目的 reference point を置き換えません。

## BoTorch との対応

| BoTorch の概念 | 公開 API |
| --- | --- |
| 線形不等式 tuple | `CandidateConstraints.inequality_constraints` |
| 線形等式 tuple | `CandidateConstraints.equality_constraints` |
| 非線形 `(callable, is_intrapoint)` | `CandidateConstraints.nonlinear_inequality_constraints` |
| 実行可能値 | `constraint(X) >= 0` |
| 明示的 restart 点 | `batch_initial_conditions` |
| 初期値 generator | `ic_generator` と `ic_gen_kwargs` |

標準 tuple の意味を保持するコンテナであり、別の非線形 DSL は設けません。`equality_constraints` は線形等式のみ、非線形は不等式のみです。非線形等式の標準対応があるかのように記述しないでください。必要なら許容誤差と実行可能性の意味が明確な backend 固有の再定式化を使うか、将来の capability として扱います。

## 点内・点間 callable

点内制約には候補 1 点 `[d]` を渡します。

```python
def inside_circle(x):
    return x.new_tensor(1.0) - x.square().sum()


constraints = CandidateConstraints(
    nonlinear_inequality_constraints=((inside_circle, True),),
)
```


点間制約には joint q-batch `[q, d]` を渡します。

```python
def separated(X):
    return (X[0] - X[1]).square().sum() - X.new_tensor(0.01)


constraints = CandidateConstraints(
    nonlinear_inequality_constraints=((separated, False),),
)
```


BoTorch と共通の微分不要経路で同じ形状を使います。勾配法では微分可能性も必要です。

## 初期化

BoTorch の非線形最適化には実行可能な初期点が必要です。`[num_restarts, q, d]` の `batch_initial_conditions` または適切な `ic_generator` を指定してください。

明示的初期値の形状、有限性、bounds、dtype、device、非線形実行可能性を事前検証します。失敗時に制約を落としたり無制約探索へ切り替えたりしません。

## 合成

backend が対応する場合、線形不等式・等式、非線形不等式、box、fixed features を併用できます。出力制約も獲得関数側で独立に併用できます。

TuRBO の許容領域は大域 bounds・現在の trust region・候補制約の共通部分です。入力摂動では nominal 候補だけに適用し、全 scenario の実行可能性を保証しません。

## 対応の境界

| 組合せ | 現在の対応 |
| --- | --- |
| q=1 の点内非線形 | 対応 |
| joint q>1 の点内非線形 | 対応 |
| joint q>1 の点間非線形 | 互換 joint backend で対応 |
| greedy sequential と点内 | 互換経路で対応 |
| greedy sequential と点間 | 未対応 |
| BoTorch mixed と点内 | 対応 |
| BoTorch mixed と点間 | 未対応 |
| fixed features | 互換 backend で対応 |
| X 内の MultiTask task 列 | 対応 |
| Kronecker の出力 task | 人工的な入力 task 座標は追加しない |
| X 内の MultiFidelity 列 | 対応 |
| qLogEHVI・qLogNEHVI | 候補制約と併用可能 |
| robust 入力摂動 | nominal X のみ |
| TuRBO | 局所 bounds と候補制約の共通部分 |
| latent・embedded 探索 | 厳密な制約写像がなければ拒否 |

metadata は backend 単位の対応を表し、逐次探索と点間制約などの条件付き組合せは runtime 検証が最終判断です。

## dtype・device・勾配

PyTorch のまま計算し、入力の dtype・device を保持してください。勾配法では `.detach()`、NumPy、Python `float`、無条件 `.cpu()` を挟まないでください。定数は `x.new_tensor(...)` で作れます。

callable が CUDA Tensor を評価できても、BoTorch・SciPy 制約付き optimizer 全体の GPU 対応を意味しません。

## エラーの意味

不正な制約構造、未対応の組合せ、不正な初期値、下流 optimizer の失敗を区別し、暗黙の無制約 fallback は行いません。任意の非線形実行可能集合が非空とは事前に証明しないため、custom initializer は問題に応じた実行可能点を構成できる必要があります。
