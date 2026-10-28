# CMA-ES backend

任意依存の微分不要な獲得関数最適化器です。BoTorch `AcquisitionFunction`、Tensor の bounds、joint batch サイズ `q` を受け取り、進化計算に `cmaes` を使います。

`q x d` 全体を 1 ベクトルに平坦化し、評価前に復元します。初期平均は box の中心、`sigma` は最大の box 幅に対する相対値です。

`pip install "robotorchan[cmaes]"` で導入します。追加パッケージは利用時のみ import されます。

## 候補制約

`constraints` に共通の `CandidateConstraints` を渡し、次の罰則付き目的を最小化します。

```text
penalized_objective = -acquisition_value + constraint_penalty * violation
```


`constraint_penalty` は正で既定値 `1e6`、`equality_tolerance` は既定値 `1e-6` です。点間制約を含む joint q-batch 全体に適用します。有限の罰則は実行可能性を保証しないため、必要なら返り値を元の制約で確認してください。

進化計算の制御は CPU 側、獲得関数・制約の評価は指定 Tensor の device・dtype で行います。返される `value` は**罰則なしの獲得値**です。

## 使用例

```python
from robotorchan.optim.backends import optimize_acqf_cmaes

candidates, value = optimize_acqf_cmaes(
    acq_function=acq,
    bounds=bounds,
    q=3,
    sigma=0.25,
    max_generations=200,
    seed=123,
    constraints=candidate_constraints,
    constraint_penalty=1e6,
    equality_tolerance=1e-6,
)
```


`acq`、`bounds`、`candidate_constraints` は呼び出し側で構築済みとします。backend 間の違いは[候補制約](candidate-constraints.md)を参照してください。
