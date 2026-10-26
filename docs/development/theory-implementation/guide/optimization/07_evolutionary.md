# Evolutionary Optimization：スカラー獲得関数と多目的探索を区別する

## 1. この章の対象と数理的境界

[進化的最適化の理論](../../../../theory/optimization/07_evolutionary.md)を正本とする。DE、CMA-ES、GA、PSOは候補群を使うが更新則・制約の処理・混合変数への適合性は異なる。NSGA-IIは**複数の実数値目的を同時に最大化するPareto探索**を行う。BoTorchのqEHVI・qNEHVIが出力するscalar acquisitionを最適化することと、NSGA-IIでvector-valued objectiveそのものを探索することは別である。

## 2. Dispatcherのoptimizer名と特殊backend

`src/robotorchan/optim/dispatch.py::optimize_acqf`に登録される11個のnamed backendは`botorch`、`torch_adam`、`torch_adamw`、`torch_sgd`、`random`、`sobol`、`de`、`cmaes`、`ga`、`pso`、`hybrid`である。これらはacquisitionのscalar値を探索する。`nsga2`はこの`optimizer=`の選択肢ではなく、別APIの`optimize_vector_nsga2`で利用する。

## 3. NSGA-IIのローカル実装

`src/robotorchan/optim/backends/nsga2.py::optimize_vector_nsga2`は、`objective(X)`に`[population,d]`を渡し、少なくとも2つの目的からなる`[population,m]`を受け取る。**全目的を最大化**すると定義されている。最小化目的はユーザー側の向きの変換が必要となる。

初期集団をbounds内で生成し、各世代で(1) nondominated ranking、(2) crowding distance、(3) binary tournament、(4) crossover、(5) mutation、(6) environmental selectionを行う。最終集団からrank=0の候補`[n_pareto,d]`と、対応する目的値`[n_pareto,m]`を返す。

| 実装契約 | ソースで確認できる意味 |
| --- | --- |
| `population_size`、`generations` | 集団数と世代数で計算費用を指定 |
| `crossover_rate`、`mutation_rate`、`mutation_scale` | 局所実装の遺伝演算を制御 |
| `seed` | ローカル乱数生成器を利用し再現性を確保 |
| `bounds` | 候補の下限・上限。結果の可行性は制約機構を別途確認 |
| `objective`の出力shape | `[population,m]`かつ`m>=2`。scalar目的は拒否 |
| 戻り値 | 非劣解の近似集合とその実目的値。BoTorch acquisition scalarの戻り値ではない |

上記の独自ranking・crowding・変異等はrobotorchan側で**Implemented**。BoTorchにNSGA-IIアルゴリズムを委譲していると説明してはいけない。

## 4. テストが証明すること・しないこと

`tests/optim/backends/test_nsga2_backend.py`には、二目的のsynthetic trade-offで(1) candidate bounds、(2)返却集団内部でのnondominance、(3)seed再現性、(4)scalar目的の拒否、(5)dtype/device保持を検査するassertionがある。

これらは返却された集合内部での性質であり、**真のPareto frontへの収束、hypervolume改善率、制約付き探索の性能**を一般に保証しない。各evolutionary backendのcandidate制約はnative enforcement、feasibility-first、penalty等で異なり、capability metadataだけで結果の厳密可行性を認定しない。

## 5. 関連する実装ガイド

[制約付き候補探索](05_constraints.md)、[多目的獲得関数](../acquisition/07_multiobjective.md)、[Optimization検証](../verification/optimization.md)、[旧Optimization Mapping](../../optimization.md)を参照する。旧Mappingの全区間検収が完了するまでは元文書を保持する。

## 6. NSGA-IIのPareto判定とscalar acquisitionの区別

旧Optimization Mappingの固定区間246–257と src/robotorchan/optim/backends/nsga2.py を照合する。最大化するm目的の値ベクトル f(x) に対し、xがyを支配する（優越する）条件は

$$
f_i(x)\ge f_i(y)\quad(\forall i),\qquad
f_j(x)>f_j(y)\quad(\exists j).
$$

NSGA-IIはnondominated sortingでrankを求め、同じrankの多様性をcrowding distanceで評価する。binary tournament、算術交叉、変異、親子集団からのelitist environmental selectionを繰り返す。これはrobotorchanが実装するベクトル目的の近似Pareto探索であり、BoTorchへ委譲するqEHVI（hypervolume improvementを**スカラー**で返すacquisition）の最適化と同じものではない。

利用例（値と候補のshapeを明示）：

~~~python
import torch
from robotorchan.optim.backends.nsga2 import optimize_vector_nsga2


def objectives(x: torch.Tensor) -> torch.Tensor:
    return torch.cat((-((x - 0.2) ** 2), -((x - 0.8) ** 2)), dim=-1)


bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
pareto_x, pareto_y = optimize_vector_nsga2(
    objectives, bounds, population_size=16, generations=5, seed=7
)
# pareto_xは[n_pareto, 1]、pareto_yは[n_pareto, 2]。
~~~

このAPIのobjectiveは入力[population,d]に対して[population,m]かつm>=2を返す必要がある。いずれの目的も最大化向きなので最小化したい量には負符号など方向変換が必要。ローカルNSGA-II APIは旧Mappingで確認した範囲にcandidate-constraint引数を持たず、constraint-dominationやfeasible Pareto frontを自動的に保証しない。実装・テストで確認できるのは返却集団内の非支配性など限定された性質であり、真のPareto frontへの収束証明ではない。
