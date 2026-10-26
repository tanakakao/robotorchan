# Bayesian Optimization：理論とrobotorchanの実装

## この章で解決する問題

高価なブラックボックス関数を少ない評価回数で最適化するために、観測からモデルを学習し、次の評価候補を選ぶまでの責務を理解する。数理的な定義は[ベイズ最適化の理論](../../../theory/01_bayesian_optimization.md)を正本とする。

## 理論からアルゴリズムへ

観測 `(X, Y)` から代理モデルのposteriorを求め、そのposteriorに基づく獲得関数を候補空間上で最大化する。獲得関数の最大化は高価な真の目的関数の最大化とは異なる。通常は「観測 → モデル更新 → posterior → acquisition → 候補最適化 → 新しい観測」を反復する。最小化する場合は目的値の符号やobjective変換の整合性が必要になる。

## robotorchanで何が実行されるか

| 計算上の責務 | 実装と所有権 | 区分 |
| --- | --- | --- |
| 単一出力GPの確率モデル | `src/robotorchan/models/standard/single_task.py`の`SingleTaskGP`はBoTorch `SingleTaskGP`を継承 | Extended（wrapper）/ Delegated（posterior） |
| 学習の目的関数 | `src/robotorchan/models/base.py`の`ExactGPModelMixin.make_mll()`が`ExactMarginalLogLikelihood`を作成 | Implemented（組み立て）/ Delegated（MLL） |
| 学習実行 | BoTorch `fit_gpytorch_mll`を利用 | Delegated |
| 獲得関数・候補探索 | モデル学習と異なる責務。[最適化ガイド](../../../optimization/README.md)参照 | モデルだけでは未検証 |

robotorchan全体を単一の閉ループ関数が自動実行すると仮定しない。候補最適化のdispatch、制約、q-batchなどはそれぞれの公開契約を確認する。

## 最小利用例：posteriorまで

```python
import torch
from botorch.fit import fit_gpytorch_mll
from robotorchan.models import SingleTaskGP

train_X = torch.linspace(0, 1, 8, dtype=torch.double).unsqueeze(-1)
train_Y = torch.sin(2 * torch.pi * train_X)
model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
fit_gpytorch_mll(model.make_mll())
candidate_X = torch.tensor([[0.25], [0.75]], dtype=torch.double)
posterior = model.posterior(candidate_X)
print(posterior.mean, posterior.variance)
```

これは学習と予測の例であり、獲得関数や実験ループが自動で実行された例ではない。コードは現在の公開シンボルと引数に合わせているが、この章の追加PRでの実行結果は別途確認する。

## 制約、検証、参照

`train_X`は入力特徴次元を持つtensor、`train_Y`は対応する観測tensorとする。実務では入力・出力のスケール、dtype/deviceの一致、観測ノイズ、bounds、候補制約を明示する必要がある。latent posteriorとノイズ付きの新規観測分布は同義ではない。

既存[verification Mapping](../verification.md)は小規模exact GPの独立条件付き分布参照を記録するが、BO全体の収束保証やすべての獲得関数・optimizer性能の実証ではない。Notebookは[標準GP](../../../../examples/notebooks/01_single_task_gp.ipynb)を参照。次は[GP](02_gaussian_process.md)、[Kernel](03_kernel.md)へ進む。
