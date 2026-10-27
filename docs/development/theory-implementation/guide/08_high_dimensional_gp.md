# High-dimensional GP：疎な有効次元と実装

## この章で理解すること

入力次元が大きく観測数が小さいとき、どのモデル仮定でGPの学習可能性を改善するかを見極める。**入力次元の問題と学習点数の問題は異なる**。理論・数式は[高次元GP](../../../theory/08_high_dimensional_gp.md)を正本とし、ここではモデルと推論方式の対応を説明する。

## 理論から実装へ

通常のARDは次元別lengthscaleを持つが、高次元・少標本では推定が不安定になりやすい。SAASは入力の大部分が弱く効くという軸方向のsparsity仮定をpriorに入れる。元の入力座標を維持するため、PCA/AEのように射影座標を生成する次元削減とは異なる。加法モデルは関数を低次元成分の和に制約する別の仮定であり、SAASと同義ではない。

## 所有権と実装経路

| モデル / 推論 | 主要実装 | 区分・注意点 |
| --- | --- | --- |
| `SaasFullyBayesianSingleTaskGP` | `src/robotorchan/models/high_dimensional/fully_bayesian.py` | **Extended** raw観測保持、**Delegated** BoTorch fully Bayesian SAAS・NUTS |
| `SaasFullyBayesianMultiTaskGP` | 同上 | **Delegated** task-feature構造・Pyro MCMC・posterior |
| `MixedSaasFullyBayesianSingleTaskGP` | 同上 | モデル所有のone-hot入力変換を使用。通常のnative categorical kernel GPと同一視しない |
| `AdditiveMapSaasSingleTaskGP` / `EnsembleMapSaasSingleTaskGP` | `src/robotorchan/models/high_dimensional/map_saas.py` | **Extended** データ契約、**Delegated** BoTorch MAP-SAAS family。MCMC posteriorとは異なる |

Fully Bayesian SAAS系は`supports_mll=False`であり、標準exact GPと同様の`model.make_mll()`→`fit_gpytorch_mll`は使わない。学習はBoTorchの`fit_fully_bayesian_model_nuts`で行う。MAP-SAASも推論方法を名称だけで決めつけず、各モデルの学習契約を確認する。

## API利用例：MCMCを実行しない基本契約の確認

Fully Bayesian SAASでは、モデル生成だけならNUTSを実行せずに入力shape、
元の観測データの保持、標準MLL非対応を確認できる。通常の軽量CIではこの範囲を検証する。

```python
import torch
from robotorchan.models import SaasFullyBayesianSingleTaskGP

X = torch.linspace(0.05, 0.95, 24, dtype=torch.double).reshape(8, 3)
Y = (2.0 * X[:, :1] - X[:, 1:2]).clone()
model = SaasFullyBayesianSingleTaskGP(train_X=X, train_Y=Y)
assert model.supports_mll is False
assert torch.equal(model.raw_train_X, X)
assert torch.equal(model.raw_train_Y, Y)
```

この段階のモデルに学習済みMCMCサンプルは存在しないため、
posteriorが使えると見なさない。次の例で推論を実行するにはPyro依存、
MCMC設定、収束診断を別途確認する。

## API利用例：モデル生成と学習境界

```python
import torch
from botorch.fit import fit_fully_bayesian_model_nuts
from robotorchan.models import SaasFullyBayesianSingleTaskGP

X = torch.rand(12, 5, dtype=torch.double)
Y = (X[:, :1] - 0.3).square()
model = SaasFullyBayesianSingleTaskGP(train_X=X, train_Y=Y)
# 実行にはMCMCの計算時間・依存関係・seed管理が必要。
fit_fully_bayesian_model_nuts(model, warmup_steps=32, num_samples=16, thinning=1)
posterior = model.posterior(X[:2])
print(posterior.mean.shape)
```

短いMCMC設定は**API説明用**であり、推定の収束・精度を保証しない。このセッションでの実行結果ではない。

## 制限、検証、参照

事後サンプル数、収束指標、seed、dtype、device、Pyro依存、観測数に留意する。次元選択の精度や大域探索性能は、モデル生成だけでは検証できない。高次元での候補探索（TuRBO、ALEBO、BAxUS等）は[高次元探索](../../../optimization/high_dimensional_search.md)と[理論第20章](../../../theory/20_high_dimensional_search.md)で扱い、モデル側の仮定と分離する。

旧[モデルMapping](../models.md)・[Verification](../verification.md)は記録の正本として維持する。使用例は[SAAS Notebook](../../../../examples/notebooks/08_saas_gp.ipynb)、比較は[モデルガイド](../../../models/high_dimensional.md)。関連：[高度な高次元モデル](21_advanced_high_dimensional_models.md)、[Variational GP](09_variational_gp.md)。
