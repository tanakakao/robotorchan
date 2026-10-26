# Gaussian Process：posterior、学習と実装の境界

## この章で解決する問題

少数の観測から、未観測入力における予測分布と不確実性を得る。数学的な定義、条件付き正規分布、ノイズモデルについては[Gaussian Processの理論](../../../theory/02_gaussian_process.md)を正本とする。

## 数理から処理へ

有限個の入力に対してGPは平均ベクトルと共分散行列を定める。学習時には観測 `train_X`, `train_Y` と、必要に応じた観測分散 `train_Yvar` を使う。学習後の `posterior(test_X)` は予測点のjoint distributionを表す。平均や分散だけでなく点間共分散も、複数候補を同時に評価する場合に重要である。

観測 `y=f(x)+ε` のノイズを条件付けに用いることと、新規観測にもノイズを加えた分布を返すことは異なる。呼び出し側はBoTorchの `observation_noise` の扱いを確認し、latent uncertaintyとobservation uncertaintyを取り違えない。

## 実装アーキテクチャ

`src/robotorchan/models/standard/single_task.py` の `SingleTaskGP` は `FittableExactGPModelMixin` とBoTorchの `SingleTaskGP` を継承する。

- **Delegated:** GPの確率モデル、posteriorの計算、基礎likelihoodはBoTorch/GPyTorchの機能を利用する。
- **Extended:** robotorchanは元の `train_X`, `train_Y`, `train_Yvar` のdetach/cloneしたスナップショットを保持し、共通のfit/MLL契約を付加する。
- **Implemented（統合部分）:** `src/robotorchan/models/base.py` の `ExactGPModelMixin.make_mll()` は `ExactMarginalLogLikelihood(self.likelihood, self)` を返し、`FittableExactGPModelMixin.fit()` がBoTorch fittingを呼び出す。

この一般的なexact-GP契約をVariational GP、Fully Bayesian GP、DeepGPにそのまま当てはめない。

## 公開APIと最小例

```python
import torch
from botorch.fit import fit_gpytorch_mll
from robotorchan.models import SingleTaskGP

X = torch.linspace(0, 1, 6, dtype=torch.double).unsqueeze(-1)
Y = (X - 0.4).square()
model = SingleTaskGP(train_X=X, train_Y=Y)
mll = model.make_mll()
fit_gpytorch_mll(mll)
test_X = torch.tensor([[0.2], [0.8]], dtype=torch.double)
p = model.posterior(test_X)
assert p.mean.shape[-2:] == (2, 1)
assert p.variance.shape[-2:] == (2, 1)
```

`train_Yvar`は省略可能であり、指定時にはその意味（観測分散）を単純な予測分散と混同しない。dtype・deviceは全入力で整合させる。tensor shape assertionはAPI形状の確認であり、数値的なposterior公式の証明ではない。

## 検証で確認すべきこと

[既存のverification Mapping](../verification.md)には固定ハイパーパラメータ下での小規模GP条件付き平均・共分散との比較が記録されている。テストが検証するのはその条件における値であり、学習最適化の収束、すべてのノイズ設定、hyperparameterの同定可能性ではない。

実装の責務確認：[single_task.py](../../../../src/robotorchan/models/standard/single_task.py)、[base.py](../../../../src/robotorchan/models/base.py)。利用者向け説明：[Standard models](../../../models/standard.md)、[Notebook](../../../../examples/notebooks/01_single_task_gp.ipynb)。関連章：[BO](01_bayesian_optimization.md)、[Kernel](03_kernel.md)。

## GP条件付き分布の数式をAPIに対応付ける

単一出力・固定hyperparameterのExact GPで、学習点X、予測点X_*、平均関数m、kernel k、観測共分散K_y=K_XX+Σ_noiseとすると、潜在関数の条件付き予測は

$$
\mu_*=m(X_*)+K_{*X}K_y^{-1}(y-m(X)),
\qquad
\Sigma_*=K_{**}-K_{*X}K_y^{-1}K_{X*}.
$$

実装は逆行列を直接形成することを要求せず、BoTorch/GPyTorchの線形代数へ委譲する。上式のmとkはmodel.mean_moduleとmodel.covar_module、観測ノイズはlikelihoodと必要に応じてtrain_Yvar、予測値はmodel.posterior(test_X)のmean/varianceに対応する。varianceはΣ_*の対角成分で、複数候補点の相関を評価する際は周辺varianceだけでは不十分である。予測入力のshapeは通常batch_shape + [q,d]、単一出力のmean/varianceはbatch_shape + [q,1]となる。

## 学習目的・ノイズ・rawデータの実装境界

| 数理対象 | robotorchan上の入口 | 実装上の所有権と注意点 |
| --- | --- | --- |
| 対数周辺尤度 log p(y\|X,θ) | model.make_mll() | ExactGPModelMixinがExactMarginalLogLikelihood(likelihood, model)を生成し、GPyTorchが計算 |
| hyperparameter最適化 | model.fit() または fit_gpytorch_mll(model.make_mll()) | FittableExactGPModelMixinがBoTorchのfittingを利用。学習収束は別途評価 |
| posteriorのjoint分布 | model.posterior(test_X) | BoTorchのSingleTaskGPから継承。robotorchan独自のGaussian conditioning式ではない |
| 観測ノイズ | train_Yvar、likelihood、posterior(observation_noise=...) | 学習時のノイズと将来観測のノイズを区別。latent posteriorとnoisy observation posteriorは同義でない |
| 元データのprovenance | raw_train_X / raw_train_Y / raw_train_Yvar | constructor引数のdetach/cloneした非永続buffer。condition_on_observationsやfantasize後の最新訓練状態ではない |

src/robotorchan/models/standard/single_task.pyでは、SingleTaskGPがFittableExactGPModelMixinとBoTorchSingleTaskGPを継承し、train_X/Y/Yvar、likelihood、covar_module、mean_module、input/outcome transformsを親classへ渡す。raw_train_*は入力時点のコピーであり、BoTorchが保持するtransform後の現在の訓練状態と混同しない。raw buffersはstate_dictに永続保存しない。

上のMLLはExact GPの式である。Variational GPはELBO、Fully Bayesianモデルはposterior hyperparameter integrationなど異なる学習契約を持ち、この共通exact MLLを自動適用してはならない。

**検証水準：** ソースの継承関係・数式の対応と、既存verification Mappingにある固定条件のGaussian conditioning参照を区別する。今回独立数値検証を実行したわけではなく、すべてのkernel・ノイズ設定・学習収束を認証しない。
