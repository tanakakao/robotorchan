# Variational GP：inducing pointsとELBOの実装

## 理論と利用場面

[Variational GP理論](../theory/09_variational_gp.md)は、観測数`n`が増えるとExact GPの密な行列計算が重くなる問題を扱う。`d`（入力次元）が大きい問題とは区別する。誘導点`Z`（通常`m<n`）とvariational distributionを用いてposteriorを近似し、ELBOを学習目的にする。これはExact GPの周辺尤度そのものを直接最大化する方法ではない。

## 実装と推論契約

`src/robotorchan/models/standard/variational.py`の`SingleTaskVariationalGP`はBoTorch `SingleTaskVariationalGP`を継承する。

- **Delegated:** inducing-point allocation、variational strategy、posterior、input/outcome transformsはBoTorchに委譲する。
- **Implemented（統合）:** robotorchanはraw dataを保持し、`make_mll(num_data)`がGPyTorchの`VariationalELBO`を構築する。
- **注意:** `make_mll()`の戻り値はExact GPの`ExactMarginalLogLikelihood`ではない。`num_data`はELBOで表現する**全観測数**で、入力がminibatchのときはそのミニバッチ行数と同一とは限らない。
- **Mixed variant:** `MixedSingleTaskVariationalGP`はnative categorical covarianceを利用する。カテゴリ変数があるからといって全variationalモデルの内部表現が同じとは限らない。

## 使用例：ELBOとinducing points

```python
import torch
from robotorchan.models import SingleTaskVariationalGP

X = torch.linspace(0, 1, 30, dtype=torch.double).unsqueeze(-1)
Y = torch.sin(6 * X)
model = SingleTaskVariationalGP(
    train_X=X,
    train_Y=Y,
    inducing_points=8,
)
elbo = model.make_mll(num_data=X.shape[-2])
print(type(elbo).__name__)
```

この例はモデルとELBOの構築を示すもので、パラメータ最適化を実行していない。学習時はVariational strategy・likelihood・ELBOのoptimizerステップに対応した処理が必要。モデルの`raw_train_Y`は任意入力により`None`になり得るため、必ず教師データが保持されるものとして扱わない。

## 何を検証するか

inducing pointの数、ELBOの符号とスケール、minibatchと`num_data`、posteriorの予測形状・数値、gradientの伝播を個別に検証する。尤度非Gaussianやmulti-outputへの拡張では同一計算路を仮定しない。shapeの一致やCI成功だけでは、近似posteriorの正確性は証明できない。

参考：[モデルガイド](../models/standard.md)、[Variational Notebook](../../examples/notebooks/06_variational_gp.ipynb)、[旧Mapping](../../.github/project-records/theory-guide/mapping/models.md)、[Verification](../../.github/project-records/theory-guide/mapping/verification.md)。関連：[Exact GP](02_gaussian_process.md)、[高次元GP](08_high_dimensional_gp.md)。

## Exact GPとの学習目的の違いを数式から読む

Exact GPは観測値の周辺対数尤度 log p(y|X,θ)を最大化する。一方、Variational GPはinducing variables uを導入し、近似分布q(u)に対して次のevidence lower boundを最大化する。

$$
\mathcal L_{\mathrm{ELBO}}=
\mathbb E_{q(f)}[\log p(y\mid f)]
-\mathrm{KL}(q(u)\Vert p(u)).
$$

第1項は尤度によるデータ適合、第2項は近似posteriorとpriorの乖離に対する正則化である。訓練時のminibatch使用では、全観測数に応じた尤度項のスケーリングが必要となる。ローカルのSingleTaskVariationalGP.make_mll(num_data)が生成するのはgpytorch.mlls.VariationalELBO(likelihood=self.likelihood, model=self.model, num_data=num_data)であり、ExactMarginalLogLikelihoodではない。

| 対象 | ローカルAPIと実装 | 境界 |
| --- | --- | --- |
| Gaussian regression Variational GP | models/standard/variational.pyのSingleTaskVariationalGP.make_mll() | BoTorch SingleTaskVariationalGPを継承し、inducing-point allocationとposterior計算を委譲 |
| 二値GP分類 | models/classification/binary/standard/single_task.pyのBinarySingleTaskGPClassifier | BernoulliLikelihoodを指定し、VariationalELBOのlikelihood項がGaussian regressionと異なる |
| Exact GP | models/base.pyのExactGPModelMixin.make_mll() | ExactMarginalLogLikelihood(self.likelihood,self)を返し、上記ELBOとは異なる |
| Fully Bayesian SAAS | models/high_dimensional/fully_bayesian.py | supports_mll=False。BoTorchのfit_fully_bayesian_model_nuts経路で推論し、MLL factoryを使用しない |

num_dataを省略するとraw_train_X.shape[-2]を用い、0以下はValueErrorとなる。raw_train_Yはconstructorで省略可能でNoneの場合がある。make_mll()は目的関数の生成であり、呼ぶだけでは学習処理が実行されない。non-Gaussian likelihoodの分類では、潜在関数posteriorと予測ラベル分布の確率を区別する必要がある。

**照合根拠：** src/robotorchan/models/standard/variational.py、models/classification/binary/standard/single_task.py、models/base.py、models/high_dimensional/fully_bayesian.py。ソース確認に基づく教育用の説明であり、ELBO値の独立数値検証や最適化収束をこの説明では保証しない。

## 理論章から分離した実装上の条件

### 9.56 実装 API との境界

学習器の生成、ELBO オブジェクトの取得、mini-batch の指定、事後分布取得などの
ライブラリ固有の呼び出し方は[モデルドキュメント](../models/README.md)で扱います。
本章では $q(u)$、ELBO、近似誤差、計算量および不確実性の理論を扱います。

---

#### 9.31 Variational Distribution：実装上の補足

代表的な構成では `CholeskyVariationalDistribution` のようなGaussian
variational distributionを利用します。実際のdefaultは利用中のupstream versionと
constructor contractを確認します。

#### 9.41 Variational GP と Random Forest / Neural Network：実装上の補足

- GP 的な posterior uncertainty
- kernel prior
- BoTorch acquisition function との統合

## 数値 API と利用上の補足

具体的なモデルの生成・学習方法は [モデルドキュメント](../models/README.md) を参照してください。

モデルの初期化に使う観測、実際に ELBO に寄与する観測、予測評価に使う観測は
区別する必要があります。データ来歴の保持方法はモデル API の責務です。


具体的なモデル API は [モデルドキュメント](../models/README.md) を参照してください。

## 数理上の概念と具体的な識別子の対応

以下は理論で一般的な概念名に分けた箇所の、具体的なモデル・変数・処理の対応です。数学的な導出は対応する Theory 章を参照してください。

1. Titsias, M. (2009).
   Variational learning of inducing variables in sparse Gaussian processes.
   *Proceedings of AISTATS*, 5, 567-574.
2. Hensman, J., Fusi, N., and Lawrence, N. D. (2013).
   Gaussian processes for big data. *Proceedings of UAI*.
3. Burt, D. R., Rasmussen, C. E., and van der Wilk, M. (2020).
   Convergence of sparse variational inference in Gaussian process regression.
   *Journal of Machine Learning Research*, 21.
4. BoTorch `SingleTaskVariationalGP` and GPyTorch `VariationalELBO` documentation.
5. [Gaussian Process](../theory/02_gaussian_process.md)
6. [High-dimensional GP](../theory/08_high_dimensional_gp.md)
