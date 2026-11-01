# Uncertain-input GP：理論と実装

## 背景と理論の入口

観測した説明変数の位置が不確かなtraining-input uncertaintyと、提案した候補を実行するときのdecision-time perturbationを区別する。Gaussian入力不確実性をRBF kernelに積分すると、入力分布の平均と共分散に依存するexpected kernelを得る。これは入力へ単発のjitterを付ける近似とは異なる。

数式と確率的仮定の正本は[対応する理論章](../theory/16_uncertain_input_gp.md)を参照。本章ではrobotorchanの実装責務・推論・制約に重点を置く。

## 実装アーキテクチャと責務

- **Implemented:** `src/robotorchan/models/uncertain/uncertain_input.py`の`GaussianUncertainInputKernel`は、平均とflattenした入力共分散を含む表現から解析的期待RBF共分散を計算する。Choleskyによる線形系とlog determinantを利用する。
- **Implemented:** `_validate_input_uncertainty`は`train_X_std`と`train_X_covar`をちょうど一方指定する契約を検査し、共分散の対称性と半正定値性をチェックする。
- **Extended:** Mixed版の期待kernelでは連続部の不確実性とカテゴリ等値kernelを区別する。カテゴリIDを正規分布の連続座標と扱わない。

## APIと実装を読むための最小例

```python
import torch
from robotorchan.models import UncertainInputSingleTaskGP

train_X = torch.linspace(0, 1, 6, dtype=torch.double).unsqueeze(-1)
train_Y = torch.sin(5 * train_X)
train_X_std = torch.full_like(train_X, 0.02)
model = UncertainInputSingleTaskGP(
    train_X=train_X,
    train_Y=train_Y,
    train_X_std=train_X_std,
)
candidate_X = torch.tensor([[0.25], [0.75]], dtype=torch.double)
posterior = model.posterior(candidate_X)
assert posterior.mean.shape[-2:] == (2, 1)
```

コード例は公開APIを示す静的な例であり、このセッションで実行して成功したとの主張ではない。

## 適用条件、テスト、限界

この解析結果はGaussian入力不確実性×RBF kernelの仮定に限定される。零入力分散で通常のRBFへ戻る極限、共分散対称性、PSD、dtype/device、zero/noisy casesを別々に検証する。candidate-time robust decisionの保証を自動的に付与しない。

旧[モデルMapping](../../.github/project-records/theory-guide/mapping/models.md)・[数学的検証Mapping](../../.github/project-records/theory-guide/mapping/verification.md)を残し、学習ガイド本文に過去のPR/CI履歴を混入させない。[モデル選択ガイド](../models/robust_noise.md)には利用者向けの比較を記す。

## Gaussian入力不確実性の積分と公開posterior座標

訓練入力の真の位置をz_i~N(mu_i,Sigma_i)とし、RBF共分散を仮定する。2つの独立なGaussian位置分布の共分散を平均化したexpected kernelは、lengthscale平方の対角行列LambdaとS=Sigma_i+Sigma_jを使い、

$$
\bar k(i,j)=|\Lambda|^{1/2}|\Lambda+S|^{-1/2}
\exp[-\tfrac12(\mu_i-\mu_j)^T(\Lambda+S)^{-1}(\mu_i-\mu_j)]
$$

となる（単位outputscaleのRBFの場合）。GaussianUncertainInputKernelは入力を[mean(d), flattened covariance(d*d)]という内部表現に拡張し、Cholesky solveとlogdetによって上式を評価する。ゼロ分散の極限で通常のRBFに戻る。

UncertainInputSingleTaskGPの公開constructorはtrain_X_stdとtrain_X_covarをちょうど一方だけ受け取る。stdはtrain_Xと同じshapeで非負・有限、full covarianceは[...,d,d]で有限・対称・PSDでなければならない。posterior(X)では元のd次元deterministic候補を受け取り、zero covarianceを連結して内部posteriorへ渡す。**不確かな訓練位置**と**不確かな実行候補**を同時に周辺化するAPIと解釈しない。

MixedUncertainInputSingleTaskGPはカテゴリ列の不確実性を正規分布として近似しない。連続列だけにGaussian uncertaintyを与え、native CategoricalKernelを積として組み合わせる。少なくとも1つの連続設計列が必要で、train_X_std/covarのshapeは連続列数に合わせる。

根拠：src/robotorchan/models/uncertain/uncertain_input.py。解析式とpublic→augmented座標変換のソース照合に基づき、zero-variance limitの独立数値計算やposterior精度の再検証は今回行っていない。

## 理論章から分離した実装上の条件

### 12. 実装との対応

対応する公開実装の契約は次のとおりです。

- continuous training-input uncertaintyはcaller-supplied Gaussianとして扱う
- uncertainty metadata自体をtrain_X / train_Yから自動推定するモデルではない
- RBF covariance を入力分布について解析的に周辺化する
- diagonal uncertainty は `train_X_std`
- correlated uncertainty は `train_X_covar`
- covariance は finite / symmetric / positive semidefinite を要求
- training representation は `[mean, flattened covariance]`
- `posterior(X)` の candidate は deterministic
- Mixed 版では continuous uncertainty と categorical identity を分離
- categorical uncertainty は専用モデルで扱う
- candidate perturbation / environmental scenarioは別layer
- robust candidate evaluationでは `InputPerturbation` とrisk objectiveを独立に合成できる
- current public uncertain-input familyからMultiTask等の未実装cross-productを推測しない

実装仕様は [uncertain-input GP](../models/uncertain_input_gp.md)、
[full-covariance uncertain-input GP](../models/full_covariance_uncertain_input_gp.md)、
[mixed uncertain-input GP](../models/mixed_uncertain_input_gp.md)、
[uncertain categorical GP](../models/uncertain_categorical_gp.md) を参照してください。

### 連続入力の不確かさと内部表現

`UncertainInputSingleTaskGP`は、次のどちらか一方を必ず受け取る。

| 引数 | shapeと意味 |
| --- | --- |
| `train_X_std` | `train_X.shape`。対角共分散`diag(std²)`の標準偏差 |
| `train_X_covar` | `train_X.shape[:-1] + (d,d)`。相関を含む共分散 |

Mixed版の不確かさのshapeは、全特徴数ではなく連続列数に合わせる。
共分散はfinite・symmetric・positive semidefiniteである必要がある。
`GaussianUncertainInputKernel`はGaussian入力分布に対するRBF covarianceの期待値を解析的に評価する。
一度random jitterした入力を通常GPへ渡す処理とは異なる。

```text
caller: train_X + train_X_std または train_X_covar
  → model内部で [mean, flattened covariance] に拡張
  → 入力分布間のexpected kernel
  → posterior
```

`raw_train_X`と公開候補`X`は元の特徴空間を維持する。
augmentationはprivate representationであり、入力誤差を`train_X` / `train_Y`から
自動推定する機能ではない。`posterior(X)`は候補の共分散を0とするdeterministic queryを扱う。

### 確定カテゴリと不確かなカテゴリ

`MixedUncertainInputSingleTaskGP`は、連続入力にGaussian uncertaintyを与え、
カテゴリは確定値としてnative categorical covarianceに保持する。
一方、`UncertainCategoricalSingleTaskGP`は1つのカテゴリ特徴について
category probability vectorを受け取る。
期待categorical covarianceは、学習されるPSDなcategory covariance matrixを
確率ベクトルで両側から平均化する。
単なるカテゴリ番号へのGaussian noiseでも、複数の不確かなカテゴリ列を
自動処理する一般的なAPIでもない。
詳細は[カテゴリ不確かさのガイド](../models/uncertain_categorical_gp.md)を参照する。

### 候補実行時の摂動との違い

訓練入力の測定誤差と、選んだ候補を実行するときの摂動は別の層である。
後者ではBoTorchの`InputPerturbation`でnominal candidateをscenarioへ展開し、
ExpectationやCVaRなどのrisk objectiveで集約する構成を検討する。
expected RBF kernelの解析式はGaussian uncertaintyとRBFの仮定に依存し、
任意の摂動分布・kernelや、全MultiTask派生への対応を意味しない。
