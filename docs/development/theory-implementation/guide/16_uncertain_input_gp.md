# Uncertain-input GP：理論と実装

## 背景と理論の入口

観測した説明変数の位置が不確かなtraining-input uncertaintyと、提案した候補を実行するときのdecision-time perturbationを区別する。Gaussian入力不確実性をRBF kernelに積分すると、入力分布の平均と共分散に依存するexpected kernelを得る。これは入力へ単発のjitterを付ける近似とは異なる。

数式と確率的仮定の正本は[対応する理論章](../../../theory/16_uncertain_input_gp.md)を参照。本章ではrobotorchanの実装責務・推論・制約に重点を置く。

## 実装アーキテクチャと責務

- **Implemented:** `src/robotorchan/models/uncertain/uncertain_input.py`の`GaussianUncertainInputKernel`は、平均とflattenした入力共分散を含む表現から解析的期待RBF共分散を計算する。Choleskyによる線形系とlog determinantを利用する。
- **Implemented:** `_validate_input_uncertainty`は`train_X_std`と`train_X_covar`をちょうど一方指定する契約を検査し、共分散の対称性と半正定値性をチェックする。
- **Extended:** Mixed版の期待kernelでは連続部の不確実性とカテゴリ等値kernelを区別する。カテゴリIDを正規分布の連続座標と扱わない。

## APIと実装を読むための最小例

```python
from robotorchan.models import UncertainInputSingleTaskGP

print(UncertainInputSingleTaskGP.__name__)
```

コード例は公開APIを示す静的な例であり、このセッションで実行して成功したとの主張ではない。

## 適用条件、テスト、限界

この解析結果はGaussian入力不確実性×RBF kernelの仮定に限定される。零入力分散で通常のRBFへ戻る極限、共分散対称性、PSD、dtype/device、zero/noisy casesを別々に検証する。candidate-time robust decisionの保証を自動的に付与しない。

旧[モデルMapping](../models.md)・[数学的検証Mapping](../verification.md)を残し、学習ガイド本文に過去のPR/CI履歴を混入させない。[モデル選択ガイド](../../../models/robust_noise.md)には利用者向けの比較を記す。

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
