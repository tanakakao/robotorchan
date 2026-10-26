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
