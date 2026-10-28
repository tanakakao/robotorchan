# Multi-Fidelity GP：理論と実装

## 解決する問題と数理の入口

[理論の正本](../theory/06_multi_fidelity.md)は数学的仮定・導出を担当する。本章は入力データ、アルゴリズム、公開API、上流委譲、数値検証のつながりを日本語で説明する。

安価で粗い評価と高価で精度の高い評価の相関を活用する。入力のfidelity列は通常の設計変数とは異なり、モデルが精度レベル間の共分散を学習するための構造的特徴である。観測のprecisionとfidelity、候補評価コストを混同しない。

## 実装の責務と公開契約

- **Extended:** `src/robotorchan/models/standard/multi_fidelity.py`の`SingleTaskMultiFidelityGP`はBoTorchモデルを継承し、`iteration_fidelity`、`data_fidelities`、`linear_truncated`等を渡しraw観測を保存する。
- **Delegated:** fidelity間共分散とGP posteriorの基礎計算はBoTorch。
- **Implemented/Extended:** `MixedSingleTaskMultiFidelityGP`はカテゴリ設計列用native covarianceを組み込み、fidelity列を構造列として除外する。このMixed版は`linear_truncated=True`を明示的に拒否する。

## 最小の利用例

```python
import torch
from robotorchan.models import SingleTaskMultiFidelityGP

X = torch.tensor([[0.1, 0.0], [0.1, 1.0], [0.8, 0.0], [0.8, 1.0]], dtype=torch.double)
Y = torch.tensor([[0.2], [0.25], [0.6], [0.7]], dtype=torch.double)
model = SingleTaskMultiFidelityGP(train_X=X, train_Y=Y, data_fidelities=[1])
print(model.posterior(X[:2]).mean)
```

コードは確認した公開APIを示す最小例であり、すべての条件での成功を保証するものではない。tensorのshape、dtype/device、入力変数の意味と、posteriorの出力構造を意識して利用する。

## 検証・未検証事項

既存の数値参照にはdata-fidelity kernelの境界値とrank-one残差の検証があるが、全fidelity posteriorや学習ハイパーパラメータの最適性まで検証したことにはならない。異なるfidelity値を同じ尺度で単純に比較できるとは限らない。Notebook: [Multi-Fidelity](../../examples/notebooks/03_multi_fidelity_gp.ipynb)。

[旧Model Mapping](../../.github/project-records/theory-guide/mapping/models.md)および[検証記録](../../.github/project-records/theory-guide/mapping/verification.md)は移行元の正本として保持する。API存在、静的assertion、CI成功、独立数値検証は区別する。使用例と詳しいモデル選択は[モデルガイド](../models/README.md)を参照する。

## Fidelity列とカーネルの所有権を具体化する

SingleTaskMultiFidelityGPはBoTorchSingleTaskMultiFidelityGPを継承し、iteration_fidelity・data_fidelities・linear_truncated・nu等を転送する。fidelityは観測の標準偏差やtrain_Yvarと異なり、評価レベルの違いを表す入力座標である。低fidelityが常に高fidelityの単調下界になるといった仮定はこのconstructorだけからは導けない。

| family | 設計共分散 | fidelityの取扱い・拒否条件 |
| --- | --- | --- |
| SingleTaskMultiFidelityGP | BoTorch標準の構成に委譲 | fidelity列は構造的な入力。linear_truncated指定はBoTorchへ転送 |
| MixedSingleTaskMultiFidelityGP | make_mixed_covar_moduleによりカテゴリ＋連続の設計部分を構成 | iteration_fidelityとdata_fidelitiesをカテゴリ列から除外し、両者の重複を拒否。linear_truncated=TrueはValueError |
| MapSaasMultiFidelityGP | 設計列のMatern kernelにSAAS prior | fidelity列をdesign_dimsから除外。linear_truncated=Trueは拒否 |
| AdditiveMapSaasMultiFidelityGP | BoTorchのadditive MAP-SAAS kernel | 少なくとも1つのfidelity列と設計列が必要。linear_truncated=Trueは拒否 |

構造列をカテゴリ列として同時指定すると、kernelで同じ入力を二重に解釈するおそれがある。Mixed版はnormalize_feature_dimsによりカテゴリ列・fidelity列の重複を明示的に検査する。学習に使う全fidelityレベルの観測数と、獲得関数のコストモデル・ターゲットfidelityを同一概念として扱わない。

posteriorのmean/varianceは指定したfidelity座標での予測であり、高精度評価を提案する際はその対象レベルとcost-aware acquisitionを別途定義する。データ形状やdtypeのAPI確認と、fidelity間相関が実データで妥当かの数値評価は別の検証となる。

**実装根拠：** src/robotorchan/models/standard/multi_fidelity.py。今回の根拠はconstructor・dimension validationの静的確認であり、全fidelity kernelの数値同値性・最適化性能を再実行していない。

## 理論章から分離した実装上の条件

#### 6.9 Cost-Aware Bayesian Optimization：実装上の補足

BoTorchではcost-aware utilityをacquisition設計へ組み込むための部品が用意されています。

## 数理上の概念と具体的な識別子の対応

以下は理論で一般的な概念名に分けた箇所の、具体的なモデル・変数・処理の対応です。数学的な導出は対応する Theory 章を参照してください。

1. Kennedy, M. C., and O'Hagan, A. (2000).
   Predicting the output from a complex computer code when fast approximations are available.
   *Biometrika*, 87(1), 1-13.
2. Kandasamy, K., Dasarathy, G., Oliva, J. B., Schneider, J., and Póczos, B. (2017).
   Multi-fidelity Bayesian optimisation with continuous approximations.
   *Proceedings of the 34th International Conference on Machine Learning*.
3. BoTorch `SingleTaskMultiFidelityGP`, multi-fidelity, and cost-aware documentation.
4. [Acquisition Function](../theory/04_acquisition_function.md)
5. [Mixed Variables](../theory/05_mixed_variables.md)
