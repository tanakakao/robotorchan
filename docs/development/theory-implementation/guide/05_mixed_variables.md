# Mixed Variables：理論と実装

## 解決する問題と数理の入口

[理論の正本](../../../theory/05_mixed_variables.md)は数学的仮定・導出を担当する。本章は入力データ、アルゴリズム、公開API、上流委譲、数値検証のつながりを日本語で説明する。

連続変数とカテゴリ変数を同時に扱う。整数のカテゴリラベルを連続量の距離と誤認すると、カテゴリの番号付けで共分散が変わってしまう。native categorical kernelは等値性に基づく共分散で、カテゴリを単純な数値距離とみなさない。学習前後でカテゴリ符号を一貫して保つ。

## 実装の責務と公開契約

- **Extended:** `src/robotorchan/models/standard/single_task.py`の`MixedSingleTaskGP`はBoTorch `MixedSingleTaskGP`のラッパーで、raw tensor保持と共通学習契約を追加する。
- **Delegated:** GP推論、native mixed covarianceとposteriorの主要処理はBoTorch/GPyTorch。
- **Implemented:** `src/robotorchan/models/base.py`の`normalize_feature_dims`と`make_mixed_covar_module`は他のmixed構造モデル向けの共通補助機能。全モデルが同じmixed kernelを用いるわけではない。
- `cat_dims`はraw入力列に対応し、task/fidelity/context等の構造列と同一視しない。

## 最小の利用例

```python
import torch
from robotorchan.models import MixedSingleTaskGP
X = torch.tensor([[0.1, 0.0], [0.3, 1.0], [0.8, 0.0], [0.9, 1.0]], dtype=torch.double)
Y = torch.tensor([[0.2], [0.8], [0.6], [0.4]], dtype=torch.double)
model = MixedSingleTaskGP(train_X=X, train_Y=Y, cat_dims=[1])
print(model.posterior(X[:2]).mean)
```

コードは確認した公開APIを示す最小例であり、今回のセッションで実行して成功したとの主張ではない。tensorのshape、dtype/device、入力変数の意味と、posteriorの出力構造を意識して利用する。

## 検証・未検証事項

カテゴリ名の置換に対する共分散の不変性は、同じ条件でのカテゴリIDの付け替えを比較するpropertyであり、全mixed variantの検証完了を意味しない。BoTorch-native categorical kernel、内部one-hot、embedding、passthroughをモデルごとに区別する。関連Notebook: [Mixed GP](../../../../examples/notebooks/02_mixed_single_task_gp.ipynb)。

[旧Model Mapping](../models.md)および[検証記録](../verification.md)は移行元の正本として保持する。API存在、静的assertion、CI成功、独立数値検証は区別する。使用例と詳しいモデル選択は[モデルガイド](../../../models/README.md)を参照する。
