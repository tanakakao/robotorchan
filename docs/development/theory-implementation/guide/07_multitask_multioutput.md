# MultiTask / MultiOutput GP：理論と実装

## 解決する問題と数理の入口

[理論の正本](../../../theory/07_multitask_multioutput.md)は数学的仮定・導出を担当する。本章は入力データ、アルゴリズム、公開API、上流委譲、数値検証のつながりを日本語で説明する。

複数タスクの関係を使って情報を共有する。long-format multi-taskでは`train_X`の一列がtask IDを表し、`train_Y`には観測行ごとに一つの出力がある。これと同じ入力点で複数出力を観測するblock-design Kroneckerモデルは、データ構造やposteriorのshapeが異なる。

## 実装の責務と公開契約

- **Extended:** `src/robotorchan/models/standard/multitask.py`の`MultiTaskGP`はBoTorch `MultiTaskGP`からlong-format契約とtask covarianceを継承し、raw-data/MLL便宜機能を付加する。
- **Delegated:** task-conditioned GPのposterior計算はBoTorch/GPyTorch。
- **Extended:** 同モジュールの`MixedMultiTaskGP`は`task_feature`を構造列に維持し、`cat_dims`を設計側のカテゴリ列として扱う。
- `KroneckerMultiTaskGP`は別のblock-design実装であり、long-formatへの単純な別名ではない。

## 最小の利用例

```python
import torch
from robotorchan.models import MultiTaskGP

X = torch.tensor([[0.1, 0.0], [0.4, 0.0], [0.1, 1.0], [0.4, 1.0]], dtype=torch.double)
Y = torch.tensor([[0.2], [0.5], [0.3], [0.7]], dtype=torch.double)
model = MultiTaskGP(train_X=X, train_Y=Y, task_feature=1)
print(model.posterior(X[:2]).mean)
```

コードは確認した公開APIを示す最小例であり、今回のセッションで実行して成功したとの主張ではない。tensorのshape、dtype/device、入力変数の意味と、posteriorの出力構造を意識して利用する。

## 検証・未検証事項

既存のKronecker数値参照は固定したinput/task covarianceブロックの配列順とPSD等を対象とし、学習後posteriorの全面的な同値性を保証しない。task-indexing、データの配置、出力形状、未知taskへの一般化の制約を個別に確認する。Notebook: [Multitask](../../../../examples/notebooks/04_multitask_gp.ipynb)。

[旧Model Mapping](../models.md)および[検証記録](../verification.md)は移行元の正本として保持する。API存在、静的assertion、CI成功、独立数値検証は区別する。使用例と詳しいモデル選択は[モデルガイド](../../../models/README.md)を参照する。
