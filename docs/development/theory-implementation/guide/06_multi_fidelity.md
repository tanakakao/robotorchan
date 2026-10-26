# Multi-Fidelity GP：理論と実装

## 解決する問題と数理の入口

[理論の正本](../../../theory/06_multi_fidelity.md)は数学的仮定・導出を担当する。本章は入力データ、アルゴリズム、公開API、上流委譲、数値検証のつながりを日本語で説明する。

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

コードは確認した公開APIを示す最小例であり、今回のセッションで実行して成功したとの主張ではない。tensorのshape、dtype/device、入力変数の意味と、posteriorの出力構造を意識して利用する。

## 検証・未検証事項

既存の数値参照にはdata-fidelity kernelの境界値とrank-one残差の検証があるが、全fidelity posteriorや学習ハイパーパラメータの最適性まで検証したことにはならない。異なるfidelity値を同じ尺度で単純に比較できるとは限らない。Notebook: [Multi-Fidelity](../../../../examples/notebooks/03_multi_fidelity_gp.ipynb)。

[旧Model Mapping](../models.md)および[検証記録](../verification.md)は移行元の正本として保持する。API存在、静的assertion、CI成功、独立数値検証は区別する。使用例と詳しいモデル選択は[モデルガイド](../../../models/README.md)を参照する。
