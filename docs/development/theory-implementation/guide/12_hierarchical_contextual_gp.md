# Hierarchical / Contextual GP：理論と実装

## 解決する問題と数理の入口

[理論の正本](../../../theory/12_hierarchical_contextual_gp.md)は数学的仮定・導出を担当する。本章は入力データ、アルゴリズム、公開API、上流委譲、数値検証のつながりを日本語で説明する。

親変数の値で子変数の有効・無効が切り替わるconditional search spaceと、context間の情報を構造的に共有するcontextual GPを区別する。カテゴリ値を持つだけの通常Mixed入力と、入力の存在条件が変わるhierarchical変数は同じではない。

## 実装の責務と公開契約

- **Implemented/Extended（family別）:** `src/robotorchan/models/structured/hierarchical.py`には`HierarchicalConditionalKernelGP`系、`src/robotorchan/models/structured/contextual.py`には`SACGP`、`LCEAGP`、`LCEMGP`等のcontext系がある。
- **Delegated:** GPの基本的なposterior演算など、BoTorch/GPyTorchへ委譲する部分と、条件付きkernel・context embeddingを構築するlocal部分はコード経路ごとに分けて確認する。
- 同一のcontext IDを設計カテゴリ、task index、あるいはlatent embeddingとして扱うモデルを一括りにしない。

## 最小の利用例

```python
import torch
from robotorchan.models import SACGP
# Context構造の詳細な入力契約はモデル別ガイドに従う。
print(SACGP.__name__)
```

コードは確認した公開APIを示す最小例であり、今回のセッションで実行して成功したとの主張ではない。tensorのshape、dtype/device、入力変数の意味と、posteriorの出力構造を意識して利用する。

## 検証・未検証事項

上の例は公開シンボルの入口確認であり、学習・posterior実行例ではない。conditional inactive dimensionsの扱い、contextごとの入力構造、観測に存在しないcontextへの汎化条件はfamilyごとに制限がある。詳細は[モデル選択ガイド](../../../models/hierarchical_contextual.md)と[Hierarchical Notebook](../../../../examples/notebooks/12_hierarchical_gp.ipynb)、[Context Notebook](../../../../examples/notebooks/14_contextual_gp.ipynb)を参照する。

[旧Model Mapping](../models.md)および[検証記録](../verification.md)は移行元の正本として保持する。API存在、静的assertion、CI成功、独立数値検証は区別する。使用例と詳しいモデル選択は[モデルガイド](../../../models/README.md)を参照する。
