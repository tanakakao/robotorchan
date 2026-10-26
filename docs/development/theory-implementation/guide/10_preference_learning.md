# Preference Learning：理論と実装

## 解決する問題と数理の入口

[理論の正本](../../../theory/10_preference_learning.md)は数学的仮定・導出を担当する。本章は入力データ、アルゴリズム、公開API、上流委譲、数値検証のつながりを日本語で説明する。

数値スコアではなく候補点の対比較（どちらが好ましいか）から潜在utilityを推定する。単一目的の数値回帰における`train_Y`と比較indexを同一視しない。対比較の向き、比較の重複統合、観測尤度の仮定を確認する。

## 実装の責務と公開契約

- **Extended:** `src/robotorchan/models/preference/pairwise.py`の`PairwiseGP`はBoTorch `PairwiseGP`を継承し、元のdatapoints/comparisonsをraw snapshotsとして保存する。
- **Delegated:** Laplace近似、重複の統合、posterior、pairwise likelihoodはBoTorch。
- **Implemented（接続）:** `make_mll()`は`PairwiseLaplaceMarginalLogLikelihood`を返す。標準GPの`ExactMarginalLogLikelihood`ではない。

## 最小の利用例

```python
import torch
from robotorchan.models import PairwiseGP

points = torch.tensor([[0.1], [0.5], [0.9]], dtype=torch.double)
comparisons = torch.tensor([[1, 0], [1, 2]], dtype=torch.long)
model = PairwiseGP(datapoints=points, comparisons=comparisons)
print(type(model.make_mll()).__name__)
```

コードは確認した公開APIを示す最小例であり、今回のセッションで実行して成功したとの主張ではない。tensorのshape、dtype/device、入力変数の意味と、posteriorの出力構造を意識して利用する。

## 検証・未検証事項

比較は`(勝者index, 敗者index)`の契約で扱う。比較グラフの連結性、正則化、Laplace approximationと潜在utilityの尺度を区別する。静的API検査を「全ての比較集合で学習成功」と読み替えない。対応する既存テスト・Notebookの正確な参照は移行時に照合する。

[旧Model Mapping](../models.md)および[検証記録](../verification.md)は移行元の正本として保持する。API存在、静的assertion、CI成功、独立数値検証は区別する。使用例と詳しいモデル選択は[モデルガイド](../../../models/README.md)を参照する。
