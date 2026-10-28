# 軌跡単位の結果保存

Python pickle を実行せず、version 付き JSON で軌跡を入出力します。

```python
from robotorchan.benchmarks import load_trajectory, save_trajectory

save_trajectory(trajectory, "run-42.json")
restored = load_trajectory("run-42.json")
```


seed、初期設計数、X、観測目的、真値目的、制約残差、費用、dtype（float32・float64）を保持し、Tensor は CPU に読み込みます。`schema_version` で互換性を明示的に確認します。

不正な形状、未対応 dtype、必須 Tensor の欠落、非有限値、非互換 version は拒否します。結果を再現可能に検査するための形式で、モデル checkpoint ではありません。完全な再現には、この最小 schema に含まれない問題定義・アルゴリズム設定・ライブラリ version・seed も保存してください。

JSON の十進表現は全浮動小数点状態や RNG 状態の bit 単位の保持を保証しません。legacy alias と pickle fallback はありません。
