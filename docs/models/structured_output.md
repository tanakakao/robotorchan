# Structured output

## HigherOrderGP

`HigherOrderGP` と `MixedHigherOrderGP` は画像、スペクトルグリッド、空間分布など tensor-valued output を扱います。出力テンソルの軸構造そのものに意味がある場合に候補になります。

## LatentKroneckerGP

`LatentKroneckerGP` と `MixedLatentKroneckerGP` は、入力 `X` と時間・波長・位置などの出力座標 `T` を明示的に扱います。新しい `T` 位置での補間・予測が必要な問題で特に自然です。

`HigherOrderGP` と `LatentKroneckerGP` は「出力次元が多い」という理由だけで交換可能ではありません。出力軸を tensor structure として扱うか、明示的座標として扱うかで選択します。

Notebook: [Structured output](../../examples/notebooks/11_structured_output_gp.ipynb)  
Theory: [Structured Output](../theory/11_structured_output.md)
