# Seed を揃えた比較

`summarize_curves({seed: curve})` で同じ長さの評価回数別曲線を集約します。seed 間の点ごとの平均、標本標準偏差（2 seed 以上で `ddof=1`）、正規近似による両側信頼区間、ソート済み seed ID を返します。

`compare_paired_curves(first, second)` は同一 seed 集合を要求し、seed ごとに `first[seed] - second[seed]` を計算して集約します。独立に推定した区間同士の減算とは異なり、seed 内の対応を保ちます。

```python
from robotorchan.benchmarks import compare_paired_curves, summarize_curves

summary = summarize_curves({0: curve_a, 1: curve_b})
paired = compare_paired_curves(strategy_a, strategy_b)
```


同じ長さ・dtype・device の有限な 1 次元曲線が必要です。区間は正規近似であり厳密な Student-t 区間ではありません。1 seed の区間幅 0 は記述上の規則で、信頼保証ではありません。

有意差検定、多重比較補正、非有限 regret の処理、共通 cost 軸への再サンプリングは行いません。seed 単位の percentile bootstrap と対応差は[多 seed 統計](multiseed_statistics.md)を参照してください。
