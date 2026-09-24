# Model selection

robotorchan では、モデルの複雑さではなく **データ生成過程と探索空間の構造** から surrogate を選びます。
まず標準 GP を基準にし、通常 GP のどの仮定が問題と合わないかを確認してください。

## 問題設定から選ぶ

| 問題設定 | 主な候補 | 判断の要点 |
|---|---|---|
| 通常の連続入力回帰 | `SingleTaskGP` | 最初のベースライン |
| 連続 + カテゴリ | `MixedSingleTaskGP` | カテゴリを連続値として扱わない |
| Fidelity がある | `SingleTaskMultiFidelityGP` | fidelity 間の情報共有 |
| 関連タスク | `MultiTaskGP`, `KroneckerMultiTaskGP` | 観測設計とタスク相関を確認 |
| 独立した複数出力 | `ModelListGP` | 出力間共分散が不要 |
| 大規模データ | `SingleTaskVariationalGP` | inducing-point approximation |
| 高次元・少数有効変数 | SAAS / MAP-SAAS | sparsity 仮定を確認 |
| 既知の低次元表現 | PCA / PLS / Random Projection 系 | reducer の意味を確認 |
| 非線形な低次元表現 | AE / VAE / Joint 系 | representation learning が必要 |
| 高次元探索空間 | REMBO / HeSBO / ALEBO / TuRBO / BAxUS | surrogate と search strategy を分けて考える |
| 外れ値・heavy tail | Relevance Pursuit / Student-t / Contaminated | 外れ値生成機構の仮定が異なる |
| 入力依存ノイズ | Heteroskedastic / Joint Heteroskedastic | mean と noise の構造を区別 |
| 反復測定 | Replicate Noise | replicate から noise を推定 |
| 入力値が不確か | Uncertain Input | observation noise とは別問題 |
| カテゴリ入力が不確か | Uncertain Categorical | カテゴリ確率を扱う |
| 非定常な応答 | Nonstationary GP | noise 非定常性との違いに注意 |
| Tensor / structured output | HigherOrder / LatentKronecker | 出力軸の構造を利用 |
| 条件付き探索空間 | Hierarchical GP | 親変数で有効次元が変化 |
| Context 構造 | SAC / LCEA / LCEM | context の意味と観測形式を確認 |
| Preference data | `PairwiseGP` | 絶対値ではなく比較を学習 |

## Mixed は独立した問題軸

Mixed は「モデルファミリー」だけではなく入力空間の性質です。Robust、Multi-task、
Multi-Fidelity、Reduced、高次元モデルにも Mixed 版があります。

カテゴリ変数があるという理由だけで `MixedSingleTaskGP` に固定せず、まず
Robust / Multi-task / Reduced などの問題構造を決め、その上で対応する Mixed 実装が
あるかを確認してください。

## Multi-task と Multi-objective を区別する

Multi-task は関連タスク間で統計的に情報共有するモデル化です。Multi-objective は複数目的を
同時に最適化する問題設定です。複数目的だから `MultiTaskGP` が必須というわけではなく、
独立 GP を `ModelListGP` でまとめる構成もあります。

## 高次元では model と search strategy を分ける

高次元問題では、surrogate model の表現能力と acquisition function の探索方法を分離して
判断します。SAAS や Reduced GP は surrogate 側、REMBO / HeSBO / ALEBO / TuRBO /
BAxUS は主に探索側の選択肢です。

詳細は [高次元モデル選択](../models/high_dimensional_model_selection.md) と
[高次元探索戦略](../optimization/high_dimensional_search.md) を参照してください。

## Robust / Noise / Uncertainty は原因から選ぶ

外れ値、heavy-tailed residual、入力依存ノイズ、反復測定、入力不確かさ、非定常性は
似た予測誤差を生むことがありますが、モデルが仮定する生成機構は異なります。

詳細は [Robust / Noise model guide](../models/robust_noise.md) と
[モデルファミリー別ガイド](../models/README.md) を参照してください。

## 詳細へ進む

このページはモデル選択の入口であり、個別モデルの説明を置き換えるものではありません。
各モデルファミリーの用途、API、制約、Mixed 対応、Notebook は
[Model guides](../models/README.md)、全体索引は [モデル概要・使い所](../models.md)、
理論的背景は [Theory](../theory/README.md)、実行例は
[Examples](../../examples/README.md) を参照してください。
