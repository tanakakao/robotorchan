# 高次元 benchmark 問題

周囲次元と有効次元を制御した、Tensor 上で評価できる決定的な最小化問題です。

| 問題 | 周囲次元 | 有効な構造 | 既知の最小値 |
| --- | ---: | --- | ---: |
| sparse_sphere_50 | 50 | 最初の 4 座標。残り 46 は無関係 | 0 |
| rotated_subspace_40 | 40 | 密な直交射影 2 個 | 0 |
| interaction_chain_30 | 30 | 最初の 10 座標と隣接相互作用 | 0 |

Sparse sphere は最初の 4 座標の 0.3 からの二乗偏差和です。Rotated subspace は `X - 0.5` の、正規化した全 1 方向と半分正・半分負の方向への射影二乗ノルムです。中心で最小となり、38 次元の零空間方向には一定です。

Interaction chain は最初の 10 座標の 0.4 からの二乗偏差と、隣接偏差間の二乗差を加えます。残り 20 座標は影響しません。

全次元の bounds は [0, 1] です。既知の最小値を用いて標準 GP・SAAS・射影・trust region を真値 regret で比較できますが、問題定義と統合テスト自体は順位付けの証拠ではありません。

`register_high_dimensional_problems(registry)` で登録します。
