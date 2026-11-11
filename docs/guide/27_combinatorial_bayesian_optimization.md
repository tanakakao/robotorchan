# 組合せベイズ最適化：理論と実装の対応

[理論](../theory/27_combinatorial_bayesian_optimization.md) /
[実行Notebook](../../examples/notebooks/49_combinatorial_bayesian_optimization.ipynb) /
[性能比較](../../examples/combinatorial_bo_benchmark.md)

## 対応表

| 機能 | 公開API / 実装 | 制約 |
| --- | --- | --- |
| Hamming距離・近傍 | `hamming_distance`, `hamming_neighbors` | 離散MixedVariableSpace |
| 局所探索 | `HammingLocalSearchStrategy` | q=1、獲得関数が必要 |
| Discrete TR | `DiscreteTrustRegionState`, `DiscreteTrustRegionStrategy`, `update_discrete_trust_region` | 状態更新と中心の変更は呼び出し側が管理 |
| BOCS二次モデル | `BOCSSurrogate`, `bocs_quadratic_features` | 二値、ガウス係数事前分布 |
| BOCS逐次探索 | `propose_bocs_candidate`, `run_bocs_sequential` | 二値、q=1、全候補列挙 |
| COMBOグラフGP | `COMBOSurrogate`, `BinaryGraphDiffusionKernel` | 二値入力 |
| COMBO逐次探索 | `propose_combo_candidate`, `run_combo_sequential` | 二値、q=1、全候補列挙 |

## 処理の流れ

1. 二値入力の `train_X`（形状 `[n,d]`）と目的値 `train_Y`（`[n,1]`）を準備します。
2. BOCSまたはCOMBOのモデルを学習し、BoTorchの獲得関数で未観測候補を評価します。
3. `propose_*_candidate` は `[1,d]` の候補を返します。
4. 逐次関数は目的関数を呼び出して観測を追加し、次の反復で再学習します。

`run_*_sequential` の目的関数は `[1,d]` を受け取り、
有限の `[1,1]` テンソルを返す必要があります。
最大化・最小化は `maximize` で切り替えられます。
候補空間を使い切るとエラーになります。評価回数は残候補数以下に設定してください。

## 実装上の区別

- BOCSの係数事前分布は等方ガウスであり、原著のスパース事前分布ではありません。
- COMBOはグラフ拡散GPを再利用しますが、逐次候補選択は列挙UCBです。
- `max_dimensions` は列挙の上限を制御します。大規模な組合せ空間に直接拡張できません。
- Hamming局所探索とDiscrete TRは候補探索層です。モデルの学習・状態更新は別責務です。
- `CandidateConstraints` はHamming探索の対応範囲内で指定できますが、
  BOCS/COMBOの逐次APIへ同じ制約を渡せるわけではありません。
- Mixed GAなどのベンチマーク比較については性能比較資料に記載した範囲を確認してください。

## 実装ファイル

- `src/robotorchan/optim/domains/combinatorial.py`
- `src/robotorchan/optim/strategies/combinatorial.py`
- `src/robotorchan/optim/trust_region/discrete.py`
- `src/robotorchan/models/standard/bocs.py`
- `src/robotorchan/models/standard/combo.py`
- `src/robotorchan/optim/strategies/bocs.py`
- `src/robotorchan/optim/strategies/combo.py`
