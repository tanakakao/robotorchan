# 異種 E2E の合成問題

実装は `src/robotorchan/benchmarks/heterogeneous_synthetic.py` です。温度 x0、圧力 x1、組成 x2 を [0,1] の連続変数とします。説明用の正規化座標であり、校正した物理量ではありません。

- Strength：`1.3 - 2(x0-.78)^2 - 1.5(x1-.28)^2 - .7(x2-.68)^2`。
- Conductivity：`1.2 - 1.8(x0-.22)^2 - 1.6(x1-.76)^2 - .9(x2-.35)^2`。
- Pass 確率：`sigmoid(9(.52 - (x0-.50)^2 - 1.2(x1-.53)^2 - .7(x2-.50)^2) - 2.5)`。

両回帰目的を最大化し、異なる最適点により trade-off が生じます。真値確率と観測二値 label `Pass ~ Bernoulli(p(x))` は別です。参照実行可能性は `p(x) >= 0.5` であり、観測 Pass=1 や予測確率で代替しません。

`observe(X, seed, noise_std=0.02)` はローカル generator で回帰の独立 Gaussian ノイズと Bernoulli label を抽出します。同じ seed・入力・形状で再現しますが、反復間の独立性は実験ループで管理してください。

`reference_front(grid_size=25, probability_threshold=0.5)` は決定的な 3 次元格子を真の実行可能性で選別し、非支配目的対を返します。連続 Pareto front の解析解ではなく離散近似です。格子解像度を報告し、細かくした際の収束を確認します。hypervolume reference point は最大化空間の実行可能目的値より低く設定して記録してください。

データ生成過程の定義だけでは、学習後の較正、制約付き獲得関数、closed-loop の優越性、CUDA 対応を保証しません。

## 契約テスト

```bash
pytest -q tests/benchmarks/test_heterogeneous_synthetic.py
```
