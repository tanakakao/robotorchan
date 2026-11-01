# Initialization：獲得関数最適化の実装ガイド

## この章の問い

raw_samplesは初期候補pool数、num_restartsは局所解探索開始点の数。
Sobol等のquasi-random初期化とacquisition-weighted selectionを区別する。

数理的な正本は[理論の対応章](../../theory/optimization/03_initialization.md)とする。本章は、そこで定義される問題の**候補の座標系、内部処理、
委譲と設計上の制限**を学ぶための日本語案内である。

## 適用条件と失敗の解釈

flat acquisition、重複、one-shot補助変数、random seedの扱いで収束性能は大きく変わる。

## 明示的な初期点とSobol poolを区別する

`raw_samples`は初期候補の探索pool、`num_restarts`は局所最適化の開始数である。
`batch_initial_conditions`を明示するときは`num_restarts × q × d`のshapeを使用する。

### 実装ガイドへの参照

実装固有の対応範囲は
[Acquisition initialization](../../optimization/initialization.md) と
[初期化の対応](../../../docs-internal/verification/optimization/initialization-compatibility.md)
を source of truth とします。

理論上利用可能な heuristic が、すべての backend で公開されているとは限りません。

## 初期候補poolとrestartの接続

[BoTorch接続部](../../../src/robotorchan/optim/backends/botorch.py)は
`batch_initial_conditions`と`ic_generator`を上流へ渡します。明示初期値がない通常経路では、
BoTorchの`gen_batch_initial_conditions()`がraw candidateを生成・評価し、`initialize_q_batch()`などでrestartを選びます。
生成poolと局所探索の初期値は、役割も軸も異なります。

one-shotでは通常のq点だけではなく補助decisionの初期化も必要になるため、上流のKG向けinitializerなどへ分岐します。
この計算は[BoTorchのinitializers.py][initializers]が担当し、
robotorchanは[非線形初期値の検査][validation]とbackendの受入条件を担当します。
`validate_nonlinear_initial_conditions()`は可行性を調べる処理で、任意の非線形可行域から初期点を生成する一般ソルバではありません。

[initializers]: https://github.com/pytorch/botorch/blob/v0.18.1/botorch/optim/initializers.py
[validation]: ../../../src/robotorchan/optim/constraints/validation.py

使用手順と結果の可視化は[対応するNotebook](../../../examples/notebooks/33_acquisition_initialization.ipynb)を参照してください
。
