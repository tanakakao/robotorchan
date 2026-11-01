# Inner optimization：獲得関数最適化の実装ガイド

## この章の問い

獲得関数の最大化はblack-box objectiveの評価とは異なる内側の最適化。q点を同時に選ぶ場合はqd次元のjoint変数を扱う。

数理的な正本は[理論の対応章](../../theory/optimization/01_foundations.md)とする。本章は、そこで定義される問題の**候補の座標系、内部処理、
委譲と設計上の制限**を学ぶための日本語案内である。

## 適用条件と失敗の解釈

同じ獲得関数でも初期化・bounds・feasibilityにより結果が変わる。局所解と大域保証を区別する。

## 補足となる実装上の設計

#### Sample Average Approximation：実装上の補足

BoTorch 系の MC acquisition では、base samples を固定したsample-average approximation を candidate input
に関して最適化できます。

## 関連する実装資料

joint と sequential は単なる API option の違いではなく、解く inner optimization problem の構造が異なります。

ライブラリ固有の対応状況と API は
[Optimization guides](../../optimization/README.md) を参照してください。

## 公開dispatcherから数値探索へ

[dispatch.py](../../../src/robotorchan/optim/dispatch.py)の`optimize_acqf()`は
bounds・q・変数型・制約を検査した後、backendごとの関数へ分岐します。`_validate_requested_capabilities()`は、
渡された制約や変数構造がbackendで扱えるかを照合します。この共通入口は獲得関数の数式や探索更新則を持ちません。

[backends](../../../src/robotorchan/optim/backends/)のうち、BoTorch経路は
上流の初期化・局所探索へ委譲し、Random / Sobolは独自の有限pool選択を行います。Theoryの内側の最大化問題は共通ですが、探索の近似と制約処理は経路ごとに異なります。
戻る獲得値は数値探索で計算したutilityであり、外部の目的関数の新しい観測ではありません。

使用手順と結果の可視化は[対応するNotebook](../../../examples/notebooks/31_acquisition_optimization.ipynb)を参照してください。
