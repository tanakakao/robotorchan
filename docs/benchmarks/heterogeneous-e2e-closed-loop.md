# Closed-loop BO の確認

合成 Strength で qEI または qNEI を使い、逐次 3 反復を実行する統合テストです。各反復で全観測から Exact GP を再学習し、semantic 合成層で新しい獲得関数を構成、Sobol optimizer で bounds 内の 1 点を選びます。反復固有の決定的ノイズ seed で観測し、データへ追加します。

16 行から 19 行への増加、初期観測の不変性、候補・観測の有限性、記録履歴と追加行の一致を検査します。qNEI の baseline は毎回現在の訓練入力で更新します。

小規模な制約なし単一目的の動作確認です。収束、Random に対する優越性、多 seed の統計的再現性、分類制約付き closed-loop を保証しません。それらには独立の実行根拠と実測 artifact が必要です。[概要](README.md)と[実測プロトコル](empirical-evaluation-protocol.md)を参照してください。

`pytest -q tests/benchmarks/test_heterogeneous_closed_loop_e2e.py` で実行します。
