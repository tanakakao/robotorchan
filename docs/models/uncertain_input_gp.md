# Gaussian 不確実入力 GP

観測された訓練座標そのものに誤差がある場合、入力分布の下でカーネルの期待値を計算します。訓練入力を一度だけ jitter する方法や、意思決定時の候補摂動とは異なる surrogate の仮定です。

## 利用方法と制約

`UncertainInputSingleTaskGP` には入力と同形状の非負標準偏差 `train_X_std`、または観測ごとの共分散 `train_X_cov` を渡します。一度に使う不確実性表現は 1 つです。Gaussian 入力と RBF カーネルの解析的な期待共分散を使います。既知の入力では不確実性を 0 とし、予測時も潜在応答と観測ノイズを区別します。

## 理論・実装・検証を辿る

[理論](../theory/16_uncertain_input_gp.md)と[実装ガイド](../guide/16_uncertain_input_gp.md)で仮定・推論方法・近似を確認できます。[ソース](../../src/robotorchan/models/uncertain/uncertain_input.py)から API と処理を確認してください。

過去の設計条件・数式・検証要件は[設計記録](../../.github/project-records/model-design/uncertain_input_gp.md)に原文を保存しています。そこでの将来計画や Phase の完了条件は現在の対応保証ではありません。現在のモデル選択は[モデル索引](README.md)を参照してください。
