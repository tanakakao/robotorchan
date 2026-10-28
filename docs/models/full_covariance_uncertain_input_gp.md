# 相関のある不確実入力

入力誤差が次元間で相関している場合、各観測の Gaussian 入力分布を完全な共分散行列で表します。対角標準偏差は独立誤差に対応し、共通の校正誤差などによる相関には完全共分散を使います。

## 利用方法と制約

`UncertainInputSingleTaskGP` の `train_X_cov` は観測ごとに対称な半正定値共分散を表します。`train_X_std` と同時に指定しません。期待 RBF 共分散の計算では安定した線形代数を使い、逆行列を直接形成することを要求しません。候補摂動・環境 scenario は別の層で扱います。

## 理論・実装・検証を辿る

[理論](../theory/16_uncertain_input_gp.md)と[実装ガイド](../guide/16_uncertain_input_gp.md)で仮定・推論方法・近似を確認できます。[ソース](../../src/robotorchan/models/uncertain/uncertain_input.py)から API と処理を確認してください。

過去の設計条件・数式・検証要件は[設計記録](../../.github/project-records/model-design/full_covariance_uncertain_input_gp.md)に原文を保存しています。そこでの将来計画や Phase の完了条件は現在の対応保証ではありません。現在のモデル選択は[モデル索引](README.md)を参照してください。
