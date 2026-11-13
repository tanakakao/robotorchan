# Binary Classification Extension — 対応表と完了判定

この表は公開モデルregistryと実装済みの分類Active Learning APIに基づきます。
**「対応」はAPI/実装の存在を意味し、全構成での精度・性能保証ではありません。**

## Phase 2〜7：モデル対応

| Registry ID | 入力 | モデル | 追加指定 | Phase |
| --- | --- | --- | --- | --- |
| `binary.multifidelity` | 連続+fidelity | `MultiFidelityBinaryGPClassifier` | `data_fidelities` | 2 |
| `binary.mixed_multifidelity` | Mixed+fidelity | `MixedMultiFidelityBinaryGPClassifier` | `cat_dims`, `data_fidelities` | 3 |
| `binary.spectral_mixture` | 連続 | `SpectralMixtureBinaryGPClassifier` | `num_mixtures` | 4 |
| `binary.mixed_spectral_mixture` | Mixed | `MixedSpectralMixtureBinaryGPClassifier` | `cat_dims`, `num_mixtures` | 5 |
| `binary.contextual` | 連続+context ID | `ContextualBinaryGPClassifier` | `context_feature` | 6 |
| `binary.hierarchical` | 連続+group/subgroup ID | `HierarchicalBinaryGPClassifier` | `group_feature`, `subgroup_feature` | 6 |
| `binary.infinite_width_bnn` | 連続 | `InfiniteWidthBNNBinaryGPClassifier` | `depth` | 7 |
| `binary.mixed_infinite_width_bnn` | Mixed | `MixedInfiniteWidthBNNBinaryGPClassifier` | `cat_dims`, `depth` | 7 |

すべてBernoulli変分GP分類器です。公開クラス名とregistry IDは
`src/robotorchan/models/classification/binary/registry.py`を正本とします。

## Phase 8：q-batch Classification Active Learning

| Acquisition | q=1 | q>1 | 候補間依存 | 備考 |
| --- | --- | --- | --- | --- |
| `qPredictiveEntropy` | 対応 | 対応 | 周辺集約のみ | ラベルの予測entropy |
| `qMarginUncertainty` | 対応 | 対応 | 周辺集約のみ | 予測確率の境界 |
| `qProbabilityVariance` | 対応 | 対応 | 周辺集約のみ | 確率サンプルの分散 |
| `qLatentStraddle` | 対応 | 対応 | 周辺集約のみ | 潜在GP境界 |
| `qBALD` | 対応 | q=2〜8 | 同時情報利得 | 二値ラベルの組合せを列挙 |

周辺集約型は候補間の冗長性を直接抑制しません。
qBALDは組合せ数が指数的に増えるため、実装上の上限はq=8です。
Mixed候補の探索にはカテゴリ値を固定できるoptimizer等を選んでください。
**すべてのoptimizer・制約との直積互換性を保証する表ではありません。**

## Phase 9：利用資料と検証

| 成果物 | パス | 状態 |
| --- | --- | --- |
| 理論 | `docs/theory/24_classification.md` | 追記済み |
| 利用ガイド | `docs/guide/24_classification.md` | 追記済み |
| モデル選択 | `docs/models/classification.md` | 追記済み |
| 日本語Notebook | `examples/notebooks/47_binary_classification_extensions.ipynb` | 追加済み |
| 比較スクリプト | `benchmarks/binary_classification_extensions.py` | 追加済み |
| q-batch統合テスト | `tests/acquisition/test_qbatch_extended_binary_models.py` | 追加済み |

ベンチマークは標準・Spectral Mixture・Infinite-width NNGPの3モデルを
合成二値ラベルで比較する**実行可能なスクリプト**です。
Brier score、NLL、accuracy、fit時間を記録します。
実行結果やモデルの優劣はこの表では主張しません。

NotebookはCIのmanifestで手動実行区分に登録されています。
**Notebookの全セル実行成功は、manifest登録やスキーマ検証だけでは確認できません。**

## 完了判定

- **機能実装:** Phase 2〜8のモデルと獲得関数の公開実装・統合テストを確認
- **資料整備:** Phase 9の理論・ガイド・Notebook・比較コードを追加
- **残る実証:** 新規Notebookの全セル実行、比較ベンチマークの実測、
  未検証optimizer/constraintの組合せは別途確認が必要

したがって、**計画した実装と資料整備は完了、実行結果に基づく性能評価は未完了**
と判定します。追加の広域監査や新規モデル実装はこのPhaseに含めません。
