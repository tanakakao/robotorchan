# 獲得関数最適化の初期化

独立した初期化体系を設けず BoTorch に合わせます。restart 軸と候補軸は optimizer が所有し、モデル・獲得関数が加える batch・sample 軸とは区別します。

## 標準の契約

通常の自動初期化は BoTorch へ委譲します。明示的初期値は `num_restarts x q x d` です。`num_restarts`、`raw_samples`、`batch_initial_conditions`、`fixed_features`、対応する候補制約は BoTorch と同じ意味を保ちます。

| 場合 | 初期化 |
| --- | --- |
| 連続 q-batch | BoTorch 標準 |
| Sequential q | 各 greedy 段階で新しい q=1 初期化 |
| 線形制約 | BoTorch の実行可能な初期化 |
| 非線形制約 | 明示的初期値または適切な `ic_generator` |
| Mixed 空間 | BoTorch mixed 最適化の初期化 |
| 固定 task・fidelity | 通常の fixed-feature 初期化 |
| MultiTask・Kronecker | モデル専用初期化は不要 |
| 多目的 | 目的専用初期化は不要 |
| 入力摂動・robust objective | nominal `q x d` を初期化し、scenario 軸は内部に保つ |
| `X_pending` | 獲得関数の文脈として扱い、初期値へ連結しない |
| Fantasy model | fantasy 軸はモデル batch 軸であり restart・q 軸ではない |

点間制約には joint-q の意味を保てる BoTorch モードが必要です。独立の q=1 部分問題を逐次解く方法では点間制約を保てません。

## One-shot 獲得関数

KG・MF-KG は BoTorch 専用の初期化を使います。明示的初期値には `acq_function.get_augmented_q_batch_size(q)` が返す拡張 q を使い、最終結果は実際の要求候補だけを含みます。

`qMultiStepLookahead` など専用初期化器のない `OneShotAcquisitionFunction` には、`ic_generator` として `gen_augmented_one_shot_initial_conditions` を使います。拡張 q を計算した後のサンプリングと restart 選択は BoTorch `gen_batch_initial_conditions` に委譲し、別のヒューリスティックは使いません。

Mixed one-shot は実候補と fantasy 行でカテゴリ割当が異なり得る専用経路です。[専用ガイド](mixed-one-shot-optimization.md)を参照してください。

## 高次元探索

モデル側の次元削減は optimizer 座標を変えず、元の公開入力空間で初期化します。REMBO・HeSBO など optimizer 側の embedding は探索座標で初期化して元空間へ戻します。

ALEBO の許容領域は box でなく polytope です。polytope sampler で実行可能な初期値を作り、逐次 q=1 でも generator から新しい実行可能点を得ます。

## Hybrid

大域探索の候補を `batch_initial_conditions` として共通 BoTorch backend へ渡して局所改良します。制約、初期値検証、dtype・device、勾配最適化を通常経路と共通化します。

## Capability metadata

`OptimizerCapabilities` に初期化専用フラグは設けません。標準経路は既存の q-batch・逐次・mixed・fixed-feature・制約 capability に従い、重複 metadata と実行契約の矛盾を防ぎます。

非線形制約の generator・明示的初期値や、one-shot の拡張 q は API 自体で要件を表します。embedding 戦略が初期化を所有するのは、その探索幾何が必要とする場合です。
