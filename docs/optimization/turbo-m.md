# TuRBO-m：複数Trust Regionによるベイズ最適化

TuRBO-mは、複数のTrust Regionを独立して管理し、各領域のGPから生成した候補をThompson Samplingで競争させる方法です。現在のrobotorchan実装は**連続変数・単一目的・同期q-batch**を対象とします。

既存の`TuRBOStrategy`（TuRBO-1）は変更せず、`TuRBOMultiRegionState`を別の管理クラスとして提供します。

## 構成と責務

| 処理 | API | 責務 |
| --- | --- | --- |
| 領域の初期化 | `TuRBOMultiRegionState(bounds, centers)` | 各領域の中心、観測データ、状態を保持 |
| モデル登録 | `set_model(region_id, model)` | **学習済み**GPを登録。学習処理は行わない |
| 候補提案 | `propose_batch(q=..., n_candidates=...)` | 領域間Thompson競争、候補・領域ID・割当数を返す |
| 評価結果の反映 | `update_regions(X, Y, region_ids)` | 担当領域ごとの最良値・中心・成功失敗カウンタ・領域幅を更新 |
| Restart | `restart_region(region_id, seed=...)` | 対象領域の学習データとモデルをクリア。履歴は保持 |

`propose_batch`の返り値は`(X, region_ids, counts)`です。候補点`X`は`[q, d]`、領域IDは`[q]`、割当数は`[m]`です。領域別の割当は固定ではなく、Thompson Samplingのスコアで決まります。必要なら`max_per_region`で1領域当たりの上限を設定できます。

## 実行手順

1. Global bounds内に`m`個の初期中心を用意し、`TuRBOMultiRegionState`を作成します。
2. 各領域に初期観測値を`append_observations`で登録し、最良観測点を中心に設定します。初期最良値は`TuRBOState.best_value`と`observed_best_value`にも反映します。
3. 各領域の`train_X`と`train_Y`からBoTorchの`SingleTaskGP`を構築・学習し、`set_model`で登録します。
4. `propose_batch(q=...)`で候補と割当領域IDを取得し、目的関数を評価します。
5. `update_regions(X, Y, region_ids)`で観測を元の領域へ返します。
6. `restart_triggered`の領域だけ`restart_region`を実行し、初期データの再取得とGP再学習を行います。

実際の学習ループは [実行用Python例](../../examples/turbo_multi_benchmark.py) と [Notebook](../../examples/notebooks/turbo_multi_benchmark.ipynb) を参照してください。Notebookは、GP学習、q-batch提案、観測更新、Restartを含む一連の処理を実行します。

## TuRBO-1との違い

| 項目 | TuRBO-1 | TuRBO-m |
| --- | --- | --- |
| Trust Region | 1領域 | 複数領域 |
| GP | 単一領域のモデル | 領域ごとに独立したモデル |
| 候補選択 | 1領域の局所探索 | 複数領域の候補をグローバルに競争 |
| 領域幅・Restart | 単一状態 | 各領域で独立 |
| 観測帰属 | 単一領域 | `region_ids`により管理 |

## ベンチマークの読み方

NotebookはAckley・Rosenbrock・Levy（2次元）で、1領域と3領域を比較します。両者の初期評価数は**合計15点**、追加評価は**q=2 × 8反復**に揃えています。スコアは目的関数の符号を反転した値で、**大きいほど良い**設定です。

**注意：** 1領域のベースラインは`TuRBOMultiRegionState`の`m=1`設定です。既存の`TuRBOStrategy`そのものを用いた比較ではありません。また、試行回数1回の結果から性能優位性を主張することはできません。

## 現在の制限

- GPの生成・学習・Restart後の再初期化は利用者側のループが担当します。高水準の自動実行APIはまだありません。
- 混合変数、非同期評価、多目的、MORBO、制約付き領域間候補選択は今回の対象外です。
- `region_ids`は候補生成時のものを評価完了時まで保持してください。
- ベンチマークは機能確認用の小規模例であり、統計的な優劣を示すものではありません。


## Phase 8：対応状況と検証の到達点

この表は実装コードと既存のTuRBO-mテストを対象とした機能棚卸しです。
性能上の優位性や、すべての実行環境での動作を証明するものではありません。

| 項目 | TuRBO-1（`TuRBOStrategy`） | TuRBO-m（`TuRBOMultiRegionState`） |
| --- | --- | --- |
| 連続変数・単一目的 | 対応 | 対応 |
| 独立した複数のTrust Region | 対象外 | 対応 |
| 領域別のGPデータとモデル参照 | 対象外 | 対応 |
| Thompson Sampling候補生成 | 対応 | 対応 |
| q-batch候補提案 | 対応 | 対応 |
| 領域間の動的q割当 | 対象外 | 対応 |
| 領域別観測帰属と状態更新 | 対象外 | 対応 |
| Restart | 対応 | 領域別に対応 |
| Mixed / discrete | 個別の既存経路あり | 未対応 |
| CandidateConstraints | 個別の既存経路あり | 未対応 |
| 非同期pending点 | 個別の既存経路あり | 未対応 |
| 多目的・multi-fidelity | 本比較の対象外 | 未対応 |

### 既存の検証資産

- `tests/optim/trust_region/test_turbo_multi_state.py`：領域別データと状態管理
- `tests/optim/trust_region/test_turbo_multi_sampling.py`：領域間候補競争と割当
- `tests/optim/trust_region/test_turbo_multi_updates.py`：観測帰属、成功・失敗カウント、局所Restart
- `examples/notebooks/turbo_multi_benchmark.ipynb`：GP fit、候補提案、観測更新、再始動をつなぐ例
- `examples/turbo_multi_benchmark.py`：Ackley、Rosenbrock、Levyの小規模比較

### 実装確認で区別すべき仕様

TuRBO-mの`update_regions`では、成功は割当領域の完了バッチにつき1回、
失敗はその領域に割り当てた点数分を加算します。
一方、TuRBO-1の`update_turbo_state`は完了バッチにつき1回の判定です。
この差はベンチマークの解釈に影響するため、同一アルゴリズムとみなして
性能を比較しません。

TuRBO-mのRestartでは対象領域の局所最良値とモデルを初期化し、
`history_X`と`history_Y`を保持します。TuRBO-1のRestartが
最良値を保持する規則とは区別します。

### 完了判定

**連続・単一目的・同期q-batchのTuRBO-m機能実装は完了。**
既存テストは上記の責務に沿って配置済みです。
ただし、以下は検証済みとは主張しません。

- 既存`TuRBOStrategy`を直接用いた、評価予算を揃えた性能比較
- 複数seedに基づく統計的な性能評価
- 新規NotebookをCI上で実行した結果（Notebookはmanual区分）
- 未対応領域（Mixed、制約、非同期、多目的など）の動作

このため**機能実装の完了**と**ベンチマークによる性能評価の完了**は
別の判定とします。性能比較は必要になった際の独立したベンチマーク作業とし、
本Phaseの完了条件に無制限な追加テストを含めません。
