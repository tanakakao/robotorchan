# Unified Factory and Capability-based Workflow API — Phase 1

基準: main / 4a8ae1fb6a34b62a70cf7bd3a7d1da167ea3deea

## 設計契約

BoTorch-first。既存クラスを直接生成し、暗黙の学習、引数破棄、fallback、不要な wrapper/alias を禁止する。既存直接利用 API を維持する。

| API | 配置 | 契約 |
|---|---|---|
| create_model, get_model_class, list_models | models | 正式登録名、クラス、Callable（生成のみ） |
| create_acquisition, get_acquisition_class, list_acquisitions | acquisition | 既存クラスまたは生成関数。関数登録は class 取得の対象外と明示 |
| create_optimizer, list_optimizers | optim | optimize_acqf の設定済み Callable または既存専用 backend |
| create_search_strategy, list_search_strategies | optim | 既存 strategy インスタンス |
| check_compatibility | workflow | 既存互換性ルールの合成 |
| recommend_compatible_workflows | workflow | 適合候補列挙 |
| recommend | workflow | 根拠のある選択。順位根拠がなければ候補提示 |
| create_study | study | 既存 API 拡張、文字列指定と明示的 auto |

Factory の名前指定とコンストラクタ引数は分離する（例: create_acquisition("qLogEI", model=model, best_f=best_f)）。識別子は positional-only を基本とし、model 等の固有引数と衝突させない。生成結果は元の型を保持する。未登録 Callable の生成可能性と capability 確認済みを区別する。optional dependency は既存遅延 import を尊重する。

## 管理元と対象境界

- models/registry.py: ModelRegistryEntry の能力情報と生成解決を接続する。models/classification/registry.py と binary/registry.py の既存登録 ID を維持し、同一情報をコピーしない。models/__all__、classification/__all__ および公開サブパッケージ、特殊モデル公開 API を対象の起点とする。補助型・尤度・メタデータはモデルと区別する。
- acquisition/registry.py: BoTorch metadata-only entry を既存上流クラスへ接続。acquisition/__all__ の全獲得関数および make_* 生成関数、特殊ワークフロー内の公開獲得関数を対象にする。bridge、sampler、objective、候補選択関数は役割を区別する。
- optim/capabilities.py と optim/dispatch.py: backend 名と能力の既存管理元を使用する。dispatch の Literal と capability のキーが一致するとは仮定しない。NSGA-II、mixed 専用、one-shot、safe、contextual の専用経路を維持し、共通 dispatch で実行できないものを誤登録しない。
- optim/base.py の SearchStrategy / SearchResult を維持。OriginalSpace、MixedSpace、Random、Sobol、HammingLocalSearch、TreeEnsemble、ALEBO、REMBO、HeSBO、BAxUS、BAxUSThompsonSampling、LatentSpace、DiscreteTrustRegion、TuRBO と公開サブパッケージの追加戦略を対象とする。状態型と候補提案関数（BOCS/COMBO 等）は専用 API と区別する。
- workflow/problem.py: 既存 purpose/input_type/task_type/output_type/objective_type 等の意味を保持する。文字列は Enum へ正規化する。観測形式（二値、多クラス、順序、異種出力）、目的数、制約種別は別概念とする。
- workflow/selector.py、acquisition/compatibility.py、acquisition/optimizer_compatibility.py: 既存ルールを拡張する。対応、条件付き、非対応、未確認、適用対象外を区別し、未確認を auto 選択しない。静的判定と実行時検証を分ける。
- workflow/recommendation.py: Recommendation を拡張し、optimizer、strategy、必要引数、理由、代替候補を保持する。互換性を性能ランキングと混同しない。
- benchmarks/capability.py: 既存 selector/recommendation を参照するため共通管理元へ追随させる。benchmarks/botorch_strategy.py の model_factory Callable (X, train_Y) を維持し、登録名解決は同じ Factory に接続する。BenchmarkProblemRegistry は問題生成専用なので統合しない。
- study/generation.py と study/study.py: 既存 CandidateGenerator と create_study を拡張する。ask()/ask_batch(q)/tell() の Trial 契約を維持する。現状はスカラー目的中心であり、多目的・分類・特殊 workflow の生成可能性と Study 実行可能性を区別する。保存・再開は既存機構を確認し、未対応設定を黙って失わない。

## 実装順とゲート

| Phase | 作業 | 完了条件 |
|---|---|---|
| 2 | 全公開モデルの生成解決、分類登録統合 | 全対象の生成契約、型、固有引数、学習・予測の代表実行 |
| 3 | 全公開獲得関数の生成解決 | 全対象の生成契約、代表評価、非対応理由 |
| 4 | 全 backend と専用 optimizer 経路 | 候補提案、mixed/q/constraints、既存 dispatch 回帰 |
| 5 | 全公開探索戦略 | 生成、状態更新、候補提案、専用 API の明示 |
| 6 | ProblemSpec と横断 selector | 条件検索、未確認と条件付きの区別、重複管理なし |
| 7 | 互換性と recommendation | 組み合わせ全体の判定、明示指定維持、生成可能な推奨 |
| 8 | Study/Ask-and-Tell/DataFrame/Benchmark | 対応ループ、保存再開契約、既存 API 回帰 |
| 9 | Notebook/Guide/README/API reference | 候補提案までの実行例、可視化、全対象の対応記録、CI 成功 |

各 Phase で公開 export と登録の差分を機械的に照合し、実装と同時に解消する。個別クラスの最終対象表は Phase 2–5 の各 PR に記録する。状態は実装済み、既存対応、条件付き、専用 API、対象外、未対応とし、対象外には理由と代替経路を記載する。未対応を対象外に変更して完了扱いしない。全直積の実行テストや無関係な理論再監査は行わない。各 Phase は独立 PR、必要なテストと CI を確認して次へ進む。

## Phase 1 の限界

主要実装と公開 export の確認に基づく設計であり、全クラスの個別生成テスト、特殊 workflow の全公開経路、保存再開の実行検証は未実施。これらは各実装 Phase のゲートで確認する。今回の PR は設計文書のみで、ランタイム変更はない。
