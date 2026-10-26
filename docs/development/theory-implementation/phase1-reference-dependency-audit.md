# Phase 1 — リンク・CI・検証参照の依存関係監査（初版）

> Baseline: `main` `9de071c899dcce5fa2c7c3c0083281d529a3b077`。既存5ファイルは変更しない。静的な参照抽出と指定ファイルの直接読取に基づく。実行検証や全パスの存在確認は未実施。

## 既存5ファイルのMarkdownリンク（重複除外）

| Source | Distinct link targets |
| --- | --- |
| `README.md` | `models.md`, `../../theory/README.md`, `verification.md#phase-7-final-exit-reconciliation`, `acquisition.md`, `verification.md`, `optimization.md`, `README.md` |
| `models.md` | `README.md`, `../../theory/README.md`, `verification.md`, `README.md#reader-navigation-phase-10`, `../../../tests/models/standard/test_phase7_kronecker_covariance_reference.py` |
| `acquisition.md` | `README.md`, `../../theory/README.md`, `verification.md`, `README.md#reader-navigation-phase-10` |
| `optimization.md` | `README.md`, `../../theory/README.md`, `verification.md`, `README.md#reader-navigation-phase-10` |
| `verification.md` | `README.md`, `models.md`, `acquisition.md`, `optimization.md`, `../../theory/README.md` |

**移行時の高リスクアンカー**: `README.md#reader-navigation-phase-10` と `verification.md#phase-7-final-exit-reconciliation`。旧ファイルの削除・改題・見出し変更で壊れるため、旧ファイル保持または互換アンカーを確保する。

## CIで直接確認した依存

`.github/workflows/documentation-links.yml` はPRとmain pushで以下を実行する。

1. `python -m unittest discover -s tests/docs -p test_docs_links.py`
2. `python scripts/check_docs_links.py`
3. `python scripts/check_notebook_links.py`
4. `python scripts/check_docs_references.py`
5. `python -m unittest discover -s tests/docs -p test_theory_implementation_mapping.py`
6. `python scripts/check_theory_implementation_mapping.py`

`scripts/check_theory_implementation_mapping.py` は **既存5ファイルの存在を必須**とし、README→models/acquisition/optimization、verification→各subject、各subject→README/verification/theory の **特定のMarkdownリンク文字列**を要求する。新しい章を追加するだけではこのチェックは失敗しないが、旧ファイルを移動・削除・リンクを書換えると失敗する。Phase 2以降はチェック契約の意図を維持しながら新構造へ拡張する必要がある。

`tests/docs/test_theory_implementation_mapping.py` は、現行ナビゲーションが空エラーであること、README欠損の検出、subject→verificationの欠落検出をassertしている。変更時にこれらの否定テストを消してはいけない。

## 本文中の実装・テスト・Notebook参照（静的抽出の例）

| Source | 確認された参照例 | 保全上の意味 |
| --- | --- | --- |
| `README.md` | `src/robotorchan/models/registry.py`, `src/robotorchan/acquisition/registry.py`, `scripts/generate_model_coverage.py`, `docs/model_coverage.json` | registryと生成成果物のsource of truth |
| `models.md` | `tests/models/standard/test_mixed_single_task_gp.py`, `tests/models/standard/test_phase7_kronecker_covariance_reference.py`, `tests/models/classification/robust/test_label_noise.py` | model family別のassertion根拠 |
| `acquisition.md` | `tests/acquisition/test_phase5_native_integration.py`, `tests/acquisition/test_epig.py`, `examples/notebooks/28_regression_active_learning.ipynb` | acquisitionの実装・テスト・実行例 |
| `optimization.md` | `tests/optim/one_shot/test_mixed.py`, `tests/optim/embedding/test_baxus_reference_schedule.py`, `tests/optim/trust_region/test_turbo_strategy.py` | one-shot、embedding、stateful strategy |
| `verification.md` | `tests/models/classification/test_phase7_binary_probit_reference.py`, `tests/acquisition/test_phase8_bald_probability_variance_reference.py`, `tests/optim/trust_region/test_phase9_turbo_geometry_reference.py` | 数値参照のproperty-specific証拠 |

`README.md` のインラインコード形式のpath候補は18件、`models.md` は158件、`acquisition.md` は42件、`optimization.md` は36件、`verification.md` は21件（各ファイル内の重複除外、正規表現による抽出）。**存在確認済み件数ではない**。 `models/...` のような省略パス、`::Class` 指定、`{a,b}.py` の集合表記は、そのままファイルパスとして検査しない。

## 外部からの参照について

GitHubコード検索で `check_theory_implementation_mapping` は `README.md`、`tests/docs/test_theory_implementation_mapping.py`、`.github/workflows/documentation-links.yml` に見つかった。 `28_regression_active_learning` は `examples/run_notebooks.py` と `acquisition.md` に見つかった。検索結果は**網羅的な被参照リストではない**ため、全リポジトリのtreeと全文走査が必要。

## 移行ガードレール

- 旧5ファイルと既存のアンカーは、新ガイドへの移行期間中維持する。
- 新しい章の相対リンクは必ず新しいファイル位置を基準に解決する。
- `docs/model_coverage.json` と生成スクリプトの所有関係を変更しない。
- 新ガイドのCIチェック追加は、現行の否定テストとリンク検査を保持してから行う。
- 数値テスト名の記載は「存在・実行・独立参照による妥当性」をそれぞれ区別する。

## 未完了

- [ ] 全repo treeに対する被参照・実装/テスト/Notebookパスの存在確認。
- [ ] 全リンクのアンカー解決（`check_docs_references.py` 実行）。
- [ ] 全174見出しの本文レベルレビュー。
- [ ] 現行CIの実行結果確認と移行後の比較。
- [ ] Phase 1の完了判定。


## 追加確認：重要参照先の実在性（個別取得）

GitHubの作業ブランチ上で、以下の10ファイルを直接取得できた。**ファイルの存在は確認済み**だが、テストの実行成功や記述内容の数学的正当性は未確認。

| Category | Path | Result |
| --- | --- | --- |
| Model numerical test | `tests/models/standard/test_phase7_kronecker_covariance_reference.py` | 存在確認 |
| Classification numerical test | `tests/models/classification/test_phase7_binary_probit_reference.py` | 存在確認 |
| Acquisition numerical test | `tests/acquisition/test_phase8_bald_probability_variance_reference.py` | 存在確認 |
| Optimization numerical test | `tests/optim/trust_region/test_phase9_turbo_geometry_reference.py` | 存在確認 |
| Notebook | `examples/notebooks/28_regression_active_learning.ipynb` | 存在確認 |
| Model registry | `src/robotorchan/models/registry.py` | 存在確認 |
| Acquisition registry | `src/robotorchan/acquisition/registry.py` | 存在確認 |
| Optimizer dispatch | `src/robotorchan/optim/dispatch.py` | 存在確認 |
| Reference checker | `scripts/check_docs_references.py` | 存在確認 |
| Theory chapter | `docs/theory/24_classification.md` | 存在確認 |

### 次のレビューで優先する点

1. `models.md` のfamily別見出しに含まれる **技術的な主張** を章別本文とAppendixに二重取りこぼしなく割り当てる。
2. `verification.md` の各数値テストについて、テスト存在・実行結果・独立参照・検証限界を別列で管理する。
3. `README.md` と各subjectファイルのリンク先を新章へ移す際、既存のナビゲーション契約を維持する。

## 追加の参照先確認（第2バッチ）

既存5ファイルの本文から、特定形式の`src/robotorchan/`、`tests/`、`examples/notebooks/`、`scripts/`、`docs/`の明示パスを抽出した結果、重複除外で**178候補**を識別した。省略パス・波括弧展開・相対リンク・外部からの被参照はこの件数に含まれない。178件すべての存在確認は未完了。

追加で以下6件をGitHub作業ブランチから直接取得し、すべて存在を確認した。

| Path | Status |
| --- | --- |
| `docs/models/classification.md` | 取得成功 |
| `docs/optimization/epig.md` | 取得成功 |
| `docs/theory/acquisition/classification-active-learning.md` | 取得成功 |
| `scripts/audit_model_api.py` | 取得成功 |
| `scripts/audit_acquisition_api.py` | 取得成功 |
| `scripts/generate_model_coverage.py` | 取得成功 |

前回の個別取得10件と合わせて、**個別取得による確認は計16件**。実行結果や全参照の網羅性は未確認。以後は新規監査ファイルを増やさず、同じ台帳を更新する。

## 数値検証テスト参照の追加確認（第3バッチ）

`verification.md`で扱う数値検証のうち、下記8テストをGitHubの作業ブランチから直接取得し、すべて存在を確認した。これは**テストファイルの存在確認**であり、実行成功や参照式の独立性を検証したものではない。

| Test | Status |
| --- | --- |
| `tests/acquisition/test_phase8_analytic_bo_gaussian_reference.py` | 取得成功 |
| `tests/acquisition/test_phase8_classification_al_reference.py` | 取得成功 |
| `tests/acquisition/test_phase8_mc_qucb_reference.py` | 取得成功 |
| `tests/acquisition/test_phase8_mc_simple_regret_reference.py` | 取得成功 |
| `tests/acquisition/test_phase8_regression_al_formula_reference.py` | 取得成功 |
| `tests/optim/backends/test_phase9_sampling_optimizer_reference.py` | 取得成功 |
| `tests/optim/trust_region/test_phase9_turbo_state_reference.py` | 取得成功 |
| `tests/optim/constraints/test_phase9_constraint_violation_reference.py` | 取得成功 |

これまでの個別取得16件と合わせ、**計24件の個別参照先の存在確認**を記録した。178件の抽出候補の網羅確認は未完了。次のゲートでは、存在確認の反復だけでなく、`verification.md`に書かれた数値的な検証条件・未検証範囲を章別ガイドとAppendixに対応付ける。

## 実装・テスト・Notebook・理論参照の直接存在確認（第4バッチ）

作業ブランチから以下の12ファイルの冒頭をGitHub connectorで直接取得し、全件の**パス存在とblob SHA取得**を確認した。テスト実行・Notebook実行・JSON生成・リンク解決・数式検証を意味しない。

| 種別 | 参照先 | 判定 |
| --- | --- | --- |
| test | `tests/acquisition/test_epig.py` | 直接取得成功 |
| test | `tests/acquisition/test_phase5_native_integration.py` | 直接取得成功 |
| test | `tests/optim/one_shot/test_mixed.py` | 直接取得成功 |
| test | `tests/optim/embedding/test_baxus_reference_schedule.py` | 直接取得成功 |
| test | `tests/optim/trust_region/test_turbo_strategy.py` | 直接取得成功 |
| test | `tests/models/standard/test_mixed_single_task_gp.py` | 直接取得成功 |
| test | `tests/models/classification/robust/test_label_noise.py` | 直接取得成功 |
| artifact | `docs/model_coverage.json` | 直接取得成功 |
| checker | `scripts/check_theory_implementation_mapping.py` | 直接取得成功 |
| notebook | `examples/notebooks/28_regression_active_learning.ipynb` | 直接取得成功 |
| theory | `docs/theory/acquisition/09_active_learning.md` | 直接取得成功 |
| theory | `docs/theory/optimization/09_high_dimensional.md` | 直接取得成功 |

以前の24件との重複はNotebook1件であるため、**直接取得で確認したユニークなパスは35件**（今回の新規11件）。178件の静的抽出候補に含まれるか否かは個別の集合突合せ未実施であり、単純に35/178を厳密な検証率とは呼ばない。

### ガイド移行時の証拠レベル

- **Path exists**：GitHubで取得成功（今回確認した範囲）。
- **Test asserts**：テスト本文を読み、対象のproperty・許容誤差・否定ケースを確認した状態。今回未実施。
- **Test passes**：対象commitの実行結果が確認できた状態。今回未実施。
- **Independent reference**：実装と独立した式・基準との比較が成立するとレビューできた状態。今回未実施。
- **Notebook runnable**：指定環境で実行成功した状態。今回未実施。

今後、各技術ガイドの「検証」節は少なくとも`source`、`test`、`notebook`、`verification level`、`limitations`を分離する。テストファイル名を挙げるだけで数値妥当性を主張しない。旧5 Mappingファイルは不変。

## 数値参照テストのassertion監査（第1バッチ）

下記6ファイルは存在確認に加えて**テスト本文を直接読み、assertion・期待値の作り方・許容誤差**を確認した。**実行はしていない**。ここでの`independent`は「対象APIの出力と別の式・計算経路を比較している」という限定的な意味で、実装から完全に独立した第三者検証の保証ではない。

| Test | 実際に検証する性質と期待値 | tolerance | 残る限界・ガイド接続 |
| --- | --- | --- | --- |
| `tests/models/standard/test_phase7_kronecker_covariance_reference.py` | input×task共分散の明示indexing、行列shape・対称性・固有値によるPSD、task順序交換時の対応 | `atol=rtol=1e-10`、最小固有値`>=-1e-10` | 特定小規模構成のkernel行列。大規模Kronecker計算・posterior性能の証明ではない。理論`07_multitask_multioutput.md` |
| `tests/models/classification/test_phase7_binary_probit_reference.py` | `f~N(mu,var)`で`E[Phi(f)]=Phi(mu/sqrt(1+var))`を用いてpositive class probabilityを直接算出、class complement・和1・[0,1] | 主要比較`atol=rtol=1e-10` | 未学習の単一binary model、3候補。予測較正・ELBO収束の証明ではない。理論`24_classification.md` |
| `tests/acquisition/test_phase8_bald_probability_variance_reference.py` | `sample_class_probabilities`を固定64標本に差替え、ProbabilityVarianceの標本分散とBALDの`H(E[p])-E[H(p)]`を直接計算 | `atol=rtol=1e-10` | **固定sampleに対する計算式テスト**。実posterior samplingの較正・Monte Carlo誤差を検証しない。理論`acquisition/09_active_learning.md` |
| `tests/acquisition/test_phase8_analytic_bo_gaussian_reference.py` | BoTorch native LogEI/LogPI/UCBとGaussian CDF/PDF閉形式を比較。`UCB=mu+sqrt(beta)*sigma` | `atol=rtol=1e-8` | 3候補・単目的`q=1`、`best_f=0`、`beta=0.4`。log-domainの極端なtail安定性やrobotorchan独自acquisitionを検証しない。理論`acquisition/02_improvement.md`・`03_confidence_bound.md` |
| `tests/optim/trust_region/test_phase9_turbo_geometry_reference.py` | ARD正規化のtrust region boundsを直接式と比較、public bounds内clamp・境界接触をassert | `atol=rtol=1e-12` | geometryのみ。TuRBOのsuccess/failure state・restart・最適化性能はこのテストの範囲外。理論`optimization/10_trust_region.md` |
| `tests/optim/constraints/test_phase9_constraint_violation_reference.py` | q=2のintra/interpoint線形・等式許容誤差・非線形違反量を手計算で加算、`feasible <=> violation==0`、feasibility-first順位をassert | 違反量`atol=rtol=1e-12`、等式許容`0.05` | 制約**評価・順位付け**のテスト。optimizerが必ず可行点を発見する保証ではない。理論`optimization/05_constraints.md` |

### 数値テストをガイドへ転記する際の原則

- テスト対象の**数式・対象次元・構成・許容誤差・期待値算出法・未検証範囲**を記述し、Phase番号を章見出しに残さない。
- BoTorch nativeの数値一致とrobotorchan local compositionの数値一致を混同しない。
- 固定sampleによるALスコア検証はsample generatorやclassification calibrationの証拠にしない。
- `candidate_constraint_violation`の正しさを、candidate optimizationの可行性保証や収束証明へ拡張しない。
- `Test asserts`までレビュー済み、`Test passes`と実行環境での独立再現は**未確認**。CIを実行したとは記載しない。

## Acquisition / Optimization数値参照テストのassertion監査（第2バッチ）

以下の6ファイルの**テスト本文**を読み、入力形状、参照式、実際のassertion、未検証範囲を確認した。**テスト・CIは実行していない**。

| Test | 参照計算とassertion | 数値条件 | 対応する理論・重要な限界 |
| --- | --- | --- | --- |
| `tests/acquisition/test_phase8_classification_al_reference.py` | PredictiveEntropyは`-sum(p log p)`、MarginUncertaintyは`1-p_top1+p_top2`、LatentStraddleは`beta*sqrt(v_latent)-abs(mu_latent)`。3候補で形状・entropy/marginの範囲もassert | `atol=rtol=1e-10`、`beta=1.5` | `acquisition/09_active_learning.md`と`10_level_set.md`、`24_classification.md`。**確率空間とlatent空間の異なるscore**。予測較正は検証しない |
| `tests/acquisition/test_phase8_mc_qucb_reference.py` | BoTorch native qUCBと、同一Sobol samplerの64 drawから計算する`E[max_q(mean_samples+sqrt(beta*pi/2)*abs(sample-mean_samples))]`を比較 | `atol=rtol=1e-8`、`beta=0.3`、`q=2`、seed=23 | `acquisition/03_confidence_bound.md`、`04_batch_noisy.md`。**sample mean**を用いる参照。analytic UCBの単純なq点選択とは異なる。sampler独立性は検証しない |
| `tests/acquisition/test_phase8_mc_simple_regret_reference.py` | BoTorch native qSimpleRegretと`mean_samples(max_q(f_sample))`を比較。同一samplerのbase samplesを再利用 | `atol=rtol=1e-8`、64 draw、`q=2`、seed=17 | `acquisition/04_batch_noisy.md`。**sample-wise q最大値**。simple regretの最終最適化性能やMC収束率は検証しない |
| `tests/acquisition/test_phase8_regression_al_formula_reference.py` | PosteriorVariance=`v`、PosteriorStd=`sqrt(v)`、Straddle=`beta*std-abs(mu-target)`、BoundaryVariance=`v*exp(-0.5*((mu-target)/max(std,eps))**2)` | `atol=rtol=1e-12`、`target=0.35`、`beta=1.7`、3候補 | `acquisition/09_active_learning.md`、`10_level_set.md`。固定パラメータ・q=1のscore式であり、境界推定性能の保証ではない |
| `tests/optim/backends/test_phase9_sampling_optimizer_reference.py` | q=2負二次目的に対し、同seedで作成した128点のRandom/Sobol候補集合の**集合内argmax**とdispatcher結果・bounds・scoreを比較 | candidate`atol=rtol=0`、score`1e-12`、seed=29 | `optimization/06_derivative_free.md`、`08_batch_one_shot.md`。**全連続空間の真の最適点**を求めた検証ではなく、同一候補集合での選択一致 |
| `tests/optim/trust_region/test_phase9_turbo_state_reference.py` | 3成功でregion lengthを倍増（maxでcap）、2失敗で半減・min割れでrestart、restart後のbest保持、denoised decision valueとraw observed bestの区別をassert | `success_tolerance=3`、`failure_tolerance=2`、length閾値と`math.isclose` | `optimization/10_trust_region.md`。TuRBO **state transition**を検証。実問題での改善率や並列async条件の性能は検証しない |

### 既存6テストとの横断整理

- これまでの**assertion本文レビューは計12ファイル**。過去の35パス存在確認とは異なる集計であり、混同しない。
- acquisitionの`qUCB`、`qSimpleRegret`はBoTorch nativeの式・MC aggregationに対する参照。local `BoundaryVariance`、`Straddle`、分類AL scoreのテストとは実装所有者が異なる。
- optimizationの`sampling optimizer`は同じ生成poolのargmax、`TuRBO state`は状態遷移、前回の`TuRBO geometry`は領域形状をそれぞれ検証する。3つを「TuRBO全体のE2E性能が検証済み」とまとめない。
- 次段階でverification guideへ移すときは、**式、参照値の生成方法、`q`、dtype、sample数、seed、tolerance、未検証範囲**を同じ表で管理する。

## 178候補のカテゴリ別突合せ：文書・スクリプト・Notebook（2026-10-26）

元の5 Mappingファイルから同一正規表現で**ユニーク178候補**を再抽出し、カテゴリを分離した：`src/robotorchan/` 24件、`tests/` 122件、`examples/notebooks/` 1件、`scripts/` 4件、`docs/` 27件（合計178）。

今回は`docs/` **27件**、`scripts/` **4件**、`examples/notebooks/` **1件**の計**32件を作業ブランチから直接取得し、32/32件のパス存在を確認**した。これはリポジトリ全treeの列挙ではなく、**原文から抽出した32候補の個別照合**である。前回までの存在確認と重複するため、過去の35件に32件を加算して累計ユニーク数を算出してはならない。

| 対象 | 候補数 | 直接取得成功 | 未確認 |
| --- | ---: | ---: | ---: |
| `docs/` | 27 | 27 | 0 |
| `scripts/` | 4 | 4 | 0 |
| `examples/notebooks/` | 1 | 1 | 0 |
| `src/robotorchan/` | 24 | 今回は未照合 | 24 |
| `tests/` | 122 | 今回は未照合 | 122 |

**検証の意味**：パスが現ブランチに存在することを確認しただけで、Markdownのアンカー・相対リンク・Notebook実行・スクリプト実行・テスト成功を確認していない。残る146候補（実装24、テスト122）は次のバッチで直接照合し、失敗したパスは`missing`と断定する前に表記ゆれ・省略形を確認する。全件完了後、元の35件の存在確認記録と集合突合せしてユニーク件数を再計算する。
