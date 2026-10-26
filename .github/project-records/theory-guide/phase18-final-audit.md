# 日本語理論・実装ガイド：最終監査と条件付き完了判定

> 監査対象：固定18 Phase・48章の技術文書整備。対象PRの先頭は #1466（最終HEAD `e5b16e05d63300d8925df0d31e2d528f807a3afe`）。この文書は**現時点の証拠範囲を明記する判定**であり、確認していない数値実験を成功と記載しない。

## 目的と判定

**判定：構造・CIレベルは達成。技術内容の完全な移行と実行の最終完了判定は保留。**

固定48章の理論対応と旧Mapping5ファイルの存在は、`scripts/check_theory_guide_quality.py`と`tests/docs/test_theory_guide_quality.py`が検査する。PR #1466最終HEADの**CIとDocumentation linksがともにsuccess**であることをGitHubワークフローで確認した。元Mappingの完全な置換、数式の全property検証、全コード例の実行を保証するものではない。

## 最終ゲートごとの証拠

| 判定対象 | 観察した根拠 | 判定 |
| --- | --- | --- |
| 理論に対応する48章 | Main24 + acquisition12 + optimization12。チェックscriptの固定basename照合 | 達成（章の存在・見出し・theoryリンクの構造） |
| 旧Mapping保全 | `README.md`、`models.md`、`acquisition.md`、`optimization.md`、`verification.md`の存在をCIで確認 | 達成（存在チェック）；全区間の内容比較ではない |
| 追加利用導線 | `guide/walkthrough/README.md`、`guide/verification/README.md`、`guide/13_model_selection.md`などを作成 | 配置済み。実例の実行は個別検証が必要 |
| Markdownリンクとアンカー | `.github/workflows/documentation-links.yml`で既存docsリンク・Notebook・referencesと新規48章checkerを実行 | PR #1466 HEADで成功 |
| パッケージと全体CI | GitHub CI workflow | PR #1466 HEADで成功。CI範囲外の実験・環境を含まない |
| 数学的検証 | `guide/verification/`を作成し旧`verification.md`のproperty別参照を保持 | 検証索引の配置は達成。全理論の独立参照完了ではない |
| 旧Mappingの区間移行 | [Phase1保全監査](phase1-preservation-audit.md)では174見出し候補、H4/表/数式の区間照合の未完了を報告 | **未確認・要照合** |
| Source/Test/Notebook対応の全件 | 旧MappingおよびPhase1の移行記録に部分的な静的監査 | **未確認**（全行の実在とassertion照合未証明） |
| 学習本文からPhase/PR/CI履歴を除去 | 主な新規ガイド本文では分離方針を採用 | **未完了**：旧Mappingに履歴が残る。原文保護のため未削除 |
| コード例の実行 | 一部の既存Notebookや統合scriptはCIの対象。新規ガイドの全コード例単体実行を確認した記録なし | **未確認** |
| 重複・不要ファイル | 旧Mappingを移行元として保持する設計 | **未完了**：全区間の照合まで整理・削除不可 |

## 学習用ガイドとして必要な残検証

- 原文の全見出し・表・数式・negative contractを移行台帳の区間へ対応付け、各行に`primary_destination`、必要なら`secondary_destinations`、具体的なassertion／参照先、`review_status`を記録する。現存するのは構造的対応であり、内容の完全移行と同義ではない。
- Model / Acquisition / Optimizerの各章にある具体的なAPI名・引数・posterior semanticsを実際のsrcおよびテストから再照合する。BoTorchへの`Delegated`とrobotorchan側の`Implemented/Extended`は推測で埋めない。
- 実行例は新規章ごとにparse/import/最小実行を検査し、CPU/float64/seed、依存環境、成功・失敗記録を保存する。CIが緑でも非実行のコードブロックまで実証済みとは扱わない。
- 履歴を学習本文から取り除くのは**技術内容を再配置した後**とする。旧原文を単純に短縮・削除するのは情報喪失になるため禁止する。
- `guide/README.md`の「予定」表現は48章作成の実態と齟齬があったため、索引表現を今回修正した。章内に残る他の時間依存表現は別途監査する。

## 今回の完了条件の扱い

固定計画には新しいPhaseを追加しない。上記の保留事項は**Phase18の最終監査ゲートの未達**として記録する。構造・リンク・CIの達成を維持しながら、意味論的移行検査・コード例実行・文書重複の解消を最終ゲートとして再判定する。

関連：[48章ガイド索引](../../../docs/development/theory-implementation/guide/README.md)、[元Mapping](../../../docs/development/theory-implementation/README.md)、[検証索引](../../../docs/development/theory-implementation/guide/verification/README.md)、[移行先設計](phase1-migration-routing.md)。

## PR #1467 の実行ゲート確認（2026-10-26）

Phase18の文書修正HEAD `07b459f98555ea5d147040cf46950934c4a0f1e1` に対し、次をGitHub Actionsで確認した。

| Workflow | Run ID | 結果 | 証拠の意味 |
| --- | --- | --- | --- |
| Documentation links | `40296208635` | **success** | Markdownリンク・参照・48章構造など、ワークフロー内検査の成功 |
| CI | `40296208646` | **success** | 現在のCIで対象としたlint / tests / notebook jobs等の成功 |

PR #1467はこの記録時点で**Open（未マージ）**である。したがって、完了をmainに反映済みとは扱わない。CI緑は変更内容をマージ可能とする必要条件の一つであり、174見出しの意味論的移行確認、H4以下の数式・negative contract保全、すべての新規コード例の実行完了を示さない。

[既存の見出し移行台帳](phase1-heading-preservation-ledger.md)は**174見出しの仮割当**で、各レコードが`planned (review required)`とされている。これを目視・機械照合したという記録なしに`verified`へ更新しない。

### Phase18で追加のゲートを閉じる際の順序

1. **台帳の内容検収：** 見出し区間の本文、H4・数式・表・参照・exception・未対応情報を原文と照合する。primaryとsecondary移行先を明示し、単にリンクを置いただけでは`migrated`にしない。
2. **APIとテスト：** 各ガイドの使用例について、コード上の公開export・constructor・実行を検査し、別途numerical propertyのassertion根拠を確認する。
3. **監査履歴の整理：** 学習用本文に歴史的なPhase/PR/CIが混入していないかを確認し、必要な技術説明を残したままauditに分離する。
4. **最終再判定：** CIとリンクが緑であるだけでなく、上記の台帳・実行証拠・情報非喪失が説明できた時点でのみPhase18全体を`complete`とする。

これらは固定Phase18の**未解決ゲート**であり、新しいPhaseの追加ではない。

## 原文区間の実照合：Models Registry（部分達成）

[旧Model Mapping](../../../docs/development/theory-implementation/models.md)の行18–91を直接取得し、以下を照合した。

| 原文区間 | 調査した事実 | 対応先 | 移行判定 |
| --- | --- | --- | --- |
| 18–31「Registry boundaries」 | 回帰registry監査時点142件／二値分類29 ID、異種modelはcoverage外、分類のhelper exportとmodel IDの区別 | [Model registry appendix](../../../docs/development/theory-implementation/guide/appendix/model-registry-and-public-api.md) | **partial, review required**：意味論は対応したが現在mainの独立カウントは未実施 |
| 32–51「Regression model inventory by existing guide」 | 11 familyの集計（合計142件）と各モデル利用ガイドへの参照 | 同Appendix | **partial, review required**：全142 model名と個別notebook/theoryの全行照合は未実施 |
| 52–91「Binary classification registry inventory」 | 分類IDとpublic classの名前空間の差。binary.kronecker_multitaskはExact Kroneckerを意味しない | 同Appendix・[Classification](../../../docs/development/theory-implementation/guide/24_classification.md) | **partial, review required**：全29 IDのassertion証拠の移管・全件検収は未実施 |

確認した本文は新しいAppendixに概念別にまとめたが、**原文のinventory全件をコピー・移行したものではない**。既存の174見出し台帳上の`planned (review required)`を`verified`へ変更せず、重複情報があるという理由で旧Mappingを削除しない。原文のsource blob SHA、詳細なテスト証拠、各レコードの完全な再配置は引き続き未達である。

## Acquisition Active Learningの実装照合（部分達成）

旧`acquisition.md`の「Distinct implementation paths」「Phase 5 — regression active-learning acquisition mappings」「Phase 5 — classification AL and Gaussian EPIG」を読んで、現行の`src/robotorchan/acquisition/active_learning/variance.py`、`classification.py`および`registry.py`のソースと照合した。[ALガイド](../../../docs/development/theory-implementation/guide/acquisition/09_active_learning.md)へ具体的な数式・クラス・posterior/predict_probaの違い・q=1・拒否条件を反映した。

**確認した範囲：** `PosteriorVariance`、`PosteriorStd`、`PredictiveEntropy`、`MarginUncertainty`、`ProbabilityVariance`、`BALD`、`LatentStraddle`の責務、ならびにregression ALのstructured/ensemble guardとclassification model interface。数値テストをこの作業で再実行したわけではなく、旧原文の全行・全entryの移行を検収したわけでもない。原文見出し台帳は依然`review required`のまま保全する。

## Constrained AcquisitionとOptimizer接続の部分照合

旧`acquisition.md`の「Distinct implementation paths」「Acquisition/optimizer interaction」の技術的主張に対し、`src/robotorchan/acquisition/composition.py`および`optimizer_compatibility.py`のソースを直接照合した。[制約付きAcquisitionガイド](../../../docs/development/theory-implementation/guide/acquisition/08_constraints.md)にsemantic binding、sample shape、分類可行性の独立性近似、candidate/outcome制約の区別、one-shot/mixed静的互換性の限界を追加した。

**状態：partial / review required。** 本作業では全constrained helperの数値的同値性、非線形候補制約の各backendでの実行、joint feasibility相関モデルは独立テストしていない。旧Mappingの原区間の数式・例外・テストパスを全件照合して`verified`にする作業は保留する。

## Optimizer Candidate Constraints：原文・ソース・ガイドの部分照合

旧`optimization.md`の「Baseline and separation of responsibilities」「Named optimization dispatcher」「Candidate-space constraints versus outcome constraints」の区間を、`src/robotorchan/optim/constraints/contracts.py`、`validation.py`および`dispatch.py`と比較した。確認対象は11 named backends、`CandidateConstraints`の線形・非線形tupleと符号、intrapoint/interpointの入力shape、`sequential=True`の拒否条件、非線形初期点・公開座標/埋め込み座標の違いである。技術ガイド[Candidate constraints](../../../docs/development/theory-implementation/guide/optimization/05_constraints.md)へ具体的な実装契約を反映した。

**移行判定：partial / review required。** この調査は制約関連のソース照合であり、全backendのconstraint enforcementの数値実行や、旧`optimization.md`の残りのstrategy/one-shot/NSGA-II区間の意味論的照合を完了したものではない。旧Mappingと174見出し台帳は変更しない。

## TuRBOの状態更新：実装と旧Mappingの部分照合

旧`optimization.md`の「Stateful and coordinate-changing algorithms」「Inspected optimizer and strategy assertion evidence」を読み、`src/robotorchan/optim/trust_region/turbo.py`の`TuRBOState`、`update_turbo_state`、`restart_turbo_state`、`generate_turbo_restart_center`の実装本文を直接照合した。実装で定義されたbatch observations/state_values区別、relative improvement閾値、success/failure counters、region lengthの拡縮、restart、ARD重みの幾何平均正規化を[Trust Regionガイド](../../../docs/development/theory-implementation/guide/optimization/10_trust_region.md)へ詳記した。

**判定：partial / review required。** 数式・sourceは照合したが、今回新たなテスト実行、全TuRBO optimizer/Thompson/pending/mixed pathのassertion-level完全照合、regret評価は行っていない。旧原文と174見出し台帳を保全する。

## Mixed one-shot：augmented batch契約のソース照合（部分達成）

旧`optimization.md`の「Specialized optimization paths: inspected source and assertions」と、`src/robotorchan/optim/one_shot/{initialization,mixed}.py`を直接比較した。[Batch / One-shotガイド](../../../docs/development/theory-implementation/guide/optimization/08_batch_one_shot.md)へ`get_augmented_q_batch_size(q)`と実候補qの区別、`extract_candidates`、one-shot初期点のshape、mixed one-shotの行別カテゴリ割当ての列挙（組合せ数`m ** augmented_q`）、`q=1`制限、`max_assignments`上限、BoTorchへの連続部分問題委譲を追記した。旧Mapping記載の`tests/optim/one_shot/test_mixed.py`はassertion追跡先として維持した。

**判定：partial / review required。** 本作業はsrc記述を確認した文書照合であり、独立の混合KG数値検証や全候補制約の組合せ、全one-shotアルゴリズムの性能検証ではない。固定174見出し台帳は保全する。

## NSGA-II：vector objectiveとscalar acquisitionの境界を照合

旧`optimization.md`の「Specialized optimization paths: inspected source and assertions」、`src/robotorchan/optim/backends/nsga2.py`および`tests/optim/backends/test_nsga2_backend.py`の実装・テスト本文を確認した。[Evolutionaryガイド](../../../docs/development/theory-implementation/guide/optimization/07_evolutionary.md)に、NSGA-IIが11種のnamed acquisition dispatcherに含まれないこと、`[population,d] → [population,m]`（`m>=2`、全目的最大化）の契約、nondominated ranking／crowding／選択・交叉・突然変異、返却される近似Pareto集合とテストの限界を明記した。

**判定：partial / review required。** この照合はソースとテストの静的確認であり、今回独立にNSGA-IIを実行した結果やhypervolume収束の証明ではない。旧Mappingは引き続き保全する。

## HeSBOとLatentSpaceStrategy：座標変換の実装照合（部分達成）

旧`optimization.md`の「HeSBO, latent-space reconstruction and backend constraint semantics」を参照し、`src/robotorchan/optim/embedding/hesbo.py`、`src/robotorchan/optim/latent/strategy.py`を直接読んで[High-dimensional optimizationガイド](../../../docs/development/theory-implementation/guide/optimization/09_high_dimensional.md)を書き直した。HeSBOの疎hash・符号・正規化座標のclamping、LatentSpaceStrategyの学習済み再構成と元空間のacquisition評価、両者の`reject_unmapped_candidate_constraints`を照合した。

**判定：partial / review required。** sourceの契約を説明し旧Mapping中の代表テスト名を参照したが、今回テスト本文全件・数値実行・REMBO/ALEBO/BAxUS全経路の検収は完了していない。座標復元のboundsと一般の非線形candidate feasibilityを同一視しない。旧174見出し台帳は依然検収待ちである。

## REMBO・ALEBO・BAxUS：embeddingと状態更新の部分照合

旧`optimization.md`の「Coordinate-changing high-dimensional strategies」を参照し、`src/robotorchan/optim/embedding/{rembo,alebo,baxus}.py`を直接確認した。[High-dimensional optimization guide](../../../docs/development/theory-implementation/guide/optimization/09_high_dimensional.md)にREMBOのGaussian投影・clipping、ALEBOのpseudoinverseに基づく埋め込みpolytopeと明示的なfeasibility判定、BAxUSの評価予算・部分空間次元・成功失敗counter・length更新・ARD重みを追加した。BAxUSの相対改善閾値とTuRBOの閾値の定義差も記録した。

**判定：partial / review required。** この更新はソース本文を読んだ監査であり、関連embeddingの数値テストやベンチマークの独立再実行、174見出し全体の内容検収は未完了。元Mappingを削除しない。

## Verification：Phase 8のMC数値参照（追加の個別検収）

旧`verification.md`の「Monte Carlo qSimpleRegret sample reference」「Monte Carlo qUCB sample reference」と、`tests/acquisition/test_phase8_mc_simple_regret_reference.py`および`tests/acquisition/test_phase8_mc_qucb_reference.py`のassertion本文を直接確認し、[Acquisition検証索引](../../../docs/development/theory-implementation/guide/verification/acquisition.md)へ計算順序・shape・seed・許容誤差を移した。

| Property | 独立参照と一致の範囲 | 定数・条件 | 判定 |
| --- | --- | --- | --- |
| `qSimpleRegret` MC aggregation | `mean_s max_q samples[s,q]`とBoTorch-native出力 | 64 Sobol samples、seed 17、2 batches×q=2、float64、atol=rtol=1e-8 | source/test assertion inspected。個別propertyだけ |
| `qUpperConfidenceBound` MC aggregation | `mean_s max_q (sample_mean+sqrt(beta*pi/2)*abs(sample-sample_mean))` | 64 Sobol samples、seed 23、beta=0.3、2 batches×q=2、float64、atol=rtol=1e-8 | source/test assertion inspected。個別propertyだけ |

元のテストが用いるbase samplesを参照式でも再利用するため、これらはsample aggregationの独立記述との一致であり、MC収束やGP conditioningの独立検証ではない。Phase 18の全体完了判定は変えず、元の174見出し台帳も一括`verified`としない。

## Phase 18 Verification：optimizerの実行assertion照合（部分達成）

`tests/optim/integration/test_nonlinear_candidate_constraints_e2e.py`と`tests/optim/trust_region/test_turbo_strategy.py`のテスト本文を直接確認し、[Optimizer verification guide](../../../docs/development/theory-implementation/guide/verification/optimization.md)にcandidate feasibility（q=1、linear/nonlinear/fixed、q=2 interpoint、qLogNParEGO）およびTuRBOの閾値・length・restart状態について具体的なassertionと許容誤差を追記した。既存のOptimization MappingとVerification Mappingは引き続き原文の証拠を保持する。

**判定：partial / review required。** このコミットはソース・テストassertionの静的照合であり、テストの新規独立実行や残余の174見出し全件のmeaning-level migrationを完了したことを意味しない。

## Phase 18 Verification：Kronecker prior / PCA reducerのassertion照合

`tests/models/standard/test_phase7_kronecker_covariance_reference.py`と`tests/models/high_dimensional/reduced/test_phase7_pca_projection_reference.py`を直接読み、[モデル検証ガイド](../../../docs/development/theory-implementation/guide/verification/models.md)に、3入力×2タスクのKronecker prior covarianceの添字・PSD・タスク置換（atol/rtol=1e-10）と、6×3→2 PCAのcovariance eigenspectrum、basis直交性、投影・残差の不変条件を追加した。

**判定：partial / review required。** 両テストのassertionを照合したにとどまり、この作業で数値実験を独立に再実行したものではない。他のモデル検証やposterior conditioningの完全性を一般化しない。元の174見出し台帳とMappingは保全する。

## Phase 18 Verification：Multi-fidelity / Heteroskedastic assertion照合

`tests/models/standard/test_phase7_multifidelity_kernel_reference.py`と`tests/models/robust/test_phase7_heteroskedastic_uncertainty_reference.py`の本文を読んで、[Models verification guide](../../../docs/development/theory-implementation/guide/verification/models.md)に数式・入力条件・許容誤差を追記した。前者はfidelity-only downsampling kernelのhigh-fidelity offset boundary、rank-one residual、PSDをfloat64/1e-10で照合した。後者は固定log-noiseからの0.01/0.09分散（1e-12）、latent varianceとの分離、手動の分散加法恒等式を確認した。

**判定：partial / review required。** テスト本文とのassertion照合であり、今回の独立再実行は行っていない。特にHeteroskedasticの`posterior(observation_noise=True)`で位置依存noiseが自動合成されること、全MF posteriorの同値性は証明されていない。旧Mappingの全区間移行は未完了。

## Phase 18 Verification：Binary probit / Gaussian posterior adapterのassertion照合

`tests/models/classification/test_phase7_binary_probit_reference.py`および`tests/models/non_gp/test_phase7_gaussian_posterior_reference.py`を直接確認し、[Models verification guide](../../../docs/development/theory-implementation/guide/verification/models.md)へ具体的な数式、入力shape、数値tolとテスト証拠の限界を追記した。前者は`E[Phi(f)] = Phi(mu/sqrt(1+variance))`、3候補の2クラス確率・正規化（1e-10）を検査。後者は固定Gaussian posteriorのbase-sample affine reparameterization（shape `[2,2,2,1]`、1e-12）とmean gradient・zero varianceを検査する。

**判定：partial / review required。** Classifierの学習・較正や外部estimatorの予測精度を証明したものではなく、本作業で独立数値テストの再実行は行っていない。旧Mappingおよび174見出し台帳は引き続き保全する。

## Phase 18 Verification：Reduced/Mixed PCAとForestのassertion照合

`tests/models/high_dimensional/reduced/test_phase7_posterior_coordinate_reference.py`、`test_phase7_mixed_reduced_categorical_reference.py`および`tests/models/non_gp/test_phase7_forest_estimator_reference.py`のテスト本文を直接確認し、[Models verification guide](../../../docs/development/theory-implementation/guide/verification/models.md)に入力shape・数値tol・negative contractを追記した。PCAGPは固定reducerの元空間/縮約空間posterior平均・分散一致（1e-10）、MixedPCAGPは`reduced_cat_dims == [2]`とカテゴリID保持・posterior同値（1e-10）、RandomForestSurrogateは7 treeの各予測とmean/不偏varianceの一致（1e-12）および`observation_noise=True`拒否を照合した。

**判定：partial / review required。** ここで得られたのはテストのassertionと数式の対応であり、今回の独立再実行や全モデル族の性能・uncertainty calibrationを示すものではない。旧Mappingの174見出し全件検収は未完了。

## Verification横断：検証証拠のレベルと履歴上の完了判定

[Verification guide index](../../../docs/development/theory-implementation/guide/verification/README.md)に、Models/Acquisition/Optimizationのassertion照合対象と、そのproperty別の未証明範囲を一つの表に整理した。旧`verification.md`内の「Phase 7 remains open」は当時のexit auditであり、後続の「Phase 7 final exit reconciliation」は**限定した代表的propertyの完了**を示すという時系列差を明記した。どちらも原文を改変しない。

検証索引にテストパスがあること、テスト本文と数式を静的に照合したこと、GitHub CIが実行成功したこと、独立の数理的propertyが成立することを**別の証拠**として扱う。例えば既存のHeteroskedastic手動分散恒等式は、位置依存の`posterior(observation_noise=True)`自動合成を保証しない。**Phase18の文書移行・174見出し意味論照合は引き続きreview required**であり、各機能検証の過去のexit結果で置き換えない。

## Phase 18：Verificationの見出し単位移行照合（部分照合）

[Phase 1の174見出し台帳](phase1-heading-preservation-ledger.md)の末尾に、歴史的な`verification.md`行区間と`guide/verification/{models,acquisition,limitations,README}.md`への受け皿の対応を追加した。Kronecker、multi-fidelity、PCA、binary probit、Gaussian adapter、heteroskedastic、MixedPCAGP、Forest、MC qSimpleRegret/qUCB、過去の代表property限定exit判定を、確認したassertionごとに**部分照合・review required**として明示した。

**未達：** 対象31見出しを含む174見出し全件の本文・表・数式・negative contract・根拠リンクの行単位検収。旧台帳の`planned (review required)`を機械的に`verified`に変更せず、旧Mapping5ファイルも保全する。Phase1の行番号は固定スナップショット由来であり現行行番号と混同しない。

## Phase 18：Phase 9 独立参照テストの部分照合

`tests/optim/backends/test_phase9_sampling_optimizer_reference.py`、`tests/optim/trust_region/test_phase9_turbo_geometry_reference.py`、`tests/optim/constraints/test_phase9_constraint_violation_reference.py`を直接読み、[Optimizer verification guide](../../../docs/development/theory-implementation/guide/verification/optimization.md)にrandom/sobol pool内argmax（seed=29、128 sample、q=2）、TuRBO ARD幾何平均正規化とboundary clipping、CandidateConstraintsの各種残差（等式許容差0.05）とfeasibility-first順位の具体的数値assertionを記載した。いずれも出力の意味・数値tolと未検証のglobal optimum/一般可行性を区別した。[Phase1 heading ledger](phase1-heading-preservation-ledger.md)に、Verificationの固定行番号408–421、436–448、449–464を部分照合として追記した。

**判定：partial / review required。** この作業は既存テスト本文の静的照合であり、独立再実行は行っていない。原文の残区間および174見出し全体のsemantic migration gateは未達。旧Mappingを保全する。

## Phase 18：Phase 9 state / PyTorch negative contractの部分照合

`tests/optim/trust_region/test_phase9_turbo_state_reference.py`と`tests/optim/backends/test_phase9_torch_constraint_contract.py`を直接確認した。[Optimizer verification guide](../../../docs/development/theory-implementation/guide/verification/optimization.md)へTuRBOの成功3回・失敗2回・restart・`state_values`とraw observationsの差、およびAdam/AdamW/SGDがlinear/equality/nonlinear candidate constraintsを9組合せで明示拒否するnegative contractを追記した。固定[見出し台帳](phase1-heading-preservation-ledger.md)では`verification.md:422–435`と`:465–475`の受け皿を部分照合とした。

**判定：partial / review required。** 既存テストの静的assertion照合であり、独立再実行・全見出しの完全移行は未達。元Mappingと全174行の`planned (review required)`は保全する。

## Phase 18：Acquisition Phase 8の独立参照assertion追加照合

`tests/acquisition/test_phase8_regression_al_formula_reference.py`、`tests/acquisition/test_phase8_classification_al_reference.py`、`tests/acquisition/test_phase8_analytic_bo_gaussian_reference.py`のテスト本文を読み、[Acquisition verification guide](../../../docs/development/theory-implementation/guide/verification/acquisition.md)へ回帰AL 4式（1e-12）、分類AL 3式（1e-10）、Gaussian log-EI/log-PI/UCB（1e-8）の入力条件と数式および限界を追記した。[174見出し台帳](phase1-heading-preservation-ledger.md)に旧`verification.md:292–306`、`:307–319`、`:335–350`を部分照合として対応付けた。

**判定：partial / review required。** 固定posteriorを入力としたacquisition算術のテストassertion静的確認であり、GP posteriorの独立検証、分類較正、全獲得関数の数値正当性は主張しない。今回のテスト独立再実行と旧Mappingの174見出し完全検収は未実施。

## Phase 18：BALD/ProbabilityVarianceとPhase 8 exit境界の照合

`tests/acquisition/test_phase8_bald_probability_variance_reference.py`のtest本文と旧`verification.md`のPhase 8 exit reconciliationを直接確認した。[Acquisition verification guide](../../../docs/development/theory-implementation/guide/verification/acquisition.md)に64固定draw・2候補・2クラス（`[64,2,1,2]`）でのProbabilityVarianceの母分散型集約とBALDのentropy差（atol/rtol=1e-10）を追記し、モデルsampling分布とMC収束は検証していないことを明示した。Phase8のexit recommendationは代表的な6つのreferenceのproperty範囲のみを対象とし、情報獲得・hypervolume・制約付き等の全獲得関数を保証しない。[移行台帳](phase1-heading-preservation-ledger.md)には固定`verification.md:320–334`および`:376–407`の部分照合を追加した。

**判定：partial / review required。** 新規の独立テスト実行や174見出しの完全移行は未達であり、旧Mappingを保全する。

## Phase 18：Replicate noiseとPhase 7〜9 exit判定の照合

`tests/models/robust/test_phase7_replicate_noise_reference.py`の本文を確認し、[Models verification guide](../../../docs/development/theory-implementation/guide/verification/models.md)へ不等replicate数（入力0.0に3件、1.0に4件）の標本平均・不偏標本分散・平均の分散（`s²/n`）と、`noise_floor=1e-12`、テストの`assert_close`がTorch既定tolを使うことを記録した。さらに旧`verification.md`のPhase7 final exit、Phase8 representative exit recommendation、Phase9 representative exitを区別し、[固定174見出し台帳](phase1-heading-preservation-ledger.md)の対象区間108–121、274–291、376–407、476–510へ部分照合を追記した。

**判定：partial / review required。** 旧Phase7〜9の限定的な数理property検証の完了と、現在のPhase18によるMapping全文の意味論的な移行完了は別。今回独立テストを再実行した証拠はなく、全174見出しの移行・検収は引き続き未達。旧5 Mappingファイルと元ステータスを保全する。

## Phase 18：Verification冒頭のevidence rulesとmodel matrixの意味論照合

旧`docs/development/theory-implementation/verification.md`冒頭1–64行（Phase1固定行番号）のPurpose、Phase7–9 ownership表、8モデルfamily検証行列、初期Phase7 execution gateを読んで、[日本語Verification入口](../../../docs/development/theory-implementation/guide/verification/README.md)へ検証証拠・入力条件・実測値記録の要件、過去の計画と後続exit判定の区別を追記した。既に直接確認済みのreplicate-noise testについて、索引内の『本文未照合』という古い説明も修正した。[174見出し台帳](phase1-heading-preservation-ledger.md)へ旧1–3、4–26、27–39、40–64行の部分照合を追加した。

**判定：partial / review required。** Exact GP条件付きposteriorとcategorical relabeling等、旧初期検証行列に明記された全propertyの独立数値検証を完了したとは宣言しない。本文・リンク・実測値の全件semantic migrationにも未達であり、旧Mappingの原文と174見出しのstatusを保全する。

## Phase 18：Exact GP conditioning / Mixed categorical relabelingの確認

旧Phase7モデル検証行列にあるExact GP kernel-conditioningとcategorical label permutationについて、既存テストの所在をGitHubで検索し`tests/models/standard/test_phase7_exact_gp_reference.py`および`tests/models/standard/test_phase7_mixed_categorical_reference.py`の本文を直接検査した。[Models verification guide](../../../docs/development/theory-implementation/guide/verification/models.md)に前者の固定RBF/Cholesky条件付け（float64、noise 0.025、lengthscale 0.4、平均/全共分散/variance対角、2e-5）、後者のラベル0→17、1→-4、2→9に対するmixed kernel covariance不変性（1e-10）を追記した。これにより従前の『テストパス未確認』説明を訂正し、[見出し台帳](phase1-heading-preservation-ledger.md)の`verification.md:27–39`へproperty別の部分照合を追記した。

**判定：partial / review required。** 学習済みGPのgeneral correctnessや全mixed variantの意味論的保証ではなく、既存テスト本文の静的照合に限定。今回の独立テスト再実行・174見出し完全移行を意味しない。

## Phase 18：Verification見出し31件のinventory確定（検収未完了）

[Phase1見出し台帳](phase1-heading-preservation-ledger.md)から旧`verification.md`のH1–H3区間31件を機械的に抽出し、それぞれの原見出し、旧固定行範囲、日本語ガイド受け皿、確認証拠の種類、完全移行に必要な残作業を31行の表へ明示した。**22件はテスト本文のassertionへの部分照合、9件は方針・初期行列・履歴・exit節の主要論点への部分照合**。見出しの索引整備という狭い意味では31/31件、原区間の本文・表・数式・PR/CI・リンクすべてのsemantic migrationを確認した`verified`は**0/31件**。この区別を持たない『31件完了』の表現を禁止する。

検査は原文との完全な差分照合ではなく、以前実施した部分照合の記録を棚卸ししたもの。旧174行の`planned (review required)`を変更せず、旧Mapping5ファイルは維持。以後はVerificationの各区間で残る全文移行の根拠を埋めるか、他4ファイルの見出し別reviewを進める。

## Phase 18：Verificationモデル検証行列8行の要件・証拠差分

旧`verification.md`冒頭1–64行と`guide/verification/README.md`を再照合し、旧Phase7 model verification matrixの8 familyについて、原文の要求と現在確認できる特定テストassertionとの相違を[日本語検証索引](../../../docs/development/theory-implementation/guide/verification/README.md)に一対一で記載した。Exact GP Cholesky参照やMixed categorical label relabelingはすでに実際のテストファイルを発見・直接確認済みであるため、索引の古い『別途確認が必要』という表現を修正した。一方でExact GPの他kernel条件、カテゴリ再学習posterior、Kronecker posterior conditioning、multi-fidelity全block、public heteroskedastic observation posteriorなどは引き続き独立reference未確認とする。

**判定：行列8/8行の要件と特定証拠との差分を文書化。完全なsemantic migrationの検収ではない。** Phase1固定の旧`verification.md:27–39`および全174見出しの`planned (review required)`を昇格せず、旧Mapping5ファイルを保持する。今回の作業で数学的テストの新規独立再実行はしていない。

## Phase 18：旧Phase 7 execution / exitの履歴整合性

旧`verification.md`の固定区間149–183、223–244、274–291の原文を再読し、[Verification日本語索引](../../../docs/development/theory-implementation/guide/verification/README.md)に時系列の表を追加した。初期の9件のrepresentative reference、#1424–#1426の追加3参照、#1426 final-head CIが当時pendingだった判定、後続#1428/#1429によるカテゴリ保持と**手動観測分散合成**の追加、最終的なproperty-specific complete判定を区別した。後続のexitはpublic `posterior(observation_noise=True)`の位置依存noise注入を認定していない。

**判定：partial / review required。** 古い`remains open`と後続の限定的`complete`を時系列で両立させ、#1426の当時のpendingを現在のCI状態と混同しない。PR番号と証拠範囲の静的照合であり、各歴史的PRの最新状態を新たに取得していない。全174見出しの完全移行は未達、旧Mapping保全。

## Phase 18：Phase 8 / Phase 9 final exitの証拠範囲と未検証契約

旧`verification.md`のPhase 8 `376–407`とPhase 9 `476–510`（Phase1固定行番号）を再確認し、[Verification日本語索引](../../../docs/development/theory-implementation/guide/verification/README.md)に過去のPR番号・代表的検証property・未検証項目を対比した。Phase8は#1431–#1436の六つの数式/MC参照を対象とする**exit recommendation**であり、noisy improvement、hypervolume、constraints、lookahead等の数値referenceを保証しない。Phase9は#1438–#1442の五つの代表propertyについての**complete decision**であり、全backendの制約可行性、global convergenceや混合/非同期最適化は対象外である。

**判定：partial / review required。** 旧Mapping内の歴史的CI/PR記録を照合したもので、過去のPR状態を今回の作業で全件再照会してはいない。数値テストを新規独立実行したわけではなく、Phase 18における174見出し全体の完全なsemantic migrationも未達。元台帳の`planned (review required)`と旧5 Mappingファイルは保全する。

## Phase 18：Verification以外の143見出しを初回棚卸し

[Phase1固定見出し台帳](phase1-heading-preservation-ledger.md)から`README.md` 13件、`models.md` 91件、`acquisition.md` 18件、`optimization.md` 21件、合計143見出しを抽出し、仮のprimary/secondary割当が全件存在することを確認した。うち25件はprimaryが`audit/`であり、Phase番号やaudit語だけで技術本文を歴史資料へ誤仕分けする危険があるため、優先レビュー表を台帳に追加した。

**判定：inventory / review required。** 143/143件は**仮の移行先が台帳にある**という意味だけであり、元区間の全文を実際に読み、技術数式・public API・negative contract・テスト結果・CI/PR履歴が移行先で完全に保全されたことを意味しない。新たな`verified`は0件。Verificationの31見出しと合わせ174件の旧台帳ステータスは変えず、旧Mapping5ファイルを保全する。
