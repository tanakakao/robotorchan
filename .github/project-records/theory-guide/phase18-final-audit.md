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

## Phase 18：audit仮割当の実体確認とPhase3 routingへの優先順位修正

Phase1見出し台帳で`audit/`が仮primaryの25区間のうち、`acquisition.md:288–308`（outcome-constrained qEI/qNEI/qEHVIの合成）、`:346–362`（qKG/MF-KG/multi-step/MES/GIBBONとPES/JESの区分）、`optimization.md:182–205`（11名optimizer dispatch）、`:246–257`（mixed native / one-shot / NSGA-II）の原文4区間を直接確認した。いずれも実装契約・数学的意味・非対応条件を含むため、見出し名のPhase番号を理由にauditのみへ移してはならない。[Phase1 ledger](phase1-heading-preservation-ledger.md)へ区間ごとの技術的受け皿とレビュー限界を追記した。

重要な調整として、[Phase3本文レビュー済みrouting](phase3-migration-ledger.md)は174区間全体の再割当を既に記録している。Phase1の`audit/`25件は**過去の仮割当件数**であり、現在の確定routing件数ではない。Phase3も`body-reviewed routing / not migrated`なので、本文移行済みと誤認しない。旧台帳174見出しの`verified`は引き続き0件。元Mapping5ファイルは保全する。

## Phase 18：旧Acquisition 288–308の制約付きacquisition本文改善（部分移行）

Phase 3台帳が旧 `acquisition.md:288–308` を `guide/acquisition/01_foundations.md` と補助索引へ割り当てたことを確認した上で、技術内容の適切な主受け皿である [`guide/acquisition/08_constraints.md`](../../../docs/development/theory-implementation/guide/acquisition/08_constraints.md) に本文を直接追記した。旧区間と直後のInterpretation、`src/robotorchan/acquisition/composition.py`、`classification_constraints.py`、`tests/acquisition/test_composition.py` および理論正本 `docs/theory/acquisition/08_constraints.md` を照合し、6種類のconstructor、native BoTorchへの委譲、同一posterior sample-wise制約と分類PoF積の相違、qEI利用例、qNEI baseline、qEHVI ref_point/eta、X_pending/fantasizeのnegative contractを文章・式・表に整理した。

**判定：partial / review required。** この変更はPhase 3 routingの受け皿を内容に即して拡張したもので、元のPhase 3行を書き換えたり `verified` に昇格させたりしない。旧区間288–308および309行以降の全文・すべてのテストassertion・参照リンクの一対一semantic migration検収は未完了。既存テストの静的ソース確認のみで、新規独立数値実験やCI成功を主張しない。旧Mapping5ファイル・174見出しは維持する。

## Phase 18：情報獲得・lookahead（旧Acquisition 346–362）の本文改善

Phase 3 routingのprimary `guide/acquisition/06_lookahead.md`と旧Mapping固定区間346–362を確認し、技術内容をその主受け皿に反映するとともに、情報獲得固有の内容を `guide/acquisition/05_information_theoretic.md` に追記。KGの意思決定価値の概念式、qKG/multi-fidelity KG/multi-stepのBoTorch委譲、one-shot augmented batchと初期化、MES/GIBBON/PES/JESの情報対象、有限candidate_set、MES/GIBBONのnative実行例を補った。根拠として `src/robotorchan/acquisition/registry.py`、`docs/optimization/lookahead.md`、`tests/acquisition/test_information_theoretic.py` の既存ソースを直接確認した。

**判定：partial / review required。** MES/GIBBONテストは有限値のみ検査し、情報量の独立数値正当性を保証しない。PES/JESのローカル公開constructorも未確認。新規数値再実行なし。174区間の全文一対一semantic migrationのverified判定は0件を維持し、旧5 Mappingファイルを保全する。

## Phase 18：旧Optimization 182–205のoptimizer dispatch本文移行（部分照合）

旧Optimization Mappingの固定区間182–205、Phase3のprimary `guide/optimization/01_foundations.md`と補助索引のroutingを確認し、読者がbackendを選択するための説明を `guide/optimization/12_selection_guide.md` に追記した。11種類のOptimizerName、最大化対象A(X)、各backendのOwned/Delegatedの違い、sampled joint q-batchのargmax式と[N,q,d] shape、BoTorch固有引数とoptimizer_options、torch_*のcandidate制約拒否、GA mixed分岐、named dispatchに含まれないNSGA-II/native mixed/one-shotを明示した。`src/robotorchan/optim/dispatch.py`、`backends/torch.py`、`backends/sampling.py`、旧Mappingの本文を静的に照合し、最小sampling dispatch例を追加した。

**判定：partial / review required。** Phase3 routingの受け皿は意味に応じて補完したが、旧区間全体の一対一semantic migrationの検収には至っていない。数値テストの新規実行や全backend正当性の証明は行っていない。174区間のverifiedは0件、旧5 Mappingは保全する。

## Phase 18：旧Optimization 246–257の専門optimizer本文改善

Phase 3台帳の旧 `optimization.md:246–257` は主受け皿 `guide/optimization/05_constraints.md` と補助appendixを指定しているが、実装別の数理・APIを読者が追えるよう、関連する48章内の `04_mixed_discrete.md`、`08_batch_one_shot.md`、`07_evolutionary.md` の本文を直接改善した。旧原文と `src/robotorchan/optim/backends/botorch.py`、`src/robotorchan/optim/one_shot/mixed.py`、`src/robotorchan/optim/backends/nsga2.py` を直接確認した。native mixedのfixed_features_listとinter-point nonlinear拒否、mixed one-shotの行ごとのカテゴリ割当てとM**augmented_q列挙・q=1制約、NSGA-IIの支配関係・vector objectiveとscalar qEHVIの相違・使用例を補った。

**判定：partial / review required。** Phase 3 routing自体や旧Mappingは書き換えず、正式な完全移行のverifiedに昇格しない。実装ソースの静的照合に基づく説明であり、今回の独立数値実験や全制約の可行性・最適性の証明ではない。174見出しは引き続き0 verifiedとして保全する。

## Phase 18：optimizer制約処理の主受け皿への補充

旧optimization.md固定区間246–257および直後のconstraint handling表について、Phase 3で主受け皿とされたguide/optimization/05_constraints.mdへbackend別のNATIVE・FEASIBILITY_FIRST・PENALTY・UNSUPPORTEDの数理的意味と、BoTorch形式の線形入力制約の最小例を追加した。src/robotorchan/optim/capabilities.py、constraints/contracts.py、backends/botorch.py、backends/torch.pyと旧Mappingを確認した。mixed BoTorchのinter-point nonlinear拒否、PyTorchの制約拒否、feasible初期点要件を区別した。

**判定：partial / review required。** 旧区間全件の意味論的移行検収ではない。独立数値実行や全backendの一般可行性検証は行っていない。旧Mapping5ファイルおよび174見出しのverified=0を維持する。

## Phase 18：Exact GP posterior・MLL・混合kernelのガイド本文改善

既存の `guide/02_gaussian_process.md` と `guide/03_kernel.md` に、理論正本のGaussian conditioning式と公開posterior API、raw data snapshots、ExactMarginalLogLikelihood/fit、latentとobservation noiseの違い、native mixed covarianceの加法・積の構成およびstructural dimension拒否の説明を追加。`src/robotorchan/models/standard/single_task.py`、`src/robotorchan/models/base.py`、`docs/theory/02_gaussian_process.md`、旧 `models.md` の代表的なモデルsource対応を直接照合した。BoTorchとGPyTorchの既存機能への委譲とrobotorchan側の拡張責務を区別した。

**判定：partial / review required。** 旧Models Mapping全91見出しの一対一移行は未完了。現時点の174見出し `verified=0`を維持し、旧Mapping5ファイルを保全する。数値実験の新規再実行や一般GP理論の実装完全性の主張ではない。

## Phase 18：Exact/Variational/Classification/Fully Bayesianの学習目的対応

既存48章内の `guide/09_variational_gp.md`、`guide/24_classification.md`、`guide/21_advanced_high_dimensional_models.md` に、Gaussian exact MLL・variational ELBO・Bernoulli分類ELBO・Fully Bayesian SAASのNUTS posteriorの相違を数式、公開API、上流委譲と照合して追記した。対象は `src/robotorchan/models/standard/variational.py`、`classification/binary/standard/single_task.py`、`high_dimensional/fully_bayesian.py`、`models/base.py`。num_dataの全観測スケール、predict_probaの潜在分布からの積分、supports_mll=Falseなどの負契約を明記した。

**判定：partial / review required。** モデルMappingの全見出しの移行は未完了。NUTS・ELBOの独立数値実行も行っていない。旧Mapping5本保全、174見出しverified=0を維持する。

## Phase 18：MultiTask／ModelList／Heterogeneousのposterior意味論

guide/07_multitask_multioutput.mdとguide/24_classification.mdを補充した。`models/standard/multitask.py`のlong-format MultiTaskGPとblock-design KroneckerMultiTaskGP、`standard/model_list.py`の独立子GPとSumMarginalLogLikelihood、`models/heterogeneous.py`のglobal output ownership、entry_posterior/entry_predict_probaを直接照合し、出力軸・相関・latentとclass probabilityを混同しない説明を追加した。

**判定：partial / review required。** 旧Mapping全区間の一対一semantic migrationの判定は未実施、旧5 Mapping保全。174見出しのverified=0を維持。独立numerical cross-task covariance検証、確率校正、end-to-end最適化実行は行っていない。

## Phase 18：Multi-objective qEHVI/qNEHVIとHeterogeneous分類制約の接続

`guide/acquisition/07_multiobjective.md`に、BoTorch-native qEHVI/qNEHVIの違い、HeterogeneousModel内の共有回帰posterior所有権、ModelListGPとの境界、continuous residualとclassification PoFの組合せ、directed ref_point/partitioningとX_baselineの拒否契約を追記した。根拠は`src/robotorchan/acquisition/composition.py`、`src/robotorchan/models/heterogeneous.py`。`make_constrained_qehvi_acquisition`は存在するが同名のconstrained qNEHVI builderは確認できず、未実装のローカルAPIを推定していない。

**判定：partial / review required。** 数値hypervolume、joint feasibility、PoF calibrationや最適化性能の新規実行はなし。旧Mappingの保全と174見出しverified=0を維持する。

## Phase 18：分類確率ObjectiveとFeasibilityの意味論的区分

`guide/24_classification.md`と`guide/acquisition/08_constraints.md`にProbabilityObjective、ClassificationProbabilityOfFeasibility、ProbabilityOfFeasibility、SampleProbabilityOfFeasibility、SampleResidualFeasibility、ProbabilityResidualFeasibilityの数学的役割・公開API・非互換条件を追加。`src/robotorchan/semantics/{objectives,feasibility,probability}.py`および`src/robotorchan/acquisition/composition.py`を直接照合し、deterministic PoFとepistemic probability samples、objectiveとconstraint、posteriorとclass probabilitiesを区別した。

**判定：partial / review required。** 独立数値・校正評価・最適化再実行はなし。固定18 Phase・48章、旧Mapping5本、174見出しverified=0は維持する。

## Phase 18：モデル主章24本の本文密度の横断点検と3章の契約照合

`guide/01_...`から`guide/24_classification.md`のファイル配置を一覧化し、全24章が存在することを確認した。モデル中心の薄い章について本文冒頭・節立てを横断点検した。特に `05_mixed_variables.md`、`06_multi_fidelity.md`、`10_preference_learning.md` はそれぞれ35行、34行、34行の概説主体だったため、今回 `standard/single_task.py`、`standard/multi_fidelity.py`、`preference/pairwise.py` および `models/base.py` と照合して追加解説を行った。

| 点検範囲 | 今回の判断 | 深いソース照合 |
| --- | --- | --- |
| 01–04：BO、Exact GP、Kernel、Acquisition基礎 | 既存本文維持。以前のGP/API加筆を保持 | 今回の新規再検証対象外 |
| 05 Mixed variables | native categorical vs helper/one-hot、structural列の境界を補強 | standard/single_task.py、models/base.py、standard/multi_fidelity.py |
| 06 Multi-fidelity | fidelity/観測ノイズ/設計列の区分、familyごとのlinear_truncated拒否を補強 | standard/multi_fidelity.py |
| 07–09 MultiTask、高次元、Variational | 既存の主題・節立て確認、本文維持 | 今回の新規再検証対象外 |
| 10 Preference learning | Pairwise likelihood、比較index、Laplace MLLの違いを補強 | preference/pairwise.py |
| 11–17 Structured、Hierarchical、Selection、Robust、Noise、Uncertain、Nonstationary | 本文冒頭と節立てを点検。短い章が残り、次回以降の深いAPI・数値根拠点検が必要 | 今回は各familyのソース深掘りなし |
| 18–21 Reduced、Neural、High-dimensional search、Advanced | 既存の概念区分・本文密度を確認、本文維持 | 今回の新規再検証対象外 |
| 22–24 Expressive、Non-GP、Classification | 既存の概念区分・本文密度を確認、本文維持 | 今回の新規再検証対象外 |

**判定：partial / review required。** 24章の存在確認と本文密度の横断点検は、各章の意味論的検収とは異なる。今回深く実装照合したのはモデル3章に限定する。特に11–17の短い章、実装のAPI例のruntime、tensor shape/dtype/device、数値参照は追加の重点確認を要する。旧Mapping保全、全174原文見出しのverified=0を維持。固定18 Phase・48章を変更しない。

## Phase 18：Structured / Hierarchical / Robustモデルの本文強化

前回のモデル24章の横断点検から、説明が薄い3章 `guide/11_structured_output.md`、`guide/12_hierarchical_contextual_gp.md`、`guide/14_robust_gaussian_process.md` を対象に、理論上の共分散・条件付き入力・外れ値モデリングと、公開実装ソースを照合した。参照ソースは `src/robotorchan/models/structured/{higher_order,hierarchical}.py`、`src/robotorchan/models/robust/robust.py`。Mixed HOGPで入力共分散因子のみ置換すること、hierarchical親次元とcat_dimsの重複拒否、RRPでのsingle-task/multitask実装経路を区別した。

**判定：partial / review required。** Contextual familyの個別内部実装、structured posteriorの数値同値性、robust support回復性能は今回は未検証。旧Mapping5本を保全し174見出しverified=0と固定18 Phase・48章を維持。

## Phase 18：Noise / Uncertain Input / Nonstationaryモデル本文の詳細照合

`guide/15_heteroskedastic_noise.md`、`guide/16_uncertain_input_gp.md`、`guide/17_nonstationary_gp.md`に、input-dependent noiseの交互学習とjoint variational loss、Gaussian uncertain-input expected RBF kernelと元入力posterior契約、Gibbs kernelの局所lengthscaleの数式およびMixed版の2分岐を追記した。確認ソースは`src/robotorchan/models/robust/{robust,joint_heteroskedastic,nonstationary}.py`および`src/robotorchan/models/uncertain/uncertain_input.py`。

**判定：partial / review required。** 各familyでsourceを照合したが、独立した数値PSD・variance calibration・posterior予測精度・dtype/device組合せの再実行はしていない。旧Mapping保全、174見出しverified=0、固定18 Phase・48章を維持。

## Expressive GPの理論・実装対応の追加照合

`guide/22_expressive_gp.md`へDeepGPのVariationalELBO/DeepApproximateMLL、MC DeepGPPosterior、tensor-valued observation_noise拒否、NNGP ReLU covariance recursion、Spectral Mixtureのkernel初期化とtask/mixed契約を追記した。根拠：`src/robotorchan/models/expressive/{deep_gp,infinite_width_bnn,spectral_mixture}.py`。DKLに対する既存記述を維持し、決定論的特徴写像と確率的層を区別。

**判定：partial / review required。** 独立数値kernel property、学習収束、MC posterior accuracyは未再実行。固定18 Phase・48章、旧Mapping保全、174見出しverified=0を維持。

## Phase 18：Non-GP posteriorとBO互換性のモデル本文照合

`guide/23_non_gp_surrogates.md`にRandom Forest・Extra Trees・Gradient Boosting・Histogram Boosting・NGBoostの経験分布／Gaussian predictive distributionの意味、ensemble `[s,q,m]`軸、complete bootstrap estimators、CPU/sklearn非微分経路、観測ノイズ拒否、NGBoost独立Gaussian posteriorを追記した。根拠：`src/robotorchan/models/non_gp/{random_forest,bootstrap,gradient_boosting,posterior,ngboost,distribution_posterior}.py`。既存24章構成を維持。

**判定：partial / review required。** APIの静的照合のみ。fit・posterior・MC acquisitionの数値再実行や予測不確実性の校正検収はしていない。旧Mapping5本の保全、174見出しverified=0、固定18 Phase・48章を維持。

## Phase 18：モデル横断の学習・posterior・acquisition契約表

`guide/13_model_selection.md`にモデル各章への章内リンクと、Exact/Variational/Pairwise/Structured/Heteroskedastic/Uncertain/Nonstationary/DeepGP/NNGP/Spectral/Non-GP/Classification/Fully Bayesianの学習方式・posterior意味論・獲得関数における注意点を横断対照できる教育用表を追加した。MultiTask、ModelListGP、HeterogeneousModelの責務の相違も読書導線として補足した。

**判定：partial / review required。** 本変更は既存章との意味論照合と読書導線の補強であり、24章の全行を対象にした網羅的ソース・API・数値テスト監査ではない。旧Mapping5本保全、174見出しverified=0と固定18 Phase・48章を維持。

## Phase 18：モデル横断ガイドのコード例契約確認

`guide/07_multitask_multioutput.md`のHeterogeneousModel例に未定義の`regression_model`・`binary_classifier`・`X`が残っていたため、`SingleTaskGP`と`BinarySingleTaskGPClassifier`のconstructorに合わせて入力tensor・学習ラベルを明示し、output ownership、分類確率と回帰posteriorのshape assertionを持つ独立したサンプルへ差し替えた。根拠は`src/robotorchan/models/{heterogeneous.py,standard/single_task.py,classification/binary/standard/single_task.py}`。新規の独立数値実験ではなく、CI実行結果はPR作成後に確認する必要がある。

**判定：partial / review required。** 他23章の全コード例の実行成功、学習収束、旧Mappingの174見出し検収を示さない。固定18 Phase・48章は維持。

## Phase 18：HeterogeneousModel掲載コードの実行回帰テスト

`tests/docs/test_theory_guide_heterogeneous_example.py`を追加し、`guide/07_multitask_multioutput.md`の掲載コードをMarkdownから抽出してそのまま実行する回帰テストを設定した。output_owner、出力数、predict_probaのshape・有限性・クラス確率の総和を検査する。従来の静的constructor照合より強い証拠とする一方、モデルfit・分類確率の校正・数値最適化の正当性を確認するテストではない。

**判定：CI待ち / partial。** 本変更前に実行成功を主張しない。旧Mapping5本、174見出しverified=0、固定18 Phase・48章を維持する。

## Phase 18：基本モデル3章の掲載例をそのまま実行するテスト

`tests/docs/test_theory_guide_core_model_examples.py`を追加した。`guide/02_gaussian_process.md`、`05_mixed_variables.md`、`09_variational_gp.md`の最初のPython fenced blockを直接抽出・実行し、Exact GPのposterior平均と分散、Mixed GPのposterior shape、Variational GPのELBO生成と訓練行数を確認する。掲載コードとテスト用コードの複製を避ける。学習の数値収束、独立条件付きGP参照、探索性能の評価とは区別する。

**判定：CI待ち / partial。** 旧Mapping5本保全、174見出しverified=0、固定18 Phase・48章は維持。

## Phase 18：Fidelity / MultiTask / Structured Outputの掲載例実行テスト

`tests/docs/test_theory_guide_structured_model_examples.py`で`guide/06_multi_fidelity.md`、`07_multitask_multioutput.md`、`11_structured_output.md`の掲載された最初のPython fenced blockをそのまま実行するテストを追加。Multi-FidelityとMultiTaskはposteriorのshapeと有限性、HigherOrderGPはtensor shapeと構築契約を確認する。これらのテストはfit完了や構造化posteriorの数値同値性を認証しない。

**判定：CI待ち / partial。** 旧Mapping5本、174見出しverified=0、固定18 Phase・48章を維持。

## Phase 18：入力不確実性・非定常GPのガイド掲載例テスト

`guide/16_uncertain_input_gp.md`のクラス名表示だけの例を`train_X_std`を与えてposteriorを取得する例に更新した。`tests/docs/test_theory_guide_kernel_model_examples.py`で第16章と第17章の掲載コードを直接抽出・実行し、UncertainInputのposterior形状・有限性・raw input std、第17章のGibbs local lengthscale形状・正値性を調べる。ソース契約に基づく実行テストであり、kernelの独立数値比較や学習収束の証明ではない。

**判定：CI待ち / partial。** 旧Mapping5本の保全、174見出しverified=0、固定18 Phase・48章を維持。

## Phase 18：Student-t Robust GPの掲載例と実行契約

`guide/14_robust_gaussian_process.md`のクラス名表示のみの例をStudentTSingleTaskGPの生成、training_lossとlatent posteriorの取得例へ更新した。`tests/docs/test_theory_guide_robust_model_example.py`では掲載コードを抽出して実行し、lossの有限性、posterior shape・有限性、make_mll()の拒否を確認する。根拠は`src/robotorchan/models/robust/student_t.py`のsupports_mll=False、StudentTLikelihoodおよび明示的training_loss経路。実行CIは作成後に確認する。

**判定：CI待ち / partial。** これはvariational fittingの収束、外れ値耐性の数値実証、独立尤度検証を意味しない。旧Mapping5本と固定18 Phase・48章を維持。

## Phase 18：Contaminated GP / RRPの学習契約照合

`guide/14_robust_gaussian_process.md`でGaussian contamination mixtureのlogsumexp尤度とMC+KL損失、潜在posteriorと近似contamination_diagnosticの相違、RRPのsparse support契約を説明した。第14章にContaminatedSingleTaskGPの使用例を追加し、`tests/docs/test_theory_guide_contaminated_example.py`でMarkdown掲載コードの直接実行、有限loss・診断確率範囲・make_mll拒否を検査する。実装照合元は`src/robotorchan/models/robust/{contaminated,student_t,robust}.py`。

**判定：CI待ち / partial。** 外れ値検出精度やRRP support回復の数値評価は対象外。旧Mapping保全、18 Phase・48章を維持。

## Phase 18：非定常Gibbs kernelの独立数値参照

`tests/models/robust/test_theory_guide_gibbs_reference.py`を追加した。1Dの局所lengthscaleを固定し、Gibbs共分散をスカラー式で独立計算して、実装の行列値との一致、対称性、対角1、有限点でのPSD、diag経路を比較する。さらにslope=0の定常RBF極限を検証する。理論第17章と実装`src/robotorchan/models/robust/nonstationary.py`の意味論対応を数値テストへ接続した。

**判定：CI待ち / partial。** このテストは非定常GPの学習収束、任意次元・パラメータでのPSD証明やBO最適化性能を保証しない。旧Mapping、18 Phase・48章を維持。

## Phase 18：Gaussian入力不確実性の期待RBF kernelの独立参照

`tests/models/uncertain/test_theory_guide_gaussian_input_kernel_reference.py`にて、入力平均と分散を連結した1DのGaussianUncertainInputKernelを対象に、解析式から個別に計算した行列と比較する。共分散の対称性、有限集合での半正定値性、diag=Trueの一致、入力分散がゼロとなる場合のRBF極限を検査する。対応する実装は`src/robotorchan/models/uncertain/uncertain_input.py`、理論ガイドは第16章。

**判定：CI待ち / partial。** 訓練収束や多次元共分散の独立参照、BO性能は対象外。旧Mapping5本、174見出しverified=0、18 Phase・48章の構成を維持。

## Phase 18：モデル24章の検証マトリクス

`model-chapter-verification-matrix.md`を新規追加し、モデル24章ごとに掲載例の直接実行テストと独立数値照合テストの対応を整理した。既存の証拠として`tests/docs/test_theory_guide_*.py`およびGibbs・Gaussian uncertain-inputの独立参照テストのみを記載し、空欄からテスト不存在を推論しない。モデル24章と獲得関数・最適化各12章の固定体系は変更しない。旧Mapping5本と174見出しは別途検証までverifiedにしない。

**判定：CI待ち / partial。** 対応表は証拠の索引であり、実装全域の品質保証でも数値収束の実証でもない。

## Phase 18：Kernel・Pairwise Preferenceガイドの掲載例実行テスト

`tests/docs/test_theory_guide_kernel_preference_examples.py`を追加し、ガイド第03章のRBFKernel/ScaleKernel付きSingleTaskGP、及び第10章のPairwiseGPの最初のPythonコードブロックを直接実行する回帰テストを定義した。第03章はposterior形状と有限値、第10章はcomparison shape、PairwiseLaplaceMarginalLogLikelihoodの型、単一出力契約を確認する。model-chapter-verification-matrix.mdに両章を掲載例実行の対象として追加した。

**判定：CI待ち / partial。** 選好推論のLaplace近似精度、独立数値照合、学習収束は未実証。固定18 Phase・48章を維持。

## Phase 18：Joint Heteroskedastic GPの掲載例実行

第15章のJointHeteroskedasticSingleTaskGPのクラス名表示例を、training_loss、response posterior、log-noise posterior、predicted_noiseの区別を示すコードへ置き換えた。`tests/docs/test_theory_guide_heteroskedastic_example.py`がMarkdownコードを直接実行し、有限損失、潜在分布の有限性、推定観測分散の正値性、make_mll()拒否を調べる。これは学習収束やnoise較正の数値実証ではない。

**判定：CI待ち / partial。** 固定18 Phase・48章、旧Mappingと174見出しの未確認状態を維持。

## Phase 18：階層条件付きGPの掲載例実行

第12章のSACGP名表示だけの例を、MixedHierarchicalConditionalKernelGPとhierarchical_dependenciesの具体的な設定へ更新。`tests/docs/test_theory_guide_hierarchical_example.py`は掲載コードを直接実行し、cat_dims、raw訓練tensor、ExactMarginalLogLikelihoodの生成を調べる。SACGPは別familyとしてtrain_Yvarとdecompositionが必要な点を説明。contextual全familyの実行、inactive invarianceの独立数値参照や学習収束は未検証。

**判定：CI待ち / partial。** 旧Mapping5本、174見出しverified=0、固定18 Phase・48章を維持。

## Phase 18：第12章 Contextual Multi-Outputの掲載例実行

第12章にMixedLCEMGPのtask列とcat_dimsを分離するコード例を追加し、`tests/docs/test_theory_guide_contextual_example.py`から当該Markdownを直接実行するテストを作成した。raw task feature、カテゴリ列、raw tensor、ExactMarginalLogLikelihood生成を確認する。既存の階層モデル例については、単なるクラス名表示とする古い説明を修正した。

**判定：CI待ち / partial。** embeddingの学習収束・context外挿・SACGPの全契約は未検証。固定18 Phase・48章と旧Mappingを保全する。

## Phase 18：PCA GP・非GP surrogate・Binary分類の掲載例実行

`tests/docs/test_theory_guide_surrogate_examples.py`を追加し、第18章PCAGP、第23章RandomForestSurrogate、第24章BinarySingleTaskGPClassifierの掲載コードを直接実行するテストを定義した。回帰posteriorのshape・有限性、分類確率の2-class軸・総和1・有限性、分類latent posteriorの有限性を確認する。第08章のSAAS MCMC例は計算コストと追加依存関係があるため対象外。検証マトリクスには該当3章を掲載例実行として登録。

**判定：CI待ち / partial。** 未学習分類モデルの校正やランダムフォレストの最適化性能を証明しない。固定18 Phase・48章と旧Mapping、174見出しの未完了判定を維持。

## Phase 18：Neural Reduction・Expressive GPの掲載例実行

`tests/docs/test_theory_guide_joint_encoder_examples.py`は第19章と第22章のJointEncoderGP掲載例をMarkdownから直接抽出して実行し、latent_dim=2の生成契約を確認する。検証マトリクスに両章の実行対象を追記する。

**判定：CI待ち / partial。** training_lossやposteriorの数値精度、encoderの収束は未検証。旧Mapping5本・174見出し、18 Phase・48章の構成を維持。
