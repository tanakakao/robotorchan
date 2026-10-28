# 数学的検証・契約テスト・未検証事項

このディレクトリは、日本語の理論→実装ガイドを読む際に必要な**検証根拠への導線**である。原文である[Verification Mapping](../../../.github/project-records/theory-guide/mapping/verification.md)の数値条件、テストファイル、過去のaudit・PR/CI記録は改変・削除しない。検証記録を広いmodel familyの品質証明へ拡大解釈しない。

## 何を「検証した」と呼ぶか

| Evidence | 読み方 | 言えること・言えないこと |
| --- | --- | --- |
| Source inspection / registry | シンボル・分岐・所有権を静的確認 | APIの存在は数式の正しさではない |
| Contract / shape test | 例外・入出力shape・bounds・dtype等 | 契約を満たすことは数値恒等式の証明ではない |
| Independent mathematical reference | 実装とは別に書いた数式・固定例・不変量と比較 | **対象property・入力条件・許容誤差に限定**して検証 |
| Integration / Notebook | fit→posterior→acquisition→optimizer等の接続 | 連結例が走っても、すべての組合せや較正を保証しない |
| CI / Documentation links | テストスイート実行とリンク検査の結果 | CI成功のみを数学的Verifiedとは呼ばない |

## 検証テーマの入口

- [モデル・posterior・noise](models.md)：固定GP共分散・PCA・異分散・分類・non-GP adapterなど。
- [獲得関数](acquisition.md)：analytic Gaussian、MC sample moment、分類AL、情報利得・制約。
- [候補最適化・探索](optimization.md)：constraint feasibility、埋め込み幾何、TuRBO state、candidate score。
- [未検証事項とnegative contracts](limitations.md)：対応しない機能、検証の外側、将来の調査点。

各テーマではテストの**ファイルパスと具体的なassertion対象**を示し、完全な設定・PR証跡・境界の正本は[Verification Mapping](../../../.github/project-records/theory-guide/mapping/verification.md)へ戻す。既存の旧Mappingは移行台帳が`verified`になるまで保持する。

## 再現時の記録単位

テストを独立に再現する場合は、検証する数学的対象、モデル/獲得関数/optimizer ID、数式または参照法、入力shape・dtype・device、seed、数値tol（atol/rtol）、実測値、実行日時/コミット、対象テストを明記する。存在するテストについて**今回の作業で新たに実行した**と誤認させない。

このディレクトリは固定48章の**補助索引**であり、49番目の理論章ではない。学習ガイド本文へPhase/PR/CIの開発履歴を流し込まない。


## 代表的な独立参照の照合状況（範囲限定）

この表の「assertionを再照合」は、元のテスト本文と数式・入力条件を**文書上で直接比較した**ことを意味する。CIで実行されたという別種の証拠と混同しない。詳細は各分野の索引に記載する。

| 分野 | assertionを再照合した対象 | 証拠が成立する範囲 | 依然として別途必要な検証 |
| --- | --- | --- | --- |
| Models | Exact GP conditioning、Mixed categorical relabeling、Kronecker covariance、Multi-fidelity downsampling、PCA reducer、Binary probit、Gaussian adapter | 固定データのprior/kernel/transform/probability/sampling不変量 | 他kernel・variantのposterior条件付け、学習精度、汎化・較正 |
| Models | Heteroskedasticの分散分離、PCAGPの座標一致、MixedPCAGPのカテゴリ保持、Random Forestのtree moments | 公開APIとの固定条件・固定数値tolでの一致 | 位置依存の公開observation posterior、較正、他reducer/estimator |
| Acquisition | regression/classification AL、BALD、ProbabilityVariance、analytic Gaussian、MC qSimpleRegret/qUCB | utilityの算術、sample集約順序、固定乱数とshape | qNEI、EHVI、情報利得、全constrained compositionの独立数値参照 |
| Optimization | nonlinear candidate feasibility E2E、TuRBO state、one-shot/mixedとNSGA-IIの実装契約 | テスト対象の可行性・state遷移・行別カテゴリ割当て等 | 一般制約下のhard feasibility、global optimum、regret |

Modelsの`replicate noise`もテスト本文と直接照合済みである。ただし、[Models verification](models.md)に名前が載ること自体は、元区間の全情報の移行完了を意味しない。上表のAcquisitionやOptimizerにも同じ注意が必要であり、すべての掲載対象を同じ確度のVerifiedに一括昇格させない。数式・shape・negative contractの実証が不足するものは[limitations](limitations.md)で区別する。

元[Verification Mapping](../../../.github/project-records/theory-guide/mapping/verification.md)の古い「Phase 7 remains open」等は**当時の履歴として保全**されている。同じファイル内に、その後の「Phase 7 final exit reconciliation」（代表的property別スコープに限定したcomplete判定）もあるため、古い節だけを引用して現在の結論にしない。最新のexit判定でも、位置依存observational noiseのpublic posteriorは保証されていない。文書移行作業そのものの完了判定とは独立である。


## 旧Verification Mapping冒頭の移行契約（原文1–64行を照合）

旧Mappingの`Purpose and evidence rules`は、Models/Acquisition/Optimizationの**静的な対応表**と、独立の数理参照に基づく**数値検証**を明確に分離する。公開registryへの登録、tensor shapeの一致、CI成功だけでは数学的propertyを`Verified`にしない。この基準を現在のガイドでも維持する。

| 旧Phase | 数理検証の対象 | 現ガイドの受け皿 | 依然として個別確認が必要な例 |
| --- | --- | --- | --- |
| 7 | モデルposterior・likelihood・covariance・入出力変換 | [Models](models.md) | Exact GP Cholesky conditioningとcategorical kernel relabelingは固定条件のテスト本文を照合済み。Kronecker priorとposteriorの区別・他variantは未証明 |
| 8 | Acquisition値・MC推定量・objective・constraints | [Acquisition](acquisition.md) | qNEI、multi-objective hypervolume、制約付き組成・情報獲得などの独立数値式 |
| 9 | Candidate最適化・可行性・状態・座標変換 | [Optimization](optimization.md) | 全backendの可行性・収束・座標変換の一般保証 |

再現可能な記録として原文が要求するのは、**対象model/method ID、独立数式/参照、仮定、dtype/device、seed、数値tol、実測結果、厳密なassertion対象**である。固定数値例の一部を確認できても、原文が提示する**モデル検証行列の全行**（Exact GP条件付きposterior、カテゴリID置換不変性を含む）の実証に置き換わるものではない。

旧`Initial Phase 7 execution gate`には、既存テストのvalue-level assertionの精査、学習を除いたfloat64 kernel conditioning oracle、カテゴリ置換不変性、分類のlatent/probability区別、focused testとCIの結果記録、他familyへの段階的拡張が含まれる。これは**当初の検証手順・到達目標**として保全し、後続の代表property限定exit判定と同一視しない。旧Mapping冒頭1–64行に由来する技術・履歴上の意味は[原文](../../../.github/project-records/theory-guide/mapping/verification.md)を正本として参照し続ける。


### 冒頭のモデル検証行列8 family：要求と実証範囲の差分

旧`verification.md`の見出し`Phase 7 — model verification matrix`について、8行の**要求property**と今確認できる独立referenceの**実証property**を分離した。旧行列の「suggested reference」はテスト結果そのものではない。

| 旧行列family | 確認できたテストと数値契約 | なお残る差分 |
| --- | --- | --- |
| Exact single-output GP | `test_phase7_exact_gp_reference.py`：固定RBF・固定noiseのCholesky平均/共分散（2e-5） | 原文の一般的なnoiseless/任意kernel参照、学習・数値条件の全範囲 |
| Mixed categorical GP | `test_phase7_mixed_categorical_reference.py`：4 queryのkernel共分散、カテゴリ0/1/2を17/-4/9へ置換（1e-10） | 学習入力の再ラベルと再学習posteriorの予測不変性は未確認 |
| Multi-task / Kronecker | `test_phase7_kronecker_covariance_reference.py`：2 taskのprior covariance、flatten順・PSD・task置換（1e-10） | posterior条件付け、variational方式への一般化 |
| Multi-fidelity | `test_phase7_multifidelity_kernel_reference.py`：downsampling fidelity-only kernelのoffset/rank-one（1e-10） | 全multi-fidelity block covarianceとposterior |
| Reduced / latent | `test_phase7_posterior_coordinate_reference.py`、`test_phase7_mixed_reduced_categorical_reference.py`：固定PCAとカテゴリ保持（1e-10） | 他reducer、reconstruction品質、学習精度 |
| Heteroskedastic / robust | `test_phase7_heteroskedastic_uncertainty_reference.py`：固定2段階noiseとlatent分離、手動観測分散（1e-12） | public観測posteriorへの位置依存noise注入、非Gaussian likelihood一般化 |
| Binary classification | `test_phase7_binary_probit_reference.py`：Gaussian-probit積分と2クラス確率（1e-10） | holdout calibration、classifier fitting、任意linkの数値積分 |
| Non-GP surrogates | `test_phase7_gaussian_posterior_reference.py`、`test_phase7_forest_estimator_reference.py`：adapter sample（1e-12）、7-tree ensemble moments（1e-12） | 他estimator・epistemic uncertaintyの較正 |

この表は原文27–39行の8行を**一対一で意味論比較するための差分台帳**であり、8 familyに対する無制限の数学的合格判定ではない。各テストの具体的assertionと例外は[Models verification](models.md)、原文の歴史的判定は[Verification Mapping](../../../.github/project-records/theory-guide/mapping/verification.md)に残す。


## 旧Phase 7の中間監査と最終exit：時系列による読み分け

原文[Verification Mapping](../../../.github/project-records/theory-guide/mapping/verification.md)の3つの区間は、互いに置き換えず、**当時の判定と後続の限定的な解消**として読む。

| 旧区間（Phase 1固定行番号） | 当時の証拠・判定 | 後続による変化と残る境界 |
| --- | --- | --- |
| `149–183` execution reconciliation | Exact GP・Mixed relabelingと#1416–#1422の代表参照を記録。CI/Docs成功を**実行証拠**として確認するが、reduced posterior座標、heteroskedastic分離、外部estimator同値には当時未達 | 後の#1424–#1426により3件の代表参照が追加。最初の「not yet complete」は当時の判断 |
| `223–244` property-specific exit audit | #1424・#1425のCI/Docs成功、#1426のfinal-head CIは**その監査時点でpending**。カテゴリ列保持とobservation posterior合成なども残り、`remains open` | 後の#1428がMixedPCAGPの元カテゴリID・posterior同値を追加。#1429は観測分散の**手動合成**のみ追加 |
| `274–291` final exit reconciliation | 後続の#1428・#1429を含め、**合意した代表property別の数学的reference範囲についてcomplete** | `posterior(observation_noise=True)`の位置依存noise自動注入は依然としてUnsupported/Unverified。全モデル・学習・較正・他reducerの認定ではない |

原文の「#1426 final-head CI pending」は**過去の監査時点の記録**であり、現時点でpendingと推定する根拠には使わない。反対に、後続のfinal exitをもって過去のCI結果を書き換えたり、未検証のAPI契約を`Verified`へ昇格させたりもしない。古い12 Phaseの開発履歴と、現在の固定18 Phase・48章の文書移行は別工程である。

**完全検収との差：** 以上は3区間の主要な表・条件付き判断・後続PR番号・negative contractの照合であり、全文章・全リンク・現在のPR状態を再確認した完全移行検収ではない。移行台帳の`planned (review required)`は維持する。


## 旧Phase 8・9のexit判定：数値propertyと未検証範囲

旧[Verification Mapping](../../../.github/project-records/theory-guide/mapping/verification.md)のPhase 8最終照合（固定行番号`376–407`）とPhase 9 exit audit（`476–510`）は、**代表的なpropertyを限定して合格とした歴史的判定**である。後続のPhase 18文書移行の検収と分離する。

| 旧exit | 原文にあるPRと確認対象 | 原文が認定しない主張 |
| --- | --- | --- |
| Phase 8：representative scopeへのexit recommendation | #1431 regression AL 4式、#1432 classification AL、#1433 BALD/ProbabilityVariance、#1434 analytic log-EI/log-PI/UCB、#1435 qSimpleRegret、#1436 qUCB。原文は6PRすべてmainへmerged・CIとDocs links成功と記録 | qNEI/qLogNEI、qEI/qLogEIのsample improvement、qEHVI/qNEHVI、制約の確率/feasibility重み、異種目的composition、情報獲得/lookaheadの**独立数学的検証** |
| Phase 9：representative property-specific scopeでcomplete | #1438 random/sobolのpool内argmax、#1439 TuRBO state、#1440 TuRBO ARD幾何、#1441 candidate違反量とfeasibility-first順位、#1442 Adam/AdamW/SGDの未対応constraint拒否。原文は5PRのCIとDocs links成功・mergedを記録 | 勾配法/derivative-free法の大域最適性・収束率、全backendの制約可行性、sampling品質、mixed/batch/async/multi-fidelity最適化の保証、学習済みARDの忠実度、restart中心の分布、PyTorch optimizerによる制約付き最適化（明示非対応） |

Phase 8の原文は**exit recommendation**、Phase 9は**exit decision**という表現の違いがあるため、同じ強さの普遍的な完了認定へ統一しない。両者とも「独立数学的参照」を要件とし、smoke・integration・contract testをそのまま数学的証明へ格上げしない。

**証拠の限界：** 上表の過去のPR/CI状態は旧Mappingに記録された当時の情報であり、この節のために個々の過去PRを再照会したものではない。個別数式・サンプル条件は[Acquisition](acquisition.md)と[Optimization](optimization.md)に、移行レビューの進捗はプロジェクト監査記録で管理する。原文の全文・表・リンクの意味論的移行検収は別途必要である。
