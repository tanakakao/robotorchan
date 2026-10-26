# 数学的検証・契約テスト・未検証事項

このディレクトリは、日本語の理論→実装ガイドを読む際に必要な**検証根拠への導線**である。原文である[Verification Mapping](../../verification.md)の数値条件、テストファイル、過去のaudit・PR/CI記録は改変・削除しない。検証記録を広いmodel familyの品質証明へ拡大解釈しない。

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

各テーマではテストの**ファイルパスと具体的なassertion対象**を示し、完全な設定・PR証跡・境界の正本は[Verification Mapping](../../verification.md)へ戻す。既存の旧Mappingは移行台帳が`verified`になるまで保持する。

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

元[Verification Mapping](../../verification.md)の古い「Phase 7 remains open」等は**当時の履歴として保全**されている。同じファイル内に、その後の「Phase 7 final exit reconciliation」（代表的property別スコープに限定したcomplete判定）もあるため、古い節だけを引用して現在の結論にしない。最新のexit判定でも、位置依存observational noiseのpublic posteriorは保証されていない。文書移行作業そのものの完了判定とは独立である。


## 旧Verification Mapping冒頭の移行契約（原文1–64行を照合）

旧Mappingの`Purpose and evidence rules`は、Models/Acquisition/Optimizationの**静的な対応表**と、独立の数理参照に基づく**数値検証**を明確に分離する。公開registryへの登録、tensor shapeの一致、CI成功だけでは数学的propertyを`Verified`にしない。この基準を現在のガイドでも維持する。

| 旧Phase | 数理検証の対象 | 現ガイドの受け皿 | 依然として個別確認が必要な例 |
| --- | --- | --- | --- |
| 7 | モデルposterior・likelihood・covariance・入出力変換 | [Models](models.md) | Exact GP Cholesky conditioningとcategorical kernel relabelingは固定条件のテスト本文を照合済み。Kronecker priorとposteriorの区別・他variantは未証明 |
| 8 | Acquisition値・MC推定量・objective・constraints | [Acquisition](acquisition.md) | qNEI、multi-objective hypervolume、制約付き組成・情報獲得などの独立数値式 |
| 9 | Candidate最適化・可行性・状態・座標変換 | [Optimization](optimization.md) | 全backendの可行性・収束・座標変換の一般保証 |

再現可能な記録として原文が要求するのは、**対象model/method ID、独立数式/参照、仮定、dtype/device、seed、数値tol、実測結果、厳密なassertion対象**である。固定数値例の一部を確認できても、原文が提示する**モデル検証行列の全行**（Exact GP条件付きposterior、カテゴリID置換不変性を含む）の実証に置き換わるものではない。

旧`Initial Phase 7 execution gate`には、既存テストのvalue-level assertionの精査、学習を除いたfloat64 kernel conditioning oracle、カテゴリ置換不変性、分類のlatent/probability区別、focused testとCIの結果記録、他familyへの段階的拡張が含まれる。これは**当初の検証手順・到達目標**として保全し、後続の代表property限定exit判定と同一視しない。旧Mapping冒頭1–64行に由来する技術・履歴上の意味は[原文](../../verification.md)を正本として参照し続ける。


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

この表は原文27–39行の8行を**一対一で意味論比較するための差分台帳**であり、8 familyに対する無制限の数学的合格判定ではない。各テストの具体的assertionと例外は[Models verification](models.md)、原文の歴史的判定は[Verification Mapping](../../verification.md)に残す。
