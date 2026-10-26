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
| Models | Kronecker covariance、Multi-fidelity downsampling、PCA reducer、Binary probit、Gaussian adapter | 固定データのprior/kernel/transform/probability/sampling不変量 | 全posterior条件付け・学習精度・異なるvariant |
| Models | Heteroskedasticの分散分離、PCAGPの座標一致、MixedPCAGPのカテゴリ保持、Random Forestのtree moments | 公開APIとの固定条件・固定数値tolでの一致 | 位置依存の公開observation posterior、較正、他reducer/estimator |
| Acquisition | regression/classification AL、BALD、ProbabilityVariance、analytic Gaussian、MC qSimpleRegret/qUCB | utilityの算術、sample集約順序、固定乱数とshape | qNEI、EHVI、情報利得、全constrained compositionの独立数値参照 |
| Optimization | nonlinear candidate feasibility E2E、TuRBO state、one-shot/mixedとNSGA-IIの実装契約 | テスト対象の可行性・state遷移・行別カテゴリ割当て等 | 一般制約下のhard feasibility、global optimum、regret |

Modelsの`replicate noise`等、[Models verification](models.md)に名前が載っていても、**この再照合作業でテスト本文をまだ読んでいない項目**はassertion-level照合済みとはしない。上表のAcquisitionやOptimizerにも同じ注意が必要であり、すべての掲載対象を同じ確度のVerifiedに一括昇格させない。数式・shape・negative contractの実証が不足するものは[limitations](limitations.md)で区別する。

元[Verification Mapping](../../verification.md)の古い「Phase 7 remains open」等は**当時の履歴として保全**されている。同じファイル内に、その後の「Phase 7 final exit reconciliation」（代表的property別スコープに限定したcomplete判定）もあるため、古い節だけを引用して現在の結論にしない。最新のexit判定でも、位置依存observational noiseのpublic posteriorは保証されていない。文書移行作業そのものの完了判定とは独立である。
