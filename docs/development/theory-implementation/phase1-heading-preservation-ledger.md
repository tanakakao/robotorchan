# Phase 1 — 見出し別保全・移行台帳（初版）

> Baseline: `9de071c899dcce5fa2c7c3c0083281d529a3b077`。対象5ファイルをブランチ上で直接取得して生成。**移行先はキーワードに基づく仮割当であり、内容確認済み・移行完了を意味しない**。既存5ファイルは不変。

## 集計

| File | H1–H3 headings | Lines | Source blob SHA |
| --- | ---: | ---: | --- |
| `README.md` | 13 | 278 | `07b12399b64da8dbc99ce92897f9ac883031dc33` |
| `models.md` | 91 | 1796 | `cd9064f6ff8ccde74defe689da64e42faf012da9` |
| `acquisition.md` | 18 | 465 | `97372e1153544b6dc807bb69dc3245bbb0b0e10e` |
| `optimization.md` | 21 | 355 | `f8062d95d39bce03f68f0bfaecc5ea3620264bb5` |
| `verification.md` | 31 | 510 | `97036dd46c6e56c8ec171a808acefff0c1134ee6` |

合計 **174見出し**。表の範囲は各見出しの開始行から次のH1–H3直前まで。H4以下は親区間に含まれる。最初の見出しより前の文章は存在しない。

## 機械割当の限界とレビュー基準

- モデル名を含む見出しであっても、本文に横断的な情報があればAppendixへの併記が必要。
- `Phase` を含む見出しを自動的にauditへ割り当てても、本文の技術情報をauditに閉じ込めてはならない。各行の本文を読んでprimary/secondaryを修正する。
- 行範囲は保全対象の所在を示す。Markdown表、脚注、数式、リンク、テスト名を省略してよいという意味ではない。
- 全行のreview statusが確認済みになるまでは元文書を置換・削除しない。
- 章番号別ガイドに収まらない情報はAppendixへ残し、関連章から逆参照する。

## 全見出し台帳

| Source | Lines | Level | Original heading | Tentative primary | Tentative secondary | Status |
| --- | --- | ---: | --- | --- | --- | --- |
| `README.md` | 1–2 | 1 | Theory–Implementation Mapping: 12-phase development cycle | `audit/` | `appendix/` | planned (review required) |
| `README.md` | 3–10 | 2 | Baseline and scope | `audit/` | `appendix/` | planned (review required) |
| `README.md` | 11–23 | 2 | Existing ownership (preserve) | `README.md` | `appendix/` | planned (review required) |
| `README.md` | 24–41 | 2 | Confirmed source-of-truth boundaries | `README.md` | `appendix/` | planned (review required) |
| `README.md` | 42–66 | 2 | Reader navigation (Phase 10) | `audit/` | `appendix/` | planned (review required) |
| `README.md` | 67–86 | 2 | Mapping record contract | `README.md` | `appendix/` | planned (review required) |
| `README.md` | 87–94 | 2 | Mapping documentation layout | `README.md` | `appendix/` | planned (review required) |
| `README.md` | 95–107 | 2 | Verification and remaining work | `README.md` | `appendix/` | planned (review required) |
| `README.md` | 108–136 | 3 | Recorded phase decisions | `audit/` | `appendix/` | planned (review required) |
| `README.md` | 137–174 | 2 | Phase 6 exit reconciliation (static scope) | `audit/` | `appendix/` | planned (review required) |
| `README.md` | 175–201 | 2 | Phase 10 navigation exit audit | `audit/` | `appendix/` | planned (review required) |
| `README.md` | 202–231 | 2 | Phase 11 mapping consistency exit audit | `audit/` | `appendix/` | planned (review required) |
| `README.md` | 232–278 | 2 | Phase 12 final audit: evidence and decision | `audit/` | `appendix/` | planned (review required) |
| `models.md` | 1–4 | 1 | Model theory–implementation mapping | `appendix/model-composition-and-delegation.md` | `appendix/model-composition-and-delegation.md` | planned (review required) |
| `models.md` | 5–17 | 2 | Audit provenance and limits | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 18–31 | 2 | Registry boundaries | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 32–51 | 2 | Regression model inventory by existing guide | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 52–91 | 2 | Binary classification registry inventory | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 92–106 | 2 | Representative code-path findings | `appendix/model-composition-and-delegation.md` | `appendix/model-composition-and-delegation.md` | planned (review required) |
| `models.md` | 107–119 | 2 | Existing audit machinery | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 120–131 | 2 | Phase 2 public-export reconciliation | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 132–146 | 3 | Regression public symbols outside the registry | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 147–185 | 3 | Binary registry IDs and exported model classes | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 186–207 | 2 | Representative test assertion mapping | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 208–225 | 2 | Component-level implementation ownership | `appendix/model-composition-and-delegation.md` | `appendix/model-composition-and-delegation.md` | planned (review required) |
| `models.md` | 226–235 | 3 | Limits of the current evidence | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 236–256 | 2 | Exhaustive static provenance export | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 257–271 | 2 | Documentation link existence validation | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 272–289 | 2 | Static test-reference discovery | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 290–315 | 2 | Inspected runtime composition and mathematical ownership | `appendix/model-composition-and-delegation.md` | `appendix/model-composition-and-delegation.md` | planned (review required) |
| `models.md` | 316–338 | 2 | Inspected test assertions and verification boundaries | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 339–350 | 2 | Optional static upstream composition scan | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 351–367 | 2 | Conditional import provenance and static scan boundaries | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 368–393 | 2 | Phase 2 evidence levels and follow-up matrix | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 394–422 | 2 | Classification model documentation ownership | `24_classification.md` | `appendix/model-composition-and-delegation.md` | planned (review required) |
| `models.md` | 423–441 | 2 | Source-inspected classification delegation: non-GP and label noise | `15_heteroskedastic_noise.md` | `appendix/model-composition-and-delegation.md` | planned (review required) |
| `models.md` | 442–458 | 2 | Source-inspected classification: heterogeneous ensembles and uncertain inputs | `16_uncertain_input_gp.md` | `appendix/model-composition-and-delegation.md` | planned (review required) |
| `models.md` | 459–477 | 2 | Inspected classification test assertions: ensemble and uncertain input | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 478–497 | 2 | Inspected classification test assertions: label noise and sklearn adapters | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 498–518 | 2 | Source and assertion audit: binary multitask and DeepGP | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 519–540 | 2 | Source and assertion audit: high-dimensional binary classifiers | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 541–563 | 2 | Source and assertion audit: neural, ensemble, robust and categorical models | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 564–588 | 2 | Source and assertion audit: contamination, replicates, bootstrap and robust composition | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 589–606 | 2 | Repeatable registry-wide static evidence summary | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 607–625 | 2 | Inspected regression non-GP implementations: NGBoost and ExtraTrees | `23_non_gp_surrogates.md` | `appendix/model-composition-and-delegation.md` | planned (review required) |
| `models.md` | 626–649 | 2 | Inspected regression noise and heterogeneous multitask implementations | `07_multitask_multioutput.md` | `appendix/model-composition-and-delegation.md` | planned (review required) |
| `models.md` | 650–675 | 2 | Source and assertion audit: multi-fidelity regression families | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 676–697 | 2 | Source and assertion audit: regression DeepGP models | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 698–723 | 2 | Source and assertion audit: expressive stationary and infinite-width kernels | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 724–746 | 2 | Source and assertion audit: joint neural representation regression | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 747–773 | 2 | Classification documentation mapping at registry-ID granularity | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 774–798 | 2 | Phase 2 inventory sign-off and scope | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 799–806 | 2 | Phase 4: theory-to-implementation records | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 807–821 | 3 | Exact single-task Gaussian-process regression | `02_gaussian_process.md` | `appendix/model-composition-and-delegation.md` | planned (review required) |
| `models.md` | 822–836 | 3 | Native mixed-input Gaussian-process regression | `05_mixed_variables.md` | `appendix/model-composition-and-delegation.md` | planned (review required) |
| `models.md` | 837–851 | 3 | Bernoulli variational Gaussian-process classification | `09_variational_gp.md` | `appendix/model-composition-and-delegation.md` | planned (review required) |
| `models.md` | 852–866 | 3 | Long-format multitask Gaussian-process regression | `07_multitask_multioutput.md` | `appendix/model-composition-and-delegation.md` | planned (review required) |
| `models.md` | 867–881 | 3 | Kronecker block-design multitask GP | `07_multitask_multioutput.md` | `appendix/model-composition-and-delegation.md` | planned (review required) |
| `models.md` | 882–896 | 3 | SAAS fully Bayesian single-task GP | `21_advanced_high_dimensional_models.md` | `appendix/model-composition-and-delegation.md` | planned (review required) |
| `models.md` | 897–911 | 3 | Stochastic deep Gaussian-process regression | `02_gaussian_process.md` | `appendix/model-composition-and-delegation.md` | planned (review required) |
| `models.md` | 912–926 | 3 | Random-forest ensemble regression | `23_non_gp_surrogates.md` | `appendix/model-composition-and-delegation.md` | planned (review required) |
| `models.md` | 927–941 | 3 | Reduced input/output Gaussian-process regression | `18_dimensionality_reduction_gp.md` | `appendix/model-composition-and-delegation.md` | planned (review required) |
| `models.md` | 942–956 | 3 | Student-t robust variational regression | `09_variational_gp.md` | `appendix/model-composition-and-delegation.md` | planned (review required) |
| `models.md` | 957–971 | 3 | NGBoost probabilistic non-GP regression | `23_non_gp_surrogates.md` | `appendix/model-composition-and-delegation.md` | planned (review required) |
| `models.md` | 972–986 | 3 | Heterogeneous regression/classification composition | `24_classification.md` | `appendix/model-composition-and-delegation.md` | planned (review required) |
| `models.md` | 987–1000 | 3 | Classification with flipped observation labels | `24_classification.md` | `appendix/model-composition-and-delegation.md` | planned (review required) |
| `models.md` | 1001–1014 | 3 | Classification with contaminating labels | `14_robust_gaussian_process.md` | `appendix/model-composition-and-delegation.md` | planned (review required) |
| `models.md` | 1015–1028 | 3 | Classification probability ensembles (GP and heterogeneous) | `24_classification.md` | `appendix/model-composition-and-delegation.md` | planned (review required) |
| `models.md` | 1029–1042 | 3 | Non-GP binary classification and bootstrap uncertainty | `23_non_gp_surrogates.md` | `appendix/model-composition-and-delegation.md` | planned (review required) |
| `models.md` | 1043–1056 | 3 | Classification with uncertain candidate inputs | `24_classification.md` | `appendix/model-composition-and-delegation.md` | planned (review required) |
| `models.md` | 1057–1070 | 3 | Multitask binary classification (long format and block input) | `07_multitask_multioutput.md` | `appendix/model-composition-and-delegation.md` | planned (review required) |
| `models.md` | 1071–1084 | 3 | Frozen reduced-input binary GP classification | `18_dimensionality_reduction_gp.md` | `appendix/model-composition-and-delegation.md` | planned (review required) |
| `models.md` | 1085–1099 | 3 | Deep variational GP binary classification | `09_variational_gp.md` | `appendix/model-composition-and-delegation.md` | planned (review required) |
| `models.md` | 1100–1131 | 3 | Remaining robust binary classification families | `14_robust_gaussian_process.md` | `appendix/model-composition-and-delegation.md` | planned (review required) |
| `models.md` | 1132–1168 | 3 | Sparse and embedded high-dimensional binary classification | `21_advanced_high_dimensional_models.md` | `appendix/model-composition-and-delegation.md` | planned (review required) |
| `models.md` | 1169–1206 | 2 | Phase 4 registry-wide reconciliation: outstanding detailed evidence | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 1207–1224 | 3 | Phase 4 exit gate (unchanged) | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 1225–1242 | 2 | Phase 4 inspected binary variant reconciliation: mixed and nonstationary | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 1243–1269 | 2 | Phase 4 inspected reduced-GP family mapping: frozen, mixed and joint | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 1270–1293 | 2 | Phase 4 inspected neural and variational reduced-GP variants | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 1294–1321 | 2 | Phase 4 inspected reduced multi-task implementations | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 1322–1344 | 2 | Phase 4 inspected jointly trained multi-task and design-reduced multi-fidelity GP | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 1345–1369 | 2 | Phase 4 reduced-family registry reconciliation (52 entries) | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 1370–1389 | 2 | Phase 4 inspected robust-noise heteroskedastic inference paths | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 1390–1404 | 2 | Phase 4 inspected contamination, replicate-noise and nonstationary regression models | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 1405–1420 | 2 | Phase 4 inspected Student-t robust regression variants | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 1421–1437 | 2 | Phase 4 inspected robust relevance-pursuit regression variants | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 1438–1460 | 2 | Phase 4 robust-noise registry reconciliation (27 entries) | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 1461–1477 | 2 | Phase 4 inspected uncertain-input registry models (3 entries) | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 1478–1493 | 2 | Phase 4 inspected structured-output regression registry (4 entries) | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 1494–1516 | 2 | Phase 4 inspected hierarchical/contextual registry (6 entries) | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 1517–1530 | 2 | Phase 4 inspected preference-learning registry (1 entry) | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 1531–1547 | 2 | Phase 4 inspected standard single-task, variational and multi-fidelity models | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 1548–1564 | 2 | Phase 4 inspected non-GP surrogate registry (5 entries) | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 1565–1581 | 2 | Phase 4 inspected core multitask and model-list registry contracts | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 1582–1600 | 2 | Phase 4 inspected specialized heterogeneous/hierarchical multitask models | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 1601–1620 | 2 | Phase 4 inspected expressive spectral and infinite-width NNGP models (12 entries) | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 1621–1637 | 2 | Phase 4 inspected expressive DeepGP hierarchy (4 entries) | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 1638–1656 | 2 | Phase 4 inspected high-dimensional SAAS, MAP-SAAS and ALEBO ownership | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 1657–1672 | 2 | Phase 4 inspected orthogonal additive high-dimensional GP variants | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 1673–1687 | 2 | Phase 4 inspected MAP-SAAS multi-fidelity model variants | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 1688–1734 | 2 | Phase 4 cross-family registry reconciliation: high-dimensional and multitask | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 1735–1785 | 2 | Phase 4 inventory-count closure audit (static, documentation-only) | `appendix/model-registry-and-public-api.md` | `audit/` | planned (review required) |
| `models.md` | 1786–1796 | 3 | KroneckerMultiTaskGP covariance verification reference | `07_multitask_multioutput.md` | `appendix/model-composition-and-delegation.md` | planned (review required) |
| `acquisition.md` | 1–4 | 1 | Acquisition theory–implementation mapping: Phase 3 inventory | `audit/` | `appendix/acquisition-contracts-and-compatibility.md` | planned (review required) |
| `acquisition.md` | 5–18 | 2 | Sources and evidence boundary | `acquisition/` | `appendix/acquisition-contracts-and-compatibility.md` | planned (review required) |
| `acquisition.md` | 19–38 | 2 | Distinct implementation paths | `acquisition/` | `appendix/acquisition-contracts-and-compatibility.md` | planned (review required) |
| `acquisition.md` | 39–47 | 2 | Acquisition/optimizer interaction | `acquisition/` | `appendix/acquisition-contracts-and-compatibility.md` | planned (review required) |
| `acquisition.md` | 48–60 | 2 | Fixed Phase 3 completion criteria | `audit/` | `appendix/acquisition-contracts-and-compatibility.md` | planned (review required) |
| `acquisition.md` | 61–78 | 2 | Inspected acquisition registry and assertion evidence | `acquisition/` | `appendix/acquisition-contracts-and-compatibility.md` | planned (review required) |
| `acquisition.md` | 79–114 | 2 | Phase 5 — regression active-learning acquisition mappings (source inspected) | `audit/` | `appendix/acquisition-contracts-and-compatibility.md` | planned (review required) |
| `acquisition.md` | 115–143 | 2 | Phase 5 — classification AL and Gaussian EPIG (source inspected) | `audit/` | `appendix/acquisition-contracts-and-compatibility.md` | planned (review required) |
| `acquisition.md` | 144–178 | 3 | ExpectedPredictiveInformationGain (regression) | `acquisition/` | `appendix/acquisition-contracts-and-compatibility.md` | planned (review required) |
| `acquisition.md` | 179–232 | 2 | Phase 5 — single-objective Monte Carlo BO (source inspected) | `audit/` | `appendix/acquisition-contracts-and-compatibility.md` | planned (review required) |
| `acquisition.md` | 233–249 | 2 | Phase 5 — multi-objective MC Bayesian optimization (source inspected) | `audit/` | `appendix/acquisition-contracts-and-compatibility.md` | planned (review required) |
| `acquisition.md` | 250–287 | 3 | Verified construction contracts | `acquisition/` | `appendix/acquisition-contracts-and-compatibility.md` | planned (review required) |
| `acquisition.md` | 288–308 | 2 | Phase 5 — outcome-constrained BO composition (source inspected) | `audit/` | `appendix/acquisition-contracts-and-compatibility.md` | planned (review required) |
| `acquisition.md` | 309–345 | 3 | Interpretation and boundaries | `acquisition/` | `appendix/acquisition-contracts-and-compatibility.md` | planned (review required) |
| `acquisition.md` | 346–362 | 2 | Phase 5 — information-theoretic and lookahead acquisitions (source inspected) | `audit/` | `appendix/acquisition-contracts-and-compatibility.md` | planned (review required) |
| `acquisition.md` | 363–405 | 3 | One-shot and information-target boundaries | `acquisition/` | `appendix/acquisition-contracts-and-compatibility.md` | planned (review required) |
| `acquisition.md` | 406–422 | 2 | Phase 5 — coverage reconciliation and exit audit | `audit/` | `appendix/acquisition-contracts-and-compatibility.md` | planned (review required) |
| `acquisition.md` | 423–465 | 3 | Confirmed limits and carry-forward | `acquisition/` | `appendix/acquisition-contracts-and-compatibility.md` | planned (review required) |
| `optimization.md` | 1–4 | 1 | Acquisition optimization theory–implementation mapping: Phase 3 inventory | `audit/` | `appendix/optimizer-dispatch-and-constraints.md` | planned (review required) |
| `optimization.md` | 5–19 | 2 | Baseline and separation of responsibilities | `optimization/` | `appendix/optimizer-dispatch-and-constraints.md` | planned (review required) |
| `optimization.md` | 20–39 | 2 | Named optimization dispatcher | `optimization/` | `appendix/optimizer-dispatch-and-constraints.md` | planned (review required) |
| `optimization.md` | 40–62 | 2 | Candidate-space constraints versus outcome constraints | `optimization/` | `appendix/optimizer-dispatch-and-constraints.md` | planned (review required) |
| `optimization.md` | 63–76 | 2 | Stateful and coordinate-changing algorithms | `optimization/` | `appendix/optimizer-dispatch-and-constraints.md` | planned (review required) |
| `optimization.md` | 77–85 | 2 | Verification boundaries | `optimization/` | `appendix/optimizer-dispatch-and-constraints.md` | planned (review required) |
| `optimization.md` | 86–99 | 2 | Inspected optimizer and strategy assertion evidence | `optimization/` | `appendix/optimizer-dispatch-and-constraints.md` | planned (review required) |
| `optimization.md` | 100–107 | 2 | Fixed follow-up audit surface | `audit/` | `appendix/optimizer-dispatch-and-constraints.md` | planned (review required) |
| `optimization.md` | 108–118 | 2 | Specialized optimization paths: inspected source and assertions | `optimization/` | `appendix/optimizer-dispatch-and-constraints.md` | planned (review required) |
| `optimization.md` | 119–133 | 3 | Coordinate-changing high-dimensional strategies | `optimization/` | `appendix/optimizer-dispatch-and-constraints.md` | planned (review required) |
| `optimization.md` | 134–140 | 3 | Remaining Phase 3 audit surface | `audit/` | `appendix/optimizer-dispatch-and-constraints.md` | planned (review required) |
| `optimization.md` | 141–152 | 2 | HeSBO, latent-space reconstruction and backend constraint semantics | `optimization/` | `appendix/optimizer-dispatch-and-constraints.md` | planned (review required) |
| `optimization.md` | 153–165 | 3 | Phase 3 evidence limits | `audit/` | `appendix/optimizer-dispatch-and-constraints.md` | planned (review required) |
| `optimization.md` | 166–181 | 2 | Phase 3 registry reconciliation gate | `audit/` | `appendix/optimizer-dispatch-and-constraints.md` | planned (review required) |
| `optimization.md` | 182–205 | 2 | Phase 6 — named optimizer mathematical and dispatch mapping | `audit/` | `appendix/optimizer-dispatch-and-constraints.md` | planned (review required) |
| `optimization.md` | 206–245 | 3 | Exact dispatch and mathematical limitations | `optimization/` | `appendix/optimizer-dispatch-and-constraints.md` | planned (review required) |
| `optimization.md` | 246–257 | 2 | Phase 6 — specialist optimizers and candidate-constraint semantics | `audit/` | `appendix/optimizer-dispatch-and-constraints.md` | planned (review required) |
| `optimization.md` | 258–274 | 3 | Constraint handling is part of the algorithm | `optimization/` | `appendix/optimizer-dispatch-and-constraints.md` | planned (review required) |
| `optimization.md` | 275–295 | 3 | Tests and mathematical verification boundary | `optimization/` | `appendix/optimizer-dispatch-and-constraints.md` | planned (review required) |
| `optimization.md` | 296–314 | 2 | Phase 6 — stateful and coordinate-changing search strategies | `audit/` | `appendix/optimizer-dispatch-and-constraints.md` | planned (review required) |
| `optimization.md` | 315–355 | 3 | Coordinate and constraint invariants | `optimization/` | `appendix/optimizer-dispatch-and-constraints.md` | planned (review required) |
| `verification.md` | 1–3 | 1 | Mathematical verification: Phases 7–9 | `verification/` | `audit/` | planned (review required) |
| `verification.md` | 4–26 | 2 | Purpose and evidence rules | `verification/` | `audit/` | planned (review required) |
| `verification.md` | 27–39 | 2 | Phase 7 — model verification matrix | `verification/` | `audit/` | planned (review required) |
| `verification.md` | 40–64 | 3 | Initial Phase 7 execution gate | `verification/` | `audit/` | planned (review required) |
| `verification.md` | 65–79 | 3 | Kronecker block-design covariance reference (Phase 7) | `verification/` | `audit/` | planned (review required) |
| `verification.md` | 80–93 | 3 | Data-fidelity downsampling kernel reference (Phase 7) | `verification/` | `audit/` | planned (review required) |
| `verification.md` | 94–107 | 3 | PCA reduced-space projection reference (Phase 7) | `verification/` | `audit/` | planned (review required) |
| `verification.md` | 108–121 | 3 | Replicate observation-noise statistical reference (Phase 7) | `verification/` | `audit/` | planned (review required) |
| `verification.md` | 122–134 | 3 | Binary GP probit predictive reference (Phase 7) | `verification/` | `audit/` | planned (review required) |
| `verification.md` | 135–148 | 3 | Non-GP Gaussian posterior reparameterization reference (Phase 7) | `verification/` | `audit/` | planned (review required) |
| `verification.md` | 149–183 | 2 | Phase 7 execution reconciliation and remaining scope | `verification/` | `audit/` | planned (review required) |
| `verification.md` | 184–197 | 3 | Reduced GP public/latent posterior coordinate reference (Phase 7) | `verification/` | `audit/` | planned (review required) |
| `verification.md` | 198–211 | 3 | Heteroskedastic latent-versus-observation noise reference (Phase 7) | `verification/` | `audit/` | planned (review required) |
| `verification.md` | 212–222 | 3 | Non-GP Random Forest estimator-level reference (Phase 7) | `verification/` | `audit/` | planned (review required) |
| `verification.md` | 223–244 | 2 | Phase 7 exit audit (property-specific) | `verification/` | `audit/` | planned (review required) |
| `verification.md` | 245–256 | 3 | Mixed reduced GP categorical-coordinate reference (Phase 7) | `verification/` | `audit/` | planned (review required) |
| `verification.md` | 257–273 | 3 | Heteroskedastic observation-variance composition boundary (Phase 7) | `verification/` | `audit/` | planned (review required) |
| `verification.md` | 274–291 | 2 | Phase 7 final exit reconciliation | `verification/` | `audit/` | planned (review required) |
| `verification.md` | 292–306 | 2 | Phase 8 — regression active-learning formula reference | `verification/` | `audit/` | planned (review required) |
| `verification.md` | 307–319 | 3 | Phase 8 — classification active-learning arithmetic reference | `verification/` | `audit/` | planned (review required) |
| `verification.md` | 320–334 | 3 | Phase 8 — BALD and probability-variance sample-moment reference | `verification/` | `audit/` | planned (review required) |
| `verification.md` | 335–350 | 3 | Phase 8 — native analytic BO Gaussian reference | `verification/` | `audit/` | planned (review required) |
| `verification.md` | 351–362 | 3 | Phase 8 — Monte Carlo qSimpleRegret sample reference | `verification/` | `audit/` | planned (review required) |
| `verification.md` | 363–375 | 3 | Phase 8 — Monte Carlo qUCB sample reference | `verification/` | `audit/` | planned (review required) |
| `verification.md` | 376–407 | 2 | Phase 8 — verification reconciliation (representative scope) | `verification/` | `audit/` | planned (review required) |
| `verification.md` | 408–421 | 2 | Phase 9 — sampling optimizer reference (initial property) | `verification/` | `audit/` | planned (review required) |
| `verification.md` | 422–435 | 3 | Phase 9 — TuRBO state-transition reference | `verification/` | `audit/` | planned (review required) |
| `verification.md` | 436–448 | 3 | Phase 9 — TuRBO trust-region geometry reference | `verification/` | `audit/` | planned (review required) |
| `verification.md` | 449–464 | 3 | Phase 9 — candidate constraint residual and feasibility ordering reference | `verification/` | `audit/` | planned (review required) |
| `verification.md` | 465–475 | 3 | Phase 9 — PyTorch optimizer candidate-constraint support boundary | `verification/` | `audit/` | planned (review required) |
| `verification.md` | 476–510 | 2 | Phase 9 — representative optimizer verification exit audit | `verification/` | `audit/` | planned (review required) |

## 残る監査

1. 各見出しの本文・数式・表・コード参照を人手相当の意味レビューで照合し、機械割当を修正する。
2. 既存の相対リンクと外部からの被参照、CI依存を確認する。
3. テスト・Notebook・実装パスの存在と意味論を確認する。
4. 検証後にAppendixの最終名称・責務をPhase 2で確定する。

## models.md 全91区間の証拠密度監査（追加レビュー）

見出し単位で本文を抽出し、`tests/*.py`、`src/robotorchan/*.py`、`examples/notebooks/*.ipynb` の明示参照を数えた。**参照数は記載数（重複除外）であり、ファイル存在・テスト成功・数理的正しさを示さない**。省略形の実装パスはこの表に含まれない。

- テスト参照を含む区間：**54/91**。
- 完全修飾src参照を含む区間：**16/91**。
- Notebook参照を含む区間：**0/91**。
- 監査/Phase系の見出しでもテスト・src参照を含む区間：**26**。これらは単純にauditへ移動せず、技術情報を本文/Appendixに残す必要がある。

| Source lines | Explicit test refs | Explicit src refs | Explicit notebook refs | Review requirement |
| --- | ---: | ---: | ---: | --- |
| 1–4 | 0 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 5–17 | 0 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 18–31 | 0 | 1 | 0 | 技術情報を本文/Appendixへ抽出（auditのみ禁止） |
| 32–51 | 0 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 52–91 | 0 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 92–106 | 0 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 107–119 | 0 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 120–131 | 0 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 132–146 | 0 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 147–185 | 0 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 186–207 | 5 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 208–225 | 0 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 226–235 | 0 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 236–256 | 0 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 257–271 | 0 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 272–289 | 0 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 290–315 | 0 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 316–338 | 4 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 339–350 | 0 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 351–367 | 0 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 368–393 | 0 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 394–422 | 0 | 1 | 0 | 章別/Appendixを本文レビューで確定 |
| 423–441 | 0 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 442–458 | 0 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 459–477 | 2 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 478–497 | 2 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 498–518 | 0 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 519–540 | 0 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 541–563 | 0 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 564–588 | 0 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 589–606 | 0 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 607–625 | 2 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 626–649 | 3 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 650–675 | 0 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 676–697 | 3 | 0 | 0 | 技術情報を本文/Appendixへ抽出（auditのみ禁止） |
| 698–723 | 2 | 0 | 0 | 技術情報を本文/Appendixへ抽出（auditのみ禁止） |
| 724–746 | 0 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 747–773 | 0 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 774–798 | 0 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 799–806 | 0 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 807–821 | 1 | 1 | 0 | 章別/Appendixを本文レビューで確定 |
| 822–836 | 1 | 1 | 0 | 章別/Appendixを本文レビューで確定 |
| 837–851 | 1 | 1 | 0 | 章別/Appendixを本文レビューで確定 |
| 852–866 | 1 | 1 | 0 | 章別/Appendixを本文レビューで確定 |
| 867–881 | 1 | 1 | 0 | 章別/Appendixを本文レビューで確定 |
| 882–896 | 1 | 1 | 0 | 章別/Appendixを本文レビューで確定 |
| 897–911 | 2 | 1 | 0 | 章別/Appendixを本文レビューで確定 |
| 912–926 | 1 | 1 | 0 | 章別/Appendixを本文レビューで確定 |
| 927–941 | 1 | 1 | 0 | 章別/Appendixを本文レビューで確定 |
| 942–956 | 1 | 1 | 0 | 章別/Appendixを本文レビューで確定 |
| 957–971 | 1 | 1 | 0 | 章別/Appendixを本文レビューで確定 |
| 972–986 | 1 | 1 | 0 | 章別/Appendixを本文レビューで確定 |
| 987–1000 | 1 | 1 | 0 | 章別/Appendixを本文レビューで確定 |
| 1001–1014 | 1 | 1 | 0 | 章別/Appendixを本文レビューで確定 |
| 1015–1028 | 1 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 1029–1042 | 1 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 1043–1056 | 1 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 1057–1070 | 1 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 1071–1084 | 1 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 1085–1099 | 1 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 1100–1131 | 3 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 1132–1168 | 2 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 1169–1206 | 0 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 1207–1224 | 0 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 1225–1242 | 1 | 0 | 0 | 技術情報を本文/Appendixへ抽出（auditのみ禁止） |
| 1243–1269 | 3 | 0 | 0 | 技術情報を本文/Appendixへ抽出（auditのみ禁止） |
| 1270–1293 | 3 | 0 | 0 | 技術情報を本文/Appendixへ抽出（auditのみ禁止） |
| 1294–1321 | 2 | 0 | 0 | 技術情報を本文/Appendixへ抽出（auditのみ禁止） |
| 1322–1344 | 2 | 0 | 0 | 技術情報を本文/Appendixへ抽出（auditのみ禁止） |
| 1345–1369 | 2 | 0 | 0 | 技術情報を本文/Appendixへ抽出（auditのみ禁止） |
| 1370–1389 | 4 | 0 | 0 | 技術情報を本文/Appendixへ抽出（auditのみ禁止） |
| 1390–1404 | 3 | 0 | 0 | 技術情報を本文/Appendixへ抽出（auditのみ禁止） |
| 1405–1420 | 1 | 0 | 0 | 技術情報を本文/Appendixへ抽出（auditのみ禁止） |
| 1421–1437 | 1 | 0 | 0 | 技術情報を本文/Appendixへ抽出（auditのみ禁止） |
| 1438–1460 | 0 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 1461–1477 | 2 | 0 | 0 | 技術情報を本文/Appendixへ抽出（auditのみ禁止） |
| 1478–1493 | 2 | 0 | 0 | 技術情報を本文/Appendixへ抽出（auditのみ禁止） |
| 1494–1516 | 1 | 0 | 0 | 技術情報を本文/Appendixへ抽出（auditのみ禁止） |
| 1517–1530 | 1 | 0 | 0 | 技術情報を本文/Appendixへ抽出（auditのみ禁止） |
| 1531–1547 | 3 | 0 | 0 | 技術情報を本文/Appendixへ抽出（auditのみ禁止） |
| 1548–1564 | 3 | 0 | 0 | 技術情報を本文/Appendixへ抽出（auditのみ禁止） |
| 1565–1581 | 3 | 0 | 0 | 技術情報を本文/Appendixへ抽出（auditのみ禁止） |
| 1582–1600 | 2 | 0 | 0 | 技術情報を本文/Appendixへ抽出（auditのみ禁止） |
| 1601–1620 | 2 | 0 | 0 | 技術情報を本文/Appendixへ抽出（auditのみ禁止） |
| 1621–1637 | 2 | 0 | 0 | 技術情報を本文/Appendixへ抽出（auditのみ禁止） |
| 1638–1656 | 4 | 0 | 0 | 技術情報を本文/Appendixへ抽出（auditのみ禁止） |
| 1657–1672 | 2 | 0 | 0 | 技術情報を本文/Appendixへ抽出（auditのみ禁止） |
| 1673–1687 | 1 | 0 | 0 | 技術情報を本文/Appendixへ抽出（auditのみ禁止） |
| 1688–1734 | 0 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 1735–1785 | 0 | 0 | 0 | 章別/Appendixを本文レビューで確定 |
| 1786–1796 | 1 | 0 | 0 | 章別/Appendixを本文レビューで確定 |

**重要：** 上記は全91区間の証拠参照密度に基づく分類支援であり、各区間の技術的意味の完全レビューではない。既存の174見出しの暫定移行先を`verified`へ昇格させない。

## 残り4ファイル・全83見出しの証拠密度監査

この節は`README.md`、`acquisition.md`、`optimization.md`、`verification.md`のH1–H3全見出しを対象とした**本文中の明示パス抽出**。参照が0でも根拠がないことは意味しない。監査見出しの中の技術情報は、auditに隔離せず章別ガイド・Appendix・verificationへ保全する。

| File | Headings | Test-ref sections | Src-ref sections | Notebook-ref sections | History-like headings with technical refs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `README.md` | 13 | 1 | 1 | 0 | 1 |
| `acquisition.md` | 18 | 10 | 6 | 1 | 6 |
| `optimization.md` | 21 | 8 | 3 | 0 | 3 |
| `verification.md` | 31 | 22 | 0 | 0 | 22 |

### README.md

| Lines | Heading | Tests | Src | Notebook | Review |
| --- | --- | ---: | ---: | ---: | --- |
| 1–2 | Theory–Implementation Mapping: 12-phase development cycle | 0 | 0 | 0 | 本文レビューが必要 |
| 3–10 | Baseline and scope | 0 | 0 | 0 | 本文レビューが必要 |
| 11–23 | Existing ownership (preserve) | 0 | 0 | 0 | 本文レビューが必要 |
| 24–41 | Confirmed source-of-truth boundaries | 0 | 4 | 0 | 本文レビューが必要 |
| 42–66 | Reader navigation (Phase 10) | 0 | 0 | 0 | 本文レビューが必要 |
| 67–86 | Mapping record contract | 0 | 0 | 0 | 本文レビューが必要 |
| 87–94 | Mapping documentation layout | 0 | 0 | 0 | 本文レビューが必要 |
| 95–107 | Verification and remaining work | 0 | 0 | 0 | 本文レビューが必要 |
| 108–136 | Recorded phase decisions | 0 | 0 | 0 | 本文レビューが必要 |
| 137–174 | Phase 6 exit reconciliation (static scope) | 0 | 0 | 0 | 本文レビューが必要 |
| 175–201 | Phase 10 navigation exit audit | 0 | 0 | 0 | 本文レビューが必要 |
| 202–231 | Phase 11 mapping consistency exit audit | 1 | 0 | 0 | 技術情報の抽出が必要 |
| 232–278 | Phase 12 final audit: evidence and decision | 0 | 0 | 0 | 本文レビューが必要 |

### acquisition.md

| Lines | Heading | Tests | Src | Notebook | Review |
| --- | --- | ---: | ---: | ---: | --- |
| 1–4 | Acquisition theory–implementation mapping: Phase 3 inventory | 0 | 0 | 0 | 本文レビューが必要 |
| 5–18 | Sources and evidence boundary | 0 | 2 | 0 | 技術情報の抽出が必要 |
| 19–38 | Distinct implementation paths | 0 | 0 | 0 | 本文レビューが必要 |
| 39–47 | Acquisition/optimizer interaction | 0 | 0 | 0 | 本文レビューが必要 |
| 48–60 | Fixed Phase 3 completion criteria | 0 | 0 | 0 | 本文レビューが必要 |
| 61–78 | Inspected acquisition registry and assertion evidence | 2 | 0 | 0 | 技術情報の抽出が必要 |
| 79–114 | Phase 5 — regression active-learning acquisition mappings (source inspected) | 5 | 0 | 1 | 技術情報の抽出が必要 |
| 115–143 | Phase 5 — classification AL and Gaussian EPIG (source inspected) | 3 | 0 | 0 | 技術情報の抽出が必要 |
| 144–178 | ExpectedPredictiveInformationGain (regression) | 1 | 0 | 0 | 本文レビューが必要 |
| 179–232 | Phase 5 — single-objective Monte Carlo BO (source inspected) | 3 | 3 | 0 | 技術情報の抽出が必要 |
| 233–249 | Phase 5 — multi-objective MC Bayesian optimization (source inspected) | 0 | 0 | 0 | 本文レビューが必要 |
| 250–287 | Verified construction contracts | 3 | 2 | 0 | 本文レビューが必要 |
| 288–308 | Phase 5 — outcome-constrained BO composition (source inspected) | 0 | 0 | 0 | 本文レビューが必要 |
| 309–345 | Interpretation and boundaries | 3 | 2 | 0 | 本文レビューが必要 |
| 346–362 | Phase 5 — information-theoretic and lookahead acquisitions (source inspected) | 1 | 0 | 0 | 技術情報の抽出が必要 |
| 363–405 | One-shot and information-target boundaries | 3 | 2 | 0 | 本文レビューが必要 |
| 406–422 | Phase 5 — coverage reconciliation and exit audit | 0 | 0 | 0 | 本文レビューが必要 |
| 423–465 | Confirmed limits and carry-forward | 2 | 2 | 0 | 本文レビューが必要 |

### optimization.md

| Lines | Heading | Tests | Src | Notebook | Review |
| --- | --- | ---: | ---: | ---: | --- |
| 1–4 | Acquisition optimization theory–implementation mapping: Phase 3 inventory | 0 | 0 | 0 | 本文レビューが必要 |
| 5–19 | Baseline and separation of responsibilities | 0 | 0 | 0 | 本文レビューが必要 |
| 20–39 | Named optimization dispatcher | 0 | 0 | 0 | 本文レビューが必要 |
| 40–62 | Candidate-space constraints versus outcome constraints | 0 | 0 | 0 | 本文レビューが必要 |
| 63–76 | Stateful and coordinate-changing algorithms | 0 | 0 | 0 | 本文レビューが必要 |
| 77–85 | Verification boundaries | 0 | 0 | 0 | 本文レビューが必要 |
| 86–99 | Inspected optimizer and strategy assertion evidence | 3 | 0 | 0 | 技術情報の抽出が必要 |
| 100–107 | Fixed follow-up audit surface | 0 | 0 | 0 | 本文レビューが必要 |
| 108–118 | Specialized optimization paths: inspected source and assertions | 3 | 0 | 0 | 本文レビューが必要 |
| 119–133 | Coordinate-changing high-dimensional strategies | 3 | 0 | 0 | 本文レビューが必要 |
| 134–140 | Remaining Phase 3 audit surface | 0 | 0 | 0 | 本文レビューが必要 |
| 141–152 | HeSBO, latent-space reconstruction and backend constraint semantics | 3 | 0 | 0 | 本文レビューが必要 |
| 153–165 | Phase 3 evidence limits | 0 | 0 | 0 | 本文レビューが必要 |
| 166–181 | Phase 3 registry reconciliation gate | 2 | 0 | 0 | 技術情報の抽出が必要 |
| 182–205 | Phase 6 — named optimizer mathematical and dispatch mapping | 0 | 1 | 0 | 技術情報の抽出が必要 |
| 206–245 | Exact dispatch and mathematical limitations | 2 | 1 | 0 | 本文レビューが必要 |
| 246–257 | Phase 6 — specialist optimizers and candidate-constraint semantics | 0 | 0 | 0 | 本文レビューが必要 |
| 258–274 | Constraint handling is part of the algorithm | 0 | 0 | 0 | 本文レビューが必要 |
| 275–295 | Tests and mathematical verification boundary | 5 | 0 | 0 | 本文レビューが必要 |
| 296–314 | Phase 6 — stateful and coordinate-changing search strategies | 0 | 0 | 0 | 本文レビューが必要 |
| 315–355 | Coordinate and constraint invariants | 4 | 2 | 0 | 本文レビューが必要 |

### verification.md

| Lines | Heading | Tests | Src | Notebook | Review |
| --- | --- | ---: | ---: | ---: | --- |
| 1–3 | Mathematical verification: Phases 7–9 | 0 | 0 | 0 | 本文レビューが必要 |
| 4–26 | Purpose and evidence rules | 0 | 0 | 0 | 本文レビューが必要 |
| 27–39 | Phase 7 — model verification matrix | 0 | 0 | 0 | 本文レビューが必要 |
| 40–64 | Initial Phase 7 execution gate | 0 | 0 | 0 | 本文レビューが必要 |
| 65–79 | Kronecker block-design covariance reference (Phase 7) | 1 | 0 | 0 | 技術情報の抽出が必要 |
| 80–93 | Data-fidelity downsampling kernel reference (Phase 7) | 1 | 0 | 0 | 技術情報の抽出が必要 |
| 94–107 | PCA reduced-space projection reference (Phase 7) | 1 | 0 | 0 | 技術情報の抽出が必要 |
| 108–121 | Replicate observation-noise statistical reference (Phase 7) | 1 | 0 | 0 | 技術情報の抽出が必要 |
| 122–134 | Binary GP probit predictive reference (Phase 7) | 1 | 0 | 0 | 技術情報の抽出が必要 |
| 135–148 | Non-GP Gaussian posterior reparameterization reference (Phase 7) | 1 | 0 | 0 | 技術情報の抽出が必要 |
| 149–183 | Phase 7 execution reconciliation and remaining scope | 0 | 0 | 0 | 本文レビューが必要 |
| 184–197 | Reduced GP public/latent posterior coordinate reference (Phase 7) | 1 | 0 | 0 | 技術情報の抽出が必要 |
| 198–211 | Heteroskedastic latent-versus-observation noise reference (Phase 7) | 1 | 0 | 0 | 技術情報の抽出が必要 |
| 212–222 | Non-GP Random Forest estimator-level reference (Phase 7) | 1 | 0 | 0 | 技術情報の抽出が必要 |
| 223–244 | Phase 7 exit audit (property-specific) | 0 | 0 | 0 | 本文レビューが必要 |
| 245–256 | Mixed reduced GP categorical-coordinate reference (Phase 7) | 1 | 0 | 0 | 技術情報の抽出が必要 |
| 257–273 | Heteroskedastic observation-variance composition boundary (Phase 7) | 1 | 0 | 0 | 技術情報の抽出が必要 |
| 274–291 | Phase 7 final exit reconciliation | 0 | 0 | 0 | 本文レビューが必要 |
| 292–306 | Phase 8 — regression active-learning formula reference | 1 | 0 | 0 | 技術情報の抽出が必要 |
| 307–319 | Phase 8 — classification active-learning arithmetic reference | 1 | 0 | 0 | 技術情報の抽出が必要 |
| 320–334 | Phase 8 — BALD and probability-variance sample-moment reference | 1 | 0 | 0 | 技術情報の抽出が必要 |
| 335–350 | Phase 8 — native analytic BO Gaussian reference | 1 | 0 | 0 | 技術情報の抽出が必要 |
| 351–362 | Phase 8 — Monte Carlo qSimpleRegret sample reference | 1 | 0 | 0 | 技術情報の抽出が必要 |
| 363–375 | Phase 8 — Monte Carlo qUCB sample reference | 1 | 0 | 0 | 技術情報の抽出が必要 |
| 376–407 | Phase 8 — verification reconciliation (representative scope) | 0 | 0 | 0 | 本文レビューが必要 |
| 408–421 | Phase 9 — sampling optimizer reference (initial property) | 1 | 0 | 0 | 技術情報の抽出が必要 |
| 422–435 | Phase 9 — TuRBO state-transition reference | 1 | 0 | 0 | 技術情報の抽出が必要 |
| 436–448 | Phase 9 — TuRBO trust-region geometry reference | 1 | 0 | 0 | 技術情報の抽出が必要 |
| 449–464 | Phase 9 — candidate constraint residual and feasibility ordering reference | 1 | 0 | 0 | 技術情報の抽出が必要 |
| 465–475 | Phase 9 — PyTorch optimizer candidate-constraint support boundary | 1 | 0 | 0 | 技術情報の抽出が必要 |
| 476–510 | Phase 9 — representative optimizer verification exit audit | 0 | 0 | 0 | 本文レビューが必要 |

### Phase 1の残ゲート（未完了）

- 全リポジトリのtreeを用いた参照先の網羅的な存在確認と外部からの被参照チェック。
- 見出し内の数式・表・注意書き・暗黙のパスを含めた意味レベルの保全レビュー。
- 各見出しの移行先（primary/secondary）を仮分類から確定に昇格するレビュー。
- リンク検証スクリプトとCIの実行。

**現時点ではPhase 1を完了と判定しない。**

## verification.md: 数値的検証条件の意味レベル移行レビュー（第1弾）

下表は`verification.md`の本文を実際に読み、理論との対応、テストが示す性質、**検証できない範囲**を明示したもの。単なるパス抽出ではない。元の本文は移行完了まで保持する。各テストの**現時点の実行結果は未確認**（原文に記載された過去のCI成功とは区別）。

| 元行 | 対応する技術ガイド | テスト・数値条件 | 確認する性質 | 保全すべき限界 |
| --- | --- | --- | --- | --- |
| 65–79 | multitask / Kronecker GP | `test_phase7_kronecker_covariance_reference.py`; float64, 3入力×2タスク, 1e-10 | `K[i*m+a,j*m+b]=K_input[i,j]*K_task[a,b]`、対称性、半正定値性、タスク置換 | 潜在事前共分散のみ。事後条件付け・学習済みパラメータ・観測雑音は対象外 |
| 80–93 | multi-fidelity GP | `test_phase7_multifidelity_kernel_reference.py`; float64, 1e-10 | データ忠実度kernelの高忠実度境界、残差rank-one、対称性・半正定値性 | 完全な多忠実度事後・linear-truncated kernel・学習は対象外 |
| 292–306 | regression AL acquisition | `test_phase8_regression_al_formula_reference.py`; CPU float64, q=1, 1e-12 | variance/std/straddle/boundary varianceの事後モーメント演算 | GP事後計算・最適化・他獲得関数は対象外 |
| 307–319 | classification AL acquisition | `test_phase8_classification_al_reference.py`; binary, q=1, 1e-10 | entropy/margin/latent straddle、ラベル確率とlatent不確実性の区別 | 較正・変分学習・BALDのMC精度・多クラスは対象外 |
| 320–334 | classification AL / information gain | `test_phase8_bald_probability_variance_reference.py`; 64 samples, CPU float64, 1e-10 | BALDのエントロピー差とProbabilityVarianceの母分散集約 | MC収束・較正・GP学習の忠実性は対象外 |
| 335–350 | analytic BO acquisition | `test_phase8_analytic_bo_gaussian_reference.py`; q=1, float64, 1e-8 | logEI/logPI/UCBの正規分布閉形式との整合 | GP条件付け・noisy/MC・制約・極端な裾の数値安定性は対象外 |
| 408–421 | optimizer sampling/dispatch | `test_phase9_sampling_optimizer_reference.py`; q=2, 128候補, float64 | random/sobol候補集合内のargmaxとdispatch | 連続空間の大域収束・制約・mixed対応は対象外 |
| 422–435 | TuRBO state | `test_phase9_turbo_state_reference.py`; deterministic float64 | 成功時拡張、失敗時縮小、restart、観測とdenoised値の区別 | regret・ARD幾何・非同期・restart点選択は対象外 |
| 436–448 | TuRBO geometry | `test_phase9_turbo_geometry_reference.py`; 3D, float64, 1e-12 | 幾何平均ARD正規化、境界clipping、ARD共通倍率不変性 | ARD学習・制約付き最適化・収束は対象外 |
| 449–464 | candidate constraints / optimizer | `test_phase9_constraint_violation_reference.py`; q=2, float64, 1e-12 | 線形/非線形、点内/点間の残差とDeb-style feasibility-first順序 | optimizerによる実際の制約充足・outcome constraints・収束は対象外 |
| 465–475 | PyTorch optimizer support matrix | `test_phase9_torch_constraint_contract.py`; Adam/AdamW/SGD | 未対応candidate constraintsの明示的拒否（negative contract） | 制約付き最適化ができることは証明しない |

### 移行上の判定

- 数式・条件・許容誤差・検証限界は各技術章の「検証範囲」に**必ず残す**。詳細なテストケース表は`verification/`またはAppendixに配置する。
- Phase番号やPR完了報告は監査履歴に置き、学習本文には入れない。ただしテストファイル名自体に含まれる`phase7`等は実在の識別子なので改名しない。
- `verification.md`の他の区間はまだ意味レベルレビューが未完了。ここにある11区間だけをレビュー済みとする。

## verification.md: 数値的検証条件の意味レベル移行レビュー（第2弾）

前節に続き、`verification.md`の未レビューの数値検証本文を読み、モデル系8区間とacquisitionの総括・optimizerの総括をレビューした。原文の過去のCI/PR記述は**歴史的証拠**であり、今回の作業でCIを再実行したことは意味しない。

| 元行 | 技術ガイド候補 | 実証する性質・条件 | 引き継ぐべき未検証境界 |
| --- | --- | --- | --- |
| 94–107 | reduced / PCA GP | `test_phase7_pca_projection_reference.py`：6×3 float64、共分散固有値・直交性・射影・残差直交性、1e-10 | PCAの基底符号は非固定。reduced GP事後、PLS、neural reducerは対象外 |
| 108–121 | robust/noise / replicate | `test_phase7_replicate_noise_reference.py`：不均等replicate群、標本分散`s²=Σ(y-ȳ)²/(n-1)`と群平均の雑音分散`s²/n` | 学習済みheteroskedastic noiseやrobust GPの精度は対象外 |
| 122–134 | binary classification | `test_phase7_binary_probit_reference.py`：`E[Φ(f)]=Φ(μ/√(1+σ²))`、float64、確率の正規化 | ELBO収束・較正・latent GP fitは対象外 |
| 135–148 | non-GP posterior adapter | `test_phase7_gaussian_posterior_reference.py`：`y=μ+√v ε`、base sample、batch/sample形状、零分散、平均への勾配 | 外部推定器の予測精度は対象外 |
| 184–197 | reduced GP coordinate API | `test_phase7_posterior_coordinate_reference.py`：`posterior_original(x)=posterior_reduced(T(x))`、固定PCA、q-batch、reducer不変 | 次元削減の有効性・カテゴリ保持は対象外 |
| 198–211 | heteroskedastic GP | `test_phase7_heteroskedastic_uncertainty_reference.py`：制御されたlog-noiseの指数変換とlatent/noise分離、0.01/0.09 | noise GP較正・任意点での観測事後は対象外 |
| 212–222 | non-GP / random forest | `test_phase7_forest_estimator_reference.py`：sklearn tree個別予測、標本分散`s-1`、q-batch、random_state=17 | 汎化精度・epistemic較正・他の非GPモデルは対象外 |
| 245–256 | mixed reduced GP | `test_phase7_mixed_reduced_categorical_reference.py`：連続/カテゴリ交互列、カテゴリID・reduced cat index保持、事後座標同値、1e-10 | MixedPCAGP固定PCA経路のみ。全mixed reducerへの一般化不可 |
| 257–273 | heteroskedastic observation | 同`test_phase7_heteroskedastic_uncertainty_reference.py`：条件付き独立雑音の手動合成`Var[y|D]=Var[f|D]+v(x)` | **`posterior(observation_noise=True)`の検証ではない**。候補依存noise注入やnoise GP不確実性積分は未対応/未検証 |

### 章別ガイドに必要な「非証明」記述

- `verification.md` 223–244と274–291のPhase 7監査は、PR #1426の最終head CIが当時未確認であったことや、カテゴリ座標・heteroskedastic観測事後の限界を明示する**歴史的な時点記録**。後続の#1428/#1429の記述を考慮しても、完全な観測事後の実装を検証済みに昇格させない。
- 376–407のPhase 8総括：独立数式参照で確認したのは代表的なAL・解析的BO・MC集約。**qNEI/qLogNEI、qEI/qLogEI、qEHVI/qNEHVI、制約付き獲得、異種合成、情報利得/lookaheadの数理的検証は別扱い**。
- 476–510のPhase 9総括：sampling pool argmax、TuRBO状態/ARD幾何、constraint residual、未対応制約の拒否を確認した歴史的判断であり、**大域収束、全backend制約充足、mixed/batch/asyncの保証は示さない**。
- これらの限界は章本文の「検証できていない性質」に残し、PR番号・Phase完了宣言はaudit側へ分離する。

### レビュー状態

今回のレビューは数値参照の具体的な意味と総括の限界を対象とする。初期execution gate・マトリクス・exit audit等の全記述を移行確定したわけではなく、**Phase 1完了判定は保留**。

## models.md: 技術内容と章構成の意味レベル監査（第1弾）

`models.md`の18–106、208–235、607–675、807–866行を本文ベースで読み、**技術本文・横断契約・歴史的監査証拠**を分離する移行判定を行った。以下はこの範囲に限った確定候補であり、全91見出しの最終移行確定ではない。

| 元行 | 移行先候補 | 章に残す技術内容 | Appendix/監査に残す内容 |
| --- | --- | --- | --- |
| 18–31 | Appendix: model registry / coverage semantics | 回帰registryと分類registryは異なるID体系。HeterogeneousModelは別扱い。分類exportのhelperをmodel IDに数えない | 142回帰entry・29分類IDは**監査時点の値**として保全し、現在値と混同しない |
| 32–91 | Appendix: model-family inventory | model→family→ユーザーガイドの分類、reduced 52/robust 27等の当時の対応 | 全142件の原表は削除せずAppendixに保管。正本`docs/model_coverage.json`との二重管理を避ける |
| 92–106 | standard GP / classification / reduced / non-GP | `SingleTaskGP`と`MixedSingleTaskGP`はBoTorchへGPを委譲。`BinarySingleTaskGPClassifier`はGPyTorch Bernoulli/ELBO。`ReducedGP`は変換をローカル実装。`RandomForestSurrogate`はsklearnを委譲 | Source-inspectedとruntime verifiedを区別するevidence legend |
| 208–235 | Appendix: ownership contract + 各モデル章 | **Delegated/Extendedはコンポーネント単位**。`posterior`・`fit`・`raw_train_X/Y`を分けて説明。RandomForestのruntime importにも注意 | 5例は代表監査であり171件の網羅的検証ではない。method-owner scanは数値検証ではない |
| 607–625 | non-GP regression | NGBoostはparametric Gaussian (`loc`,`scale²`)。ExtraTreesは木予測の経験分布。両者の不確実性はGP事後とは異なる | optional `ngboost`未導入時のskipと、較正/数値一致の未検証 |
| 626–649 | robust/noise + heterogeneous multitask | 交互fitのlog-noise GP、joint variational二潜在GP、Kronecker heteroskedastic prototype、mixed task conditional kernelを**別アルゴリズム**として記述 | finite 1-iterationは収束を示さない。prototypeの観測事後は未対応。テスト関数数は検証保証ではない |
| 650–675 | multi-fidelity GP | fidelity列とcategorical列の分離、BoTorch DownsamplingKernel保持、linear-truncated拒否、cost-aware KG接続 | upstream posterior一致は固定構成のみ。samplerのshape/gradientは統計的無偏性を示さない |
| 807–821 | standard exact GP | exact MLL・BoTorch posterior委譲とlocal fit/raw data contract | upstream equalityは検査した設定に限定 |
| 822–836 | mixed GP | native categorical kernel、`cat_dims`負index、one-hot/順序距離の誤解を避ける | kernel/optimizerの全組合せ検証なし |
| 837–851 | classification GP | `posterior`はlatent、`predict_proba`はlabel probability、variational ELBO | calibration・fit収束・実行Notebookは未検証 |
| 852–866 | multitask GP | long-format task featureとblock-design Kroneckerを混同しない | task相関の数値同定・全task coverageは未検証 |

### 横断的な設計原則（技術ガイド本文へ反映）

1. `Implemented`等のstatusは**モデル名一括ではなくアルゴリズム構成要素単位**で説明する。ローカルsubclassの存在だけでGP posteriorを独自実装と判定しない。
2. 分類のlatent uncertainty、predictive class probability、label uncertaintyは別物として扱う。Regressionと同じ`posterior`という名称だけで確率の意味を混同しない。
3. Heteroskedastic、non-GP、multi-fidelityは、単にBoTorch互換APIがあることと、数理的に同じposteriorがあることを区別する。
4. 既存の英語表とregistry全件リストはAppendixへ保全し、日本語の章本文には学習上必要な説明を再構成する。Phase/PR番号はauditへ分離する。

**保留:** 原文のregistry件数や過去のテスト状況は今回再計算・再実行していないため、「現時点の実装保証」に書き換えない。既存5ファイルは維持し、CIを壊す移動・改名は行わない。

## models.md: 高次元・分類・表現学習の意味レベル監査（第2弾）

本文の498–588、676–773、1100–1168行を確認。理論章は原理の説明、技術ガイドはAPI・数値手法・委譲境界・検証限界、Appendixは全ID一覧と横断契約、auditは過去の実施記録を担当する。

| 元行 | 技術ガイドへの移行候補 | 絶対に失ってはいけない実装上の区別 | 限界・Appendix保全事項 |
| --- | --- | --- | --- |
| 498–518 | multitask classification / deep GP classification | **KroneckerMultiTaskBinaryGPClassifierはblock→long形式へ展開してvariational ICMを使う**。厳密Kronecker行列計算ではない。DeepGP分類はDeepApproximateMLL+Bernoulliでlatent sample平均 | タスク間相関の数値一致・ELBO/MC収束・較正は未検証 |
| 519–540 | high-dimensional binary classification | SAASとMAP-SAASのHalf-Cauchy inverse-ARD prior、ALEBO Mahalanobis metric、frozen PCA/PLS/random projectionの元/削減座標対応 | SAAS分類は**NUTS/Pyro完全Bayesianではない**。MAP最適化成功・次元削減の情報保持は未検証 |
| 541–563 | joint encoder / classification ensembles / uncertain input | encoderのELBO勾配、GP ensembleのclass probability経験分布、カテゴリ不確実性の有限support重み付き周辺化、Gibbs非定常は**latent covariance**の変更 | 勾配存在≠学習品質。ensemble variance≠較正。非定常≠label noise |
| 564–588 | robust classification / non-GP bootstrap | contamination `(1-c)p_clean+c p_contaminant`、replicate disagreement `2p(1-p)`、bootstrap推定器間分散、mixed Gibbs+categorical kernel | replicateの不一致はposterior varianceではない。bootstrap分散は較正保証ではない |
| 676–697 | expressive / DeepGP | `DeepGPPosterior`は**MC sample-backed**で解析的正規事後ではない。task embedding/categorical embeddingは局所実装。BoTorch獲得関数への接続 | finite qLogEI/optimize_acqfはregret・MC分布一致の証明ではない。legacy `DeepKernelGP` aliasは公開不可 |
| 698–723 | expressive / spectral & NNGP | spectral kernel計算はGPyTorchへ委譲、構造列/カテゴリとの合成はlocal。**InfiniteWidthReLUKernelのarc-cosine再帰はlocal実装** | Gram PSD数値チェックだけで公開理論式との同値性は証明されない |
| 724–746 | reduced / joint encoder / hybrid AE / VAE | joint encoderはGP MLLと同時最適化し**frozen reducerではない**。Hybridは再構成MSE、VAEはKL+stochastic reconstruction。mixedでは連続のみ削減 | VAE標準`posterior`はlatent平均経路、`uncertainty_aware_posterior`は別のMC moment-matched近似。後者の較正・収束未検証 |
| 747–773 | Appendix: classification registry documentation | 29分類IDは個別guide/theory/notebookメタデータを持たず、共通分類ガイドへ案内する | 共通理論章はsklearn/ensemble/robust変種の個別数理導出ではない。IDごとのNotebook実行未確認 |
| 1100–1131 | robust binary classification | replicateは個々の0/1ラベルで学習。input-dependent flipは`p_obs=FP+(1-FP-FN)p_clean`。Gibbs非定常はflip確率ではなくlatent kernel | identical floating rowsの要件、flip rate同定/較正未検証、mixed/multitaskの数値同値性未検証 |
| 1132–1168 | sparse/embedded binary classification | MAP-SAAS/SAASのpriorとNUTSを混同しない。ALEBO分類は**埋込座標Z**を入力としambient projectionはstrategy側。joint encoderはlossで特徴へ勾配伝播 | ALEBO ambient可行性・prior収束・joint encoder予測較正は未検証 |

### 検証状態と移行先のルール

- 上記は**既存Mapping本文の意味を保全するためのレビュー**であり、実装をこの作業で再実行したわけではない。
- `Implemented/Delegated/Extended/Approximated/Unsupported/Unverified`は構成要素ごとに付す。名称から数学的アルゴリズムを推測しない（特にKronecker分類、SAAS分類、VAEの事後）。
- 分類全29 ID・回帰全142 entryの歴史的registry表と根拠テストはAppendixに保存し、学習本文に巨大な監査表をそのまま埋め込まない。
- 元の`models.md`は引き続き保持する。**Phase 1の全内容レビューは未完了**。
