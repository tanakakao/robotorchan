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
