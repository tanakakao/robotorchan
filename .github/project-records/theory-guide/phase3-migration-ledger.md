# Phase 3 — 原文174区間から日本語ガイドへの移行割当台帳

> **現行判定（Phase 3）:** 174/174区間の本文確認・一次移行先割当が完了。`migrated` 0件、`verified` 0件。原文は未変更。本文レビュー完了はガイド執筆・数学的検証の完了を意味しない。

## 入力と判定基準

- 入力: Phase 1の174見出し台帳（5ファイル、元行範囲・元blob SHAは原台帳に保持）。
- 章構成: `docs/development/theory-implementation/guide/README.md`の24+12+12章。
- 一次移行先は**本文を読み進める主章**、二次移行先は横断的な技術契約や監査情報の保全先。
- 一次移行先は本文レビューに基づく**移行計画**。本文に複数テーマが混在する区間は後続の章執筆時に分割・二次参照を確定する。
- `verification.md`の全区間はverificationの章へ一次割当済み。propertyごとのモデル・獲得・最適化章への二次参照は後続の執筆で確定する。
- `Phase`と付いた見出しでも数式、テストassertion、未検証条件は技術ガイドまたはverificationに残す。
- `audit-history`は旧Phase/PR/CIの履歴を指す**分類ラベル**であり、新しいファイルが作成済みという意味ではない。

## 174区間の移行割当

| 原文 | 行範囲 | 見出し | 第一移行先（予定） | 第二移行先・保全分類 | 状態 |
| --- | --- | --- | --- | --- | --- |
| `README.md` | 1–2 | Theory–Implementation Mapping: 12-phase development cycle | `guide/README.md` | `audit-history` | **body-reviewed routing / not migrated** |
| `README.md` | 3–10 | Baseline and scope | `guide/README.md` | `audit-history` | **body-reviewed routing / not migrated** |
| `README.md` | 11–23 | Existing ownership (preserve) | `guide/README.md` | `guide/appendix/cross-cutting-implementation-contracts.md` | **body-reviewed routing / not migrated** |
| `README.md` | 24–41 | Confirmed source-of-truth boundaries | `guide/appendix/cross-cutting-implementation-contracts.md` | `guide/appendix/model-registry-and-public-api.md` | **body-reviewed routing / not migrated** |
| `README.md` | 42–66 | Reader navigation (Phase 10) | `guide/README.md` | `audit-history` | **body-reviewed routing / not migrated** |
| `README.md` | 67–86 | Mapping record contract | `guide/appendix/cross-cutting-implementation-contracts.md` | `guide/verification/README.md` | **body-reviewed routing / not migrated** |
| `README.md` | 87–94 | Mapping documentation layout | `guide/README.md` | `audit-history` | **body-reviewed routing / not migrated** |
| `README.md` | 95–107 | Verification and remaining work | `guide/verification/README.md` | `audit-history` | **body-reviewed routing / not migrated** |
| `README.md` | 108–136 | Recorded phase decisions | `audit-history` | `guide/appendix/limitations-and-unverified.md` | **body-reviewed routing / not migrated** |
| `README.md` | 137–174 | Phase 6 exit reconciliation (static scope) | `guide/appendix/optimizer-dispatch-and-constraints.md` | `audit-history` | **body-reviewed routing / not migrated** |
| `README.md` | 175–201 | Phase 10 navigation exit audit | `guide/README.md` | `audit-history` | **body-reviewed routing / not migrated** |
| `README.md` | 202–231 | Phase 11 mapping consistency exit audit | `guide/verification/README.md` | `audit-history` | **body-reviewed routing / not migrated** |
| `README.md` | 232–278 | Phase 12 final audit: evidence and decision | `guide/appendix/limitations-and-unverified.md` | `audit-history` | **body-reviewed routing / not migrated** |
| `models.md` | 1–4 | Model theory–implementation mapping | `guide/README.md` | `guide/appendix/model-registry-and-public-api.md` | **body-reviewed routing / not migrated** |
| `models.md` | 5–17 | Audit provenance and limits | `guide/appendix/model-registry-and-public-api.md` | `audit-history` | **body-reviewed routing / not migrated** |
| `models.md` | 18–31 | Registry boundaries | `guide/appendix/model-registry-and-public-api.md` | `guide/24_classification.md` | **body-reviewed routing / not migrated** |
| `models.md` | 32–51 | Regression model inventory by existing guide | `guide/appendix/model-registry-and-public-api.md` | `guide/13_model_selection.md` | **body-reviewed routing / not migrated** |
| `models.md` | 52–91 | Binary classification registry inventory | `guide/appendix/model-registry-and-public-api.md` | `guide/24_classification.md` | **body-reviewed routing / not migrated** |
| `models.md` | 92–106 | Representative code-path findings | `guide/appendix/model-composition-and-delegation.md` | `guide/02_gaussian_process.md` | **body-reviewed routing / not migrated** |
| `models.md` | 107–119 | Existing audit machinery | `guide/appendix/model-registry-and-public-api.md` | `audit-history` | **body-reviewed routing / not migrated** |
| `models.md` | 120–131 | Phase 2 public-export reconciliation | `guide/appendix/model-registry-and-public-api.md` | `audit-history` | **body-reviewed routing / not migrated** |
| `models.md` | 132–146 | Regression public symbols outside the registry | `guide/appendix/model-registry-and-public-api.md` | `guide/13_model_selection.md` | **body-reviewed routing / not migrated** |
| `models.md` | 147–185 | Binary registry IDs and exported model classes | `guide/appendix/model-registry-and-public-api.md` | `guide/24_classification.md` | **body-reviewed routing / not migrated** |
| `models.md` | 186–207 | Representative test assertion mapping | `guide/verification/README.md` | `guide/appendix/limitations-and-unverified.md` | **body-reviewed routing / not migrated** |
| `models.md` | 208–225 | Component-level implementation ownership | `guide/appendix/model-composition-and-delegation.md` | `guide/02_gaussian_process.md` | **body-reviewed routing / not migrated** |
| `models.md` | 226–235 | Limits of the current evidence | `guide/appendix/limitations-and-unverified.md` | `guide/verification/README.md` | **body-reviewed routing / not migrated** |
| `models.md` | 236–256 | Exhaustive static provenance export | `guide/appendix/model-registry-and-public-api.md` | `guide/appendix/limitations-and-unverified.md` | **body-reviewed routing / not migrated** |
| `models.md` | 257–271 | Documentation link existence validation | `guide/appendix/model-registry-and-public-api.md` | `guide/appendix/limitations-and-unverified.md` | **body-reviewed routing / not migrated** |
| `models.md` | 272–289 | Static test-reference discovery | `guide/verification/README.md` | `guide/appendix/limitations-and-unverified.md` | **body-reviewed routing / not migrated** |
| `models.md` | 290–315 | Inspected runtime composition and mathematical ownership | `guide/appendix/model-composition-and-delegation.md` | `guide/02_gaussian_process.md` | **body-reviewed routing / not migrated** |
| `models.md` | 316–338 | Inspected test assertions and verification boundaries | `guide/verification/README.md` | `guide/appendix/limitations-and-unverified.md` | **body-reviewed routing / not migrated** |
| `models.md` | 339–350 | Optional static upstream composition scan | `guide/appendix/model-composition-and-delegation.md` | `guide/appendix/limitations-and-unverified.md` | **body-reviewed routing / not migrated** |
| `models.md` | 351–367 | Conditional import provenance and static scan boundaries | `guide/appendix/model-composition-and-delegation.md` | `guide/appendix/limitations-and-unverified.md` | **body-reviewed routing / not migrated** |
| `models.md` | 368–393 | Phase 2 evidence levels and follow-up matrix | `guide/verification/README.md` | `guide/appendix/limitations-and-unverified.md` | **body-reviewed routing / not migrated** |
| `models.md` | 394–422 | Classification model documentation ownership | `guide/24_classification.md` | `guide/appendix/model-registry-and-public-api.md` | **body-reviewed routing / not migrated** |
| `models.md` | 423–441 | Source-inspected classification delegation: non-GP and label noise | `guide/24_classification.md` | `guide/23_non_gp_surrogates.md` | **body-reviewed routing / not migrated** |
| `models.md` | 442–458 | Source-inspected classification: heterogeneous ensembles and uncertain inputs | `guide/24_classification.md` | `guide/16_uncertain_input_gp.md` | **body-reviewed routing / not migrated** |
| `models.md` | 459–477 | Inspected classification test assertions: ensemble and uncertain input | `guide/verification/README.md` | `guide/24_classification.md` | **body-reviewed routing / not migrated** |
| `models.md` | 478–497 | Inspected classification test assertions: label noise and sklearn adapters | `guide/verification/README.md` | `guide/24_classification.md` | **body-reviewed routing / not migrated** |
| `models.md` | 498–518 | Source and assertion audit: binary multitask and DeepGP | `guide/24_classification.md` | `guide/07_multitask_multioutput.md` | **body-reviewed routing / not migrated** |
| `models.md` | 519–540 | Source and assertion audit: high-dimensional binary classifiers | `guide/24_classification.md` | `guide/08_high_dimensional_gp.md` | **body-reviewed routing / not migrated** |
| `models.md` | 541–563 | Source and assertion audit: neural, ensemble, robust and categorical models | `guide/24_classification.md` | `guide/14_robust_gaussian_process.md` | **body-reviewed routing / not migrated** |
| `models.md` | 564–588 | Source and assertion audit: contamination, replicates, bootstrap and robust composition | `guide/24_classification.md` | `guide/14_robust_gaussian_process.md` | **body-reviewed routing / not migrated** |
| `models.md` | 589–606 | Repeatable registry-wide static evidence summary | `guide/appendix/model-registry-and-public-api.md` | `audit-history` | **body-reviewed routing / not migrated** |
| `models.md` | 607–625 | Inspected regression non-GP implementations: NGBoost and ExtraTrees | `guide/23_non_gp_surrogates.md` | `appendix/model-registry-and-public-api.md` | **body-reviewed routing / not migrated** |
| `models.md` | 626–649 | Inspected regression noise and heterogeneous multitask implementations | `guide/15_heteroskedastic_noise.md` | `appendix/model-registry-and-public-api.md` | **body-reviewed routing / not migrated** |
| `models.md` | 650–675 | Source and assertion audit: multi-fidelity regression families | `guide/06_multi_fidelity.md` | `audit-history` | **body-reviewed routing / not migrated** |
| `models.md` | 676–697 | Source and assertion audit: regression DeepGP models | `guide/22_expressive_gp.md` | `audit-history` | **body-reviewed routing / not migrated** |
| `models.md` | 698–723 | Source and assertion audit: expressive stationary and infinite-width kernels | `guide/22_expressive_gp.md` | `audit-history` | **body-reviewed routing / not migrated** |
| `models.md` | 724–746 | Source and assertion audit: joint neural representation regression | `guide/19_neural_reduction_gp.md` | `audit-history` | **body-reviewed routing / not migrated** |
| `models.md` | 747–773 | Classification documentation mapping at registry-ID granularity | `guide/24_classification.md` | `appendix/model-registry-and-public-api.md` | **body-reviewed routing / not migrated** |
| `models.md` | 774–798 | Phase 2 inventory sign-off and scope | `guide/appendix/model-registry-and-public-api.md` | `audit-history` | **body-reviewed routing / not migrated** |
| `models.md` | 799–806 | Phase 4: theory-to-implementation records | `guide/appendix/cross-cutting-implementation-contracts.md` | `appendix/model-registry-and-public-api.md` | **body-reviewed routing / not migrated** |
| `models.md` | 807–821 | Exact single-task Gaussian-process regression | `guide/02_gaussian_process.md` | `appendix/model-registry-and-public-api.md` | **body-reviewed routing / not migrated** |
| `models.md` | 822–836 | Native mixed-input Gaussian-process regression | `guide/05_mixed_variables.md` | `appendix/model-registry-and-public-api.md` | **body-reviewed routing / not migrated** |
| `models.md` | 837–851 | Bernoulli variational Gaussian-process classification | `guide/24_classification.md` | `appendix/model-registry-and-public-api.md` | **body-reviewed routing / not migrated** |
| `models.md` | 852–866 | Long-format multitask Gaussian-process regression | `guide/07_multitask_multioutput.md` | `appendix/model-registry-and-public-api.md` | **body-reviewed routing / not migrated** |
| `models.md` | 867–881 | Kronecker block-design multitask GP | `guide/07_multitask_multioutput.md` | `appendix/model-registry-and-public-api.md` | **body-reviewed routing / not migrated** |
| `models.md` | 882–896 | SAAS fully Bayesian single-task GP | `guide/08_high_dimensional_gp.md` | `appendix/model-registry-and-public-api.md` | **body-reviewed routing / not migrated** |
| `models.md` | 897–911 | Stochastic deep Gaussian-process regression | `guide/22_expressive_gp.md` | `appendix/model-registry-and-public-api.md` | **body-reviewed routing / not migrated** |
| `models.md` | 912–926 | Random-forest ensemble regression | `guide/23_non_gp_surrogates.md` | `guide/24_classification.md` / `guide/verification/README.md` | **body-reviewed routing / not migrated** |
| `models.md` | 927–941 | Reduced input/output Gaussian-process regression | `guide/18_dimensionality_reduction_gp.md` | `guide/24_classification.md` / `guide/verification/README.md` | **body-reviewed routing / not migrated** |
| `models.md` | 942–956 | Student-t robust variational regression | `guide/14_robust_gaussian_process.md` | `guide/24_classification.md` / `guide/verification/README.md` | **body-reviewed routing / not migrated** |
| `models.md` | 957–971 | NGBoost probabilistic non-GP regression | `guide/23_non_gp_surrogates.md` | `guide/24_classification.md` / `guide/verification/README.md` | **body-reviewed routing / not migrated** |
| `models.md` | 972–986 | Heterogeneous regression/classification composition | `guide/appendix/model-composition-and-delegation.md` | `guide/24_classification.md` / `guide/verification/README.md` | **body-reviewed routing / not migrated** |
| `models.md` | 987–1000 | Classification with flipped observation labels | `guide/24_classification.md` | `guide/24_classification.md` / `guide/verification/README.md` | **body-reviewed routing / not migrated** |
| `models.md` | 1001–1014 | Classification with contaminating labels | `guide/24_classification.md` | `guide/24_classification.md` / `guide/verification/README.md` | **body-reviewed routing / not migrated** |
| `models.md` | 1015–1028 | Classification probability ensembles (GP and heterogeneous) | `guide/24_classification.md` | `guide/24_classification.md` / `guide/verification/README.md` | **body-reviewed routing / not migrated** |
| `models.md` | 1029–1042 | Non-GP binary classification and bootstrap uncertainty | `guide/24_classification.md` | `guide/24_classification.md` / `guide/verification/README.md` | **body-reviewed routing / not migrated** |
| `models.md` | 1043–1056 | Classification with uncertain candidate inputs | `guide/16_uncertain_input_gp.md` | `guide/24_classification.md` / `guide/verification/README.md` | **body-reviewed routing / not migrated** |
| `models.md` | 1057–1070 | Multitask binary classification (long format and block input) | `guide/07_multitask_multioutput.md` | `guide/24_classification.md` / `guide/verification/README.md` | **body-reviewed routing / not migrated** |
| `models.md` | 1071–1084 | Frozen reduced-input binary GP classification | `guide/18_dimensionality_reduction_gp.md` | `guide/24_classification.md` / `guide/verification/README.md` | **body-reviewed routing / not migrated** |
| `models.md` | 1085–1099 | Deep variational GP binary classification | `guide/22_expressive_gp.md` | `guide/24_classification.md` / `guide/verification/README.md` | **body-reviewed routing / not migrated** |
| `models.md` | 1100–1131 | Remaining robust binary classification families | `guide/24_classification.md` | `appendix/model-registry-and-public-api.md` | **body-reviewed routing / not migrated** |
| `models.md` | 1132–1168 | Sparse and embedded high-dimensional binary classification | `guide/08_high_dimensional_gp.md` | `appendix/model-registry-and-public-api.md` | **body-reviewed routing / not migrated** |
| `models.md` | 1169–1206 | Phase 4 registry-wide reconciliation: outstanding detailed evidence | `guide/appendix/model-registry-and-public-api.md` | `audit-history` | **body-reviewed routing / not migrated** |
| `models.md` | 1207–1224 | Phase 4 exit gate (unchanged) | `guide/verification/README.md` | `appendix/model-registry-and-public-api.md` | **body-reviewed routing / not migrated** |
| `models.md` | 1225–1242 | Phase 4 inspected binary variant reconciliation: mixed and nonstationary | `guide/24_classification.md` | `appendix/model-registry-and-public-api.md` | **body-reviewed routing / not migrated** |
| `models.md` | 1243–1269 | Phase 4 inspected reduced-GP family mapping: frozen, mixed and joint | `guide/18_dimensionality_reduction_gp.md` | `appendix/model-registry-and-public-api.md` | **body-reviewed routing / not migrated** |
| `models.md` | 1270–1293 | Phase 4 inspected neural and variational reduced-GP variants | `guide/19_neural_reduction_gp.md` | `appendix/model-registry-and-public-api.md` | **body-reviewed routing / not migrated** |
| `models.md` | 1294–1321 | Phase 4 inspected reduced multi-task implementations | `guide/07_multitask_multioutput.md` | `appendix/model-registry-and-public-api.md` | **body-reviewed routing / not migrated** |
| `models.md` | 1322–1344 | Phase 4 inspected jointly trained multi-task and design-reduced multi-fidelity GP | `guide/06_multi_fidelity.md` | `appendix/model-registry-and-public-api.md` | **body-reviewed routing / not migrated** |
| `models.md` | 1345–1369 | Phase 4 reduced-family registry reconciliation (52 entries) | `guide/18_dimensionality_reduction_gp.md` | `appendix/model-registry-and-public-api.md` | **body-reviewed routing / not migrated** |
| `models.md` | 1370–1389 | Phase 4 inspected robust-noise heteroskedastic inference paths | `guide/15_heteroskedastic_noise.md` | `appendix/model-registry-and-public-api.md` | **body-reviewed routing / not migrated** |
| `models.md` | 1390–1404 | Phase 4 inspected contamination, replicate-noise and nonstationary regression models | `guide/14_robust_gaussian_process.md` | `appendix/model-registry-and-public-api.md` | **body-reviewed routing / not migrated** |
| `models.md` | 1405–1420 | Phase 4 inspected Student-t robust regression variants | `guide/14_robust_gaussian_process.md` | `appendix/model-registry-and-public-api.md` | **body-reviewed routing / not migrated** |
| `models.md` | 1421–1437 | Phase 4 inspected robust relevance-pursuit regression variants | `guide/14_robust_gaussian_process.md` | `appendix/model-registry-and-public-api.md` | **body-reviewed routing / not migrated** |
| `models.md` | 1438–1460 | Phase 4 robust-noise registry reconciliation (27 entries) | `guide/appendix/model-registry-and-public-api.md` | `appendix/model-registry-and-public-api.md` | **body-reviewed routing / not migrated** |
| `models.md` | 1461–1477 | Phase 4 inspected uncertain-input registry models (3 entries) | `guide/16_uncertain_input_gp.md` | `audit-history` | **body-reviewed routing / not migrated** |
| `models.md` | 1478–1493 | Phase 4 inspected structured-output regression registry (4 entries) | `guide/11_structured_output.md` | `audit-history` | **body-reviewed routing / not migrated** |
| `models.md` | 1494–1516 | Phase 4 inspected hierarchical/contextual registry (6 entries) | `guide/12_hierarchical_contextual_gp.md` | `audit-history` | **body-reviewed routing / not migrated** |
| `models.md` | 1517–1530 | Phase 4 inspected preference-learning registry (1 entry) | `guide/10_preference_learning.md` | `audit-history` | **body-reviewed routing / not migrated** |
| `models.md` | 1531–1547 | Phase 4 inspected standard single-task, variational and multi-fidelity models | `guide/06_multi_fidelity.md` | `appendix/model-registry-and-public-api.md` | **body-reviewed routing / not migrated** |
| `models.md` | 1548–1564 | Phase 4 inspected non-GP surrogate registry (5 entries) | `guide/23_non_gp_surrogates.md` | `audit-history` | **body-reviewed routing / not migrated** |
| `models.md` | 1565–1581 | Phase 4 inspected core multitask and model-list registry contracts | `guide/07_multitask_multioutput.md` | `appendix/model-registry-and-public-api.md` | **body-reviewed routing / not migrated** |
| `models.md` | 1582–1600 | Phase 4 inspected specialized heterogeneous/hierarchical multitask models | `guide/07_multitask_multioutput.md` | `appendix/model-registry-and-public-api.md` | **body-reviewed routing / not migrated** |
| `models.md` | 1601–1620 | Phase 4 inspected expressive spectral and infinite-width NNGP models (12 entries) | `guide/22_expressive_gp.md` | `appendix/model-registry-and-public-api.md` | **body-reviewed routing / not migrated** |
| `models.md` | 1621–1637 | Phase 4 inspected expressive DeepGP hierarchy (4 entries) | `guide/22_expressive_gp.md` | `appendix/model-registry-and-public-api.md` | **body-reviewed routing / not migrated** |
| `models.md` | 1638–1656 | Phase 4 inspected high-dimensional SAAS, MAP-SAAS and ALEBO ownership | `guide/08_high_dimensional_gp.md` | `appendix/model-registry-and-public-api.md` | **body-reviewed routing / not migrated** |
| `models.md` | 1657–1672 | Phase 4 inspected orthogonal additive high-dimensional GP variants | `guide/21_advanced_high_dimensional_models.md` | `appendix/model-registry-and-public-api.md` | **body-reviewed routing / not migrated** |
| `models.md` | 1673–1687 | Phase 4 inspected MAP-SAAS multi-fidelity model variants | `guide/06_multi_fidelity.md` | `appendix/model-registry-and-public-api.md` | **body-reviewed routing / not migrated** |
| `models.md` | 1688–1734 | Phase 4 cross-family registry reconciliation: high-dimensional and multitask | `guide/appendix/model-registry-and-public-api.md` | `appendix/model-registry-and-public-api.md` | **body-reviewed routing / not migrated** |
| `models.md` | 1735–1785 | Phase 4 inventory-count closure audit (static, documentation-only) | `guide/appendix/model-registry-and-public-api.md` | `audit-history` | **body-reviewed routing / not migrated** |
| `models.md` | 1786–1796 | KroneckerMultiTaskGP covariance verification reference | `guide/verification/README.md` | `appendix/model-registry-and-public-api.md` | **body-reviewed routing / not migrated** |
| `acquisition.md` | 1–4 | Acquisition theory–implementation mapping: Phase 3 inventory | `guide/acquisition/01_foundations.md` | `appendix/acquisition-contracts-and-compatibility.md` | **body-reviewed routing / not migrated** |
| `acquisition.md` | 5–18 | Sources and evidence boundary | `guide/appendix/acquisition-contracts-and-compatibility.md` | `appendix/acquisition-contracts-and-compatibility.md` | **body-reviewed routing / not migrated** |
| `acquisition.md` | 19–38 | Distinct implementation paths | `guide/acquisition/01_foundations.md` | `appendix/acquisition-contracts-and-compatibility.md` | **body-reviewed routing / not migrated** |
| `acquisition.md` | 39–47 | Acquisition/optimizer interaction | `guide/acquisition/01_foundations.md` | `appendix/acquisition-contracts-and-compatibility.md` | **body-reviewed routing / not migrated** |
| `acquisition.md` | 48–60 | Fixed Phase 3 completion criteria | `guide/appendix/acquisition-contracts-and-compatibility.md` | `appendix/acquisition-contracts-and-compatibility.md` | **body-reviewed routing / not migrated** |
| `acquisition.md` | 61–78 | Inspected acquisition registry and assertion evidence | `guide/appendix/acquisition-contracts-and-compatibility.md` | `appendix/acquisition-contracts-and-compatibility.md` | **body-reviewed routing / not migrated** |
| `acquisition.md` | 79–114 | Phase 5 — regression active-learning acquisition mappings (source inspected) | `guide/acquisition/09_active_learning.md` | `appendix/acquisition-contracts-and-compatibility.md` | **body-reviewed routing / not migrated** |
| `acquisition.md` | 115–143 | Phase 5 — classification AL and Gaussian EPIG (source inspected) | `guide/acquisition/09_active_learning.md` | `appendix/acquisition-contracts-and-compatibility.md` | **body-reviewed routing / not migrated** |
| `acquisition.md` | 144–178 | ExpectedPredictiveInformationGain (regression) | `guide/acquisition/05_information_theoretic.md` | `appendix/acquisition-contracts-and-compatibility.md` | **body-reviewed routing / not migrated** |
| `acquisition.md` | 179–232 | Phase 5 — single-objective Monte Carlo BO (source inspected) | `guide/acquisition/04_batch_noisy.md` | `appendix/acquisition-contracts-and-compatibility.md` | **body-reviewed routing / not migrated** |
| `acquisition.md` | 233–249 | Phase 5 — multi-objective MC Bayesian optimization (source inspected) | `guide/acquisition/07_multiobjective.md` | `appendix/acquisition-contracts-and-compatibility.md` | **body-reviewed routing / not migrated** |
| `acquisition.md` | 250–287 | Verified construction contracts | `guide/acquisition/01_foundations.md` | `appendix/acquisition-contracts-and-compatibility.md` | **body-reviewed routing / not migrated** |
| `acquisition.md` | 288–308 | Phase 5 — outcome-constrained BO composition (source inspected) | `guide/acquisition/01_foundations.md` | `appendix/acquisition-contracts-and-compatibility.md` | **body-reviewed routing / not migrated** |
| `acquisition.md` | 309–345 | Interpretation and boundaries | `guide/acquisition/01_foundations.md` | `appendix/acquisition-contracts-and-compatibility.md` | **body-reviewed routing / not migrated** |
| `acquisition.md` | 346–362 | Phase 5 — information-theoretic and lookahead acquisitions (source inspected) | `guide/acquisition/06_lookahead.md` | `appendix/acquisition-contracts-and-compatibility.md` | **body-reviewed routing / not migrated** |
| `acquisition.md` | 363–405 | One-shot and information-target boundaries | `guide/acquisition/06_lookahead.md` | `appendix/acquisition-contracts-and-compatibility.md` | **body-reviewed routing / not migrated** |
| `acquisition.md` | 406–422 | Phase 5 — coverage reconciliation and exit audit | `guide/acquisition/01_foundations.md` | `appendix/acquisition-contracts-and-compatibility.md` | **body-reviewed routing / not migrated** |
| `acquisition.md` | 423–465 | Confirmed limits and carry-forward | `guide/acquisition/01_foundations.md` | `appendix/acquisition-contracts-and-compatibility.md` | **body-reviewed routing / not migrated** |
| `optimization.md` | 1–4 | Acquisition optimization theory–implementation mapping: Phase 3 inventory | `guide/optimization/01_foundations.md` | `appendix/optimizer-dispatch-and-constraints.md` | **body-reviewed routing / not migrated** |
| `optimization.md` | 5–19 | Baseline and separation of responsibilities | `guide/optimization/01_foundations.md` | `appendix/optimizer-dispatch-and-constraints.md` | **body-reviewed routing / not migrated** |
| `optimization.md` | 20–39 | Named optimization dispatcher | `guide/optimization/01_foundations.md` | `appendix/optimizer-dispatch-and-constraints.md` | **body-reviewed routing / not migrated** |
| `optimization.md` | 40–62 | Candidate-space constraints versus outcome constraints | `guide/optimization/05_constraints.md` | `appendix/optimizer-dispatch-and-constraints.md` | **body-reviewed routing / not migrated** |
| `optimization.md` | 63–76 | Stateful and coordinate-changing algorithms | `guide/optimization/09_high_dimensional.md` | `appendix/optimizer-dispatch-and-constraints.md` | **body-reviewed routing / not migrated** |
| `optimization.md` | 77–85 | Verification boundaries | `guide/optimization/01_foundations.md` | `appendix/optimizer-dispatch-and-constraints.md` | **body-reviewed routing / not migrated** |
| `optimization.md` | 86–99 | Inspected optimizer and strategy assertion evidence | `guide/optimization/01_foundations.md` | `appendix/optimizer-dispatch-and-constraints.md` | **body-reviewed routing / not migrated** |
| `optimization.md` | 100–107 | Fixed follow-up audit surface | `guide/optimization/01_foundations.md` | `appendix/optimizer-dispatch-and-constraints.md` | **body-reviewed routing / not migrated** |
| `optimization.md` | 108–118 | Specialized optimization paths: inspected source and assertions | `guide/optimization/01_foundations.md` | `appendix/optimizer-dispatch-and-constraints.md` | **body-reviewed routing / not migrated** |
| `optimization.md` | 119–133 | Coordinate-changing high-dimensional strategies | `guide/optimization/09_high_dimensional.md` | `appendix/optimizer-dispatch-and-constraints.md` | **body-reviewed routing / not migrated** |
| `optimization.md` | 134–140 | Remaining Phase 3 audit surface | `guide/optimization/01_foundations.md` | `appendix/optimizer-dispatch-and-constraints.md` | **body-reviewed routing / not migrated** |
| `optimization.md` | 141–152 | HeSBO, latent-space reconstruction and backend constraint semantics | `guide/optimization/09_high_dimensional.md` | `appendix/optimizer-dispatch-and-constraints.md` | **body-reviewed routing / not migrated** |
| `optimization.md` | 153–165 | Phase 3 evidence limits | `guide/optimization/01_foundations.md` | `appendix/optimizer-dispatch-and-constraints.md` | **body-reviewed routing / not migrated** |
| `optimization.md` | 166–181 | Phase 3 registry reconciliation gate | `guide/optimization/01_foundations.md` | `appendix/optimizer-dispatch-and-constraints.md` | **body-reviewed routing / not migrated** |
| `optimization.md` | 182–205 | Phase 6 — named optimizer mathematical and dispatch mapping | `guide/optimization/01_foundations.md` | `appendix/optimizer-dispatch-and-constraints.md` | **body-reviewed routing / not migrated** |
| `optimization.md` | 206–245 | Exact dispatch and mathematical limitations | `guide/optimization/01_foundations.md` | `appendix/optimizer-dispatch-and-constraints.md` | **body-reviewed routing / not migrated** |
| `optimization.md` | 246–257 | Phase 6 — specialist optimizers and candidate-constraint semantics | `guide/optimization/05_constraints.md` | `appendix/optimizer-dispatch-and-constraints.md` | **body-reviewed routing / not migrated** |
| `optimization.md` | 258–274 | Constraint handling is part of the algorithm | `guide/optimization/05_constraints.md` | `appendix/optimizer-dispatch-and-constraints.md` | **body-reviewed routing / not migrated** |
| `optimization.md` | 275–295 | Tests and mathematical verification boundary | `guide/optimization/01_foundations.md` | `appendix/optimizer-dispatch-and-constraints.md` | **body-reviewed routing / not migrated** |
| `optimization.md` | 296–314 | Phase 6 — stateful and coordinate-changing search strategies | `guide/optimization/09_high_dimensional.md` | `appendix/optimizer-dispatch-and-constraints.md` | **body-reviewed routing / not migrated** |
| `optimization.md` | 315–355 | Coordinate and constraint invariants | `guide/optimization/09_high_dimensional.md` | `appendix/optimizer-dispatch-and-constraints.md` | **body-reviewed routing / not migrated** |
| `verification.md` | 1–3 | Mathematical verification: Phases 7–9 | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **body-reviewed routing / not migrated** |
| `verification.md` | 4–26 | Purpose and evidence rules | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **body-reviewed routing / not migrated** |
| `verification.md` | 27–39 | Phase 7 — model verification matrix | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **body-reviewed routing / not migrated** |
| `verification.md` | 40–64 | Initial Phase 7 execution gate | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **body-reviewed routing / not migrated** |
| `verification.md` | 65–79 | Kronecker block-design covariance reference (Phase 7) | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **body-reviewed routing / not migrated** |
| `verification.md` | 80–93 | Data-fidelity downsampling kernel reference (Phase 7) | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **body-reviewed routing / not migrated** |
| `verification.md` | 94–107 | PCA reduced-space projection reference (Phase 7) | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **body-reviewed routing / not migrated** |
| `verification.md` | 108–121 | Replicate observation-noise statistical reference (Phase 7) | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **body-reviewed routing / not migrated** |
| `verification.md` | 122–134 | Binary GP probit predictive reference (Phase 7) | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **body-reviewed routing / not migrated** |
| `verification.md` | 135–148 | Non-GP Gaussian posterior reparameterization reference (Phase 7) | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **body-reviewed routing / not migrated** |
| `verification.md` | 149–183 | Phase 7 execution reconciliation and remaining scope | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **body-reviewed routing / not migrated** |
| `verification.md` | 184–197 | Reduced GP public/latent posterior coordinate reference (Phase 7) | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **body-reviewed routing / not migrated** |
| `verification.md` | 198–211 | Heteroskedastic latent-versus-observation noise reference (Phase 7) | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **body-reviewed routing / not migrated** |
| `verification.md` | 212–222 | Non-GP Random Forest estimator-level reference (Phase 7) | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **body-reviewed routing / not migrated** |
| `verification.md` | 223–244 | Phase 7 exit audit (property-specific) | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **body-reviewed routing / not migrated** |
| `verification.md` | 245–256 | Mixed reduced GP categorical-coordinate reference (Phase 7) | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **body-reviewed routing / not migrated** |
| `verification.md` | 257–273 | Heteroskedastic observation-variance composition boundary (Phase 7) | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **body-reviewed routing / not migrated** |
| `verification.md` | 274–291 | Phase 7 final exit reconciliation | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **body-reviewed routing / not migrated** |
| `verification.md` | 292–306 | Phase 8 — regression active-learning formula reference | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **body-reviewed routing / not migrated** |
| `verification.md` | 307–319 | Phase 8 — classification active-learning arithmetic reference | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **body-reviewed routing / not migrated** |
| `verification.md` | 320–334 | Phase 8 — BALD and probability-variance sample-moment reference | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **body-reviewed routing / not migrated** |
| `verification.md` | 335–350 | Phase 8 — native analytic BO Gaussian reference | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **body-reviewed routing / not migrated** |
| `verification.md` | 351–362 | Phase 8 — Monte Carlo qSimpleRegret sample reference | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **body-reviewed routing / not migrated** |
| `verification.md` | 363–375 | Phase 8 — Monte Carlo qUCB sample reference | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **body-reviewed routing / not migrated** |
| `verification.md` | 376–407 | Phase 8 — verification reconciliation (representative scope) | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **body-reviewed routing / not migrated** |
| `verification.md` | 408–421 | Phase 9 — sampling optimizer reference (initial property) | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **body-reviewed routing / not migrated** |
| `verification.md` | 422–435 | Phase 9 — TuRBO state-transition reference | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **body-reviewed routing / not migrated** |
| `verification.md` | 436–448 | Phase 9 — TuRBO trust-region geometry reference | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **body-reviewed routing / not migrated** |
| `verification.md` | 449–464 | Phase 9 — candidate constraint residual and feasibility ordering reference | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **body-reviewed routing / not migrated** |
| `verification.md` | 465–475 | Phase 9 — PyTorch optimizer candidate-constraint support boundary | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **body-reviewed routing / not migrated** |
| `verification.md` | 476–510 | Phase 9 — representative optimizer verification exit audit | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **body-reviewed routing / not migrated** |

## README.md（13区間）の本文レビュー結果

旧MappingのREADME全文を確認し、**技術契約と歴史的判定が同一区間に混在する**箇所を区別した。これは移行先の確定に向けた内容レビューであり、技術ガイドへの本文転記やソースコードの再検証ではない。

| 原文行 | 技術本文として残す内容 | 別途保全する履歴・制約 |
| --- | --- | --- |
| 1–10 | Mappingの目的、対象範囲、数学的正しさを保証しない境界 | 旧12 Phase、当時のbaseline SHA |
| 11–23 | theory / models / optimization / architecture / notebooks / tests / coverage JSONの正本責任 | 正本ファイルを新ガイドが上書きしない |
| 24–41 | regression registryとclassification registryの相違、acquisition metadataとoptimizer実行可否の非同一性、生成coverageの責務 | 旧監査時点のsource-of-truth判定 |
| 42–66 | 理論→実装→assertion単位の検証という読者導線、6種類の実装関係ラベル | 旧Phase 10のナビゲーション履歴 |
| 67–86 | method ID、数学的対象、API/実装symbol、上流委譲、数値手法、test、Notebook、limitationsの9項目契約 | 「存在する」と「数値的に検証済み」の区別 |
| 87–94 | 旧5 Mappingのcanonical契約、wrapperやregistryを文書のために増設しない原則 | 旧Phase 4–9の執筆順序 |
| 95–107 | 静的監査と独立数値参照の証拠水準の違い、監査スクリプト | 旧Phaseごとの未完了項目 |
| 108–136 | inventory / mapping / property-specific verification / CIの証拠水準 | 旧12 Phaseの判定、PR、当時の件数 |
| 137–174 | 11 named backendとspecialist pathの区別、stateful strategy、candidate constraint、unsupported combinations | 旧Phase 6のexit gate |
| 175–201 | 理論・Mapping・検証間の双方向ナビゲーション、リンク検査の限界 | 旧Phase 10のPR・CI |
| 202–231 | 旧5ファイル存在・戻りリンクの構造的CI契約、数値検証ではない限界 | 旧Phase 11のPR・CI |
| 232–278 | heteroskedastic posterior、制約・HV・lookahead、optimizer収束、async等の未検証範囲 | 旧Phase 12の条件付き完了判定・PR履歴 |

**保全上の重要な修正:** 原台帳で `README.md` の13区間を機械的に `audit/` 中心へ振っていた部分を、本文の技術的主張に基づいてガイド本文・Appendix・verificationへ再分類した。旧Phase番号を理由に技術情報を監査履歴だけへ移してはならない。

**状態:** README 13/13区間は「本文を読んで移行先を再判定」まで完了。ただし各主張の実装照合・新章への文章移行・assertion検証は未完了であり、`migrated` / `verified` は0件。残り161区間は暫定分類のまま。

## models.md（冒頭14区間）の本文レビュー

原文 `models.md` の**1–256行**を確認し、次の技術的区別を保全対象として抽出した。移行先の確定は原文の記述に基づくが、現行コードを再実行して数値検証したものではない。

| 原文行 | 主な技術的主張・保全項目 | 移行時の注意 |
| --- | --- | --- |
| 1–17 | 元のMappingの位置付け、初期baseline SHA、静的MRO・source inspection・assertionの証拠水準 | 原文の当時の状態と現在の実装を同一視しない |
| 18–31 | regression registryの142件、classification registryの29 ID、heterogeneous layerの別扱い、helper exportsはmodel IDでない | 件数は原文時点の記録。最新件数はregistryから再計算 |
| 32–51 | regression 142件のfamily別全件表。特にreduced 52、robust/noise 27、expressive 16など | 全名称表をAppendixへ保全し、単一モデル章へ押し込まない |
| 52–91 | binary 29 IDの一覧と代表的source/test evidence。Kronecker分類のlong-format variationalとexact Kroneckerの相違 | registry IDと公開class名を混同しない。個別IDのruntime検証を主張しない |
| 92–106 | SingleTaskGP・MixedSingleTaskGP・binary classifier・ReducedGP・RandomForestのcomponent-level ownership | local wrapperがあることと確率アルゴリズムの実装主体を区別 |
| 107–119 | 既存audit scriptとgenerated coverageの所有権 | `docs/model_coverage.json`を手編集しない |
| 120–131 | regression registry 142件、public exports 154名、classification exports 76名、binary model ID 29件 | 数字は静的監査の時点値。exportとregistryの集合は同じではない |
| 132–146 | registry外の公開12シンボル（`HeterogeneousModel`等） | infrastructure symbolを未登録モデルと誤分類しない |
| 147–185 | binary registry ID→公開class名の全29対応 | 分類registryは理論・Notebookリンクを持たず、別途照合が必要 |
| 186–207 | 代表的test assertionの範囲と非保証事項（RF calibration、posterior approximation等） | test file存在≠全モデルの数学的検証 |
| 208–225 | fit/raw-data/GP posteriorの委譲、分類ELBO、reducer、sklearn estimatorのcomponent ownership | `Extended`と`Delegated`をコンポーネントごとに併記 |
| 226–235 | 171 registry entriesはinventory済みだがsource inspectionは代表5パス | 監査scriptのmethod ownerは数値正しさの証拠ではない |
| 236–256 | `implementation_module`、`upstream_bases`、`provenance_verification`、documentation metadata | MROに上流baseがなくてもruntime composition/delegationはあり得る |

### 実装章への分岐が必要な箇所

- `SingleTaskGP`はGP本体とfit/data契約を分け、`guide/02_gaussian_process.md`と横断契約に割り当てる。
- `MixedSingleTaskGP`はnative categorical GPとデータ契約を区別し、`guide/05_mixed_variables.md`にも二次参照が必要。
- `BinarySingleTaskGPClassifier`はlatent posterior、class probability、Bernoulli likelihood、ELBOを区別し、`guide/24_classification.md`へ接続する。
- `ReducedGP`はreducer・latent GP・posterior reconstructionを分離し、`guide/18_dimensionality_reduction_gp.md`にも接続する。
- `RandomForestSurrogate`はsklearnへの委譲とBoTorch-compatible posterior facadeを区別し、`guide/23_non_gp_surrogates.md`へ接続する。

**進捗:** `models.md`冒頭14/91区間を本文レビュー済み。READMEの13区間と合わせて27/174区間。未レビュー147区間。新ガイドへの移行・検証は依然0件。

## models.md（257–393行、追加7区間）の本文レビュー

既存の静的監査・委譲・テスト証拠を、ガイド本文と検証Appendixへ分離するために原文を確認した。

| 原文行 | 残す技術情報 | 証拠水準・限界 |
| --- | --- | --- |
| 257–271 | `audit_model_api.py --check-documentation-paths` と `generate_model_coverage.py --check` | regression registryに記録されたguide/theory/notebookの**パス存在**のみ。Notebook実行や分類ドキュメントを保証しない |
| 272–289 | `--include-test-references` によるAST参照抽出、`syntactic_reference_only` | alias・動的生成による偽陰性と同名参照による偽陽性。assertionや実行成功を保証しない |
| 290–315 | RandomForestの`self._forest`、Binary GPのBernoulli/ELBO、ReducedGPのreducerとBoTorch GPの**合成** | MROだけではsklearn/GPyTorch/reducerの委譲を発見できない。source-inspectedだが全runtime pathは未確認 |
| 316–338 | Mixed GP・RandomForest・Binary GP・ReducedGPの代表テスト、named test数とassertion scope | RFのcalibration、分類のELBO最適値、reduction covariance誤差などは未検証 |
| 339–350 | `--include-upstream-calls` によるBoTorch/GPyTorch/sklearnの直接import-call静的検出 | helper内委譲・間接factory・inherited method・runtime aliasを網羅しない |
| 351–367 | try/except ImportError内の条件付きimport、alias、文字列の偽検出を区別するscanner regression test | syntax scan成功はfit、posterior、likelihoodの数値正確性を証明しない |
| 368–393 | registry/export、MRO、direct composition、test AST、inspected assertion、numerical referenceの**6段階の証拠** | 静的スキャンの空結果は`Unsupported`ではなく未検出。`Unverified`を維持 |

### モデル章への二次接続（本文レビューで必要と判明）

- `models.md` 290–315：`guide/23_non_gp_surrogates.md`、`guide/24_classification.md`、`guide/18_dimensionality_reduction_gp.md`をそれぞれ追加参照する。1区間に複数の数学的対象が含まれるため、単一のGP章だけへ移さない。
- `models.md` 316–338：`guide/05_mixed_variables.md`、`guide/23_non_gp_surrogates.md`、`guide/24_classification.md`、`guide/18_dimensionality_reduction_gp.md`とverificationを接続する。
- `models.md` 368–393：全モデルに共通するevidence taxonomyとしてverification/Appendixに配置する。過去のPhase番号だけを監査履歴に残す。

**進捗:** README 13区間 + models.md 21区間 = 34/174区間を本文レビュー済み。140区間は未レビュー。ガイド本文への移行・検証完了は0件。

## models.md（394–588行、分類9区間）の本文レビュー

分類モデルの原文を読み、latent posterior、class probability、観測ラベルノイズ、ensemble probability posterior、非GP adapterを混同しないよう移行先を再判定した。以下は**原文に記録されたsource/test inspectionの整理**であり、このコミットでコードを再実行した結果ではない。

| 原文行 | 技術的保全対象 | 章の接続・未検証境界 |
| --- | --- | --- |
| 394–422 | 分類29 registry IDはregression registryのper-model guide/theory/notebook metadataを持たない。共有分類ガイドは個別modelの実行例を保証しない | `24_classification` + model-registry Appendix。ID別ドキュメントとNotebookは別途確認 |
| 423–441 | sklearn classifier adapterは`estimator_class`経由で委譲。label-flip channelは `p_observed = a + (1-a-b)*p_clean` | `24_classification` + `23_non_gp_surrogates` + robust/noise。quadrature、識別可能性、calibrationは未検証 |
| 442–458 | heterogeneous ensembleはmember probabilityの重み付き経験分布。共通latentがなく`latent_posterior`/`sample_latent`を明示的に非対応。連続入力不確実性はGaussian摂動のMC平均 | `24_classification` + `16_uncertain_input_gp`。probability posteriorをGP latent posteriorと呼ばない |
| 459–477 | heterogeneous ensemble 7 named tests、continuous uncertain input 5 named tests。重み正規化、mean/variance、zero-std極限など | entropy/MI identity、相関共分散の積分精度、near-singular安定性は未検証 |
| 478–497 | label noise 5 named tests、sklearn adapters 4 named tests（parametrizationは別）。非対称flip formulaを固定値で検証 | class-order、quadrature、calibration、単一クラス学習などは未検証 |
| 498–518 | multitask variational ICM（IndexKernel×data kernel）、block design→long-format変換、DeepGPのDeepApproximateMLL/Bernoulli | `07_multitask_multioutput` + `22_expressive_gp`。**Kronecker名の分類モデルはexact Kronecker推論を実装したという意味ではない** |
| 519–540 | reduced binary GP（PCA/PLS/random projection）、variational SAASとMAP-SAASの異なるprior設定、ALEBO Mahalanobis metric | `08_high_dimensional_gp` + `18_dimensionality_reduction_gp`。SAAS分類をNUTS/Pyro fully Bayesianと誤記しない |
| 541–563 | joint neural encoder、homogeneous GP ensemble、input-dependent flip、Gibbs nonstationary、categorical finite-support marginalization | `19_neural_reduction_gp`、`17_nonstationary_gp`、`16_uncertain_input_gp`も参照。gradient到達≠学習品質 |
| 564–588 | contamination `(1-c)*p_clean+c*p_contaminant`、replicate disagreement `2*p*(1-p)`、bootstrap probability moments、mixed/task Gibbs kernel | `14_robust_gaussian_process` + `15_heteroskedastic_noise`。replicate disagreementはposterior varianceでも代替likelihoodでもない |

### 本文移行時の必須注意

1. `KroneckerMultiTaskBinaryGPClassifier`はblock-to-long変換後のvariational ICM backendであり、exact Kronecker algebraの検証と区別する。
2. `SaasBinarySingleTaskGPClassifier`と`MapSaasBinarySingleTaskGPClassifier`のprior・推論契約を別々に記載し、NUTSの実行を主張しない。
3. Heterogeneous ensembleではlatent APIが非対応であることを**肯定的な仕様**として残し、probability-based ALとの接続を記載する。
4. ノイズ付き分類はclean probabilityとobserved probabilityを区別する。fixed-value testはcalibrationを証明しない。
5. 不確実入力のGaussian MC平均とcategorical finite-support marginalizationは異なる計算法として説明する。

**進捗:** README 13 + models 30 = 43/174区間レビュー済み。残り131区間。新ガイドへの移行・検証は0件。

## models.md（589–798行）の本文レビュー記録

以下9区間の原文を確認した。いずれも移行先の判定であり、新章への移行・現行コードの数値再検証ではない。

| 原文行 | 保存する実装上の区別 |
| --- | --- |
| 589–606 | 静的test/upstream call検出はassertion実行や数学的正しさを証明しない |
| 607–625 | NGBoostのGaussian予測分布とExtraTreesの経験的ensemble posteriorを区別。optional NGBoostテストはskipされ得る |
| 626–649 | 異分散GPの交互学習とjoint variational学習を区別。Kronecker異分散prototypeのobservation posteriorは未対応 |
| 650–675 | BoTorch委譲のMultiFidelity GPと、mixed設計変数・構造fidelity変数の分離を維持 |
| 676–697 | DeepGPのsample-based posteriorは厳密なGaussian posteriorではない。旧DeepKernelGP aliasは非公開 |
| 698–723 | SpectralMixtureKernelのGPyTorch委譲と独自InfiniteWidthReLUKernel再帰を区別 |
| 724–746 | JointEncoderの同時学習、HybridAEの再構成項、JointVAEのKLと不確実性伝播を区別 |
| 747–773 | 分類29 registry IDの共有ガイドは個別Notebook実行の証拠ではない |
| 774–798 | 旧静的監査の完了記録はauditへ、未検証範囲はverificationへ残す |

複数テーマを含む区間は関連するMixed、Noise、MultiTask、Kernel、Reduced、Classification各章への二次参照を後続の本文移行で追加する。

> 過去の作業時点の進捗記録（現在値ではない）：README 13 + models 39 = 52/174区間レビュー済み。未レビュー122区間。移行・検証完了0件。

## models.md（799–911行、method-level mapping 8区間）のレビュー

原文のmethod-level mapping recordを読み、API・委譲・数値手法・assertion・制約を章ごとに再割当した。

| 原文行 | 移行時の技術契約 |
| --- | --- |
| 799–806 | registry metadataの正本はgenerated coverage。記録のsource/test inspection baselineを保持 |
| 807–821 | SingleTaskGPはBoTorch exact posteriorをDelegated、raw dataとfit/MLLをExtended。状態転送後のposterior比較は限定条件 |
| 822–836 | MixedSingleTaskGPはBoTorch native mixed kernelをDelegated。負のcat_dimsを許容し、カテゴリ値を順序付き連続量と見なさない |
| 837–851 | BinarySingleTaskGPClassifierはlatent posterior、Bernoulli predictive distribution、predict_proba、predict_classを分離。ELBOはGPyTorchに委譲 |
| 852–866 | MultiTaskGPはtask featureを含むlong-format exact GP。block-designのKronecker GPとは異なる |
| 867–881 | KroneckerMultiTaskGPはtrain_Y [n,m]のblock design。未観測タスクを含む不完全gridと同一視しない |
| 882–896 | SaasFullyBayesianSingleTaskGPはBoTorchのMCMC/NUTSへ委譲。make_mllを前提とするexact GPと区別 |
| 897–911 | SingleTaskDeepGPはGPyTorch variational layersとsample-based DeepGPPosterior。exact Gaussian posteriorではない |

**進捗:** README 13 + models 47 = 60/174区間レビュー済み。未レビュー114区間。移行・検証完了0件。

## models.md（912–1099行、13 method records）の本文レビュー

各method recordの公開API・数値計算法・委譲境界・assertion・未検証事項を原文から確認した。表のsecondaryは複数章にまたがる場合の参照先であり、分類ではない回帰レコードについてはverificationのみが適用される。

| 原文行 | 保存すべき相違 |
| --- | --- |
| 912–926 | RandomForestはsklearn tree predictionをBoTorch EnsemblePosteriorに変換。ensemble spreadはcalibrated Bayesian uncertaintyではない |
| 927–941 | ReducedGPはinput/output reducerとBoTorch GPを合成。reduced outputでtrain_Yvar非対応 |
| 942–956 | Student-tはheavy-tailed observation likelihoodとvariational training_loss。exact Gaussian MLLではない |
| 957–971 | NGBoostはNormal(loc,scale)のparametric posterior adapter。optional dependencyのテスト実行は未確認 |
| 972–986 | HeterogeneousModelはregression/classification childのoutput ownershipとdispatch。global posterior/ELBOを定義しない |
| 987–1000 | asymmetric label flipはclean/observed probabilityを区別。p_obs=FP+(1-FP-FN)*p_clean |
| 1001–1014 | contaminationは(1-epsilon)*p_clean+epsilon*p_contaminant。入力依存ノイズとは異なる |
| 1015–1028 | probability ensembleはmember確率の経験分布。共通latent GP posteriorを定義しない |
| 1029–1042 | sklearn非GP分類とbootstrap empirical uncertainty。単体推定器はepistemic samplingを保証しない |
| 1043–1056 | 連続候補入力はGaussian MC積分、カテゴリ入力は有限支持の重み付き厳密和。訓練入力やラベルの不確実性とは区別 |
| 1057–1070 | multitask binary GPはvariational ICM。Kronecker名のblock入力でもexact Kronecker推論ではない |
| 1071–1084 | PCA/PLS/random projectionは凍結した入力削減器。encoderとGPのjoint trainingではない |
| 1085–1099 | DeepGP binaryはDeepApproximateMLLとBernoulli likelihood、finite MC予測。校正・収束は未検証 |

> 過去の作業時点の進捗記録（現在値ではない）：README 13 + models 60 = 73/174区間本文レビュー済み。未レビュー101区間。新ガイドへの本文移行・検証完了は0件。

## models.md（1100–1344行、9区間）の本文レビュー

モデル別記録と旧Phaseの照合手順を読み、教育用ガイドと監査履歴を区別した。

| 原文行 | 保存する技術内容・注意点 |
| --- | --- |
| 1100–1131 | replicate labelsは個別0/1観測を維持し、2r(1-r)は記述的な不一致率。input-dependent flipとGibbs nonstationarityは観測ノイズとlatent covarianceの別概念 |
| 1132–1168 | MAP-SAAS/SAAS binaryはHalf-Cauchy prior付きvariational BernoulliでNUTSではない。ALEBOはembedded Mahalanobis metric、JointEncoderはencoderへELBO勾配が流れる |
| 1169–1206 | 142 regression + 29 binary IDのregistry別の詳細照合未完了事項。静的inventoryと数学的検証を区別 |
| 1207–1224 | 旧Phaseのexit gateは監査履歴へ、assertion/数値検証の証拠区分はverificationへ |
| 1225–1242 | binary.mixedはnative categorical kernelでone-hot不要。mixed Gibbsはcontinuousのみ、multitask Gibbsはtask identityをlengthscale入力から除外 |
| 1243–1269 | frozen ReducedGP、output reduction、mixed continuous-only reduction、joint encoderを異なるtraining contractとして整理。original/latent同次元では入力判定が曖昧 |
| 1270–1293 | pretrained VAE/SAEのfrozen mean codeとJointVAEのjoint trainingを区別。uncertainty_aware_posteriorはsingle-outputのMC moment approximation |
| 1294–1321 | reduced multitask long-formatのtask列とKronecker block outputを区別。reducerはtask identityを変換しない。同次元入力の判定曖昧性も保持 |
| 1322–1344 | **訂正の継承が必須**: ReducedMultiFidelityGPという名前がなくてもPCAMultiFidelityGP/PLSMultiFidelityGP/RandomProjectionMultiFidelityGPが存在。designのみ削減しfidelity列を保存。Joint multi-task encoder/AE/VAEのlossとlayoutを区別 |

教育用本文では旧Phase番号・完了宣言を除去し、実装境界と未検証事項のみを残す。複数章への二次参照は各章執筆時に追記する。

> 過去の作業時点の進捗記録（現在値ではない）：README 13 + models 69 = 82/174区間レビュー済み。未レビュー92区間。本文移行・検証完了0件。

## models.md（1345–1547行、11区間）の本文レビュー

原文のregistry family reconciliationとsource/test assertion記録を確認した。登録数は静的照合であり数値的正しさの検証ではない。

| 原文行 | 移行時に保存する実装契約 |
| --- | --- |
| 1345–1369 | reduced family 52件 = deterministic 10 + frozen multitask 10 + neural single/mixed 14 + neural multitask 15 + design-reduced fidelity 3。family countは数学的証明ではない |
| 1370–1389 | alternating heteroskedastic、joint variational two-latent、block task-noise、multi-fidelity noiseを区別。joint posteriorはresponse latentとnoiseを別APIで公開 |
| 1390–1404 | contaminationはGaussian mixture likelihood、replicate noiseは平均の固定観測分散s²/n、Gibbs nonstationarityはlatent covariance。相互に置換不可 |
| 1405–1420 | Student-tはvariational Gaussian latent GPのheavy-tail観測尤度。make_mll非対応でtraining_lossを使用 |
| 1421–1437 | BoTorch relevance pursuit sparse-outlier likelihoodはStudent-t/contaminationと別。mixed/task structural featureを分離 |
| 1438–1460 | robust_noise 27件のregistry照合。4 relevance + 4 contamination + 4 Student-t + 10 heteroskedastic/replicate + 5 Gibbs |
| 1461–1477 | Gaussian uncertain inputのexpected RBF kernel、mixed continuous-only uncertainty、categorical simplex expected kernelは異なる。candidate uncertaintyをposteriorが自動周辺化するとは限らない |
| 1478–1493 | HigherOrderGPはtensor output factors、LatentKroneckerGPはX×T product space。mixed kernelはX factorのみに適用 |
| 1494–1516 | hierarchical parent indicesは設計カテゴリと別。LCEAGPとLCEMGPのcontext/task契約を分ける |
| 1517–1530 | preference-learningのpairwise比較を通常の数値観測回帰と同一視しない |
| 1531–1547 | standard exact、variational、multi-fidelityの異なるfit/MLLとfidelity covarianceを分離 |

> 過去の作業時点の進捗記録（現在値ではない）：README 13 + models 80 = 93/174区間レビュー済み。未レビュー81区間。ガイド本文移行・検証完了0件。

## 各区間の移行検収条件

1. 原文の全行を読み、数式、API、型、例外、candidate/outcome制約、実装ownership、upstream symbol、テストassertion、許容誤差、未検証事項を抽出する。
2. 第一・第二移行先を実際の本文に基づいて確定し、必要なら**1区間を複数章に分割**する。
3. 移行先の章には説明文と正確なソース・テストへの参照を記述する。表のリンクだけで移行済みにしない。
4. レジストリや静的存在チェックをruntime/数学的同値性の証拠に格上げしない。
5. 移行先の本文と原文を照合してから `planned → migrated → verified` と変更する。未確認のまま原文を削除しない。
6. 原文5ファイルと既存anchorを保全し、ドキュメントリンクCIを通す。

## 今回の作業範囲

このPhaseでは**全174区間の本文確認と一次配置の確定**までを実施した。数式・API・実装・テストの独立した再検証と48章の本文執筆は後続Phaseで実施する。移行先パスは未作成を含むため、Markdownリンクではなくコード表記とする。


## 最終整合確認（Phase 3）

| 確認対象 | 結果 |
| --- | --- |
| 元ファイルと台帳 | README 13、models 91、acquisition 18、optimization 21、verification 31 = **174区間** |
| 元ファイルの行範囲 | 5ファイルとも台帳範囲がファイル末尾まで連続し、各区間の見出し開始位置が一致 |
| 台帳ステータス | **174/174 body-reviewed routing / not migrated**、planned 0、migrated 0、verified 0 |
| 本文と監査履歴 | 数式・API・実装制約・assertionは技術ガイド、旧Phase・PR・CIの時系列は本監査記録で保持 |
| 新ガイドの章構成 | Phase 2の成果物を積み上げブランチから復元し、READMEの24+12+12章を確認。各章本文は未作成（予定パス） |
| 移行先の章内容 | 本文未執筆につき照合未実施。全174区間の移行検収は後続Phase |

### 履歴の読み方

以下の各区間レビューに記載された過去の「累計」は作業途中のスナップショットであり、現在の進捗ではない。最終値は本節の174/174を唯一の集計基準とする。前掲の詳細レビューは技術的判断の根拠として保存し、重複した進捗宣言を完了証明には使用しない。

### 追加の是正が必要な点

Phase 2のPR #1451はPhase 1の積み上げブランチをbaseとしてマージされており、Phase 3ブランチ（main起点）にはガイドREADMEが含まれていなかった。元ブランチから同一内容のREADMEを復元した。台帳primaryの不一致2件（hierarchical/contextual、preference）をPhase 2目次の正式パスに修正した。48章の各本文は未作成であり、移行先パスは**計画上のラベル**。後続Phaseでは実際の章作成とsecondary参照の確定を行う。

## 時系列レビュー補足（当時の作業記録）

## verification.md（184–510行、20区間）の本文レビュー

- 184–291: PCA public/latent posterior consistency、heteroskedastic latent/noise separation、RandomForest estimator-level moments、mixed reduced categorical preservation、manual conditional observation varianceを区別する。**posterior(observation_noise=True)のheteroskedastic candidate-specific varianceは検証済みではない**。過去の暫定exit判定と最終property-specific判定を履歴として分離。
- 292–407: regression AL 4種の数式、classification entropy/margin/latent straddle、BALD/probability varianceの標本集約、analytic log-EI/log-PI/UCB、MC qSimpleRegret/qUCBの独立参照を整理。共通posteriorやQMC base samplesの再利用による比較は推定量の収束・校正を証明しない。qNEI、EHVI、制約付き、lookahead等は未検証のまま保持。
- 408–510: Random/Sobol pool argmax、TuRBO success/failure/restart、ARD幾何、linear/nonlinear intra/inter-point constraint residualとfeasibility-first ordering、torch系の候補制約明示拒否を整理。最適化の大域収束・任意制約の実行可能性を主張しない。
- 全区間の過去PR番号、CI、exit decision、旧Phase番号は監査記録へ。教育用verificationガイドには検証命題・前提・再現手順・許容誤差・未検証境界を記載する。

**全174区間の本文レビュー完了（README 13、models 91、acquisition 18、optimization 21、verification 31）。移行・照合検証完了0件。**

## acquisition.md（1–178行、9区間）の本文レビュー

- 1–78: registryのlocal acquisition、BoTorch-native metadata、composition helper、Thompson selectionを別種として扱う。mixed one-shot optimizerは普通のmixed routeとは異なり、compatibility assertionは実際の最適化成功を保証しない。旧Phaseのcompletion gateは監査履歴へ。
- 79–114: regression ALはq=1のscalar marginal score。Variance、Std、Straddle、RandomizedStraddle、BoundaryVarianceの式・係数・random coefficient cacheを区別。ensembleと非scalar outputのguardを保持。
- 115–143: classification ALのPredictiveEntropy、Margin、ProbabilityVariance、BALD、LatentStraddleはclass probability、label entropy、latent uncertaintyを混同しない。すべてq=1。
- 144–178: regression EPIGはfinite target set上の二点Gaussian mutual information平均。joint covarianceとnoise補正を必要とし、classification BALDや一般非Gaussian EPIGと同一視しない。
- 原文のnamed testsはassertion inspectionであり、CI実行・独立数値検証とは別。

**累計:** README 13 + models 91 + acquisition 9 = 113/174区間。未レビュー61区間。本文移行・検証完了0件。
