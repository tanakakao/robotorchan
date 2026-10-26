# Phase 3 — 原文174区間から日本語ガイドへの移行割当台帳

> 作業記録。原文を削除・改名しない。**未レビュー行のprimaryは見出しベースの暫定割当**であり、`verified`は0件。機械的に分類した結果を移行完了と誤認しないこと。

## 入力と判定基準

- 入力: Phase 1の174見出し台帳（5ファイル、元行範囲・元blob SHAは原台帳に保持）。
- 章構成: `docs/development/theory-implementation/guide/README.md`の24+12+12章。
- 一次移行先は**本文を読み進める主章**、二次移行先は横断的な技術契約や監査情報の保全先。
- **見出し語による暫定割当**のため、本文に複数テーマが混在する区間は後続の逐条レビューで分割する。
- `verification.md`は全区間をverificationに暫定割当し、propertyごとの対応章を本文レビューで追加する。
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
| `models.md` | 799–806 | Phase 4: theory-to-implementation records | `guide/appendix/model-composition-and-delegation.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 807–821 | Exact single-task Gaussian-process regression | `guide/02_gaussian_process.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 822–836 | Native mixed-input Gaussian-process regression | `guide/05_mixed_variables.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 837–851 | Bernoulli variational Gaussian-process classification | `guide/24_classification.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 852–866 | Long-format multitask Gaussian-process regression | `guide/07_multitask_multioutput.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 867–881 | Kronecker block-design multitask GP | `guide/07_multitask_multioutput.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 882–896 | SAAS fully Bayesian single-task GP | `guide/08_high_dimensional_gp.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 897–911 | Stochastic deep Gaussian-process regression | `guide/appendix/model-composition-and-delegation.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 912–926 | Random-forest ensemble regression | `guide/23_non_gp_surrogates.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 927–941 | Reduced input/output Gaussian-process regression | `guide/18_dimensionality_reduction_gp.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 942–956 | Student-t robust variational regression | `guide/14_robust_gaussian_process.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 957–971 | NGBoost probabilistic non-GP regression | `guide/23_non_gp_surrogates.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 972–986 | Heterogeneous regression/classification composition | `guide/24_classification.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 987–1000 | Classification with flipped observation labels | `guide/24_classification.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 1001–1014 | Classification with contaminating labels | `guide/24_classification.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 1015–1028 | Classification probability ensembles (GP and heterogeneous) | `guide/24_classification.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 1029–1042 | Non-GP binary classification and bootstrap uncertainty | `guide/24_classification.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 1043–1056 | Classification with uncertain candidate inputs | `guide/24_classification.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 1057–1070 | Multitask binary classification (long format and block input) | `guide/24_classification.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 1071–1084 | Frozen reduced-input binary GP classification | `guide/24_classification.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 1085–1099 | Deep variational GP binary classification | `guide/24_classification.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 1100–1131 | Remaining robust binary classification families | `guide/24_classification.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 1132–1168 | Sparse and embedded high-dimensional binary classification | `guide/24_classification.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 1169–1206 | Phase 4 registry-wide reconciliation: outstanding detailed evidence | `guide/appendix/model-registry-and-public-api.md` | `audit-history` | **planned / needs body review** |
| `models.md` | 1207–1224 | Phase 4 exit gate (unchanged) | `guide/appendix/model-composition-and-delegation.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 1225–1242 | Phase 4 inspected binary variant reconciliation: mixed and nonstationary | `guide/24_classification.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 1243–1269 | Phase 4 inspected reduced-GP family mapping: frozen, mixed and joint | `guide/05_mixed_variables.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 1270–1293 | Phase 4 inspected neural and variational reduced-GP variants | `guide/09_variational_gp.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 1294–1321 | Phase 4 inspected reduced multi-task implementations | `guide/07_multitask_multioutput.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 1322–1344 | Phase 4 inspected jointly trained multi-task and design-reduced multi-fidelity GP | `guide/06_multi_fidelity.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 1345–1369 | Phase 4 reduced-family registry reconciliation (52 entries) | `guide/18_dimensionality_reduction_gp.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 1370–1389 | Phase 4 inspected robust-noise heteroskedastic inference paths | `guide/15_heteroskedastic_noise.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 1390–1404 | Phase 4 inspected contamination, replicate-noise and nonstationary regression models | `guide/15_heteroskedastic_noise.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 1405–1420 | Phase 4 inspected Student-t robust regression variants | `guide/14_robust_gaussian_process.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 1421–1437 | Phase 4 inspected robust relevance-pursuit regression variants | `guide/14_robust_gaussian_process.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 1438–1460 | Phase 4 robust-noise registry reconciliation (27 entries) | `guide/15_heteroskedastic_noise.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 1461–1477 | Phase 4 inspected uncertain-input registry models (3 entries) | `guide/appendix/model-registry-and-public-api.md` | `audit-history` | **planned / needs body review** |
| `models.md` | 1478–1493 | Phase 4 inspected structured-output regression registry (4 entries) | `guide/appendix/model-registry-and-public-api.md` | `audit-history` | **planned / needs body review** |
| `models.md` | 1494–1516 | Phase 4 inspected hierarchical/contextual registry (6 entries) | `guide/appendix/model-registry-and-public-api.md` | `audit-history` | **planned / needs body review** |
| `models.md` | 1517–1530 | Phase 4 inspected preference-learning registry (1 entry) | `guide/appendix/model-registry-and-public-api.md` | `audit-history` | **planned / needs body review** |
| `models.md` | 1531–1547 | Phase 4 inspected standard single-task, variational and multi-fidelity models | `guide/06_multi_fidelity.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 1548–1564 | Phase 4 inspected non-GP surrogate registry (5 entries) | `guide/appendix/model-registry-and-public-api.md` | `audit-history` | **planned / needs body review** |
| `models.md` | 1565–1581 | Phase 4 inspected core multitask and model-list registry contracts | `guide/07_multitask_multioutput.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 1582–1600 | Phase 4 inspected specialized heterogeneous/hierarchical multitask models | `guide/07_multitask_multioutput.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 1601–1620 | Phase 4 inspected expressive spectral and infinite-width NNGP models (12 entries) | `guide/22_expressive_gp.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 1621–1637 | Phase 4 inspected expressive DeepGP hierarchy (4 entries) | `guide/12_hierarchical_contextual_gp.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 1638–1656 | Phase 4 inspected high-dimensional SAAS, MAP-SAAS and ALEBO ownership | `guide/08_high_dimensional_gp.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 1657–1672 | Phase 4 inspected orthogonal additive high-dimensional GP variants | `guide/08_high_dimensional_gp.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 1673–1687 | Phase 4 inspected MAP-SAAS multi-fidelity model variants | `guide/06_multi_fidelity.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 1688–1734 | Phase 4 cross-family registry reconciliation: high-dimensional and multitask | `guide/07_multitask_multioutput.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 1735–1785 | Phase 4 inventory-count closure audit (static, documentation-only) | `guide/appendix/model-registry-and-public-api.md` | `audit-history` | **planned / needs body review** |
| `models.md` | 1786–1796 | KroneckerMultiTaskGP covariance verification reference | `guide/07_multitask_multioutput.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `acquisition.md` | 1–4 | Acquisition theory–implementation mapping: Phase 3 inventory | `guide/acquisition/01_foundations.md` | `appendix/acquisition-contracts-and-compatibility.md` | **planned / needs body review** |
| `acquisition.md` | 5–18 | Sources and evidence boundary | `guide/acquisition/10_level_set.md` | `appendix/acquisition-contracts-and-compatibility.md` | **planned / needs body review** |
| `acquisition.md` | 19–38 | Distinct implementation paths | `guide/acquisition/01_foundations.md` | `appendix/acquisition-contracts-and-compatibility.md` | **planned / needs body review** |
| `acquisition.md` | 39–47 | Acquisition/optimizer interaction | `guide/acquisition/01_foundations.md` | `appendix/acquisition-contracts-and-compatibility.md` | **planned / needs body review** |
| `acquisition.md` | 48–60 | Fixed Phase 3 completion criteria | `guide/acquisition/01_foundations.md` | `appendix/acquisition-contracts-and-compatibility.md` | **planned / needs body review** |
| `acquisition.md` | 61–78 | Inspected acquisition registry and assertion evidence | `guide/acquisition/01_foundations.md` | `appendix/acquisition-contracts-and-compatibility.md` | **planned / needs body review** |
| `acquisition.md` | 79–114 | Phase 5 — regression active-learning acquisition mappings (source inspected) | `guide/acquisition/09_active_learning.md` | `appendix/acquisition-contracts-and-compatibility.md` | **planned / needs body review** |
| `acquisition.md` | 115–143 | Phase 5 — classification AL and Gaussian EPIG (source inspected) | `guide/acquisition/09_active_learning.md` | `appendix/acquisition-contracts-and-compatibility.md` | **planned / needs body review** |
| `acquisition.md` | 144–178 | ExpectedPredictiveInformationGain (regression) | `guide/acquisition/05_information_theoretic.md` | `appendix/acquisition-contracts-and-compatibility.md` | **planned / needs body review** |
| `acquisition.md` | 179–232 | Phase 5 — single-objective Monte Carlo BO (source inspected) | `guide/acquisition/04_batch_noisy.md` | `appendix/acquisition-contracts-and-compatibility.md` | **planned / needs body review** |
| `acquisition.md` | 233–249 | Phase 5 — multi-objective MC Bayesian optimization (source inspected) | `guide/acquisition/07_multiobjective.md` | `appendix/acquisition-contracts-and-compatibility.md` | **planned / needs body review** |
| `acquisition.md` | 250–287 | Verified construction contracts | `guide/acquisition/01_foundations.md` | `appendix/acquisition-contracts-and-compatibility.md` | **planned / needs body review** |
| `acquisition.md` | 288–308 | Phase 5 — outcome-constrained BO composition (source inspected) | `guide/acquisition/01_foundations.md` | `appendix/acquisition-contracts-and-compatibility.md` | **planned / needs body review** |
| `acquisition.md` | 309–345 | Interpretation and boundaries | `guide/acquisition/01_foundations.md` | `appendix/acquisition-contracts-and-compatibility.md` | **planned / needs body review** |
| `acquisition.md` | 346–362 | Phase 5 — information-theoretic and lookahead acquisitions (source inspected) | `guide/acquisition/06_lookahead.md` | `appendix/acquisition-contracts-and-compatibility.md` | **planned / needs body review** |
| `acquisition.md` | 363–405 | One-shot and information-target boundaries | `guide/acquisition/06_lookahead.md` | `appendix/acquisition-contracts-and-compatibility.md` | **planned / needs body review** |
| `acquisition.md` | 406–422 | Phase 5 — coverage reconciliation and exit audit | `guide/acquisition/01_foundations.md` | `appendix/acquisition-contracts-and-compatibility.md` | **planned / needs body review** |
| `acquisition.md` | 423–465 | Confirmed limits and carry-forward | `guide/acquisition/01_foundations.md` | `appendix/acquisition-contracts-and-compatibility.md` | **planned / needs body review** |
| `optimization.md` | 1–4 | Acquisition optimization theory–implementation mapping: Phase 3 inventory | `guide/optimization/01_foundations.md` | `appendix/optimizer-dispatch-and-constraints.md` | **planned / needs body review** |
| `optimization.md` | 5–19 | Baseline and separation of responsibilities | `guide/optimization/01_foundations.md` | `appendix/optimizer-dispatch-and-constraints.md` | **planned / needs body review** |
| `optimization.md` | 20–39 | Named optimization dispatcher | `guide/optimization/01_foundations.md` | `appendix/optimizer-dispatch-and-constraints.md` | **planned / needs body review** |
| `optimization.md` | 40–62 | Candidate-space constraints versus outcome constraints | `guide/optimization/05_constraints.md` | `appendix/optimizer-dispatch-and-constraints.md` | **planned / needs body review** |
| `optimization.md` | 63–76 | Stateful and coordinate-changing algorithms | `guide/optimization/09_high_dimensional.md` | `appendix/optimizer-dispatch-and-constraints.md` | **planned / needs body review** |
| `optimization.md` | 77–85 | Verification boundaries | `guide/optimization/01_foundations.md` | `appendix/optimizer-dispatch-and-constraints.md` | **planned / needs body review** |
| `optimization.md` | 86–99 | Inspected optimizer and strategy assertion evidence | `guide/optimization/01_foundations.md` | `appendix/optimizer-dispatch-and-constraints.md` | **planned / needs body review** |
| `optimization.md` | 100–107 | Fixed follow-up audit surface | `guide/optimization/01_foundations.md` | `appendix/optimizer-dispatch-and-constraints.md` | **planned / needs body review** |
| `optimization.md` | 108–118 | Specialized optimization paths: inspected source and assertions | `guide/optimization/01_foundations.md` | `appendix/optimizer-dispatch-and-constraints.md` | **planned / needs body review** |
| `optimization.md` | 119–133 | Coordinate-changing high-dimensional strategies | `guide/optimization/09_high_dimensional.md` | `appendix/optimizer-dispatch-and-constraints.md` | **planned / needs body review** |
| `optimization.md` | 134–140 | Remaining Phase 3 audit surface | `guide/optimization/01_foundations.md` | `appendix/optimizer-dispatch-and-constraints.md` | **planned / needs body review** |
| `optimization.md` | 141–152 | HeSBO, latent-space reconstruction and backend constraint semantics | `guide/optimization/09_high_dimensional.md` | `appendix/optimizer-dispatch-and-constraints.md` | **planned / needs body review** |
| `optimization.md` | 153–165 | Phase 3 evidence limits | `guide/optimization/01_foundations.md` | `appendix/optimizer-dispatch-and-constraints.md` | **planned / needs body review** |
| `optimization.md` | 166–181 | Phase 3 registry reconciliation gate | `guide/optimization/01_foundations.md` | `appendix/optimizer-dispatch-and-constraints.md` | **planned / needs body review** |
| `optimization.md` | 182–205 | Phase 6 — named optimizer mathematical and dispatch mapping | `guide/optimization/01_foundations.md` | `appendix/optimizer-dispatch-and-constraints.md` | **planned / needs body review** |
| `optimization.md` | 206–245 | Exact dispatch and mathematical limitations | `guide/optimization/01_foundations.md` | `appendix/optimizer-dispatch-and-constraints.md` | **planned / needs body review** |
| `optimization.md` | 246–257 | Phase 6 — specialist optimizers and candidate-constraint semantics | `guide/optimization/05_constraints.md` | `appendix/optimizer-dispatch-and-constraints.md` | **planned / needs body review** |
| `optimization.md` | 258–274 | Constraint handling is part of the algorithm | `guide/optimization/05_constraints.md` | `appendix/optimizer-dispatch-and-constraints.md` | **planned / needs body review** |
| `optimization.md` | 275–295 | Tests and mathematical verification boundary | `guide/optimization/01_foundations.md` | `appendix/optimizer-dispatch-and-constraints.md` | **planned / needs body review** |
| `optimization.md` | 296–314 | Phase 6 — stateful and coordinate-changing search strategies | `guide/optimization/09_high_dimensional.md` | `appendix/optimizer-dispatch-and-constraints.md` | **planned / needs body review** |
| `optimization.md` | 315–355 | Coordinate and constraint invariants | `guide/optimization/09_high_dimensional.md` | `appendix/optimizer-dispatch-and-constraints.md` | **planned / needs body review** |
| `verification.md` | 1–3 | Mathematical verification: Phases 7–9 | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **planned / needs body review** |
| `verification.md` | 4–26 | Purpose and evidence rules | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **planned / needs body review** |
| `verification.md` | 27–39 | Phase 7 — model verification matrix | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **planned / needs body review** |
| `verification.md` | 40–64 | Initial Phase 7 execution gate | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **planned / needs body review** |
| `verification.md` | 65–79 | Kronecker block-design covariance reference (Phase 7) | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **planned / needs body review** |
| `verification.md` | 80–93 | Data-fidelity downsampling kernel reference (Phase 7) | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **planned / needs body review** |
| `verification.md` | 94–107 | PCA reduced-space projection reference (Phase 7) | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **planned / needs body review** |
| `verification.md` | 108–121 | Replicate observation-noise statistical reference (Phase 7) | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **planned / needs body review** |
| `verification.md` | 122–134 | Binary GP probit predictive reference (Phase 7) | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **planned / needs body review** |
| `verification.md` | 135–148 | Non-GP Gaussian posterior reparameterization reference (Phase 7) | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **planned / needs body review** |
| `verification.md` | 149–183 | Phase 7 execution reconciliation and remaining scope | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **planned / needs body review** |
| `verification.md` | 184–197 | Reduced GP public/latent posterior coordinate reference (Phase 7) | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **planned / needs body review** |
| `verification.md` | 198–211 | Heteroskedastic latent-versus-observation noise reference (Phase 7) | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **planned / needs body review** |
| `verification.md` | 212–222 | Non-GP Random Forest estimator-level reference (Phase 7) | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **planned / needs body review** |
| `verification.md` | 223–244 | Phase 7 exit audit (property-specific) | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **planned / needs body review** |
| `verification.md` | 245–256 | Mixed reduced GP categorical-coordinate reference (Phase 7) | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **planned / needs body review** |
| `verification.md` | 257–273 | Heteroskedastic observation-variance composition boundary (Phase 7) | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **planned / needs body review** |
| `verification.md` | 274–291 | Phase 7 final exit reconciliation | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **planned / needs body review** |
| `verification.md` | 292–306 | Phase 8 — regression active-learning formula reference | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **planned / needs body review** |
| `verification.md` | 307–319 | Phase 8 — classification active-learning arithmetic reference | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **planned / needs body review** |
| `verification.md` | 320–334 | Phase 8 — BALD and probability-variance sample-moment reference | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **planned / needs body review** |
| `verification.md` | 335–350 | Phase 8 — native analytic BO Gaussian reference | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **planned / needs body review** |
| `verification.md` | 351–362 | Phase 8 — Monte Carlo qSimpleRegret sample reference | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **planned / needs body review** |
| `verification.md` | 363–375 | Phase 8 — Monte Carlo qUCB sample reference | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **planned / needs body review** |
| `verification.md` | 376–407 | Phase 8 — verification reconciliation (representative scope) | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **planned / needs body review** |
| `verification.md` | 408–421 | Phase 9 — sampling optimizer reference (initial property) | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **planned / needs body review** |
| `verification.md` | 422–435 | Phase 9 — TuRBO state-transition reference | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **planned / needs body review** |
| `verification.md` | 436–448 | Phase 9 — TuRBO trust-region geometry reference | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **planned / needs body review** |
| `verification.md` | 449–464 | Phase 9 — candidate constraint residual and feasibility ordering reference | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **planned / needs body review** |
| `verification.md` | 465–475 | Phase 9 — PyTorch optimizer candidate-constraint support boundary | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **planned / needs body review** |
| `verification.md` | 476–510 | Phase 9 — representative optimizer verification exit audit | `guide/verification/README.md` | `appendix/limitations-and-unverified.md` | **planned / needs body review** |

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

**累計:** README 13 + models 39 = 52/174区間レビュー済み。未レビュー122区間。移行・検証完了0件。

## 各区間の移行検収条件

1. 原文の全行を読み、数式、API、型、例外、candidate/outcome制約、実装ownership、upstream symbol、テストassertion、許容誤差、未検証事項を抽出する。
2. 第一・第二移行先を実際の本文に基づいて確定し、必要なら**1区間を複数章に分割**する。
3. 移行先の章には説明文と正確なソース・テストへの参照を記述する。表のリンクだけで移行済みにしない。
4. レジストリや静的存在チェックをruntime/数学的同値性の証拠に格上げしない。
5. 移行先の本文と原文を照合してから `planned → migrated → verified` と変更する。未確認のまま原文を削除しない。
6. 原文5ファイルと既存anchorを保全し、ドキュメントリンクCIを通す。

## 今回の作業範囲

このPhaseでは**移行対象の漏れ防止と一次配置の具体化**まで行う。174区間の逐語的な技術監査・48章の執筆は後続Phaseで実施する。移行先パスは未作成を含むため、Markdownリンクではなくコード表記とする。
