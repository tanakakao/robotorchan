# Phase 3 — 原文174区間から日本語ガイドへの移行割当台帳

> 作業記録。原文を削除・改名しない。**全行のprimaryは本文未レビューの暫定割当**であり、`verified`は0件。機械的に分類した結果を移行完了と誤認しないこと。

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
| `README.md` | 1–2 | Theory–Implementation Mapping: 12-phase development cycle | `guide/appendix/cross-cutting-implementation-contracts.md` | `audit-history` | **planned / needs body review** |
| `README.md` | 3–10 | Baseline and scope | `guide/appendix/cross-cutting-implementation-contracts.md` | `audit-history` | **planned / needs body review** |
| `README.md` | 11–23 | Existing ownership (preserve) | `guide/appendix/cross-cutting-implementation-contracts.md` | `audit-history` | **planned / needs body review** |
| `README.md` | 24–41 | Confirmed source-of-truth boundaries | `guide/appendix/cross-cutting-implementation-contracts.md` | `audit-history` | **planned / needs body review** |
| `README.md` | 42–66 | Reader navigation (Phase 10) | `guide/appendix/cross-cutting-implementation-contracts.md` | `audit-history` | **planned / needs body review** |
| `README.md` | 67–86 | Mapping record contract | `guide/appendix/cross-cutting-implementation-contracts.md` | `audit-history` | **planned / needs body review** |
| `README.md` | 87–94 | Mapping documentation layout | `guide/appendix/cross-cutting-implementation-contracts.md` | `audit-history` | **planned / needs body review** |
| `README.md` | 95–107 | Verification and remaining work | `guide/appendix/cross-cutting-implementation-contracts.md` | `audit-history` | **planned / needs body review** |
| `README.md` | 108–136 | Recorded phase decisions | `guide/appendix/cross-cutting-implementation-contracts.md` | `audit-history` | **planned / needs body review** |
| `README.md` | 137–174 | Phase 6 exit reconciliation (static scope) | `guide/appendix/cross-cutting-implementation-contracts.md` | `audit-history` | **planned / needs body review** |
| `README.md` | 175–201 | Phase 10 navigation exit audit | `guide/appendix/cross-cutting-implementation-contracts.md` | `audit-history` | **planned / needs body review** |
| `README.md` | 202–231 | Phase 11 mapping consistency exit audit | `guide/appendix/cross-cutting-implementation-contracts.md` | `audit-history` | **planned / needs body review** |
| `README.md` | 232–278 | Phase 12 final audit: evidence and decision | `guide/appendix/cross-cutting-implementation-contracts.md` | `audit-history` | **planned / needs body review** |
| `models.md` | 1–4 | Model theory–implementation mapping | `guide/appendix/model-composition-and-delegation.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 5–17 | Audit provenance and limits | `guide/appendix/model-registry-and-public-api.md` | `audit-history` | **planned / needs body review** |
| `models.md` | 18–31 | Registry boundaries | `guide/appendix/model-registry-and-public-api.md` | `audit-history` | **planned / needs body review** |
| `models.md` | 32–51 | Regression model inventory by existing guide | `guide/appendix/model-registry-and-public-api.md` | `audit-history` | **planned / needs body review** |
| `models.md` | 52–91 | Binary classification registry inventory | `guide/24_classification.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 92–106 | Representative code-path findings | `guide/appendix/model-composition-and-delegation.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 107–119 | Existing audit machinery | `guide/appendix/model-registry-and-public-api.md` | `audit-history` | **planned / needs body review** |
| `models.md` | 120–131 | Phase 2 public-export reconciliation | `guide/appendix/model-registry-and-public-api.md` | `audit-history` | **planned / needs body review** |
| `models.md` | 132–146 | Regression public symbols outside the registry | `guide/appendix/model-registry-and-public-api.md` | `audit-history` | **planned / needs body review** |
| `models.md` | 147–185 | Binary registry IDs and exported model classes | `guide/appendix/model-registry-and-public-api.md` | `audit-history` | **planned / needs body review** |
| `models.md` | 186–207 | Representative test assertion mapping | `guide/appendix/model-registry-and-public-api.md` | `audit-history` | **planned / needs body review** |
| `models.md` | 208–225 | Component-level implementation ownership | `guide/appendix/model-composition-and-delegation.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 226–235 | Limits of the current evidence | `guide/appendix/model-registry-and-public-api.md` | `audit-history` | **planned / needs body review** |
| `models.md` | 236–256 | Exhaustive static provenance export | `guide/appendix/model-registry-and-public-api.md` | `audit-history` | **planned / needs body review** |
| `models.md` | 257–271 | Documentation link existence validation | `guide/appendix/model-registry-and-public-api.md` | `audit-history` | **planned / needs body review** |
| `models.md` | 272–289 | Static test-reference discovery | `guide/appendix/model-registry-and-public-api.md` | `audit-history` | **planned / needs body review** |
| `models.md` | 290–315 | Inspected runtime composition and mathematical ownership | `guide/appendix/model-composition-and-delegation.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 316–338 | Inspected test assertions and verification boundaries | `guide/appendix/model-registry-and-public-api.md` | `audit-history` | **planned / needs body review** |
| `models.md` | 339–350 | Optional static upstream composition scan | `guide/appendix/model-registry-and-public-api.md` | `audit-history` | **planned / needs body review** |
| `models.md` | 351–367 | Conditional import provenance and static scan boundaries | `guide/appendix/model-registry-and-public-api.md` | `audit-history` | **planned / needs body review** |
| `models.md` | 368–393 | Phase 2 evidence levels and follow-up matrix | `guide/appendix/model-registry-and-public-api.md` | `audit-history` | **planned / needs body review** |
| `models.md` | 394–422 | Classification model documentation ownership | `guide/24_classification.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 423–441 | Source-inspected classification delegation: non-GP and label noise | `guide/24_classification.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 442–458 | Source-inspected classification: heterogeneous ensembles and uncertain inputs | `guide/24_classification.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 459–477 | Inspected classification test assertions: ensemble and uncertain input | `guide/24_classification.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 478–497 | Inspected classification test assertions: label noise and sklearn adapters | `guide/24_classification.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 498–518 | Source and assertion audit: binary multitask and DeepGP | `guide/24_classification.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 519–540 | Source and assertion audit: high-dimensional binary classifiers | `guide/24_classification.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 541–563 | Source and assertion audit: neural, ensemble, robust and categorical models | `guide/05_mixed_variables.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 564–588 | Source and assertion audit: contamination, replicates, bootstrap and robust composition | `guide/15_heteroskedastic_noise.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 589–606 | Repeatable registry-wide static evidence summary | `guide/appendix/model-registry-and-public-api.md` | `audit-history` | **planned / needs body review** |
| `models.md` | 607–625 | Inspected regression non-GP implementations: NGBoost and ExtraTrees | `guide/23_non_gp_surrogates.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 626–649 | Inspected regression noise and heterogeneous multitask implementations | `guide/07_multitask_multioutput.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 650–675 | Source and assertion audit: multi-fidelity regression families | `guide/appendix/model-registry-and-public-api.md` | `audit-history` | **planned / needs body review** |
| `models.md` | 676–697 | Source and assertion audit: regression DeepGP models | `guide/appendix/model-registry-and-public-api.md` | `audit-history` | **planned / needs body review** |
| `models.md` | 698–723 | Source and assertion audit: expressive stationary and infinite-width kernels | `guide/appendix/model-registry-and-public-api.md` | `audit-history` | **planned / needs body review** |
| `models.md` | 724–746 | Source and assertion audit: joint neural representation regression | `guide/appendix/model-registry-and-public-api.md` | `audit-history` | **planned / needs body review** |
| `models.md` | 747–773 | Classification documentation mapping at registry-ID granularity | `guide/24_classification.md` | `appendix/model-registry-and-public-api.md` | **planned / needs body review** |
| `models.md` | 774–798 | Phase 2 inventory sign-off and scope | `guide/appendix/model-registry-and-public-api.md` | `audit-history` | **planned / needs body review** |
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

## 各区間の移行検収条件

1. 原文の全行を読み、数式、API、型、例外、candidate/outcome制約、実装ownership、upstream symbol、テストassertion、許容誤差、未検証事項を抽出する。
2. 第一・第二移行先を実際の本文に基づいて確定し、必要なら**1区間を複数章に分割**する。
3. 移行先の章には説明文と正確なソース・テストへの参照を記述する。表のリンクだけで移行済みにしない。
4. レジストリや静的存在チェックをruntime/数学的同値性の証拠に格上げしない。
5. 移行先の本文と原文を照合してから `planned → migrated → verified` と変更する。未確認のまま原文を削除しない。
6. 原文5ファイルと既存anchorを保全し、ドキュメントリンクCIを通す。

## 今回の作業範囲

このPhaseでは**移行対象の漏れ防止と一次配置の具体化**まで行う。174区間の逐語的な技術監査・48章の執筆は後続Phaseで実施する。移行先パスは未作成を含むため、Markdownリンクではなくコード表記とする。
