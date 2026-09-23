# High-dimensional models

高次元問題では、surrogate の仮定と candidate search strategy を分けて考えます。このページは surrogate 側を扱います。REMBO / HeSBO / ALEBO / TuRBO / BAxUS などの探索側は [高次元探索戦略](../optimization/high_dimensional_search.md) を参照してください。

## Fully Bayesian SAAS

`SaasFullyBayesianSingleTaskGP`、`SaasFullyBayesianMultiTaskGP` とそれぞれの Mixed 版は、少数の入力次元のみが重要という sparsity 仮定を SAAS prior で表します。NUTS を利用し、`supports_mll=False` です。

## MAP-SAAS

`AdditiveMapSaasSingleTaskGP`、`EnsembleMapSaasSingleTaskGP` と Mixed 版は、NUTS より軽量な MAP 系の高次元候補です。計算量と posterior uncertainty の扱いは Fully Bayesian SAAS と同一ではありません。

## High-dimensional × Multi-Fidelity

高次元の design 変数と fidelity 変数を同時に扱う場合、fidelity を単なる削減対象へ含めません。robotorchan では design と structural fidelity の役割を分離します。

- `PCAMultiFidelityGP`: design 次元だけを PCA で削減します。
- `PLSMultiFidelityGP`: design 次元だけを PLS で削減します。
- `RandomProjectionMultiFidelityGP`: design 次元だけを random projection します。
- `MapSaasMultiFidelityGP`: design covariance に MAP-SAAS を使い、BoTorch の native fidelity covariance と合成します。
- `AdditiveMapSaasMultiFidelityGP`: public BoTorch additive MAP-SAAS covariance factory を design 次元へ限定し、native fidelity covariance と合成します。

これらの public model は posterior / sampling に加えて、代表的な `qMultiFidelityKnowledgeGradient`、cost utility、target-fidelity projection の runtime evidence を持ちます。PCA / PLS / Random Projection 版は raw-space conditioning と fantasize も明示的に検証しています。

Ensemble MAP-SAAS × Multi-Fidelity は、member-specific SAAS state と `GaussianMixturePosterior` を維持する必要があり、Additive版と同じ薄い covariance wrapper にはできません。実装価値はありますが、専用 prototype と実装・保守コスト評価を経て判断する候補です。

## Orthogonal additive structure

`OrthogonalAdditiveGP` と `MixedOrthogonalAdditiveGP` は低次の加法構造を仮定します。単に「高次元だから」ではなく、目的関数が低次相互作用へ分解できるという構造仮定が妥当かを確認します。

## ALEBO surrogate

`ALEBOGP` は ALEBO の埋め込みに対応する full Mahalanobis metric を持つ surrogate です。単独の汎用 high-dimensional GP としてではなく、ALEBO search strategy と組み合わせる設計意図を確認してください。

## Relevance pursuit

`RobustRelevancePursuitSingleTaskGP` と Mixed 版は robust observation のためのモデルであり、「高次元」という理由だけで選ぶモデルではありません。robust性の詳細は [Robust / Noise](robust_noise.md) を参照してください。

Notebookは代表例を共有します。ALEBOは独立したNotebookを持たず、現在は高次元モデル/探索ガイドと benchmark を主な実行導線とします。

Notebook: [SAAS](../../examples/notebooks/08_saas_gp.ipynb)、[MAP-SAAS / Additive](../../examples/notebooks/09_map_saas_and_additive_gp.ipynb)、[Multi-Fidelity](../../examples/notebooks/03_multi_fidelity_gp.ipynb)  
Theory: [High-dimensional GP](../theory/08_high_dimensional_gp.md)、[Advanced high-dimensional models](../theory/21_advanced_high_dimensional_models.md)、[Multi-Fidelity](../theory/06_multi_fidelity.md)
