# Optimization guides

surrogate model とは独立した candidate search / optimization の利用者向けガイドです。

- [Standard acquisition functions](standard_acquisition.md): single-objective BO の推奨獲得関数とBoTorch直接利用方針
- [Thompson sampling](thompson_sampling.md): finite candidate set向けposterior samplingと連続探索との責務分離
- [Regression active learning](regression_active_learning.md): posterior uncertaintyとintegrated variance reduction
- [Level-set and boundary learning](level_set_learning.md): Straddleと境界推定
- [Information-theoretic acquisitions](information_theoretic.md): MES、GIBBON、Randomized Straddle
- [High-dimensional search](high_dimensional_search.md): Original space、Random Search、latent search、REMBO、HeSBO、ALEBO、TuRBO、BAxUS
- [TuRBO](turbo.md): stateful trust-region strategy の詳細

高次元 surrogate の選択は [High-dimensional models](../models/high_dimensional.md) を参照してください。
