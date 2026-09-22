# Changelog

robotorchanの主な変更を記録します。

形式は [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) を参考にし、
リリース番号は [Semantic Versioning](https://semver.org/) に従います。

## [Unreleased]

### Added

- BoTorch-first の共通モデル契約として raw training data 保持と `make_mll()` / `supports_mll` を整備。
- Continuous / Mixed、Multi-task / Kronecker、Multi-Fidelity、Variational、Fully Bayesian SAAS を含む surrogate model 群。
- PCA / PLS / Random Projection、AE / VAE、supervised / joint / hybrid encoder などの高次元・次元削減モデル。
- Student-t、Contaminated、Relevance Pursuit、heteroskedastic、replicate-noise、uncertain-input、nonstationary などの robust / noise / uncertainty モデル。
- DeepGP、Infinite-width BNN GP、Spectral Mixture GP などの expressive surrogate。
- Random Forest、Extra Trees、Gradient Boosting、Histogram Gradient Boosting の Non-GP surrogate。
- REMBO、HeSBO、ALEBO、TuRBO、BAxUS、tree-ensemble search などの探索戦略。
- Posterior variance / std、Straddle、Randomized Straddle、Boundary Variance、EPIG、Thompson candidate selection などの Active Learning / acquisition 拡張。
- Public model API を model guide、Theory、代表 Notebook に対応付ける `docs/model_coverage.json`。
- Acquisition Function の詳細理論を独立した `docs/theory/acquisition/` として整備。
- Getting Started、Models、Optimization、Theory、Benchmarks、Development に責務分離したドキュメント構成。
- Public library metadata、BSD-3-Clause license、package / release workflow、OSS project files。

### Changed

- Mixed model API を raw input + `cat_dims` に統一し、利用者側の one-hot 前処理を要求しない設計へ整理。
- 互換 alias / deprecated wrapper を残さず、公開 API・テスト・Notebook・ドキュメントを現行仕様へ同期。
- README とドキュメント索引を現在の public API、23の主要 Theory 章、25 Notebook に同期。

## [0.1.0] - Unreleased

Initial public release. The final release date and release notes will be fixed when `v0.1.0` is published.

[Unreleased]: https://github.com/tanakakao/robotorchan/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tanakakao/robotorchan/releases/tag/v0.1.0
