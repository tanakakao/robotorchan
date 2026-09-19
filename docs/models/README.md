# Model guides

このディレクトリは、robotorchan の public model を問題設定・モデルファミリー単位で説明する利用者向けガイドです。

[Getting Started のモデル選択](../getting_started/model_selection.md) で候補を絞り、このディレクトリで個別モデルの違いを確認し、[Theory](../theory/README.md) で統計的仮定を確認してください。

## ファミリー

| ガイド | 主な対象 |
|---|---|
| [Standard / Fidelity / Variational](standard.md) | 標準回帰、Mixed、Multi-Fidelity、大規模データ |
| [Multi-task / Multi-output](multitask_multioutput.md) | task共有、block design、独立出力、heterogeneous task |
| [High-dimensional](high_dimensional.md) | SAAS、MAP-SAAS、additive、ALEBO |
| [Reduced](reduced.md) | PCA、PLS、Random Projection、AE/VAE、Joint、Mixed、Multi-task |
| [Robust / Noise](robust_noise.md) | outlier、heavy tail、heteroskedastic、replicate noise、nonstationary |
| [Input uncertainty](uncertain_input.md) | continuous / categorical input uncertainty |
| [Structured output](structured_output.md) | HigherOrder、LatentKronecker |
| [Hierarchical / Contextual](hierarchical_contextual.md) | conditional space、heterogeneous task、context |
| [Preference](preference.md) | pairwise preference learning |

## 共通規約

多くの supervised wrapper は raw training data、`supports_mll`、`make_mll()` を共通化します。ただし、Variational GP、Fully Bayesian SAAS、Pairwise GP などは学習契約が異なります。モデル固有の契約を優先してください。

この再編ではファイル数削減を目的にせず、public model の用途・仮定・制約・関連する Theory / Notebook への到達性を維持します。機械可読な対応表は [model_coverage.json](../model_coverage.json) です。
