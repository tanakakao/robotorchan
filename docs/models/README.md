# Model guides

このディレクトリは、robotorchan の public model を問題設定・モデルファミリー単位で説明する利用者向け詳細ガイドです。
`docs/models.md` は全体索引と共通APIの入口、このディレクトリは family 固有の用途・仮定・制約・variant 差分の source of truth とします。

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
| [Expressive GP](expressive.md) | DKL、DeepGP、NNGP、Spectral Mixture |

## 共通規約

多くの supervised wrapper は raw training data、`supports_mll`、`make_mll()` を共通化します。ただし、Variational GP、Fully Bayesian SAAS、Pairwise GP などは学習契約が異なります。モデル固有の契約を優先してください。

この再編ではファイル数削減を目的にせず、public model の用途・仮定・制約・関連する Theory / Notebook への到達性を維持します。機械可読な対応表は [model_coverage.json](../model_coverage.json) です。

## Theory との責務分離

このディレクトリでは public model の選択・利用に必要な情報を優先します。数式や推論原理を省略するのではなく、詳細な導出や一般理論は対応する [Theory](../theory/README.md) へ集約し、モデルガイドから必ず到達できるようにします。

各モデルファミリーでは、少なくとも「用途」「主要な統計的仮定」「入力・出力契約」「学習方法」「適用上の制約」「関連する Mixed / MultiTask / reduced variant」「Theory」「Notebook」を追跡可能にします。
