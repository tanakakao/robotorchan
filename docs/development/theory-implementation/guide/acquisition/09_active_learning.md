# Active Learning：数式とローカル実装の接続

## 1. 何を最大化するか

Active Learning (AL) は目的関数の最大値を探すBOとは区別し、未観測の応答・分類境界について情報を得るために候補を選ぶ。[理論9章](../../../../theory/acquisition/09_active_learning.md)を数理的正本とする。

回帰ではposteriorの平均 `mu(x)` と分散 `v(x)` を取り、`PosteriorVariance(x)=v(x)`、`PosteriorStd(x)=sqrt(max(v(x),0))` を評価する。この二つは非負分散上で順位は単調に対応するが、utilityの**尺度**は異なる。これらはq=1の周辺情報量であり、q個を同時に評価するjoint acquisitionではない。

分類では観測ラベル確率 `p(y|x,D)` と潜在GPの確率変数を区別する。ラベルのpredictive entropy `H(Y|x,D)=-sum_c p_c log p_c` と、モデルの`mutual_information`で評価するBALDは同じ計算ではない。

## 2. 実装への対応

| 公開Acquisition | 数学的score / 参照 | 実装所有権と使用するモデルAPI |
| --- | --- | --- |
| `PosteriorVariance` | `v(x)` | `active_learning/variance.py::PosteriorVariance.forward`でImplemented。momentsは`model.posterior(X)`へDelegated |
| `PosteriorStd` | `sqrt(max(v(x),0))` | `variance.py::PosteriorStd.forward`がvarianceの返値にclampとsqrtを適用（Extended） |
| `PredictiveEntropy` | `-sum p_c log p_c` | `classification.py::PredictiveEntropy.forward`が`predict_proba`を呼びentropyを計算 |
| `MarginUncertainty` | `1-(p_(1)-p_(2))` | `classification.py::MarginUncertainty.forward`が上位2クラスの確率差を使用 |
| `ProbabilityVariance` | class-probability varianceのclass平均 | `classification.py`からモデルの`probability_variance(X,num_samples)`へDelegated |
| `BALD` | predictive entropyと期待conditional entropyの差に対応するmutual information | `classification.py`から`mutual_information(X,num_samples)`へDelegated。posterior latent varianceではない |
| `LatentStraddle` | `beta * sigma_latent - abs(mu_latent)` | `classification.py`で`latent_posterior(X)`に基づき境界を評価 |

`Straddle`、`RandomizedStraddle`、`BoundaryVariance`は[Level Set章](10_level_set.md)へ、`ExpectedPredictiveInformationGain`は[情報理論章](05_information_theoretic.md)へ分ける。上表のALクラスはBoTorch-native EI等とは異なるローカル実装である。実装の全監査記録は[Acquisition Mapping](../../acquisition.md)のPhase 5 regression/classification AL区間に残す。

## 3. 入出力shapeと拒否条件

`X`の形は通常`batch_shape × 1 × d`であり、acquisitionの返値はcandidate batchに対するscalar scoreとなる。対象のregression/classification AL registryはいずれも`max_q=1`。よって`q=2`のjoint最適化を保証しない。

`PosteriorVariance`はmulti-output posteriorで`output_index`が無ければ明示的に拒否する。指定時はその出力次元だけを抽出する。またstructured-output posteriorやensemble posteriorも明示的に拒否する（decoratorがあってもguardは優先される）。

分類の`PredictiveEntropy`は`predict_proba`を要求する。確率varianceとBALDは`num_samples>=2`を要求し、対応するモデルAPIが無ければconstruct時にTypeErrorとなる。単にGPのlatent `posterior`があれば分類ALを計算できる、という契約ではない。

## 4. 具体的な使い分け

回帰の関数値をできるだけ広く学習する単一点候補には`PosteriorVariance(model)`が基準となる。未知境界`f(x)=t`の周辺へ集中したい場合は[Level Set章](10_level_set.md)のStraddle系を選ぶ。分類ラベルの不確実性を狙う場合は`PredictiveEntropy(classifier)`、モデルパラメータ不確実性によるラベル情報を狙う場合は`BALD(classifier)`とし、予測確率の意味論を確認する。

この使い分けは推薦基準であり、特定のモデル・optimizer組合せの実行可能性を保証しない。AL optimizerはcandidate constraints、continuous/discrete、batch shapeに別途対応している必要がある。

## 5. 数学的検証と未検証の境界

[検証索引](../verification/acquisition.md)で参照する`tests/acquisition/test_phase8_regression_al_formula_reference.py`は固定float64のposterior moment scoreを独立数式と比較する。分類entropyとlatent boundaryには`test_phase8_classification_al_reference.py`、BALDの固定64サンプルmomentには`test_phase8_bald_probability_variance_reference.py`がある。

検証されたのは**指定したscore計算**であり、posteriorの較正、MC収束、joint q-batchのAL最適性は証明されない。旧Mappingのassertion・数値tol・未対応条件は移行台帳が検収されるまで保全する。
