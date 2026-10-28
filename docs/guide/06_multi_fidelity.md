# Multi-Fidelity GP：設計共分散と忠実度カーネルの接続

## この章で学ぶこと

[Multi-FidelityのTheory](../theory/06_multi_fidelity.md)の6.5・6.13・6.14節を、
設計入力と忠実度入力に対する共分散部品の構成へ対応付けます。
モデルが評価レベル間の依存を表す処理と、獲得関数が評価費用を考慮する処理を分けて読みます。

忠実度は評価レベルを表す入力座標です。
観測分散を与える`train_Yvar`とも、実験の費用とも異なります。
忠実度に応じた共分散を使うだけでは、評価値の単調性や安価な評価の有用性は保証されません。

## 1. クラス構造と数式の対応

[models/standard/multi_fidelity.py][local]の`SingleTaskMultiFidelityGP`は、
`ExactGPModelMixin`とBoTorchの`SingleTaskMultiFidelityGP`を継承します。
raw観測のコピーとMLL構成の共通入口を加え、確率モデルの構成と予測は親へ委譲します。
親クラスはBoTorchの`SingleTaskGP`を拡張しています。

| Theory上の構成要素 | 計算を担当する部品 |
| --- | --- |
| 設計入力の共分散 | 非忠実度列を使うカーネル、または指定された`covar_module` |
| データ量などの忠実度 | `DownsamplingKernel` |
| 反復回数などの忠実度 | `ExponentialDecayKernel` |
| 目標過程と忠実度依存の偏り | `LinearTruncatedFidelityKernel` |
| 観測への条件付け | 共分散を受け取るGPyTorchのExact GP推論 |
| 費用と目標忠実度への意思決定 | モデルの外に置くcost-aware utilityと獲得関数 |

BoTorchの[gp_regression_fidelity.py][b-models]にある
`_setup_multifidelity_covar_module()`が、指定された忠実度の種類と
`linear_truncated`に応じて共分散部品を組み立てます。
同じモデル名でも、この分岐によって部品の構造が異なります。

## 2. 分離したカーネルを積にする経路

`linear_truncated=False`では、忠実度列を除いた入力列から設計共分散を作ります。
指定された設計カーネルがあればそれを使い、なければ
`get_covar_module_with_dim_scaled_prior()`を使います。
反復忠実度には`ExponentialDecayKernel`、データ忠実度の各列には
`DownsamplingKernel`を加え、全体を`ProductKernel`で接続します。
この経路に、最上位の`ScaleKernel`を必ず追加するわけではありません。

`DownsamplingKernel.forward()`は、忠実度sについて`1-s`を作り、
`1 + power`乗した値の積へoffsetを加えます。
単一の忠実度列なら、その積が忠実度に依存する残差共分散になります。
s=1では残差の因子が消えますが、offsetまで零になるわけではありません。
これは全予測分散や測定ノイズが消えるという意味でもありません。
この部品は入力をclipしないため、忠実度の尺度も式が適用できる範囲にそろえる必要があります。

`ExponentialDecayKernel.forward()`は、長さ尺度で割った入力と
符号を反転した相手側の入力との距離を作り、`(1 + distance)`の負のpower乗へoffsetを加えます。
非負の単一忠実度では入力の和に対応する量を使い、通常のRBFの差による距離とは異なります。
具体的な計算は[BoTorchのkernels][b-kernels]内の
`downsampling.py`と`exponential_decay.py`にあります。

## 3. 目標過程と偏りを組み合わせる経路

`linear_truncated=True`では、`LinearTruncatedFidelityKernel`を構成します。
この部品は忠実度だけの関数ではなく、設計入力側の
`covar_module_unbiased`と`covar_module_biased`も所有します。
`forward()`は忠実度を0〜1にclipした値から`bias_factor`を計算し、
偏りを表す共分散へ掛けて目標側の共分散に加えます。

この構造は、Theoryで説明する目標過程と評価レベルに依存するずれを、
別の共分散部品に分ける箇所に対応します。
一つの設計共分散と忠実度共分散を単に掛ける経路と同一ではありません。
実装は`linear_truncated_fidelity.py`にあります。

setup関数は、これらの部品を積でまとめた後に`ScaleKernel`で包みます。
設計カーネルもこの部品が所有するため、同時に別の`data_covar_module`を渡すことは拒否します。
この所有関係が、独自の設計カーネルを組み込む派生モデルの制約につながります。

## 4. robotorchanが設計側を拡張するモデル

| 派生モデル | 設計側の構成 |
| --- | --- |
| `MixedSingleTaskMultiFidelityGP` | 共通mixedヘルパーで連続・カテゴリ・交互作用を構成 |
| `MapSaasMultiFidelityGP` | 設計列だけのMatérnへ`add_saas_prior()`を適用 |
| `AdditiveMapSaasMultiFidelityGP` | `get_additive_map_saas_covar_module()`で設計共分散を構成 |

これらは`SingleTaskMultiFidelityGP`を拡張し、`linear_truncated=True`を拒否します。
独自に構成した設計カーネルと、親が構成する忠実度カーネルを積で接続するためです。
忠実度カーネルの数値演算をrobotorchan側で複製する構成ではありません。

Mixed版は、反復・データ忠実度の列を集め、
`normalize_feature_dims()`と共通mixedヘルパーで設計カテゴリ列から除外します。
忠実度列同士の重複や、カテゴリ列との重なりも拒否します。
構造列を通常の設計カテゴリとして二重に解釈することを防ぎます。

MAP-SAAS版は、忠実度以外の`design_dims`にMatérnを置き、
SAASの事前分布を加えて`ScaleKernel`で包みます。
Additive版は、同じ設計列に複数のSAASカーネルを加算します。
設計列がない場合は構成できず、忠実度列の指定や重複も検査します。
現行の二つのSAAS派生経路では、`data_fidelities`の正規化で非空の列集合を要求します。
反復忠実度だけを指定した場合も、この列検査で拒否されます。
これは多忠実度モデル一般の数学的な必要条件ではなく、現行実装の範囲です。
SAASによる縮小の仕組みは[高次元GP Guide](08_high_dimensional_gp.md)を参照してください。

## 5. 学習・予測と費用の境界

`make_mll()`は、[共通mixin][base]が構成する`ExactMarginalLogLikelihood`です。
標準Exact GPと同じく、学習目的の生成とパラメータ更新は別の操作です。
その後の条件付けは、組み立てた共分散を使って上流の推論が行います。

`posterior()`は、渡された入力の忠実度での分布を返します。
全てを自動的に目標忠実度へ射影する処理ではありません。
目標忠実度への射影、現在価値の計算、費用による効用の調整は、
[多忠実度の獲得関数Guide](acquisition/11_multifidelity_cost_aware.md)の別の部品です。
Theory 6.7〜6.9・6.19〜6.22節の区別を、実装の担当範囲にも保ちます。

## 6. 実装の担当範囲と参照先

標準モデルと忠実度カーネルの計算はDelegatedです。
raw観測保持とMLL構成はExtended、設計列の検査と派生モデルの部品構成は
robotorchanのImplementedに当たります。
モデル、ターゲットへの写像、費用、候補探索を分けてソースを読むことが重要です。

使用・可視化は[Multi-Fidelity Notebook][notebook]、
カーネルの数値的な性質は[参照テスト][reference-test]、
mixed構成の境界は[Mixedモデルのテスト][mixed-test]を参照できます。
数学的な文献はTheoryの参考文献へ集約しています。
上流コードの参照先はBoTorch 0.18.1です。

[local]: ../../src/robotorchan/models/standard/multi_fidelity.py
[base]: ../../src/robotorchan/models/base.py
[b-models]: https://github.com/pytorch/botorch/tree/v0.18.1/botorch/models
[b-kernels]: https://github.com/pytorch/botorch/tree/v0.18.1/botorch/models/kernels
[notebook]: ../../examples/notebooks/03_multi_fidelity_gp.ipynb
[reference-test]: ../../tests/models/standard/test_phase7_multifidelity_kernel_reference.py
[mixed-test]: ../../tests/models/standard/test_mixed_multi_fidelity.py
