# 回帰 Active Learning

連続回帰出力の不確実性を使う選択です。分類専用の獲得関数とは区別します。

## 点ごとの不確実性

q=1 の `PosteriorVariance` は事後分散、`PosteriorStd` は事後標準偏差を最大化します。いずれも BoTorch `AcquisitionFunction` に従い、単一候補の選択に `optimize_acqf` を使えます。

```python
from robotorchan.acquisition import PosteriorVariance

acqf = PosteriorVariance(model)
```


多出力では `output_index` を指定します。問題固有の効用を勝手に課すことになるため、出力の合計・平均を暗黙に取りません。

## Batch の積分分散削減

純粋探索には BoTorch 標準の `qNegIntegratedPosteriorVariance` を直接使います。

```python
from botorch.acquisition.active_learning import qNegIntegratedPosteriorVariance

acqf = qNegIntegratedPosteriorVariance(model=model, mc_points=target_points)
```


積分点上の事後不確実性の期待削減量で batch を評価します。wrapper や再公開は追加しません。

## 適用範囲

回帰の Active Learning 用です。level-set・境界学習、EPIG など予測情報量に基づく選択は別のガイドを参照してください。[理論](../theory/acquisition/09_active_learning.md)に関係を示します。

## 確率的な non-GP surrogate

意味のある周辺モーメントを公開する non-GP にも `PosteriorVariance`・`PosteriorStd` を使えます。`NGBoostSurrogate` は Gaussian 予測分布からモーメントを供給する基準経路です。

すべての Gaussian posterior 用獲得関数が使える意味ではありません。`ExpectedPredictiveInformationGain` は joint Gaussian 共分散を必要としますが、通常の NGBoost は独立な予測周辺分布だけを公開します。予測分散だけでは epistemic・aleatoric 不確実性を分離できないため、BALD もこの契約の対象外です。

## 経験 ensemble surrogate

現在の `PosteriorVariance`・`PosteriorStd` は ensemble posterior を拒否します。`_is_ensemble` のモデルと BoTorch `EnsemblePosterior` の直接の返り値を含みます。

tree ensemble のメンバー間の広がりは、この獲得関数の不確実性契約としては認定していません。RandomForest・ExtraTrees の Active Learning には ensemble を明示的に扱う別の基準と検証が必要です。[理論](../theory/acquisition/09_active_learning.md)と[Notebook](../../examples/notebooks/28_active_learning_acquisition.ipynb)に示す実行範囲と一致させています。
