# Classification and ordinal research gates

この文書は classification / ordinal model と対応する Active Learning を public API に
追加する前に満たすべき、現在有効な統計・実装契約を定義します。

## Binary GP classification

binary classifier は次の二つを明示的に分離します。

1. latent posterior `p(f | D, x)`
2. Bernoulli link 後の class probability `p(y=1 | D, x)`

latent mean を probability と呼んではいけません。最初の public implementation は、
binary target validation、Bernoulli likelihood、non-Gaussian inference、latent posterior
sampling、class probability prediction、finite training objective / gradients を備えます。

BoTorch / GPyTorch の public variational primitives を優先し、robotorchan は raw-data
retention、training API、probability surface、capability metadata、acquisition integration
に必要な薄い integration layer を所有します。

## Ordinal GP

ordinal target を continuous regression として代用しません。public contract には
ordered contiguous labels、monotone cut-points、latent posterior、class ordering metadata、
合計1になる per-class probabilities、non-Gaussian inference が必要です。

binary classification の latent / probability infrastructure を先に確立し、その後に
ordinal threshold semantics を追加します。

## Classification Active Learning

classification acquisition は class-probability / epistemic contract が成立してから
追加します。

- predictive entropy は predictive class probabilities を使用する。
- margin uncertainty は probability margin を使用する。
- Bernoulli `p(1-p)` は latent posterior variance と同一視しない。
- BALD は marginal probability だけでは実装しない。link 前の epistemic variation を
  保持する posterior samples が必要。
- latent straddle は latent link scale 上の threshold を明示する。

regression の `PosteriorVariance`、`PosteriorStd`、`Straddle`、
`BoundaryVariance` を名前だけ変えて classification uncertainty として公開しません。

## Expansion order

cross-product を最初から作りません。基本順序は次です。

1. binary single-task GP classifier
2. validated classification AL
3. raw categorical semantics が必要なら Mixed binary classifier
4. explicit output-probability contract を持つ multi-class / multi-task classification
5. ordinal GP と ordinal-specific acquisition

Mixed / MultiTask / ordinal variants は base classification contract の runtime evidence を
継承できる部分と、専用検証が必要な部分を分離します。

## Public promotion checklist

public model / acquisition を追加する前に少なくとも次を確認します。

- raw training-data retention
- target-domain validation
- finite training objective and gradients
- latent posterior sampling
- calibrated surface の意味を明示した probability prediction
- probability range / normalization
- posterior/probability shape contract
- representative acquisition runtime
- capability metadata と実装の一致
- no regression-shaped fake classifier API

この領域は regression correctness defect ではなく独立した future capability です。
実装を開始した後も、未検証の Mixed / MultiTask / ordinal cross-product を自動的に
capability claim へ昇格させません。
