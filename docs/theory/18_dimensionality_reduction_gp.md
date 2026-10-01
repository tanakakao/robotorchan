# Dimensionality Reduction Gaussian Process

## 直感

入力次元 `D` が大きくても、目的関数が実質的に低次元の構造へ依存している場合があります。

```text
x in R^D
  -> z = r_x(x) in R^d
  -> GP(z)

d << D
```

また、高次元 output に対しては

```text
y in R^M
  -> h = r_y(y) in R^m
  -> GP latent outputs
  -> reconstructed posterior in R^M
```

という output reduction も可能です。

robotorchan の reduced GP は、**公開 API では元の feature / output space を保ちながら、
内部 GP だけを reduced space で動かす**ことを基本契約とします。

## 1. Modeling dimension と search dimension

最初に重要なのは、次の2つを分けることです。

### Modeling dimension reduction

```text
x in R^D -> r(x) in R^d -> GP
```

GP が covariance を計算する次元を減らします。

### Search-space reduction

```text
z in R^d -> x = A z in R^D -> acquisition evaluation
```

acquisition optimization 自体を低次元空間で行います。

`PCAGP` や `PLSGP` は前者です。public posterior は original-space `X` を受け取り、
内部で reducer を適用します。そのため通常の `optimize_acqf` に original-space bounds を
渡すなら、candidate search 自体は依然として `D` 次元です。

REMBO / ALEBO / BAxUS などは
[High-dimensional search](20_high_dimensional_search.md) の問題です。

## 2. PCA input reduction

PCA は centered input matrix `X` の分散を最大化する直交方向を求めます。

```text
z = W^T (x - mu_x)
```

ここで `W` は上位 principal directions です。

PCA は `Y` を使わない unsupervised reduction です。そのため、

- noise を含む高次元入力を圧縮できる
- reducer が目的値へ直接 overfit しにくい

という利点がある一方、

- X の分散が小さくても Y に重要な方向を落とす

可能性があります。

robotorchan:

- `PCAGP`
- Mixed / MultiTask / Kronecker 系の PCA reduced model

## 3. PLS input reduction

PLS は input variation だけでなく response `Y` との関係を使って latent component を
構成する supervised reduction です。

概念的には component

```text
t = X w
```

が response と強く共変するように projection direction `w` を選びます。

PCA と比較すると predictive direction を残しやすい一方、reducer 自体が `Y` を見るため、
小標本では component 数の選択や過学習に注意が必要です。

robotorchan:

- `PLSGP`
- Mixed / MultiTask / Kronecker 系の PLS reduced model

## 4. Random projection

Random Projection は projection matrix をデータから最適化せず、固定されたランダム射影を
使います。

```text
z = A x
```

高次元 Euclidean geometry は、適切な reduced dimension を取ればランダム射影でも
近似的に保持できることが Johnson-Lindenstrauss lemma から知られています。

robotorchan の `RandomProjectionGP` は Gaussian random projection を frozen reducer として
使います。`random_state` により再現可能な projection を構成します。

PCA / PLS のような fitted direction と比較する baseline としても有用です。

## 5. ReducedGP の公開空間と内部空間

`ReducedGP` は次の dimension を明示的に区別します。

- `original_input_dim`
- `reduced_input_dim`
- `original_output_dim`
- `reduced_output_dim`

training 時には raw `train_X` / `train_Y` を保存し、内部 GP には reduced data を渡します。

posterior では original-space input と already-reduced input の両方を受けられます。

```text
original X -> reducer.transform -> GP
reduced X ---------------------> GP
```

dimension がどちらにも一致しなければ error とします。

この契約により、通常利用では original feature space を維持しつつ、内部処理や高度な利用では
reduced coordinates を直接渡すこともできます。

## 6. Reducer lifecycle

reducer はモデル構築時に1度だけ fit することを基本とします。

- unfitted reducer: constructor 内で `fit_transform`
- already-fitted reducer: 再 fit せず `transform` のみ

したがって externally pre-fitted reducer や pretrained neural reducer の latent coordinate
system を保持できます。

BO iteration ごとに reducer を無条件に再 fit すると latent basis が変わり、以前の GP
parameter や candidate geometry との整合性を失う可能性があります。

reducer を更新する場合は raw training data からモデルを再構築し、座標系の変更を明示的に
扱う方が安全です。

## 7. BoTorch transform との順序

input reduction と BoTorch `input_transform` は同じものではありません。

robotorchan の処理順序は概念的に

```text
raw X
 -> input reducer
 -> reduced X
 -> BoTorch input_transform
 -> GP
```

です。

したがって `Normalize(d=...)` などの `input_transform` を渡す場合、その dimension は
**reduced input dimension** に合わせる必要があります。

output 側も同様に

```text
raw Y
 -> output reducer
 -> latent Y
 -> BoTorch outcome_transform
 -> GP
```

なので、`outcome_transform` は latent output dimension を対象とします。

## 8. Output reduction

高次元 output `Y` では、すべての output をそのまま GP で扱う代わりに latent output space
へ圧縮できます。

robotorchan:

- `OutputPCAGP`
- `OutputPLSGP`

training は

```text
Y -> latent Y -> GP
```

ですが、public posterior は latent component をそのまま返すのではなく、reducer を通じて
original output space へ復元します。

```text
GP posterior in latent Y
 -> restored posterior
 -> original Y space
```

したがって downstream objective / constraint は原則として original output semantics で
定義できます。

## 9. Posterior restoration と posterior transform

output reduction を使う場合、posterior transform の適用順序が重要です。

robotorchan はまず latent posterior を original output space へ復元し、その後
`posterior_transform` を適用します。

```text
latent posterior
 -> restore original outputs
 -> posterior_transform
```

これにより objective semantics を original outputs に対して定義できます。

ただし restored posterior が要求された transform をサポートしない場合は
`NotImplementedError` になります。

また output reduction 使用時は現在、

- `output_indices`
- Tensor-valued `observation_noise`

に制約があります。

## 10. Output noise の制約

output reduction と explicit `train_Yvar` を同時に使うには、observation-noise covariance
自体も latent output coordinates へ正しく変換する必要があります。

現在の `ReducedGP` はこの変換を暗黙には行わないため、

```text
output_reducer + explicit train_Yvar
```

をサポートしません。

これは単なる API 制限ではなく、noise semantics を誤って変換しないための correctness
boundary です。

同様に `condition_on_observations` でも output reduction と explicit observation noise の
組み合わせには制約があります。

## 11. condition_on_observations

fantasization や sequential BO では、新しい観測を既存モデルへ condition することがあります。

`ReducedGP.condition_on_observations(X, Y)` は public API では original spaces を受け取り、

```text
X -> input reduction
Y -> output reduction
```

して underlying GP へ渡します。

これにより model construction 時だけでなく conditioning 時にも同じ coordinate contract を
維持します。

## 12. state_dict と latent coordinate synchronization

neural reducer のように model construction 時と state restoration 時で latent coordinate が
変わり得る reducer では、parameter の load だけでは underlying GP の cached training input と
reducer が不整合になる可能性があります。

`ReducedGP.load_state_dict` は load 後に raw training inputs を現在の reducer で再変換し、
GP training inputs を同期します。

この挙動は PCA / PLS だけを見ると目立ちませんが、共通 reduced-model infrastructure の
重要な correctness contract です。

neural reduction の詳細は
[Neural reduction GP](19_neural_reduction_gp.md) を参照してください。

## 13. Mixed input reduction

Mixed input では categorical feature を PCA / PLS / random projection の連続線形空間へ
そのまま入れません。

基本構造は

```text
continuous X -> reducer -> latent continuous Z
categorical X -----------> categorical covariance
```

です。

つまり continuous dimensions だけを削減し、category identity は GP 側の mixed covariance
へ渡します。

これは「カテゴリコード 0, 1, 2 の数値距離」を PCA や PLS に学習させることを避けるためです。

## 14. MultiTask と Kronecker

long-format MultiTask では task feature は reducer に含めません。

```text
[data features, task ID]
       |
       +-> reduce data features only
```

task identity は structural column として保持します。

Kronecker形式では task identity は `Y` の task/output axis にあり、`X` に task column が
存在しません。そのため data features `X` の reduction と task covariance を自然に分離できます。

PCA / PLS reduced model が MultiTask / Kronecker と組み合わされる場合でも、この
structural-feature contract を維持する必要があります。

## 15. Input reduction と output reduction の違い

| Reduction | 目的 | public interface |
| --- | --- | --- |
| Input reduction | GP covariance の入力次元を下げる | original X |
| Output reduction | GP が扱う output 次元を下げる | original Y posterior |
| Search reduction | acquisition search 次元を下げる | method dependent |

この3つを同じ「高次元対策」として混同しないことが重要です。

特に input PCA/PLS を使っても、original-space `optimize_acqf` の optimization difficulty が
自動的に低次元になるわけではありません。

## 16. Bayesian optimization での注意

dimensionality reduction は GP の statistical efficiency を改善できる一方、reducer が
目的関数に重要な方向を捨てれば BO の optimum 自体を見失う可能性があります。

PCA は `X` の variance、PLS は `X-Y` relationship、random projection は geometric
preservation という異なる基準を使います。

したがって component 数は単に reconstruction ratio だけでなく、

- predictive validation
- posterior calibration
- BO regret / best-observed trajectory
- reduced basis の安定性

なども見て選択します。

## 17. 実装との対応

現在の理論章で保持すべき実装契約は次です。

- public training data は original space で保持
- underlying GP は reduced space で学習
- unfitted reducer は construction 時に1回 fit
- fitted reducer は再 fit せず再利用
- posterior は original-space / reduced-space X の両方を受理可能
- BoTorch `input_transform` は input reduction の後に適用
- BoTorch `outcome_transform` は output reduction の後に適用
- output posterior は original output space へ復元
- output reduction 時の `posterior_transform` は復元後に適用
- output reduction + explicit `train_Yvar` は未対応
- conditioning でも input/output reduction を適用
- state restoration 後は reducer と GP training input を再同期
- Mixed では continuous features のみ reduction
- task / category は structural feature として保持
- input reduction と acquisition search reduction は別機能

実装詳細は [high-dimensional inputs](../models/high_dimensional_inputs.md) と
[high-dimensional outputs](../models/high_dimensional_outputs.md) を参照してください。

## 参考文献

1. Jolliffe, I. T. and Cadima, J. (2016).
   Principal component analysis: A review and recent developments.
   *Philosophical Transactions of the Royal Society A*, 374.
2. Wold, S., Sjöström, M., and Eriksson, L. (2001).
   PLS-regression: A basic tool of chemometrics.
   *Chemometrics and Intelligent Laboratory Systems*, 58(2), 109-130.
3. Johnson, W. B. and Lindenstrauss, J. (1984).
   Extensions of Lipschitz mappings into a Hilbert space.
   *Contemporary Mathematics*, 26, 189-206.
4. Bingham, E. and Mannila, H. (2001).
   Random projection in dimensionality reduction: Applications to image and text data.
   *Proceedings of the Seventh ACM SIGKDD International Conference on Knowledge Discovery
   and Data Mining*.
5. Constantine, P. G., Dow, E., and Wang, Q. (2014).
   Active subspace methods in theory and practice: Applications to kriging surfaces.
   *SIAM Journal on Scientific Computing*, 36(4).
6. Rasmussen, C. E. and Williams, C. K. I. (2006).
   *Gaussian Processes for Machine Learning*. MIT Press.

## 関連章

- [Gaussian Process](02_gaussian_process.md)
- [High-dimensional GP](08_high_dimensional_gp.md)
- [Neural reduction GP](19_neural_reduction_gp.md)
- [High-dimensional search](20_high_dimensional_search.md)
- [MultiTask / MultiOutput](07_multitask_multioutput.md)
