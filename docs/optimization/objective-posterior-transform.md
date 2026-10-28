# Objective と PosteriorTransform

objective、posterior transform、制約、robust scenario 集約の責務を分けます。

## 標準の処理順

```text
X
-> candidate feasibility
-> model.posterior(X)
-> PosteriorTransform when required
-> posterior sampling when required
-> MC Objective when required
-> outcome feasibility / acquisition utility
-> acquisition optimization
```


意思決定時の入力摂動では posterior 評価前に scenario を導入します。

```text
X -> perturbation scenarios -> posterior -> sampling -> objective -> scenario risk aggregation
```


入力摂動は汎用 objective の責務ではありません。

## PosteriorTransform

解析的・MC 獲得関数の評価前に posterior 分布を変換します。モデルが対応する BoTorch 標準 transform を直接使います。

アフィンなスカラー化には通常 `ScalarizedPosteriorTransform` を使います。`Z = offset + w^T Y` では Gaussian posterior の平均と出力間共分散の両方を変換します。

```text
E[Z] = offset + w^T E[Y]
Var[Z] = w^T Cov[Y] w
```


周辺分散だけから共分散を作り出してはいけません。必要な分布構造を公開しない特殊 posterior は、その transform を明示的に拒否できます。

## MC Objective

`MCAcquisitionObjective` は分布自体でなく posterior sample を変換します。

```text
sample_shape x batch_shape x q x m
-> sample_shape x batch_shape x q
```


必要な意味を表せる標準 objective を使い、非線形の custom sample objective には `GenericMCObjective` を使います。posterior transform と MC objective は交換可能ではありません。

## 多目的獲得関数

qLogEHVI・qLogNEHVI は hypervolume に必要な目的ベクトルを維持します。スカラー化は多目的の既定動作ではなく、qLogNParEGO などの別経路です。

## 制約

出力制約はモデル化した未知の実行可能性を獲得関数と合成します。候補制約は既知の X の関数であり `robotorchan.optim` が担当します。両者を一つの抽象化にまとめません。

## Robust scenario 集約

入力摂動から scenario を作り、posterior 評価・sampling 後に scenario 軸を risk 集約します。posterior sample 軸と scenario 軸は別に保ちます。

`robotorchan.objectives.risk` の class は scenario 集約器です。実際に native 契約を満たさない限り `MCAcquisitionObjective` の代用品としません。BoTorch の risk objective が目的を満たすなら標準経路を使います。

## モデル固有の対応

`posterior_transform=` が同じ書式でも、すべての変換を受け付けるわけではありません。

- 標準・Mixed Gaussian Exact GP は通常 BoTorch へ委譲する。
- 入力削減モデルでは posterior 構築前に入力を変換し、transform は公開 posterior に作用する。
- 出力削減モデルは元出力の意味を保つ必要がある。
- 経験 ensemble・確率的 posterior は Gaussian 分布構造が必要な変換を拒否し得る。
- Kronecker など特殊経路は、数学的に正当化した追加実装がなければ上流の制約に従う。

未対応の組合せは意味を暗黙に変更せず拒否します。

## 拡張方針

標準 API で埋まらない意味上の不足がある場合に限り追加します。Tensor 軸、勾配、獲得関数との合成、モデル・posterior 互換性を実行テストで定義してください。
