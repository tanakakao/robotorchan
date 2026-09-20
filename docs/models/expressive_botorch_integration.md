# 表現力モデルのBoTorch統合契約

robotorchanの表現力モデルは内部構造が異なっていても、ベイズ最適化から利用するときは
可能な範囲で共通のBoTorch contractを提供します。

## 対象モデル

- `JointEncoderGP`: 決定論的なニューラル特徴抽出器とExact GPを結合するDKL
- `SingleTaskDeepGP`: 確率的なGP階層を持つDeep GP
- `InfiniteWidthBNNGP`: 無限幅ReLUネットワークに対応するNNGP kernelのExact GP
- `SpectralMixtureGP`: スペクトル混合kernelのExact GP

## 共通して確認する契約

Exact GP系の3モデルでは、元の入力空間 `X` をそのまま `posterior()` と獲得関数へ
渡せます。候補点に対する勾配も元入力空間まで伝播するため、`optimize_acqf` に
手動の潜在変換や逆変換を挟む必要はありません。

Phase 11の統合テストでは、Exact GP系について以下を同じ条件で確認します。

- posteriorから候補点へのgradient
- qLogExpectedImprovement
- qLogNoisyExpectedImprovement
- qUpperConfidenceBound
- optimize_acqfによる元入力空間での候補生成

Deep GPも元入力空間を直接受け取り、posterior samplingを通じてMC acquisitionへ接続します。
ただしExact GP posteriorとはサンプリング構造が異なるため、現在は
`StochasticSampler` を用いるqLogEI/qUCBと `optimize_acqf` を統合契約として検証します。

この違いを隠すための互換wrapperは追加しません。各モデルがBoTorchの期待するposterior
contractを直接満たす設計を維持します。
