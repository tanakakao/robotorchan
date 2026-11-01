# Foundations：理論から実装へ

## この章で解決する問い

posterior → objective / utility → acquisition score → optimizer → candidate。モデル予測は意思決定価値そのものではない。

数学的な定義・仮定・導出は[理論の対応章](../../theory/acquisition/01_foundations.md)が正本であり、
この章は**数式の計算対象 → クラス構造 → 内部処理 → 近似と制限**をつなぐ日本語実装ガイドである。

## 実装の対応と委譲

BoTorch標準acquisitionとrobotorchanのlocal ALクラスを区別。
`src/robotorchan/acquisition/registry.py`のmetadataは数式実装や全optimizer互換性の証明ではない。

## 採用時の確認と注意事項

観測値の尺度、posterior shape、最大化/最小化の向きを統一する。

### BoTorch / robotorchan との対応

BoTorchはanalytic / MC acquisition、objective、posterior transform、sampler、
acquisition optimizerを分離した設計を採用しています。

robotorchanのcapability metadataではposterior requirementをmarginal moments、joint Gaussian、
posterior samplesに分けます。さらにq、multi-output、ensemble、fantasization等が重なります。

robotorchanもBoTorch-firstとし、BoTorch native acquisitionが目的を満たす場合はローカルwrapperを作りません。現在の対応状況は
[統合契約](../../../docs-internal/contracts/optimization/acquisition-integration.md)
を参照してください。

## acquisitionの基底と計算の所有権

BoTorchの`AcquisitionFunction`はモデルを保持し、具体的な`forward()`は派生クラスが実装します。analytic系はposteriorのmoment、
MC系はsamplerで得た応答をobjectiveへ通した量を計算します。これらの共通構造は[BoTorchのacquisition基底][b-base]が担当します。

robotorchanの[registry](../../../src/robotorchan/acquisition/registry.py)は計算クラスとcapabilityを結び付け、
[compatibility](../../../src/robotorchan/acquisition/compatibility.py)が
posteriorの表現・出力数・samplingなどの条件を照合します。registry自体は獲得関数値を計算しません。
[composition](../../../src/robotorchan/acquisition/composition.py)は
目的の方向とモデル内の出力位置をBoTorchのobjectiveへ接続します。

[b-base]: https://github.com/pytorch/botorch/blob/v0.18.1/botorch/acquisition/acquisition.py
