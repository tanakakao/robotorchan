# 未検証事項とNegative Contracts

実装が存在すること、testを実行したこと、数学的referenceと一致したことは異なる。ここでは確かめる際の**誤った一般化を防ぐ境界**を列挙する。現時点の全未検証事項の完全棚卸しではなく、原文[Verification Mapping](../../../.github/project-records/theory-guide/mapping/verification.md)・[Mapping index](../../../.github/project-records/theory-guide/mapping/README.md)の原記録を優先する。

| 機能・関係 | 避ける主張 | 根拠と次の確認 |
| --- | --- | --- |
| Heteroskedastic観測posterior | `posterior(observation_noise=True)`が入力依存noiseを自動統合済み | 手動の`Var[y]=Var[f]+v`検証はあるが、public APIの候補点noise注入は別 |
| Binary GP probit | closed-form probability参照が学習後の分類較正を保証する | Brier/NLL/ECEなどholdout指標を要する |
| Non-GP ensemble | ensemble spreadがBayesian epistemic uncertainty | estimator disagreementの性質と分布的妥当性を分ける |
| Mixed reduced GP | 1つのPCAカテゴリ保持検証から全mixed reducerを保証 | PLS・neural・別feature位置は別確認 |
| Fully Bayesian SAAS | `supports_mll=False`でもExact MLLで通常通り学習できる | NUTS/Pyro fitting contractに従う |
| One-shot・embedding strategy | 通常のcandidate constraintを無変換で潜在空間へ転送できる | optimizer coordinateで制約を再表現する必要がある |
| CI成功 | 全モデル/獲得関数/optimizerの数式が検証された | テストの個々のassertion・tol・対象propertyを記録する |

**証拠不足と実装非対応を区別する。** テスト未確認なら`Unverified`、ソース上で意図的に拒否するなら`Unsupported`、近似の仮定付きなら`Approximated`とし、両者を単なる「未完了」に潰さない。

旧文書にあるPhase/PR等の履歴は監査資料として有効なので、この索引へコピーせず原文のまま保存する。原文区間の移行証跡が揃う前にVerification Mappingを縮小・削除しない。
