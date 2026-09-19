# Nonstationary Gaussian Process

## 直感

stationary kernel は、同じ距離だけ離れた二点の関係が入力空間のどこでも同じという仮定を
置きます。現実には、ある領域では滑らかで、相転移や境界付近では急激に変化する関数があります。

## Stationary kernel

典型的な stationary kernel は

```text
k(x, x') = k(x - x')
```

であり、局所的な smoothness の尺度である lengthscale が空間全体で共有されます。

## Local lengthscale

nonstationary GP では概念的に

```text
ell = ell(x)
```

として、入力位置に応じて局所 metric / lengthscale を変化させます。これにより smooth region と
rapidly varying region を同じ global lengthscale で妥協する必要がなくなります。

robotorchan: `NonstationarySingleTaskGP`, `MixedNonstationarySingleTaskGP`。

## Heteroskedasticity との違い

- nonstationary: latent function の covariance / smoothness が変化
- heteroskedastic: observation noise variance が変化

見かけ上どちらも「場所によってばらつきが違う」ように見えることがありますが、統計的な原因は
異なります。

## Mixed variables

Mixed 版では continuous geometry の非定常性と categorical covariance を共存させます。
カテゴリコードを continuous local-lengthscale process にそのまま混ぜる設計ではありません。

## BO との関係

局所的に急変する領域で stationary GP が過度に平滑化すると、posterior mean と uncertainty の
双方が acquisition を誤誘導する可能性があります。一方、nonstationary model は自由度も増える
ため、小標本では過学習や識別性を確認します。

詳細は [Nonstationary GP](../models/nonstationary_gp.md) を参照してください。
