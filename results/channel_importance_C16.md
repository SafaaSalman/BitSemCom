loaded Pretrain\BitSemCom_C16_stage2_ep480.model  (missing=0 unexpected=0)

C=16, 24 images, noiseless PSNR 22.23 dB. Per-channel PSNR drop when flipping a fraction of that channel's bits:

| channel | mean confidence | share of top-2%% confident bits | P(bit=1) | drop @10% flips (dB) | drop @50% flips (dB) |
|---|---|---|---|---|---|
| 11 | 0.476 | 8.5% | 0.32 | 0.06 | 0.37 |
| 7 | 0.460 | 3.6% | 0.58 | 0.06 | 0.41 |
| 0 | 0.478 | 9.0% | 0.71 | 0.05 | 0.50 |
| 15 | 0.467 | 8.2% | 0.61 | 0.05 | 0.37 |
| 9 | 0.478 | 8.9% | 0.34 | 0.05 | 0.31 |
| 3 | 0.476 | 7.1% | 0.39 | 0.05 | 0.31 |
| 13 | 0.454 | 2.1% | 0.59 | 0.04 | 0.42 |
| 6 | 0.457 | 3.8% | 0.54 | 0.04 | 0.37 |
| 8 | 0.462 | 5.6% | 0.55 | 0.04 | 0.37 |
| 2 | 0.475 | 10.7% | 0.27 | 0.04 | 0.36 |
| 5 | 0.477 | 7.8% | 0.41 | 0.04 | 0.36 |
| 4 | 0.473 | 6.8% | 0.54 | 0.04 | 0.23 |
| 1 | 0.448 | 1.3% | 0.47 | 0.03 | 0.35 |
| 14 | 0.466 | 5.8% | 0.38 | 0.03 | 0.30 |
| 10 | 0.473 | 6.3% | 0.51 | 0.03 | 0.29 |
| 12 | 0.467 | 4.4% | 0.62 | 0.03 | 0.32 |

Sum of per-channel drops @10%: 0.67 dB; max 0.06, median 0.04, min 0.03
Correlation(mean confidence, drop @10%) = 0.20
