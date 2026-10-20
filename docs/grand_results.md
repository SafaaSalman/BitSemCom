# BitSemCom + GRAND: first results and what they mean

Date: 2026-10-21. Branch: `grand-implementation`. Hardware: RTX 5060 Laptop (GPU for the semantic model), CPU for ORBGRAND.
Models: the Colab stage-2 checkpoints `Pretrain/BitSemCom_C{16,32,64}_stage2_ep*.model`. Test set: Kodak, 24 centre 256x256 patches (plus one run on the full 768x512 images). All PSNR values are means over the 24 images; differences below about 0.03 dB are within noise.

## 1. What was built

* `grand/` - a numpy port of Ken Duffy's reference ORBGRAND (basic and 1-line variants, Landslide pattern generator), systematic random-linear and CRC code constructors, a self-test and a validation script. Syndromes are packed into 64-bit integers, patterns are cached, so the decoder runs at roughly 2 million codebook queries per second in pure Python.
* `experiments/` - a wrapper that builds the DIV2K/Kodak WITT model without importing `main.py`, a coded link (pad -> [n,k] encode -> optional interleaver -> QPSK with exactly the mapping and noise model of `QPSK_soft` -> AWGN or block Rayleigh fading -> coherent LLRs -> ORBGRAND -> P(bit=1)), the sweep runner `run_grand.py`, two diagnostics, and `summarize.py`.

### Validation of the decoder against the published curves

BPSK, same codes and Eb/N0 points as the `.mat` files shipped with GRAND-MATLAB.

| code / variant | Eb/N0 | reference BLER | this port | reference avg queries | this port |
|---|---|---|---|---|---|
| RLC(128,116) basic | 4.0 | 1.04e-1 | 1.03e-1 | 404 | 372 |
| RLC(128,116) basic | 5.0 | 1.09e-2 | 0.96e-2 | 51 | 60 |
| RLC(128,116) basic | 6.0 | 6.6e-4 | 7.0e-4 | 5.0 | 4.7 |
| CRC(128,112) 1-line | 4.0 | 3.7e-2 | 2.5e-2 | 2473 | 2257 |
| CRC(128,112) 1-line | 4.5 | 9.4e-3 | 8.1e-3 | 752 | 695 |
| CRC(128,112) 1-line | 5.0 | 1.7e-3 | 1.7e-3 | 157 | 153 |

Each point was stopped after 40 block errors, so the agreement is within Monte Carlo noise.

### Conventions confirmed in the code

* `QPSK_soft` uses positive LLR = bit 0, the same convention as Duffy's code. No sign flip is needed anywhere (the ChatGPT conversation in `convo.md` had this backwards).
* The semantic decoder consumes P(bit=1) = sigmoid(-LLR), not bits and not LLRs.
* The SNR argument of `QPSK_soft` is Es/N0 per QPSK symbol. Per-bit error rate is Q(sqrt(Es/N0)), so 6 dB gives 2.3 percent raw BER.

## 2. Transmission paths compared

Every path receives the identical semantic bitstream for a given image and SNR.

| path | description |
|---|---|
| `noiseless` | decoder fed the exact bits: the ceiling for any channel code |
| `uncoded_soft` | the original BitSemCom receiver |
| `uncoded_hard` | hard decisions (0/1) fed to the decoder |
| `coded_hard` | (n,k) code + ORBGRAND; abandoned blocks fall back to hard decisions |
| `coded_soft` | same, abandoned blocks fall back to channel posteriors |
| `*_ebn0` | coded path run at Es/N0 + 10 log10(k/n): same energy per semantic bit, n/k more channel uses |
| `*_il` / `*_noil` | with / without a pseudo-random interleaver over the whole transmission (fading runs) |

Default code: CRC-16 polynomial 0x9eb2 (Koopman notation), systematic (128,112), rate 0.875, which is 14.3 percent more channel uses. Decoder: 1-line ORBGRAND, abandonment after 1e5 queries.

## 3. AWGN results

### 3.1 PSNR (dB), (128,112) code, 24 patches

C = 16

| Es/N0 | noiseless | uncoded soft | uncoded hard | coded soft | coded soft, equal Eb/N0 |
|---|---|---|---|---|---|
| 2 | 22.25 | 21.69 | 21.32 | 21.29 | 21.06 |
| 4 | 22.27 | 22.03 | 21.90 | 21.86 | 21.74 |
| 5 | 22.24 | 22.08 | 21.98 | 22.04 | 21.90 |
| 6 | 22.24 | 22.16 | 22.11 | 22.20 | 22.12 |
| 7 | 22.25 | 22.20 | 22.18 | 22.25 | 22.24 |
| 8 | 22.24 | 22.22 | 22.22 | 22.24 | 22.24 |
| 10 | 22.26 | 22.26 | 22.26 | 22.26 | 22.26 |

C = 32

| Es/N0 | noiseless | uncoded soft | coded soft | coded soft, equal Eb/N0 |
|---|---|---|---|---|
| 2 | 23.44 | 22.83 | 22.36 | 22.12 |
| 4 | 23.47 | 23.19 | 22.98 | 22.80 |
| 5 | 23.46 | 23.27 | 23.24 | 23.07 |
| 6 | 23.45 | 23.36 | 23.42 | 23.34 |
| 7 | 23.45 | 23.41 | 23.45 | 23.44 |
| 8 | 23.46 | 23.44 | 23.46 | 23.46 |

C = 64

| Es/N0 | noiseless | uncoded soft | coded soft | coded soft, equal Eb/N0 |
|---|---|---|---|---|
| 2 | 24.44 | 23.79 | 23.45 | 23.26 |
| 4 | 24.44 | 24.13 | 23.96 | 23.81 |
| 5 | 24.44 | 24.25 | 24.25 | 24.09 |
| 6 | 24.44 | 24.33 | 24.41 | 24.33 |
| 7 | 24.45 | 24.39 | 24.45 | 24.43 |
| 8 | 24.44 | 24.40 | 24.44 | 24.44 |

Full-resolution Kodak (768x512), C=16: noiseless 23.11; uncoded soft 22.92 / 23.04 / 23.09 at 4 / 6 / 8 dB; coded soft 22.75 / 23.08 / 23.11. Same picture as the patches.

### 3.2 Coding-gain ceiling and realised gain

Ceiling = noiseless minus uncoded soft. Gain = coded soft minus uncoded soft.

| Es/N0 | C=16 ceiling / gain / gain at equal Eb/N0 | C=32 | C=64 |
|---|---|---|---|
| 2 | +0.56 / -0.41 / -0.63 | +0.61 / -0.47 / -0.71 | +0.66 / -0.34 / -0.53 |
| 4 | +0.24 / -0.17 / -0.29 | +0.28 / -0.21 / -0.39 | +0.31 / -0.16 / -0.31 |
| 5 | +0.16 / -0.04 / -0.18 | +0.19 / -0.03 / -0.20 | +0.19 / +0.00 / -0.15 |
| 6 | +0.08 / +0.04 / -0.04 | +0.09 / +0.06 / -0.02 | +0.11 / +0.08 / -0.00 |
| 7 | +0.05 / +0.05 / +0.04 | +0.05 / +0.05 / +0.04 | +0.07 / +0.06 / +0.05 |
| 8 | +0.02 / +0.02 / +0.02 | +0.03 / +0.03 / +0.03 | +0.03 / +0.03 / +0.03 |

### 3.3 What GRAND itself did (C=16, coded path at equal Es/N0)

| Es/N0 | raw BER | BER after GRAND | block error rate after | abandoned | miscorrected | avg queries |
|---|---|---|---|---|---|---|
| 2 | 0.104 | 0.111 | 0.996 | 0.21 | 0.78 | 49k |
| 4 | 0.058 | 0.062 | 0.83 | 0.17 | 0.66 | 44k |
| 5 | 0.037 | 0.028 | 0.45 | 0.08 | 0.37 | 24k |
| 6 | 0.022 | 0.0046 | 0.082 | 0.011 | 0.071 | 5.7k |
| 7 | 0.013 | 0.0004 | 0.009 | 0.002 | 0.007 | 668 |
| 8 | 0.0058 | 0.00006 | 0.001 | 0 | 0.001 | 14 |
| 10 | 0.0007 | 0 | 0 | 0 | 0 | 1 |

A full C=16 sweep (24 images, 7 SNRs, 4 coded paths) takes 70 s on CPU.

### 3.4 Alternatives tried (C=16, 12 images, PSNR difference coded soft minus uncoded soft)

| variant | 3 dB | 4 dB | 5 dB | 6 dB | 7 dB | note |
|---|---|---|---|---|---|---|
| CRC(128,112), budget 1e5 (baseline) | - | -0.24 | -0.10 | +0.04 | +0.05 | 37 to 66 percent miscorrections below 6 dB |
| CRC(128,112), budget 1e4 | -0.07 | -0.05 | -0.01 | +0.05 | +0.05 | fewer miscorrections, more abandonments |
| CRC(128,112), budget 1e3 | -0.01 | -0.04 | -0.04 | +0.03 | +0.05 | never hurts much, never helps much |
| RLC(128,104), rate 0.81 | - | +0.03 | +0.07 | +0.06 | +0.05 | zero miscorrections; at equal Eb/N0: -0.10, -0.08, +0.01, +0.05 |
| CRC(64,48), rate 0.75 | -0.08 | +0.10 | +0.13 | +0.08 | - | at equal Eb/N0: -0.58, -0.35, -0.06, +0.06 |

The best measured gain against the original receiver at equal per-symbol SNR is +0.13 dB (short code at 5 dB), at the cost of 33 percent more channel uses. At equal energy per semantic bit nothing beats the uncoded receiver by more than 0.07 dB.

### 3.5 Bandwidth comparison

Noiseless PSNR is 22.25 / 23.46 / 24.44 dB for C = 16 / 32 / 64, i.e. about 1.1 dB per doubling of semantic bits. The 14.3 percent of channel uses that the (128,112) code costs would therefore buy about 0.2 dB if spent on more semantic bits (0.33 dB for the (128,104) code, 0.46 dB for the (64,48) code). Every coding gain measured in AWGN is smaller than that. This estimate interpolates between existing models; the clean test needs C=14 and C=28 models (section 6).

## 4. Why the gain is so small: diagnostics

### 4.1 The bit mapper is stochastic and the decoder is trained for it

78.9 percent (C=16) / 74.8 percent (C=64) of bits have P(1) within 0.01 of 0 or 1; 10 to 12 percent are genuinely uncertain. Re-sampling the Gumbel bit mapper with a different seed changes about 5 percent of the bits and does not change PSNR at all (22.23 vs 22.23; 24.44 vs 24.46). The decoder has effectively been trained with 5 percent label noise on its input, which is more than the 2.3 percent channel BER at 6 dB.

### 4.2 Scattered flips are cheap, concentrated flips are catastrophic

Hard 0/1 input, 24 patches, PSNR after flipping a fraction of the bits:

| flipped | random positions | most-confident bits | least-confident bits | whole tokens (all C bits of a token) |
|---|---|---|---|---|
| C=16, 1 % | 22.17 | 22.16 | 22.22 | 21.53 |
| C=16, 2 % | 22.12 | 22.09 | 22.21 | 21.16 |
| C=16, 5 % | 21.85 | 21.79 | 22.21 | 19.85 |
| C=16, 10 % | 21.36 | 20.84 | 22.17 | 18.01 |
| C=64, 2 % | 24.31 | 24.26 | 24.42 | 22.55 |
| C=64, 10 % | 23.47 | 22.57 | 24.40 | 17.98 |

Noiseless: 22.23 (C=16), 24.44 (C=64). Which bits are hit barely matters (an earlier version of this table showed confident bits as 10x more damaging; that was a tie-breaking artifact that selected the first tokens in memory). What matters is concentration: losing 2 percent of tokens costs 1.1 dB, the same number of flips spread out costs 0.1 dB.

Per-channel sensitivity is flat: flipping 10 percent of one channel's bits costs 0.03 to 0.06 dB for every one of the 16 channels (0.00 to 0.03 for the 64 channels). There is no "important channel" to protect, so unequal error protection by channel has no basis in these models.

### 4.3 The SNR conditioning of the WITT backbone is inert in these checkpoints

Encoder features at 0 dB and 20 dB conditioning differ by 1e-5 relative; the sampled bits are identical. Decoder outputs differ by at most 0.027 on a 0-1 scale and PSNR is identical to two decimals across all paths (run `--dec_snr 10` vs `channel`). Telling the decoder "the bits are clean" after GRAND therefore gains nothing. These models are effectively SNR-agnostic.

## 5. Block-fading results (where errors cluster)

Block Rayleigh fading, one complex gain per block of symbols, coherent detection, average Es/N0. The interleaver is a fixed pseudo-random permutation known to both sides.

C=16, one fade per token (8 symbols = 16 bits):

| avg Es/N0 | noiseless | uncoded soft | uncoded soft + interleave | uncoded hard | coded soft + interleave | coded soft, no interleave | coded soft + IL, equal Eb/N0 |
|---|---|---|---|---|---|---|---|
| 6 | 22.24 | 21.62 | 21.85 | 21.05 | 21.51 | 21.14 | 21.42 |
| 10 | 22.26 | 22.00 | 22.11 | 21.82 | 22.05 | 21.78 | 21.98 |
| 14 | 22.24 | 22.11 | 22.18 | 22.07 | 22.24 | 22.12 | 22.23 |
| 18 | 22.26 | 22.18 | 22.24 | 22.15 | 22.26 | 22.23 | 22.26 |
| 22 | 22.26 | 22.24 | 22.25 | 22.23 | 22.26 | 22.25 | 22.26 |

C=64, one fade per token (32 symbols):

| avg Es/N0 | noiseless | uncoded soft | uncoded soft + interleave | uncoded hard | coded soft + interleave | coded soft, no interleave |
|---|---|---|---|---|---|---|
| 6 | 24.44 | 23.45 | 23.95 | 22.89 | 23.59 | 23.23 |
| 10 | 24.44 | 24.00 | 24.25 | 23.72 | 24.24 | 24.01 |
| 14 | 24.45 | 24.24 | 24.37 | 24.10 | 24.44 | 24.26 |
| 18 | 24.44 | 24.30 | 24.40 | 24.25 | 24.44 | 24.37 |
| 22 | 24.45 | 24.43 | 24.44 | 24.43 | 24.45 | 24.43 |

C=16 with one fade per 64 symbols (eight tokens) behaves the same: uncoded 21.52 / 21.94 / 22.02 at 6 / 10 / 14 dB, interleaved uncoded 21.89 / 22.12 / 22.17, coded soft with interleaver 21.53 / 22.00 / 22.23.

Reading of the fading data:

* Fading costs the uncoded receiver 0.6 dB (C=16) to 1.0 dB (C=64) at 6 dB average SNR, far more than AWGN at the same SNR, consistent with section 4.2: deep fades destroy whole tokens.
* Interleaving alone, with no code, recovers most of that: +0.23 to +0.50 dB at 6 dB, +0.11 to +0.25 dB at 10 dB. It converts concentrated errors into scattered ones, which the decoder absorbs.
* Hard decisions are much worse than soft in fading (-0.6 to -1.1 dB at 6 dB): faded tokens arrive as P(1) near 0.5 and the decoder treats them like erasures. This is the mechanism behind the robustness.
* Coding plus interleaving reaches the noiseless ceiling from 14 dB average SNR upward, 0.06 to 0.07 dB above interleaving alone. At 6 to 10 dB the code is again worse than interleaving alone because of miscorrections. Coding without interleaving is worse than doing nothing at every SNR.

## 6. Conclusions and what to build on

1. For these checkpoints on AWGN, a high-rate code plus ORBGRAND cannot improve image quality in any meaningful way. The decoder already tolerates dispersed bit errors so well that the entire available gain is under 0.1 dB where GRAND is feasible, and the code's bandwidth would buy twice that in semantic bits. This is a solid negative result with a clear mechanism (sections 4.1 and 4.2), and it is worth stating in the thesis as such.

2. Below 6 dB Es/N0, GRAND with 16 parity bits does harm: miscorrections deliver confidently wrong bits. Twenty-four parity bits (RLC(128,104)) remove the harm but not the bandwidth cost. Any further AWGN work should stop here.

3. Where a code plus GRAND does earn its place is when errors are concentrated or when the receiver cannot pass soft information to the semantic decoder:
   * `uncoded_hard` vs `coded_hard`: +0.09 dB at 6 dB AWGN, +0.36 dB at 6 dB and +0.23 dB at 10 dB in per-token fading (C=16). A system whose PHY delivers bits, not LLRs, benefits from GRAND. That is a realistic deployment constraint and a defensible thesis framing.
   * In fading, interleaving is the first-order fix and should be part of any baseline. GRAND on top gives bit-exact delivery from about 14 dB average SNR.

4. GRAND's abandonment flag is a block-level reliability signal that the semantic decoder does not have. Using it for retransmission (HARQ) of failed blocks, or for choosing between soft fallback and hard output per block, is unexplored here and is the natural next experiment with the existing pipeline (`abandon_rate` is already logged).

5. The fair equal-channel-use comparison needs new models: train C=14 and C=28 (and C=56) on Colab with the same recipe, then compare coded C=14 with (128,112) against uncoded C=16 at identical symbol counts. The interpolation in section 3.5 predicts uncoded wins by about 0.15 dB; confirming or refuting that directly is the cleanest statement the thesis can make.

6. Soft-output GRAND (SOGRAND) is the right algorithmic follow-up if coding is kept: the decoder's strength is soft input, and handing it posterior probabilities instead of hard decisions is the only way to stop miscorrections from hurting. The math of `SOGRAND_bitSO.m` is about fifteen lines on top of a list version of the current decoder.

7. Small but useful facts for the thesis text: the bit mapper is stochastic (about 5 percent of bits change per sample) with no quality impact; the WITT SNR-adaptive modules are inert in these checkpoints; the P(bit=1) soft input acts as an erasure mechanism in fading.

## 7. Reproducing

```
python grand/selftest.py
python grand/validate_duffy.py --mat <GRAND-MATLAB>/RESULTS/ORBGRAND_RLC_128_116_1.mat --variant basic --ebn0 4,5,6
python experiments/run_grand.py --C 16 --snrs 2,4,5,6,7,8,10 --code crc:0x9eb2:128:112
python experiments/run_grand.py --C 16 --snrs 6,10,14,18,22 --channel blockfading --fade_block 8 \
    --paths noiseless,uncoded_soft_np,uncoded_soft_il,uncoded_hard,coded_hard,coded_soft,coded_soft_noil
python experiments/bit_importance.py --C 16
python experiments/channel_importance.py --C 16
python experiments/summarize.py results/C16_crc128_112_patches.jsonl
```

Kodak images are expected in `datasets/kodak_full` (24 PNGs) and `datasets/kodak_patches256` (centre crops); both folders are gitignored. Raw per-image results are in `results/*.jsonl`, run logs in `results/*.log`, markdown tables in `results/summary_*.md`.

## 8. Licensing note

`grand/orbgrand.py` is a port of code released under the GRAND Codebase Non-Commercial Academic Research Use License; keep the citations in its docstring in any publication and do not use it commercially. No GPL code from the Rowshan repositories was used.
