# Plan v2: C=32 benchmark, GRAND-in-the-loop training, imperfect CSI, better metrics, LLR plots

Date: 2026-10-25. Branch `grand-implementation`. Follows the supervisor comments on the first results (`docs/grand_results.md`).

## 0. What the supervisor asked for, and the decision taken on each point

| # | Request | Decision |
|---|---|---|
| 1 | Pick one C, train it "properly" (no fixed LR, no 10M epochs) as the benchmark | C=32. New trainer `train_bitsemcom.py`: fixed step budget, warmup + cosine LR (1e-4 -> 1e-6), per-batch random SNR, grad clipping, validation every N steps on Kodak patches, best-checkpoint selection, early stop. Starts from `Pretrain/BitSemCom_C32_stage2_ep480.model` (the only C=32 model we have; the original training stopped it at ~24k steps with the LR still at 8e-5). |
| 2 | Train with GRAND in the loop and compare "without GRAND" vs "with GRAND" | Same trainer, `--grand crc:0x9eb2:128:112`. Channel block becomes: bits -> (128,112) CRC encode -> QPSK -> channel -> LLR -> ORBGRAND -> P(1) (decoded blocks exact, abandoned blocks soft fallback) -> semantic decoder. Non-differentiable part uses a straight-through estimator, exactly like the Gumbel bit mapper already does. Identical step budget, data, schedule and SNR range for both runs. |
| 3 | Use the MATLAB GRAND (Duffy), not a Python replica | Two uses. (a) Verification: `experiments/grand_verify.py` exports LLR blocks plus the Python decisions to a `.mat`; `matlab/verify_port.m` decodes them with Duffy's unmodified `bin_ORBGRAND1.m` and reports mismatches. ORBGRAND is deterministic, so the expected result is zero mismatches, not "statistically similar". (b) Evaluation: `--grand_backend matlab` (or `octave`) in `evaluate.py` runs the final sweeps through Duffy's code. The training loop itself uses the verified Python port: Duffy's implementation tests one noise pattern per loop iteration (his README calls it instructional, not optimized), and one training epoch needs on the order of 1e9 queries. **This split (MATLAB for verification and final numbers, verified port inside training) is the one thing Safaa should confirm with the supervisor.** |
| 4 | GRAND looked bad on bad channels; evaluate with imperfect CSI and SNR from 0 dB | SNR grid 0..12 dB (Es/N0 per QPSK symbol, as everywhere in this repo). Conditions: AWGN perfect CSI; AWGN with SNR-estimate error (receiver assumes SNR+d dB, d in {-3,+3} and random +-3); Rayleigh block fading perfect CSI; Rayleigh with channel-estimate error h_hat = h + e, e ~ CN(0, eps), eps in {0.05, 0.1, 0.2}. All implemented in `experiments/channel_coding.py` (numpy, evaluation) and `experiments/torch_channel.py` (torch, training). Training can optionally augment with the same errors. |
| 5 | A better semantic metric than PSNR | `experiments/metrics.py`: PSNR, MS-SSIM (existing), LPIPS-AlexNet (perceptual), CLIP ViT-B/32 cosine similarity (semantic). Primary pair for the thesis: PSNR + LPIPS; CLIP similarity as the "semantic" view. |
| 6 | Plot the LLRs under different channels | `experiments/plot_llr.py`: per condition and SNR, histogram of channel LLRs split by the true bit (with the theoretical Gaussian for AWGN), histogram of P(1) the decoder sees, and the same after GRAND. Also the bit-mapper confidence histogram the supervisor has seen before. |

## 1. Experiments (in order)

1. **LLR shape study** (no training; local or Colab, minutes). Stage-2 C=32 model, a few Kodak patches, SNR {0, 4, 8, 12} dB x the six conditions above. Output `results/figures/llr_*.png`. This is the first thing to show the supervisor; it explains the earlier fading/hard-decision results.
2. **Benchmark run** `runs/C32_bench`: `--grand none`, AWGN, SNR U[0, 10] dB, 30k steps, batch 16.
3. **GRAND run** `runs/C32_grand`: same, `--grand crc:0x9eb2:128:112 --grand_budget 2000 --grand_fallback soft`.
4. (optional, if 2 and 3 differ) `runs/C32_grand_csi`: GRAND run with imperfect-CSI augmentation.
5. **Evaluation** `experiments/evaluate.py`: both models x receivers {uncoded_soft, uncoded_hard, coded_soft(+interleaver)} x the six conditions x SNR 0..12. JSONL -> `experiments/plot_results.py`.
6. **MATLAB verification** on Safaa's MATLAB (or Octave): `python experiments/grand_verify.py export` -> run `matlab/verify_port.m` -> `python experiments/grand_verify.py compare`.

## 2. Figures (the minimal set)

1. LLR histograms by condition and SNR (supervisor's request).
2. PSNR and LPIPS vs SNR, one panel per condition, lines = model x receiver.
3. GRAND behaviour vs SNR per condition: BLER after decoding, abandonment, miscorrection, average queries.
4. Training curves of both runs: validation PSNR vs step, LR schedule.
5. Reliability diagram of P(1) under SNR mismatch (why imperfect CSI hurts the soft receiver and not the GRAND output). Cheap, optional.

## 3. Training recipe (both runs)

* Data: DIV2K train HR, random 256x256 crops, horizontal flip. Validation: 24 Kodak 256x256 patches at SNR {0, 3, 6, 10} with fixed seeds.
* Init: `Pretrain/BitSemCom_C32_stage2_ep480.model`. Loss: MSE on the 0..255 scale (unchanged).
* AdamW (weight decay 0), peak LR 1e-4, 500 warmup steps, cosine to 1e-6 over the budget, grad clip 1.0, bf16 autocast for the Swin blocks (bit mapper and channel in fp32).
* SNR per batch ~ U[0, 10] dB continuous (the old code drew one integer SNR per epoch).
* GRAND run: CRC(128,112) 0x9eb2, 1-line ORBGRAND, budget 2000 queries, soft fallback, interleaver on, decoding parallelised over the images of a batch (`--grand_workers`).
* Budget: 30k steps (DIV2K: 600 epochs). Validation every 1000 steps; keep `best.model` and `last.model`; `--patience` early stop; `--max_hours` for Colab sessions; `--resume` from `ckpt_last.pt`.

## 4. Status (end of 2026-10-25)

Built and smoke-tested locally (RTX 5060, small samples only; full runs are for Colab, see
`docs/colab_runbook.md`):

| piece | file | tested |
|---|---|---|
| trainer (benchmark and GRAND-in-the-loop, resume, CSI augmentation) | `train_bitsemcom.py` | 20-30 step runs with and without GRAND, resume, bf16 |
| differentiable training channel with imperfect CSI | `experiments/torch_channel.py` | identical to `QPSK_soft` (max diff 1e-7); statistics match the numpy channel |
| evaluation channel with imperfect CSI, pluggable decoder | `experiments/channel_coding.py` | reproduces the stored first-round rows to all digits (`--no_interleave`) |
| GRAND layer with exact straight-through and process pool | `experiments/grand_layer.py` | pool == in-process, gradient = identity |
| metrics PSNR / MS-SSIM / LPIPS / CLIP | `experiments/metrics.py` | LPIPS and CLIP weights download (slow on this laptop; instant on Colab) |
| LLR study | `experiments/plot_llr.py` | figures in `results/figures/llr_*.png` (4 patches, budget 1e4) |
| evaluation sweep and figures | `experiments/evaluate.py`, `plot_results.py`, `plot_training.py` | tiny sweeps, all figure types rendered |
| MATLAB reference bridge and bit-exact verification | `experiments/grand_backend.py`, `grand_verify.py`, `matlab/*.m` | file exchange and compare logic tested with a simulated reference output; the real MATLAB/Octave run is still to do |
| ORBGRAND budget semantics of the reference | `grand/orbgrand.py` (`budget_mode='level'`) | self-test passes; 'level' reproduces the per-level budget check of `bin_ORBGRAND*.m` |

First observations from the local LLR study (C=32, 4 patches, CRC(128,112), budget 1e4), to be
confirmed on Colab with the full budget:

* A +3 dB SNR over-estimate doubles the LLR magnitudes the decoder sees (mean |LLR| 4.7 vs 2.3
  at 0 dB) and halves the near-erasure fraction; a -3 dB under-estimate does the opposite. GRAND is
  unaffected (its ordering is scale invariant), so the coded receiver is immune to this CSI error
  while the soft receiver is miscalibrated. This is the cleanest argument for GRAND the thesis has.
* Channel-estimate error in block fading (variance 0.1) raises the raw BER from 3.2% to 6.3% at
  12 dB average SNR and pushes ORBGRAND from 13% to 96% block failures: with 16 parity bits the
  decoder cannot absorb the extra errors. This is the mechanism behind "GRAND looks bad on bad
  channels": the code is too weak for the error rate, not the decoder.
* Below about 6 dB AWGN, 13-16% of the blocks are miscorrected at a 1e4 budget: the decoder returns
  confidently wrong bits. Training with GRAND in the loop is the experiment that tests whether the
  semantic decoder can learn to live with that.

Note for comparisons with `docs/grand_results.md`: the stored AWGN rows were produced before the
interleaver existed; `run_grand.py` now interleaves by default (use `--no_interleave` to reproduce them).

## 5. Conventions kept from the first round

* LLR positive = bit 0. Decoder consumes P(bit=1) = sigmoid(-LLR).
* SNR = Es/N0 per QPSK symbol in dB; raw BER = Q(sqrt(Es/N0)).
* Equal-Eb/N0 comparison for coded paths = Es/N0 + 10log10(k/n).
* Seeds: same (image, SNR) -> same semantic bits and same channel noise for every receiver.

## 6. Review fixes (2026-10-27), applied before the Colab runs

* **Evaluation budget = training budget.** The GRAND run trains at 2000 queries (0-4 dB: 95-99 % of
  blocks abandoned, i.e. soft fallback, 1-5 % miscorrected) while 1e5 queries give 40-65 %
  miscorrected blocks at 4-5 dB. The primary `evaluate.py` sweep therefore uses `--budget 2000`
  (now the default); the 1e5 sweep is reported separately (`results/eval_C32_b1e5.jsonl`).
* **Gradient surrogate through the coded link.** The identity straight-through gave the GRAND run a
  different encoder gradient from the benchmark run, whose gradient is the soft channel's Jacobian
  2 p1 (1 - p1) / sigma_hat^2 (zero on confidently received bits, noise dependent).
  `GrandChannel(grad='soft')` (default, `--grand_grad`) keeps GRAND's exact forward value and uses
  the uncoded soft channel's Jacobian, so both runs share one gradient path and differ only in the
  forward signal.
* **Query accounting.** All experiment entry points (`evaluate.py`, `plot_llr.py`, `GrandChannel`,
  `grand_backend`) use `budget_mode='level'`, the per-weight-level check of `bin_ORBGRAND*.m`, so
  Python and MATLAB numbers are bit-identical (a block may use up to ~15 % more queries than the
  nominal budget). `run_grand.py` keeps `'query'` to reproduce the first-round rows.
* **Trainer.** Validation saves and restores the global RNG (it used to reseed it, so every
  1000-step interval replayed the same Gumbel and channel noise); the Swin resolution is reset to
  `--crop` every step (validation images of another size crashed the next step); a fresh run into
  an existing `--out` overwrites `log.csv`/`val.csv` instead of appending a second header;
  `fade_block=0` raises instead of silently meaning fast fading; the GRAND pool uses
  forkserver/spawn (never fork after CUDA init).
* **Caveat for runbook question 1.** `stage2` (-3..6 dB) and `bench` (0..10 dB) differ in training
  SNR range, not only in schedule.
