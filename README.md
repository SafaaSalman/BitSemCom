# BitSemCom + GRAND

Bit-level semantic image communication (BitSemCom, Shanghai Jiao Tong University, [arXiv 2510.26225](https://arxiv.org/abs/2510.26225)) with an explicit channel code decoded by GRAND / ORBGRAND added between the bit mapper and the semantic decoder. This repository holds the original BitSemCom code, our GRAND experiments, the results, and two reports:

- [docs/grand_results.md](docs/grand_results.md): round 1 (October 2026): C = 16 / 32 / 64, AWGN and block fading, the diagnostics that explain the numbers.
- [docs/grand_review.md](docs/grand_review.md): round 2 (October 2026): a properly retrained C = 32 benchmark, a model trained with GRAND in the loop, imperfect CSI, extra metrics, LLR study.
- [docs/latex/](docs/latex/): the same material as two LaTeX documents, a guide to the repository and a results report (PDFs included).

**Result in one sentence.** For these models, a high-rate CRC code plus ORBGRAND does not improve image quality on AWGN (at most +0.08 dB PSNR) or fading channels (the gains come from the interleaver), training with GRAND in the loop makes the model slightly worse, and the only measurable win for GRAND is for receivers that can only deliver hard bits.

Branches: `main` is the untouched BitSemCom code, `grand-implementation` has everything described here.

## 1. Repository map

| Path | What it is |
| --- | --- |
| `main.py`, `network/`, `data/`, `loss/`, `utils.py` | the original BitSemCom code (Swin/WITT autoencoder, Gumbel bit mapper, QPSK soft channel, training and test loop) |
| `train_bitsemcom.py` | our trainer: warmup + cosine schedule, fixed step budget, validation, optional GRAND in the loop |
| `grand/` | pure-numpy ORBGRAND decoder (port of Ken Duffy's MATLAB code), CRC and random linear codes, self-test, validation against Duffy's published curves |
| `experiments/` | coded link and channels, evaluation sweeps, metrics, diagnostics, all plotting scripts (see the table in section 4) |
| `matlab/` | bridge to Duffy's MATLAB GRAND for a bit-exact check of the Python port and for decoding with the reference code |
| `BitSemCom_Colab.ipynb` | runs the round-2 experiments on Google Colab, cell by cell |
| `results/` | raw per-image results (`*.jsonl`), summary tables, figures, training logs |
| `docs/` | the two reports, their figures, and the LaTeX documents |
| `Pretrain/` (git-ignored) | model checkpoints |
| `datasets/` (git-ignored) | Kodak images and DIV2K |

## 2. Setup

Python 3.10 or newer and PyTorch 2.x. A CUDA GPU is needed for everything that runs the autoencoder (training, the sweeps, the LLR study): the original Swin code places its attention masks on the GPU (`network/encoder.py`). The GRAND decoder, its self-test and validation, and the plotting scripts run on the CPU.

```
git clone https://github.com/SafaaSalman/BitSemCom -b grand-implementation
cd BitSemCom
pip install -r requirements.txt
python grand/selftest.py          # decoder and code self-test, a few seconds, no GPU needed
```

`lpips` and `open_clip_torch` are only needed for the LPIPS and CLIP metrics; `thop` and `tensorboardX` only for the original `main.py`.

### Data

```
# Kodak, 24 images (test set of everything)
mkdir -p datasets/kodak_full
for i in $(seq -w 1 24); do wget -q -O datasets/kodak_full/kodim$i.png https://r0k.us/graphics/kodak/kodak/kodim$i.png; done
python experiments/make_kodak_patches.py        # -> datasets/kodak_patches256 (centre 256x256 crops)

# DIV2K train HR, 800 images, ~3.5 GB (training only)
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip && unzip -q DIV2K_train_HR.zip -d datasets/
```

Every number in the reports was measured on the 24 centre crops `datasets/kodak_patches256`.

### Checkpoints

Put model files in `Pretrain/`. Three kinds exist:

| File | What it is | Where to get it |
| --- | --- | --- |
| `WITT_BitSC_C{16,32,64}.model` | the authors' models trained on an error-free channel (stage 1) | Baidu Netdisk link in section 3, extraction code `SJTU` |
| `BitSemCom_C{16,32,64}_stage2_ep*.model` | the same models after fine-tuning on a noisy QPSK channel with `main.py --pass_channel` (stage 2). Starting point of all our experiments | the project's shared Drive folder `BitSemCom/Pretrain` (ask Safaa) |
| `runs/C32_bench/best.model`, `runs/C32_grand/best.model` | round-2 models: C = 32 retrained with a proper schedule, without and with GRAND in the loop | same Drive folder, `BitSemCom/runs` |

Our scripts find the newest `Pretrain/*C{C}_*.model` automatically when `--ckpt` / `--init` is not given.

## 3. Reproducing the original BitSemCom (before GRAND)

The original pipeline is `main.py`: image → Swin encoder → bit mapper (Gumbel-softmax, C bits per token, 256 tokens for a 256×256 image) → QPSK → AWGN → soft demodulation → P(bit = 1) → Swin decoder → image. Three things to know before running it:

1. The dataset folders are hard-coded in the `config` class of `main.py`: `train_data_dir` (DIV2K) and `test_data_dir` (`kodak_patches_all` for `--testset kodak`). Edit them to your paths, for example `datasets/DIV2K_train_HR` and `datasets/kodak_patches256`.
2. It needs a CUDA GPU and the packages `lpips`, `thop`, `tensorboardX`.
3. The training loop has no end (`tot_epoch = 10,000,000`): it saves a checkpoint every 20 epochs to `history/BitSC_Prob/<timestamp>/models/` and runs a test at the same time. Stop it by hand when the test PSNR stops improving. The learning rate decays by `--decay` every `--step` epochs down to `--min_lr`, and each epoch uses one integer SNR drawn from `[--snr_min, --snr_max]`.

**Stage 1, error-free channel** (what the authors' `WITT_BitSC_*` checkpoints are): the same command without `--pass_channel`.

**Stage 2, noisy channel** (C = 32 example from the authors' readme; `--pretrain` is the model to start from):

```
python main.py --training --gpu 0 --C 32 --lr 1e-4 --step 200 --min_lr 5e-6 --pass_channel \
    --px BitSC --snr_min -3 --snr_max 6 --pretrain Pretrain/WITT_BitSC_C32.model
```

**Testing** a model at one SNR (no `--training`): encodes the test images, sends them through the QPSK channel at `--test_snr` dB, decodes, and prints PSNR, MS-SSIM and LPIPS. It also saves histograms of the bit-mapper logits and received probabilities to `test_probs_C<C>/`.

```
python main.py --gpu 0 --C 32 --pass_channel --pretrain Pretrain/BitSemCom_C32_stage2_ep480.model --test_snr 6
```

Model sizes: C = 16 / 32 / 64 bits per token give 4096 / 8192 / 16384 bits per 256×256 image (compression ratios 1/32, 1/16, 1/8). The authors' pretrained stage-1 models: [Baidu Netdisk](https://pan.baidu.com/s/1rRuC-7AAAQCqAOIS23HnNg), extraction code `SJTU`. Their visual comparison is `Visual_results.png`.

Our evaluation code never imports `main.py`; `experiments/bitsemcom_io.py` rebuilds the same model configuration and loads the checkpoints with a strict key check.

## 4. What we added

### The coded link

Between the bit mapper and the semantic decoder we insert: pad the bits to blocks of k = 112 → systematic CRC(128,112) code (polynomial 0x9eb2, 16 parity bits per block) → fixed pseudo-random interleaver over the whole image → QPSK → channel (AWGN or block Rayleigh fading, with optional CSI errors) → LLR per bit → ORBGRAND decodes every block with a guess budget → the first 112 bits of each decoded block go to the decoder as exact 0/1 values; blocks the decoder gave up on (abandoned) are passed as the plain soft values instead (soft fallback). The code is implemented once in numpy (`experiments/channel_coding.py`, `CodedLink.transmit`) and reused inside training through a process pool (`experiments/grand_layer.py`).

### Scripts

| Script | Role |
| --- | --- |
| `experiments/run_grand.py` | round-1 sweep: one model, many SNRs, all receivers; writes `results/<name>.jsonl` |
| `experiments/evaluate.py` | round-2 sweep: several models × named channel conditions × SNRs × receivers, with PSNR, MS-SSIM, LPIPS, CLIP; optional MATLAB decoding |
| `experiments/channel_coding.py` | the coded link, QPSK channel, interleaver (numpy, evaluation) |
| `experiments/torch_channel.py` | the same QPSK channel in torch, differentiable, for training |
| `experiments/grand_layer.py` | GRAND in the training loop: forward = coded link + ORBGRAND, backward = soft-channel gradient |
| `experiments/conditions.py` | names of channel conditions: `awgn`, `awgn_over3`, `awgn_under3`, `fade8`, `fade8_h0.1`, ... |
| `experiments/metrics.py` | PSNR, MS-SSIM, LPIPS, CLIP similarity |
| `experiments/plot_llr.py` | LLR study: what the decoder and GRAND receive under each condition |
| `experiments/plot_results.py`, `plot_training.py`, `plot_review_gains.py` | figures from the JSONLs and training logs |
| `experiments/summarize.py` | markdown tables from round-1 JSONLs |
| `experiments/bit_importance.py`, `channel_importance.py` | round-1 diagnostics: which bit errors hurt the decoder |
| `experiments/grand_verify.py`, `grand_backend.py`, `matlab/` | bit-exact check against Duffy's MATLAB code; MATLAB / Octave decoding backend |
| `grand/selftest.py`, `grand/validate_duffy.py` | decoder self-test; reproduction of Duffy's published BLER curves |

### Receivers compared

| Name | What the semantic decoder receives |
| --- | --- |
| `noiseless` | the exact bits (ceiling for any channel code) |
| `uncoded_soft` | P(bit = 1) from the QPSK soft demodulator: the original BitSemCom receiver |
| `uncoded_soft_il` | the same, with the coded link's interleaver but no code (isolates the interleaving gain) |
| `uncoded_hard` | hard 0/1 decisions |
| `coded_soft` | interleaver + CRC(128,112) + ORBGRAND; abandoned blocks fall back to the soft values |
| `coded_hard` | the same with hard-decision fallback |
| `*_ebn0`, `*_noil` | coded path at equal energy per semantic bit; coded path without interleaver (round 1) |

### Conventions

- LLR positive means bit 0 (same as Duffy's code); the decoder consumes P(bit = 1) = sigmoid(−LLR).
- SNR is Es/N0 per QPSK symbol in dB. Raw bit error rate = Q(√(Es/N0)): 16% at 0 dB, 2.3% at 6 dB.
- Same (image, SNR, seed) gives the same semantic bits and the same channel noise for every receiver, so every comparison is paired.
- Equal Eb/N0 for a rate-R code is Es/N0 + 10 log10(R), i.e. −0.58 dB for 112/128.

## 5. Reproducing round 1 (C = 16 / 32 / 64, stage-2 checkpoints)

Needs a CUDA GPU; runs in minutes. Outputs go to `results/<name>.jsonl`; `summarize.py` turns them into the tables of `docs/grand_results.md`.

```
python grand/selftest.py
python grand/validate_duffy.py --mat matlab/GRAND-MATLAB/RESULTS/ORBGRAND_RLC_128_116_1.mat --variant basic --ebn0 4,5,6
python experiments/run_grand.py --C 16 --snrs 2,4,5,6,7,8,10 --code crc:0x9eb2:128:112 --out results/C16_crc128_112_patches.jsonl
python experiments/run_grand.py --C 16 --snrs 6,10,14,18,22 --channel blockfading --fade_block 8 \
    --paths noiseless,uncoded_soft_np,uncoded_soft_il,uncoded_hard,coded_hard,coded_soft,coded_soft_noil \
    --out results/C16_crc128_112_fading8.jsonl
python experiments/bit_importance.py --C 16
python experiments/channel_importance.py --C 16
python experiments/summarize.py results/C16_crc128_112_patches.jsonl
```

Other round-1 variants (`--code rlc:128:104:0`, `--code crc:0x9eb2:64:48`, `--max_queries 1000`, `--dec_snr 10`, `--C 32`, `--C 64`) are the other files in `results/`. A code spec is `crc:<Koopman polynomial>:<n>:<k>` or `rlc:<n>:<k>:<seed>`. The first round ran the coded paths without an interleaver on AWGN; `run_grand.py` now interleaves by default, so add `--no_interleave` to reproduce those rows exactly.

## 6. Reproducing round 2 (C = 32 benchmark vs GRAND-in-the-loop)

Everything below runs locally with a GPU or on Colab; `BitSemCom_Colab.ipynb` contains the same commands as cells, keeps models, data and results on Google Drive under `MyDrive/BitSemCom`, and clones the code fresh each session.

**LLR study** (minutes, no training):

```
python experiments/plot_llr.py --C 32 --limit 8 --snrs 0,4,8,12 \
    --conditions awgn,awgn_over3,awgn_under3,fade8,fade8_h0.1,fade1 --budget 100000 --out results/figures
```

**Benchmark model** (30k steps, 5.8 h on an L4, 3.9 h on an A100):

```
python train_bitsemcom.py --C 32 --init Pretrain/BitSemCom_C32_stage2_ep480.model \
    --train_dir datasets/DIV2K_train_HR --val_dir datasets/kodak_patches256 \
    --steps 30000 --batch 16 --lr 1e-4 --min_lr 1e-6 --warmup 500 \
    --snr_min 0 --snr_max 10 --val_every 1000 --patience 0 --num_workers 4 --out runs/C32_bench
```

**GRAND-in-the-loop model** (same recipe; add `--resume` to continue an interrupted run):

```
python train_bitsemcom.py --C 32 --init Pretrain/BitSemCom_C32_stage2_ep480.model \
    --train_dir datasets/DIV2K_train_HR --val_dir datasets/kodak_patches256 \
    --steps 30000 --batch 16 --lr 1e-4 --min_lr 1e-6 --warmup 500 \
    --snr_min 0 --snr_max 10 --val_every 1000 --patience 0 --num_workers 4 \
    --grand crc:0x9eb2:128:112 --grand_budget 2000 --grand_fallback soft --grand_workers 8 --out runs/C32_grand
```

Leave early stopping off (`--patience 0`): with a 1e-4 peak learning rate the validation score dips below its starting value for the first 6000 to 7000 steps and only recovers as the learning rate decays. Each run writes `log.csv`, `val.csv`, `best.model`, `last.model` and `ckpt_last.pt` to `--out`. On a T4 use `--amp_dtype fp16`. If ORBGRAND dominates the step time, lower `--grand_budget` or set `--grand_images 8` (half of each batch coded).

**Evaluation** (the budget-2000 sweep is the one in the reports; the 100,000 sweep is the budget study):

```
python experiments/evaluate.py --C 32 \
    --ckpt stage2=Pretrain/BitSemCom_C32_stage2_ep480.model \
    --ckpt bench=runs/C32_bench/best.model --ckpt grand=runs/C32_grand/best.model \
    --snrs 0,2,4,6,8,10,12 --conditions awgn,awgn_over3,awgn_under3,fade8,fade8_h0.1 \
    --paths noiseless,uncoded_soft,uncoded_soft_il,uncoded_hard,coded_soft --budget 2000 \
    --save_images 3 --save_snr 4 --out results/eval_C32.jsonl
python experiments/evaluate.py --C 32 \
    --ckpt bench=runs/C32_bench/best.model --ckpt grand=runs/C32_grand/best.model \
    --snrs 0,2,4,6,8,10,12 --conditions awgn,fade8 --paths noiseless,uncoded_soft,uncoded_soft_il,coded_soft \
    --budget 100000 --out results/eval_C32_b1e5.jsonl
```

**Figures and tables**:

```
python experiments/plot_results.py results/eval_C32.jsonl --out results/figures
python experiments/plot_results.py results/eval_C32_b1e5.jsonl --out results/figures --tag b1e5
python experiments/plot_training.py results/train_logs/C32_bench results/train_logs/C32_grand --out results/figures
python experiments/plot_review_gains.py results/eval_C32.jsonl --b1e5 results/eval_C32_b1e5.jsonl \
    --images results/images_C32 --out docs/figures/grand_review
```

## 7. The thesis questions these runs answer

1. `bench` vs `stage2`: did the proper schedule improve the model (section 3 of the review)?
2. `grand` vs `bench` with the coded receiver: does training with GRAND in the loop change what the decoder makes of GRAND's output (section 4)?
3. `coded_soft` vs `uncoded_soft_il`: does the code add anything on top of the interleaver, on AWGN and in fading (section 5)?
4. `awgn_over3`, `awgn_under3`, `fade8_h0.1`: what imperfect channel knowledge does to each receiver (section 6)?
5. LPIPS and CLIP vs PSNR: does a perceptual or semantic metric change the ranking of receivers (section 8)?

## 8. MATLAB reference decoder (Duffy's code)

The Python decoder is a port of [GRAND-MATLAB](https://github.com/kenrduffy/GRAND-MATLAB). To check it bit for bit against the original, export a test set, decode it with the MATLAB functions and compare. ORBGRAND is deterministic and both sides use the same query accounting, so the expected result is zero mismatches.

```
bash matlab/fetch_grand_matlab.sh                 # downloads Duffy's .m files into matlab/GRAND-MATLAB
python experiments/grand_verify.py export         # -> results/grand_verify/testset.mat (740 blocks)
# MATLAB:  >> cd matlab; addpath('GRAND-MATLAB'); verify_port('../results/grand_verify/testset.mat')
python experiments/grand_verify.py run --backend octave   # or: decode with Octave, then compare
```

`matlab/README.md` has the details, including the one Octave incompatibility (`cumsum(d,'reverse')`) and the shim in `matlab/octave_compat/`. This check has not been run with real MATLAB yet. `evaluate.py --grand_backend matlab` runs a whole sweep through the reference decoder (slow).

## 9. Where the results are

| What | Where |
| --- | --- |
| Round-1 raw results and tables | `results/C*_*.jsonl`, `results/summary_*.md`, `results/bit_importance_*.md`, `results/channel_importance_*.md` |
| Round-2 raw results | `results/eval_C32.jsonl` (12,600 rows), `results/eval_C32_b1e5.jsonl` |
| Round-2 training logs | `results/train_logs/C32_bench`, `C32_grand`, `C32_grand_abandoned` (`log.csv`, `val.csv`, console output) |
| Figures | `results/figures/` (all), `docs/figures/grand_review/` (the ones in the review) |
| Example reconstructions | `results/images_C32/` (git-ignored, on Drive) |

Each JSONL row is one (model, image, condition, SNR, receiver) with the four metrics and, for coded receivers, the block statistics: `ber_channel`, `ber_after`, `bler_after`, `abandon_rate`, `miscorrect_rate`, `avg_queries`.

## 10. Licenses and citations

- BitSemCom: the original authors' code and models (SJTU); cite arXiv 2510.26225.
- `grand/orbgrand.py` is a port of Ken Duffy's GRAND-MATLAB, released under the GRAND Codebase Non-Commercial Academic Research Use License: non-commercial use only, and cite Duffy et al. (ICASSP 2021, IEEE Trans. Signal Processing 2022) in any publication.
