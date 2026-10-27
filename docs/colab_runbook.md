# Colab runbook (plan v2: C=32 benchmark vs GRAND-in-the-loop)

Everything below was smoke-tested locally on small samples; the full runs are meant for
Colab Pro (A100/L4). Paths assume the repo is cloned to `/content/BitSemCom` and Google
Drive is mounted at `/content/drive` with a folder `MyDrive/BitSemCom` holding the
checkpoints and receiving the outputs.

## 0. Setup (every session)

```python
from google.colab import drive; drive.mount('/content/drive')
%cd /content
!git clone https://github.com/SafaaSalman/BitSemCom -b grand-implementation   # private: use a token or upload a zip
%cd /content/BitSemCom
!pip -q install timm lpips open_clip_torch scipy
!mkdir -p Pretrain datasets runs results/figures
!cp /content/drive/MyDrive/BitSemCom/Pretrain/BitSemCom_C32_stage2_ep480.model Pretrain/
```

Data (once, then keep on Drive):

```python
# DIV2K train HR (800 images, ~3.5 GB) and Kodak (24 images)
!wget -q http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip && unzip -q DIV2K_train_HR.zip -d datasets/
!mkdir -p datasets/kodak_full && for i in $(seq -w 1 24); do wget -q -O datasets/kodak_full/kodim$i.png https://r0k.us/graphics/kodak/kodak/kodim$i.png; done
# 256x256 centre crops for validation / evaluation (same crops as the first round)
!python - <<'EOF'
import os; from PIL import Image
os.makedirs('datasets/kodak_patches256', exist_ok=True)
for f in sorted(os.listdir('datasets/kodak_full')):
    im = Image.open('datasets/kodak_full/'+f); w,h = im.size; l,t = (w-256)//2,(h-256)//2
    im.crop((l,t,l+256,t+256)).save('datasets/kodak_patches256/'+f)
EOF
```

GPU note: bf16 autocast (default) is native on A100/L4. On a T4 use `--amp_dtype fp16`
(or `--amp 0`).

## 1. LLR shape study (minutes, no training) - first thing to show the supervisor

```bash
python experiments/plot_llr.py --C 32 --limit 8 --snrs 0,4,8,12 \
    --conditions awgn,awgn_over3,awgn_under3,fade8,fade8_h0.1,fade1 --budget 100000 --out results/figures
```

Produces `results/figures/llr_<condition>.png`, `llr_summary.png/.md`, `bitmapper.png`.

## 2. Benchmark model (no GRAND)

```bash
python train_bitsemcom.py --C 32 --init Pretrain/BitSemCom_C32_stage2_ep480.model \
    --train_dir datasets/DIV2K_train_HR --val_dir datasets/kodak_patches256 \
    --steps 30000 --batch 16 --lr 1e-4 --min_lr 1e-6 --warmup 500 \
    --snr_min 0 --snr_max 10 --val_every 1000 --patience 8 \
    --num_workers 4 --max_hours 11 --out runs/C32_bench
# if the session dies: add --resume and rerun the same command
# (without --resume the same --out starts a fresh run and overwrites log.csv / val.csv)
```

Expected speed: ~0.15-0.25 s/step on A100 (batch 16), i.e. 1.5-2 h for 30k steps.

## 3. GRAND-in-the-loop model

```bash
python train_bitsemcom.py --C 32 --init Pretrain/BitSemCom_C32_stage2_ep480.model \
    --train_dir datasets/DIV2K_train_HR --val_dir datasets/kodak_patches256 \
    --steps 30000 --batch 16 --lr 1e-4 --min_lr 1e-6 --warmup 500 \
    --snr_min 0 --snr_max 10 --val_every 1000 --patience 8 \
    --grand crc:0x9eb2:128:112 --grand_budget 2000 --grand_fallback soft --grand_workers 8 \
    --num_workers 4 --max_hours 11 --out runs/C32_grand
```

The gradient through the coded link is the uncoded soft channel's Jacobian (`--grand_grad soft`,
default), i.e. the same gradient path as the benchmark run, so the two runs differ only in the
forward signal the decoder sees; `--grand_grad identity` is plain straight-through. The query
budget is accounted like Duffy's MATLAB code (checked once per weight level), so a block may use
somewhat more than `--grand_budget` queries.

The log line shows the GRAND cost per step (`... | GRAND aband .. misc .. bler .. q .. 0.xxs`).
If the decoder time dominates (few vCPUs), lower `--grand_budget` to 1000 or set
`--grand_images 8` (half of each batch coded, half uncoded). If the process pool ever hangs
on Colab, `--grand_workers 0` decodes in-process (slower, always works).

Optional third run, imperfect-CSI augmentation: add `--snr_err_train 3 --h_err_train 0.1`
and `--out runs/C32_grand_csi` (and `--channel blockfading --fade_block 8` if fading is
the target channel).

Keep the outputs: `!cp -r runs /content/drive/MyDrive/BitSemCom/` (best.model, last.model,
log.csv, val.csv per run).

## 4. Evaluation (all models x conditions x SNR x receivers, four metrics)

```bash
# primary sweep: the query budget the GRAND model was trained with. Its decoder learned the
# abandonment / miscorrection statistics of that budget (2000: 0-4 dB mostly abandoned -> soft
# fallback, 1-5 % miscorrected); at 1e5 queries 4-5 dB AWGN has 40-65 % miscorrected blocks.
python experiments/evaluate.py --C 32 \
    --ckpt stage2=Pretrain/BitSemCom_C32_stage2_ep480.model \
    --ckpt bench=runs/C32_bench/best.model --ckpt grand=runs/C32_grand/best.model \
    --snrs 0,2,4,6,8,10,12 --conditions awgn,awgn_over3,awgn_under3,fade8,fade8_h0.1 \
    --paths noiseless,uncoded_soft,uncoded_hard,coded_soft --budget 2000 \
    --save_images 3 --save_snr 4 --out results/eval_C32.jsonl
python experiments/plot_results.py results/eval_C32.jsonl --out results/figures
python experiments/plot_training.py runs/C32_bench runs/C32_grand --out results/figures
# secondary sweep: large budget (fewer abandonments, more miscorrections at low SNR)
python experiments/evaluate.py --C 32 \
    --ckpt bench=runs/C32_bench/best.model --ckpt grand=runs/C32_grand/best.model \
    --snrs 0,2,4,6,8,10,12 --conditions awgn,fade8 --paths noiseless,uncoded_soft,coded_soft \
    --budget 100000 --out results/eval_C32_b1e5.jsonl
python experiments/plot_results.py results/eval_C32_b1e5.jsonl --out results/figures --tag b1e5
```

Figures: `quality_<condition>.png`, `gain_<metric>.png`, `grand_<condition>.png`,
`training.png`, tables in `results/figures/summary.md`, example reconstructions in
`results/images_C32/`.

## 5. MATLAB reference decoder

Bit-exact check of the Python port (run on any machine with MATLAB, or Octave on Colab):

```bash
python experiments/grand_verify.py export               # results/grand_verify/testset.mat
# MATLAB:  >> cd matlab; addpath('GRAND-MATLAB'); verify_port('../results/grand_verify/testset.mat')
# Octave on Colab:
!apt-get -qq install octave && bash matlab/fetch_grand_matlab.sh
python experiments/grand_verify.py run --backend octave   # decodes with Duffy's .m files, then compares
```

The python backend uses the same query accounting as the reference (`--budget_mode level`,
default), so the two backends must agree bit for bit, not just statistically.

Final numbers through the reference decoder (slow; use a reduced grid):

```bash
python experiments/evaluate.py --C 32 --ckpt bench=runs/C32_bench/best.model --ckpt grand=runs/C32_grand/best.model \
    --snrs 4,6,8,10 --conditions awgn,fade8 --paths noiseless,uncoded_soft,coded_soft \
    --budget 10000 --grand_backend octave --out results/eval_C32_octave.jsonl
```

## 6. What to compare (the thesis questions)

1. `bench` vs `stage2`: did the proper schedule improve the model at all (noiseless and uncoded curves)? Caveat: `stage2` was trained at -3..6 dB and `bench` at 0..10 dB, so above 6 dB the two differ in training SNR range as well as in schedule. If that matters for the write-up, add a run with `--snr_min -3 --snr_max 6 --out runs/C32_bench_origrange`.
2. `grand` vs `bench` with the `coded_soft` receiver: does training with GRAND in the loop change what the decoder makes of GRAND's output, especially at 0-6 dB where miscorrections live?
3. `coded_soft` vs `uncoded_soft` under `awgn_over3` / `awgn_under3`: GRAND's decoded bits do not depend on the SNR estimate, the soft receiver's P(1) does (gain_*.png).
4. `fade8` vs `fade8_h0.1`: what channel-estimation error does to the block error rate and to image quality (grand_*.png, quality_*.png).
5. LPIPS and CLIP vs PSNR: whether the ranking of receivers changes with a perceptual / semantic metric.
