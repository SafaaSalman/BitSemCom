# BitSemCom: bit-level semantic image communication

This branch holds the original BitSemCom code (Shanghai Jiao Tong University, [arXiv 2510.26225](https://arxiv.org/abs/2510.26225)) with a small clean-up so that it is easier to run and to reproduce. The method is unchanged: same model, same bit mapper, same channel, same loss.

What BitSemCom does, in one sentence: a neural network turns an image into a short sequence of **bits**, the bits are sent over a noisy radio channel, and a second network rebuilds the image from what was received, without a separate channel code.

## 1. What is in this branch

| Path | What it is | Origin |
| --- | --- | --- |
| `main.py` | the authors' training and test script | original, untouched |
| `network/` | the model: Swin encoder and decoder, bit mapper, QPSK channel | original, untouched |
| `data/`, `loss/`, `utils.py` | data loaders, MSE and MS-SSIM, logging helpers | original, untouched |
| `Visual_results.png` | the authors' visual comparison | original |
| `train_bitsemcom.py` | a second trainer for the same model, with a fixed length and a learning-rate schedule (section 6) | added |
| `experiments/bitsemcom_io.py` | builds the model and loads checkpoints without importing `main.py` | added |
| `experiments/make_kodak_patches.py` | makes the 24 Kodak test crops | added |
| `experiments/plot_training.py` | plots the training curves of `train_bitsemcom.py` | added |
| `BitSemCom_Colab.ipynb` | runs the training on Google Colab | added |
| `Pretrain/`, `datasets/`, `runs/` | checkpoints, data, training outputs. Not in git, you create them | |

The clean-up compared with the authors' upload: this README, a short `requirements.txt` (the original pinned 186 packages of a whole conda environment), a `.gitignore`, and the removal of an empty `train.py` and of `Pretrain/model.txt` (its download link is now in section 5).

## 2. How the model works

```
image 256x256 ──> Swin encoder ──> bit mapper ──> QPSK ──> AWGN channel ──> soft demodulator ──> Swin decoder ──> image
                  256 tokens       C bits per     2 bits    adds noise      P(bit = 1) for
                                   token          per symbol                every bit
```

1. **Encoder** (`network/encoder.py`). A Swin Transformer reduces the image by a factor 16 in each direction. A 256×256 image becomes 16×16 = 256 tokens, each with C numbers.
2. **Bit mapper** (`Resample_Layer` in `network/network.py`). Every number becomes one bit, 0 or 1. The layer computes two scores per bit and samples the bit with the Gumbel-softmax trick, which lets gradients pass through a discrete decision during training.
3. **Channel** (`QPSK_soft` in `network/network.py`). Two bits form one QPSK symbol. Gaussian noise is added (AWGN). The SNR is Es/N0 per QPSK symbol, in dB.
4. **Soft demodulator** (same function). For every bit the receiver computes the probability that a 1 was sent. A clean bit gives a value near 0 or 1, a noisy bit gives something near 0.5.
5. **Decoder** (`network/decoder.py`). A second Swin Transformer takes these probabilities, not hard 0/1 decisions, and rebuilds the image.

Encoder and decoder are also told the SNR, so one model works over a range of SNRs. The training loss is the mean squared error between the input image and the rebuilt image.

Model sizes (256×256 image):

| C (bits per token) | bits per image | QPSK symbols per pixel (k in the authors' table) |
| --- | --- | --- |
| 16 | 16 × 256 = 4096 | 1/32 |
| 32 | 32 × 256 = 8192 | 1/16 |
| 64 | 64 × 256 = 16384 | 1/8 |

## 3. Setup

Python 3.10 or newer, PyTorch 2.x, and a CUDA GPU. A GPU is required: the original encoder places its attention masks on the GPU (`network/encoder.py`, line 138), so nothing that runs the model works on the CPU.

```
git clone https://github.com/SafaaSalman/BitSemCom -b bitsemcom-original
cd BitSemCom
pip install -r requirements.txt
```

`lpips`, `thop` and `tensorboardX` are only needed by `main.py`. `train_bitsemcom.py` runs without them.

## 4. Data

```
# Kodak, 24 images (test set)
mkdir -p datasets/kodak_full
for i in $(seq -w 1 24); do wget -q -O datasets/kodak_full/kodim$i.png https://r0k.us/graphics/kodak/kodak/kodim$i.png; done
python experiments/make_kodak_patches.py        # -> datasets/kodak_patches256 (centre 256x256 crops)

# DIV2K train HR, 800 images, about 3.5 GB (training set)
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip && unzip -q DIV2K_train_HR.zip -d datasets/
```

## 5. Checkpoints

Put model files in `Pretrain/` (create the folder).

| File | What it is | Where to get it |
| --- | --- | --- |
| `WITT_BitSC_C{16,32,64}.model` | the authors' pretrained models. Their readme describes them as trained "under error-free channel conditions", to be fine-tuned on a noisy channel afterwards | [Baidu Netdisk](https://pan.baidu.com/s/1rRuC-7AAAQCqAOIS23HnNg), extraction code `SJTU` |
| `BitSemCom_C{16,32,64}_stage2_ep*.model` | the same models after fine-tuning on the noisy QPSK channel with `main.py --pass_channel` | the project's shared Drive folder (ask Safaa) |

## 6. Training with `train_bitsemcom.py`

`main.py` trains without an end and with an almost constant learning rate (section 7). `train_bitsemcom.py` trains the same model on the same channel with the same loss, but for a fixed number of steps, with a learning-rate schedule, a regular validation and automatic saving of the best model.

```
python train_bitsemcom.py --C 32 --init Pretrain/BitSemCom_C32_stage2_ep480.model \
    --train_dir datasets/DIV2K_train_HR --val_dir datasets/kodak_patches256 \
    --steps 30000 --batch 16 --lr 1e-4 --min_lr 1e-6 --warmup 500 \
    --snr_min 0 --snr_max 10 --val_every 1000 --out runs/C32_bench
```

This run took 5.8 hours on an L4 GPU on Colab. `--init` is the model to start from; `--init none` starts from random weights. If the run is interrupted, run the same command again with `--resume`. On a T4 GPU add `--amp_dtype fp16`. For C = 32, batch 16 needs about 13 GB of GPU memory and batch 8 about 7 GB; lower `--batch` on a smaller GPU.

What the run writes to `--out`:

| File | Content |
| --- | --- |
| `best.model` | weights with the best validation score. Use this one |
| `last.model` | weights at the last validation |
| `ckpt_last.pt` | full training state for `--resume` |
| `log.csv` | training loss, PSNR and learning rate every 50 steps |
| `val.csv` | validation PSNR and MS-SSIM per SNR |
| `config.json` | the arguments of the run |

Every 1000 steps the trainer rebuilds the 24 Kodak crops at 0, 3, 6 and 10 dB and prints one line. Each image is decoded twice:

- `noiseless`: the decoder receives the exact bits. This is the best the model can do at any SNR.
- `uncoded_soft`: the bits go through the QPSK + AWGN channel and the soft demodulator. This is the normal BitSemCom receiver, and its mean PSNR is the score that selects `best.model`.

Plot the curves with:

```
python experiments/plot_training.py runs/C32_bench      # -> runs/C32_bench/training.png
```

Leave early stopping off (`--patience 0`, the default). With a peak learning rate of 1e-4 the validation score first drops below its starting value and only recovers after several thousand steps, when the learning rate has decayed.

### What differs from `main.py`

| | `main.py` | `train_bitsemcom.py` |
| --- | --- | --- |
| Model, bit mapper, channel, loss | original | the same code |
| Length | 10,000,000 epochs, stopped by hand | `--steps` (30,000 steps = 600 passes over DIV2K at batch 16) |
| Learning rate | multiplied by `--decay` (0.9) every `--step` epochs | 500 warmup steps, then cosine decay from `--lr` to `--min_lr` |
| Optimiser | Adam | AdamW with weight decay 0, which is the same update |
| SNR during training | one whole-number SNR per epoch | a new SNR for every batch, any value in the range |
| Gradient clipping | none | norm 1.0 |
| Precision | 32-bit | 16-bit (bf16) inside the Swin blocks, 32-bit for bit mapper and channel |
| Training crops | random 256×256 crop | random 256×256 crop and horizontal flip |
| Testing during training | every 20 epochs at that epoch's SNR | every `--val_every` steps at fixed SNRs with fixed noise |
| Saved models | one every 20 epochs | best and last |

## 7. Running the original `main.py`

Read these five points first. The first three stop the script before it starts.

1. **Use Linux, WSL or Colab.** `main.py` creates a folder named after the current time, for example `history/BitSC_Prob/2026-11-10 10:33:41`. Windows does not allow `:` in a folder name, so the script fails there with `OSError: [WinError 123]`.
2. **Training needs Python 3.11 or older, or a one-line fix.** Line 684 calls `random.randint(args.snr_min, args.snr_max)` with decimal numbers. Since Python 3.12 this raises `TypeError: 'float' object cannot be interpreted as an integer`. The fix is `random.randint(int(args.snr_min), int(args.snr_max))`. Testing is not affected.
3. **Set the dataset folders.** They are fixed in the `config` class of `main.py`: `train_data_dir` (line 262, DIV2K) and `test_data_dir` (line 265, `kodak_patches_all`). Change them to your paths, for example `datasets/DIV2K_train_HR` and `datasets/kodak_patches256`. Both folders must contain images, also when you only test: an empty training folder stops the script with `ValueError: num_samples should be a positive integer value`.
4. **Always pass `--gpu 0` and `--C`.** The defaults are GPU number 2 and C = 96.
5. **Training never ends by itself** (`tot_epoch = 10,000,000`). Every 20 epochs it saves a model to `history/BitSC_Prob/<time>/models/` and runs a test. Stop it by hand when the test PSNR no longer improves. One epoch is 50 steps (800 DIV2K images, batch 16).

**Training on the noisy channel** (C = 32):

```
python main.py --training --gpu 0 --C 32 --lr 1e-4 --step 200 --min_lr 5e-6 --pass_channel \
    --px BitSC --snr_min -3 --snr_max 6 --pretrain Pretrain/WITT_BitSC_C32.model
```

`--pass_channel` switches the bit mapper on; keep it in every command. `--pretrain` is the model to start from. `--snr_min` and `--snr_max` give the SNR range in dB; each epoch uses one whole-number SNR from it.

**Testing** a model at one SNR (no `--training`):

```
python main.py --gpu 0 --C 32 --pass_channel --pretrain Pretrain/BitSemCom_C32_stage2_ep480.model --test_snr 6
```

This sends the test images through the channel at `--test_snr` dB and prints PSNR, MS-SSIM and LPIPS. It also saves histograms of the bit-mapper scores and of the received probabilities to `test_probs_C32/`. Add `--save_img` to save the rebuilt images to `Image_samples/`. A model from `train_bitsemcom.py` is tested the same way: `--pretrain runs/C32_bench/best.model`.

## 8. Google Colab

`BitSemCom_Colab.ipynb` runs section 6 on Colab cell by cell: it mounts Google Drive, clones this branch, downloads DIV2K and Kodak once, trains, and plots the curves. Checkpoints, data and runs are kept on Drive under `MyDrive/BitSemCom`, so a session that ends can be continued by running the training cell again. The first cell of the notebook lists what to prepare.

## 9. Questions you may have

**What does "semantic communication" mean here?** The transmitter does not try to deliver every pixel exactly. It sends a compact description that the receiver's network can turn back into an image that looks right. Compression and protection against noise are learned together.

**Why bits? Other learned systems send real numbers.** Real radio hardware and standards work with bits and standard modulations such as QPSK. BitSemCom keeps the learned encoder and decoder but makes the interface in between a plain bit sequence.

**How can a network be trained through a 0/1 decision?** A hard decision has no gradient. The bit mapper uses Gumbel-softmax with the straight-through trick: the forward pass uses the hard bit, the backward pass uses the gradient of a smooth approximation.

**Why does the decoder get probabilities instead of bits?** A probability also says how sure the receiver is. A value of 0.5 tells the decoder "this bit is unknown", so it can rely on the neighbouring bits instead. This is what replaces the channel code.

**What do the SNR numbers mean?** SNR is Es/N0 per QPSK symbol in dB. At 0 dB about 16 % of the bits are wrong before the decoder, at 6 dB about 2.3 %, at 10 dB about 0.08 %.

**Which C should I start with?** C = 32. It is the size used in all examples here.

**Why is the PSNR printed during training different from the validation PSNR?** The training value is measured on random DIV2K crops at a random SNR, the validation value on the fixed Kodak crops at fixed SNRs.

**The validation score is worse than at step 0. Is something broken?** Not during the first thousands of steps when you start from a trained model, see the note on early stopping in section 6.

**`RuntimeError: checkpoint/model mismatch` or `size mismatch`.** The `--C` you passed is not the C of the checkpoint.

**`FileNotFoundError: no checkpoint for C=32 in Pretrain/`.** No `--init` was given and `Pretrain/` holds no model for this C. Pass `--init <file>` or `--init none`.

**Can I run it without a GPU?** No, see section 3.

## 10. Citation

The method, the model code and the pretrained models are the work of the BitSemCom authors (SJTU). Cite arXiv 2510.26225 when you use them.
