"""
Train or fine-tune BitSemCom (WITT + Gumbel bit mapper) with a fixed step budget and a
learning-rate schedule. The model, the bit mapper, the QPSK + AWGN channel and the MSE loss
are the original ones (network/); only the training procedure differs from main.py, whose
loop is open-ended (tot_epoch = 10,000,000) with an almost constant learning rate.

Example, C=32, starting from a stage-2 checkpoint:
  python train_bitsemcom.py --C 32 --init Pretrain/BitSemCom_C32_stage2_ep480.model \
      --train_dir datasets/DIV2K_train_HR --val_dir datasets/kodak_patches256 \
      --steps 30000 --batch 16 --snr_min 0 --snr_max 10 --out runs/C32_bench

Recipe: AdamW, linear warmup then cosine decay from --lr to --min_lr over --steps,
gradient clipping, SNR drawn per batch ~ U[snr_min, snr_max] (continuous), bf16 autocast
for the Swin blocks (bit mapper and channel always in fp32), validation on fixed images
and seeds every --val_every steps, best/last checkpoints, --patience early stop,
--max_hours session guard, --resume.

Validation decodes every image twice: "noiseless" (the decoder receives the exact bits) and
"uncoded_soft" (the bits go through QPSK + AWGN + soft demodulation, the original BitSemCom
receiver). The best checkpoint is the one with the highest mean "uncoded_soft" PSNR.

Outputs in --out: config.json, log.csv (training), val.csv (validation), ckpt_last.pt
(full state for --resume), best.model and last.model (state_dicts loadable with
experiments.bitsemcom_io.load_model or main.py --pretrain).
"""
import argparse
import csv
import glob
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from experiments.bitsemcom_io import load_images, load_model  # noqa: E402
from loss.distortion import MS_SSIM  # noqa: E402
from network.network import QPSK_soft  # noqa: E402

IMG_EXT = ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.PNG", "*.bmp")


# ----------------------------------------------------------------------------- data
class CropDataset(Dataset):
    def __init__(self, dirs, size=256, flip=True):
        self.files = []
        for d in dirs:
            for ext in IMG_EXT:
                self.files += glob.glob(os.path.join(d, "**", ext), recursive=True)
        self.files = sorted(set(self.files))
        if not self.files:
            raise FileNotFoundError("no images under %s" % dirs)
        self.size, self.flip = size, flip

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        im = Image.open(self.files[i]).convert("RGB")
        w, h = im.size
        s = self.size
        if w < s or h < s:
            scale = max(s / w, s / h)
            im = im.resize((math.ceil(w * scale), math.ceil(h * scale)), Image.BICUBIC)
            w, h = im.size
        x0 = int(torch.randint(0, w - s + 1, (1,)))
        y0 = int(torch.randint(0, h - s + 1, (1,)))
        im = im.crop((x0, y0, x0 + s, y0 + s))
        x = torch.from_numpy(np.array(im, dtype=np.uint8)).permute(2, 0, 1).float() / 255.0
        if self.flip and torch.rand(1).item() < 0.5:
            x = x.flip(-1)
        return x


def infinite(loader):
    while True:
        for b in loader:
            yield b


# ----------------------------------------------------------------------------- model helpers
def set_resolution(net, H, W):
    if H != net.H or W != net.W:
        net.encoder.update_resolution(H, W)
        net.decoder.update_resolution(H // (2 ** net.downsample), W // (2 ** net.downsample))
        net.H, net.W = H, W


def encode(net, x, snr_tx, amp, amp_dtype):
    """image -> (logits, y_soft, bits (B,T,C) float {0,1} with Gumbel straight-through)."""
    with torch.autocast("cuda", dtype=amp_dtype, enabled=amp):
        feat = net.encoder(x, snr_tx, net.model)
    with torch.autocast("cuda", enabled=False):
        logits, y_soft, bits = net.resample(feat.float(), net.tau, net.tau_lg)
    return logits, y_soft, bits


def channel(bits, snr):
    """The original BitSemCom channel: QPSK + AWGN + soft demodulation -> P(bit = 1)."""
    _, p1 = QPSK_soft(bits, snr)
    return p1


def decode(net, p1, snr_rx, amp, amp_dtype):
    with torch.autocast("cuda", dtype=amp_dtype, enabled=amp):
        recon = net.decoder(p1, snr_rx, net.model)
    return recon.float()


def mse_loss_255(x, recon):
    return F.mse_loss(x * 255.0, recon.clamp(0.0, 1.0) * 255.0)


def psnr_from_mse(mse):
    """PSNR (dB) from a mean squared error on the 0..255 scale; 100 dB for mse == 0."""
    return 10 * math.log10(255.0 ** 2 / mse) if mse > 0 else 100.0


# ----------------------------------------------------------------------------- validation
@torch.no_grad()
def validate(net, val_images, snrs, msssim_fn, device, amp, amp_dtype):
    """Fixed images and seeds. The global RNG state is saved and restored so that the training
    noise stream (Gumbel, channel) does not restart from the same point after every validation."""
    rng_cpu, rng_cuda = torch.get_rng_state(), torch.cuda.get_rng_state(device)
    net.eval()
    acc = defaultdict(lambda: defaultdict(list))
    try:
        for si, snr in enumerate(snrs):
            for ii, (_, img) in enumerate(val_images):
                img = img.to(device)
                set_resolution(net, img.shape[2], img.shape[3])
                seed = 7_000_000 + ii * 100 + si
                torch.manual_seed(seed)
                _, _, bits = encode(net, img, snr, amp, amp_dtype)
                recv = {"noiseless": bits}
                torch.manual_seed(seed + 1)
                recv["uncoded_soft"] = channel(bits, snr)
                for name, p1 in recv.items():
                    rec = decode(net, p1, snr, amp, amp_dtype).clamp(0, 1)
                    mse = F.mse_loss(img * 255.0, rec * 255.0).item()
                    acc[(snr, name)]["psnr"].append(psnr_from_mse(mse))
                    acc[(snr, name)]["msssim"].append(1.0 - msssim_fn(img, rec).mean().item())
    finally:
        net.train()
        torch.set_rng_state(rng_cpu)
        torch.cuda.set_rng_state(rng_cuda, device)
    rows = []
    for (snr, name), d in acc.items():
        rows.append(dict(snr=snr, receiver=name, psnr=float(np.mean(d["psnr"])), msssim=float(np.mean(d["msssim"]))))
    return rows


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--C", type=int, default=32)
    ap.add_argument("--init", default="", help="state_dict to start from ('' = newest Pretrain/*C{C}_*.model, 'none' = scratch)")
    ap.add_argument("--train_dir", nargs="+", required=True, help="one or more folders of training images (searched recursively)")
    ap.add_argument("--val_dir", default="datasets/kodak_patches256")
    ap.add_argument("--val_limit", type=int, default=0)
    ap.add_argument("--val_snrs", default="0,3,6,10")
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--out", required=True)
    # optimisation
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--amp", type=int, default=1, help="1 = autocast for the Swin blocks")
    ap.add_argument("--amp_dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1024)
    ap.add_argument("--patience", type=int, default=0, help="stop after this many validations without improvement (0 = off)")
    ap.add_argument("--max_hours", type=float, default=0.0, help="save and exit after this wall time (0 = off)")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/ckpt_last.pt")
    ap.add_argument("--log_every", type=int, default=50)
    # channel (QPSK + AWGN, SNR = Es/N0 in dB)
    ap.add_argument("--snr_min", type=float, default=0.0)
    ap.add_argument("--snr_max", type=float, default=10.0)
    args = ap.parse_args()

    device = torch.device("cuda")
    amp = bool(args.amp)
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    use_scaler = amp and amp_dtype == torch.float16
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # model
    init = args.init if args.init else None
    net = load_model(args.C, device, checkpoint=init)
    net.train()
    set_resolution(net, args.crop, args.crop)

    # data
    ds = CropDataset(args.train_dir, size=args.crop)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=args.num_workers,
                        pin_memory=True, persistent_workers=args.num_workers > 0)
    it = infinite(loader)
    val_images = load_images(args.val_dir, limit=args.val_limit or None)
    val_snrs = [float(s) for s in args.val_snrs.split(",")]
    msssim_fn = MS_SSIM(data_range=1.0, levels=4, channel=3).to(device)

    # optimiser + schedule
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    min_ratio = args.min_lr / args.lr

    def lr_lambda(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        t = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # resume
    step, best_score, bad_vals = 0, -1e9, 0
    ckpt_path = os.path.join(args.out, "ckpt_last.pt")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step, best_score, bad_vals = ck["step"], ck["best"], ck.get("bad_vals", 0)
        rng = np.random.default_rng(args.seed + step)
        print("resumed from step %d (best %.3f)" % (step, best_score))
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=1)

    log_path = os.path.join(args.out, "log.csv")
    val_path = os.path.join(args.out, "val.csv")
    log_cols = ["step", "lr", "loss", "psnr", "snr", "step_time"]
    val_cols = ["step", "snr", "receiver", "psnr", "msssim"]
    new_log = step == 0 or not (os.path.exists(log_path) and os.path.exists(val_path))
    mode = "w" if new_log else "a"  # a fresh run (step 0) overwrites old logs; --resume appends
    flog = open(log_path, mode, newline="")
    wlog = csv.DictWriter(flog, fieldnames=log_cols)
    fval = open(val_path, mode, newline="")
    wval = csv.DictWriter(fval, fieldnames=val_cols)
    if new_log:
        wlog.writeheader()
        wval.writeheader()

    def save_ckpt():
        torch.save(dict(model=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step,
                        best=best_score, bad_vals=bad_vals, args=vars(args)), ckpt_path)
        torch.save(net.state_dict(), os.path.join(args.out, "last.model"))

    def do_validation():
        nonlocal best_score, bad_vals
        rows = validate(net, val_images, val_snrs, msssim_fn, device, amp, amp_dtype)
        for r in rows:
            wval.writerow(dict(step=step, **r))
        fval.flush()
        key = "uncoded_soft"
        score = float(np.mean([r["psnr"] for r in rows if r["receiver"] == key]))
        summary = " ".join("%s@%g=%.2f" % (r["receiver"][:8], r["snr"], r["psnr"]) for r in sorted(rows, key=lambda r: (r["receiver"], r["snr"])))
        improved = score > best_score
        if improved:
            best_score, bad_vals = score, 0
            torch.save(net.state_dict(), os.path.join(args.out, "best.model"))
        else:
            bad_vals += 1
        print("[val step %d] %s score(%s)=%.3f best=%.3f%s" % (step, summary, key, score, best_score, " *" if improved else ""), flush=True)
        return improved

    print("training C=%d from %s | %d images | steps=%d batch=%d lr=%g->%g warmup=%d | SNR U[%g,%g] awgn | amp=%s"
          % (args.C, init or "Pretrain default", len(ds), args.steps, args.batch, args.lr, args.min_lr, args.warmup,
             args.snr_min, args.snr_max, args.amp_dtype if amp else "off"), flush=True)
    if step == 0:
        do_validation()

    t_start = time.time()
    meter = defaultdict(float)
    n_meter = 0
    stop_reason = "done"
    while step < args.steps:
        t0 = time.time()
        set_resolution(net, args.crop, args.crop)  # validation may have switched the Swin masks to the val size
        x = next(it).to(device, non_blocking=True)
        snr = float(rng.uniform(args.snr_min, args.snr_max))

        _, _, bits = encode(net, x, snr, amp, amp_dtype)
        p1 = channel(bits, snr)
        recon = decode(net, p1, snr, amp, amp_dtype)
        loss = mse_loss_255(x, recon)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1

        meter["loss"] += loss.item()
        meter["psnr"] += psnr_from_mse(loss.item())
        meter["step_time"] += time.time() - t0
        n_meter += 1

        if step % args.log_every == 0:
            m = {k: v / n_meter for k, v in meter.items()}
            lr = opt.param_groups[0]["lr"]
            wlog.writerow(dict(step=step, lr=lr, loss=m["loss"], psnr=m["psnr"], snr=snr, step_time=m["step_time"]))
            flog.flush()
            eta = (args.steps - step) * m["step_time"] / 3600
            print("step %6d/%d lr %.2e loss %7.2f psnr %6.2f snr %4.1f %.2fs/it eta %.1fh" % (
                step, args.steps, lr, m["loss"], m["psnr"], snr, m["step_time"], eta), flush=True)
            meter.clear()
            n_meter = 0

        if step % args.val_every == 0 or step == args.steps:
            do_validation()
            save_ckpt()
            if args.patience and bad_vals >= args.patience:
                stop_reason = "early stop (%d validations without improvement)" % bad_vals
                break
        if args.max_hours and (time.time() - t_start) > args.max_hours * 3600:
            save_ckpt()
            stop_reason = "time budget reached at step %d; rerun with --resume" % step
            break

    save_ckpt()
    flog.close()
    fval.close()
    print("finished: %s | best val score %.3f | %s" % (stop_reason, best_score, args.out))


if __name__ == "__main__":
    main()
