"""
Training curves from one or more train_bitsemcom.py run folders.

  python experiments/plot_training.py runs/C32_bench runs/C32_grand --out results/figures

training.png: (a) validation PSNR of each run's own receiver vs step (mean over validation
SNRs, with the per-SNR curves faint), (b) smoothed training loss, (c) learning rate,
(d) GRAND block outcomes during training (if any run used GRAND).
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.plot_style import color, finish, use_style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402


def read_csv(path):
    with open(path) as f:
        return [{k: float(v) if k not in ("receiver",) else v for k, v in r.items()} for r in csv.DictReader(f)]


def ema(x, alpha=0.05):
    out, m = [], None
    for v in x:
        m = v if m is None else (1 - alpha) * m + alpha * v
        out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--receiver", default="", help="validation receiver to plot ('' = each run's own: grand if present else uncoded_soft)")
    args = ap.parse_args()
    use_style()
    os.makedirs(args.out, exist_ok=True)
    any_grand = False
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.1))
    for ri, run in enumerate(args.runs):
        name = os.path.basename(os.path.normpath(run))
        log = read_csv(os.path.join(run, "log.csv"))
        val = read_csv(os.path.join(run, "val.csv"))
        receivers = {r["receiver"] for r in val}
        rec = args.receiver or ("grand" if "grand" in receivers else "uncoded_soft")
        by_step = defaultdict(list)
        per_snr = defaultdict(list)
        for r in val:
            if r["receiver"] == rec:
                by_step[r["step"]].append(r["psnr"])
                per_snr[r["snr"]].append((r["step"], r["psnr"]))
        steps = sorted(by_step)
        for snr, pts in sorted(per_snr.items()):
            pts.sort()
            axes[0].plot([p[0] for p in pts], [p[1] for p in pts], color=color(ri), lw=0.7, alpha=0.25)
        axes[0].plot(steps, [np.mean(by_step[s]) for s in steps], color=color(ri), marker="o", ms=3, label="%s (%s)" % (name, rec))
        axes[1].plot([r["step"] for r in log], ema([r["loss"] for r in log]), color=color(ri), label=name)
        axes[2].plot([r["step"] for r in log], [r["lr"] for r in log], color=color(ri), label=name)
        if any(r.get("grand_queries", 0) > 0 for r in log):
            any_grand = True
            axes[3].plot([r["step"] for r in log], ema([r["grand_abandon"] for r in log]), color=color(ri), ls="-", label="%s abandoned" % name)
            axes[3].plot([r["step"] for r in log], ema([r["grand_miscorrect"] for r in log]), color=color(ri), ls="--", label="%s miscorrected" % name)
    finish(axes[0], "step", "validation PSNR (dB), mean over SNRs", "validation (faint: per SNR)")
    finish(axes[1], "step", "training MSE loss (EMA)", "training loss", legend=False)
    axes[2].set_yscale("log")
    finish(axes[2], "step", "learning rate (log)", "schedule", legend=False)
    if any_grand:
        finish(axes[3], "step", "rate (EMA)", "GRAND during training")
    else:
        axes[3].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "training.png"))
    print("wrote", os.path.join(args.out, "training.png"))


if __name__ == "__main__":
    main()
