"""
Training curves from one or more train_bitsemcom.py run folders.

  python experiments/plot_training.py runs/C32_bench
  python experiments/plot_training.py runs/C32_a runs/C32_b --out figures

Writes training.png (into the first run folder unless --out is given): (a) validation PSNR
vs step, mean over the validation SNRs with the per-SNR curves faint, (b) smoothed training
loss, (c) learning rate.
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def read_csv(path):
    with open(path) as f:
        return [{k: v if k == "receiver" else float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def ema(x, alpha=0.05):
    out, m = [], None
    for v in x:
        m = v if m is None else (1 - alpha) * m + alpha * v
        out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="", help="output folder ('' = the first run folder)")
    ap.add_argument("--receiver", default="uncoded_soft", choices=["uncoded_soft", "noiseless"],
                    help="which validation result to plot")
    args = ap.parse_args()
    out = args.out or args.runs[0]
    os.makedirs(out, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e5e5e2", "grid.linewidth": 0.6, "legend.frameon": False, "savefig.dpi": 200,
                         "savefig.bbox": "tight"})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.1))
    for ri, run in enumerate(args.runs):
        c = COLORS[ri % len(COLORS)]
        name = os.path.basename(os.path.normpath(run))
        log = read_csv(os.path.join(run, "log.csv"))
        val = read_csv(os.path.join(run, "val.csv"))
        by_step = defaultdict(list)
        per_snr = defaultdict(list)
        for r in val:
            if r["receiver"] == args.receiver:
                by_step[r["step"]].append(r["psnr"])
                per_snr[r["snr"]].append((r["step"], r["psnr"]))
        steps = sorted(by_step)
        for snr, pts in sorted(per_snr.items()):
            pts.sort()
            axes[0].plot([p[0] for p in pts], [p[1] for p in pts], color=c, lw=0.7, alpha=0.25)
        axes[0].plot(steps, [np.mean(by_step[s]) for s in steps], color=c, marker="o", ms=3, label=name)
        axes[1].plot([r["step"] for r in log], ema([r["loss"] for r in log]), color=c, label=name)
        axes[2].plot([r["step"] for r in log], [r["lr"] for r in log], color=c, label=name)
    axes[0].set(xlabel="step", ylabel="validation PSNR (dB), mean over SNRs", title="validation, %s (faint: per SNR)" % args.receiver)
    axes[0].legend(loc="best")
    axes[1].set(xlabel="step", ylabel="training MSE loss (EMA)", title="training loss")
    axes[2].set(xlabel="step", ylabel="learning rate (log)", title="schedule", yscale="log")
    fig.tight_layout()
    path = os.path.join(out, "training.png")
    fig.savefig(path)
    print("wrote", path)


if __name__ == "__main__":
    main()
