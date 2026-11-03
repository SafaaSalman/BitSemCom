"""
Paired-gain charts for docs/grand_review.md: PSNR differences drawn as bars about zero, whiskers = +-1 s.e.
over the 24 Kodak patches. Every bar compares the same image, the same bits and the same channel noise.

  review_gain_awgn.png       bench model, AWGN. Left: the room up to the perfect-bit ceiling (grey) next to
                             what CRC(128,112)+ORBGRAND recovered. Right: zoom on the interleaver-only and
                             the coded gains over the plain soft receiver.
  review_gain_fade8.png      the same for block Rayleigh fading (one fade per 8 symbols)
  review_grand_vs_bench.png  GRAND-trained model minus bench model on AWGN, one bar per receiver
  review_csi.png             cost of imperfect CSI per receiver: each condition minus its perfect-CSI twin
  review_budget.png          coding gain at the training budget (2000) vs 100,000 guesses (needs --b1e5)

  python experiments/plot_review_gains.py results/eval_C32.jsonl --b1e5 results/eval_C32_b1e5.jsonl \
      --images results/images_C32 --out docs/figures/grand_review
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.plot_style import INK2, INK3, PALETTE, use_style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

GREY = "#c3c2b7"  # the ceiling / reference, deliberately recessive
BLUE, ORANGE, AQUA, YELLOW, VIOLET = PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3], PALETTE[6]
COND_LABEL = {"awgn": "AWGN, perfect CSI", "fade8": "block Rayleigh fading (8 symbols), perfect CSI",
              "fade8_h0.1": "block Rayleigh fading, channel-estimate error var 0.1",
              "awgn_over3": "AWGN, receiver assumes SNR 3 dB too high", "awgn_under3": "AWGN, receiver assumes SNR 3 dB too low"}


def load(path):
    with open(path) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    idx = defaultdict(dict)  # (model, condition, snr, path) -> {image: psnr}
    for r in rows:
        idx[(r["model"], r["condition"], r["snr"], r["path"])][r["image"]] = r["psnr"]
    return idx


def paired(idx, a, b):
    """mean and standard error of PSNR(a) - PSNR(b) over the images both keys share"""
    A, B = idx[a], idx[b]
    d = np.array([A[i] - B[i] for i in sorted(set(A) & set(B))])
    if len(d) == 0:
        return np.nan, np.nan
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0


def series(idx, model, cond, snrs, a_path, b_path, b_model=None):
    vals = [paired(idx, (model, cond, s, a_path), (b_model or model, cond, s, b_path)) for s in snrs]
    return np.array([v[0] for v in vals]), np.array([v[1] for v in vals])


def bars(ax, snrs, groups, width, ylim, label=(), label_min=0.0):
    """grouped bars about zero. groups: list of (name, colour, means, s.e.)"""
    n = len(groups)
    x = np.arange(len(snrs))
    pad = 0.015 * (ylim[1] - ylim[0])
    for j, (name, col, m, se) in enumerate(groups):
        off = (j - (n - 1) / 2) * width
        ax.bar(x + off, m, width * 0.9, color=col, label=name, zorder=3, linewidth=0)
        ax.errorbar(x + off, m, yerr=se, fmt="none", ecolor=INK3, elinewidth=0.8, capsize=1.5, zorder=4)
        if name in label:
            for xi, mi, si in zip(x + off, m, se):
                if abs(mi) < label_min:
                    continue
                y = mi + (si + pad) if mi >= 0 else mi - (si + pad)
                ax.text(xi, y, "%+.2f" % mi, ha="center", va="bottom" if mi >= 0 else "top", fontsize=7, color=INK2, zorder=5)
    ax.axhline(0, color=INK3, linewidth=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(["%g" % s for s in snrs])
    ax.set_ylim(*ylim)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Es/N0 (dB)")


def gain_chart(idx, model, cond, out, budget):
    snrs = sorted({k[2] for k in idx if k[0] == model and k[1] == cond})
    room = series(idx, model, cond, snrs, "noiseless", "uncoded_soft")
    il = series(idx, model, cond, snrs, "uncoded_soft_il", "uncoded_soft")
    coded = series(idx, model, cond, snrs, "coded_soft", "uncoded_soft")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.7))
    name_room, name_il, name_coded = "room up to perfect bits", "interleaver only, no code", "interleaver + CRC(128,112) + ORBGRAND"
    top = float(np.nanmax(room[0])) * 1.2
    bars(a1, snrs, [(name_room, GREY, *room), (name_coded, ORANGE, *coded)], 0.36,
         (min(-0.06, float(np.nanmin(coded[0])) * 1.6), top), label=(name_room,), label_min=0.1)
    a1.set_title("what a perfect channel would add, and what the code added")
    a1.set_ylabel("PSNR gain over the plain soft receiver (dB)")
    a1.legend(loc="upper right")
    vals = np.concatenate([il[0], coded[0]])
    span = float(np.nanmax(np.abs(vals)))
    lo = min(0.0, float(np.nanmin(vals))) - 0.35 * span - 0.02
    hi = max(0.0, float(np.nanmax(vals))) + 0.45 * span + 0.02
    bars(a2, snrs, [(name_il, BLUE, *il), (name_coded, ORANGE, *coded)], 0.36, (lo, hi), label=(name_coded,))
    a2.set_title("zoom: the two receivers that share the interleaver")
    a2.legend(loc="upper left" if cond.startswith("fade") else "upper right")
    fig.text(0.0, 1.0, "%s model, %s, ORBGRAND budget %s guesses, 24 Kodak patches, whiskers = ±1 s.e. of the paired mean"
             % (model, COND_LABEL.get(cond, cond), budget), ha="left", va="bottom", fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "review_gain_%s.png" % cond))
    plt.close(fig)


def model_chart(idx, cond, out, model="grand", ref="bench"):
    snrs = sorted({k[2] for k in idx if k[0] == model and k[1] == cond})
    receivers = [("noiseless", "perfect bits", GREY), ("uncoded_soft", "plain soft", AQUA), ("uncoded_hard", "hard 0/1", YELLOW),
                 ("uncoded_soft_il", "soft + interleaver", BLUE), ("coded_soft", "interleaver + CRC + ORBGRAND", ORANGE)]
    groups = [(name, col, *series(idx, model, cond, snrs, p, p, b_model=ref)) for p, name, col in receivers]
    vals = np.concatenate([g[2] for g in groups])
    span = float(np.nanmax(np.abs(vals)))
    lo = min(0.0, float(np.nanmin(vals))) - 0.3 * span - 0.02
    hi = max(0.0, float(np.nanmax(vals))) + 0.25 * span + 0.02
    fig, ax = plt.subplots(figsize=(10.5, 3.7))
    bars(ax, snrs, groups, 0.16, (lo, hi), label=("perfect bits", "interleaver + CRC + ORBGRAND"))
    ax.set_title("%s-trained model minus %s model, same receiver, same image, same channel noise" % (model, ref))
    ax.set_ylabel("PSNR difference, %s minus %s (dB)" % (model, ref))
    ax.legend(loc="lower right", ncol=5)
    fig.text(0.0, 1.0, "%s, 24 Kodak patches, whiskers = ±1 s.e. of the paired mean; below zero = the %s-trained model is worse"
             % (COND_LABEL.get(cond, cond), model), ha="left", va="bottom", fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "review_%s_vs_%s.png" % (model, ref)))
    plt.close(fig)


def csi_chart(idx, model, out):
    panels = [("awgn_over3", "awgn", "AWGN, receiver assumes SNR 3 dB too high"),
              ("awgn_under3", "awgn", "AWGN, receiver assumes SNR 3 dB too low"),
              ("fade8_h0.1", "fade8", "fading, channel-estimate error var 0.1")]
    receivers = [("uncoded_soft", "plain soft", AQUA), ("uncoded_soft_il", "soft + interleaver", BLUE),
                 ("coded_soft", "interleaver + CRC + ORBGRAND", ORANGE)]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7), sharey=True)
    for ax, (cond, ref, title) in zip(axes, panels):
        snrs = sorted({k[2] for k in idx if k[0] == model and k[1] == cond})
        groups = []
        for p, name, col in receivers:
            vals = [paired(idx, (model, cond, s, p), (model, ref, s, p)) for s in snrs]
            groups.append((name, col, np.array([v[0] for v in vals]), np.array([v[1] for v in vals])))
        bars(ax, snrs, groups, 0.27, (-0.95, 0.12), label=("plain soft", "interleaver + CRC + ORBGRAND"), label_min=0.1)
        ax.set_title(title)
    axes[0].set_ylabel("PSNR change against perfect CSI (dB)")
    axes[0].legend(loc="lower right")
    fig.text(0.0, 1.0, "%s model, each receiver under the CSI error minus the same receiver with perfect CSI, 24 Kodak patches, "
             "whiskers = ±1 s.e.; below zero = the error costs quality" % model, ha="left", va="bottom", fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "review_csi.png"))
    plt.close(fig)


def budget_chart(idx, idx_big, model, out, conds=("awgn", "fade8")):
    fig, axes = plt.subplots(1, len(conds), figsize=(10.5, 3.7))
    n1, n2 = "budget 2000 guesses (the training budget)", "budget 100,000 guesses"
    for ax, cond in zip(np.atleast_1d(axes), conds):
        snrs = sorted({k[2] for k in idx if k[0] == model and k[1] == cond})
        small = series(idx, model, cond, snrs, "coded_soft", "uncoded_soft_il")
        big = series(idx_big, model, cond, snrs, "coded_soft", "uncoded_soft_il")
        vals = np.concatenate([small[0], big[0]])
        span = float(np.nanmax(np.abs(vals)))
        lo = min(0.0, float(np.nanmin(vals))) - 0.25 * span - 0.02
        hi = max(0.0, float(np.nanmax(vals))) + 0.4 * span + 0.02
        bars(ax, snrs, [(n1, ORANGE, *small), (n2, VIOLET, *big)], 0.36, (lo, hi), label=(n2,), label_min=0.1)
        ax.set_title(COND_LABEL.get(cond, cond))
        ax.legend(loc="lower right")
    np.atleast_1d(axes)[0].set_ylabel("PSNR gain of the code over interleaver-only (dB)")
    fig.text(0.0, 1.0, "%s model, coded receiver minus the same receiver without the code, 24 Kodak patches, whiskers = ±1 s.e."
             % model, ha="left", va="bottom", fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "review_budget.png"))
    plt.close(fig)


def example_chart(idx, images_dir, out, image="kodim01", model="bench", cond="fade8", snr=4.0):
    """one reconstructed patch per receiver, from evaluate.py --save_images, with its PSNR"""
    import matplotlib.image as mpimg
    panels = [("noiseless", "perfect bits"), ("uncoded_soft", "plain soft"), ("uncoded_soft_il", "soft + interleaver"),
              ("coded_soft", "interleaver + CRC + ORBGRAND"), ("uncoded_hard", "hard 0/1")]
    fig, axes = plt.subplots(1, len(panels), figsize=(12.5, 3.1))
    for ax, (p, name) in zip(axes, panels):
        f = os.path.join(images_dir, "%s_%s_%s_%gdB_%s.png" % (image, model, cond, snr, p))
        ax.imshow(mpimg.imread(f))
        ax.set_axis_off()
        d = idx.get((model, cond, snr, p), {})
        key = next((k for k in d if str(k).startswith(image)), None)
        ax.set_title("%s\n%.2f dB" % (name, d[key]) if key is not None else name, fontsize=8.5, loc="center")
    fig.text(0.0, 1.0, "%s, %s model, %s, Es/N0 = %g dB; the same bits and the same channel noise for every receiver"
             % (image, model, COND_LABEL.get(cond, cond), snr), ha="left", va="bottom", fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "review_example_%s_%s_%gdB.png" % (image, cond, snr)))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl")
    ap.add_argument("--b1e5", default=None, help="JSONL of the large-budget sweep, for review_budget.png")
    ap.add_argument("--images", default=None, help="folder written by evaluate.py --save_images, for the example strip")
    ap.add_argument("--out", default="docs/figures/grand_review")
    ap.add_argument("--model", default="bench", help="model for the coding-gain charts")
    ap.add_argument("--conds", default="awgn,fade8", help="conditions for the coding-gain charts")
    ap.add_argument("--budget", default="2000", help="query budget, for the caption only")
    args = ap.parse_args()
    use_style()
    os.makedirs(args.out, exist_ok=True)
    idx = load(args.jsonl)
    for cond in args.conds.split(","):
        gain_chart(idx, args.model, cond, args.out, args.budget)
    if any(k[0] == "grand" for k in idx) and any(k[0] == "bench" for k in idx):
        model_chart(idx, "awgn", args.out)
    if all(any(k[1] == c for k in idx) for c in ("awgn_over3", "awgn_under3", "fade8_h0.1")):
        csi_chart(idx, args.model, args.out)
    if args.b1e5:
        budget_chart(idx, load(args.b1e5), args.model, args.out)
    if args.images:
        for cond in ("awgn", "fade8"):
            example_chart(idx, args.images, args.out, model=args.model, cond=cond)
    print("wrote", sorted(f for f in os.listdir(args.out) if f.startswith("review_")))


if __name__ == "__main__":
    main()
