"""
LLR shape study: what the semantic decoder (and GRAND) actually receive under different
channels, SNRs and CSI errors.

For every condition one figure (rows = SNR):
  col 1  channel LLR density split by the true bit (bit 0 / bit 1), with the theoretical
         Gaussian for AWGN (also for mismatched SNR estimate), BER and near-erasure fraction
  col 2  P(bit=1) seen by the semantic decoder with the plain soft receiver
  col 3  P(bit=1) after the coded link + ORBGRAND (decoded blocks exact, abandoned soft)
Plus `llr_summary.png` (BER, mean |LLR|, near-erasure fraction vs SNR for all conditions)
and `bitmapper.png` (confidence of the bit mapper itself: logits and P(1) before the channel).

  python experiments/plot_llr.py --C 32 --limit 4 --snrs 0,4,8,12 \
      --conditions awgn,awgn_over3,awgn_under3,fade8,fade8_h0.1,fade1 --out results/figures
"""
import argparse
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.bitsemcom_io import load_images, load_model  # noqa: E402
from experiments.channel_coding import CodedLink, make_code, qpsk_channel  # noqa: E402
from experiments.conditions import describe, parse_condition  # noqa: E402
from experiments.plot_style import INK2, INK3, color, finish, use_style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

ERASURE_T = 1.0  # |LLR| below this = "near erasure" (P(1) within 0.27..0.73)


@torch.no_grad()
def collect(net, images, snr, seed=0):
    """Semantic bits plus the bit mapper's own confidence, concatenated over images."""
    bits, logits1, p1_map = [], [], []
    for i, (_, img) in enumerate(images):
        img = img.to(next(net.parameters()).device)
        torch.manual_seed(seed + i)
        lg, y_soft, b = net.Encoder(img, float(snr))
        bits.append(b[0].cpu().numpy().astype(np.uint8).reshape(-1))
        logits1.append((lg[..., 1] - lg[..., 0])[0].cpu().numpy().reshape(-1))
        p1_map.append(y_soft[..., 1][0].cpu().numpy().reshape(-1))
    return bits, np.concatenate(logits1), np.concatenate(p1_map)


def gaussian_overlay(ax, x, snr_db, snr_err_db, weight, col):
    """LLR | bit ~ N(+-2 g_hat, 4 g_hat^2 / g) for AWGN with estimated SNR g_hat (g_hat=g if perfect)."""
    g = 10 ** (snr_db / 10)
    gh = 10 ** ((snr_db + snr_err_db) / 10)
    mu, var = 2 * gh, 4 * gh * gh / g
    for sign in (+1, -1):
        pdf = weight * np.exp(-(x - sign * mu) ** 2 / (2 * var)) / math.sqrt(2 * math.pi * var)
        ax.plot(x, pdf, color=col, lw=1.0, ls="--", alpha=0.9, label="theory" if sign > 0 else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--C", type=int, default=32)
    ap.add_argument("--ckpt", default="", help="model state_dict ('' = newest Pretrain/*C{C}_*.model)")
    ap.add_argument("--images", default="datasets/kodak_patches256")
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--snrs", default="0,4,8,12")
    ap.add_argument("--conditions", default="awgn,awgn_over3,awgn_under3,fade8,fade8_h0.1,fade1")
    ap.add_argument("--code", default="crc:0x9eb2:128:112")
    ap.add_argument("--budget", type=int, default=10 ** 4)
    ap.add_argument("--no_interleave", action="store_true")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    use_style()
    os.makedirs(args.out, exist_ok=True)
    snrs = [float(s) for s in args.snrs.split(",")]
    conds = [parse_condition(c) for c in args.conditions.split(",")]
    net = load_model(args.C, args.device, checkpoint=args.ckpt or None)
    images = load_images(args.images, limit=args.limit)
    G, H = make_code(args.code)
    link = CodedLink(G, H, max_queries=args.budget, budget_mode="level")
    tag = ("_" + args.tag) if args.tag else ""

    # bit mapper confidence (independent of the channel)
    bits_sets, logits1, p1_map = collect(net, images, snrs[len(snrs) // 2])
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
    axes[0].hist(logits1, bins=120, color=color(0), alpha=0.9)
    finish(axes[0], "logit(1) - logit(0)", "bits", "Bit mapper logits (C=%d, %d images)" % (args.C, len(images)), legend=False)
    axes[1].hist(p1_map, bins=100, color=color(0), alpha=0.9, log=True)
    conf = np.mean((p1_map < 0.01) | (p1_map > 0.99))
    finish(axes[1], "P(bit=1) from the bit mapper", "bits (log)", "%.0f%% of bits are within 0.01 of 0/1" % (100 * conf), legend=False)
    fig.savefig(os.path.join(args.out, "bitmapper%s.png" % tag))
    plt.close(fig)

    summary = {}
    for ci, cond in enumerate(conds):
        fig, axes = plt.subplots(len(snrs), 3, figsize=(11, 2.4 * len(snrs)), squeeze=False)
        for si, snr in enumerate(snrs):
            bits_all = np.concatenate(bits_sets)
            llr_u, p1_u, aft, lbl = [], [], [], []
            stats = []
            for bi, bits in enumerate(bits_sets):
                # same noise / fading for every condition of the same channel type (seed = image, snr only)
                rng = np.random.default_rng(10 * si + bi)
                l, p = qpsk_channel(bits, snr, rng, channel=cond["channel"], fade_block=cond["fade_block"],
                                    snr_err_db=cond["snr_err_db"], h_err_var=cond["h_err_var"])
                llr_u.append(l)
                p1_u.append(p)
                pg, st, det = link.transmit(bits, snr, np.random.default_rng(5000 + 10 * si + bi), fallback="soft",
                                            channel=cond["channel"], fade_block=cond["fade_block"], interleave=not args.no_interleave,
                                            snr_err_db=cond["snr_err_db"], h_err_var=cond["h_err_var"], return_detail=True)
                aft.append(pg)
                lbl.append(det["abandoned"])
                stats.append(st)
            llr = np.concatenate(llr_u)
            p1 = np.concatenate(p1_u)
            after = np.concatenate(aft)
            aband_bits = np.concatenate(lbl)
            ber = float(((llr < 0) != bits_all).mean())
            erasure = float((np.abs(llr) < ERASURE_T).mean())
            m = lambda k: float(np.mean([s[k] for s in stats]))
            summary.setdefault(cond["name"], []).append(dict(snr=snr, ber=ber, erasure=erasure, mean_abs_llr=float(np.abs(llr).mean()),
                                                              bler=m("bler_after"), aband=m("abandon_rate"), misc=m("miscorrect_rate"),
                                                              ber_after=m("ber_after"), queries=m("avg_queries")))

            # col 1: LLR density by true bit
            ax = axes[si, 0]
            lim = np.percentile(np.abs(llr), 99.5)
            lim = max(lim, 4.0)
            edges = np.linspace(-lim, lim, 121)
            for b, lab, col in ((0, "true bit 0", color(0)), (1, "true bit 1", color(1))):
                sel = bits_all == b
                ax.hist(np.clip(llr[sel], -lim, lim), bins=edges, density=False, weights=np.full(sel.sum(), 1.0 / llr.size),
                        histtype="stepfilled", alpha=0.45, color=col, label=lab)
            if cond["channel"] == "awgn":
                x = np.linspace(-lim, lim, 400)
                w = (edges[1] - edges[0])
                frac0, frac1 = (bits_all == 0).mean(), (bits_all == 1).mean()
                gaussian_overlay(ax, x, snr, cond["snr_err_db"], w * max(frac0, frac1), INK2)
            ax.axvspan(-ERASURE_T, ERASURE_T, color=INK3, alpha=0.12, lw=0)
            ax.text(0.02, 0.95, "BER %.3f\n|LLR|<%g: %.1f%%\nmean|LLR| %.1f" % (ber, ERASURE_T, 100 * erasure, np.abs(llr).mean()),
                    transform=ax.transAxes, va="top", fontsize=7.5, color=INK2)
            finish(ax, "LLR (positive = bit 0)", "fraction of bits", "Es/N0 = %g dB: channel LLR" % snr, legend=si == 0)

            # col 2: P(1) to the decoder, uncoded soft receiver
            ax = axes[si, 1]
            ax.hist(p1, bins=100, range=(0, 1), color=color(2), alpha=0.9, log=True, weights=np.full(p1.size, 1.0 / p1.size))
            mid = float(((p1 > 0.1) & (p1 < 0.9)).mean())
            ax.text(0.5, 0.95, "%.1f%% in (0.1, 0.9)" % (100 * mid), transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color=INK2)
            ax.set_ylim(1e-5, 1.0)
            finish(ax, "P(bit=1) at the decoder input", "fraction (log)", "uncoded soft receiver", legend=False)

            # col 3: after GRAND
            ax = axes[si, 2]
            ax.hist(after[~aband_bits], bins=100, range=(0, 1), color=color(0), alpha=0.9, log=True,
                    weights=np.full((~aband_bits).sum(), 1.0 / after.size), label="decoded blocks")
            if aband_bits.any():
                ax.hist(after[aband_bits], bins=100, range=(0, 1), color=color(1), alpha=0.7, log=True,
                        weights=np.full(aband_bits.sum(), 1.0 / after.size), label="abandoned (soft fallback)")
            ax.text(0.5, 0.95, "BLER %.3f  aband %.2f  miscorr %.3f\nBER %.4f -> %.4f, avg %.0f queries" % (
                m("bler_after"), m("abandon_rate"), m("miscorrect_rate"), m("ber_channel"), m("ber_after"), m("avg_queries")),
                transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color=INK2)
            ax.set_ylim(1e-5, 1.0)
            finish(ax, "P(bit=1) at the decoder input", "fraction (log)", "after (%d,%d) code + ORBGRAND" % (link.n, link.k), legend=si == 0)
        fig.suptitle("%s  -  C=%d, %d Kodak patches, %d bits" % (describe(cond), args.C, len(images), bits_all.size), x=0.01, ha="left", fontsize=10)
        fig.tight_layout()
        path = os.path.join(args.out, "llr_%s%s.png" % (cond["name"], tag))
        fig.savefig(path)
        plt.close(fig)
        print("wrote", path)

    # summary across conditions
    fig, axes = plt.subplots(1, 4, figsize=(13, 2.9))
    keys = [("ber", "raw BER (log)", True), ("erasure", "fraction |LLR| < %g" % ERASURE_T, False),
            ("bler", "BLER after ORBGRAND (log)", True), ("miscorrect", "miscorrection rate", False)]
    for ci, (name, rows) in enumerate(summary.items()):
        x = [r["snr"] for r in rows]
        for ax, (k, _, _) in zip(axes, keys):
            kk = "misc" if k == "miscorrect" else k
            ax.plot(x, [max(r[kk], 1e-5) for r in rows], marker="o", color=color(ci), label=name)
    for ax, (k, ylab, log) in zip(axes, keys):
        if log:
            ax.set_yscale("log")
        finish(ax, "Es/N0 (dB)", ylab, legend=ax is axes[0])
    fig.suptitle("LLR summary across conditions (C=%d)" % args.C, x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "llr_summary%s.png" % tag))
    plt.close(fig)
    with open(os.path.join(args.out, "llr_summary%s.md" % tag), "w") as f:
        f.write("| condition | Es/N0 | BER | mean abs LLR | frac abs LLR<%g | BLER after | abandoned | miscorrected | BER after | avg queries |\n" % ERASURE_T)
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for name, rows in summary.items():
            for r in rows:
                f.write("| %s | %g | %.4f | %.2f | %.3f | %.4f | %.3f | %.4f | %.5f | %.0f |\n" % (
                    name, r["snr"], r["ber"], r["mean_abs_llr"], r["erasure"], r["bler"], r["aband"], r["misc"], r["ber_after"], r["queries"]))
    print("wrote", os.path.join(args.out, "llr_summary%s.png/.md" % tag))


if __name__ == "__main__":
    main()
