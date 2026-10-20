"""
Per-channel importance of the semantic bitstream (bits are (tokens, C)).

For each channel c: flip a fraction of that channel's bits at random and measure the
PSNR drop; also report the channel's mean bit-mapper confidence and the share of the
globally most-confident 2% of bits that live in it.

Usage: python experiments/channel_importance.py --C 16 --limit 24
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.bitsemcom_io import Metrics, decode_probs, load_images, load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=int, default=16)
    ap.add_argument("--images", default="datasets/kodak_patches256")
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--snr", type=float, default=6.0)
    ap.add_argument("--flip", default="0.1,0.5")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    flips = [float(f) for f in args.flip.split(",")]
    net = load_model(args.C, args.device)
    met = Metrics(torch.device(args.device))
    images = load_images(args.images, limit=args.limit)
    C = args.C

    drop = {f: np.zeros(C) for f in flips}
    conf_mean = np.zeros(C)
    top_share = np.zeros(C)
    ones_frac = np.zeros(C)
    base_all = []
    for ii, (name, img) in enumerate(images):
        img = img.to(args.device)
        with torch.no_grad():
            torch.manual_seed(1000 + ii)
            logits, _, bits = net.Encoder(img, args.snr)  # (1, T, C)
            p1 = torch.softmax(logits, dim=-1)[..., 1][0]  # (T, C)
            conf = (p1 - 0.5).abs()
            T = bits.shape[1]
            base = met(img, decode_probs(net, bits, args.snr))[0]
            base_all.append(base)
            conf_mean += conf.mean(0).cpu().numpy()
            ones_frac += bits[0].mean(0).cpu().numpy()
            m = int(round(0.02 * T * C))
            top_idx = torch.argsort(conf.reshape(-1), descending=True)[:m]
            top_ch = (top_idx % C).cpu().numpy()
            top_share += np.bincount(top_ch, minlength=C) / m
            rng = np.random.default_rng(ii)
            for f in flips:
                k = int(round(f * T))
                for c in range(C):
                    b = bits.clone()
                    rows = torch.from_numpy(rng.choice(T, k, replace=False)).to(b.device)
                    b[0, rows, c] = 1.0 - b[0, rows, c]
                    drop[f][c] += base - met(img, decode_probs(net, b, args.snr))[0]
        print("[%2d/%2d] %s" % (ii + 1, len(images), name), flush=True)

    n = len(images)
    conf_mean /= n
    top_share /= n
    ones_frac /= n
    for f in flips:
        drop[f] /= n
    print("\nC=%d, %d images, noiseless PSNR %.2f dB. Per-channel PSNR drop when flipping a fraction of that channel's bits:\n" % (C, n, np.mean(base_all)))
    print("| channel | mean confidence | share of top-2%% confident bits | P(bit=1) | " + " | ".join("drop @%d%% flips (dB)" % (100 * f) for f in flips) + " |")
    print("|---|---|---|---|" + "---|" * len(flips))
    order = np.argsort(-drop[flips[0]])
    for c in order:
        print("| %d | %.3f | %.1f%% | %.2f | " % (c, conf_mean[c], 100 * top_share[c], ones_frac[c]) + " | ".join("%.2f" % drop[f][c] for f in flips) + " |")
    print("\nSum of per-channel drops @%d%%: %.2f dB; max %.2f, median %.2f, min %.2f" % (
        100 * flips[0], drop[flips[0]].sum(), drop[flips[0]].max(), np.median(drop[flips[0]]), drop[flips[0]].min()))
    r = np.corrcoef(conf_mean, drop[flips[0]])[0, 1]
    print("Correlation(mean confidence, drop @%d%%) = %.2f" % (100 * flips[0], r))


if __name__ == "__main__":
    main()
