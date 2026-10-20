"""
Which semantic bits matter? Inject a fixed fraction of bit flips into the bit mapper's
output, choosing the flipped positions (a) uniformly at random, (b) among the bits the
encoder is MOST confident about, (c) among the bits it is LEAST confident about, and
measure the decoder's PSNR. Also measures the effect of simply re-sampling the Gumbel
bit mapper with another seed (its intrinsic randomness).

Usage: python experiments/bit_importance.py --C 16 --limit 24
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
    ap.add_argument("--rates", default="0.01,0.02,0.05,0.10")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    rates = [float(r) for r in args.rates.split(",")]
    net = load_model(args.C, args.device)
    met = Metrics(torch.device(args.device))
    images = load_images(args.images, limit=args.limit)

    modes = ("random", "confident", "uncertain", "tokens")
    keys = ["noiseless", "resampled"] + ["%s@%.2f" % (m, r) for r in rates for m in modes]
    acc = {k: [] for k in keys}
    conf_stats = []
    for ii, (name, img) in enumerate(images):
        img = img.to(args.device)
        with torch.no_grad():
            torch.manual_seed(1000 + ii)
            logits, _, bits = net.Encoder(img, args.snr)
            p1 = torch.softmax(logits, dim=-1)[..., 1]
            conf = (p1 - 0.5).abs()  # 0 = coin flip, 0.5 = certain
            conf_stats.append([(conf > 0.49).float().mean().item(), (conf < 0.4).float().mean().item()])
            torch.manual_seed(5000 + ii)
            _, _, bits2 = net.Encoder(img, args.snr)
            acc["noiseless"].append(met(img, decode_probs(net, bits, args.snr))[0])
            acc["resampled"].append(met(img, decode_probs(net, bits2, args.snr))[0])
            flat = bits.reshape(-1)
            cflat = conf.reshape(-1)
            n = flat.numel()
            T, Cc = bits.shape[1], bits.shape[2]
            rng = np.random.default_rng(ii)
            # break ties (saturated probabilities) at random so "most confident" is not "first in memory"
            jitter = torch.from_numpy(rng.uniform(0, 1e-6, n).astype(np.float32)).to(cflat.device)
            order_conf = torch.argsort(cflat + jitter, descending=True).cpu().numpy()
            for r in rates:
                m = int(round(r * n))
                ntok = max(1, int(round(r * T)))
                tok = rng.choice(T, ntok, replace=False)
                sel = {
                    "random": rng.choice(n, m, replace=False),
                    "confident": order_conf[:m],
                    "uncertain": order_conf[::-1][:m].copy(),
                    "tokens": (tok[:, None] * Cc + np.arange(Cc)[None, :]).reshape(-1),
                }
                for mode, idx in sel.items():
                    b = flat.clone()
                    idx_t = torch.from_numpy(np.asarray(idx)).to(b.device)
                    b[idx_t] = 1.0 - b[idx_t]
                    acc["%s@%.2f" % (mode, r)].append(met(img, decode_probs(net, b.reshape(bits.shape), args.snr))[0])
        print("[%2d/%2d] %s done" % (ii + 1, len(images), name), flush=True)

    cs = np.mean(conf_stats, axis=0)
    print("\nC=%d, %d images. Bit-mapper confidence: %.1f%% of bits have |P(1)-0.5|>0.49, %.1f%% have |P(1)-0.5|<0.4"
          % (args.C, len(images), 100 * cs[0], 100 * cs[1]))
    print("Resampling the Gumbel bit mapper with another seed flips ~5%% of bits.\n")
    print("| flipped fraction | random positions | most-confident bits (ties random) | least-confident bits | whole tokens (all C bits) |")
    print("|---|---|---|---|---|")
    base = np.mean(acc["noiseless"])
    print("| 0 (noiseless) | %.2f | %.2f | %.2f | %.2f |" % (base, base, base, base))
    print("| resampled bits (seed) | %.2f | - | - | - |" % np.mean(acc["resampled"]))
    for r in rates:
        print("| %.0f%% | %.2f | %.2f | %.2f | %.2f |" % (100 * r, np.mean(acc["random@%.2f" % r]),
                                                       np.mean(acc["confident@%.2f" % r]), np.mean(acc["uncertain@%.2f" % r]),
                                                       np.mean(acc["tokens@%.2f" % r])))


if __name__ == "__main__":
    main()
