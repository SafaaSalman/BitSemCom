"""Aggregate one or more results/*.jsonl files from run_grand.py into markdown tables.

Usage: python experiments/summarize.py results/C16_*.jsonl [more.jsonl ...]
"""
import json
import sys
from collections import defaultdict

import numpy as np


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            rows += [json.loads(l) for l in f if l.strip()]
    return rows


def table(rows, metric, fmt):
    paths = []
    for r in rows:
        if r["path"] not in paths:
            paths.append(r["path"])
    snrs = sorted({r["snr"] for r in rows})
    agg = defaultdict(list)
    for r in rows:
        agg[(r["snr"], r["path"])].append(r[metric])
    out = ["| Es/N0 (dB) | " + " | ".join(paths) + " |", "|---|" + "---|" * len(paths)]
    for s in snrs:
        out.append("| %.0f | " % s + " | ".join(fmt % np.mean(agg[(s, p)]) if agg[(s, p)] else "-" for p in paths) + " |")
    return "\n".join(out)


def grand_table(rows, path):
    snrs = sorted({r["snr"] for r in rows})
    sel = [r for r in rows if r["path"] == path]
    out = ["| Es/N0 | channel Es/N0 | BER channel | BER after GRAND | BLER channel | BLER after | abandoned | miscorrected | avg queries | symbols per payload bit |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for s in snrs:
        rr = [r for r in sel if r["snr"] == s]
        if not rr:
            continue
        m = lambda k: float(np.mean([r[k] for r in rr]))
        out.append("| %.0f | %.2f | %.4f | %.5f | %.3f | %.4f | %.3f | %.4f | %.0f | %.3f |" % (
            s, m("channel_snr"), m("ber_channel"), m("ber_after"), m("bler_channel"), m("bler_after"),
            m("abandon_rate"), m("miscorrect_rate"), m("avg_queries"), m("symbols") / (m("payload_bits") / 2)))
    return "\n".join(out)


def main():
    rows = load(sys.argv[1:])
    Cs = sorted({r["C"] for r in rows})
    for C in Cs:
        rc = [r for r in rows if r["C"] == C]
        n_img = len({r["image"] for r in rc})
        print("\n### C=%d  (%d images)\n" % (C, n_img))
        print("PSNR (dB), mean over images:\n")
        print(table(rc, "psnr", "%.2f"))
        print("\nMS-SSIM, mean over images:\n")
        print(table(rc, "msssim", "%.4f"))
        for p in ["coded_hard", "coded_hard_ebn0"]:
            if any(r["path"] == p for r in rc):
                print("\nGRAND statistics, path `%s`:\n" % p)
                print(grand_table(rc, p))


if __name__ == "__main__":
    main()
