"""
Figures from evaluate.py output (results/*.jsonl):

  quality_<condition>.png   metric vs Es/N0, one panel per metric; colour = model (fixed order),
                            line style + marker = receiver; noiseless ceiling as a thin line
  gain_<metric>.png         coded_soft minus uncoded_soft per condition (does GRAND help?), colour = model
  grand_<condition>.png     BLER after decoding / abandoned / miscorrected (log) and average queries vs Es/N0
  summary.md                tables (mean over images) for every model x condition x path

  python experiments/plot_results.py results/eval_C32.jsonl --out results/figures
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.conditions import describe, parse_condition  # noqa: E402
from experiments.plot_style import INK3, color, finish, use_style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

PATH_STYLE = {  # receiver -> (linestyle, marker, label)
    "uncoded_soft": ("-", "o", "uncoded soft (BitSemCom)"),
    "coded_soft": ("--", "s", "coded + ORBGRAND (soft fallback)"),
    "uncoded_hard": (":", "^", "uncoded hard"),
    "uncoded_soft_il": ("-", "d", "uncoded soft + interleaver"),
    "uncoded_hard_il": (":", "<", "uncoded hard + interleaver"),
    "coded_hard": ("-.", "v", "coded + ORBGRAND (hard fallback)"),
    "coded_soft_noil": ("--", "D", "coded, no interleaver"),
    "coded_soft_ebn0": ("--", "P", "coded, equal Eb/N0"),
    "coded_hard_ebn0": ("-.", "X", "coded hard, equal Eb/N0"),
}
METRIC_LABEL = {"psnr": "PSNR (dB)", "msssim": "MS-SSIM", "lpips": "LPIPS (lower is better)", "clip": "CLIP similarity"}


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            rows += [json.loads(l) for l in f if l.strip()]
    return rows


def mean_table(rows, key):
    agg = defaultdict(list)
    for r in rows:
        if key in r and r[key] is not None:
            agg[(r["model"], r["condition"], r["snr"], r["path"])].append(r[key])
    return {k: float(np.mean(v)) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--metrics", default="psnr,lpips,msssim,clip")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    use_style()
    os.makedirs(args.out, exist_ok=True)
    rows = load(args.jsonl)
    models = list(dict.fromkeys(r["model"] for r in rows))
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    paths = list(dict.fromkeys(r["path"] for r in rows))
    snrs = sorted({r["snr"] for r in rows})
    metrics = [m for m in args.metrics.split(",") if any(m in r for r in rows)]
    tables = {m: mean_table(rows, m) for m in metrics}
    tag = ("_" + args.tag) if args.tag else ""
    print("models %s | conditions %s | paths %s | metrics %s" % (models, conds, paths, metrics))

    # quality vs SNR per condition
    for cond in conds:
        fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.2), squeeze=False)
        for mi, metric in enumerate(metrics):
            ax = axes[0, mi]
            t = tables[metric]
            for ci, model in enumerate(models):
                if "noiseless" in paths:
                    y = [t.get((model, cond, s, "noiseless"), np.nan) for s in snrs]
                    ax.plot(snrs, y, color=color(ci), lw=0.9, alpha=0.5, label="%s noiseless" % model if mi == 0 else None)
                for path in paths:
                    if path not in PATH_STYLE:
                        continue
                    ls, mk, lab = PATH_STYLE[path]
                    y = [t.get((model, cond, s, path), np.nan) for s in snrs]
                    ax.plot(snrs, y, color=color(ci), ls=ls, marker=mk, label="%s: %s" % (model, lab) if mi == 0 else None)
            finish(ax, "Es/N0 (dB)", METRIC_LABEL.get(metric, metric), legend=False)
        axes[0, 0].legend(loc="lower right", fontsize=7)
        fig.suptitle(describe(parse_condition(cond)), x=0.01, ha="left", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "quality_%s%s.png" % (cond, tag)))
        plt.close(fig)

    # coding gain per condition (coded_soft - uncoded_soft)
    if "coded_soft" in paths and "uncoded_soft" in paths:
        for metric in metrics:
            t = tables[metric]
            fig, axes = plt.subplots(1, len(conds), figsize=(3.4 * len(conds), 3.0), squeeze=False, sharey=True)
            for ci_c, cond in enumerate(conds):
                ax = axes[0, ci_c]
                ax.axhline(0, color=INK3, lw=0.8)
                for ci, model in enumerate(models):
                    y = [t.get((model, cond, s, "coded_soft"), np.nan) - t.get((model, cond, s, "uncoded_soft"), np.nan) for s in snrs]
                    ax.plot(snrs, y, color=color(ci), marker="o", label=model)
                finish(ax, "Es/N0 (dB)", ("%s gain of coded+ORBGRAND over uncoded" % METRIC_LABEL.get(metric, metric)) if ci_c == 0 else None,
                       cond, legend=ci_c == 0)
            fig.tight_layout()
            fig.savefig(os.path.join(args.out, "gain_%s%s.png" % (metric, tag)))
            plt.close(fig)

    # GRAND behaviour per condition
    gkeys = [("bler_after", "BLER after ORBGRAND"), ("abandon_rate", "abandoned"), ("miscorrect_rate", "miscorrected"), ("avg_queries", "avg queries")]
    coded = [p for p in paths if p.startswith("coded")]
    if coded:
        path = "coded_soft" if "coded_soft" in coded else coded[0]
        for cond in conds:
            sub = [r for r in rows if r["path"] == path and r["condition"] == cond]
            if not sub:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))
            for ci, model in enumerate(models):
                for ki, (k, lab) in enumerate(gkeys[:3]):
                    y = [max(np.mean([r[k] for r in sub if r["model"] == model and r["snr"] == s] or [np.nan]), 1e-5) for s in snrs]
                    axes[0].plot(snrs, y, color=color(ci), ls=["-", "--", ":"][ki], marker=["o", "s", "^"][ki], label="%s: %s" % (model, lab))
                y = [np.mean([r["avg_queries"] for r in sub if r["model"] == model and r["snr"] == s] or [np.nan]) for s in snrs]
                axes[1].plot(snrs, y, color=color(ci), marker="o", label=model)
            axes[0].set_yscale("log")
            axes[1].set_yscale("log")
            finish(axes[0], "Es/N0 (dB)", "rate (log)", "block outcomes, %s" % path)
            finish(axes[1], "Es/N0 (dB)", "average codebook queries (log)", "decoding effort", legend=len(models) > 1)
            fig.suptitle(describe(parse_condition(cond)), x=0.01, ha="left", fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(args.out, "grand_%s%s.png" % (cond, tag)))
            plt.close(fig)

    # markdown summary
    with open(os.path.join(args.out, "summary%s.md" % tag), "w") as f:
        for metric in metrics:
            t = tables[metric]
            f.write("\n## %s (mean over images)\n" % METRIC_LABEL.get(metric, metric))
            for model in models:
                for cond in conds:
                    f.write("\n**%s, %s**\n\n| Es/N0 | %s |\n|---|%s\n" % (model, cond, " | ".join(paths), "---|" * len(paths)))
                    for s in snrs:
                        f.write("| %g | %s |\n" % (s, " | ".join("%.3f" % t[(model, cond, s, p)] if (model, cond, s, p) in t else "-" for p in paths)))
    print("wrote figures and summary%s.md to %s" % (tag, args.out))


if __name__ == "__main__":
    main()
