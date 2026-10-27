"""
Evaluation sweep: models x channel conditions x SNR x receivers, with PSNR, MS-SSIM,
LPIPS and CLIP similarity, optionally decoding with Duffy's MATLAB code.

  python experiments/evaluate.py --C 32 \
      --ckpt stage2=Pretrain/BitSemCom_C32_stage2_ep480.model \
      --ckpt bench=runs/C32_bench/best.model --ckpt grand=runs/C32_grand/best.model \
      --snrs 0,2,4,6,8,10,12 --conditions awgn,awgn_over3,awgn_under3,fade8,fade8_h0.1 \
      --paths noiseless,uncoded_soft,uncoded_hard,coded_soft --out results/eval_C32.jsonl

Receivers (paths), all fed the same semantic bits per (model, image, SNR):
  noiseless        exact bits (ceiling)
  uncoded_soft     QPSK soft demod -> P(1) (the BitSemCom receiver), with the condition's CSI error
  uncoded_hard     hard decisions
  uncoded_*_il     same with the coded link's interleaver but no code (isolates the interleaving
                   gain from the coding gain in fading)
  coded_soft       (n,k) code + interleaver + ORBGRAND, abandoned blocks -> channel posteriors
  coded_hard       same, abandoned blocks -> hard decisions
  *_noil           coded without interleaver;  *_ebn0: coded at Es/N0 + 10log10(k/n)

--grand_backend matlab|octave routes every ORBGRAND call through matlab/orbgrand_decode_file.m
(see matlab/README.md). The python backend uses the reference's query accounting
(--budget_mode level), so both backends give identical decisions.

--budget: the primary sweep uses the budget the GRAND model was trained with
(train_bitsemcom.py --grand_budget, default 2000), because its decoder learned the
abandonment / miscorrection statistics of that budget; a larger budget is a separate sweep.

One JSON row per (model, image, condition, snr, path) in --out;
a PSNR summary table is printed at the end (experiments/plot_results.py draws the figures).
"""
import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.bitsemcom_io import decode_probs, encode_bits, load_images, load_model  # noqa: E402
from experiments.channel_coding import CodedLink, interleaver, make_code, qpsk_channel  # noqa: E402
from experiments.conditions import parse_condition  # noqa: E402
from experiments.grand_backend import make_decode_fn  # noqa: E402
from experiments.metrics import Metrics  # noqa: E402


def to_tensor(p1_flat, like):
    return torch.from_numpy(np.asarray(p1_flat, dtype=np.float32)).reshape(like.shape).to(like.device)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--C", type=int, default=32)
    ap.add_argument("--ckpt", action="append", default=[], help="name=path (repeatable); default: stage2=Pretrain/*C{C}_*")
    ap.add_argument("--images", default="datasets/kodak_patches256")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--snrs", default="0,2,4,6,8,10,12")
    ap.add_argument("--conditions", default="awgn,awgn_over3,awgn_under3,fade8,fade8_h0.1")
    ap.add_argument("--paths", default="noiseless,uncoded_soft,uncoded_hard,coded_soft")
    ap.add_argument("--code", default="crc:0x9eb2:128:112")
    ap.add_argument("--variant", default="1line", choices=["1line", "basic"])
    ap.add_argument("--budget", type=int, default=2000, help="ORBGRAND query budget (primary sweep: the --grand_budget of the model under test)")
    ap.add_argument("--budget_mode", default="level", choices=["level", "query"], help="'level' = query accounting of Duffy's MATLAB code")
    ap.add_argument("--grand_backend", default="python", choices=["python", "matlab", "octave"])
    ap.add_argument("--no_interleave", action="store_true")
    ap.add_argument("--metrics", default="all", help="'all' or comma list from psnr,msssim,lpips,clip")
    ap.add_argument("--dec_snr", default="estimated", help="SNR fed to the semantic decoder: 'estimated' (channel SNR + CSI error), 'true', or a number")
    ap.add_argument("--save_images", type=int, default=0, help="save reconstructions of the first N images (all paths) at --save_snr")
    ap.add_argument("--save_snr", type=float, default=4.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpts = [c.split("=", 1) for c in args.ckpt] or [["stage2", ""]]
    snrs = [float(s) for s in args.snrs.split(",")]
    conds = [parse_condition(c) for c in args.conditions.split(",")]
    paths = args.paths.split(",")
    want = set(["psnr", "msssim", "lpips", "clip"] if args.metrics == "all" else args.metrics.split(","))
    met = Metrics(args.device, use_lpips="lpips" in want, use_clip="clip" in want)
    G, H = make_code(args.code)
    decode_fn = None if args.grand_backend == "python" else make_decode_fn(args.grand_backend, H, args.budget, args.variant)
    link = CodedLink(G, H, max_queries=args.budget, variant=args.variant, budget_mode=args.budget_mode, decode_fn=decode_fn)
    rate_db = 10 * math.log10(link.rate)
    images = load_images(args.images, limit=args.limit or None)
    out_path = args.out or os.path.join(ROOT, "results", "eval_C%d.jsonl" % args.C)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img_dir = os.path.join(os.path.dirname(out_path), "images_C%d" % args.C)
    fout = open(out_path, "w")
    print("models=%s images=%d snrs=%s conditions=%s paths=%s code=%s budget=%d backend=%s metrics=%s" % (
        [n for n, _ in ckpts], len(images), snrs, [c["name"] for c in conds], paths, args.code, args.budget,
        args.grand_backend, met.names), flush=True)

    agg = defaultdict(list)
    t_start = time.time()
    for mname, mpath in ckpts:
        net = load_model(args.C, args.device, checkpoint=mpath or None)
        for ii, (iname, img) in enumerate(images):
            img = img.to(args.device)
            for snr in snrs:
                seed = args.seed * 100000 + ii * 1000 + int(round(snr * 10))
                bits = encode_bits(net, img, snr, seed=seed)  # same Gumbel seed for every model
                bits_np = bits[0].detach().cpu().numpy().astype(np.uint8).reshape(-1)
                for ci, cond in enumerate(conds):
                    # channel randomness depends on (image, snr) only: conditions that share a channel
                    # type see the same noise and fading, and differ only in the receiver's CSI error
                    cseed = seed
                    snr_err = cond["snr_err_db"]
                    if cond["snr_err_rand"]:
                        snr_err += float(np.random.default_rng(cseed + 11).uniform(-cond["snr_err_rand"], cond["snr_err_rand"]))
                    ch = dict(channel=cond["channel"], fade_block=cond["fade_block"], snr_err_db=snr_err, h_err_var=cond["h_err_var"])
                    results = {}
                    for path in paths:
                        stats = {}
                        ch_snr = snr
                        if path == "noiseless":
                            p1 = bits
                        elif path in ("uncoded_soft", "uncoded_hard", "uncoded_soft_il", "uncoded_hard_il"):
                            if path.endswith("_il"):  # same interleaver as the coded link, no code
                                perm, inv = interleaver(bits_np.size)
                                llr_p, p1_p = qpsk_channel(bits_np[perm], snr, np.random.default_rng(cseed + 2), **ch)
                                llr, p1f = llr_p[inv], p1_p[inv]
                            else:
                                llr, p1f = qpsk_channel(bits_np, snr, np.random.default_rng(cseed + 2), **ch)
                            p1 = to_tensor((llr < 0).astype(np.float32) if path.startswith("uncoded_hard") else p1f, bits)
                            stats = dict(ber_channel=float(((llr < 0) != bits_np).mean()), symbols=int(bits_np.size // 2))
                        elif path.startswith("coded"):
                            ch_snr = snr + rate_db if "_ebn0" in path else snr
                            fb = "soft" if "soft" in path else "hard"
                            il = (not args.no_interleave) and ("_noil" not in path)
                            p1f, stats = link.transmit(bits_np, ch_snr, np.random.default_rng(cseed + 3), fallback=fb, interleave=il, **ch)
                            p1 = to_tensor(p1f, bits)
                        else:
                            raise ValueError(path)
                        if args.dec_snr == "estimated":
                            dsnr = ch_snr + snr_err
                        elif args.dec_snr == "true":
                            dsnr = ch_snr
                        else:
                            dsnr = float(args.dec_snr)
                        rec = decode_probs(net, p1, dsnr)
                        m = met(img, rec)
                        stats.pop("payload_bits", None)
                        row = dict(model=mname, image=iname, C=args.C, snr=snr, condition=cond["name"], channel=cond["channel"],
                                   fade_block=cond["fade_block"], snr_err_db=snr_err, h_err_var=cond["h_err_var"], path=path,
                                   channel_snr=ch_snr, payload_bits=int(bits_np.size), **m, **stats)
                        fout.write(json.dumps(row) + "\n")
                        results[path] = row
                        agg[(mname, cond["name"], snr, path)].append(row)
                        if args.save_images and ii < args.save_images and snr == args.save_snr:
                            os.makedirs(img_dir, exist_ok=True)
                            arr = (rec[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
                            Image.fromarray(arr).save(os.path.join(img_dir, "%s_%s_%s_%gdB_%s.png" % (
                                os.path.splitext(iname)[0], mname, cond["name"], snr, path)))
                    fout.flush()
                    line = " ".join("%s=%.2f" % (p[:11], results[p]["psnr"]) for p in paths)
                    extra = ""
                    for p in paths:
                        if p.startswith("coded"):
                            r = results[p]
                            extra = " | %s: BER %.4f->%.5f aband %.2f misc %.3f q %.0f" % (
                                p[:10], r["ber_channel"], r["ber_after"], r["abandon_rate"], r["miscorrect_rate"], r["avg_queries"])
                            break
                    print("[%s %2d/%2d %-10s %-14s snr=%4.1f] %s%s" % (mname, ii + 1, len(images), iname[:10], cond["name"], snr, line, extra), flush=True)
        del net
        torch.cuda.empty_cache() if args.device.startswith("cuda") else None
    fout.close()

    print("\n=== PSNR (dB), mean over %d images, %.0fs ===" % (len(images), time.time() - t_start))
    for mname, _ in ckpts:
        for cond in conds:
            print("\nmodel %s | condition %s" % (mname, cond["name"]))
            print("%-7s" % "Es/N0" + "".join("%14s" % p[:13] for p in paths))
            for snr in snrs:
                print("%-7.1f" % snr + "".join("%14.2f" % np.mean([r["psnr"] for r in agg[(mname, cond["name"], snr, p)]]) for p in paths))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
