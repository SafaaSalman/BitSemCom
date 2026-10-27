"""
BitSemCom + GRAND experiment runner.

For every image and every Es/N0 it transmits the SAME semantic bitstream through:
  noiseless      : decoder fed the exact bits (ceiling for any channel code)
  uncoded_soft   : original BitSemCom receiver (QPSK soft demod -> P(1) -> decoder)
  uncoded_hard   : hard decisions (0/1) -> decoder
  coded_hard     : [n,k] code + ORBGRAND, abandoned blocks fall back to hard decisions
  coded_soft     : same, abandoned blocks fall back to channel posteriors
  *_ebn0         : coded paths run at Es/N0 + 10log10(k/n), i.e. equal energy per
                   semantic bit (coded path spends n/k more channel uses)

Example:
  python experiments/run_grand.py --C 16 --snrs 4,6,8,10 --limit 12 --code crc:0x9eb2:128:112
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.bitsemcom_io import (Metrics, decode_probs, encode_bits, load_images,  # noqa: E402
                                      load_model, qpsk_soft_original)
from experiments.channel_coding import CodedLink, interleaver, make_code, qpsk_channel  # noqa: E402


def to_tensor(p1_flat, like):
    return torch.from_numpy(np.asarray(p1_flat, dtype=np.float32)).reshape(like.shape).to(like.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=int, default=16)
    ap.add_argument("--snrs", default="4,6,8,10", help="Es/N0 in dB (QPSK), comma separated")
    ap.add_argument("--images", default="datasets/kodak_patches256")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--code", default="crc:0x9eb2:128:112")
    ap.add_argument("--variant", default="1line", choices=["1line", "basic"])
    ap.add_argument("--max_queries", type=int, default=10 ** 5)
    ap.add_argument("--paths", default="noiseless,uncoded_soft,uncoded_hard,coded_hard,coded_soft,coded_hard_ebn0,coded_soft_ebn0")
    ap.add_argument("--dec_snr", default="channel", help="'channel' or a fixed dB value fed to the semantic decoder")
    ap.add_argument("--channel", default="awgn", choices=["awgn", "blockfading"])
    ap.add_argument("--fade_block", type=int, default=8, help="symbols per fading block (8 = one C=16 token)")
    ap.add_argument("--no_interleave", action="store_true", help="coded paths: do not interleave codewords")
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    snrs = [float(s) for s in args.snrs.split(",")]
    paths = args.paths.split(",")
    G, H = make_code(args.code)
    link = CodedLink(G, H, max_queries=args.max_queries, variant=args.variant)
    rate_db = 10 * math.log10(link.rate)
    net = load_model(args.C, args.device)
    met = Metrics(torch.device(args.device))
    images = load_images(args.images, limit=args.limit or None)
    out_path = args.out or "results/C%d_%s_%s.jsonl" % (args.C, args.code.replace(":", "_"), args.variant)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fout = open(out_path, "w")
    print("C=%d code=%s (n=%d,k=%d, R=%.3f -> %.2f dB) variant=%s max_queries=%d images=%d channel=%s fade_block=%d interleave=%s"
          % (args.C, args.code, link.n, link.k, link.rate, rate_db, args.variant, args.max_queries, len(images),
             args.channel, args.fade_block, not args.no_interleave))

    agg = defaultdict(list)
    t_start = time.time()
    for ii, (name, img) in enumerate(images):
        img = img.to(args.device)
        for snr in snrs:
            seed = args.seed * 100000 + ii * 1000 + int(round(snr * 10))
            bits = encode_bits(net, img, snr, seed=seed)
            bits_np = bits[0].detach().cpu().numpy().astype(np.uint8).reshape(-1)
            rng = np.random.default_rng(seed)

            def dec_snr(channel_snr):
                return channel_snr if args.dec_snr == "channel" else float(args.dec_snr)

            results = {}
            for path in paths:
                stats = {}
                if path == "noiseless":
                    p1 = bits
                    ch_snr = snr
                elif path == "uncoded_soft":
                    if args.channel != "awgn":
                        raise ValueError("uncoded_soft uses the original QPSK_soft (AWGN only); use uncoded_soft_np")
                    torch.manual_seed(seed + 1)
                    p1 = qpsk_soft_original(bits, snr)
                    ch_snr = snr
                elif path in ("uncoded_soft_np", "uncoded_soft_il", "uncoded_hard"):
                    # numpy channel; '_il' interleaves the semantic bits over the transmission
                    rng_u = np.random.default_rng(seed + 2)
                    if path == "uncoded_soft_il":
                        perm, inv = interleaver(bits_np.size)
                        llr_p, p1_p = qpsk_channel(bits_np[perm], snr, rng_u, args.channel, args.fade_block)
                        llr, p1f = llr_p[inv], p1_p[inv]
                    else:
                        llr, p1f = qpsk_channel(bits_np, snr, rng_u, args.channel, args.fade_block)
                    if path == "uncoded_hard":
                        p1 = to_tensor((llr < 0).astype(np.float32), bits)
                    else:
                        p1 = to_tensor(p1f, bits)
                    stats = dict(ber_channel=float(((llr < 0) != bits_np).mean()), symbols=int(bits_np.size // 2))
                    ch_snr = snr
                elif path.startswith("coded"):
                    ch_snr = snr + rate_db if "_ebn0" in path else snr
                    fb = "soft" if "soft" in path else "hard"
                    il = (not args.no_interleave) and ("_noil" not in path)
                    p1f, stats = link.transmit(bits_np, ch_snr, np.random.default_rng(seed + 3), fallback=fb,
                                               channel=args.channel, fade_block=args.fade_block, interleave=il)
                    p1 = to_tensor(p1f, bits)
                else:
                    raise ValueError(path)
                rec = decode_probs(net, p1, dec_snr(ch_snr))
                psnr, ms = met(img, rec)
                stats.pop("payload_bits", None)
                rec_row = dict(image=name, C=args.C, snr=snr, path=path, channel_snr=ch_snr,
                               psnr=psnr, msssim=ms, payload_bits=int(bits_np.size), **stats)
                fout.write(json.dumps(rec_row) + "\n")
                results[path] = rec_row
                agg[(snr, path)].append(rec_row)
            fout.flush()
            line = " ".join("%s=%.2f" % (p[:12], results[p]["psnr"]) for p in paths)
            extra = ""
            if "coded_hard" in results:
                r = results["coded_hard"]
                extra = " | BERch=%.4f BERgrand=%.5f aband=%.2f EG=%.0f t=%.1fs" % (
                    r["ber_channel"], r["ber_after"], r["abandon_rate"], r["avg_queries"], r["decode_time_s"])
            print("[%2d/%2d %s snr=%4.1f] %s%s" % (ii + 1, len(images), name, snr, line, extra), flush=True)
    fout.close()

    print("\n=== Summary (mean over %d images), C=%d, code %s, %.0fs ===" % (len(images), args.C, args.code, time.time() - t_start))
    hdr = "%-7s" % "Es/N0" + "".join("%16s" % p[:15] for p in paths)
    print(hdr)
    for snr in snrs:
        print("%-7.1f" % snr + "".join("%16.2f" % np.mean([r["psnr"] for r in agg[(snr, p)]]) for p in paths))
    print("MS-SSIM:")
    for snr in snrs:
        print("%-7.1f" % snr + "".join("%16.4f" % np.mean([r["msssim"] for r in agg[(snr, p)]]) for p in paths))
    for p in paths:
        if p.startswith("coded"):
            print("GRAND stats for %s:" % p)
            print("  Es/N0  chan_snr  BERch     BERafter  BLERch  BLERafter aband  miscorr  avgQ    symbols/payload")
            for snr in snrs:
                rows = agg[(snr, p)]
                m = lambda key: float(np.mean([r[key] for r in rows]))
                print("  %5.1f  %7.2f   %.5f  %.6f  %.3f   %.4f    %.3f  %.4f  %7.0f  %.3f" % (
                    snr, m("channel_snr"), m("ber_channel"), m("ber_after"), m("bler_channel"), m("bler_after"),
                    m("abandon_rate"), m("miscorrect_rate"), m("avg_queries"), m("symbols") / (m("payload_bits") / 2)))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
