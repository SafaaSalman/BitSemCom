"""
Bit-exact verification of the Python ORBGRAND port against Ken Duffy's MATLAB code.

ORBGRAND is deterministic: for the same LLR vector, the same code and the same query
budget, the reference and the port must return the same codeword, the same number of
guesses and the same abandonment flag. This script makes that comparison on real
semantic bits sent through the channels used in the thesis.

Step 1 (any machine):
    python experiments/grand_verify.py export --out results/grand_verify
        -> results/grand_verify/testset.mat  (LLR blocks, H, budget, Python decisions)

Step 2 (a machine with MATLAB or Octave and Duffy's code, see matlab/README.md):
    MATLAB:  >> cd matlab; verify_port('../results/grand_verify/testset.mat')
    or from Python if matlab/octave is on PATH:
    python experiments/grand_verify.py run --backend matlab   (or --backend octave)
        -> results/grand_verify/matlab_out.mat

Step 3:
    python experiments/grand_verify.py compare
        -> prints mismatch counts (expected: 0)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.channel_coding import CodedLink, make_code  # noqa: E402

DEFAULT_OUT = os.path.join(ROOT, "results", "grand_verify")


def export(args):
    from scipy.io import savemat
    G, H = make_code(args.code)
    link = CodedLink(G, H, max_queries=args.budget, variant=args.variant, budget_mode="level")
    rng = np.random.default_rng(args.seed)
    if args.from_model:
        import torch
        from experiments.bitsemcom_io import encode_bits, load_images, load_model
        net = load_model(args.C, "cuda" if torch.cuda.is_available() else "cpu")
        imgs = load_images(os.path.join(ROOT, "datasets", "kodak_patches256"), limit=args.images)
        bit_sets = [encode_bits(net, img.to(next(net.parameters()).device), 6.0, seed=i)[0].cpu().numpy().astype(np.uint8).reshape(-1)
                    for i, (_, img) in enumerate(imgs)]
    else:
        bit_sets = [rng.integers(0, 2, size=256 * args.C, dtype=np.uint8) for _ in range(args.images)]

    conds = []
    for s in args.snrs.split(","):
        conds.append(dict(channel="awgn", snr=float(s), fade_block=0, snr_err_db=0.0, h_err_var=0.0))
    for s in args.fading_snrs.split(","):
        conds.append(dict(channel="blockfading", snr=float(s), fade_block=8, snr_err_db=0.0, h_err_var=0.0))
        conds.append(dict(channel="blockfading", snr=float(s), fade_block=8, snr_err_db=0.0, h_err_var=0.1))
    llr_all, words_all, g_all, a_all, meta = [], [], [], [], []
    t0 = time.time()
    for ci, cond in enumerate(conds):
        for bi, bits in enumerate(bit_sets):
            r = np.random.default_rng(args.seed + 1000 * ci + bi)
            _, st, det = link.transmit(bits, cond["snr"], r, fallback="soft", channel=cond["channel"],
                                       fade_block=cond["fade_block"], interleave=True, snr_err_db=cond["snr_err_db"],
                                       h_err_var=cond["h_err_var"], return_detail=True)
            llr_all.append(det["llr_blocks"])
            words_all.append(det["words"])
            g_all.append(det["guesses"])
            a_all.append(det["aband_blocks"])
            meta += [dict(cond=ci, **cond)] * det["llr_blocks"].shape[0]
            print("cond %d %-11s snr=%4.1f h_err=%.2f img %d: blocks=%d bler=%.3f aband=%.3f avgQ=%.0f" % (
                ci, cond["channel"], cond["snr"], cond["h_err_var"], bi, st["blocks"], st["bler_after"],
                st["abandon_rate"], st["avg_queries"]))
    llr = np.concatenate(llr_all)
    words = np.concatenate(words_all)
    guesses = np.concatenate(g_all)
    aband = np.concatenate(a_all)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "testset.mat")
    savemat(path, dict(llr=llr.astype(np.float64), H=H.astype(np.float64), G=G.astype(np.float64),
                       max_queries=float(args.budget), variant=args.variant, code=args.code,
                       words_py=words.astype(np.float64), guesses_py=guesses.astype(np.float64),
                       abandoned_py=aband.astype(np.float64), cond=np.array([m["cond"] for m in meta], dtype=np.float64)))
    with open(os.path.join(args.out, "conditions.json"), "w") as f:
        json.dump(conds, f, indent=1)
    print("wrote %s: %d blocks of n=%d (%.1f%% abandoned by Python, avg %.0f queries) in %.0fs" % (
        path, llr.shape[0], llr.shape[1], 100 * aband.mean(), guesses.mean(), time.time() - t0))
    print("Python decisions were made with budget_mode='level' (the MATLAB semantics).")


def run(args):
    from scipy.io import loadmat, savemat
    from experiments.grand_backend import decode_matlab_files
    ts = loadmat(os.path.join(args.out, "testset.mat"))
    variant = str(np.asarray(ts["variant"]).reshape(-1)[0]).strip()
    max_queries = float(np.asarray(ts["max_queries"]).reshape(-1)[0])
    t0 = time.time()
    words, guesses, aband = decode_matlab_files(ts["llr"], ts["H"], max_queries, variant, engine=args.backend)
    path = os.path.join(args.out, "matlab_out.mat")
    savemat(path, dict(words=words.astype(np.float64), guesses=guesses.astype(np.float64),
                       abandoned=aband.astype(np.float64), backend=args.backend))
    print("wrote %s (%d blocks, %.0fs with %s)" % (path, words.shape[0], time.time() - t0, args.backend))
    compare(args)


def compare(args):
    from scipy.io import loadmat
    ts = loadmat(os.path.join(args.out, "testset.mat"))
    mo = loadmat(os.path.join(args.out, "matlab_out.mat"))
    wp = np.rint(ts["words_py"]).astype(np.int64)
    gp = np.rint(ts["guesses_py"]).reshape(-1).astype(np.int64)
    ap = np.rint(ts["abandoned_py"]).reshape(-1).astype(bool)
    wm = np.rint(mo["words"]).astype(np.int64)
    gm = np.rint(mo["guesses"]).reshape(-1).astype(np.int64)
    am = np.rint(mo["abandoned"]).reshape(-1).astype(bool)
    nb = wp.shape[0]
    both_dec = ~ap & ~am
    word_mis = int((wp[both_dec] != wm[both_dec]).any(axis=1).sum())
    aband_mis = int((ap != am).sum())
    guess_mis = int((gp != gm).sum())
    cond = np.rint(ts["cond"]).reshape(-1).astype(int) if "cond" in ts else np.zeros(nb, int)
    print("blocks: %d | abandoned python %d, matlab %d" % (nb, ap.sum(), am.sum()))
    print("abandonment flag mismatches : %d" % aband_mis)
    print("guess-count mismatches      : %d" % guess_mis)
    print("codeword mismatches (both decoded): %d of %d" % (word_mis, both_dec.sum()))
    for c in sorted(set(cond.tolist())):
        m = cond == c
        print("  condition %d: %d blocks, aband mism %d, guess mism %d, word mism %d" % (
            c, m.sum(), (ap[m] != am[m]).sum(), (gp[m] != gm[m]).sum(),
            (wp[m & both_dec] != wm[m & both_dec]).any(axis=1).sum()))
    ok = aband_mis == 0 and guess_mis == 0 and word_mis == 0
    print("RESULT: %s" % ("IDENTICAL - the Python port reproduces the MATLAB reference bit for bit" if ok else "MISMATCH - inspect"))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["export", "run", "compare"])
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--code", default="crc:0x9eb2:128:112")
    ap.add_argument("--variant", default="1line", choices=["1line", "basic"])
    ap.add_argument("--budget", type=int, default=10 ** 4, help="query budget (keep moderate: MATLAB is slow)")
    ap.add_argument("--C", type=int, default=32)
    ap.add_argument("--images", type=int, default=2, help="bit sets per condition (1 image = 256*C bits = 74 blocks)")
    ap.add_argument("--from_model", action="store_true", help="use real semantic bits from the C model (needs checkpoint)")
    ap.add_argument("--snrs", default="3,5,7", help="AWGN Es/N0 points")
    ap.add_argument("--fading_snrs", default="10", help="block-fading average Es/N0 points")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default="matlab", choices=["matlab", "octave"])
    args = ap.parse_args()
    {"export": export, "run": run, "compare": compare}[args.mode](args)


if __name__ == "__main__":
    main()
