"""
Validate the Python ORBGRAND port against reference results shipped with Duffy's
GRAND-MATLAB repository (RESULTS/*.mat). Reproduces driver_GRAND.m: BPSK over a
complex AWGN channel with total noise variance sigma2 = 10^(-snr_db/10), so the
effective real-valued channel is y = (1 - 2c) + N(0, sigma2/2).

Usage:
  python grand/validate_duffy.py --mat <path to ORBGRAND_RLC_128_116_1.mat> --variant basic --ebn0 4,5,6,7
"""
import argparse
import os
import sys
import time

import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grand import ORBGRAND, encode  # noqa: E402


def load_code(mat_path):
    m = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)["code"]
    G = np.asarray(m.G).astype(np.uint8)
    k, n = G.shape
    assert (G[:, :k] == np.eye(k, dtype=np.uint8)).all(), "G is not systematic"
    P = G[:, k:]
    H = np.concatenate([P.T, np.eye(n - k, dtype=np.uint8)], axis=1)
    ref = {
        "ebn0": np.atleast_1d(m.ebn0).astype(float),
        "snr": np.atleast_1d(m.snr).astype(float),
        "BLER": np.atleast_1d(m.BLER).astype(float),
        "BLERdemod": np.atleast_1d(m.BLERdemod).astype(float),
        "EG": np.atleast_1d(m.EG).astype(float),
    }
    return G, H, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", required=True)
    ap.add_argument("--variant", default="basic", choices=["basic", "1line"])
    ap.add_argument("--ebn0", default="", help="comma list of Eb/N0 points to run (default: all)")
    ap.add_argument("--errors", type=int, default=30, help="stop an SNR point after this many block errors")
    ap.add_argument("--max_packets", type=int, default=20000)
    ap.add_argument("--max_queries", type=int, default=10 ** 6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    G, H, ref = load_code(args.mat)
    k, n = G.shape
    dec = ORBGRAND(H, max_queries=args.max_queries, variant=args.variant)
    rng = np.random.default_rng(args.seed)
    sel = list(range(len(ref["ebn0"])))
    if args.ebn0:
        want = [float(x) for x in args.ebn0.split(",")]
        sel = [i for i in sel if any(abs(ref["ebn0"][i] - w) < 1e-6 for w in want)]
    print("code (%d,%d)  variant=%s  file=%s" % (n, k, args.variant, os.path.basename(args.mat)))
    print(" ebn0  | ref BLERdemod  my BLERdemod | ref BLER   my BLER    | ref EG    my EG   | packets aband  s/pkt")
    for i in sel:
        snr_db = ref["snr"][i]
        sigma2 = 10 ** (-snr_db / 10)
        sigma = np.sqrt(sigma2 / 2)
        pk = errs = demod_errs = aband = 0
        tot_g = 0
        t0 = time.time()
        while errs < args.errors and pk < args.max_packets:
            u = rng.integers(0, 2, k, dtype=np.uint8)
            c = encode(u, G)
            y = (1.0 - 2.0 * c) + sigma * rng.standard_normal(n)
            llr = 4.0 * y / sigma2
            chat, ng, ab = dec.decode(llr)
            pk += 1
            tot_g += ng
            demod_errs += int(((llr < 0) != c).any())
            if ab or (chat != c).any():
                errs += 1
            aband += int(ab)
        dt = time.time() - t0
        print(" %4.2f  |  %.3e    %.3e   | %.3e  %.3e  | %8.1f  %8.1f | %6d  %4d   %.4f"
              % (ref["ebn0"][i], ref["BLERdemod"][i], demod_errs / pk, ref["BLER"][i], errs / pk,
                 ref["EG"][i], tot_g / pk, pk, aband, dt / pk))


if __name__ == "__main__":
    main()
