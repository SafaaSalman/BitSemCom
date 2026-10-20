"""Quick self-test of the grand package: Landslide against brute force, codes, decoder."""
import itertools
import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grand import landslide, make_rlc, make_crc, encode, is_codeword, koopman_to_poly, ORBGRAND  # noqa: E402


def main():
    print("landslide(5,2,4)+1 ->", (landslide(5, 2, 4) + 1).tolist())
    print("landslide(10,3,6)+1 ->", (landslide(10, 3, 6) + 1).tolist())
    for n in [8, 12]:
        for w in range(1, 5):
            for W in range(w * (w + 1) // 2, w * (2 * n + 1 - w) // 2 + 1):
                z = landslide(W, w, n) + 1
                ref = sorted(c for c in itertools.combinations(range(1, n + 1), w) if sum(c) == W)
                got = sorted(map(tuple, z.tolist()))
                assert got == ref, (n, w, W, got, ref)
                assert len(set(got)) == len(got)
    print("landslide brute-force OK for n in {8,12}, w<=4")

    rng = np.random.default_rng(0)
    G, H = make_rlc(116, 128, seed=1)
    u = rng.integers(0, 2, (5, 116))
    c = encode(u, G)
    assert all(is_codeword(x, H) for x in c)
    print("RLC(128,116) ok", G.shape, H.shape)

    print("koopman 0x8f3 ->", koopman_to_poly("0x8f3"))
    G2, H2 = make_crc(116, koopman_to_poly("0x8f3"))
    c2 = encode(u, G2)
    assert all(is_codeword(x, H2) for x in c2)
    print("CRC(128,116) ok", G2.shape, H2.shape)

    for variant in ["basic", "1line"]:
        dec = ORBGRAND(H2, max_queries=10 ** 5, variant=variant)
        llr = (1 - 2 * c2[0].astype(float)) * 5.0
        chat, ng, ab = dec.decode(llr)
        assert (chat == c2[0]).all() and ng == 1 and not ab
        llr2 = llr.copy()
        llr2[[3, 70]] *= -0.3  # two confident-ish flips among the least reliable bits
        llr2[[10, 50, 90]] *= 0.5  # some unreliable but correct bits
        t0 = time.time()
        chat, ng, ab = dec.decode(llr2)
        print("%-5s 2-flip decode ok: %s guesses=%d abandoned=%s  (%.1f ms)"
              % (variant, (chat == c2[0]).all(), ng, ab, 1e3 * (time.time() - t0)))

    # random-noise block test: BPSK, Es/N0 = 7 dB, RLC(128,116)
    dec = ORBGRAND(H, max_queries=10 ** 5)
    snr = 10 ** (7 / 10)
    sigma = np.sqrt(1 / (2 * snr))
    nblk, nerr, naband, tot_g = 200, 0, 0, 0
    t0 = time.time()
    for i in range(nblk):
        ui = rng.integers(0, 2, 116)
        ci = encode(ui, G)
        y = (1 - 2 * ci.astype(float)) + sigma * rng.standard_normal(128)
        llr = 2 * y / sigma ** 2
        chat, ng, ab = dec.decode(llr)
        nerr += int((chat != ci).any())
        naband += int(ab)
        tot_g += ng
    dt = time.time() - t0
    print("RLC(128,116) BPSK 7dB: BLER=%.3f aband=%d avg_guesses=%.1f  %.1f ms/block"
          % (nerr / nblk, naband, tot_g / nblk, 1e3 * dt / nblk))


if __name__ == "__main__":
    main()
