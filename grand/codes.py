"""
Systematic binary linear block codes for GRAND decoding.

All codes are systematic: G = [I_k | P], H = [P^T | I_{n-k}], so the payload is the
first k bits of a codeword and c is a codeword iff H c = 0 (mod 2).
"""
import numpy as np


def make_rlc(k, n, p=0.5, seed=0):
    """Random linear code (after Duffy's make_RLC): no all-zero rows, no duplicate parity columns."""
    rng = np.random.default_rng(seed)
    while True:
        P = rng.binomial(1, p, size=(k, n - k)).astype(np.uint8)
        while (P.sum(axis=1) == 0).any():
            bad = P.sum(axis=1) == 0
            P[bad] = rng.binomial(1, p, size=(int(bad.sum()), n - k)).astype(np.uint8)
        if len({P[:, j].tobytes() for j in range(n - k)}) == n - k:
            break
    return _systematic(P)


def koopman_to_poly(hex_poly):
    """Koopman notation (implicit +1, explicit x^r term) -> coefficient list [x^r ... x^0]."""
    v = int(hex_poly, 16)
    bits = [int(b) for b in bin(v)[2:]]
    return bits + [1]


def make_crc(k, poly):
    """Systematic CRC code of dimension k from generator polynomial coefficients [x^r ... x^0]."""
    poly = [int(b) & 1 for b in poly]
    assert poly[0] == 1 and poly[-1] == 1
    r = len(poly) - 1
    g = int("".join(map(str, poly)), 2)
    P = np.zeros((k, r), dtype=np.uint8)
    for i in range(k):
        # message bit i is the coefficient of x^(k-1-i); parity = x^(k-1-i+r) mod g(x)
        rem = 1 << (k - 1 - i + r)
        for d in range(k - 1 - i + r, r - 1, -1):
            if (rem >> d) & 1:
                rem ^= g << (d - r)
        P[i] = [(rem >> (r - 1 - j)) & 1 for j in range(r)]
    return _systematic(P)


def _systematic(P):
    k, m = P.shape
    G = np.concatenate([np.eye(k, dtype=np.uint8), P], axis=1)
    H = np.concatenate([P.T, np.eye(m, dtype=np.uint8)], axis=1)
    return G, H


def encode(u, G):
    """Encode payload bits u (..., k) with generator G (k, n) -> codewords (..., n)."""
    u = np.asarray(u, dtype=np.uint8)
    return (u.astype(np.int64) @ G.astype(np.int64) % 2).astype(np.uint8)


def is_codeword(c, H):
    c = np.asarray(c, dtype=np.int64)
    return not ((c @ H.T.astype(np.int64)) % 2).any()
