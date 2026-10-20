"""
Ordered Reliability Bits GRAND (ORBGRAND) in pure numpy.

A port of the reference MATLAB implementation by Ken R. Duffy
(https://github.com/kenrduffy/GRAND-MATLAB): basic ORBGRAND, 1-line ORBGRAND
and the Landslide integer-partition generator. The MATLAB code is subject to the
"GRAND Codebase Non-Commercial Academic Research Use License"; this port is for
academic research only and the following should be cited with any results:

  K. R. Duffy, "Ordered reliability bits guessing random additive noise decoding,"
  IEEE ICASSP, 2021, pp. 8268-8272.
  K. R. Duffy, W. An, M. Medard, "Ordered reliability bits guessing random additive
  noise decoding," IEEE Trans. Signal Process., vol. 70, pp. 4528-4542, 2022.

Conventions
-----------
* LLR sign: positive LLR means bit 0 is more likely, i.e. LLR = log P(0)/P(1).
  This is the convention of Duffy's code and of BitSemCom's QPSK_soft.
* H is an (n-k) x n binary parity-check matrix; c is a codeword iff H c = 0 (mod 2).
* Syndromes are packed into uint64 bit masks, so n-k must be <= 63.
"""
import math
import numpy as np

_LANDSLIDE_CACHE = {}


def _mountain_build(u, k, w, W1, n1):
    base = u[k]
    for i in range(k + 1, w):
        u[i] = base
    W2 = W1 - sum(u)
    span = n1 - base
    if span > 0:
        q, r = divmod(W2, span)
    else:
        q, r = 0, W2
    if q:
        for i in range(w - q, w):
            u[i] = n1
    if w - q > 0:
        u[w - q - 1] += r
    return u


def landslide(W, w, n):
    """All error patterns of Hamming weight w and logistic weight W over n sorted positions.

    Returns an int64 array of shape (num_patterns, w) holding 0-indexed positions in
    the reliability-sorted order (position 0 = least reliable bit). Patterns are
    generated in Landslide order and cached, since they depend only on (W, w, n).
    """
    key = (W, w, n)
    z = _LANDSLIDE_CACHE.get(key)
    if z is not None:
        return z
    W1 = W - w * (w + 1) // 2
    n1 = n - w
    u = [0] * w
    u = _mountain_build(u, 0, w, W1, n1)
    out = [list(u)]
    # Accumulated drop D[i] = u[w-1] - u[i]; loop while D[0] >= 2.
    while u[w - 1] - u[0] >= 2:
        k = w - 1
        while u[w - 1] - u[k] < 2:
            k -= 1
        u[k] += 1
        u = _mountain_build(u, k, w, W1, n1)
        out.append(list(u))
    z = np.asarray(out, dtype=np.int64) + np.arange(w, dtype=np.int64)
    _LANDSLIDE_CACHE[key] = z
    return z


def _pack_columns(H):
    r, n = H.shape
    if r > 63:
        raise ValueError("n-k must be <= 63 for packed syndromes (got %d)" % r)
    weights = np.left_shift(np.uint64(1), np.arange(r, dtype=np.uint64))
    return (H.astype(np.uint64).T * weights).sum(axis=1).astype(np.uint64)


def _basic_schedule(n):
    """Yield (W, w) in basic ORBGRAND order: increasing logistic weight W."""
    W = 0
    W_max = n * (n + 1) // 2
    while W < W_max:
        W += 1
        # smallest w whose maximal logistic weight w(2n+1-w)/2 reaches W
        w = math.ceil((1 + 2 * n - math.sqrt((1 + 2 * n) ** 2 - 8 * W)) / 2)
        w = max(w, 1)
        while w * (2 * n + 1 - w) < 2 * W:
            w += 1
        while w > 1 and (w - 1) * (2 * n + 1 - (w - 1)) >= 2 * W:
            w -= 1
        w_max = (math.isqrt(1 + 8 * W) - 1) // 2
        while w <= w_max:
            yield W, w
            w += 1


def _oneline_schedule(n, L):
    """Yield (W, w) in 1-line ORBGRAND order. L: sorted ascending reliabilities."""
    half = int(math.floor(n / 2 + 0.5))  # MATLAB round(n/2)
    beta = (L[half - 1] - L[0]) / (half - 1)
    if beta > 0:
        c = max(int(math.floor(L[0] / beta - 1 + 0.5)), 0)
    else:
        c = 0
    wt = c + 1
    wt_max = c * n + n * (n + 1) // 2
    while wt <= wt_max:
        w = math.ceil((1 + 2 * (n + c) - math.sqrt((1 + 2 * (n + c)) ** 2 - 8 * wt)) / 2)
        w = max(1, w)
        while w <= n:
            W = wt - c * w
            if W < w * (w + 1) // 2:
                break
            if w * (2 * n + 1 - w) >= 2 * W:  # feasibility guard against float error
                yield W, w
            w += 1
        wt += 1


class ORBGRAND:
    """ORBGRAND decoder for a binary linear code given by parity-check matrix H.

    Parameters
    ----------
    H : (n-k, n) binary array
    max_queries : abandon after this many codebook queries
    variant : '1line' (default, better at low SNR) or 'basic'
    """

    def __init__(self, H, max_queries=10 ** 5, variant="1line"):
        self.H = np.asarray(H, dtype=np.uint8) & 1
        self.r, self.n = self.H.shape
        self.cols = _pack_columns(self.H)
        self.max_queries = int(max_queries)
        if variant not in ("1line", "basic"):
            raise ValueError("variant must be '1line' or 'basic'")
        self.variant = variant

    def syndrome(self, bits):
        bits = np.asarray(bits, dtype=bool)
        if not bits.any():
            return np.uint64(0)
        return np.bitwise_xor.reduce(self.cols[bits])

    def decode(self, llr):
        """Decode one block from channel LLRs (positive = bit 0).

        Returns (codeword_hat, n_guesses, abandoned). When abandoned, codeword_hat is
        the hard-decision word (not a codeword), which is the receiver-side fallback.
        """
        llr = np.asarray(llr, dtype=np.float64)
        n = self.n
        if llr.shape[0] != n:
            raise ValueError("expected %d LLRs, got %d" % (n, llr.shape[0]))
        y = (llr < 0).astype(np.uint8)
        s = self.syndrome(y)
        n_guesses = 1
        if s == 0:
            return y, n_guesses, False
        order = np.argsort(np.abs(llr), kind="stable")
        cols_sorted = self.cols[order]
        if self.variant == "basic":
            schedule = _basic_schedule(n)
        else:
            schedule = _oneline_schedule(n, np.abs(llr)[order])
        for W, w in schedule:
            remaining = self.max_queries - n_guesses
            if remaining <= 0:
                break
            z = landslide(W, w, n)
            if z.shape[0] > remaining:
                z = z[:remaining]
            synd = np.bitwise_xor.reduce(cols_sorted[z], axis=1)
            hit = np.flatnonzero(synd == s)
            if hit.size:
                j = int(hit[0])
                n_guesses += j + 1
                e = np.zeros(n, dtype=np.uint8)
                e[order[z[j]]] = 1
                return y ^ e, n_guesses, False
            n_guesses += z.shape[0]
        return y, n_guesses, True

    def decode_many(self, llrs):
        """Decode a (num_blocks, n) array of LLRs. Returns (words, n_guesses, abandoned)."""
        llrs = np.asarray(llrs, dtype=np.float64)
        words = np.zeros(llrs.shape, dtype=np.uint8)
        guesses = np.zeros(llrs.shape[0], dtype=np.int64)
        aband = np.zeros(llrs.shape[0], dtype=bool)
        for i in range(llrs.shape[0]):
            words[i], guesses[i], aband[i] = self.decode(llrs[i])
        return words, guesses, aband
