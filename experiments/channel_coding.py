"""
Channel-coding layer between BitSemCom's bit mapper and its semantic decoder.

Transmitter:  semantic bits (flattened) -> pad -> k-bit blocks -> [n,k] systematic code
              -> optional interleaver -> QPSK (Gray, same mapping as network.QPSK_soft)
              -> AWGN or block-Rayleigh fading
Receiver:     coherent LLRs (positive = bit 0) -> de-interleave -> ORBGRAND per block
              -> payload bits -> unpad -> P(bit=1) for the semantic decoder

All randomness goes through an explicit numpy Generator so every path in an experiment
sees the same transmitted bits.
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grand import ORBGRAND, encode, make_crc, make_rlc, koopman_to_poly  # noqa: E402


def make_code(spec):
    """spec examples: 'crc:0x9eb2:128:112', 'crc:0x8f3:128:116', 'rlc:128:112:0' (seed)."""
    parts = spec.split(":")
    if parts[0] == "crc":
        _, hexpoly, n, k = parts
        n, k = int(n), int(k)
        poly = koopman_to_poly(hexpoly)
        assert len(poly) - 1 == n - k, "polynomial degree %d != n-k=%d" % (len(poly) - 1, n - k)
        G, H = make_crc(k, poly)
    elif parts[0] == "rlc":
        _, n, k, seed = parts
        G, H = make_rlc(int(k), int(n), seed=int(seed))
    else:
        raise ValueError(spec)
    return G, H


def qpsk_channel(bits, snr_db, rng, channel="awgn", fade_block=0):
    """QPSK + AWGN exactly as network.QPSK_soft (consecutive bit pairs -> (I,Q), Gray map
    0 -> +1/sqrt2, 1 -> -1/sqrt2, per-component noise variance 1/(2*Es/N0)), optionally
    with block Rayleigh fading: one complex gain h ~ CN(0,1) per `fade_block` consecutive
    symbols, known at the receiver (coherent detection). snr_db is the AVERAGE Es/N0.
    Returns (llr, p1) flattened to the length of bits (positive LLR = bit 0)."""
    bits = np.asarray(bits, dtype=np.float64)
    assert bits.size % 2 == 0
    b = bits.reshape(-1, 2)
    s = (1.0 - 2.0 * b) / math.sqrt(2.0)  # (nsym, 2) real/imag
    snr = 10 ** (snr_db / 10.0)
    nv = 1.0 / (2.0 * snr)
    noise = math.sqrt(nv) * rng.standard_normal(s.shape)
    if channel == "awgn":
        r = s + noise
        gain = np.ones(s.shape[0])
    elif channel == "blockfading":
        nsym = s.shape[0]
        nblk = int(math.ceil(nsym / fade_block))
        h = (rng.standard_normal(nblk) + 1j * rng.standard_normal(nblk)) / math.sqrt(2.0)
        hs = np.repeat(h, fade_block)[:nsym]
        sc = (s[:, 0] + 1j * s[:, 1]) * hs
        rc = sc + (noise[:, 0] + 1j * noise[:, 1])
        # coherent matched filter: rotate by conj(h)/|h| -> |h| s + noise'
        gain = np.abs(hs)
        y = rc * np.conj(hs) / np.maximum(gain, 1e-12)
        r = np.stack([y.real, y.imag], axis=1)
    else:
        raise ValueError(channel)
    llr = (math.sqrt(2.0) * gain[:, None] * r / nv).reshape(-1)
    p1 = 1.0 / (1.0 + np.exp(np.clip(llr, -500, 500)))  # sigmoid(-llr)
    return llr, p1


def qpsk_awgn(bits, snr_db, rng):
    return qpsk_channel(bits, snr_db, rng, "awgn")


class CodedLink:
    def __init__(self, G, H, max_queries=10 ** 5, variant="1line"):
        self.G, self.H = G, H
        self.k, self.n = G.shape
        self.dec = ORBGRAND(H, max_queries=max_queries, variant=variant)

    @property
    def rate(self):
        return self.k / self.n

    def transmit(self, bits_flat, snr_db, rng, fallback="hard", channel="awgn", fade_block=0, interleave=False):
        """bits_flat: 1-D 0/1 array of semantic bits. Returns (p1_out, stats).

        p1_out is the P(bit=1) sequence handed to the semantic decoder: decoded blocks
        give exact 0/1 values; abandoned blocks give either the hard decisions
        ('hard') or the channel posteriors of their systematic bits ('soft').
        With interleave=True a fixed pseudo-random permutation (known to both sides)
        spreads every codeword over the whole transmission.
        """
        bits_flat = np.asarray(bits_flat, dtype=np.uint8).reshape(-1)
        L = bits_flat.size
        nb = math.ceil(L / self.k)
        pad = nb * self.k - L
        u = np.concatenate([bits_flat, np.zeros(pad, dtype=np.uint8)]).reshape(nb, self.k)
        c = encode(u, self.G)  # (nb, n)
        tx = c.reshape(-1)
        extra = tx.size % 2
        if extra:
            tx = np.concatenate([tx, np.zeros(1, dtype=np.uint8)])
        if interleave:
            perm = np.random.default_rng(12345).permutation(tx.size)
            inv = np.empty_like(perm)
            inv[perm] = np.arange(tx.size)
            llr_tx, p1_tx = qpsk_channel(tx[perm], snr_db, rng, channel, fade_block)
            llr, p1 = llr_tx[inv], p1_tx[inv]
        else:
            llr, p1 = qpsk_channel(tx, snr_db, rng, channel, fade_block)
        llr_b = llr[: nb * self.n].reshape(nb, self.n)
        p1_b = p1[: nb * self.n].reshape(nb, self.n)

        t0 = time.time()
        words, guesses, aband = self.dec.decode_many(llr_b)
        dec_time = time.time() - t0

        hard_b = (llr_b < 0).astype(np.uint8)
        payload_hat = words[:, : self.k]
        out = payload_hat.astype(np.float64)
        if fallback == "soft":
            out[aband] = p1_b[aband, : self.k]
        p1_out = out.reshape(-1)[:L]

        payload_hard = hard_b[:, : self.k].reshape(-1)[:L]
        payload_dec = payload_hat.reshape(-1)[:L]
        stats = dict(
            n=self.n, k=self.k, blocks=int(nb), payload_bits=int(L),
            coded_bits=int(nb * self.n), symbols=int(tx.size // 2),
            ber_channel=float((payload_hard != bits_flat).mean()),
            ber_after=float((payload_dec != bits_flat).mean()),
            bler_channel=float((hard_b != c).any(axis=1).mean()),
            bler_after=float(((words != c).any(axis=1)).mean()),
            abandon_rate=float(aband.mean()),
            miscorrect_rate=float(((words != c).any(axis=1) & ~aband).mean()),
            avg_queries=float(guesses.mean()),
            max_queries_seen=int(guesses.max()),
            decode_time_s=float(dec_time),
        )
        return p1_out, stats
