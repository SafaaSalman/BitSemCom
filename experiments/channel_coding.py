"""
Channel-coding layer between BitSemCom's bit mapper and its semantic decoder.

Transmitter:  semantic bits (flattened) -> pad -> k-bit blocks -> [n,k] systematic code
              -> optional interleaver -> QPSK (Gray, same mapping as network.QPSK_soft)
              -> AWGN or block-Rayleigh fading
Receiver:     coherent LLRs (positive = bit 0), optionally with imperfect CSI
              -> de-interleave -> ORBGRAND per block -> payload bits -> unpad
              -> P(bit=1) for the semantic decoder

Imperfect CSI (both evaluation here and training in experiments/torch_channel.py):
  * snr_err_db : the receiver computes LLRs with the noise variance of snr_db + snr_err_db
                 (positive = receiver over-confident, LLR magnitudes too large)
  * h_err_var  : the receiver uses h_hat = h + e, e ~ CN(0, h_err_var) for the matched
                 filter and the LLR scaling (fading channels only)

All randomness goes through an explicit numpy Generator so every path in an experiment
sees the same transmitted bits and the same noise. The draw order (noise, then fading
gains, then estimation errors) keeps the perfect-CSI results of the first round
reproducible.
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grand import ORBGRAND, encode, make_crc, make_rlc, koopman_to_poly  # noqa: E402

SQRT2 = math.sqrt(2.0)


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


def noise_var(snr_db):
    return 1.0 / (2.0 * 10 ** (snr_db / 10.0))


def interleaver(n, seed=12345):
    """Fixed pseudo-random permutation known to both sides (the coded link's interleaver,
    also used by the uncoded '_il' reference paths). Returns (perm, inv): transmit x[perm],
    de-interleave the received sequence with y[inv]."""
    perm = np.random.default_rng(seed).permutation(n)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(n)
    return perm, inv


def qpsk_channel(bits, snr_db, rng, channel="awgn", fade_block=0, snr_err_db=0.0, h_err_var=0.0,
                 return_extra=False):
    """QPSK + AWGN exactly as network.QPSK_soft (consecutive bit pairs -> (I,Q), Gray map
    0 -> +1/sqrt2, 1 -> -1/sqrt2, per-component noise variance 1/(2*Es/N0)), optionally
    with block Rayleigh fading: one complex gain h ~ CN(0,1) per `fade_block` consecutive
    symbols (fade_block=1: fast fading). snr_db is the AVERAGE Es/N0 per symbol.
    Returns (llr, p1) flattened to the length of bits (positive LLR = bit 0); with
    return_extra=True also a dict with the per-symbol quantities used by the plots."""
    bits = np.asarray(bits, dtype=np.float64)
    assert bits.size % 2 == 0
    b = bits.reshape(-1, 2)
    s = (1.0 - 2.0 * b) / SQRT2  # (nsym, 2) real/imag
    nsym = s.shape[0]
    nv = noise_var(snr_db)
    nv_hat = noise_var(snr_db + snr_err_db)
    noise = math.sqrt(nv) * rng.standard_normal(s.shape)
    hs = hhs = None
    if channel == "awgn":
        r = s + noise
        gain = np.ones(nsym)
    elif channel == "blockfading":
        fb = int(fade_block)
        if fb < 1:
            raise ValueError("blockfading needs fade_block >= 1 symbols (1 = fast fading), got %r" % (fade_block,))
        nblk = int(math.ceil(nsym / fb))
        h = (rng.standard_normal(nblk) + 1j * rng.standard_normal(nblk)) / SQRT2
        hs = np.repeat(h, fb)[:nsym]
        sc = (s[:, 0] + 1j * s[:, 1]) * hs
        rc = sc + (noise[:, 0] + 1j * noise[:, 1])
        if h_err_var > 0:
            e = math.sqrt(h_err_var / 2.0) * (rng.standard_normal(nblk) + 1j * rng.standard_normal(nblk))
            hhs = np.repeat(h + e, fb)[:nsym]
        else:
            hhs = hs
        # coherent matched filter with the receiver's channel estimate
        gain = np.abs(hhs)
        y = rc * np.conj(hhs) / np.maximum(gain, 1e-12)
        r = np.stack([y.real, y.imag], axis=1)
    else:
        raise ValueError(channel)
    llr = (SQRT2 * gain[:, None] * r / nv_hat).reshape(-1)
    p1 = 1.0 / (1.0 + np.exp(np.clip(llr, -500, 500)))  # sigmoid(-llr)
    if return_extra:
        return llr, p1, dict(nv=nv, nv_hat=nv_hat, gain=gain, h=hs, h_hat=hhs, r=r)
    return llr, p1


def qpsk_awgn(bits, snr_db, rng):
    return qpsk_channel(bits, snr_db, rng, "awgn")


class CodedLink:
    """[n,k] systematic code + QPSK link + ORBGRAND receiver.

    decode_fn: optional replacement for the numpy decoder with the signature
        decode_fn(llr_blocks (nb, n)) -> (words (nb, n) 0/1, n_guesses (nb,), abandoned (nb,)),
    e.g. experiments.grand_backend.make_decode_fn('matlab', ...). Like the numpy decoder it
    returns the hard decisions in abandoned rows.
    budget_mode: 'query' (stop exactly at max_queries) or 'level' (the per-weight-level check
    of Duffy's MATLAB code; use it whenever numbers are compared with the reference).
    """

    def __init__(self, G, H, max_queries=10 ** 5, variant="1line", budget_mode="query", decode_fn=None):
        self.G, self.H = G, H
        self.k, self.n = G.shape
        self.dec = ORBGRAND(H, max_queries=max_queries, variant=variant, budget_mode=budget_mode)
        self.decode_fn = decode_fn

    @property
    def rate(self):
        return self.k / self.n

    def transmit(self, bits_flat, snr_db, rng, fallback="hard", channel="awgn", fade_block=0, interleave=False,
                 snr_err_db=0.0, h_err_var=0.0, return_detail=False):
        """bits_flat: 1-D 0/1 array of semantic bits. Returns (p1_out, stats[, detail]).

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
        ch_kw = dict(channel=channel, fade_block=fade_block, snr_err_db=snr_err_db, h_err_var=h_err_var)
        if interleave:
            perm, inv = interleaver(tx.size)
            llr_tx, p1_tx = qpsk_channel(tx[perm], snr_db, rng, **ch_kw)
            llr, p1 = llr_tx[inv], p1_tx[inv]
        else:
            llr, p1 = qpsk_channel(tx, snr_db, rng, **ch_kw)
        llr_b = llr[: nb * self.n].reshape(nb, self.n)
        p1_b = p1[: nb * self.n].reshape(nb, self.n)
        hard_b = (llr_b < 0).astype(np.uint8)

        t0 = time.time()
        words, guesses, aband = (self.decode_fn or self.dec.decode_many)(llr_b)
        dec_time = time.time() - t0
        words = np.asarray(words, dtype=np.uint8)
        guesses = np.asarray(guesses, dtype=np.int64).reshape(-1)
        aband = np.asarray(aband, dtype=bool).reshape(-1)

        payload_hat = words[:, : self.k]
        out = payload_hat.astype(np.float64)
        if fallback == "soft":
            out[aband] = p1_b[aband, : self.k]
        p1_out = out.reshape(-1)[:L]

        payload_hard = hard_b[:, : self.k].reshape(-1)[:L]
        payload_dec = payload_hat.reshape(-1)[:L]
        block_err = (words != c).any(axis=1)
        stats = dict(
            n=self.n, k=self.k, blocks=int(nb), payload_bits=int(L),
            coded_bits=int(nb * self.n), symbols=int(tx.size // 2),
            ber_channel=float((payload_hard != bits_flat).mean()),
            ber_after=float((payload_dec != bits_flat).mean()),
            bler_channel=float((hard_b != c).any(axis=1).mean()),
            bler_after=float(block_err.mean()),
            abandon_rate=float(aband.mean()),
            miscorrect_rate=float((block_err & ~aband).mean()),
            avg_queries=float(guesses.mean()),
            max_queries_seen=int(guesses.max()),
            decode_time_s=float(dec_time),
        )
        if return_detail:
            per_bit = lambda v: np.repeat(v, self.k)[:L]  # block flag -> payload-bit flag
            detail = dict(
                llr_payload=llr_b[:, : self.k].reshape(-1)[:L],
                p1_payload=p1_b[:, : self.k].reshape(-1)[:L],
                abandoned=per_bit(aband),
                miscorrected=per_bit(block_err & ~aband),
                llr_blocks=llr_b, words=words, codewords=c, guesses=guesses, aband_blocks=aband,
            )
            return p1_out, stats, detail
        return p1_out, stats
