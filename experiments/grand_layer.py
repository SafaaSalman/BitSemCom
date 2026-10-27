"""
GRAND-in-the-loop channel block for training.

    grand = GrandChannel("crc:0x9eb2:128:112", max_queries=2000, fallback="soft", workers=4)
    p1, stats = grand.transmit(bits, snr_db, seed)        # bits: (B, T, C) torch {0,1}

The forward value of `p1` is exactly what the coded link + ORBGRAND delivers to the
semantic decoder (decoded blocks: exact bits; abandoned blocks: hard decisions or channel
posteriors). The code, the channel and the decoder are not differentiable, so the gradient
to the encoder needs a surrogate (`grad`):

  * 'soft' (default): the Jacobian of the uncoded soft channel. The same bits are also
    pushed through experiments.torch_channel.qpsk_channel_torch with the same channel and
    CSI settings (fresh noise draw) and only its gradient is kept:
        p1 = p1_grand + (p1_soft - stop_gradient(p1_soft))
    This is the gradient path of the benchmark (no-GRAND) run, bits -> QPSK -> LLR ->
    sigmoid, so a GRAND run and a benchmark run differ only in the forward signal the
    decoder sees, not in how the encoder is trained.
  * 'identity': plain straight-through, d p1 / d bits = I (the Gumbel bit mapper's trick).

The numpy decoding runs in a process pool, one image per job, so a batch of 16 images uses
up to 16 cores. Everything else (channel model, CSI errors, code, budget accounting,
fallback, interleaver) is identical to experiments.channel_coding.CodedLink, which is what
evaluation uses; budget_mode='level' is the query accounting of Duffy's MATLAB code.
"""
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.channel_coding import CodedLink, make_code  # noqa: E402
from experiments.torch_channel import qpsk_channel_torch  # noqa: E402

_LINK = None
_LINK_KEY = None


def _get_link(code_spec, max_queries, variant, budget_mode):
    global _LINK, _LINK_KEY
    key = (code_spec, int(max_queries), variant, budget_mode)
    if _LINK is None or _LINK_KEY != key:
        G, H = make_code(code_spec)
        _LINK = CodedLink(G, H, max_queries=max_queries, variant=variant, budget_mode=budget_mode)
        _LINK_KEY = key
    return _LINK


def _init_worker(code_spec, max_queries, variant, budget_mode):
    _get_link(code_spec, max_queries, variant, budget_mode)


def _decode_one(job):
    (bits_np, snr_db, seed, code_spec, max_queries, variant, budget_mode,
     fallback, channel, fade_block, interleave, snr_err_db, h_err_var) = job
    link = _get_link(code_spec, max_queries, variant, budget_mode)
    p1, stats = link.transmit(bits_np, snr_db, np.random.default_rng(seed), fallback=fallback, channel=channel,
                              fade_block=fade_block, interleave=interleave, snr_err_db=snr_err_db,
                              h_err_var=h_err_var)
    return p1.astype(np.float32), stats


STAT_KEYS = ("ber_channel", "ber_after", "bler_after", "abandon_rate", "miscorrect_rate", "avg_queries",
             "decode_time_s")


class _StraightThrough(torch.autograd.Function):
    """forward: the receiver output p1, exactly; backward: gradient passed to `bits` unchanged."""

    @staticmethod
    def forward(ctx, bits, p1):
        return p1

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class GrandChannel:
    def __init__(self, code_spec="crc:0x9eb2:128:112", max_queries=2000, variant="1line", fallback="soft",
                 channel="awgn", fade_block=8, interleave=True, workers=0, budget_mode="level", grad="soft"):
        if grad not in ("soft", "identity"):
            raise ValueError("grad must be 'soft' or 'identity'")
        self.code_spec = code_spec
        self.max_queries = int(max_queries)
        self.variant = variant
        self.budget_mode = budget_mode
        self.fallback = fallback
        self.channel = channel
        self.fade_block = int(fade_block)
        self.interleave = bool(interleave)
        self.workers = int(workers)
        self.grad = grad
        self.pool = None
        link = _get_link(code_spec, max_queries, variant, budget_mode)
        self.n, self.k = link.n, link.k

    @property
    def rate(self):
        return self.k / self.n

    def _ensure_pool(self):
        if self.workers > 0 and self.pool is None:
            # never fork a process that has initialised CUDA and started DataLoader workers
            ctx = mp.get_context("spawn" if os.name == "nt" else "forkserver")
            self.pool = ctx.Pool(self.workers, initializer=_init_worker,
                                 initargs=(self.code_spec, self.max_queries, self.variant, self.budget_mode))
        return self.pool

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def _soft_surrogate(self, bits, snrs, channel, fade_block, snr_err_db, h_err_var, generator):
        kw = dict(channel=channel, fade_block=fade_block, snr_err_db=snr_err_db, h_err_var=h_err_var,
                  generator=generator)
        if len(set(snrs)) == 1:
            return qpsk_channel_torch(bits, snrs[0], **kw)[1]
        return torch.cat([qpsk_channel_torch(bits[i:i + 1], s, **kw)[1] for i, s in enumerate(snrs)], dim=0)

    def transmit(self, bits, snr_db, seed, snr_err_db=0.0, h_err_var=0.0, channel=None, fade_block=None,
                 generator=None):
        """bits: (B, ...) torch tensor of {0,1} (float, may require grad). snr_db: float or
        sequence of B floats. generator: torch.Generator for the surrogate soft channel
        (grad='soft'; None = global RNG). Returns (p1 with surrogate gradient, mean stats)."""
        B = bits.shape[0]
        snrs = [float(snr_db)] * B if np.isscalar(snr_db) else [float(s) for s in snr_db]
        channel = channel or self.channel
        fade_block = self.fade_block if fade_block is None else int(fade_block)
        bits_np = bits.detach().cpu().numpy().astype(np.uint8).reshape(B, -1)
        jobs = [(bits_np[i], snrs[i], int(seed) + i, self.code_spec, self.max_queries, self.variant,
                 self.budget_mode, self.fallback, channel, fade_block, self.interleave,
                 float(snr_err_db), float(h_err_var)) for i in range(B)]
        t0 = time.time()
        pool = self._ensure_pool()
        if pool is not None:
            res = pool.map(_decode_one, jobs, chunksize=1)
        else:
            res = [_decode_one(j) for j in jobs]
        p1 = torch.from_numpy(np.stack([r[0] for r in res])).reshape(bits.shape).to(bits.device, bits.dtype)
        stats = {k: float(np.mean([r[1][k] for r in res])) for k in STAT_KEYS}
        stats["wall_time_s"] = time.time() - t0
        if not bits.requires_grad:
            return p1, stats
        if self.grad == "identity":
            return _StraightThrough.apply(bits, p1), stats
        # value = GRAND output exactly (x - x.detach() is exactly 0); gradient = soft-channel Jacobian
        p1_soft = self._soft_surrogate(bits, snrs, channel, fade_block, snr_err_db, h_err_var, generator)
        return p1.detach() + (p1_soft - p1_soft.detach()), stats
