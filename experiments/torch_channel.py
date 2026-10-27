"""
Differentiable torch version of the QPSK link used for training.

Mirrors experiments.channel_coding.qpsk_channel (numpy, evaluation) including the
imperfect-CSI options, so that a model trained here is evaluated under exactly the
same channel model:

  * channel='awgn'          : y = s + n
  * channel='blockfading'   : one complex Rayleigh gain h ~ CN(0,1) per `fade_block`
                              consecutive symbols (fade_block=1 is fast fading), coherent
                              receiver
  * snr_err_db              : the receiver computes LLRs with the noise variance of
                              (snr_db + snr_err_db); >0 means the receiver is over-confident
  * h_err_var               : the receiver uses h_hat = h + e, e ~ CN(0, h_err_var)

With channel='awgn' and no CSI error this is numerically identical to network.QPSK_soft
(same symbol mapping, same noise draw order, same LLR formula), so the "benchmark"
training sees the original BitSemCom channel.

Gradients flow to `bits` through the symbol mapping s = (1 - 2 b) / sqrt(2), as in the
original code.
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.channel_coding import SQRT2, noise_var  # noqa: E402


def qpsk_channel_torch(bits, snr_db, channel="awgn", fade_block=8, snr_err_db=0.0, h_err_var=0.0,
                       generator=None):
    """bits: (B, ..., C) float tensor with values in {0,1} (last dim even).
    Returns (llr, p1) with the shape of bits. Positive LLR = bit 0, p1 = P(bit=1)."""
    shape = bits.shape
    B = shape[0]
    flat = bits.reshape(B, -1).float()
    if flat.shape[1] % 2:
        raise ValueError("number of bits per sample must be even for QPSK")
    s = (1.0 - 2.0 * flat).reshape(B, -1, 2) / SQRT2  # (B, nsym, 2)
    nsym = s.shape[1]
    nv = noise_var(snr_db)
    nv_hat = noise_var(snr_db + snr_err_db)
    noise = math.sqrt(nv) * torch.randn(s.shape, device=s.device, dtype=s.dtype, generator=generator)

    if channel == "awgn":
        r = s + noise
        llr = SQRT2 * r / nv_hat
    elif channel == "blockfading":
        fb = int(fade_block)
        if fb < 1:
            raise ValueError("blockfading needs fade_block >= 1 symbols (1 = fast fading), got %r" % (fade_block,))
        nblk = (nsym + fb - 1) // fb
        hr = torch.randn(B, nblk, 2, device=s.device, dtype=s.dtype, generator=generator) / SQRT2
        h = torch.complex(hr[..., 0], hr[..., 1])
        h = h.repeat_interleave(fb, dim=1)[:, :nsym]
        if h_err_var > 0:
            er = torch.randn(B, nblk, 2, device=s.device, dtype=s.dtype, generator=generator) * math.sqrt(h_err_var / 2.0)
            e = torch.complex(er[..., 0], er[..., 1]).repeat_interleave(fb, dim=1)[:, :nsym]
            h_hat = h + e
        else:
            h_hat = h
        sc = torch.complex(s[..., 0], s[..., 1])
        nc = torch.complex(noise[..., 0], noise[..., 1])
        rc = h * sc + nc
        gain = h_hat.abs().clamp_min(1e-12)
        y = rc * torch.conj(h_hat) / gain
        r = torch.stack([y.real, y.imag], dim=-1)
        llr = SQRT2 * gain.unsqueeze(-1) * r / nv_hat
    else:
        raise ValueError(channel)
    llr = llr.reshape(shape)
    p1 = torch.sigmoid(-llr)
    return llr, p1
