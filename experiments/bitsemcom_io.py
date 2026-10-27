"""
Thin wrapper around the BitSemCom (WITT + bit mapper) model for the GRAND experiments.

Builds the DIV2K/Kodak configuration used in main.py without importing main.py (which
needs lpips/thop/tensorboardX and a CUDA device at import time).
"""
import glob
import os
import sys
import types

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from network.network import WITT, QPSK_soft  # noqa: E402
from loss.distortion import MS_SSIM  # noqa: E402
from experiments.metrics import psnr_255  # noqa: E402


def find_checkpoint(C):
    cands = sorted(glob.glob(os.path.join(ROOT, "Pretrain", "*C%d_*.model" % C)))
    if not cands:
        raise FileNotFoundError("no checkpoint for C=%d in Pretrain/" % C)
    return cands[-1]


def make_args(C):
    return types.SimpleNamespace(
        tau=1.0, tau_lg=1.0, C=C, model="WITT", multiple_snr="10", channel_type="awgn",
        distortion_metric="MSE", trainset="DIV2K", logger=None,
    )


def make_config(C, device):
    cfg = types.SimpleNamespace()
    cfg.logger = None
    cfg.pass_channel = True
    cfg.CUDA = device.type == "cuda"
    cfg.device = device
    cfg.norm = False
    cfg.snr_min, cfg.snr_max = -3, 6
    cfg.downsample = 4
    cfg.image_dims = (3, 256, 256)
    cfg.encoder_kwargs = dict(
        img_size=(256, 256), patch_size=2, in_chans=3,
        embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10],
        C=C, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, patch_norm=True,
    )
    cfg.decoder_kwargs = dict(
        img_size=(256, 256),
        embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2], num_heads=[10, 8, 6, 4],
        C=C, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, patch_norm=True,
    )
    return cfg


def build_model(C, device="cuda"):
    """Fresh (randomly initialised) WITT + bit mapper with the DIV2K/Kodak configuration."""
    device = torch.device(device)
    return WITT(make_args(C), make_config(C, device)).to(device)


def load_model(C, device="cuda", checkpoint=None, verbose=True):
    """checkpoint: path to a state_dict (.model) or a trainer checkpoint (ckpt_last.pt);
    None -> newest Pretrain/*C{C}_*.model; 'none' -> random initialisation."""
    device = torch.device(device)
    net = build_model(C, device)
    if checkpoint == "none":
        net.eval()
        return net
    ckpt = checkpoint or find_checkpoint(C)
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if verbose:
        print("loaded %s  (missing=%d unexpected=%d)" % (os.path.relpath(ckpt, ROOT), len(missing), len(unexpected)))
    if missing or unexpected:
        raise RuntimeError("checkpoint/model mismatch: missing=%s unexpected=%s" % (missing[:5], unexpected[:5]))
    net.eval()
    return net


def load_images(folder, limit=None):
    paths = sorted(glob.glob(os.path.join(folder, "*.png")) + glob.glob(os.path.join(folder, "*.jpg")))
    if limit:
        paths = paths[:limit]
    out = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        w2, h2 = w - w % 128, h - h % 128
        if (w2, h2) != (w, h):
            l, t = (w - w2) // 2, (h - h2) // 2
            im = im.crop((l, t, l + w2, t + h2))
        x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        out.append((os.path.basename(p), x))
    return out


@torch.no_grad()
def encode_bits(net, img, snr_db, seed=None):
    """Semantic encoder + bit mapper. Returns float tensor of 0/1 bits, shape (1, tokens, C)."""
    if seed is not None:
        torch.manual_seed(seed)
    _, _, bits = net.Encoder(img, float(snr_db))
    return bits


@torch.no_grad()
def decode_probs(net, p1, snr_db):
    """Semantic decoder from P(bit=1) tensor (1, tokens, C) -> image in [0,1]."""
    recon = net.decoder(p1, float(snr_db), net.model)
    return recon.clamp(0.0, 1.0)


@torch.no_grad()
def qpsk_soft_original(bits, snr_db):
    """BitSemCom's own QPSK + AWGN + soft demod; returns P(bit=1) with the shape of bits."""
    _, p1 = QPSK_soft(bits, float(snr_db))
    return p1


class Metrics:
    def __init__(self, device):
        self.msssim = MS_SSIM(data_range=1.0, levels=4, channel=3).to(device)

    @torch.no_grad()
    def __call__(self, img, recon):
        psnr = psnr_255(img, recon)
        ms = 1.0 - self.msssim(img, recon).mean().item()
        return psnr, ms
