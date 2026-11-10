"""
Build and load the BitSemCom model (WITT + bit mapper) without importing main.py.

main.py parses the command line, needs lpips / thop / tensorboardX and a CUDA device as soon
as it is imported, so other scripts cannot reuse it. This module rebuilds the same DIV2K /
Kodak model configuration as the `config` class of main.py and loads checkpoints with a
strict key check (main.py loads with strict=False, which hides a wrong --C).
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
from network.network import WITT  # noqa: E402


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
        print("loaded %s  (missing=%d unexpected=%d)" % (ckpt, len(missing), len(unexpected)))
    if missing or unexpected:
        raise RuntimeError("checkpoint/model mismatch: missing=%s unexpected=%s" % (missing[:5], unexpected[:5]))
    net.eval()
    return net


def load_images(folder, limit=None):
    """All .png / .jpg images of a folder as (name, tensor (1,3,H,W) in [0,1]); H and W are
    centre-cropped to multiples of 128, as the test loader of data/datasets.py does."""
    paths = sorted(glob.glob(os.path.join(folder, "*.png")) + glob.glob(os.path.join(folder, "*.jpg")))
    if not paths:
        raise FileNotFoundError("no .png / .jpg images in %s" % folder)
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
