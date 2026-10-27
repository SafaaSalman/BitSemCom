"""
Image quality metrics for the thesis: PSNR, MS-SSIM, LPIPS (perceptual) and CLIP
image-image cosine similarity (semantic).

LPIPS and CLIP are optional: if `lpips` / `open_clip_torch` are not installed (or
`use_lpips=False` / `use_clip=False`), those keys are simply absent from the result.

    met = Metrics(device)
    d = met(img, recon)   # img, recon: (1,3,H,W) in [0,1]
    d -> {'psnr': ..., 'msssim': ..., 'lpips': ..., 'clip': ...}

Conventions: higher is better for psnr, msssim, clip; lower is better for lpips.
"""
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loss.distortion import MS_SSIM  # noqa: E402

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def psnr_from_mse(mse):
    """PSNR (dB) from a mean squared error on the 0..255 scale; 100 dB for mse == 0."""
    return 10 * math.log10(255.0 ** 2 / mse) if mse > 0 else 100.0


def psnr_255(img, recon):
    return psnr_from_mse(((img * 255.0 - recon * 255.0) ** 2).mean().item())


class Metrics:
    def __init__(self, device, use_lpips=True, use_clip=True, clip_model="ViT-B-32", clip_pretrained="openai",
                 verbose=True):
        self.device = torch.device(device)
        self.msssim = MS_SSIM(data_range=1.0, levels=4, channel=3).to(self.device)
        self.lpips = None
        self.clip = None
        if use_lpips:
            try:
                import lpips  # noqa: F401
                self.lpips = lpips.LPIPS(net="alex", verbose=False).to(self.device).eval()
                for p in self.lpips.parameters():
                    p.requires_grad_(False)
            except Exception as e:  # pragma: no cover
                if verbose:
                    print("[metrics] LPIPS unavailable (%s); skipping" % e)
        if use_clip:
            try:
                import open_clip
                model, _, _ = open_clip.create_model_and_transforms(clip_model, pretrained=clip_pretrained)
                self.clip = model.to(self.device).eval()
                for p in self.clip.parameters():
                    p.requires_grad_(False)
                self._clip_mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
                self._clip_std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)
            except Exception as e:  # pragma: no cover
                if verbose:
                    print("[metrics] CLIP unavailable (%s); skipping" % e)

    @property
    def names(self):
        n = ["psnr", "msssim"]
        if self.lpips is not None:
            n.append("lpips")
        if self.clip is not None:
            n.append("clip")
        return n

    @torch.no_grad()
    def _clip_embed(self, x):
        x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        x = (x - self._clip_mean) / self._clip_std
        f = self.clip.encode_image(x).float()
        return F.normalize(f, dim=-1)

    @torch.no_grad()
    def __call__(self, img, recon):
        img = img.to(self.device).float()
        recon = recon.to(self.device).float().clamp(0, 1)
        out = {"psnr": psnr_255(img, recon),
               "msssim": 1.0 - self.msssim(img, recon).mean().item()}
        if self.lpips is not None:
            out["lpips"] = self.lpips(recon * 2 - 1, img * 2 - 1).mean().item()
        if self.clip is not None:
            out["clip"] = (self._clip_embed(img) * self._clip_embed(recon)).sum(-1).mean().item()
        return out
