"""Real HDRUNet architecture loader behind the ResidualAdapter contract.

Architecture follows chxy95/HDRUNet at revision e9bd4b90a27d7531a38b8c97db05feb368b96dc2
(MIT, codes/models/modules/UNet_arch.py and arch_util.py). No weights are bundled or
downloaded: the caller supplies a checkpoint path and an independently obtained SHA-256.
The model learns NTIRE2021 single-frame HDR reconstruction, not Sony calibration, and
its output is never treated as recovered clipped-highlight truth.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from prd_v5.hypothesis import CameraHypothesis
from prd_v5.learned import ResidualAdapter, camera_domain

HDRUNET_REVISION = "e9bd4b90a27d7531a38b8c97db05feb368b96dc2"
HDRUNET_SOURCE = f"https://github.com/chxy95/HDRUNet/tree/{HDRUNET_REVISION}"
HDRUNET_LICENCE = "MIT (chxy95/HDRUNet LICENSE, copyright 2021 XyChen)"
STRIDE = 4  # two stride-2 reductions: tiles are padded to, and halos kept at, multiples of four


def build_hdrunet(torch, nf: int = 64):
    """Instantiate HDRUNet(in_nc=3, out_nc=3, nf=nf, act_type='relu') with upstream key names."""
    nn, functional = torch.nn, torch.nn.functional

    class SFTLayer(nn.Module):
        def __init__(self, in_nc=32, out_nc=64, nf=32):
            super().__init__()
            self.SFT_scale_conv0 = nn.Conv2d(in_nc, nf, 1)
            self.SFT_scale_conv1 = nn.Conv2d(nf, out_nc, 1)
            self.SFT_shift_conv0 = nn.Conv2d(in_nc, nf, 1)
            self.SFT_shift_conv1 = nn.Conv2d(nf, out_nc, 1)

        def forward(self, x):
            scale = self.SFT_scale_conv1(functional.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1))
            shift = self.SFT_shift_conv1(functional.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1))
            return x[0] * (scale + 1) + shift

    class ResBlockWithSFT(nn.Module):
        def __init__(self, nf=64):
            super().__init__()
            self.sft1 = SFTLayer(in_nc=32, out_nc=64, nf=32)
            self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1)
            self.sft2 = SFTLayer(in_nc=32, out_nc=64, nf=32)
            self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1)

        def forward(self, x):
            fea = functional.relu(self.conv1(self.sft1(x)))
            fea = self.conv2(self.sft2((fea, x[1])))
            return (x[0] + fea, x[1])

    def trunk(count):
        return nn.Sequential(*[ResBlockWithSFT(nf=nf) for _ in range(count)])

    class HDRUNet(nn.Module):
        def __init__(self, in_nc=3, out_nc=3):
            super().__init__()
            self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
            self.SFT_layer1 = SFTLayer()
            self.HR_conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.down_conv1 = nn.Conv2d(nf, nf, 3, 2, 1)
            self.down_conv2 = nn.Conv2d(nf, nf, 3, 2, 1)
            self.recon_trunk1 = trunk(2)
            self.recon_trunk2 = trunk(8)
            self.recon_trunk3 = trunk(2)
            self.up_conv1 = nn.Sequential(nn.Conv2d(nf, nf * 4, 3, 1, 1), nn.PixelShuffle(2))
            self.up_conv2 = nn.Sequential(nn.Conv2d(nf, nf * 4, 3, 1, 1), nn.PixelShuffle(2))
            self.SFT_layer2 = SFTLayer()
            self.HR_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
            cond_nf = 64
            self.cond_first = nn.Sequential(
                nn.Conv2d(3, cond_nf, 3, 1, 1),
                nn.LeakyReLU(0.1, True),
                nn.Conv2d(cond_nf, cond_nf, 1),
                nn.LeakyReLU(0.1, True),
                nn.Conv2d(cond_nf, cond_nf, 1),
                nn.LeakyReLU(0.1, True),
            )
            self.CondNet1 = nn.Sequential(
                nn.Conv2d(cond_nf, cond_nf, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(cond_nf, 32, 1)
            )
            self.CondNet2 = nn.Sequential(
                nn.Conv2d(cond_nf, cond_nf, 3, 2, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(cond_nf, 32, 1)
            )
            self.CondNet3 = nn.Sequential(
                nn.Conv2d(cond_nf, cond_nf, 3, 2, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(cond_nf, 32, 3, 2, 1)
            )
            self.mask_est = nn.Sequential(
                nn.Conv2d(in_nc, nf, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(nf, nf, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(nf, nf, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(nf, out_nc, 1),
            )
            self.act = nn.ReLU(inplace=False)

        def forward(self, x):
            mask = self.mask_est(x[0])
            cond = self.cond_first(x[1])
            cond1, cond2, cond3 = self.CondNet1(cond), self.CondNet2(cond), self.CondNet3(cond)
            fea0 = self.act(self.conv_first(x[0]))
            fea0 = self.SFT_layer1((fea0, cond1))
            fea0 = self.act(self.HR_conv1(fea0))
            fea1 = self.act(self.down_conv1(fea0))
            fea1, _ = self.recon_trunk1((fea1, cond2))
            fea2 = self.act(self.down_conv2(fea1))
            out, _ = self.recon_trunk2((fea2, cond3))
            out = out + fea2
            out = self.act(self.up_conv1(out)) + fea1
            out, _ = self.recon_trunk3((out, cond2))
            out = self.act(self.up_conv2(out)) + fea0
            out = self.SFT_layer2((out, cond1))
            out = self.act(self.HR_conv2(out))
            return mask * x[0] + self.conv_last(out)

    return HDRUNet()


def _srgb_to_linear(encoded):
    encoded = np.clip(encoded, 0.0, 1.0)
    return np.where(encoded <= 0.04045, encoded / 12.92, ((encoded + 0.055) / 1.055) ** 2.4)


def make_predictor(model, torch, *, device: str, max_delta: float):
    """Tile predictor: encoded-domain HDRUNet delta mapped to a linear camera residual.

    The (img, cond) pair is the same encoded sRGB tile, as in upstream LQ_condition_dataset.
    Output minus input, in encoded units, becomes a linear-light difference in
    normalized units (1.0 = camera.saturation_e), matching the clean buffer it is added to.
    Confidence falls linearly to zero as the per-pixel encoded change approaches `max_delta`;
    the network's own mask is not a calibrated confidence.
    """

    def predict(encoded, clean, origin, camera):
        height, width = encoded.shape[:2]
        pad_h, pad_w = -height % STRIDE, -width % STRIDE
        padded = np.pad(np.clip(encoded, 0.0, 1.0), ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        tensor = torch.from_numpy(np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32)).to(device)
        with torch.no_grad():
            output = model((tensor, tensor))
        out = np.asarray(output.float().cpu().numpy())[0].transpose(1, 2, 0)[:height, :width]
        if not np.isfinite(out).all():
            raise ValueError("HDRUNet produced nonfinite output")
        base = np.clip(encoded, 0.0, 1.0).astype(np.float64)
        residual = _srgb_to_linear(np.clip(out, 0.0, 1.0)) - _srgb_to_linear(base)
        change = np.abs(np.clip(out, 0.0, 1.0) - base).max(axis=-1)
        return residual.astype(np.float32), np.clip(1.0 - change / max_delta, 0.0, 1.0)

    return predict


def load_hdrunet_adapter(
    checkpoint_path: str | Path,
    expected_sha256: str,
    camera: CameraHypothesis,
    *,
    licence_accepted: bool = False,
    experimental_accepted: bool = False,
    device: str = "cuda",
    tile: int = 1024,
    halo: int = 128,
    max_delta: float = 0.25,
    torch_module=None,
    source: str = HDRUNET_SOURCE,
    licence: str = HDRUNET_LICENCE,
) -> ResidualAdapter:
    """Verify bytes, load strictly with weights_only, and wrap as a ResidualAdapter."""
    if tile % STRIDE or halo % STRIDE or tile < STRIDE:
        raise ValueError(f"HDRUNet tile and halo must be multiples of {STRIDE}")
    if not 0 < max_delta <= 1:
        raise ValueError("max_delta must be in (0, 1]")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("expected_sha256 must be a 64 hex digit string supplied by the caller")
    digest = hashlib.sha256()
    with Path(checkpoint_path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    if digest.hexdigest() != expected_sha256.lower():
        raise ValueError("HDRUNet checkpoint SHA-256 mismatch; refusing to unpickle")
    torch = torch_module
    if torch is None:
        import torch

    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state and "conv_first.weight" not in state:
        state = state["state_dict"]
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model = build_hdrunet(torch)
    model.load_state_dict(state, strict=True)  # rejects missing, unexpected and mis-shaped keys
    model = model.to(device).eval()
    return ResidualAdapter(
        predict=make_predictor(model, torch, device=device, max_delta=max_delta),
        checkpoint_path=str(checkpoint_path),
        sha256=expected_sha256.lower(),
        camera_sha256=camera_domain(camera),
        source=source,
        licence=licence,
        licence_accepted=licence_accepted,
        experimental_accepted=experimental_accepted,
        tile=tile,
        halo=halo,
    )
