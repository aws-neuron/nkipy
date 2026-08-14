"""On-device SD VAE decoder for PixArt.

Compiles the ``vae_decode`` kernel and runs the convolutional decoder on
Trainium. Runs in fp32 (the decoder is numerically sensitive and it's a
one-shot at the end of sampling). Conv is the weakest path on the PE array, so
this is correctness-first, not a perf win.
"""

import copy
import time

import numpy as np
import torch
from config import Config
from kernels.vae import vae_decode
from nkipy.runtime import DeviceKernel, DeviceTensor

BUILD_DIR = "./build"


class VAEDecoder:
    def __init__(self, weights, config: Config, latent_size):
        self.config = config
        self.latent_size = latent_size
        # VAE weights are fp32
        self.device_weights = {
            k: DeviceTensor.from_numpy(v.numpy().astype(np.float32), name=k)
            for k, v in weights.items()
        }
        self._prepare_kernel()

    def _prepare_kernel(self):
        t = time.time()
        h = self.latent_size
        print(f"[vae] compiling vae_decode (latent={h})")
        cfg = self.config
        latents = DeviceTensor.from_numpy(
            np.empty((1, 4, h, h), dtype=np.float32), "latents"
        )
        weight_kwargs = dict(self.device_weights)
        # VAE runs in fp32 (numerically sensitive); clone config and override dtype
        cfg_fp32 = copy.copy(cfg)
        cfg_fp32.dtype = np.float32
        self._cfg_fp32 = cfg_fp32
        self.kernel = DeviceKernel.compile_and_load(
            vae_decode,
            name=f"vae_decode_{h}",
            latents=latents,
            configs=cfg_fp32,
            build_dir=BUILD_DIR,
            additional_compiler_args=cfg.additional_compiler_args_nkipy,
            **weight_kwargs,
        )
        out_hw = h * (2 ** (len(cfg.vae_block_out_channels) - 1))
        self._out = DeviceTensor.from_numpy(
            np.empty((1, 3, out_hw, out_hw), dtype=np.float32), "pixels"
        )
        print(f"[vae] --> kernel ready in {time.time() - t:.2f}s")

    def decode(self, latents):
        """Args: latents torch (1, 4, h, w). Returns torch (1, 3, H, W) pixels."""
        scaled = (latents / self.config.vae_scaling_factor).to(torch.float32)
        inputs = {
            "latents": DeviceTensor.from_numpy(
                scaled.cpu().numpy().astype(np.float32), "latents"
            ),
        }
        inputs.update(self.device_weights)
        self.kernel(inputs=inputs, outputs={"output0": self._out})
        return self._out.torch().to(torch.float32)
