import re
import torch
from .tools import VariantSupport
from .base_node import NODE_NAME, LatentNode
import comfy.model_management

MAX_RESOLUTION=32768

# from  https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475
def slerp(val, low, high):
    dims = low.shape

    #flatten to batches
    low = low.reshape(dims[0], -1)
    high = high.reshape(dims[0], -1)

    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)

    # in case we divide by zero
    low_norm[low_norm != low_norm] = 0.0
    high_norm[high_norm != high_norm] = 0.0

    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res.reshape(dims)

@VariantSupport()
class SeedInterpNoise(LatentNode):
    """
    Produce a batch of noise tensors whose seeds advance by +1 after
    every (interp_steps + 1) frames, with SLERP interpolation between
    the two anchor seeds.

    Example (start_seed=1, interp_steps=2, frames=7):
        seeds: 1   1--2   1--2   2   2--3   2--3   3
                ε0  ε1    ε2    ε3  ε4     ε5     ε6
                ▲   ▲          ▲           ▲
                |   |__________|           |
                |______ segment 1 _____|   |___ seg 2 …
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "source": (["CPU", "GPU"],),
                    "start_seed": ("INT",  {"default": 0, "min": 0,
                                            "max": 0xffffffffffffffff}),
                    "frames":     ("INT",  {"default": 8, "min": 1,
                                            "max": 9999999}),
                    "interp_steps": ("INT", {"default": 1, "min": 0,
                                             "max": 1024}),
                    "width":  ("INT", {"default": 512, "min": 64,
                                       "max": MAX_RESOLUTION, "step": 8}),
                    "height": ("INT", {"default": 512, "min": 64,
                                       "max": MAX_RESOLUTION, "step": 8}),
               }}

    RETURN_TYPES = ("LATENT",)          # matches NoisyLatentImage
    FUNCTION = "build"

    def build(self, source, start_seed, frames, interp_steps,
              width, height):
        device = "cpu" if source == "CPU" \
                 else comfy.model_management.get_torch_device()
        c, h, w = 4, height // 8, width // 8

        # pre-allocate output tensor
        batch = torch.empty((frames, c, h, w),
                            dtype=torch.float32, device=device)

        # helper: epsilon(seed)
        def eps(seed):
            g = torch.Generator(device).manual_seed(seed)
            return torch.randn((c, h, w), generator=g, device=device)

        seg_len = interp_steps + 1      # ε_a + k in-betweens + ε_b
        cur_seed = start_seed
        frame_idx = 0

        while frame_idx < frames:
            eps_a = eps(cur_seed)
            eps_b = eps(cur_seed + 1)

            # anchor A
            batch[frame_idx] = eps_a
            frame_idx += 1
            if frame_idx >= frames:
                break

            # interpolated frames
            for k in range(1, seg_len):
                if frame_idx >= frames:
                    break
                t = k / seg_len
                batch[frame_idx] = slerp(t, eps_a, eps_b)
                frame_idx += 1

            # anchor B (start of next segment) only if we still need frames
            cur_seed += 1

        return ({"samples": batch.cpu()}, )

LATENT_NODE_CLASS_MAPPINGS = {
    "SeedInterpNoise": SeedInterpNoise,
}

LATENT_NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedInterpNoise": f"Seed Interp Noise | {NODE_NAME}",
}