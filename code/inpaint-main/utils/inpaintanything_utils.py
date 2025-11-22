# inpaint_anything_utils.py
from ..config import change_dir
change_dir("inpaintanything")

import torch
from diffusers import StableDiffusionInpaintPipeline
from stable_diffusion_inpaint import fill_img_with_sd_v2

class InpaintAnythingInitializer:
    def __init__(self):
        self.pipe = self.initialize_inpaint_anything()

    def initialize_inpaint_anything(self):
        device = 'cuda'
        inner_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float32,
        ).to(device)
        outer_pipe = lambda init_image, mask_image, prompt: fill_img_with_sd_v2(init_image, mask_image, prompt, device=device, pipe=inner_pipe)

        #pipe = fill_img_with_sd
        
        return outer_pipe
