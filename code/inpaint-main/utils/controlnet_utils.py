#controlnet_utils.py
from ..config import change_dir
change_dir("controlnet")

from diffusers import StableDiffusionInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from src.pipeline_stable_diffusion_controlnet_inpaint import *

class ControlNetInitializer:
    def __init__(self, base_model_path):
        self.base_model_path = base_model_path
        self.pipe = self.initialize_controlnet()

    def initialize_controlnet(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", controlnet=controlnet, torch_dtype=torch.float16
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.to('cuda')
        
        return pipe