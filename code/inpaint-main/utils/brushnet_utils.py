# brushnet_utils.py
from ..config import change_dir
change_dir("brushnet")
from diffusers import BrushNetModel
import torch

class BrushNetInitializer:
    def __init__(self, base_model_path, diffusion_model):
        self.base_model_path = base_model_path
        self.diffusion_model = diffusion_model
        self.pipe = self.initialize_brushnet()

    def initialize_brushnet(self):
        dtype = torch.float16

        if self.diffusion_model in ["realisticvision", "dreamshaper", "epicrealism", "sd1-5"]:
            from diffusers import UniPCMultistepScheduler, StableDiffusionBrushNetPipeline
            
            brushnet_path = "data/ckpt/segmentation_mask_brushnet_ckpt"
            brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=dtype)
            
            pipe = StableDiffusionBrushNetPipeline.from_pretrained(
                self.base_model_path, brushnet=brushnet, torch_dtype=dtype, low_cpu_mem_usage=False
            )
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.enable_model_cpu_offload()
        elif self.diffusion_model in ["juggernaut", "sdxl"]:
            from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLBrushNetPipeline, AutoencoderKL
            
            brushnet_path = "data/ckpt/segmentation_mask_brushnet_ckpt_sdxl_v0"
            brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=dtype)
            
            pipe = StableDiffusionXLBrushNetPipeline.from_pretrained(
                self.base_model_path, brushnet=brushnet, torch_dtype=dtype, low_cpu_mem_usage=False, use_safetensors=True
            )
            pipe.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.enable_model_cpu_offload()
        else:
            raise ValueError("Unsupported diffusion model.")

        return pipe
