# powerpaint_anything_utils.py
from ..config import change_dir
change_dir("powerpaint")

import torch
from transformers import CLIPTextModel
from model.BrushNet_CA import BrushNetModel
from pipeline.pipeline_PowerPaint_Brushnet_CA import StableDiffusionPowerPaintBrushNetPipeline
from diffusers import UniPCMultistepScheduler
from safetensors.torch import load_model
from model.diffusers_c.models import UNet2DConditionModel
from utils.utils import TokenizerWrapper, add_tokens

class PowerPaintInitializer:
    def __init__(self, base_model_path):
        self.base_model_path = base_model_path
        self.pipe = self.initialize_powerpaint()
        self.current_control = 'canny'
        
    def initialize_powerpaint(self):
        dtype = torch.float16
        unet = UNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="unet", revision=None, torch_dtype=dtype
        )
        text_encoder_brushnet = CLIPTextModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="text_encoder", revision=None, torch_dtype=dtype
        )
        brushnet = BrushNetModel.from_unet(unet)

        pipe = StableDiffusionPowerPaintBrushNetPipeline.from_pretrained(
            self.base_model_path, brushnet=brushnet, text_encoder_brushnet=text_encoder_brushnet, torch_dtype=dtype, low_cpu_mem_usage=False, safety_checker=None
        )

        pipe.unet = UNet2DConditionModel.from_pretrained(
            self.base_model_path, subfolder="unet", revision=None, torch_dtype=dtype
        )
        pipe.tokenizer = TokenizerWrapper(from_pretrained=self.base_model_path, subfolder="tokenizer", revision=None)

        add_tokens(
            tokenizer=pipe.tokenizer,
            text_encoder=pipe.text_encoder_brushnet,
            placeholder_tokens=['P_ctxt', 'P_shape', 'P_obj'],
            initialize_tokens=['a', 'a', 'a'],
            num_vectors_per_token=10
        )

        load_model(pipe.brushnet, "./PowerPaint_v2/PowerPaint_Brushnet/diffusion_pytorch_model.safetensors")
        pipe.text_encoder_brushnet.load_state_dict(torch.load("./PowerPaint_v2/PowerPaint_Brushnet/pytorch_model.bin"), strict=False)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()

        return pipe
