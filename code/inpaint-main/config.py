# config.py
import os

base_path = "/path/to/inpainting/repos" 

def change_dir(inpainting_model):
    if inpainting_model == "powerpaint":
        os.chdir(f'{base_path}/PowerPaint')
    elif inpainting_model == 'brushnet':
        os.chdir(f'{base_path}/BrushNet2/examples/brushnet')
    elif inpainting_model == 'controlnet':
        os.chdir(f'{base_path}/ControlNetInpaint')
    elif inpainting_model == 'inpaintanything':
        os.chdir(f'{base_path}/Inpaint-Anything-main')
    elif inpainting_model == 'hdpainter':
        os.chdir(f'{base_path}/HD-Painter-main')
    else:
        raise ValueError(f"Unsupported inpainting model: {inpainting_model}")   