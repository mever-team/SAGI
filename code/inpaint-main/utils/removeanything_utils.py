# inpaint_anything_utils.py
from ..config import change_dir
change_dir("inpaintanything")

from lama_inpaint import inpaint_img_with_lama

class InpaintAnythingInitializer:
    def __init__(self):
        self.pipe = self.initialize_inpaint_anything()

    def initialize_inpaint_anything(self):
        device = 'cuda'
        pipe = lambda image, mask: inpaint_img_with_lama(image, mask, "./lama/configs/prediction/default.yaml", "./pretrained_models/big-lama", device=device)
        
        
        return pipe
