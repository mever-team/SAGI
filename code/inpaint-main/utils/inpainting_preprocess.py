import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Union

def dilate_mask_per(mask, expansion_percent):
    """
    Dilates the mask until its area has increased by the specified percentage.
    
    Parameters:
    - mask: A binary mask as a numpy array (non-zero for the mask).
    - expansion_percent: Desired expansion in percentage.
    
    Returns:
    - Dilated mask as a numpy array.
    """
    mask = np.array(mask)

    original_area = np.sum(mask > 0)
    target_area = original_area * (1 + expansion_percent / 100)
    total_area = mask.size

    if target_area >= total_area:
        # Create and return a fully white mask
        white_mask = np.ones_like(mask)
        return white_mask
    
    # Structuring element for dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    dilated_mask = mask.copy()
    while np.sum(dilated_mask > 0) < target_area:
        dilated_mask = cv2.dilate(dilated_mask, kernel, iterations=1)
    
    return dilated_mask

def load_img_to_array(img_p):
    img = Image.open(img_p)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    return np.array(img)


def save_array_to_img(img_arr, img_p):
    Image.fromarray(img_arr.astype(np.uint8)).save(img_p)


def dilate_mask(mask, dilate_factor=15):
    mask = mask.astype(np.uint8)
    mask = cv2.dilate(
        mask,
        np.ones((dilate_factor, dilate_factor), np.uint8),
        iterations=1
    )
    return mask

def painter_resize(image: Image, size: Union[int, Tuple[int, int]], resample=Image.BICUBIC):
    if isinstance(size, int):
        w, h = image.size
        aspect_ratio = w / h
        size = (min(size, int(size * aspect_ratio)),
                min(size, int(size / aspect_ratio)))
    return image.resize(size, resample=resample)

def preprocess_powerpaint(image_path, mask_path, expansion_percent):
    init_image = cv2.imread(image_path)[:, :, ::-1]
    mask_image = 1. * (cv2.imread(mask_path).sum(-1) > 255)[:, :, np.newaxis]

    mask_image = dilate_mask_per(mask_image, expansion_percent)
    if len(mask_image.shape) < 3:
        mask_image = mask_image[:, :, np.newaxis]

    new_height, new_width, _ = init_image.shape
    mask_image = cv2.resize(mask_image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    if mask_image.ndim == 2:
        mask_image = mask_image[:, :, np.newaxis]

    init_image = init_image * (1 - mask_image)
    init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")
    mask_image = Image.fromarray(mask_image.astype(np.uint8).repeat(3, -1) * 255).convert("RGB")

    return init_image, mask_image

def preprocess_brushnet(image_path, mask_path, expansion_percent, diffusion_model):
    if diffusion_model in ["sdxl", "juggernaut"]:
        is_sdxl = True
    else:
        is_sdxl = False
    
    init_image = cv2.imread(image_path)[:, :, ::-1]
    mask_image = 1. * (cv2.imread(mask_path).sum(-1) > 255)[:, :, np.newaxis]
    mask_image = dilate_mask_per(mask_image, expansion_percent)
    if len(mask_image.shape) < 3:
        mask_image = mask_image[:, :, np.newaxis]
    new_height, new_width, _ = init_image.shape
    mask_image = cv2.resize(mask_image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    if mask_image.ndim == 2:
        mask_image = mask_image[:, :, np.newaxis]
        
    if not is_sdxl:
        init_image = init_image * (1 - mask_image)
        init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")
        mask_image = Image.fromarray(mask_image.astype(np.uint8).repeat(3, -1) * 255).convert("RGB")
    else:
        h, w, _ = init_image.shape
        if w < h:
            scale = 1024 / w
        else:
            scale = 1024 / h
        new_h = int(h * scale)
        new_w = int(w * scale)

        init_image = cv2.resize(init_image, (new_w, new_h))
        mask_image = cv2.resize(mask_image, (new_w, new_h))[:, :, np.newaxis]

        init_image = init_image * (1 - mask_image)
        init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")
        mask_image = Image.fromarray(mask_image.astype(np.uint8).repeat(3, -1) * 255).convert("RGB")

    return init_image, mask_image

def preprocess_controlnet(image_path, mask_path, expansion_percent):
    init_image = cv2.imread(image_path)[:, :, ::-1]
    mask_image = 1. * (cv2.imread(mask_path).sum(-1) > 255)[:, :, np.newaxis]

    mask_image = dilate_mask_per(mask_image, expansion_percent)
    if len(mask_image.shape) < 3:
        mask_image = mask_image[:, :, np.newaxis]

    new_height, new_width, _ = init_image.shape
    mask_image = cv2.resize(mask_image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    if mask_image.ndim == 2:
        mask_image = mask_image[:, :, np.newaxis]

    init_image = init_image * (1 - mask_image)
    init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")
    mask_image = Image.fromarray(mask_image.astype(np.uint8).repeat(3, -1) * 255).convert("RGB")

    return init_image, mask_image

def preprocess_hdpainter(image_path, mask_path, expansion_percent):
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('RGB')
    mask = mask.resize(image.size)
    
    resized_image = painter_resize(image, 512)
    resized_mask = painter_resize(mask, 512)
    
    resized_mask = dilate_mask_per(resized_mask, expansion_percent)
    resized_mask = Image.fromarray(resized_mask)
    resized_mask = resized_mask.point(lambda p: p > 128 and 255)
    
    init_image = resized_image
    mask_image = resized_mask
    
    return init_image, mask_image

def preprocess_inpaint_anything(image_path, mask_path, expansion_percent):
    init_image = load_img_to_array(image_path)
    
    if len(init_image.shape) == 2:  # Check if the image is grayscale
        init_image = np.stack((init_image,) * 3, axis=-1)  # Stack the grayscale image into 3 channels

    
    mask_image = load_img_to_array(mask_path)
    mask_image = np.where(mask_image < 128, 0, 255).astype(np.uint8)
    mask_image = dilate_mask_per(mask_image, expansion_percent)
    
    new_height, new_width, _ = init_image.shape
    mask_image = cv2.resize(mask_image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    
    return init_image, mask_image