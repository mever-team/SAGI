import argparse
import random
import os
import cv2
import numpy as np
import torch
from PIL import Image
import pandas as pd
from tqdm import tqdm
import contextlib
from pathlib import Path

from utils.inpainting_preprocess import (
    preprocess_powerpaint,
    preprocess_brushnet,
    preprocess_controlnet,
    preprocess_hdpainter,
    preprocess_inpaint_anything
)


class NoOutput:
    """Suppress output from external libraries"""
    def write(self, x): pass
    def flush(self): pass


def resize_image(img, dim_limit=1024, is_inpaint_anything=False):
    """Resize image if it exceeds dimension limit"""
    if is_inpaint_anything:
        # For inpaint_anything, work with numpy arrays
        if isinstance(img, Image.Image):
            img = np.array(img)
        height, width = img.shape[:2]
        
        if max(width, height) > dim_limit:
            if width > height:
                new_width = dim_limit
                new_height = int((new_width / width) * height)
            else:
                new_height = dim_limit
                new_width = int((new_height / height) * width)
            img = Image.fromarray(img).resize((new_width, new_height))
            return np.array(img)
        return img
    else:
        # For other models, work with PIL Images
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        width, height = img.size
        
        if max(width, height) > dim_limit:
            if width > height:
                new_width = dim_limit
                new_height = int((new_width / width) * height)
            else:
                new_height = dim_limit
                new_width = int((new_height / height) * width)
            img = img.resize((new_width, new_height))
        return img


def run_powerpaint_inference(pipe, init_image, mask_image, prompt, seed, promptA, promptB, 
                             negative_promptA, negative_promptB, negative_prompt):
    """Run inference for PowerPaint model"""
    img = np.array(init_image)
    W = int(np.shape(img)[0] - np.shape(img)[0] % 8)
    H = int(np.shape(img)[1] - np.shape(img)[1] % 8)
    init_image = init_image.resize((H, W))
    mask_image = mask_image.resize((H, W))
    
    generator = torch.Generator("cuda").manual_seed(seed)
    
    with contextlib.redirect_stdout(NoOutput()), contextlib.redirect_stderr(NoOutput()):
        image = pipe(
            promptA=promptA,
            promptB=promptB,
            promptU=prompt,
            tradoff=1,
            tradoff_nag=1,
            image=init_image,
            mask=mask_image,
            num_inference_steps=50,
            generator=generator,
            brushnet_conditioning_scale=1.0,
            negative_promptA=negative_promptA,
            negative_promptB=negative_promptB,
            negative_promptU=negative_prompt,
            guidance_scale=12,
            width=H,
            height=W,
        ).images[0]
    
    return image


def run_brushnet_inference(pipe, init_image, mask_image, prompt, seed):
    """Run inference for BrushNet model"""
    generator = torch.Generator("cuda").manual_seed(seed)
    
    with contextlib.redirect_stdout(NoOutput()), contextlib.redirect_stderr(NoOutput()):
        image = pipe(
            prompt=prompt,
            image=init_image,
            mask=mask_image,
            num_inference_steps=50,
            generator=generator,
            brushnet_conditioning_scale=1.0
        ).images[0]
    
    return image


def run_controlnet_inference(pipe, init_image, mask_image, prompt, seed):
    """Run inference for ControlNet model"""
    canny_image = cv2.Canny(np.array(init_image), 100, 200)
    canny_image = canny_image[:, :, None]
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
    control_image = Image.fromarray(canny_image)
    
    generator = torch.Generator("cuda").manual_seed(seed)
    
    with contextlib.redirect_stdout(NoOutput()), contextlib.redirect_stderr(NoOutput()):
        image = pipe(
            prompt,
            num_inference_steps=50,
            generator=generator,
            image=init_image,
            control_image=control_image,
            controlnet_conditioning_scale=1,
            mask_image=mask_image
        ).images[0]
    
    return image


def run_hdpainter_inference(pipe, init_image, mask_image, prompt, seed):
    """Run inference for HD-Painter model"""
    with contextlib.redirect_stdout(NoOutput()), contextlib.redirect_stderr(NoOutput()):
        image = pipe(init_image, mask_image, prompt, seed=seed)
    return image


def run_inpaint_anything_inference(pipe, init_image, mask_image, prompt, seed):
    """Run inference for Inpaint-Anything model"""
    torch.manual_seed(seed)
    with contextlib.redirect_stdout(NoOutput()), contextlib.redirect_stderr(NoOutput()):
        image = pipe(init_image, mask_image, prompt)
    return image


def process_single_image(row, inpainting_model, diffusion_model, pipe, preprocess_func, 
                        output_dir, dim_limit, promptA=None, promptB=None, 
                        negative_promptA=None, negative_promptB=None, negative_prompt=None):
    """Process a single image with the specified inpainting model"""
    
    
    image_path = row['image_path']
    mask_path = row['mask_path']
    prompt = row['prompt']
    
    
    expansion_percent = row.get('expansion_percent', 0)
    
    
    seed = random.randint(0, 2**32 - 1)
    
    
    image_name = Path(image_path).stem
    output_filename = f"{image_name}_{inpainting_model}.png"
    output_path = os.path.join(output_dir, output_filename)
    
    
    if inpainting_model == 'brushnet':
        init_image, mask_image = preprocess_func(image_path, mask_path, expansion_percent, diffusion_model)
    else:
        init_image, mask_image = preprocess_func(image_path, mask_path, expansion_percent)
    
    
    cuda_error_count = 0
    for size in [dim_limit, dim_limit // 2, dim_limit // 4]:
        try:
            
            is_inpaint_anything = (inpainting_model == 'inpaintanything')
            init_image_resized = resize_image(init_image, size, is_inpaint_anything)
            mask_image_resized = resize_image(mask_image, size, is_inpaint_anything)
            
            
            if inpainting_model == "powerpaint":
                image = run_powerpaint_inference(
                    pipe, init_image_resized, mask_image_resized, prompt, seed,
                    promptA, promptB, negative_promptA, negative_promptB, negative_prompt
                )
            elif inpainting_model == "brushnet":
                image = run_brushnet_inference(pipe, init_image_resized, mask_image_resized, prompt, seed)
            elif inpainting_model == "controlnet":
                image = run_controlnet_inference(pipe, init_image_resized, mask_image_resized, prompt, seed)
            elif inpainting_model == "hdpainter":
                image = run_hdpainter_inference(pipe, init_image, mask_image, prompt, seed)
            elif inpainting_model == "inpaintanything":
                image = run_inpaint_anything_inference(pipe, init_image_resized, mask_image_resized, prompt, seed)
            else:
                raise ValueError(f"Unsupported inpainting model: {inpainting_model}")
            
            break  # Success, exit retry loop
            
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                cuda_error_count += 1
                if cuda_error_count >= 3:
                    raise RuntimeError(f"Failed to process {image_path} after trying multiple sizes")
                continue
            else:
                raise e
    
    # Convert to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Save output
    image.save(output_path)
    
    return output_path, seed


def initialize_model(inpainting_model, diffusion_model, base_model_path, 
                     promptA=None, promptB=None, negative_promptA=None, 
                     negative_promptB=None, negative_prompt=None):
    """Initialize the inpainting model and return pipe and preprocess function"""
    
    if inpainting_model == 'powerpaint':
        from utils.powerpaint_utils import PowerPaintInitializer
        initializer = PowerPaintInitializer(base_model_path, diffusion_model, promptA, 
                                          promptB, negative_promptA, negative_promptB, 
                                          negative_prompt)
        pipe = initializer.pipe
        preprocess_func = preprocess_powerpaint
        
    elif inpainting_model == 'brushnet':
        from utils.brushnet_utils import BrushNetInitializer
        initializer = BrushNetInitializer(base_model_path, diffusion_model)
        pipe = initializer.pipe
        preprocess_func = preprocess_brushnet
        
    elif inpainting_model == 'controlnet':
        from utils.controlnet_utils import ControlNetInitializer
        initializer = ControlNetInitializer(base_model_path, diffusion_model)
        pipe = initializer.pipe
        preprocess_func = preprocess_controlnet
        
    elif inpainting_model == 'hdpainter':
        from utils.hdpainter_utils import HDPainterInitializer
        initializer = HDPainterInitializer()
        pipe = initializer.pipe
        preprocess_func = preprocess_hdpainter
        
    elif inpainting_model == 'inpaintanything':
        from utils.inpaintanything_utils import InpaintAnythingInitializer
        initializer = InpaintAnythingInitializer()
        pipe = initializer.pipe
        preprocess_func = preprocess_inpaint_anything
        
    else:
        raise ValueError(f"Unsupported inpainting model: {inpainting_model}")
    
    return pipe, preprocess_func


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Image Inpainting Script')
    
    # Required arguments
    parser.add_argument('--inpainting_model', type=str, required=True,
                       choices=['powerpaint', 'brushnet', 'controlnet', 'hdpainter', 'inpaintanything'],
                       help='Inpainting model to use')
    parser.add_argument('--csv_file', type=str, required=True,
                       help='Path to CSV file with columns: image_path, mask_path, prompt')
    parser.add_argument('--output_folder', type=str, required=True,
                       help='Output folder for inpainted images')
    
    # Model-specific arguments
    parser.add_argument('--diffusion_model', type=str, default='sd1-5',
                       help='Diffusion model for brushnet/controlnet/powerpaint')
    parser.add_argument('--base_model_path', type=str, default=None,
                       help='Base model path')
    
    # PowerPaint specific arguments
    parser.add_argument('--promptA', type=str, default='P_ctxt',
                       help='PowerPaint promptA (context prompt)')
    parser.add_argument('--promptB', type=str, default='P_obj',
                       help='PowerPaint promptB (object prompt)')
    parser.add_argument('--negative_promptA', type=str, default='',
                       help='PowerPaint negative promptA')
    parser.add_argument('--negative_promptB', type=str, default='',
                       help='PowerPaint negative promptB')
    parser.add_argument('--negative_prompt', type=str, default='',
                       help='PowerPaint negative promptU')
    
    # Processing arguments
    parser.add_argument('--dim_limit', type=int, default=1024,
                       help='Maximum dimension for images')
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip processing if output already exists')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    
    
    os.makedirs(args.output_folder, exist_ok=True)
    
    
    print(f"Reading CSV file: {args.csv_file}")
    df = pd.read_csv(args.csv_file)
    
    
    required_columns = ['image_path', 'mask_path', 'prompt']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV must contain column: {col}")
    
    print(f"Found {len(df)} images to process")
    
    
    print(f"Initializing {args.inpainting_model} model...")
    pipe, preprocess_func = initialize_model(
        args.inpainting_model,
        args.diffusion_model,
        args.base_model_path,
        args.promptA if args.inpainting_model == 'powerpaint' else None,
        args.promptB if args.inpainting_model == 'powerpaint' else None,
        args.negative_promptA if args.inpainting_model == 'powerpaint' else None,
        args.negative_promptB if args.inpainting_model == 'powerpaint' else None,
        args.negative_prompt if args.inpainting_model == 'powerpaint' else None
    )
    
    
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        try:
            # Check if output already exists
            image_name = Path(row['image_path']).stem
            output_filename = f"{image_name}_{args.inpainting_model}.png"
            output_path = os.path.join(args.output_folder, output_filename)
            
            if args.skip_existing and os.path.exists(output_path):
                print(f"Skipping {image_name} (already exists)")
                results.append({
                    'image_path': row['image_path'],
                    'mask_path': row['mask_path'],
                    'prompt': row['prompt'],
                    'output_path': output_path,
                    'seed': None,
                    'status': 'skipped'
                })
                continue
            
            
            output_path, seed = process_single_image(
                row,
                args.inpainting_model,
                args.diffusion_model,
                pipe,
                preprocess_func,
                args.output_folder,
                args.dim_limit,
                args.promptA if args.inpainting_model == 'powerpaint' else None,
                args.promptB if args.inpainting_model == 'powerpaint' else None,
                args.negative_promptA if args.inpainting_model == 'powerpaint' else None,
                args.negative_promptB if args.inpainting_model == 'powerpaint' else None,
                args.negative_prompt if args.inpainting_model == 'powerpaint' else None
            )
            
            results.append({
                'image_path': row['image_path'],
                'mask_path': row['mask_path'],
                'prompt': row['prompt'],
                'output_path': output_path,
                'seed': seed,
                'status': 'success'
            })
            
        except Exception as e:
            print(f"Error processing {row['image_path']}: {e}")
            results.append({
                'image_path': row['image_path'],
                'mask_path': row['mask_path'],
                'prompt': row['prompt'],
                'output_path': None,
                'seed': None,
                'status': f'failed: {str(e)}'
            })
    
    
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(args.output_folder, 'results.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nResults saved to: {results_csv_path}")
    print(f"Successfully processed: {len([r for r in results if r['status'] == 'success'])}/{len(df)} images")