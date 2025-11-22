# Image Inpainting with Multiple Models

A unified interface for running image inpainting with 5 different models: HD-Painter, PowerPaint, Inpaint-Anything, ControlNet, and BrushNet.

## Setup

Due to dependency conflicts between models, each requires its own conda environment.

### 1. Install Model Repositories

Clone and set up each model you want to use:

- **HD-Painter**: https://github.com/Picsart-AI-Research/HD-Painter
- **PowerPaint**: https://github.com/open-mmlab/PowerPaint
- **Inpaint-Anything**: https://github.com/geekyutao/Inpaint-Anything
- **ControlNet**: https://github.com/mikonvergence/ControlNetInpaint
- **BrushNet**: https://github.com/TencentARC/BrushNet

Follow each repository's installation instructions to create separate conda environments.

### 2. Configure Paths

Edit `config.py` and set `BASE_PATH` to the directory containing all your model repositories:

```python
BASE_PATH = "/path/to/inpainting/repos"  # Directory containing HD-Painter, PowerPaint, etc.
```

The script expects this structure:
```
/path/to/your/models/
├── HD-Painter-main/
├── PowerPaint/
├── Inpaint-Anything-main/
├── ControlNetInpaint/
└── BrushNet2/examples/brushnet/
```

### 3. Prepare Input CSV

Create a CSV file with three columns:

| image_path | mask_path | prompt |
|------------|-----------|--------|
| /path/to/image1.jpg | /path/to/mask1.png | a red sports car |
| /path/to/image2.jpg | /path/to/mask2.png | a wooden table |

Optional fourth column: `expansion_percent` (defaults to 0 if not provided)

## Usage

Activate the appropriate conda environment for your chosen model, then run:

```bash
# HD-Painter
conda activate hdpainter
python inpaint.py \
    --inpainting_model hdpainter \
    --csv_file input.csv \
    --output_folder ./outputs

# PowerPaint
conda activate powerpaint
python inpaint.py \
    --inpainting_model powerpaint \
    --diffusion_model sd1-5 \
    --csv_file input.csv \
    --output_folder ./outputs

# BrushNet
conda activate diffusers
python inpaint.py \
    --inpainting_model brushnet \
    --diffusion_model realisticvision \
    --base_model_path /path/to/base/model \
    --csv_file input.csv \
    --output_folder ./outputs

# ControlNet
conda activate controlnet
python inpaint.py \
    --inpainting_model controlnet \
    --diffusion_model sd1-5 \
    --csv_file input.csv \
    --output_folder ./outputs

# Inpaint-Anything
conda activate inpaintanything
python inpaint.py \
    --inpainting_model inpaintanything \
    --csv_file input.csv \
    --output_folder ./outputs
```

## Arguments

**Required:**
- `--inpainting_model`: Model to use (`hdpainter`, `powerpaint`, `brushnet`, `controlnet`, `inpaintanything`)
- `--csv_file`: Path to input CSV file
- `--output_folder`: Where to save inpainted images

**Optional:**
- `--diffusion_model`: Base diffusion model (for brushnet/controlnet/powerpaint). Options: `sd1-5`, `realisticvision`, `dreamshaper`, `epicrealism`, `sdxl`, `juggernaut`
- `--base_model_path`: Path to base model weights
- `--dim_limit`: Maximum image dimension (default: 1024)
- `--skip_existing`: Skip processing if output already exists

**PowerPaint-specific:**
- `--promptA`: Context prompt (default: `P_ctxt`)
- `--promptB`: Object prompt (default: `P_obj`)
- `--negative_promptA`, `--negative_promptB`, `--negative_prompt`: Negative prompts

## Output

The script creates:
- Inpainted images: `{image_name}_{model}.png`
- `results.csv`: Summary with columns: `image_path`, `mask_path`, `prompt`, `output_path`, `seed`, `status`

## Notes

- The script automatically handles CUDA memory issues by trying progressively smaller image sizes
- Each image gets a random seed (saved in results.csv)
- Mask dilation can be controlled via the `expansion_percent` column in your CSV
