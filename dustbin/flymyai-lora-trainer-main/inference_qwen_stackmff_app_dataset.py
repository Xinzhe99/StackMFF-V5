import argparse
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import math
import sys
import datetime

# Add parent directory to path to import network.py
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from model.network import DepthTransformer

from optimum.quanto import quantize, qfloat8, freeze, qint8
from diffusers import (
    AutoencoderKLQwenImage,
    QwenImageTransformer2DModel,
    FlowMatchEulerDiscreteScheduler,
    QwenImageEditPipeline
)

def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return width, height

def real_fusion_interface(latents, fusion_model):
    """
    Real fusion interface using DepthTransformer.
    Input: latents of shape (Stack_Size, C, T, H, W) where T=1
           OR (Stack_Size, C, H, W).
    Output: fused latent of shape (C, 1, H, W).
    """
    stack_input = latents.unsqueeze(0) # [1, Stack, ...]
    if len(stack_input.shape) == 6 and stack_input.shape[3] == 1:
         stack_input = stack_input.squeeze(3) # [1, Stack, C, H, W]
    
    with torch.no_grad():
        fused_output = fusion_model(stack_input) # [1, C, H, W]
    
    return fused_output.squeeze(0).unsqueeze(1) # [C, 1, H, W]

def load_stack_images(stack_dir, width, height, processor):
    stack_files = sorted([f for f in os.listdir(stack_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not stack_files:
        raise ValueError(f"No images found in {stack_dir}")
    
    images = []
    print(f"Loading {len(stack_files)} images from {stack_dir}...")
    for f in stack_files:
        img = Image.open(os.path.join(stack_dir, f)).convert('RGB')
        # Simple resize to target dims
        img = img.resize((width, height), Image.LANCZOS)
        images.append(img)
    return images

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack_dir", type=str, default="/home/ot/Students/xxz/datasets/test_datasets/Mobile Depth/image stack/keyboard", help="Path to the folder containing stack images")
    parser.add_argument("--prompt", type=str, default="High quality, sharp, crisp, clear, well-focused, high-resolution, detailed, noise-free, smooth, natural textures, realistic textures, rich textures", help="Text prompt")
    parser.add_argument("--fusion_model_path", type=str, default='/home/ot/Students/xxz/projects_mff/StackMFFV5/train_runs/train_runs9/model_save/epoch_5.pth', help="Path to trained DepthTransformer checkpoint (e.g. .pth)")
    parser.add_argument("--model_path", type=str, default="/home/ot/.cache/huggingface/hub/models--Qwen--Qwen-Image-Edit-2509/snapshots/d3968ef930e841f4c73640fb8afa3b306a78167e", help="Pretrained Qwen-Image-Edit path")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--output", type=str, default="output.png")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--quantize", type=bool, default=True, help="Enable 8-bit quantization for lower VRAM usage")
    args = parser.parse_args()

    # 1. Setup Data Type & Device
    dtype = torch.float32
    if args.dtype == "bf16": dtype = torch.bfloat16
    elif args.dtype == "fp16": dtype = torch.float16
    device = torch.device(args.device)

    print(f"Initializing models on {device} ({dtype})...")

    # 2. Load Models
    # A. Fusion Model
    fusion_model = DepthTransformer(input_channels=16, embed_dim=256, num_heads=8, num_layers=4)
    
    # Load state dict safely
    print(f"Loading fusion model from {args.fusion_model_path}")
    state_dict = torch.load(args.fusion_model_path, map_location='cpu')
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('depth_transformer.'):
            new_state_dict[k[len('depth_transformer.'):]] = v
        else:
            new_state_dict[k] = v
    fusion_model.load_state_dict(new_state_dict)
    fusion_model.to(device, dtype=dtype).eval()

    # B. Qwen Components
    print("Loading Qwen Pipeline...")
    if args.quantize:
        print("Model will be loaded on CPU first for quantization...")
        pipeline = QwenImageEditPipeline.from_pretrained(args.model_path, torch_dtype=dtype)
    else:
        pipeline = QwenImageEditPipeline.from_pretrained(args.model_path, torch_dtype=dtype).to(device)
    
    # We detach components for clearer manual loop
    vae = pipeline.vae
    transformer = pipeline.transformer
    scheduler = pipeline.scheduler
    
    # Ensure transformer patch is applied to BLOCKS (where _modulate actually lives)
    # Patch removed as requested by user - assuming using compatible model version (e.g. 2509)
    pass

    # C. Quantization Logic
    if args.quantize:
        print("Quantizing transformer blocks to int8 (weights only, matching train code)...")
        # Mimicking train_4090.py logic:
        # 1. Quantize blocks individually and move to CPU to save memory during process
        # 2. Move full model to Device
        # 3. Quantize remaining parts
        
        all_blocks = list(transformer.transformer_blocks)
        if hasattr(transformer, "single_transformer_blocks"):
             all_blocks.extend(list(transformer.single_transformer_blocks))

        for block in tqdm(all_blocks, desc="Quantizing blocks"):
            block.to(device, dtype=dtype)
            quantize(block, weights=qint8) # Only quantize weights, keep activations in precision
            freeze(block)
            block.to('cpu')
        
        print("Moving full transformer to device and quantizing remaining layers...")
        transformer.to(device, dtype=dtype)
        quantize(transformer, weights=qint8)
        freeze(transformer)
        
        print("Moving remaining components to device...")
        vae.to(device)
        if hasattr(pipeline, "text_encoder") and pipeline.text_encoder:
            pipeline.text_encoder.to(device)

    # 3. Process Input Stack
    # Identify stack folders
    root_dir = args.stack_dir
    if not os.path.exists(root_dir):
        print(f"Error: Directory {root_dir} not found.")
        return

    # Find subdirectories
    possible_stacks = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    stack_paths = []
    stack_names = []
    
    if not possible_stacks:
        # Check if it contains images directly (backward compatibility)
        if any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in os.listdir(root_dir)):
            print(f"No subdirectories found, assuming {root_dir} is a single stack.")
            stack_paths = [root_dir]
            stack_names = [os.path.basename(root_dir)] # Or custom output name
        else:
            print("No stack folders or images found.")
            return
    else:
        # Sort for deterministic order
        possible_stacks.sort()
        stack_paths = [os.path.join(root_dir, d) for d in possible_stacks]
        stack_names = possible_stacks
        print(f"Found {len(stack_paths)} stacks in {root_dir}")

    # Calculate dimensions once
    # Default aspect ratio 1:1 for now if not specified via images, assuming user passes 1024x1024 args
    w, h = calculate_dimensions(args.width * args.height, args.width / args.height)

    # Prepare VAE constants
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(device, dtype)
    latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(device, dtype)
    
    # Determine VAE scale factor securely
    if hasattr(pipeline, "vae_scale_factor"):
        vae_scale_factor = pipeline.vae_scale_factor
    elif hasattr(vae.config, "block_out_channels"):
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    else:
        print("Warning: Could not determine VAE scale factor from config. Defaulting to 16.")
        vae_scale_factor = 16

    latent_h = h // vae_scale_factor
    latent_w = w // vae_scale_factor
    num_channels_latents = getattr(vae.config, "z_dim", 16)
    bsz = 1

    print("Models loaded successfully.")
    print("Interactive mode: Input a prompt and press Enter to generate images for ALL stacks.")
    print("Press Ctrl+C or Ctrl+D (Unix) or Ctrl+Z (Windows) to exit.")
    
    gen_count = 0 
    
    while True:
        try:
            print(f"\n--- Batch Generation {gen_count} ---")
            user_input = input(f"Enter prompt: ")
            current_prompt = user_input.strip()
            
            print(f"Using prompt: '{current_prompt}'")

            # Create Timestamped Directory
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', f"results_{timestamp}_{gen_count}")
            os.makedirs(save_dir, exist_ok=True)
            print(f"Results will be saved to: {save_dir}")

            # Loop over stacks
            for idx, stack_path in enumerate(stack_paths):
                current_stack_name = stack_names[idx]
                print(f"Processing stack [{idx+1}/{len(stack_paths)}]: {current_stack_name} ...")
                
                try:
                    # Load images for this stack
                    stack_imgs_pil = load_stack_images(stack_path, w, h, pipeline.image_processor)
                    
                    # Encode Stack Images to Latents
                    stack_latents = []
                    for img in stack_imgs_pil:
                         # Standardize processing
                         img_tensor = torch.from_numpy((np.array(img) / 127.5) - 1)
                         img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
                         pixel_values = img_tensor.unsqueeze(2) 
                         latent = vae.encode(pixel_values).latent_dist.sample()
                         stack_latents.append(latent)

                    stack_latents = torch.cat(stack_latents, dim=0)
                    
                    # Fusion
                    control_latent = real_fusion_interface(stack_latents, fusion_model).to(dtype=dtype)
                    
                    # VAE Normalization
                    control_latent = control_latent.unsqueeze(0)
                    control_latent = (control_latent - latents_mean) * (1.0 / latents_std)

                    # Encode Prompt (needs image context - using middle image or first?)
                    # Using first image as per original code
                    prompt_embeds, prompt_masks = pipeline.encode_prompt(
                        prompt=current_prompt,
                        image=stack_imgs_pil[0], 
                        device=device,
                        num_images_per_prompt=1
                    )

                    # Prepare Latents
                    latents = torch.randn(
                        (bsz, num_channels_latents, 1, latent_h, latent_w),
                        device=device,
                        dtype=dtype
                    )

                    # Sampling Loop
                    mu = None
                    if getattr(scheduler.config, "use_dynamic_shifting", False):
                        seq_len = latent_h * latent_w
                        base_seq_len = getattr(scheduler.config, "base_image_seq_len", 256)
                        max_seq_len = getattr(scheduler.config, "max_image_seq_len", 4096)
                        base_shift = getattr(scheduler.config, "base_shift", 0.5)
                        max_shift = getattr(scheduler.config, "max_shift", 1.15)
                        mu = (seq_len - base_seq_len) / (max_seq_len - base_seq_len) * (max_shift - base_shift) + base_shift

                    try:
                        scheduler.set_timesteps(args.steps, mu=mu)
                    except TypeError:
                        scheduler.set_timesteps(args.steps)
                    
                    # Only show progress bar for first image or silence it?
                    # keeping tqdm is fine but might be noisy. `disable=False` default.
                    for t in scheduler.timesteps: # Removing tqdm for inner loop to avoid clutter, or maybe keep it? Let's keep loop simple
                        model_input = latents
                        timestep = t.expand(bsz).to(device, dtype)

                        packed_latents = QwenImageEditPipeline._pack_latents(
                            model_input, bsz, num_channels_latents, latent_h, latent_w
                        )
                        packed_condition = QwenImageEditPipeline._pack_latents(
                            control_latent.clone(), bsz, num_channels_latents, latent_h, latent_w
                        )
                        model_input_packed = torch.cat([packed_latents, packed_condition], dim=1)
                        
                        img_shapes = [[(1, latent_h // 2, latent_w // 2), (1, latent_h // 2, latent_w // 2)]] * bsz
                        txt_seq_lens = prompt_masks.sum(dim=1).tolist()
                        
                        noise_pred_packed = transformer(
                            hidden_states=model_input_packed,
                            timestep=timestep / 1000,
                            encoder_hidden_states=prompt_embeds,
                            encoder_hidden_states_mask=prompt_masks,
                            img_shapes=img_shapes,
                            txt_seq_lens=txt_seq_lens,
                            return_dict=False
                        )[0]
                        
                        dataset_token_len = packed_latents.size(1)
                        noise_pred_packed = noise_pred_packed[:, :dataset_token_len]
                        noise_pred = QwenImageEditPipeline._unpack_latents(
                            noise_pred_packed, height=latent_h * vae_scale_factor, width=latent_w * vae_scale_factor, vae_scale_factor=vae_scale_factor
                        )
                        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                    # Decode
                    latents = (latents * latents_std) + latents_mean
                    decoded = vae.decode(latents).sample.squeeze(2)
                    decoded = (decoded / 2 + 0.5).clamp(0, 1)
                    decoded = decoded.cpu().permute(0, 2, 3, 1).float().numpy()
                    img = Image.fromarray((decoded[0] * 255).astype(np.uint8))
                    
                    out_path = os.path.join(save_dir, f"{current_stack_name}.png")
                    img.save(out_path)
                    print(f"Saved: {out_path}")
                
                except Exception as e:
                    print(f"Failed to process stack {current_stack_name}: {e}")
                    import traceback
                    traceback.print_exc()

            print(f"Batch generation completed. All results in {save_dir}")
            gen_count += 1

        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive session.")
            break


if __name__ == "__main__":
    main()
