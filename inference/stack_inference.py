import os
import torch
import numpy as np
import time
from tqdm import tqdm
from PIL import Image
from omegaconf import OmegaConf
from typing import List, Generator
from argparse import Namespace
from torchvision import transforms

from model import ControlLDM, Diffusion
from model.fusion_encoder import FusionEncoder
from sampler import SpacedSampler
from utils.common import instantiate_from_config, VRAMPeakMonitor

class StackInferenceLoop:
    def __init__(self, args: Namespace):
        self.args = args
        self.train_cfg = OmegaConf.load(args.train_cfg)
        self.output_dir = args.output
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 1. Load Models
        self.device = torch.device(args.device)
        self._load_models()
        
        # 2. Setup Transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def _load_models(self):
        print(f"Loading models to {self.device}...")
        
        # --- Fusion Network (Stage 1) ---
        print("Loading Fusion Network...")
        fusion_cfg = self.train_cfg.model.fusion_network
        latent_scale_factor = self.train_cfg.model.cldm.params.latent_scale_factor
        
        self.fusion_encoder = FusionEncoder(
            vae_model_id=fusion_cfg.vae_model_id,
            vae_subfolder=fusion_cfg.vae_subfolder,
            depth_transformer_cfg=fusion_cfg.depth_transformer,
            fusion_weights_path=fusion_cfg.fusion_weights_path,
            latent_scale_factor=latent_scale_factor
        )
        self.fusion_encoder.to(self.device).eval()
        
        # --- ControlNet + Diffusion (Stage 2) ---
        print("Loading ControlNet & Diffusion...")
        self.cldm: ControlLDM = instantiate_from_config(self.train_cfg.model.cldm)
        
        # Load SD weights
        if hasattr(self.train_cfg.train, 'sd_path'):
            sd_path = self.train_cfg.train.sd_path
            print(f"Loading SD weights from {sd_path}")
            sd_weight = torch.load(sd_path, map_location="cpu")
            if "state_dict" in sd_weight:
                sd_weight = sd_weight["state_dict"]
            unused, missing = self.cldm.load_pretrained_sd(sd_weight)
        
        # Load ControlNet weights
        print(f"Loading ControlNet weights from {self.args.ckpt}")
        control_weight = torch.load(self.args.ckpt, map_location="cpu")
        self.cldm.load_controlnet_from_ckpt(control_weight)
        
        self.cldm.to(self.device).eval()
        
        # Cast dtype
        cast_type = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.args.precision]
        self.cldm.cast_dtype(cast_type)
        
        # --- Diffusion ---
        self.diffusion: Diffusion = instantiate_from_config(self.train_cfg.model.diffusion)
        self.diffusion.to(self.device)
        
        # --- Sampler ---
        self.sampler = SpacedSampler(
            self.diffusion.betas, 
            self.diffusion.parameterization, 
            rescale_cfg=self.args.rescale_cfg
        )

    def _load_stack(self, stack_dir: str, image_size: int = 512):
        """Load images from a directory as a stack."""
        images = []
        # Sort using numeric value if possible
        image_files = sorted(
            [f for f in os.listdir(stack_dir) if f.lower().endswith(('.png', '.jpg', '.bmp', '.jpeg'))],
            key=lambda x: int(''.join(filter(str.isdigit, x)) or '0')
        )
        
        if not image_files:
            print(f"Warning: No images found in {stack_dir}")
            return None

        for img_name in image_files:
            img_path = os.path.join(stack_dir, img_name)
            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    img = img.resize((image_size, image_size), Image.BILINEAR)
                    img_tensor = self.transform(img)
                    images.append(img_tensor)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
        
        if not images:
            return None
            
        return torch.stack(images) # [N, C, H, W]

    def _get_stacks(self) -> Generator:
        """Yields (stack_name, stack_tensor)"""
        input_root = self.args.input
        
        # Check if input is likely a single stack (contains images) 
        # or a root folder of stacks (contains folders)
        items = os.listdir(input_root)
        has_images = any(f.lower().endswith(('.png', '.jpg')) for f in items)
        
        if has_images:
            # Treat input_root as a single stack
            print(f"Treating {input_root} as a single image stack.")
            stack = self._load_stack(input_root, self.args.image_size)
            if stack is not None:
                yield os.path.basename(input_root), stack
        else:
            # Treat input_root as containing multiple stacks
            subdirs = [os.path.join(input_root, d) for d in items if os.path.isdir(os.path.join(input_root, d))]
            subdirs = sorted(subdirs)
            print(f"Found {len(subdirs)} potential stack directories in {input_root}")
            
            for stack_dir in subdirs:
                stack_name = os.path.basename(stack_dir)
                stack = self._load_stack(stack_dir, self.args.image_size)
                if stack is not None:
                    yield stack_name, stack

    @torch.no_grad()
    def run(self):
        print(f"Starting inference with Image Size: {self.args.image_size}...")
        
        for stack_name, stack_tensor in self._get_stacks():
            print(f"Processing stack: {stack_name} (Size: {len(stack_tensor)})")
            start_time = time.time()
            
            # Batch size for stack? 
            # The FusionEncoder expects [B, N, C, H, W]. Here B=1.
            stack_tensor = stack_tensor.unsqueeze(0).to(self.device) # [1, N, C, H, W]
            
            # 1. Fusion: Get Fused Latent
            with VRAMPeakMonitor("Fusion"):
                fused_latent = self.fusion_encoder(stack_tensor) # [1, 4, h, w]
            
            # 2. Prepare Conditions
            B = fused_latent.shape[0]
            prompt = [self.args.pos_prompt] * B
            neg_prompt = [self.args.neg_prompt] * B
            
            c_txt = self.cldm.clip.encode(prompt)
            c_txt_uncond = self.cldm.clip.encode(neg_prompt)
            
            cond = {"c_txt": c_txt, "c_img": fused_latent}
            uncond = {"c_txt": c_txt_uncond, "c_img": fused_latent} 
            
            # 3. Sampling
            # Prepare shape
            _, _, h, w = fused_latent.shape
            shape = (B, 4, h, w)
            
            # Start noise
            x_T = torch.randn(shape, device=self.device)
            
            with VRAMPeakMonitor("Sampling"):
                samples = self.sampler.sample(
                    model=self.cldm,
                    device=self.device,
                    steps=self.args.steps,
                    x_size=shape,
                    cond=cond,
                    uncond=uncond,
                    cfg_scale=self.args.cfg_scale,
                    x_T=x_T,
                    progress=True
                )
            
            # 4. Decode
            with VRAMPeakMonitor("Decoding"):
                # VAE decode
                # Samples are latents here
                x_samples = self.fusion_encoder.decode_latent(samples)
                
                # Normalize to [0, 1] for saving
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                
            # 5. Save
            for i in range(x_samples.shape[0]):
                img_tensor = x_samples[i].cpu()
                img = transforms.ToPILImage()(img_tensor)
                save_path = os.path.join(self.output_dir, f"{stack_name}_fused.png")
                img.save(save_path)
                print(f"Saved: {save_path}")
            
            print(f"Time: {time.time() - start_time:.2f}s")
