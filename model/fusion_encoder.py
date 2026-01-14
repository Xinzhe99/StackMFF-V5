import torch
import torch.nn as nn
import torch.nn.functional as F
import os
# 设置 Hugging Face 镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from diffusers import AutoencoderKL
from model.network import DepthTransformer


class FusionEncoder(nn.Module):
    """
    VAE + DepthTransformer 融合编码器
    将图像栈编码为单个融合后的隐变量
    """
    def __init__(self, vae_model_id, vae_subfolder, depth_transformer_cfg, 
                 fusion_weights_path, latent_scale_factor=0.18215):
        super().__init__()
        self.latent_scale_factor = latent_scale_factor
        
        # 加载 VAE
        print(f"Loading VAE from {vae_model_id}...")
        self.vae = AutoencoderKL.from_pretrained(
            vae_model_id,
            subfolder=vae_subfolder,
            torch_dtype=torch.float32
        )
        
        # 冻结 VAE
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()
        
        # 创建 DepthTransformer
        print(f"Creating DepthTransformer...")
        self.depth_transformer = DepthTransformer(
            input_channels=depth_transformer_cfg.input_channels,
            embed_dim=depth_transformer_cfg.embed_dim,
            num_heads=depth_transformer_cfg.num_heads,
            num_layers=depth_transformer_cfg.num_layers
        )
        
        # 加载融合网络权重
        print(f"Loading fusion weights from {fusion_weights_path}...")
        fusion_state_dict = torch.load(fusion_weights_path, map_location='cpu', weights_only=False)
        
        # 处理可能的 key 前缀
        if any(k.startswith('depth_transformer.') for k in fusion_state_dict.keys()):
            # 移除前缀
            new_state_dict = {}
            for k, v in fusion_state_dict.items():
                if k.startswith('depth_transformer.'):
                    new_state_dict[k.replace('depth_transformer.', '')] = v
            fusion_state_dict = new_state_dict
        
        self.depth_transformer.load_state_dict(fusion_state_dict, strict=True)
        print("✓ Fusion network weights loaded successfully")
        
        # 冻结 DepthTransformer
        for param in self.depth_transformer.parameters():
            param.requires_grad = False
        self.depth_transformer.eval()

    @torch.no_grad()
    def encode_single(self, x):
        """编码单张图像，返回带scale的latent"""
        latent_dist = self.vae.encode(x).latent_dist
        return latent_dist.mode() * self.latent_scale_factor

    @torch.no_grad()
    def encode_image(self, image):
        """
        编码单张或批量图像 (用于编码GT等)
        Args:
            image: [B, C, H, W] 图像，值域 [-1, 1]
        Returns:
            latent: [B, 4, H//8, W//8]
        """
        latent_dist = self.vae.encode(image).latent_dist
        return latent_dist.mode() * self.latent_scale_factor

    @torch.no_grad()
    def decode_latent(self, z):
        """
        解码隐变量为图像
        Args:
            z: [B, 4, H//8, W//8] 隐变量
        Returns:
            image: [B, C, H, W] 图像，值域 [-1, 1]
        """
        return self.vae.decode(z / self.latent_scale_factor).sample

    @torch.no_grad()
    def forward(self, image_stack):
        """
        Args:
            image_stack: [B, N, C, H, W] 图像栈，值域 [-1, 1]
        Returns:
            fused_latent: [B, 4, H//8, W//8] 融合后的隐变量
        """
        B, N, C, H, W = image_stack.shape
        
        # 1. VAE 编码每张图像
        x_reshaped = image_stack.view(B * N, C, H, W)
        latents = self.encode_single(x_reshaped)  # [B*N, 4, H//8, W//8]
        
        _, latent_c, latent_h, latent_w = latents.shape
        latents = latents.view(B, N, latent_c, latent_h, latent_w)  # [B, N, 4, H//8, W//8]
        
        # 2. DepthTransformer 融合
        fused_latent = self.depth_transformer(latents)  # [B, 4, H//8, W//8]
        
        return fused_latent