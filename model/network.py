# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
"""
Network architecture for StackMFF-V5.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
# 设置 Hugging Face 镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from diffusers import AutoencoderKL


class DepthTransformerLayer(nn.Module):
    """
    单层 Depth Transformer Layer (带 2D 空间卷积)
    结构: Pre-LN Attention -> Pre-LN FFN -> Pre-LN Spatial Conv2d
    """
    def __init__(self, embed_dim, num_heads, ff_dim=None, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 1. Self-Attention (处理 N 维度，即深度/时序信息)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # 2. Feed-forward (处理特征维度)
        self.ff_dim = ff_dim or embed_dim * 4
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, self.ff_dim),
            nn.GELU(),
            nn.Linear(self.ff_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # 3. 2D Depthwise Convolution (处理 H, W 维度，即空间信息)
        # groups=embed_dim 保证是 Depthwise (通道独立)，只做空间平滑，不混合通道
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, 
                      kernel_size=3, padding=1, groups=embed_dim),
            nn.GELU()
        )
        self.norm3 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, height, width):
        """
        x: [B*H*W, N, C]  (注意：这里的 N 可能包含 fusion token)
        height, width: 原始图像的高和宽，用于还原空间结构
        """
        # --- Block 1: Attention (Depth interaction) ---
        x_norm1 = self.norm1(x)
        x_attn = self.attn(x_norm1, x_norm1, x_norm1)[0]
        x = x + self.dropout(x_attn)

        # --- Block 2: FFN (Channel interaction) ---
        x_norm2 = self.norm2(x)
        x_ffn = self.ffn(x_norm2)
        x = x + self.dropout(x_ffn)
        
        # --- Block 3: Spatial Conv2d (Spatial interaction) ---
        # 这一步进行空间平滑
        x_norm3 = self.norm3(x) # [Total_Pixels, N, C]
        
        # 1. 还原维度
        total_pixels, seq_len, c = x_norm3.shape
        b = total_pixels // (height * width)
        
        # 变换流程:
        # [B*H*W, N, C] -> [B, H, W, N, C] -> [B, N, C, H, W]
        # 我们把 (B, N) 合并视为 Conv2d 的 Batch 维度，这样对每一张图、每一个深度层都做卷积
        x_img = x_norm3.view(b, height, width, seq_len, c).permute(0, 3, 4, 1, 2)
        x_img = x_img.reshape(b * seq_len, c, height, width) # [B*N, C, H, W]
        
        # 2. 执行 2D 卷积
        x_conv_out = self.spatial_conv(x_img) # [B*N, C, H, W]
        
        # 3. 展平回 Transformer 格式
        # [B*N, C, H, W] -> [B, N, C, H, W] -> [B, H, W, N, C] -> [B*H*W, N, C]
        x_conv_out = x_conv_out.view(b, seq_len, c, height, width).permute(0, 3, 4, 1, 2)
        x_conv_out = x_conv_out.reshape(total_pixels, seq_len, c)
        
        # 4. 残差连接
        x = x + self.dropout(x_conv_out)

        return x


class DepthTransformer(nn.Module):
    def __init__(self, input_channels, embed_dim, num_heads, num_layers, ff_dim=None, dropout=0.0):
        super().__init__()
        self.input_projection = nn.Linear(input_channels, embed_dim)
        self.output_projection = nn.Linear(embed_dim, input_channels)
        
        self.fusion_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.layers = nn.ModuleList([
            DepthTransformerLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        x: [B, N, C, H, W]
        """
        B, N, C, H, W = x.shape
        
        # 1. 展平并投影
        x_flat = x.permute(0, 3, 4, 1, 2).contiguous().reshape(B * H * W, N, C)
        x_proj = self.input_projection(x_flat) 
        
        # 2. 插入 Fusion Token
        # 扩展 Fusion Token 到每个像素位置 [B*H*W, 1, embed_dim]
        f_token = self.fusion_token.expand(B * H * W, -1, -1)
        x_combined = torch.cat([f_token, x_proj], dim=1) # [Total_Pixels, N+1, embed_dim]
        
        # 3. Transformer 处理
        # 关键点：将 H 和 W 传进去
        for layer in self.layers:
            x_combined = layer(x_combined, H, W)
        
        # 4. 提取输出
        fused_out = x_combined[:, 0, :] # [B*H*W, embed_dim]
        
        # 5. 投影回原始通道
        x_out = self.output_projection(fused_out) 
    
        # 6. 重塑回图像形状
        x_processed = x_out.reshape(B, H, W, C).permute(0, 3, 1, 2)
            
        return x_processed

class StackMFF_V5(nn.Module):
    def __init__(self, vae_model_id="/home/ot/.cache/huggingface/hub/models--Manojb--stable-diffusion-2-1-base/snapshots/0094d483a120f3f33dafbd187ea4aa60d10de75c", vae_subfolder="vae"):
        super(StackMFF_V5, self).__init__()
        
        print(f"正在加载 VAE 模型从 {vae_model_id}...")

        # 默认使用 float32
        self.vae_dtype = torch.float32
        
        # 初始化 DepthTransformer
        # Stable Diffusion VAE channels = 4
        self.depth_transformer = DepthTransformer(input_channels=4, embed_dim=256, num_heads=8, num_layers=4)
        self.depth_transformer = self.depth_transformer.to(self.vae_dtype)
            
        try:
            self.vae = AutoencoderKL.from_pretrained(
                vae_model_id,
                subfolder=vae_subfolder,
                torch_dtype=self.vae_dtype
            )
        except Exception as e:
            print(f"首选模型加载失败: {e}...")
        
        # 冻结 VAE 参数
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()
        print("VAE 模型加载完成，所有参数已冻结") 

    def encode_one_step(self, x):
        """
        单个 batch 的编码步骤，方便用于 DataParallel 或 chunking
        Args:
            x: [B, C, H, W]
        Returns:
            latents: [B, C_lat, H_lat, W_lat]
        """
        with torch.no_grad():
            latent_dist = self.vae.encode(x).latent_dist
            latents = latent_dist.mode()
        return latents

    def forward(self, x, chunk_size=None):
        """
        前向传播: VAE encoder -> DepthTransformer -> VAE decoder
        Args:
            x: [B, N, C, H, W]
            chunk_size: 如果指定，则在 VAE 编码阶段对 N 进行分块处理以节省显存
        """
        batch_size, num_images, channels, height, width = x.shape
        assert num_images >= 2, f"图像数量必须至少为2，当前为{num_images}"
        
        # 1. VAE Encode
        x_reshaped = x.view(batch_size * num_images, channels, height, width)
        x_reshaped = x_reshaped.to(dtype=self.vae_dtype)
   
        # 使用 chunk_size 避免 OOM
        if chunk_size is not None and chunk_size > 0:
            latents_list = []
            total = x_reshaped.shape[0]
            for i in range(0, total, chunk_size):
                end = min(i + chunk_size, total)
                batch_x = x_reshaped[i:end]
                # 编码
                batch_latents = self.encode_one_step(batch_x)
                latents_list.append(batch_latents)
            latents = torch.cat(latents_list, dim=0) # [B*N, C, H, W]
        else:
            latents = self.encode_one_step(x_reshaped)
        
        _, latent_channels, latent_h, latent_w = latents.shape  # [B*N, C, H, W]
        latents = latents.view(batch_size, num_images, latent_channels, latent_h, latent_w)  # [B, N, C, H, W]
        
        # 2. DepthTransformer 融合
        fused_latent = self.depth_transformer(latents)  # [B, C, H, W]
        
        # 3. VAE Decode
        fused_latent = fused_latent.to(dtype=self.vae_dtype)
        fused_output = self.vae.decode(fused_latent).sample
        fused_output = fused_output.to(dtype=x.dtype)
        
        return fused_output



if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    from PIL import Image
    import numpy as np
    
    # 创建模型并移动到GPU
    model = StackMFF_V5().to("cuda:0")
    
    # 读取真实的测试图像
    img_path_a = "/home/ot/Students/xxz/projects_mff/StackMFFV5/Visualization/test_pair/test_A.jpg"
    img_path_b = "/home/ot/Students/xxz/projects_mff/StackMFFV5/Visualization/test_pair/test_B.jpg"
    
    # 加载图像并预处理
    img_a = Image.open(img_path_a).convert('RGB')
    img_b = Image.open(img_path_b).convert('RGB')
    
    # 确保图像尺寸是16的倍数
    w, h = img_a.size
    new_w = w - (w % 16)
    new_h = h - (h % 16)
    img_a = img_a.resize((new_w, new_h), Image.LANCZOS)
    img_b = img_b.resize((new_w, new_h), Image.LANCZOS)
    
    # 转换为tensor并归一化到[-1, 1]
    def img_to_tensor(img):
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # [C, H, W]
        img_tensor = (img_tensor * 2.0) - 1.0  # [-1, 1]
        return img_tensor.unsqueeze(0)  # [1, C, H, W]
    
    img_a_tensor = img_to_tensor(img_a)
    img_b_tensor = img_to_tensor(img_b)
    
    # 组合图像栈 [B, N, C, H, W]
    x = torch.stack([img_a_tensor.squeeze(0), img_b_tensor.squeeze(0)], dim=0).unsqueeze(0)  # [1, 2, 3, H, W]
    x = x.to("cuda:0")

    print(f"Input shape: {x.shape}")
    
    # 测试推理模式
    model.eval()
    with torch.no_grad():
        fused_image = model(x)
    
    print(f"Fused image shape: {fused_image.shape}")
    print(f"Output value range: [{fused_image.min().item():.3f}, {fused_image.max().item():.3f}]")
    
    # 保存融合结果
    # 将输出从[-1, 1]范围转换回[0, 1]并转换为图像格式
    fused_image_np = fused_image.squeeze(0).permute(1, 2, 0).cpu().float().numpy()  # [H, W, C], 转换为float32避免BFloat16问题
    fused_image_np = (fused_image_np + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    fused_image_np = np.clip(fused_image_np, 0, 1)  # 确保值在[0, 1]范围内
    fused_image_pil = Image.fromarray((fused_image_np * 255).astype(np.uint8))
    fused_image_pil.save("fused_result_real_images.jpg")
    print("融合结果已保存为 fused_result_real_images.jpg")
    
    # 内存使用情况
    print('{:>16s} : {:<.3f} [M]'.format('Max Memory', torch.cuda.max_memory_allocated(torch.cuda.current_device())/1024**2))
    
    # 计算FLOPs和参数量
    try:
        flops = FlopCountAnalysis(model, (x,))
        print(flop_count_table(flops))
    except Exception as e:
        print(f"FLOPs计算跳过: {e}")
    
    # 额外测试不同图像数量
    print("\n=== Testing with different stack sizes ===")
    test_cases = [3, 5]
    for num_images in test_cases:
        print(f"\n--- {num_images} images ---")
        x_test = torch.randn(1, num_images, 3, new_h, new_w).to("cuda:0") * 2 - 1
        
        model.eval()
        with torch.no_grad():
            fused_output = model(x_test)
        print(f"Input: {x_test.shape} -> Output: {fused_output.shape}")
        print(f"Output range: [{fused_output.min().item():.3f}, {fused_output.max().item():.3f}]")

