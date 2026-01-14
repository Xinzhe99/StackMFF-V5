# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
"""
Network architecture for StackMFF-V5.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKLWan
import os

# 设置 Hugging Face 镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


# =====================
# Depth-wise Transformer
# =====================
class DepthTransformerLayer(nn.Module):
    """
    单层 Depth Transformer Layer
    """
    def __init__(self, embed_dim, num_heads, ff_dim=None, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # Multi-head attention
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ff_dim = ff_dim or embed_dim * 4
        # Feed-forward 网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, self.ff_dim),
            nn.GELU(),
            nn.Linear(self.ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B*H*W, N, C]
        """
        # We apply LayerNorm *before* attention, which is a common and stable practice (Pre-LN)
        x_norm = self.norm1(x) 

        # Self-attention 沿 depth 方向
        # x_norm is used as query, key, and value
        x_attn = self.attn(x_norm, x_norm, x_norm)[0]  # [B*H*W, N, C]
        x_attn = self.dropout(x_attn)

        # 残差连接
        x = x + x_attn  # [B*H*W, N, C]

        # Feed-forward 网络
        x_norm2 = self.norm2(x) # Pre-LN for the FFN
        x_ffn = self.ffn(x_norm2)  # [B*H*W, N, C]
        x_ffn = self.dropout(x_ffn)
        
        # 残差连接
        x = x + x_ffn  # [B*H*W, N, C]

        return x


class DepthTransformer(nn.Module):
    """
    沿 depth 方向 (num_images) 建立 Transformer，用于捕捉图层间关系
    输入输出都为 [B, N, C, H, W]
    支持自定义层数
    """
    def __init__(self, input_channels, embed_dim, num_heads, num_layers=1, ff_dim=None, dropout=0.0):
        super().__init__()
        self.input_channels = input_channels  # VAE的潜在空间通道数
        self.embed_dim = embed_dim  # Transformer的隐藏维度
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # 输入投影层：将输入通道映射到Transformer的隐藏维度
        self.input_projection = nn.Linear(input_channels, embed_dim)
        
        # 输出投影层：将Transformer的输出映射回原始通道数
        self.output_projection = nn.Linear(embed_dim, input_channels)
        
        # 创建多层 transformer layers
        self.layers = nn.ModuleList([
            DepthTransformerLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        x: [B, N, C, H, W]
        """
        B, N, C, H, W = x.shape
            
        # 将输入从 [B, N, C, H, W] 转换为 [B*H*W, N, C]
        x_flat = x.permute(0, 3, 4, 1, 2).contiguous().reshape(B * H * W, N, C)
        
        # 通过输入投影层映射到Transformer的隐藏维度
        x_proj = self.input_projection(x_flat)  # [B*H*W, N, embed_dim]
        
        # 在Transformer中处理
        for layer in self.layers:
            x_proj = layer(x_proj)
        
        # 通过输出投影层映射回原始通道数
        x_out = self.output_projection(x_proj)  # [B*H*W, N, C]
    
        # 重塑回原始形状
        x_processed = x_out.reshape(B, H, W, N, C).permute(0, 3, 4, 1, 2)
            
        # 使用残差连接保留高频细节信息
        x_output = x + x_processed
            
        return x_output
        
class StackMFF_V5(nn.Module):
    def __init__(self, num_transformer_layers=4, vae_model_id="Qwen/Qwen-Image", vae_subfolder="vae", vae_dtype="bfloat16"):
        super(StackMFF_V5, self).__init__()
        
        # 加载VAE encoder和decoder
        print(f"正在加载 VAE 模型从 {vae_model_id}...")
        if vae_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif vae_dtype == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        
        self.vae_dtype = torch_dtype  # 保存VAE的数据类型
        
        # 初始化DepthTransformer用于融合latent特征
        # Wan VAE的潜在空间通道数为16
        # 使用投影层将低维VAE特征映射到更高维空间以增强表达能力
        self.depth_transformer = DepthTransformer(input_channels=16, embed_dim=256, num_heads=8, num_layers=num_transformer_layers)
        
        # 确保DepthTransformer使用正确的数据类型
        self.depth_transformer = self.depth_transformer.to(self.vae_dtype)
            
        try:
            self.vae = AutoencoderKLWan.from_pretrained(
                vae_model_id,
                subfolder=vae_subfolder,
                torch_dtype=torch_dtype
            )
        except Exception as e:
            print(f"首选模型加载失败: {e}...")
           
        
        # 冻结VAE所有参数
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()
        print("VAE 模型加载完成，所有参数已冻结") 
        
        # 将"层间-层内"打包成一个修正循环
        # TODO: 添加你的特征处理网络模块
        
    def forward(self, x):
        """
        前向传播 - VAE encoder -> 网络处理 -> VAE decoder
        """
        batch_size, num_images, channels, height, width = x.shape
        assert num_images >= 2, f"图像数量必须至少为2，当前为{num_images}"
        
        # 1. VAE Encode
        x_reshaped = x.view(batch_size * num_images, channels, height, width)
        x_with_time = x_reshaped.unsqueeze(2) # [B*N, C, 1, H, W]
        x_with_time = x_with_time.to(dtype=self.vae_dtype)
   
        with torch.no_grad():
            latent_dist = self.vae.encode(x_with_time).latent_dist
            latents = latent_dist.mode() 
        
        latents = latents.squeeze(2)
        _, latent_channels, latent_h, latent_w = latents.shape # [B*N, C, H, W]
        latents = latents.view(batch_size, num_images, latent_channels, latent_h, latent_w) # [B, N, C, H, W]
        
        fused_latent_features = self.depth_transformer(latents) # [B, N, C, H, W]
        fused_latent = fused_latent_features.max(dim=1)[0]# [B, C, H, W]
        
        # 3. VAE Decode
        fused_latent_with_time = fused_latent.unsqueeze(2)
        fused_latent_with_time = fused_latent_with_time.to(dtype=self.vae_dtype)
        
        fused_output = self.vae.decode(fused_latent_with_time).sample
        
        fused_output = fused_output.squeeze(2)
        fused_output = fused_output.to(dtype=x.dtype)
        
        return fused_output



if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    from PIL import Image
    import numpy as np
    
    # 创建模型并移动到GPU
    model = StackMFF_V5().to("cuda:0")
    
    # 读取真实的测试图像
    img_path_a = "/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/project_image_stack_fusion/StackMFFV5/test_A.jpg"
    img_path_b = "/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/project_image_stack_fusion/StackMFFV5/test_B.jpg"
    
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
    fused_image_np = fused_image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H, W, C]
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

