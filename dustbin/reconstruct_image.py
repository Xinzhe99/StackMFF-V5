# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
"""
使用VAE对单张图像进行编码再解码重建
"""

import torch
import torch.nn as nn
import os
# 设置 Hugging Face 镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from diffusers import AutoencoderKL
from PIL import Image
import numpy as np


class VAEImageReconstructor:
    def __init__(self, vae_model_id="/home/ot/.cache/huggingface/hub/models--Manojb--stable-diffusion-2-1-base/snapshots/0094d483a120f3f33dafbd187ea4aa60d10de75c", vae_subfolder="vae", vae_dtype=None):
        """
        初始化VAE重建器
        """
        print(f"正在加载 VAE 模型从 {vae_model_id}...")
        
        # 使用指定的数据类型，如果没有指定则使用float32
        if vae_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif vae_dtype == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        
        self.vae_dtype = torch_dtype  # 保存VAE的数据类型
        
        try:
            self.vae = AutoencoderKL.from_pretrained(
                vae_model_id,
                subfolder=vae_subfolder,
                torch_dtype=torch_dtype
            )
        except Exception as e:
            print(f"首选模型加载失败: {e}")
            raise e
            
        # 冻结VAE所有参数
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()
        print("VAE 模型加载完成，所有参数已冻结")
        
        # 将VAE移到GPU（如果可用）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae = self.vae.to(self.device)
        print(f"VAE 已移动到设备: {self.device}")

    def preprocess_image(self, image_path):
        """
        预处理图像：加载、调整大小（确保是16的倍数）、归一化到[-1, 1]
        """
        # 加载图像
        img = Image.open(image_path).convert('RGB')
        
        # 确保图像尺寸是16的倍数（VAE要求）
        w, h = img.size
        new_w = w - (w % 16)
        new_h = h - (h % 16)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        
        # 转换为tensor并归一化到[-1, 1]
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # [C, H, W]
        img_tensor = (img_tensor * 2.0) - 1.0  # [-1, 1]
        
        # 添加批次维度 [1, C, H, W]
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor.to(self.device, dtype=self.vae_dtype)

    def reconstruct_image(self, image_path):
        """
        对图像进行编码再解码重建
        """
        print(f"正在处理图像: {image_path}")
        
        # 预处理图像
        img_tensor = self.preprocess_image(image_path)
        print(f"输入图像形状: {img_tensor.shape}")
        
        # 编码：将图像转换为潜在空间表示
        with torch.no_grad():
            # VAE编码不需要添加时间维度
            latent_dist = self.vae.encode(img_tensor).latent_dist
            latents = latent_dist.mode()
        
        print(f"潜在空间形状: {latents.shape}")
        
        # 解码：将潜在表示转换回图像
        with torch.no_grad():
            # 解码时也不需要时间维度
            latents = latents.to(dtype=self.vae_dtype)
            reconstructed = self.vae.decode(latents).sample
        
        print(f"重建图像形状: {reconstructed.shape}")
        
        # 将重建图像从[-1, 1]范围转换回[0, 1]并转换为PIL图像
        reconstructed_np = reconstructed.squeeze(0).permute(1, 2, 0).cpu().float().numpy()  # [H, W, C], 转换为float32避免BFloat16问题
        reconstructed_np = (reconstructed_np + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        reconstructed_np = np.clip(reconstructed_np, 0, 1)  # 确保值在[0, 1]范围内
        reconstructed_img = Image.fromarray((reconstructed_np * 255).astype(np.uint8))
        
        # 需要先加载原始图像返回
        original_img = Image.open(image_path).convert('RGB')
        # 确保原始图像尺寸与重建图像一致
        w, h = original_img.size
        new_w = w - (w % 16)
        new_h = h - (h % 16)
        original_img = original_img.resize((new_w, new_h), Image.LANCZOS)
        
        return reconstructed_img, original_img

    def save_reconstructed_image(self, reconstructed_img, original_path, output_path=None):
        """
        保存重建的图像
        """
        if output_path is None:
            # 生成默认输出路径
            base_path = os.path.splitext(original_path)[0]
            output_path = f"{base_path}_reconstructed.jpg"
        
        reconstructed_img.save(output_path)
        print(f"重建图像已保存到: {output_path}")
        
        return output_path


if __name__ == "__main__":
    # 创建重建器实例
    reconstructor = VAEImageReconstructor()
    
    # 指定要重建的图像路径
    input_image_path = "Visualization/12.jpg"
    
    # 检查输入图像是否存在
    if not os.path.exists(input_image_path):
        print(f"错误: 输入图像不存在: {input_image_path}")
        # 尝试使用绝对路径
        input_image_path = "/home/ot/Students/xxz/projects_mff/StackMFFV5/Visualization/test_pair/test_A.jpg"
        if not os.path.exists(input_image_path):
            # 最后尝试项目内的路径
            input_image_path = "c:/Users/dell/Desktop/Working/StackMFF V5/Visualization/12.jpg"
            if not os.path.exists(input_image_path):
                print(f"错误: 输入图像不存在: {input_image_path}")
                exit(1)
    
    try:
        # 执行重建
        reconstructed_img, original_img = reconstructor.reconstruct_image(input_image_path)
        
        # 生成输出路径
        output_path = os.path.join(os.path.dirname(input_image_path), 
                                  os.path.splitext(os.path.basename(input_image_path))[0] + "_reconstructed.jpg")
        
        # 保存重建图像
        reconstructor.save_reconstructed_image(reconstructed_img, input_image_path, output_path)
        
        print("图像重建完成！")
        print(f"原始图像: {input_image_path}")
        print(f"重建图像: {output_path}")
        
    except Exception as e:
        print(f"重建过程中出现错误: {e}")
        import traceback
        traceback.print_exc()