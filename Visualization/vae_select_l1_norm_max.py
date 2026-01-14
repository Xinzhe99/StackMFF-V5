import torch
from diffusers import AutoencoderKLWan
from PIL import Image
import numpy as np
import os
import argparse
import sys
import logging

# 设置 HuggingFace 镜像，防止连接超时
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="使用Wan VAE进行基于L1范数的潜空间融合")
    
    # 默认输入文件路径
    default_path_a = "/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/project_image_stack_fusion/StackMFFV5/test_A.jpg"
    default_path_b = "/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/project_image_stack_fusion/StackMFFV5/test_B.jpg"
    
    parser.add_argument("--input_path_a", type=str, default=default_path_a, help="输入图像 A (清晰前景)")
    parser.add_argument("--input_path_b", type=str, default=default_path_b, help="输入图像 B (清晰背景)")
    
    # 默认输出文件名
    parser.add_argument("--output_path", type=str, default="fused_result_l1.png",
                        help="输出路径 (默认: fused_result_l1.png)")
    
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen-Image", help="模型ID")
    parser.add_argument("--device", type=str, default=None, help="设备类型 (cuda/cpu)")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--vae_subfolder", type=str, default="vae", help="VAE子文件夹")
    parser.add_argument("--fallback_model", type=str, default="Wan-AI/Wan2.1-T2V-14B", help="备用模型ID")
    
    return parser.parse_args()


def load_vae_model(model_id, vae_subfolder, dtype, device, fallback_model):
    """加载 VAE 模型"""
    logger.info(f"正在加载 Wan VAE (来自 {model_id})...")
    
    if dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    try:
        vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder=vae_subfolder, torch_dtype=torch_dtype
        ).to(device)
        vae.eval()
        logger.info("VAE 加载成功！")
        return vae, torch_dtype
    except Exception as e:
        logger.warning(f"首选模型加载失败: {e}，尝试加载备用模型...")
        try:
            vae = AutoencoderKLWan.from_pretrained(
                fallback_model, subfolder=vae_subfolder, torch_dtype=torch_dtype
            ).to(device)
            vae.eval()
            return vae, torch_dtype
        except Exception as fallback_e:
            logger.error(f"模型加载完全失败: {fallback_e}")
            sys.exit(1)


def load_image_for_wan(image_path, device, dtype, target_size=None):
    """加载并预处理图像，支持强制尺寸对齐"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到文件: {image_path}")

    img = Image.open(image_path).convert("RGB")
    
    # 尺寸对齐逻辑
    if target_size is not None:
        if img.size != target_size:
            logger.info(f"调整图像尺寸以匹配: {img.size} -> {target_size}")
            img = img.resize(target_size, Image.LANCZOS)
    else:
        w, h = img.size
        new_w = w - (w % 16)
        new_h = h - (h % 16)
        if new_w != w or new_h != h:
            img = img.resize((new_w, new_h), Image.LANCZOS)
            logger.info(f"图像裁剪为16倍数: {new_w}x{new_h}")

    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = (img_np * 2.0) - 1.0
    
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
    # [C, H, W] -> [B, C, T, H, W]
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(2)
    
    return img_tensor.to(device=device, dtype=dtype), img.size


def save_wan_reconstruction(tensor, output_path):
    """保存结果"""
    tensor = tensor.squeeze(2).squeeze(0)
    tensor = (tensor / 2.0 + 0.5).clamp(0, 1)
    img_np = tensor.permute(1, 2, 0).float().cpu().numpy()
    img_np = (img_np * 255).round().astype("uint8")
    Image.fromarray(img_np).save(output_path)
    logger.info(f"结果已保存至: {output_path}")


# =========================================================================
# 核心策略函数：L1 Norm Fusion
# =========================================================================
def process_fusion_l1(vae, tensor_a, tensor_b):
    """
    基于 L1 范数的融合策略 (修复了数据类型报错问题)。
    """
    with torch.no_grad():
        logger.info("正在编码图像 A...")
        latent_a = vae.encode(tensor_a).latent_dist.mode()
        
        logger.info("正在编码图像 B...")
        latent_b = vae.encode(tensor_b).latent_dist.mode()
        
        # 1. 计算 L1 能量
        activity_a = torch.sum(torch.abs(latent_a), dim=1, keepdim=True)
        activity_b = torch.sum(torch.abs(latent_b), dim=1, keepdim=True)
        
        # 2. 生成掩膜 (Mask)
        # [关键修复]: 不要用 .float()，而是转为与 latent 相同的类型 (bfloat16)
        target_dtype = latent_a.dtype 
        mask = (activity_a > activity_b).to(dtype=target_dtype)
        
        # 统计信息
        ratio_a = mask.mean().item()
        logger.info(f"融合统计: {ratio_a:.1%} 选 A，{1-ratio_a:.1%} 选 B")
        
        # 3. 融合
        # 此时 mask 和 latent 类型一致，结果也会保持为 bfloat16
        latent_fused = latent_a * mask + latent_b * (1 - mask)
        
        # [双重保险]: 如果因为任何运算变成了 float32，这里强制转回模型的 dtype
        latent_fused = latent_fused.to(dtype=vae.dtype)

        # 4. 解码
        logger.info(f"解码输入类型: {latent_fused.dtype}, 模型类型: {vae.dtype}")
        decoded_tensor = vae.decode(latent_fused).sample
        
        return decoded_tensor


def main():
    args = parse_args()
    
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载模型
    vae, torch_dtype = load_vae_model(args.model_id, args.vae_subfolder, args.dtype, device, args.fallback_model)
    
    # 加载图像 (确保 B 对齐 A 的尺寸)
    logger.info(f"加载图像 A: {args.input_path_a}")
    tensor_a, size_a = load_image_for_wan(args.input_path_a, device, torch_dtype)
    
    logger.info(f"加载图像 B: {args.input_path_b}")
    tensor_b, _ = load_image_for_wan(args.input_path_b, device, torch_dtype, target_size=size_a)
    
    # 执行 L1 融合
    fused_tensor = process_fusion_l1(vae, tensor_a, tensor_b)
    
    # 保存
    save_wan_reconstruction(fused_tensor, args.output_path)


if __name__ == "__main__":
    main()