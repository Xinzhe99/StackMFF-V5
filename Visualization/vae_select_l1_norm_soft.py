import torch
from diffusers import AutoencoderKLWan
from PIL import Image
import numpy as np
import os
import argparse
import sys
import logging
import torch.nn.functional as F  # 确保导入这个

# 设置 HuggingFace 镜像，防止连接超时
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="使用Wan VAE进行基于局部方差的软融合")
    
    # 默认输入文件路径
    default_path_a = "/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/project_image_stack_fusion/StackMFFV5/test_A.jpg"
    default_path_b = "/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/project_image_stack_fusion/StackMFFV5/test_B.jpg"
    
    parser.add_argument("--input_path_a", type=str, default=default_path_a, help="输入图像 A (清晰前景)")
    parser.add_argument("--input_path_b", type=str, default=default_path_b, help="输入图像 B (清晰背景)")
    
    # 默认输出文件名
    parser.add_argument("--output_path", type=str, default="fused_result_variance_soft.png",
                        help="输出路径")
    
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
    """加载并预处理图像"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到文件: {image_path}")

    img = Image.open(image_path).convert("RGB")
    
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
# 修复后的核心策略函数：基于局部方差的软融合
# =========================================================================
def process_fusion_variance_soft(vae, tensor_a, tensor_b):
    """
    改进策略：基于局部方差的软融合
    """
    with torch.no_grad():
        logger.info("正在编码...")
        latent_a = vae.encode(tensor_a).latent_dist.mode()
        latent_b = vae.encode(tensor_b).latent_dist.mode()
        
        # --- [关键修改] 修复维度的局部方差计算函数 ---
        def compute_local_variance(feat):
            # feat shape: [B, C, T, H, W] -> [1, 16, 1, 64, 64]
            
            # 1. 降维: 去掉 Time 维度，变成 4D [B, C, H, W]
            feat_2d = feat.squeeze(2) 
            
            # 2. 计算局部均值 (3x3 窗口)
            pad = 1
            avg_pool = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=pad)
            
            mean = avg_pool(feat_2d)
            mean_sq = avg_pool(feat_2d ** 2)
            
            # 3. 方差 = E[x^2] - (E[x])^2
            variance = mean_sq - mean ** 2
            
            # 4. 对所有通道求和 -> [B, 1, H, W]
            total_variance = torch.sum(variance, dim=1, keepdim=True)
            
            # 5. 升维: 把 Time 维度加回来 -> [B, 1, 1, H, W]
            return total_variance.unsqueeze(2)

        logger.info("正在计算纹理方差图...")
        var_a = compute_local_variance(latent_a)
        var_b = compute_local_variance(latent_b)
        
        # --- 软掩膜 ---
        gain = 10.0 
        diff = (var_a - var_b)
        mask = torch.sigmoid(diff * gain)
        
        # 类型转换 (bfloat16)
        mask = mask.to(dtype=latent_a.dtype)
        
        logger.info(f"Mask 统计: 平均值 {mask.mean():.4f}")

        # --- 融合 ---
        logger.info("执行加权融合...")
        latent_fused = latent_a * mask + latent_b * (1 - mask)
        latent_fused = latent_fused.to(dtype=vae.dtype)
        
        # --- 解码 ---
        decoded_tensor = vae.decode(latent_fused).sample
        return decoded_tensor


def main():
    args = parse_args()
    
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载模型
    vae, torch_dtype = load_vae_model(args.model_id, args.vae_subfolder, args.dtype, device, args.fallback_model)
    
    # 加载图像
    logger.info(f"加载图像 A: {args.input_path_a}")
    tensor_a, size_a = load_image_for_wan(args.input_path_a, device, torch_dtype)
    
    logger.info(f"加载图像 B: {args.input_path_b}")
    tensor_b, _ = load_image_for_wan(args.input_path_b, device, torch_dtype, target_size=size_a)
    
    # 执行融合
    fused_tensor = process_fusion_variance_soft(vae, tensor_a, tensor_b)
    
    # 保存
    save_wan_reconstruction(fused_tensor, args.output_path)


if __name__ == "__main__":
    main()