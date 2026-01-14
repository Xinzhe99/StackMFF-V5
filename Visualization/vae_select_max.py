import torch
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from diffusers import AutoencoderKLWan
from PIL import Image
import numpy as np
import argparse
import sys
import logging


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="使用Wan VAE对两张图像进行潜空间Max融合")
    
    # 修改：默认路径改为你指定的 A 和 B
    default_path_a = "/home/ot/Students/xxz/projects_mff/StackMFF V5/Visualization/test_pair/test_A.jpg"
    default_path_b = "/home/ot/Students/xxz/projects_mff/StackMFF V5/Visualization/test_pair/test_B.jpg"
    
    parser.add_argument("--input_path_a", type=str, default=default_path_a,
                        help="输入图像 A 路径")
    parser.add_argument("--input_path_b", type=str, default=default_path_b,
                        help="输入图像 B 路径")
    parser.add_argument("--output_path", type=str, default="fused_result_max.png",
                        help="输出融合图像路径")
    
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen-Image",
                        help="模型ID (默认: Qwen/Qwen-Image)")
    parser.add_argument("--device", type=str, default=None,
                        help="设备类型 (cuda, cpu) (默认: 自动检测)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="数据类型 (默认: bfloat16)")
    parser.add_argument("--vae_subfolder", type=str, default="vae",
                        help="VAE子文件夹名称")
    parser.add_argument("--fallback_model", type=str, default="Wan-AI/Wan2.1-T2V-14B",
                        help="备用模型ID")
    
    return parser.parse_args()


def load_vae_model(model_id, vae_subfolder, dtype, device, fallback_model):
    """加载VAE模型"""
    logger.info(f"正在加载 Wan VAE (来自 {model_id})...")
    
    if dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "float32":
        torch_dtype = torch.float32
    else:
        raise ValueError(f"不支持的数据类型: {dtype}")
    
    try:
        vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder=vae_subfolder,
            torch_dtype=torch_dtype
        ).to(device)
        vae.eval() # 开启评估模式
        logger.info("VAE 加载成功！")
        return vae, torch_dtype
    except Exception as e:
        logger.warning(f"首选模型加载失败: {e}，尝试加载备用模型...")
        try:
            vae = AutoencoderKLWan.from_pretrained(
                fallback_model, 
                subfolder=vae_subfolder, 
                torch_dtype=torch_dtype
            ).to(device)
            vae.eval()
            return vae, torch_dtype
        except Exception as fallback_e:
            logger.error(f"模型加载完全失败: {fallback_e}")
            sys.exit(1)


def load_image_for_wan(image_path, device, dtype, target_size=None):
    """
    加载并预处理图像以适配Wan VAE。
    :param target_size: (width, height) 元组。如果提供，强制Resize到该尺寸（用于对齐多图）。
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到文件: {image_path}")

    img = Image.open(image_path).convert("RGB")
    
    # 尺寸处理逻辑
    if target_size is not None:
        # 如果指定了目标尺寸（通常是为了匹配图A），直接Resize
        if img.size != target_size:
            logger.info(f"正在调整图像尺寸以匹配基准: {img.size} -> {target_size}")
            img = img.resize(target_size, Image.LANCZOS)
    else:
        # 如果没指定，则进行 16 倍数对齐
        w, h = img.size
        new_w = w - (w % 16)
        new_h = h - (h % 16)
        if new_w != w or new_h != h:
            img = img.resize((new_w, new_h), Image.LANCZOS)
            logger.info(f"图像已裁剪为 16 的倍数: {new_w}x{new_h}")

    # 归一化到 [-1, 1]
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = (img_np * 2.0) - 1.0
    
    # Tensor 转换 [C, H, W]
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
    
    # 增加维度 -> [Batch, Channels, Time, Height, Width]
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(2)
    
    # 返回 Tensor 和 最终尺寸（用于给第二张图做参考）
    return img_tensor.to(device=device, dtype=dtype), img.size


def save_wan_reconstruction(tensor, output_path):
    """保存重建结果"""
    tensor = tensor.squeeze(2).squeeze(0) # 去掉 Time 和 Batch
    tensor = (tensor / 2.0 + 0.5).clamp(0, 1) # 反归一化
    img_np = tensor.permute(1, 2, 0).float().cpu().numpy()
    img_np = (img_np * 255).round().astype("uint8")
    Image.fromarray(img_np).save(output_path)
    logger.info(f"结果已保存至: {output_path}")


def process_fusion(vae, tensor_a, tensor_b):
    """核心融合逻辑：Encode -> Max Fusion -> Decode"""
    with torch.no_grad():
        # 1. 编码 (Encode)
        # 使用 mode() 而不是 sample()。
        # sample() 会加入随机噪声，对于融合任务，我们希望比较的是特征本身的强度，而不是噪声。
        logger.info("正在编码图像 A...")
        latent_a = vae.encode(tensor_a).latent_dist.mode()
        
        logger.info("正在编码图像 B...")
        latent_b = vae.encode(tensor_b).latent_dist.mode()
        
        logger.info(f"Latent A 形状: {latent_a.shape}")
        
        # 2. 融合 (Fusion) - Element-wise Max
        # 逻辑：取两个特征图中每个位置的最大值
        logger.info("正在执行 Max 融合...")
        
        # 方式 A: 直接取数值最大 (Signed Max)
        # 如果 latent 正负值代表特征方向，这可能导致只保留正向特征。
        latent_fused = torch.max(latent_a, latent_b)
        
        # 方式 B: 取绝对值最大 (Activity Max) - 可选备用方案
        # 这种方式通常在多焦融合中更常用，意味着保留“反应更强烈”的特征，无论正负
        # mask = torch.abs(latent_a) > torch.abs(latent_b)
        # latent_fused = torch.where(mask, latent_a, latent_b)
        
        # 这里默认使用你要求的 "取max的方式" (方式 A)
        
        # 3. 解码 (Decode)
        logger.info("正在解码融合后的 Latent...")
        decoded_tensor = vae.decode(latent_fused).sample
        
        return decoded_tensor


def main():
    args = parse_args()
    
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 1. 加载模型
    vae, torch_dtype = load_vae_model(args.model_id, args.vae_subfolder, args.dtype, device, args.fallback_model)
    
    # 2. 加载图像 A
    logger.info(f"加载图像 A: {args.input_path_a}")
    tensor_a, size_a = load_image_for_wan(args.input_path_a, device, torch_dtype)
    
    # 3. 加载图像 B (强制对齐到 A 的尺寸)
    logger.info(f"加载图像 B: {args.input_path_b}")
    tensor_b, _ = load_image_for_wan(args.input_path_b, device, torch_dtype, target_size=size_a)
    
    # 4. 执行融合流程
    fused_tensor = process_fusion(vae, tensor_a, tensor_b)
    
    # 5. 保存
    save_wan_reconstruction(fused_tensor, args.output_path)


if __name__ == "__main__":
    main()