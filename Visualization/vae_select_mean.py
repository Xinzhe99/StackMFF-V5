import torch
from diffusers import AutoencoderKLWan
from PIL import Image
import numpy as np
import os
import argparse
import sys
import logging

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="使用Wan VAE对两张图像进行潜空间均值(Mean)融合")
    
    # 默认路径
    default_path_a = "/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/project_image_stack_fusion/StackMFFV5/test_A.jpg"
    default_path_b = "/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/project_image_stack_fusion/StackMFFV5/test_B.jpg"
    
    parser.add_argument("--input_path_a", type=str, default=default_path_a, help="输入图像 A 路径")
    parser.add_argument("--input_path_b", type=str, default=default_path_b, help="输入图像 B 路径")
    
    # 修改：默认输出文件名改为 fused_result_mean.png
    parser.add_argument("--output_path", type=str, default="fused_result_mean.png",
                        help="输出融合图像路径")
    
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen-Image", help="模型ID")
    parser.add_argument("--device", type=str, default=None, help="设备类型")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--vae_subfolder", type=str, default="vae", help="VAE子文件夹名称")
    parser.add_argument("--fallback_model", type=str, default="Wan-AI/Wan2.1-T2V-14B", help="备用模型ID")
    
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
            logger.info(f"正在调整图像尺寸以匹配基准: {img.size} -> {target_size}")
            img = img.resize(target_size, Image.LANCZOS)
    else:
        w, h = img.size
        new_w = w - (w % 16)
        new_h = h - (h % 16)
        if new_w != w or new_h != h:
            img = img.resize((new_w, new_h), Image.LANCZOS)
            logger.info(f"图像已裁剪为 16 的倍数: {new_w}x{new_h}")

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


def process_fusion_mean(vae, tensor_a, tensor_b):
    """核心融合逻辑：Encode -> Mean Fusion -> Decode"""
    with torch.no_grad():
        logger.info("正在编码图像 A...")
        latent_a = vae.encode(tensor_a).latent_dist.mode()
        
        logger.info("正在编码图像 B...")
        latent_b = vae.encode(tensor_b).latent_dist.mode()
        
        # ==========================================
        # 修改点：执行均值融合 (Average / Mean)
        # ==========================================
        logger.info("正在执行 Mean (均值) 融合...")
        latent_fused = (latent_a + latent_b) / 2.0
        
        logger.info("正在解码融合后的 Latent...")
        decoded_tensor = vae.decode(latent_fused).sample
        
        return decoded_tensor


def main():
    args = parse_args()
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    vae, torch_dtype = load_vae_model(args.model_id, args.vae_subfolder, args.dtype, device, args.fallback_model)
    
    logger.info(f"加载图像 A: {args.input_path_a}")
    tensor_a, size_a = load_image_for_wan(args.input_path_a, device, torch_dtype)
    
    logger.info(f"加载图像 B: {args.input_path_b}")
    tensor_b, _ = load_image_for_wan(args.input_path_b, device, torch_dtype, target_size=size_a)
    
    # 使用均值融合处理
    fused_tensor = process_fusion_mean(vae, tensor_a, tensor_b)
    
    save_wan_reconstruction(fused_tensor, args.output_path)


if __name__ == "__main__":
    main()