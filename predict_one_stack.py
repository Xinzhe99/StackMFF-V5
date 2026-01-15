# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
"""
StackMFF V5 多焦点图像栈融合推理脚本
用于加载训练好的DepthTransformer模型并进行图像栈融合推理
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import argparse
from network import StackMFF_V5
from tools import load_trainable_model


def preprocess_image(image_path):
    """
    预处理单张图像
    Args:
        image_path: 图像路径
    Returns:
        预处理后的tensor和原始尺寸
    """
    # 加载图像
    img = Image.open(image_path).convert('RGB')
    
    # 记录原始尺寸
    original_size = img.size  # (w, h)
    
    # 检查图像尺寸是否为16的倍数，如果不是则调整为16的倍数
    w, h = img.size
    if w % 16 != 0 or h % 16 != 0:
        new_w = w - (w % 16)
        new_h = h - (h % 16)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    
    # 转换为numpy数组并归一化到[-1, 1]
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # [C, H, W]
    img_tensor = (img_tensor * 2.0) - 1.0  # [-1, 1]
    
    return img_tensor.unsqueeze(0), original_size  # [1, C, H, W] and (w, h)


class VAEEncoderWrapper(nn.Module):
    """
    Wrapper to enable DataParallel on VAE encoding only
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # x: [B, C, H, W], where B is number of images in this context
        # Convert to [B, C, 1, H, W] as expected by encode_one_step
        if x.dim() == 4:
            x = x.unsqueeze(2)
        return self.model.encode_one_step(x)


def inference_single_image_stack(model, image_paths, device='cuda:0', multi_gpu=False):
    """
    对图像栈进行推理
    Args:
        model: 加载了权重的模型
        image_paths: 图像路径列表
        device: 推理设备
        multi_gpu: 是否启用了多GPU (DataParallel wrapped model)
    Returns:
        融合后的图像tensor和原始尺寸
    """
    # 预处理所有图像
    processed_images = []
    original_sizes = []
    for img_path in image_paths:
        img_tensor, original_size = preprocess_image(img_path)
        processed_images.append(img_tensor)
        original_sizes.append(original_size)  # 记录第一张图像的原始尺寸作为输出尺寸参考
    
    # 将预处理的图像合并成一个张量 [N, C, H, W]
    # 每个img_tensor的形状是[1, C, H, W]
    image_stack_flat = torch.stack([img.squeeze(0) for img in processed_images], dim=0)  # [N, C, H, W]
    N = image_stack_flat.size(0)
    
    model.eval()
    
    if multi_gpu:
        # 如果启用了多GPU，我们只需要并行化 VAE 编码部分
        # 此时 model 应该是 nn.DataParallel 或者是原始模型
        # 但如果是 DataParallel(StackMFF_V5)，它期望 [B, N, ...] 且 B 分割。
        # 这里我们需要手动处理：
        
        # 1. 准备并行编码器
        if isinstance(model, nn.DataParallel):
            # 获取原始模型
            raw_model = model.module
            # 如果 model 已经是 DataParallel，通常意味着用户原本想直接 run，但在 batch=1 时失效
            # 我们这里利用 DataParallel 来运行编码器 wrapper
            
            # 创建临时 wrapper 并封装进 DataParallel
            # 注意: 这里有点动态，但为了利用 device_ids
            encoder_parallel = nn.DataParallel(VAEEncoderWrapper(raw_model), device_ids=model.device_ids)
            encoder_parallel = encoder_parallel.to(device)
            
            # 2. 并行编码 [N, C, H, W] -> [N, LatentC, LatentH, LatentW]
            # DataParallel 会自动切分 N 到不同 GPU
            print(f"并行编码中 (N={N})...")
            with torch.no_grad():
                latents = encoder_parallel(image_stack_flat.to(device)) # [N, C, H, W]
            
            # 3. 融合 (在主 GPU 上)
            # 整理形状 [1, N, C, H, W]
            latents = latents.unsqueeze(0) # [1, N, C, H, W]
            
            # 使用 raw_model 的后续部分
            with torch.no_grad():
                # DepthTransformer
                fused_latent = raw_model.depth_transformer(latents)
                
                # Decode (单个图像，不需要并行)
                fused_latent_with_time = fused_latent.unsqueeze(2).to(dtype=raw_model.vae_dtype)
                fused_output = raw_model.vae.decode(fused_latent_with_time).sample
                fused_output = fused_output.squeeze(2).to(dtype=image_stack_flat.dtype)
                
    else:
        # 单 GPU 模式
        # 使用 chunk_size 防止 OOM
        image_stack = image_stack_flat.unsqueeze(0).to(device) # [1, N, C, H, W]
        with torch.no_grad():
            # 使用新添加的 chunk_size 参数，例如 4 或 8，根据显存调整
            fused_output = model(image_stack, chunk_size=4)
    
    # 返回融合结果和原始尺寸（使用第一张图像的原始尺寸）
    return fused_output, original_sizes[0] if original_sizes else None



def save_tensor_as_image(tensor, save_path, original_size=None):
    """
    将tensor保存为图像
    Args:
        tensor: 图像tensor [B, C, H, W]
        save_path: 保存路径
        original_size: 原始图像尺寸 (w, h)，如果提供则将输出resize回原始尺寸
    """
    # 转换到CPU并获取numpy数组
    img_tensor = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H, W, C]
    img_tensor = (img_tensor + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    img_tensor = np.clip(img_tensor, 0, 1)  # 确保值在[0, 1]范围内
    img_tensor = (img_tensor * 255).astype(np.uint8)  # [0, 255]
    
    # 创建PIL图像
    img_pil = Image.fromarray(img_tensor)
    
    # 如果提供了原始尺寸，则将图像resize回原始尺寸
    if original_size is not None:
        img_pil = img_pil.resize(original_size, Image.LANCZOS)
    
    # 保存图像
    img_pil.save(save_path)
    print(f"融合图像已保存到: {save_path}")


def get_image_files_from_directory(directory):
    """
    从目录中获取所有图像文件
    Args:
        directory: 图像目录路径
    Returns:
        图像文件路径列表
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = []
    
    for filename in os.listdir(directory):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(directory, filename))
    
    # 按文件名排序以确保一致性
    image_files.sort()
    return image_files


def main():
    parser = argparse.ArgumentParser(description="StackMFF V5 多焦点图像栈融合推理")
    parser.add_argument("--model_path", type=str, default='weights/stackmffv5.pth',
                        help="训练好的模型权重路径")
    parser.add_argument("--input_dir", type=str, default='/home/ot/Students/xxz/datasets/test_datasets/Mobile Depth/image stack/keyboard',
                        help="包含图像栈的目录路径")
    parser.add_argument("--output_dir", type=str, default='outputs',
                        help="融合结果保存路径")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="推理设备 (默认: cuda:0)")
    parser.add_argument("--gpus", type=str, default=None,
                        help="使用的GPU列表，使用逗号分隔，例如 '0,1'。")
    
    args = parser.parse_args()
    
    # 获取目录中的所有图像文件
    image_paths = get_image_files_from_directory(args.input_dir)
    
    if len(image_paths) < 2:
        print(f"错误: 目录 {args.input_dir} 中图像数量少于2张，无法进行融合")
        return
    
    print(f"找到 {len(image_paths)} 张图像:")

    # 检查输入图像是否存在
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"错误: 图像文件不存在 {img_path}")
            return
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在 {args.model_path}")
        return
    
    # 设置设备
    if args.gpus:
        try:
            device_ids = [int(x.strip()) for x in args.gpus.split(',')]
            device = torch.device(f"cuda:{device_ids[0]}")
            print(f"启用多GPU模式，主设备: {device}, GPU列表: {device_ids}")
        except ValueError:
            print("错误: --gpus 参数格式不正确")
            return
    else:
        device_ids = None
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
    
    # 创建完整模型
    print("创建模型...")
    model = StackMFF_V5()
    
    # 加载训练好的可训练参数
    print("加载训练好的模型参数...")
    model = load_trainable_model(args.model_path, StackMFF_V5, device=device)
    
    # 移动模型到设备
    model = model.to(device)
    
    is_multi_gpu = False
    if device_ids and len(device_ids) > 1:
        print(f"使用DataParallel封装模型，GPUs: {device_ids}")
        model = nn.DataParallel(model, device_ids=device_ids)
        is_multi_gpu = True
    
    # 执行推理
    print("开始推理...")
    fused_image, original_size = inference_single_image_stack(model, image_paths, device, multi_gpu=is_multi_gpu)
    
    # 使用输入目录的名称作为输出文件名
    input_dir_name = os.path.basename(os.path.normpath(args.input_dir))
    output_filename = f"{input_dir_name}.png"
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, output_filename)
    
    # 保存结果，如果original_size存在则resize回原始尺寸
    save_tensor_as_image(fused_image, output_path, original_size)
    
    print("推理完成！")


if __name__ == "__main__":
    main()