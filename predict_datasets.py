# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
"""
StackMFF V5 批量推理脚本
用于对多个数据集中的多个图像栈进行批量融合推理
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import argparse
from network import StackMFF_V5
from tools import load_trainable_model
from tqdm import tqdm
import time
from datetime import datetime


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


def preprocess_image(image_path, target_size=None):
    """
    预处理单张图像
    Args:
        image_path: 图像路径
        target_size: 目标尺寸 (w, h)，如果提供则将图像调整为此尺寸
    Returns:
        预处理后的tensor和原始尺寸
    """
    # 加载图像
    img = Image.open(image_path).convert('RGB')
    
    # 记录原始尺寸
    original_size = img.size  # (w, h)
    
    # 如果提供了目标尺寸，则调整图像到目标尺寸
    if target_size is not None:
        img = img.resize(target_size, Image.LANCZOS)
    else:
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
    # 首先获取所有图像的尺寸，以确定统一尺寸
    temp_sizes = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        # 调整为16的倍数
        if w % 16 != 0 or h % 16 != 0:
            new_w = (w // 16) * 16  # 确保是16的倍数
            new_h = (h // 16) * 16
            temp_sizes.append((new_w, new_h))
        else:
            temp_sizes.append((w, h))
    
    # 找到最小的公共尺寸作为目标尺寸，以避免内存问题
    if temp_sizes:
        min_width = min(size[0] for size in temp_sizes)
        min_height = min(size[1] for size in temp_sizes)
        target_size = (min_width, min_height)
    else:
        raise ValueError("无法确定图像尺寸")
    
    # 预处理所有图像到统一尺寸
    processed_images = []
    original_sizes = []
    for img_path in image_paths:
        img_tensor, original_size = preprocess_image(img_path, target_size=target_size)
        processed_images.append(img_tensor)
        original_sizes.append(original_size)  # 记录第一张图像的原始尺寸作为输出尺寸参考
    
    # 将预处理的图像合并成一个张量 [N, C, H, W]
    # 每个img_tensor的形状是[1, C, H, W]
    image_stack_flat = torch.stack([img.squeeze(0) for img in processed_images], dim=0)  # [N, C, H, W]
    N = image_stack_flat.size(0)
    
    model.eval()
    
    if multi_gpu:
        # 如果启用了多GPU，我们只需要并行化 VAE 编码部分
        if isinstance(model, nn.DataParallel):
            # 获取原始模型
            raw_model = model.module
            
            # 创建临时 wrapper 并封装进 DataParallel
            encoder_parallel = nn.DataParallel(VAEEncoderWrapper(raw_model), device_ids=model.device_ids)
            encoder_parallel = encoder_parallel.to(device)
            
            # 并行编码 [N, C, H, W] -> [N, LatentC, LatentH, LatentW]
            with torch.no_grad():
                latents = encoder_parallel(image_stack_flat.to(device)) # [N, C, H, W]
            
            # 融合 (在主 GPU 上)
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


def get_image_stacks_from_dataset(dataset_path):
    """
    从数据集目录中获取所有图像栈
    Args:
        dataset_path: 数据集路径
    Returns:
        图像栈列表，每个元素是图像路径列表
    """
    image_stacks = []
    stack_names = []
    
    # 遍历数据集目录下的所有子目录（每个子目录是一个图像栈）
    for stack_name in os.listdir(dataset_path):
        stack_path = os.path.join(dataset_path, stack_name)
        
        # 检查是否为目录
        if os.path.isdir(stack_path):
            # 获取该目录下的所有图像文件
            image_files = get_image_files_from_directory(stack_path)
            
            # 如果该目录包含至少2张图像，则认为是一个有效的图像栈
            if len(image_files) >= 2:
                image_stacks.append(image_files)
                stack_names.append(stack_name)
    
    return image_stacks, stack_names


def process_single_stack(model, image_paths, device, original_size, multi_gpu=False):
    """
    处理单个图像栈
    Args:
        model: 模型
        image_paths: 图像路径列表
        device: 设备
        original_size: 原始尺寸
        multi_gpu: 是否多GPU模式
    Returns:
        融合图像tensor和推理时间
    """
    # 记录推理开始时间
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    
    # 进行推理
    try:
        fused_image, _ = inference_single_image_stack(model, image_paths, device, multi_gpu=multi_gpu)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"CUDA内存不足，尝试清理缓存...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # 再次尝试，如果还是失败则抛出异常
            fused_image, _ = inference_single_image_stack(model, image_paths, device, multi_gpu=multi_gpu)
        else:
            raise e
    
    # 记录推理结束时间
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()
    
    inference_time = end_time - start_time
    
    return fused_image, inference_time


def main():
    parser = argparse.ArgumentParser(description="StackMFF V5 批量多焦点图像栈融合推理")
    parser.add_argument("--model_path", type=str, default='weights/stackmffv5.pth',
                        help="训练好的模型权重路径")
    parser.add_argument("--data_root", type=str, default='/home/ot/Students/xxz/datasets/test_datasets',
                        help="数据根目录路径，包含多个数据集子目录")
    parser.add_argument("--output_dir", type=str, default='stackmffv5_outputs',
                        help="融合结果保存路径")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="推理设备 (默认: cuda:0)")
    parser.add_argument("--gpus", type=str, default=None,
                        help="使用的GPU列表，使用逗号分隔，例如 '0,1'。")
    parser.add_argument("--datasets", nargs='+', default=['Mobile Depth','Middlebury','Road-MF','FlyingThings3D'],
                        help="要处理的数据集名称列表，如果未指定则处理所有数据集")

    args = parser.parse_args()
    
    # 获取GPU列表
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
        # 设置设备
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在 {args.model_path}")
        return
    
    # 检查数据根目录是否存在
    if not os.path.exists(args.data_root):
        print(f"错误: 数据根目录不存在 {args.data_root}")
        return
    
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
    
    # 获取所有数据集目录
    all_datasets = [d for d in os.listdir(args.data_root) 
                   if os.path.isdir(os.path.join(args.data_root, d))]
    
    # 如果指定了特定数据集，则只处理这些数据集
    if args.datasets:
        datasets_to_process = [d for d in args.datasets if d in all_datasets]
        invalid_datasets = [d for d in args.datasets if d not in all_datasets]
        if invalid_datasets:
            print(f"警告: 以下数据集不存在: {invalid_datasets}")
    else:
        datasets_to_process = all_datasets
    
    print(f"找到 {len(datasets_to_process)} 个数据集: {datasets_to_process}")
    
    # 创建输出根目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = os.path.join(args.output_dir, f"batch_results_{timestamp}")
    os.makedirs(output_root, exist_ok=True)
    
    # 统计信息
    total_stacks_processed = 0
    total_inference_time = 0
    
    # 遍历每个数据集
    for dataset_name in datasets_to_process:
        print(f"\n处理数据集: {dataset_name}")
        
        dataset_path = os.path.join(args.data_root,dataset_name,'image stack')
        
        # 获取该数据集中的所有图像栈
        image_stacks, stack_names = get_image_stacks_from_dataset(dataset_path)
        
        print(f"在数据集 {dataset_name} 中找到 {len(image_stacks)} 个图像栈")
        
        if len(image_stacks) == 0:
            print(f"数据集 {dataset_name} 中没有找到有效的图像栈，跳过...")
            continue
        
        # 为当前数据集创建输出目录
        dataset_output_dir = os.path.join(output_root, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # 处理当前数据集中的每个图像栈
        dataset_time = 0
        dataset_processed = 0
        
        for i, (image_paths, stack_name) in enumerate(tqdm(zip(image_stacks, stack_names), 
                                                          total=len(image_stacks),
                                                          desc=f"处理 {dataset_name}")):
            # 检查图像是否存在
            valid_images = [img_path for img_path in image_paths if os.path.exists(img_path)]
            if len(valid_images) != len(image_paths):
                print(f"警告: 图像栈 {stack_name} 中部分图像不存在，跳过...")
                continue
            
            if len(valid_images) < 2:
                print(f"警告: 图像栈 {stack_name} 中图像数量少于2张，跳过...")
                continue
            
            # 处理单个图像栈
            try:
                # 获取第一张图像的原始尺寸用于后续恢复
                first_img = Image.open(valid_images[0])
                original_size = first_img.size
                
                fused_image, inference_time = process_single_stack(model, valid_images, device, original_size, multi_gpu=is_multi_gpu)
                
                # 生成输出文件名
                output_filename = f"{stack_name}.png"
                output_path = os.path.join(dataset_output_dir, output_filename)
                
                # 保存结果，resize回原始尺寸
                save_tensor_as_image(fused_image, output_path, original_size)
                
                # 更新统计信息
                dataset_time += inference_time
                dataset_processed += 1
                total_inference_time += inference_time
                total_stacks_processed += 1
                
                # 定期清理CUDA缓存以避免内存累积
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"处理图像栈 {stack_name} 时出错: {str(e)}")
                # 即使出错也要清理缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        
        # 输出当前数据集的统计信息
        if dataset_processed > 0:
            avg_time = dataset_time / dataset_processed
            print(f"数据集 {dataset_name} 处理完成: {dataset_processed} 个图像栈，"
                  f"平均推理时间: {avg_time:.4f} 秒")
        else:
            print(f"数据集 {dataset_name} 没有成功处理任何图像栈")
    
    # 输出总体统计信息
    if total_stacks_processed > 0:
        overall_avg_time = total_inference_time / total_stacks_processed
        print(f"\n批量处理完成！")
        print(f"总共处理: {total_stacks_processed} 个图像栈")
        print(f"总推理时间: {total_inference_time:.4f} 秒")
        print(f"平均推理时间: {overall_avg_time:.4f} 秒")
        print(f"结果保存在: {output_root}")
    else:
        print("\n没有成功处理任何图像栈")


if __name__ == "__main__":
    main()