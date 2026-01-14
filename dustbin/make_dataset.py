#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import cv2
import numpy as np
import argparse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm

def generate_ordered_stack_and_soft_gt(img, img_depth, num_regions):
    """
    核心生成函数：生成有序的模糊图像栈 和 对应的线性 Soft GT。
    
    Args:
        img (np.array): 全清晰原图 [H, W, 3]
        img_depth (np.array): 深度图 [H, W] (0-255)
        num_regions (int): 将深度划分为多少层 (例如 5, 10)
        
    Returns:
        ordered_layers (list): 包含 num_regions 张图片的列表，顺序为从近到远。
        soft_gt (np.array): 形状为 [num_regions, H, W] 的概率图，类型 float16。
    """
    H, W = img_depth.shape
    
    # ==========================================
    # 第一步：准备模糊金字塔 (Blur Pyramid)
    # ==========================================
    # 我们预先计算好不同程度的模糊图，后续直接查表取用，提高效率。
    # 模糊等级从 0 (清晰) 到 num_regions-1 (最模糊)
    # Kernel size 必须是奇数：1, 3, 5, 7, ...
    blurred_pyramid = []
    for i in range(num_regions):
        k_size = 2 * i + 1
        if k_size == 1:
            blurred_pyramid.append(img)
        else:
            blurred_pyramid.append(cv2.GaussianBlur(img, (k_size, k_size), 0))

    # ==========================================
    # 第二步：合成有序图像栈 (Image Synthesis)
    # ==========================================
    # 1. 将 0-255 的深度图量化为 0 到 N-1 的离散整数层级
    #    这模拟了物理世界中，相机聚焦在某一个特定的焦平面上
    thresholds = np.linspace(0, 255, num_regions + 1)
    # digitize 返回 1..N, 减1 变为 0..N-1
    quantized_depth_map = np.digitize(img_depth, thresholds) - 1
    quantized_depth_map = np.clip(quantized_depth_map, 0, num_regions - 1)
    
    ordered_layers = []
    
    # 循环生成每一层 (Layer 0 -> Layer N-1)
    for focus_idx in range(num_regions):
        current_layer_img = np.zeros_like(img)
        
        # 对于当前聚焦层 focus_idx，遍历图像中所有可能的真实深度 actual_depth_idx
        for actual_depth_idx in range(num_regions):
            # 找出属于这个真实深度的像素掩码
            mask = (quantized_depth_map == actual_depth_idx)
            if not np.any(mask):
                continue
            
            # 物理逻辑：
            # 如果 聚焦层(focus_idx) == 真实层(actual_depth_idx)，距离为0，清晰。
            # 距离越远，模糊程度越高。
            dist = abs(focus_idx - actual_depth_idx)
            
            # 限制最大模糊等级不越界
            blur_level = min(dist, len(blurred_pyramid) - 1)
            
            # 填入像素
            current_layer_img[mask] = blurred_pyramid[blur_level][mask]
            
        ordered_layers.append(current_layer_img)

    # ==========================================
    # 第三步：生成线性 Soft GT (Linear Triangular Distribution)
    # ==========================================
    # 这里的逻辑是：概率只分配给最近的 1 到 2 个图层，远处严格为 0。
    
    # 1. 将深度图映射到连续的索引坐标系 [0.0, (N-1).0]
    #    例如 10层：0 -> 0.0, 128 -> 4.5, 255 -> 9.0
    continuous_depth_val = (img_depth.astype(np.float32) / 255.0) * (num_regions - 1)
    
    # 2. 创建层索引的 3D 网格: [0, 1, 2, ..., N-1]
    #    Shape: [num_regions, H, W]
    layer_indices = np.arange(num_regions).reshape(num_regions, 1, 1)
    
    # 3. 计算每个像素的真实深度距离每一层有多远
    #    Distance Shape: [num_regions, H, W]
    distance_map = np.abs(layer_indices - continuous_depth_val)
    
    # 4. 线性三角分布公式
    #    Prob = 1 - distance (如果距离小于1)
    #    Prob = 0 (如果距离大于1)
    #    物理含义：如果深度是 4.5，Layer 4 距离0.5(Prob=0.5)，Layer 5 距离0.5(Prob=0.5)，其他层Prob=0
    # 原始 (Radius = 1.0, 只有2层)
    soft_gt = np.maximum(0, 1.0 - distance_map)

    # # 更软 (Radius = 2.0, 会有4层有值)
    # # 物理含义：允许一定程度的离焦模糊参与融合，增加平滑度，降低锐度
    # soft_gt = np.maximum(0, 1.0 - (distance_map / 2.0))
    
    # 5. 归一化 (Normalize)
    #    虽然理论上 sum 应该是 1，但防止浮点误差，做一次归一化
    gt_sum = np.sum(soft_gt, axis=0, keepdims=True)
    soft_gt = soft_gt / (gt_sum + 1e-8)
    
    # 6. 转为 float16 以节省大量硬盘空间 (概率值不需要 float32 精度)
    soft_gt = soft_gt.astype(np.float16)
    
    return ordered_layers, soft_gt

def process_single_image(args):
    """
    单个图像处理函数 (Worker Function)
    """
    pic_path, depth_path, output_path, num_regions_list = args
    name = "unknown"
    
    try:
        # 获取文件名
        filename = os.path.basename(pic_path)
        name, _ = os.path.splitext(filename)

        # 读取原图
        img = cv2.imread(pic_path)
        if img is None:
            return False, name, "Failed to load image"

        # 读取深度图 (尝试 png 和 jpg)
        img_depth_path = os.path.join(depth_path, name + '.png')
        if not os.path.exists(img_depth_path):
             img_depth_path = os.path.join(depth_path, name + '.jpg')
        
        if not os.path.exists(img_depth_path):
            return False, name, "Depth map not found"
            
        img_depth = cv2.imread(img_depth_path, 0) # 0 表示读取灰度
        if img_depth is None:
            return False, name, "Failed to load depth map"

        # 尺寸对齐：如果深度图和原图尺寸不一致，Resize深度图
        if img.shape[:2] != img_depth.shape[:2]:
            img_depth = cv2.resize(img_depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 随机选择一个层数 (增加数据的多样性)
        num_regions = random.choice(num_regions_list)
        
        # === 核心生成逻辑 ===
        ordered_stack, soft_gt = generate_ordered_stack_and_soft_gt(img, img_depth, num_regions)

        # === 保存路径设置 ===
        # 1. 图像栈: output/focus_stack/<name>/0.jpg, 1.jpg ...
        stack_dir = os.path.join(output_path, 'focus_stack', name)
        os.makedirs(stack_dir, exist_ok=True)
        
        # 2. GT: output/soft_gt/<name>.npy
        gt_dir = os.path.join(output_path, 'soft_gt')
        os.makedirs(gt_dir, exist_ok=True)

        # 保存图片
        for i, layer_img in enumerate(ordered_stack):
            cv2.imwrite(os.path.join(stack_dir, f'{i}.jpg'), layer_img)
            
        # 保存 GT
        np.save(os.path.join(gt_dir, f'{name}.npy'), soft_gt)
        
        # 可选：保存层数信息，方便 Dataset 读取时知道这一组有多少层
        # 或者直接在 Dataset 里用 len(glob) 判断
        
        return True, name, f"Layers: {num_regions}"

    except Exception as e:
        return False, name, str(e)

def main():
    parser = argparse.ArgumentParser(description="Generate Ordered Multi-Focus Dataset for OT Training")
    parser.add_argument("--original_path", type=str, default="C:/Users/dell/Desktop/aif", help="Path to sharp images folder")
    parser.add_argument("--depth_path", type=str, default="C:/Users/dell/Desktop/depth", help="Path to depth maps folder")
    parser.add_argument("--output_path", type=str, default="C:/Users/dell/Desktop/output", help="Output root folder")
    parser.add_argument("--max_workers", type=int, default=8, help="Number of threads")
    
    args = parser.parse_args()
    
    # 定义生成的层数范围
    # 建议：OT 训练初期不要太深，4-10层是比较合理的范围
    num_regions_list = list(range(4, 11)) 
    
    # 搜集图片
    extensions = ['*.jpg', '*.png', '*.jpeg']
    original_images = []
    for ext in extensions:
        original_images.extend(glob.glob(os.path.join(args.original_path, ext)))
    
    if not original_images:
        print("No images found!")
        return
        
    print(f"Found {len(original_images)} images.")
    print(f"Outputting to: {args.output_path}")
    print(f"Using Linear Triangular GT Distribution.")

    # 准备任务参数
    task_args = [(p, args.depth_path, args.output_path, num_regions_list) for p in original_images]
    
    success_cnt = 0
    fail_cnt = 0
    
    # 多线程处理
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_single_image, arg): arg[0] for arg in task_args}
        
        with tqdm(total=len(original_images), desc="Generating Data") as pbar:
            for future in as_completed(futures):
                success, name, msg = future.result()
                if success:
                    success_cnt += 1
                else:
                    fail_cnt += 1
                    # print(f"Failed {name}: {msg}") # 避免刷屏
                pbar.update(1)
                
    print(f"\nProcessing Done.")
    print(f"Success: {success_cnt}")
    print(f"Failed: {fail_cnt}")

if __name__ == "__main__":
    multiprocessing.freeze_support() # Windows下需要
    main()