# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University

from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from Evaluate.MI import MI_function
from Evaluate.VIF import vifp_mscale
from Evaluate.niqe import niqe
from Evaluate.simple_metric import *
import matplotlib.colors
import re
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import sys
import platform
from datetime import datetime
def count_parameters(model):
    """
    Calculate the total number of trainable parameters in the model
    
    Args:
        model: Neural network model
    
    Returns:
        int: Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_images(save_path, filename, images, subdirs):
    """
    Save multiple images to specified subdirectories
    
    Args:
        save_path: Base path for saving images
        filename: Name of the file to save
        images: List of images to save
        subdirs: List of subdirectories corresponding to each image
    
    Returns:
        bool: True if all images were saved successfully, False otherwise
    """
    if len(images) != len(subdirs):
        print("Number of images does not match number of subdirectories")
        return False

    for img, subdir in zip(images, subdirs):
        path = os.path.join(save_path, subdir, filename)
        if img is None:
            print(f"Image is None for {subdir}/{filename}")
            return False

        # Check if image is already in uint8 format
        if img.dtype == np.uint8:
            save_img = img
        else:
            # Check the range of the image
            img_min, img_max = img.min(), img.max()

            if img_max <= 1.0 and img_min >= 0.0:
                # Image is in [0, 1] range, scale to [0, 255]
                save_img = (img * 255).astype(np.uint8)
            elif img_max <= 255 and img_min >= 0:
                # Image is already in [0, 255] range, just convert to uint8
                save_img = img.astype(np.uint8)
            else:
                # Image is in an unknown range, normalize to [0, 255]
                save_img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                print(
                    f"Image {subdir}/{filename} had an unexpected range [{img_min}, {img_max}]. Normalized to [0, 255].")

        if not cv2.imwrite(path, save_img):
            print(f"Failed to save image: {path}")
            return False

    return True

def calculate_metrics(fused_image, all_in_focus_gt, estimated_depth, input_depth_map):
    """
    Calculate various evaluation metrics for image fusion and depth estimation
    
    Args:
        fused_image: The fused image
        all_in_focus_gt: All-in-focus ground truth image
        estimated_depth: Estimated depth map
        input_depth_map: Input depth map ground truth
    
    Returns:
        dict: Dictionary containing various evaluation metrics
    """
    metrics = {}

    if all_in_focus_gt is not None:
        # 计算与全清晰图像相关的指标
        metrics['SSIM'] = compare_ssim(fused_image, all_in_focus_gt)
        metrics['PSNR'] = compare_psnr(fused_image, all_in_focus_gt)
        metrics['MSE'] = MSE_function(fused_image, all_in_focus_gt)
        metrics['MAE'] = MAE_function(fused_image, all_in_focus_gt)
        metrics['RMSE'] = RMSE_function(fused_image, all_in_focus_gt)
        metrics['logRMS'] = logRMS_function(fused_image, all_in_focus_gt)
        metrics['abs_rel_error'] = abs_rel_error_function(fused_image, all_in_focus_gt)
        metrics['sqr_rel_error'] = sqr_rel_error_function(fused_image, all_in_focus_gt)
        metrics['VIF'] = vifp_mscale(fused_image, all_in_focus_gt)
        metrics['MI'] = MI_function(fused_image, all_in_focus_gt)
        metrics['NIQE'] = niqe(fused_image)
        metrics['SF'] = SF_function(fused_image)
        metrics['AVG'] = AG_function(fused_image)
        metrics['EN'] = EN_function(fused_image)
        metrics['STD'] = SD_function(fused_image)

    if input_depth_map is not None:
        # 计算与深度图相关的指标
        metrics['depth_mse'] = MSE_function(estimated_depth, input_depth_map)
        metrics['depth_mae'] = MAE_function(estimated_depth, input_depth_map)
    return metrics

def to_image(tensor_batch, epoch, tag, path, nrow=6):
    """
    Convert tensor batch to image and save it
    
    Args:
        tensor_batch: Input tensor batch
        epoch: Current training epoch
        tag: Image tag
        path: Save path
        nrow: Number of images per row in the grid, default is 6
    """
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

    # Ensure the input is a 4D tensor (batch_size, channels, height, width)
    if tensor_batch.dim() == 3:
        tensor_batch = tensor_batch.unsqueeze(0)

    # Normalize the tensor if it's not in the range [0, 1]
    if tensor_batch.min() < 0 or tensor_batch.max() > 1:
        tensor_batch = (tensor_batch - tensor_batch.min()) / (tensor_batch.max() - tensor_batch.min())

    # Create a grid of images
    grid = make_grid(tensor_batch, nrow=nrow, padding=2, normalize=True)

    # Save the grid as an image
    save_image(grid, os.path.join(path, f'{epoch}_{tag}.jpg'))

def resize_to_multiple_of_32(image):
    """
    Resize image to be multiple of 32 in both dimensions
    
    Args:
        image: Input image tensor
    
    Returns:
        tuple: (Resized image, Original image dimensions)
    """
    h, w = image.shape[-2:]
    new_h = ((h - 1) // 32 + 1) * 32
    new_w = ((w - 1) // 32 + 1) * 32
    resized_image = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)
    return resized_image, (h, w)


def gray_to_colormap(img, cmap='rainbow'):
    """
    Convert grayscale image to colormap
    
    Args:
        img: Input grayscale image, must be 2D array
        cmap: Matplotlib colormap name, default is 'rainbow'
    
    Returns:
        ndarray: Converted colormap image in uint8 format
    
    Note:
        - Input image should be 2D
        - Negative values will be set to 0
        - Values less than 1e-10 will be masked as invalid
    """
    assert img.ndim == 2

    img[img < 0] = 0
    mask_invalid = img < 1e-10
    img = img / (img.max() + 1e-8)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.1)
    cmap_m = matplotlib.colormaps.get_cmap(cmap)
    colormap_scaler = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    colormap = (colormap_scaler.to_rgba(img)[:, :, :3] * 255).astype(np.uint8)
    colormap[mask_invalid] = 0
    return colormap

def config_model_dir(resume=False, subdir_name='train_runs'):
    """
    Configure and create model directory for saving training results
    
    Args:
        resume: Boolean flag indicating whether to resume training from existing directory
        subdir_name: Base name for the subdirectory, default is 'train_runs'
    
    Returns:
        str: Path to the model directory
        
    Note:
        - Creates a new numbered directory if resume=False
        - Returns existing directory path if resume=True
        - Directory naming format: {subdir_name}{number} (e.g., train_runs1, train_runs2)
    """
    # Get current script's directory (project directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = script_dir
    # Get path to models directory
    models_dir = os.path.join(project_dir, subdir_name)
    # Create models directory if it doesn't exist (use makedirs to create parent directories)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)

    # Create first directory if none exists
    if not os.path.exists(os.path.join(models_dir, subdir_name+'1')):
        os.makedirs(os.path.join(models_dir, subdir_name+'1'), exist_ok=True)
        return os.path.join(models_dir, subdir_name+'1')
    else:
        # Get existing subdirectories
        sub_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
        
        def extract_number(dir_name):
            numbers = re.findall(r'\d+', dir_name)
            return int(numbers[0]) if numbers else 0
        
        sub_dirs.sort(key=extract_number)
        last_numbers = re.findall(r"\d+", sub_dirs[-1])

        if not resume:
            # Create new directory with incremented number
            next_number = int(last_numbers[0]) + 1 if last_numbers else 1
            new_sub_dir_name = subdir_name + str(next_number)
        else:
            # Use existing directory for resume
            new_sub_dir_name = subdir_name + str(int(last_numbers[0])) if last_numbers else subdir_name + '1'

        model_dir_path = os.path.join(models_dir, new_sub_dir_name)
        
        if not resume:
            os.makedirs(model_dir_path, exist_ok=True)
            
        return model_dir_path

# ==================== 训练输出格式化函数 ====================

def print_banner():
    """打印训练开始的横幅信息"""
    banner = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                              StackMFF V5 Training                           ║
║                     Multi-Focus Image Fusion Neural Network                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)
    print(f"📅 Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💻 Platform: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"🔥 PyTorch: {torch.__version__}")
    print(f"🖥️  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎯 CUDA Version: {torch.version.cuda}")
        print(f"📊 GPU Count: {torch.cuda.device_count()}")
    print("\n" + "="*80)

def print_model_info(model, num_params):
    """打印模型信息"""
    print("\n🧠 MODEL INFORMATION")
    print("─" * 40)
    print(f"📐 Architecture: StackMFF_V3")
    print(f"🔢 Parameters: {num_params:,}")
    print(f"💾 Model Size: {num_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
def print_device_info(device, use_parallel=False, gpu_count=0):
    """打印设备配置信息"""
    print("\n⚡ DEVICE CONFIGURATION")
    print("─" * 40)
    if use_parallel:
        print(f"🔄 Training Mode: DataParallel ({gpu_count} GPUs)")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i) if torch.cuda.is_available() else "Unknown"
            print(f"   └── GPU {i}: {gpu_name}")
    else:
        if device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(device.index) if torch.cuda.is_available() else "Unknown"
            print(f"🎯 Training Mode: Single GPU ({device})")
            print(f"   └── {gpu_name}")
        else:
            print(f"🖥️  Training Mode: CPU ({device})")

def print_dataset_info(train_loader, val_loaders, args):
    """打印数据集信息"""
    print("\n📊 DATASET INFORMATION")
    print("─" * 40)
    print(f"📁 Root Path: {args.datasets_root}")
    print(f"🔧 Image Size: {args.training_image_size}x{args.training_image_size}")
    print(f"📦 Batch Size: {args.batch_size}")
    print(f"👷 Workers: {args.num_workers}")
    print(f"\n📈 Training Datasets:")
    if train_loader:
        print(f"   └── Total Samples: {len(train_loader.dataset):,}")
        print(f"   └── Batches: {len(train_loader):,}")
        for dataset in args.train_datasets:
            print(f"   └── {dataset}")
    else:
        print("   └── No training data found")
    
    print(f"\n📉 Validation Datasets:")
    if val_loaders:
        total_val_samples = sum(len(loader.dataset) for loader in val_loaders)
        print(f"   └── Total Samples: {total_val_samples:,}")
        for i, dataset in enumerate(args.val_datasets):
            if i < len(val_loaders):
                print(f"   └── {dataset}: {len(val_loaders[i].dataset):,} samples")
    else:
        print("   └── No validation data found")

def print_training_config(args, optimizer, scheduler):
    """打印训练配置信息"""
    print("\n⚙️  TRAINING CONFIGURATION")
    print("─" * 40)
    print(f"🔄 Epochs: {args.num_epochs}")
    print(f"📈 Learning Rate: {args.lr}")
    print(f"📉 LR Decay: {args.lr_decay}")
    print(f"🎯 Optimizer: {optimizer.__class__.__name__}")
    print(f"📅 Scheduler: {scheduler.__class__.__name__}")
    print(f"💾 Save Path: {args.save_name}")
    print("\n" + "="*80)

def print_epoch_results(epoch, total_epochs, train_loss, val_results, dataset_names, lr, best_loss, improved, **kwargs):
    """打印epoch结果"""
    print(f"\n{'='*80}")
    print(f"📊 EPOCH {epoch+1}/{total_epochs} SUMMARY")
    print(f"{'='*80}")
    
    if train_loss is not None:
        print(f"🔥 Training Loss: {train_loss:.6f}")
    
    print(f"📚 Learning Rate: {lr:.2e}")
    
    if val_results:
        print(f"\n📈 Validation Results:")
        for i, (dataset, result) in enumerate(zip(dataset_names, val_results)):
            # 检查result是否为序列（如元组或列表），如果不是，则直接使用它作为损失值
            if isinstance(result, (list, tuple)):
                # 如果result是序列且长度>=3
                if len(result) >= 3:
                    # 格式: (val_loss, val_focus_loss, metric_value)
                    val_loss, val_focus_loss, metric_value = result[0], result[1], result[2]
                else:
                    # 兼容旧格式
                    val_loss, val_focus_loss, metric_value = result[0], result[1], 0
            else:
                # 如果result是单个值（如浮点数），则将其作为损失值
                val_loss = result
                val_focus_loss = 0  # 设置默认值
                metric_value = 0  # 设置默认值
            
            status = "🏆" if improved else "📊"
            # 检查是否通过kwargs传递了自定义指标名称和格式
            if 'metric_name' in kwargs and 'metric_format' in kwargs:
                metric_name = kwargs['metric_name']
                metric_format = kwargs['metric_format']
            else:
                # 默认情况下，如果result不是序列，说明只返回了损失值
                if not isinstance(result, (list, tuple)):
                    # 只显示验证损失，不显示额外的指标
                    print(f"  {status} {dataset:15} | Loss: {val_loss:.6f}")
                    continue  # 跳过当前循环迭代
                else:
                    # 默认使用Loss作为指标名称
                    metric_name = "Loss"
                    metric_format = "{:.6f}"
            
            print(f"  {status} {dataset:15} | Loss: {val_loss:.6f} | {metric_name}: {metric_format.format(metric_value)}")
    
    if best_loss:
        status_icon = "🆕" if improved else "📈"
        print(f"\n{status_icon} Best Model Loss: {best_loss:.6f}")
    
    print(f"{'='*80}\n")

def print_training_complete(start_time, model_save_path):
    """打印训练完成信息"""
    end_time = time.time()
    training_time_hours = (end_time - start_time) / 3600
    
    print(f"\n{'🎉'*30}")
    print("🎯 TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'🎉'*30}")
    print(f"⏱️  Total Training Time: {training_time_hours:.2f} hours")
    print(f"💾 Model Saved At: {model_save_path}")
    print(f"🏁 Training finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

def get_depth_transformer_state_dict(model):
    """
    提取模型中DepthTransformer部分的state_dict
    
    Args:
        model: 完整的StackMFF_V5模型（可能被Accelerator、DataParallel或DistributedDataParallel包装）
        
    Returns:
        dict: 仅包含DepthTransformer参数的state_dict
    """
    # 获取原始模型，处理各种包装器情况
    raw_model = model
    
    # 循环解包，处理多种包装器
    # Accelerator 可能使用 'module', '_orig_mod', 或其他属性名
    unwrap_attrs = ['module', '_orig_mod', 'model']
    max_iterations = 10  # 防止无限循环
    iteration = 0
    
    while iteration < max_iterations:
        unwrapped = False
        for attr in unwrap_attrs:
            if hasattr(raw_model, attr):
                raw_model = getattr(raw_model, attr)
                unwrapped = True
                break
        if not unwrapped:
            break
        iteration += 1
    
    # 获取state_dict
    model_state_dict = raw_model.state_dict()
    
    # 调试输出：显示所有键的前缀
    all_prefixes = set()
    for name in model_state_dict.keys():
        prefix = name.split('.')[0]
        all_prefixes.add(prefix)
    print(f"   🔍 State dict contains prefixes: {sorted(all_prefixes)}")
    print(f"   📊 Total parameters in state_dict: {len(model_state_dict)}")
    
    # 提取DepthTransformer部分的参数
    depth_transformer_state_dict = {}
    for name, param in model_state_dict.items():
        if name.startswith('depth_transformer.'):
            depth_transformer_state_dict[name] = param
    
    # 打印调试信息（可以帮助确认是否正确提取）
    print(f"   📦 Extracted {len(depth_transformer_state_dict)} DepthTransformer parameter tensors")
    
    if len(depth_transformer_state_dict) == 0:
        print(f"   ⚠️  警告: 未找到 depth_transformer 参数!")
        print(f"   📋 前5个键名: {list(model_state_dict.keys())[:5]}")
    
    return depth_transformer_state_dict

def load_trainable_model(model_path, model_class, device='cpu'):
    """
    加载只包含可训练参数的模型文件，并正确初始化完整模型
    
    Args:
        model_path (str): 模型文件路径
        model_class: 模型类 (如 StackMFF_V5)
        device (str): 设备名称
        
    Returns:
        model: 加载了可训练参数的完整模型
    """
    # 首先创建完整模型（包含冻结的VAE）
    model = model_class()
    model = model.to(device)
    
    # 加载保存的可训练参数
    trainable_state_dict = torch.load(model_path, map_location=device, weights_only=False)
    
    # 打印调试信息
    print(f"   📂 Loaded checkpoint contains {len(trainable_state_dict)} parameter tensors")
    if len(trainable_state_dict) > 0:
        first_key = list(trainable_state_dict.keys())[0]
        print(f"   🔑 First key in checkpoint: {first_key}")
    
    # 将可训练参数加载到完整模型中
    model_dict = model.state_dict()
    
    # 过滤出只属于depth_transformer的参数
    filtered_state_dict = {k: v for k, v in trainable_state_dict.items() 
                           if k in model_dict and k.startswith('depth_transformer.')}
    
    # 检查是否有匹配的参数
    if len(filtered_state_dict) == 0:
        print(f"   ⚠️  警告: 没有找到匹配的 depth_transformer 参数!")
        print(f"   📋 Checkpoint keys (前5个): {list(trainable_state_dict.keys())[:5]}")
        print(f"   📋 Model keys (前5个): {[k for k in model_dict.keys() if 'depth_transformer' in k][:5]}")
    
    # 更新模型参数
    model_dict.update(filtered_state_dict)
    model.load_state_dict(model_dict)
    
    # 确保模型处于评估模式
    model.eval()
    
    print(f"✅ Successfully loaded trainable parameters from {model_path}")
    print(f"   📊 Loaded {len(filtered_state_dict)} parameter tensors")
    
    return model