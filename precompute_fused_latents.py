"""
预处理脚本：预先计算所有图像栈的融合隐变量和GT图像隐变量
运行一次即可，后续训练直接加载预计算的隐变量，无需FusionEncoder

保存内容：
- fused_latents: 融合后的隐变量，作为ControlNet的条件 c_img
- gt_latents: GT图像(AiF)的隐变量，作为diffusion的目标 z_0
- stack_names: 对应的图像栈名称（用于追溯）
- stack_sizes: 图像栈大小

使用方法：
    python precompute_fused_latents.py --config configs/train/train_stage2.yaml --output_dir precomputed_latents --batch_size 8
"""

import os
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from argparse import ArgumentParser
from model.fusion_encoder import FusionEncoder
from torchvision import transforms
from PIL import Image
import numpy as np
from collections import defaultdict


def get_stack_paths(datasets_root, train_datasets):
    """
    收集所有图像栈和GT图像的路径
    
    Returns:
        list of dict: 每个dict包含 {stack_dir, gt_path, stack_name, dataset_name, stack_size}
    """
    all_samples = []
    
    for dataset_name in train_datasets:
        dataset_path = os.path.join(datasets_root, dataset_name, 'TR')
        stack_root = os.path.join(dataset_path, 'focus_stack')
        gt_root = os.path.join(dataset_path, 'AiF')
        
        if not os.path.exists(stack_root):
            print(f"⚠️  Focus stack directory not found: {stack_root}")
            continue
        if not os.path.exists(gt_root):
            print(f"⚠️  AiF directory not found: {gt_root}")
            continue
            
        # 遍历所有图像栈
        for stack_name in sorted(os.listdir(stack_root)):
            stack_dir = os.path.join(stack_root, stack_name)
            if not os.path.isdir(stack_dir):
                continue
            
            # 预先计算 stack_size（用于分组）
            image_files = [f for f in os.listdir(stack_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.bmp')) and f != 'layer_order.npy']
            stack_size = len(image_files)
            if stack_size == 0:
                continue
                
            # 查找对应的GT图像
            gt_path = None
            for ext in ['.png', '.jpg', '.bmp', '.PNG', '.JPG', '.BMP']:
                candidate_path = os.path.join(gt_root, stack_name + ext)
                if os.path.exists(candidate_path):
                    gt_path = candidate_path
                    break
            
            if gt_path is None:
                print(f"⚠️  GT image not found for stack: {stack_name}")
                continue
                
            all_samples.append({
                'stack_dir': stack_dir,
                'gt_path': gt_path,
                'stack_name': stack_name,
                'dataset_name': dataset_name,
                'stack_size': stack_size
            })
    
    return all_samples


def load_image_stack(stack_dir, transform, image_size):
    """
    加载图像栈中的所有图像
    
    Returns:
        torch.Tensor: [N, C, H, W]
        int: stack_size
    """
    images = []
    image_files = sorted(
        [f for f in os.listdir(stack_dir) 
         if f.lower().endswith(('.png', '.jpg', '.bmp')) and f != 'layer_order.npy'],
        key=lambda x: int(''.join(filter(str.isdigit, x)) or '0')
    )
    
    for img_name in image_files:
        img_path = os.path.join(stack_dir, img_name)
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            img = img.resize((image_size, image_size), Image.BILINEAR)
            img_tensor = transform(img)
            images.append(img_tensor)
    
    if not images:
        return None, 0
        
    return torch.stack(images), len(images)


def load_gt_image(gt_path, transform, image_size):
    """
    加载GT图像
    
    Returns:
        torch.Tensor: [C, H, W]
    """
    with Image.open(gt_path) as img:
        img = img.convert('RGB')
        img = img.resize((image_size, image_size), Image.BILINEAR)
        return transform(img)


def group_samples_by_stack_size(all_samples):
    """
    按 stack_size 分组样本，以便批量处理
    
    Returns:
        dict: {stack_size: [sample_info, ...]}
    """
    groups = defaultdict(list)
    for sample in all_samples:
        groups[sample['stack_size']].append(sample)
    return groups


def process_batch(batch_samples, fusion_encoder, transform, image_size, device):
    """
    批量处理一组样本（必须具有相同的 stack_size）
    
    Returns:
        fused_latents: [B, 4, H, W]
        gt_latents: [B, 4, H, W]
        valid_indices: 成功处理的样本索引
    """
    batch_stacks = []
    batch_gts = []
    valid_indices = []
    
    # 加载batch中的所有图像
    for i, sample_info in enumerate(batch_samples):
        try:
            image_stack, stack_size = load_image_stack(
                sample_info['stack_dir'], transform, image_size
            )
            if image_stack is None:
                continue
                
            gt_image = load_gt_image(sample_info['gt_path'], transform, image_size)
            
            batch_stacks.append(image_stack)
            batch_gts.append(gt_image)
            valid_indices.append(i)
        except Exception as e:
            print(f"Error loading {sample_info['stack_name']}: {e}")
            continue
    
    if not batch_stacks:
        return None, None, []
    
    # 堆叠成batch
    batch_stacks = torch.stack(batch_stacks).to(device)  # [B, N, C, H, W]
    batch_gts = torch.stack(batch_gts).to(device)        # [B, C, H, W]
    
    with torch.no_grad():
        # 批量计算融合隐变量
        fused_latents = fusion_encoder(batch_stacks)        # [B, 4, H//8, W//8]
        # 批量计算GT隐变量
        gt_latents = fusion_encoder.encode_image(batch_gts) # [B, 4, H//8, W//8]
    
    return fused_latents.cpu(), gt_latents.cpu(), valid_indices


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("=" * 60)
    
    # 加载配置
    cfg = OmegaConf.load(args.config)
    
    # 设置参数
    image_size = args.image_size
    batch_size = args.batch_size
    latent_scale_factor = cfg.model.cldm.params.latent_scale_factor
    
    # 创建融合编码器
    fusion_cfg = cfg.model.fusion_network
    fusion_encoder = FusionEncoder(
        vae_model_id=fusion_cfg.vae_model_id,
        vae_subfolder=fusion_cfg.vae_subfolder,
        depth_transformer_cfg=fusion_cfg.depth_transformer,
        fusion_weights_path=fusion_cfg.fusion_weights_path,
        latent_scale_factor=latent_scale_factor
    )
    fusion_encoder.to(device)
    fusion_encoder.eval()
    print("✓ Fusion encoder created and loaded")
    
    # 图像预处理（转换为[-1, 1]范围）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 收集所有样本路径
    datasets_root = cfg.dataset.train.params.datasets_root
    train_datasets = cfg.dataset.train.params.train_datasets
    
    all_samples = get_stack_paths(datasets_root, train_datasets)
    print(f"Found {len(all_samples)} image stacks in total")
    
    if len(all_samples) == 0:
        print("❌ No samples found! Check your dataset paths.")
        return
    
    # 按 stack_size 分组
    grouped_samples = group_samples_by_stack_size(all_samples)
    print(f"Grouped into {len(grouped_samples)} different stack sizes:")
    for size, samples in sorted(grouped_samples.items()):
        print(f"  stack_size={size}: {len(samples)} samples")
    
    # 预计算结果存储
    all_fused_latents = []
    all_gt_latents = []
    all_stack_names = []
    all_stack_sizes = []
    all_dataset_names = []
    
    # 按分组批量处理（从大到小排序）
    print(f"\nProcessing samples with batch_size={batch_size}...")
    
    for stack_size, samples in sorted(grouped_samples.items(), reverse=True):
        num_batches = (len(samples) + batch_size - 1) // batch_size
        
        pbar = tqdm(range(0, len(samples), batch_size), 
                   desc=f"stack_size={stack_size}", 
                   total=num_batches)
        
        for batch_start in pbar:
            batch_end = min(batch_start + batch_size, len(samples))
            batch_samples = samples[batch_start:batch_end]
            
            try:
                # 批量处理
                fused_latents, gt_latents, valid_indices = process_batch(
                    batch_samples, fusion_encoder, transform, image_size, device
                )
                
                if fused_latents is None:
                    continue
                
                # 存储结果
                for i, idx in enumerate(valid_indices):
                    sample_info = batch_samples[idx]
                    all_fused_latents.append(fused_latents[i])
                    all_gt_latents.append(gt_latents[i])
                    all_stack_names.append(sample_info['stack_name'])
                    all_stack_sizes.append(sample_info['stack_size'])
                    all_dataset_names.append(sample_info['dataset_name'])
                
                # 清理显存
                del fused_latents, gt_latents
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue
    
    total_samples = len(all_fused_latents)
    print(f"\n✓ Successfully processed {total_samples} samples")
    
    if total_samples == 0:
        print("❌ No latents were computed!")
        return
    
    # 创建输出目录
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查隐变量形状
    sample_shape = all_fused_latents[0].shape
    print(f"Latent shape: {sample_shape}")
    
    # 堆叠隐变量
    all_same_shape = all(latent.shape == sample_shape for latent in all_fused_latents)
    
    if all_same_shape:
        fused_latents_tensor = torch.stack(all_fused_latents)
        gt_latents_tensor = torch.stack(all_gt_latents)
        print(f"Fused latents shape: {fused_latents_tensor.shape}")
        print(f"GT latents shape: {gt_latents_tensor.shape}")
    else:
        print("⚠️  Latents have different shapes, saving as list")
        fused_latents_tensor = all_fused_latents
        gt_latents_tensor = all_gt_latents
    
    # 保存结果
    output_path = os.path.join(output_dir, args.output_filename)
    
    save_dict = {
        'fused_latents': fused_latents_tensor,    # 融合隐变量 [N, 4, 64, 64]
        'gt_latents': gt_latents_tensor,           # GT隐变量 [N, 4, 64, 64]
        'stack_names': all_stack_names,            # 图像栈名称列表
        'stack_sizes': all_stack_sizes,            # 图像栈大小列表
        'dataset_names': all_dataset_names,        # 数据集名称列表
        'num_samples': total_samples,
        'latent_shape': sample_shape,
        'image_size': image_size,
        'latent_scale_factor': latent_scale_factor,
        'config_path': args.config
    }
    
    torch.save(save_dict, output_path)
    print(f"\n✓ Saved precomputed latents to {output_path}")
    
    # 打印存储信息
    storage_size = os.path.getsize(output_path) / (1024**2)
    print(f"Storage size: {storage_size:.2f} MB")
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Total samples: {total_samples}")
    print(f"  Latent shape: {sample_shape}")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Storage: {storage_size:.2f} MB")
    print(f"  Output: {output_path}")
    print("=" * 60)
    
    # 按数据集统计
    from collections import Counter
    dataset_counts = Counter(all_dataset_names)
    print("\nSamples per dataset:")
    for name, count in dataset_counts.items():
        print(f"  {name}: {count}")


def verify_precomputed(precomputed_path):
    """验证预计算文件"""
    print(f"\nVerifying: {precomputed_path}")
    
    data = torch.load(precomputed_path, map_location='cpu', weights_only=False)
    
    print(f"Keys: {list(data.keys())}")
    print(f"Num samples: {data['num_samples']}")
    print(f"Latent shape: {data['latent_shape']}")
    
    fused = data['fused_latents']
    gt = data['gt_latents']
    
    if isinstance(fused, torch.Tensor):
        print(f"Fused latents tensor shape: {fused.shape}")
        print(f"  Mean: {fused.mean():.4f}, Std: {fused.std():.4f}")
        print(f"  Range: [{fused.min():.4f}, {fused.max():.4f}]")
        
    if isinstance(gt, torch.Tensor):
        print(f"GT latents tensor shape: {gt.shape}")
        print(f"  Mean: {gt.mean():.4f}, Std: {gt.std():.4f}")
        print(f"  Range: [{gt.min():.4f}, {gt.max():.4f}]")
    
    print("✓ Verification complete")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, 
                       default='configs/train/train_stage2.yaml',
                       help="训练配置文件路径")
    parser.add_argument("--output_dir", type=str, 
                       default='precomputed_latents',
                       help="输出目录")
    parser.add_argument("--output_filename", type=str, 
                       default='fused_latents.pth',
                       help="输出文件名")
    parser.add_argument("--image_size", type=int,
                       default=512,
                       help="图像尺寸")
    parser.add_argument("--batch_size", type=int,
                       default=8,
                       help="批处理大小（越大越快，但需要更多显存）")
    parser.add_argument("--verify", type=str,
                       default=None,
                       help="验证已存在的预计算文件")
    args = parser.parse_args()
    
    if args.verify:
        verify_precomputed(args.verify)
    else:
        main(args)
