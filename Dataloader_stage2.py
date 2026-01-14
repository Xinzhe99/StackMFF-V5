"""
Stage2 数据加载器（使用预计算隐变量版本）

功能：
- 直接加载预计算的融合隐变量和GT隐变量
- 完全不需要FusionEncoder，大幅减少显存占用
- 支持隐变量空间的数据增强

使用方法：
    from Dataloader_stage2_new import create_dataloader
    
    loader, dataset = create_dataloader(
        precomputed_latents_path='precomputed_latents/fused_latents.pth',
        batch_size=4,
        num_workers=4
    )
"""

import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler, ConcatDataset
from collections import defaultdict


class PrecomputedLatentDataset(Dataset):
    """
    预计算隐变量数据集
    
    直接从预计算文件加载 fused_latents 和 gt_latents，
    无需任何图像加载或模型推理
    
    数据增强（在隐变量空间）：
    - 随机翻转（水平/垂直）
    - 可选的轻微噪声注入
    """
    
    def __init__(
        self,
        precomputed_latents_path: str,
        augment: bool = True,
        noise_augment_prob: float = 0.0,  # 默认关闭噪声增强
        noise_augment_scale: float = 0.02
    ):
        """
        Args:
            precomputed_latents_path: 预计算隐变量 .pth 文件路径
            augment: 是否启用数据增强
            noise_augment_prob: 噪声增强概率（建议训练时关闭或设很小）
            noise_augment_scale: 噪声增强强度
        """
        self.precomputed_latents_path = precomputed_latents_path
        self.augment = augment
        self.noise_augment_prob = noise_augment_prob
        self.noise_augment_scale = noise_augment_scale
        
        # 加载预计算数据
        self._load_data()
        
    def _load_data(self):
        """加载预计算的隐变量数据"""
        print(f"Loading precomputed latents from: {self.precomputed_latents_path}")
        
        if not os.path.exists(self.precomputed_latents_path):
            raise FileNotFoundError(f"Precomputed latents file not found: {self.precomputed_latents_path}")
        
        data = torch.load(self.precomputed_latents_path, map_location='cpu', weights_only=False)
        
        # 必需的字段
        self.fused_latents = data['fused_latents']  # [N, 4, H, W] 融合后的隐变量
        self.gt_latents = data['gt_latents']        # [N, 4, H, W] GT图像的隐变量
        
        # 可选的元数据
        self.stack_names = data.get('stack_names', [f'sample_{i}' for i in range(len(self.fused_latents))])
        self.stack_sizes = data.get('stack_sizes', [1] * len(self.fused_latents))
        self.dataset_names = data.get('dataset_names', ['unknown'] * len(self.fused_latents))
        self.latent_shape = data.get('latent_shape', self.fused_latents[0].shape if len(self.fused_latents) > 0 else None)
        self.num_samples = data.get('num_samples', len(self.fused_latents))
        
        # 转换为tensor（如果是列表）
        if isinstance(self.fused_latents, list):
            self.fused_latents = torch.stack(self.fused_latents)
        if isinstance(self.gt_latents, list):
            self.gt_latents = torch.stack(self.gt_latents)
        
        print(f"✓ Loaded {self.num_samples} samples")
        print(f"  Fused latents shape: {self.fused_latents.shape}")
        print(f"  GT latents shape: {self.gt_latents.shape}")
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """获取单个样本"""
        # 获取隐变量
        fused_latent = self.fused_latents[idx].clone()  # [4, H, W]
        gt_latent = self.gt_latents[idx].clone()        # [4, H, W]
        
        # 数据增强（对 fused_latent 和 gt_latent 应用相同的变换）
        if self.augment:
            fused_latent, gt_latent = self._latent_augment(fused_latent, gt_latent)
        
        # 生成 prompt
        prompt = "a high quality all-in-focus image"
        
        return {
            "fused_latent": fused_latent,  # [4, H, W] 作为 ControlNet 条件 c_img
            "gt_latent": gt_latent,        # [4, H, W] 作为 diffusion 目标 z_0
            "prompt": prompt,
            "stack_size": self.stack_sizes[idx] if idx < len(self.stack_sizes) else 1,
            "stack_name": self.stack_names[idx] if idx < len(self.stack_names) else f'sample_{idx}',
            "index": idx
        }
    
    def _latent_augment(self, fused_latent: torch.Tensor, gt_latent: torch.Tensor):
        """
        隐变量空间的数据增强
        
        重要：必须对 fused_latent 和 gt_latent 应用相同的几何变换！
        """
        # 1. 随机水平翻转
        if random.random() > 0.5:
            fused_latent = torch.flip(fused_latent, dims=[-1])
            gt_latent = torch.flip(gt_latent, dims=[-1])
        
        # 2. 随机垂直翻转
        if random.random() > 0.5:
            fused_latent = torch.flip(fused_latent, dims=[-2])
            gt_latent = torch.flip(gt_latent, dims=[-2])
        
        # 3. 可选的轻微噪声注入（只对fused_latent，不对gt_latent）
        # 这模拟了一些不确定性，但要谨慎使用
        if random.random() < self.noise_augment_prob:
            noise = torch.randn_like(fused_latent) * self.noise_augment_scale
            fused_latent = fused_latent + noise
        
        return fused_latent, gt_latent


class GroupedBatchSampler(Sampler):
    """
    按 stack size 分组的批次采样器
    确保每个 batch 内的样本具有相同的 stack_size
    """
    
    def __init__(self, stack_sizes, batch_size):
        self.stack_size_groups = defaultdict(list)
        for idx, size in enumerate(stack_sizes):
            self.stack_size_groups[size].append(idx)
        self.batch_size = batch_size
        self.batches = self._create_batches()
    
    def _create_batches(self):
        batches = []
        for size, indices in self.stack_size_groups.items():
            random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) > 0:
                    batches.append(batch)
        random.shuffle(batches)
        return batches
    
    def __iter__(self):
        # 每个epoch重新生成batches以增加随机性
        self.batches = self._create_batches()
        return iter(self.batches)
    
    def __len__(self):
        return len(self.batches)


class RandomBatchSampler(Sampler):
    """
    完全随机的批次采样器（不考虑stack_size分组）
    当所有隐变量尺寸相同时可以使用
    """
    
    def __init__(self, num_samples, batch_size, drop_last=False):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.drop_last = drop_last
    
    def __iter__(self):
        indices = list(range(self.num_samples))
        random.shuffle(indices)
        
        batches = []
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i:i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)
        
        return iter(batches)
    
    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        return (self.num_samples + self.batch_size - 1) // self.batch_size


def collate_fn_precomputed(batch):
    """
    自定义 collate 函数
    """
    fused_latents = torch.stack([item["fused_latent"] for item in batch])  # [B, 4, H, W]
    gt_latents = torch.stack([item["gt_latent"] for item in batch])        # [B, 4, H, W]
    prompts = [item["prompt"] for item in batch]
    stack_sizes = [item["stack_size"] for item in batch]
    stack_names = [item["stack_name"] for item in batch]
    indices = [item["index"] for item in batch]
    
    return {
        "fused_latent": fused_latents,  # [B, 4, H, W]
        "gt_latent": gt_latents,        # [B, 4, H, W]
        "prompt": prompts,
        "stack_size": stack_sizes,
        "stack_name": stack_names,
        "index": indices
    }


def create_dataloader(
    precomputed_latents_path: str,
    batch_size: int,
    num_workers: int = 4,
    augment: bool = True,
    use_grouped_sampler: bool = False,  # 默认禁用分组采样器，避免分布式训练问题
    noise_augment_prob: float = 0.0,
    noise_augment_scale: float = 0.02,
    pin_memory: bool = True
):
    """
    创建 Stage2 训练数据加载器（使用预计算隐变量）
    
    Args:
        precomputed_latents_path: 预计算隐变量文件路径
        batch_size: 批大小
        num_workers: 数据加载线程数
        augment: 是否启用数据增强
        use_grouped_sampler: 是否使用分组采样器（分布式训练时建议禁用）
        noise_augment_prob: 噪声增强概率
        noise_augment_scale: 噪声增强强度
        pin_memory: 是否使用 pin_memory
    
    Returns:
        loader: DataLoader
        dataset: PrecomputedLatentDataset
    """
    # 创建数据集
    dataset = PrecomputedLatentDataset(
        precomputed_latents_path=precomputed_latents_path,
        augment=augment,
        noise_augment_prob=noise_augment_prob,
        noise_augment_scale=noise_augment_scale
    )
    
    # 对于分布式训练，使用标准的 DataLoader（让 accelerator 处理分布式采样）
    # 不使用自定义的 batch_sampler，避免分布式同步问题
    if use_grouped_sampler:
        # 仅在单卡训练时使用分组采样器
        sampler = GroupedBatchSampler(dataset.stack_sizes, batch_size)
        loader = DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn_precomputed,
            pin_memory=pin_memory,
        )
    else:
        # 分布式训练友好的标准 DataLoader
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,  # 确保所有进程有相同数量的 batch
            num_workers=num_workers,
            collate_fn=collate_fn_precomputed,
            pin_memory=pin_memory,
        )
    
    return loader, dataset


def create_dataloader_from_config(cfg, precomputed_latents_path: str = None):
    """
    从配置文件创建数据加载器
    
    Args:
        cfg: OmegaConf 配置对象
        precomputed_latents_path: 预计算隐变量文件路径（如果为None，从config获取）
    
    Returns:
        loader: DataLoader
        dataset: PrecomputedLatentDataset
    """
    # 获取参数
    batch_size = cfg.train.batch_size
    num_workers = cfg.train.num_workers
    augment = cfg.dataset.train.params.get('augment', True)
    
    # 确定预计算文件路径
    if precomputed_latents_path is None:
        # 尝试从config获取
        precomputed_latents_path = cfg.dataset.train.params.get(
            'precomputed_latents_path',
            'precomputed_latents/fused_latents.pth'
        )
    
    return create_dataloader(
        precomputed_latents_path=precomputed_latents_path,
        batch_size=batch_size,
        num_workers=num_workers,
        augment=augment,
        use_grouped_sampler=True
    )


# ==================== 验证和调试工具 ====================

def verify_dataloader(loader, num_batches=3):
    """
    验证 DataLoader 输出
    """
    print("\n" + "=" * 60)
    print("Verifying DataLoader...")
    print("=" * 60)
    
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
            
        print(f"\nBatch {i + 1}:")
        print(f"  fused_latent shape: {batch['fused_latent'].shape}")
        print(f"  gt_latent shape: {batch['gt_latent'].shape}")
        print(f"  fused_latent range: [{batch['fused_latent'].min():.4f}, {batch['fused_latent'].max():.4f}]")
        print(f"  gt_latent range: [{batch['gt_latent'].min():.4f}, {batch['gt_latent'].max():.4f}]")
        print(f"  prompts: {batch['prompt'][:2]}...")
        print(f"  stack_sizes: {batch['stack_size']}")
    
    print("\n✓ DataLoader verification complete")
    print("=" * 60)


if __name__ == "__main__":
    # 测试代码
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents_path", type=str, 
                       default="precomputed_latents/fused_latents.pth",
                       help="预计算隐变量文件路径")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()
    
    # 创建 DataLoader
    loader, dataset = create_dataloader(
        precomputed_latents_path=args.latents_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=True
    )
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Number of batches: {len(loader)}")
    
    # 验证
    verify_dataloader(loader, num_batches=3)
