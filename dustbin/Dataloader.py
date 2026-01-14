# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University

import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from PIL import Image
from collections import defaultdict
import torchvision.transforms.functional as TF
from torch.utils.data import ConcatDataset

class FocusStackDataset(Dataset):
    """
    Dataset class for handling stacks of focus images and their corresponding Soft GT (probability maps).
    Supports data augmentation, subset sampling, and layer shuffling.
    """
    def __init__(self, root_dir, soft_gt_dir, transform=None, augment=True, subset_fraction=1, shuffle_order=False):
        """
        Initialize the dataset.
        Args:
            root_dir: Directory containing focus image stacks  
            soft_gt_dir: Directory containing soft GT maps (.npy files, shape [N, H, W])
            transform: Optional transforms to be applied
            augment: Whether to apply data augmentation (Spatial flips)
            subset_fraction: Fraction of the dataset to use (0-1)
            shuffle_order: Whether to shuffle the order of layers in the stack (for Unordered/Ablation study)
        """
        self.root_dir = root_dir
        self.soft_gt_dir = soft_gt_dir
        self.transform = transform
        self.augment = augment
        self.shuffle_order = shuffle_order  # 新增：控制是否打乱顺序
        self.image_stacks = []
        self.soft_gt_maps = []
        self.stack_sizes = []

        # 获取所有图像栈文件夹
        all_stacks = sorted(os.listdir(root_dir))
        
        # 数据集子集采样
        subset_size = int(len(all_stacks) * subset_fraction)
        if subset_size < len(all_stacks):
            selected_stacks = random.sample(all_stacks, subset_size)
        else:
            selected_stacks = all_stacks

        for stack_name in selected_stacks:
            stack_path = os.path.join(root_dir, stack_name)
            if os.path.isdir(stack_path):
                image_stack = []
                # 排除非图像文件和元数据文件
                for img_name in sorted(os.listdir(stack_path), key=self.sort_key):
                    if img_name.lower().endswith(('.png', '.jpg', '.bmp', '.jpeg')) and not img_name.endswith('.npy'):
                        img_path = os.path.join(stack_path, img_name)
                        image_stack.append(img_path)

                if image_stack:
                    # 查找对应的 Soft GT (.npy)
                    gt_path = os.path.join(soft_gt_dir, stack_name + '.npy')
                    
                    if os.path.exists(gt_path):
                        self.image_stacks.append(image_stack)
                        self.soft_gt_maps.append(gt_path)
                        self.stack_sizes.append(len(image_stack))
                    else:
                        print(f"Warning: Soft GT not found for {stack_name} at {gt_path}")
                else:
                    print(f'Failed to read image stack: {stack_name}')

    def __len__(self):
        return len(self.image_stacks)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        Returns:
            stack_tensor: Tensor of stacked images (N, H, W)
            soft_gt: Corresponding probability map [N, H, W] as torch.float32
            len(images): Number of images in the stack
        """
        image_stack = self.image_stacks[idx]
        gt_path = self.soft_gt_maps[idx]

        # 1. 加载图像栈
        images = []
        for img_path in image_stack:
            image = Image.open(img_path).convert('YCbCr')
            image = image.split()[0]  # 只保留 Y 通道 (灰度)
            images.append(image)

        # 2. 加载 Soft GT
        # 期望格式: numpy array [N, H, W], float16 或 float32
        soft_gt_np = np.load(gt_path)
        soft_gt = torch.from_numpy(soft_gt_np).float()
        
        # 校验层数是否匹配
        if soft_gt.shape[0] != len(images):
            # 简单的容错处理
            min_len = min(len(images), soft_gt.shape[0])
            images = images[:min_len]
            soft_gt = soft_gt[:min_len, :, :]

        # =========================================================
        # 新增：处理无序模式 (Shuffle Order)
        # =========================================================
        if self.shuffle_order:
            # 生成随机排列索引
            num_layers = len(images)
            perm = torch.randperm(num_layers)
            
            # 1. 打乱图像列表
            images = [images[i] for i in perm]
            
            # 2. 同步打乱 Soft GT (沿着第0维: 层级维度)
            soft_gt = soft_gt[perm]

        # 3. 数据增强 (Spatial Flip)
        if self.augment:
            images, soft_gt = self.consistent_transform(images, soft_gt)

        # 4. Resize 与 转换
        if self.transform:
            images = [self.transform(img) for img in images]
            
            target_size = None
            for t in self.transform.transforms:
                if isinstance(t, transforms.Resize):
                    target_size = t.size
                    break
            
            if target_size is not None:
                # 对 Soft GT 进行 Resize (BILINEAR)
                soft_gt = TF.resize(
                    soft_gt, 
                    target_size, 
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True
                )
                # Resize 后重新归一化
                soft_gt = torch.clamp(soft_gt, min=0.0)
                sum_prob = soft_gt.sum(dim=0, keepdim=True) + 1e-8
                soft_gt = soft_gt / sum_prob

        # 5. 堆叠图像
        images = [img.squeeze(0) for img in images] 
        stack_tensor = torch.stack(images)  # (N, H, W)

        return stack_tensor, soft_gt, len(images)

    def consistent_transform(self, images, soft_gt):
        """
        Apply consistent transformations to both images and Soft GT.
        """
        # 随机水平翻转
        if random.random() > 0.5:
            images = [TF.hflip(img) for img in images]
            soft_gt = TF.hflip(soft_gt)

        # 随机垂直翻转
        if random.random() > 0.5:
            images = [TF.vflip(img) for img in images]
            soft_gt = TF.vflip(soft_gt)

        return images, soft_gt

    @staticmethod
    def sort_key(filename):
        digits = ''.join(filter(str.isdigit, filename))
        return int(digits) if digits else 0

class GroupedBatchSampler(Sampler):
    """
    Custom batch sampler that groups samples by stack size.
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
                batches.append(indices[i:i + self.batch_size])
        random.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

def get_updated_dataloader(dataset_params, batch_size, num_workers=4, augment=True, target_size=256, shuffle_order=False):
    """
    Create a DataLoader with multiple datasets combined.
    Args:
        dataset_params: List of params dict.
        batch_size: Int
        num_workers: Int
        augment: Bool (Spatial augmentation)
        target_size: Int
        shuffle_order: Bool (Whether to shuffle layer order, for Unordered Mode)
    """
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
    ])

    datasets = []
    for params in dataset_params:
        gt_dir_key = 'soft_gt_dir' if 'soft_gt_dir' in params else 'focus_index_gt'
        
        dataset = FocusStackDataset(
            root_dir=params['root_dir'],
            soft_gt_dir=params[gt_dir_key], 
            transform=transform,
            augment=augment,
            subset_fraction=params.get('subset_fraction', 1.0),
            shuffle_order=shuffle_order  # 传递参数
        )
        if len(dataset) > 0:
            datasets.append(dataset)

    if not datasets:
        return None

    combined_dataset = CombinedDataset(datasets)
    sampler = GroupedBatchSampler(combined_dataset.stack_sizes, batch_size)
    dataloader = DataLoader(combined_dataset, batch_sampler=sampler, num_workers=num_workers)
    
    return dataloader

class CombinedDataset(ConcatDataset):
    def __init__(self, datasets):
        super(CombinedDataset, self).__init__(datasets)
        self.stack_sizes = []
        for dataset in datasets:
            self.stack_sizes.extend(dataset.stack_sizes)

    def __getitem__(self, idx):
        return super(CombinedDataset, self).__getitem__(idx)