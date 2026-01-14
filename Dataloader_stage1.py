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
    Dataset class for handling stacks of focus images and their corresponding fused GT images.
    Supports data augmentation and subset sampling.
    """
    def __init__(self, root_dir, fused_gt_dir, transform=None, augment=True, subset_fraction=1, training_image_size=384):
        """
        Initialize the dataset.
        Args:
            root_dir: Directory containing focus image stacks  
            fused_gt_dir: Directory containing fused GT images
            transform: Optional transforms to be applied
            augment: Whether to apply data augmentation
            subset_fraction: Fraction of the dataset to use (0-1)
            training_image_size: Size for random crop during training
        """
        self.root_dir = root_dir
        self.fused_gt_dir = fused_gt_dir
        self.transform = transform
        self.augment = augment
        self.training_image_size = training_image_size
        self.image_stacks = []
        self.fused_gt_paths = []
        self.stack_sizes = []

        # 检查目录是否存在
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Dataset root directory not found: {root_dir}")

        all_stacks = sorted(os.listdir(root_dir))
        
        if not all_stacks:
            raise ValueError(f"No stacks found in: {root_dir}")
        
        # 计算子集大小
        subset_size = max(1, int(len(all_stacks) * subset_fraction)) if subset_fraction > 0 else 0
        subset_size = min(subset_size, len(all_stacks))
        
        if subset_size == 0:
            print(f"Warning: subset_fraction={subset_fraction} results in 0 samples, using all data")
            selected_stacks = all_stacks
        else:
            selected_stacks = random.sample(all_stacks, subset_size)

        for stack_name in selected_stacks:
            stack_path = os.path.join(root_dir, stack_name)
            if os.path.isdir(stack_path):
                image_stack = []
                # 排除layer_order.npy文件，只加载图像文件
                for img_name in sorted(os.listdir(stack_path), key=self.sort_key):
                    if img_name.lower().endswith(('.png', '.jpg', '.bmp')) and img_name != 'layer_order.npy':
                        img_path = os.path.join(stack_path, img_name)
                        image_stack.append(img_path)

                if image_stack:
                    # 查找GT融合图像（支持多种格式）
                    fused_gt_path = None
                    for ext in ['.png', '.jpg', '.bmp']:
                        candidate_path = os.path.join(fused_gt_dir, stack_name + ext)
                        if os.path.exists(candidate_path):
                            fused_gt_path = candidate_path
                            break
                    
                    if fused_gt_path:
                        self.image_stacks.append(image_stack)
                        self.fused_gt_paths.append(fused_gt_path)
                        self.stack_sizes.append(len(image_stack))
                    else:
                        print(f"Warning: Fused GT image not found for {stack_name}")
                else:
                    print(f'Failed to read image stack: {stack_name}')

    def __len__(self):
        return len(self.image_stacks)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        Returns:
            stack_tensor: Tensor of stacked images (N, C, H, W)
            fused_gt: Corresponding fused GT image [C, H, W] as torch.float32
            len(images): Number of images in the stack
        """
        image_stack = self.image_stacks[idx]
        fused_gt_path = self.fused_gt_paths[idx]

        images = []
        for img_path in image_stack:
            with Image.open(img_path) as img:
                images.append(img.convert('RGB'))

        # 加载GT融合图像
        with Image.open(fused_gt_path) as img:
            fused_gt = img.convert('RGB')

        # 随机打乱图像栈中图层的顺序
        if self.augment:
            # 创建图像索引列表并随机打乱
            indices = list(range(len(images)))
            random.shuffle(indices)
            # 根据打乱后的索引重新排列图像
            images = [images[i] for i in indices]

        if self.augment:
            images, fused_gt = self.consistent_transform(images, fused_gt)

        # 应用其他变换
        if self.transform:
            images = [self.transform(img) for img in images]
            fused_gt = self.transform(fused_gt)
            
        # 堆叠图像栈
        stack_tensor = torch.stack(images)  # 形状将是 (N, C, H, W)

        return stack_tensor, fused_gt, len(images)

    def consistent_transform(self, images, fused_gt):
        """
        Apply consistent transformations to both images and fused GT image.
        Includes random crop, color jitter, and random horizontal and vertical flips.
        """
        # 随机裁剪
        images, fused_gt = self.consistent_random_crop(images, fused_gt)
        
        # 颜色扰动 (Color Jitter)
        if random.random() < 0.1:
            # 1. 定义扰动范围 (与你之前的一致)
            brightness = (0.8, 1.2)
            contrast = (0.8, 1.2)
            saturation = (0.8, 1.2)
            hue = (-0.1, 0.1)

            # 2. 获取随机参数
            # 返回: 
            # fn_idx (list): 扰动应用的顺序索引
            # brightness_factor, contrast_factor, saturation_factor, hue_factor: 具体的倍率
            fn_idx, b_f, c_f, s_f, h_f = transforms.ColorJitter.get_params(
                brightness, contrast, saturation, hue
            )

            # 3. 定义一个应用函数，确保顺序和参数完全一致
            def apply_jitter(img):
                for i in fn_idx:
                    if i == 0 and b_f is not None:
                        img = TF.adjust_brightness(img, b_f)
                    elif i == 1 and c_f is not None:
                        img = TF.adjust_contrast(img, c_f)
                    elif i == 2 and s_f is not None:
                        img = TF.adjust_saturation(img, s_f)
                    elif i == 3 and h_f is not None:
                        img = TF.adjust_hue(img, h_f)
                return img

            # 4. 应用到整个栈和 GT
            images = [apply_jitter(img) for img in images]
            fused_gt = apply_jitter(fused_gt)

        # 随机水平翻转
        if random.random() > 0.5:
            images = [TF.hflip(img) for img in images]
            fused_gt = TF.hflip(fused_gt)

        # 随机垂直翻转
        if random.random() > 0.5:
            images = [TF.vflip(img) for img in images]
            fused_gt = TF.vflip(fused_gt)

        return images, fused_gt
    
    def consistent_random_crop(self, images, fused_gt):
            # 1. 强制对齐尺寸（非常重要！）
            # 如果你的数据集里 GT 和 Stack 尺寸不一致，必须先 Resize 到相同尺寸
            # 否则任何 Crop 都会导致空间失配
            w, h = images[0].size
            if fused_gt.size != (w, h):
                fused_gt = TF.resize(fused_gt, size=(h, w), interpolation=TF.InterpolationMode.BILINEAR)

            # 2. 确定裁剪尺寸
            # 确保裁剪尺寸不会超过图像本身
            th = tw = self.training_image_size
            if w < tw or h < th:
                # 如果原图比目标裁剪尺寸还小，则先 padding 或者调整裁剪尺寸
                padding_h = max(0, th - h)
                padding_w = max(0, tw - w)
                if padding_h > 0 or padding_w > 0:
                    images = [TF.pad(img, padding=(0, 0, padding_w, padding_h)) for img in images]
                    fused_gt = TF.pad(fused_gt, padding=(0, 0, padding_w, padding_h))
                    w, h = images[0].size # 更新尺寸

            # 3. 使用 torchvision 的内置函数获取随机坐标
            # 这样可以保证所有图像用的 (i, j, h, w) 完全一致
            i, j, h_crop, w_crop = transforms.RandomCrop.get_params(
                images[0], output_size=(self.training_image_size, self.training_image_size))

            # 4. 应用裁剪
            cropped_images = [TF.crop(img, i, j, h_crop, w_crop) for img in images]
            cropped_fused_gt = TF.crop(fused_gt, i, j, h_crop, w_crop)

            return cropped_images, cropped_fused_gt

    @staticmethod
    def sort_key(filename):
        """
        Helper function to sort filenames based on their numerical values.
        Returns 0 if no digits are found to handle non-numeric filenames.
        """
        digits = ''.join(filter(str.isdigit, filename))
        return int(digits) if digits else 0

class GroupedBatchSampler(Sampler):
    """
    Custom batch sampler that groups samples by stack size for efficient batching.
    Ensures that each batch contains stacks of the same size.
    """
    def __init__(self, stack_sizes, batch_size):
        """
        Initialize the sampler.
        Args:
            stack_sizes: List of stack sizes for each sample
            batch_size: Number of samples per batch
        """
        self.stack_size_groups = defaultdict(list)
        for idx, size in enumerate(stack_sizes):
            self.stack_size_groups[size].append(idx)
        self.batch_size = batch_size
        self.batches = self._create_batches()

    def _create_batches(self):
        """
        Create batches of indices grouped by stack size.
        Returns shuffled batches for random sampling.
        """
        batches = []
        for size, indices in self.stack_size_groups.items():
            for i in range(0, len(indices), self.batch_size):
                batches.append(indices[i:i + self.batch_size])
        random.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

def get_updated_dataloader(dataset_params, batch_size, num_workers=4, augment=True, target_size=384, training_image_size=384):
    """
    Create a DataLoader with multiple datasets combined.
    Args:
        dataset_params: List of parameter dictionaries for each dataset
        batch_size: Number of samples per batch
        num_workers: Number of worker processes for data loading
        augment: Whether to apply data augmentation
        target_size: Size to resize images to
        training_image_size: Size for random crop during training
    Returns:
        DataLoader object with combined datasets
    """
    # 归一化到[-1, 1]范围以适配VAE输入
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),  # 先转换到[0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到[-1, 1]
    ])

    datasets = []
    for params in dataset_params:
        dataset = FocusStackDataset(
            root_dir=params['root_dir'],
            fused_gt_dir=params['fused_gt'],  # 改为融合图像GT路径
            transform=transform,
            augment=augment,
            subset_fraction=params['subset_fraction'],
            training_image_size=training_image_size
        )
        datasets.append(dataset)

    combined_dataset = CombinedDataset(datasets)

    sampler = GroupedBatchSampler(combined_dataset.stack_sizes, batch_size)

    dataloader = DataLoader(combined_dataset, batch_sampler=sampler, num_workers=num_workers)
    return dataloader

class CombinedDataset(ConcatDataset):
    """
    Extension of ConcatDataset that maintains stack size information
    when combining multiple datasets.
    """
    def __init__(self, datasets):
        """
        Initialize the combined dataset.
        Args:
            datasets: List of FocusStackDataset objects to combine
        """
        super(CombinedDataset, self).__init__(datasets)
        self.stack_sizes = []
        for dataset in datasets:
            self.stack_sizes.extend(dataset.stack_sizes)

    def __getitem__(self, idx):
        return super(CombinedDataset, self).__getitem__(idx)