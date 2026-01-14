
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
# ==================== 数据集定义 ====================
class FocusStackDatasetStage2(Dataset):
    """
    用于 Stage2 训练的图像栈数据集
    返回图像栈、融合后的GT图像和prompt
    """
    def __init__(self, root_dir, fused_gt_dir, transform=None, augment=True, 
                 subset_fraction=1.0, training_image_size=512):
        self.root_dir = root_dir
        self.fused_gt_dir = fused_gt_dir
        self.transform = transform
        self.augment = augment
        self.training_image_size = training_image_size
        self.image_stacks = []
        self.fused_gt_paths = []
        self.stack_sizes = []

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Dataset root directory not found: {root_dir}")

        all_stacks = sorted(os.listdir(root_dir))
        
        if not all_stacks:
            raise ValueError(f"No stacks found in: {root_dir}")
        
        subset_size = max(1, int(len(all_stacks) * subset_fraction)) if subset_fraction > 0 else 0
        subset_size = min(subset_size, len(all_stacks))
        
        if subset_size == 0:
            print(f"Warning: subset_fraction={subset_fraction} results in 0 samples, using all data")
            selected_stacks = all_stacks
        else:
            selected_stacks = random.sample(all_stacks, subset_size) if subset_fraction < 1.0 else all_stacks

        for stack_name in selected_stacks:
            stack_path = os.path.join(root_dir, stack_name)
            if os.path.isdir(stack_path):
                image_stack = []
                for img_name in sorted(os.listdir(stack_path), key=self._sort_key):
                    if img_name.lower().endswith(('.png', '.jpg', '.bmp')) and img_name != 'layer_order.npy':
                        img_path = os.path.join(stack_path, img_name)
                        image_stack.append(img_path)

                if image_stack:
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

    def __len__(self):
        return len(self.image_stacks)

    def __getitem__(self, idx):
        image_stack = self.image_stacks[idx]
        fused_gt_path = self.fused_gt_paths[idx]

        images = []
        for img_path in image_stack:
            with Image.open(img_path) as img:
                images.append(img.convert('RGB'))

        with Image.open(fused_gt_path) as img:
            fused_gt = img.convert('RGB')

        # 随机打乱图像栈中图层的顺序
        if self.augment:
            indices = list(range(len(images)))
            random.shuffle(indices)
            images = [images[i] for i in indices]

        # 数据增强
        if self.augment:
            images, fused_gt = self._consistent_transform(images, fused_gt)

        # 应用transforms
        if self.transform:
            images = [self.transform(img) for img in images]
            fused_gt = self.transform(fused_gt)
            
        stack_tensor = torch.stack(images)  # [N, C, H, W]
        
        # 生成简单的prompt
        prompt = "a high quality all-in-focus image"

        return {
            "image_stack": stack_tensor,  # [N, C, H, W]
            "aif": fused_gt,              # [C, H, W]
            "prompt": prompt,
            "stack_size": len(image_stack)
        }

    def _consistent_transform(self, images, fused_gt):
        """一致的数据增强"""
        # 随机裁剪
        images, fused_gt = self._consistent_random_crop(images, fused_gt)
        
        # 颜色扰动
        if random.random() < 0.1:
            brightness = (0.8, 1.2)
            contrast = (0.8, 1.2)
            saturation = (0.8, 1.2)
            hue = (-0.1, 0.1)
            fn_idx, b_f, c_f, s_f, h_f = transforms.ColorJitter.get_params(
                brightness, contrast, saturation, hue
            )
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
            images = [apply_jitter(img) for img in images]
            fused_gt = apply_jitter(fused_gt)

        # 随机翻转
        if random.random() > 0.5:
            images = [TF.hflip(img) for img in images]
            fused_gt = TF.hflip(fused_gt)
        if random.random() > 0.5:
            images = [TF.vflip(img) for img in images]
            fused_gt = TF.vflip(fused_gt)

        return images, fused_gt
    
    def _consistent_random_crop(self, images, fused_gt):
        w, h = images[0].size
        if fused_gt.size != (w, h):
            fused_gt = TF.resize(fused_gt, size=(h, w), interpolation=TF.InterpolationMode.BILINEAR)

        th = tw = self.training_image_size
        if w < tw or h < th:
            padding_h = max(0, th - h)
            padding_w = max(0, tw - w)
            if padding_h > 0 or padding_w > 0:
                images = [TF.pad(img, padding=(0, 0, padding_w, padding_h)) for img in images]
                fused_gt = TF.pad(fused_gt, padding=(0, 0, padding_w, padding_h))
                w, h = images[0].size

        i, j, h_crop, w_crop = transforms.RandomCrop.get_params(
            images[0], output_size=(self.training_image_size, self.training_image_size))
        cropped_images = [TF.crop(img, i, j, h_crop, w_crop) for img in images]
        cropped_fused_gt = TF.crop(fused_gt, i, j, h_crop, w_crop)

        return cropped_images, cropped_fused_gt

    @staticmethod
    def _sort_key(filename):
        digits = ''.join(filter(str.isdigit, filename))
        return int(digits) if digits else 0


class GroupedBatchSampler(Sampler):
    """按stack size分组的批次采样器"""
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


class CombinedDatasetStage2(ConcatDataset):
    """合并多个数据集"""
    def __init__(self, datasets):
        super().__init__(datasets)
        self.stack_sizes = []
        for dataset in datasets:
            self.stack_sizes.extend(dataset.stack_sizes)


def collate_fn_stage2(batch):
    """自定义collate函数，处理变长的图像栈"""
    image_stacks = torch.stack([item["image_stack"] for item in batch])  # [B, N, C, H, W]
    aifs = torch.stack([item["aif"] for item in batch])  # [B, C, H, W]
    prompts = [item["prompt"] for item in batch]
    stack_sizes = [item["stack_size"] for item in batch]
    
    return {
        "image_stack": image_stacks,
        "aif": aifs,
        "prompt": prompts,
        "stack_size": stack_sizes
    }


def create_dataloader(cfg, batch_size, num_workers, training_image_size):
    """创建Stage2训练数据加载器"""
    transform = transforms.Compose([
        transforms.Resize((training_image_size, training_image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
    ])
    
    datasets = []
    datasets_root = cfg.dataset.train.params.datasets_root
    train_datasets = cfg.dataset.train.params.train_datasets
    subset_fraction = cfg.dataset.train.params.get('subset_fraction', 1.0)
    augment = cfg.dataset.train.params.get('augment', True)
    
    for dataset_name in train_datasets:
        dataset_path = os.path.join(datasets_root, dataset_name, 'TR')
        if os.path.exists(dataset_path):
            dataset = FocusStackDatasetStage2(
                root_dir=os.path.join(dataset_path, 'focus_stack'),
                fused_gt_dir=os.path.join(dataset_path, 'AiF'),
                transform=transform,
                augment=augment,
                subset_fraction=subset_fraction,
                training_image_size=training_image_size
            )
            datasets.append(dataset)
            print(f"✓ Loaded dataset: {dataset_name} with {len(dataset)} samples")
        else:
            print(f"⚠️  Dataset path not found: {dataset_path}")
    
    if not datasets:
        raise ValueError("No valid datasets found!")
    
    combined_dataset = CombinedDatasetStage2(datasets)
    sampler = GroupedBatchSampler(combined_dataset.stack_sizes, batch_size)
    
    loader = DataLoader(
        dataset=combined_dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn_stage2,
        pin_memory=True,
    )
    
    return loader, combined_dataset