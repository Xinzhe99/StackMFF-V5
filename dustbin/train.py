import argparse
import time
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import platform
from datetime import datetime
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# === 自定义模块导入 ===
# 确保 Dataloader.py, network.py, loss.py, utils.py 在同一目录下
from Dataloader import get_updated_dataloader
from network_v1 import StackMFF_V5 
# 修改点1: 导入 SpatialSmoothnessLoss 以便在验证集手动计算
from loss import FusionLoss, SpatialSmoothnessLoss
from utils import (to_image, count_parameters, config_model_dir, 
                   print_banner, print_model_info, print_device_info, 
                   print_dataset_info, print_training_config, 
                   print_epoch_results, print_training_complete)

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="StackMFF Training Script (Ordered/Unordered)")
    
    # === 核心实验参数 ===
    parser.add_argument('--loss_mode', type=str, default='ordered', choices=['ordered', 'unordered'],
                        help="Training mode: 'ordered' (OT Loss) or 'unordered' (KL Divergence)")
    parser.add_argument('--lambda_spatial', type=float, default=0.01,
                        help="Weight for spatial smoothness regularization (Horizontal Loss)")
    parser.add_argument('--save_name', default='stackmff_training',
                        help="Name of the experiment for saving logs and models")
    
    # === 数据集配置 ===
    # 请修改为你的实际数据集根目录
    parser.add_argument('--datasets_root', 
                        default=r'/media/user/dataset/stackmff_v3_dataset',
                        type=str, help='Root path to all datasets')

    parser.add_argument('--train_datasets', nargs='+', 
                        default=['NYU-V2', 'DUTS', 'DIODE', 'Cityscapes', 'ADE'],
                        help='List of datasets to use for training')
    parser.add_argument('--val_datasets', nargs='+',
                        default=['NYU-V2', 'DUTS', 'DIODE', 'Cityscapes', 'ADE'],
                        help='List of datasets to use for validation')
    
    parser.add_argument('--subset_fraction_train', type=float, default=0.2,
                        help='Fraction of training data to use (0-1)')
    parser.add_argument('--subset_fraction_val', type=float, default=0.05,
                        help='Fraction of validation data to use (0-1)')

    # === 训练超参数 ===
    parser.add_argument('--training_image_size', type=int, default=256,
                        help='Target image size for training (Resize)')
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, 
                        help='Number of training epochs')
    parser.add_argument('--eval_interval', type=int, default=1, 
                        help='Interval of epochs between evaluations')
    parser.add_argument('--lr', type=float, default=1e-3, 
                        help='Initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.9, 
                        help='Learning rate decay factor per epoch')
    parser.add_argument('--num_workers', type=int, default=8, 
                        help='Number of data loading workers')
    
    # === 硬件配置 ===
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=None,
                        help='Specific GPU IDs to use (e.g., 0 1). Default: use all.')
    
    return parser.parse_args()

def create_dataset_loaders(args):
    """
    Create training and validation data loaders.
    """
    # === 关键逻辑：根据 loss_mode 决定是否打乱顺序 ===
    should_shuffle = (args.loss_mode == 'unordered')
    
    if should_shuffle:
        print("🔀 Dataloader: Layer shuffling ENABLED (Unordered Mode)")
    else:
        print("⬇️ Dataloader: Layer shuffling DISABLED (Ordered Mode)")

    # -------------------------------------------------
    # 1. 创建训练集 Loader
    # -------------------------------------------------
    train_dataset_params = []
    for dataset_name in args.train_datasets:
        dataset_path = os.path.join(args.datasets_root, dataset_name, 'TR')
        
        # 严格指定 soft_gt 路径
        gt_path = os.path.join(dataset_path, 'soft_gt')
        img_path = os.path.join(dataset_path, 'focus_stack')
             
        # 检查图像和GT路径是否都存在
        if os.path.exists(img_path) and os.path.exists(gt_path):
            train_dataset_params.append({
                'root_dir': img_path,
                'soft_gt_dir': gt_path,
                'subset_fraction': args.subset_fraction_train
            })
        else:
            print(f"⚠️  Warning: Training dataset or Soft GT not found at: {dataset_path}")
    
    train_loader = None
    if train_dataset_params:
        train_loader = get_updated_dataloader(
            train_dataset_params,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            augment=True,
            target_size=args.training_image_size,
            shuffle_order=should_shuffle 
        )
    
    # -------------------------------------------------
    # 2. 创建验证集 Loader (列表)
    # -------------------------------------------------
    val_loaders = []
    for dataset_name in args.val_datasets:
        dataset_path = os.path.join(args.datasets_root, dataset_name, 'TE')
        
        gt_path = os.path.join(dataset_path, 'soft_gt')
        img_path = os.path.join(dataset_path, 'focus_stack')

        if os.path.exists(img_path) and os.path.exists(gt_path):
            val_loader = get_updated_dataloader(
                [{
                    'root_dir': img_path,
                    'soft_gt_dir': gt_path,
                    'subset_fraction': args.subset_fraction_val
                }],
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                augment=False,
                target_size=args.training_image_size,
                shuffle_order=should_shuffle 
            )
            if val_loader is not None:
                val_loaders.append(val_loader)
        else:
            print(f"⚠️  Warning: Validation dataset or Soft GT not found at: {dataset_path}")
    
    return train_loader, val_loaders

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    """
    训练单个 Epoch
    """
    model.train()
    total_loss = 0.0
    
    progress_bar = tqdm(
        train_loader, 
        desc=f"🔥 Epoch {epoch+1}/{total_epochs} [Train]",
        ncols=120,
        bar_format='{l_bar}{bar:20}{r_bar}'
    )

    for batch_idx, (image_stack, soft_gt, stack_size) in enumerate(progress_bar):
        # image_stack: [B, N, H, W]
        # soft_gt: [B, N, H, W] (Float32)
        image_stack, soft_gt = image_stack.to(device), soft_gt.to(device)

        optimizer.zero_grad()
        
        # 训练模式下，Network 返回 logits [B, N, H, W]
        logits = model(image_stack)
        
        # 计算 Loss (FusionLoss 内部处理 OT/KL 和 Spatial)
        loss = criterion(logits, soft_gt)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 更新进度条
        progress_bar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Avg": f"{total_loss/(batch_idx+1):.4f}",
        })

    return total_loss / len(train_loader)

def validate_dataset(model, val_loader, device, epoch, save_path, dataset_name, loss_mode, lambda_spatial):
    """
    验证函数：
    确保 Loss 计算与训练时逻辑一致 (Total = Main + Spatial)。
    由于 Eval 模式下模型返回 probs，我们需要手动组合 loss。
    
    指标说明：
    - Depth MAE: 预测期望深度与GT期望深度之间的平均绝对误差（越小越好）
    - 这是 Soft GT 下更合理的评估指标，衡量"聚焦位置"的准确性
    """
    model.eval()
    val_loss_accum = 0.0
    val_main_loss_accum = 0.0 # 单独记录主损失以便观察
    val_spatial_loss_accum = 0.0 # 单独记录空间损失
    
    # 期望深度误差累积
    depth_mae_accum = 0.0
    
    # 实例化空间损失计算模块 (用于验证集)
    spatial_criterion = SpatialSmoothnessLoss(mode=loss_mode).to(device) if lambda_spatial > 0 else None
    
    os.makedirs(save_path, exist_ok=True)

    progress_bar = tqdm(
        val_loader, 
        desc=f"📊 Val {dataset_name}",
        ncols=140, # 稍微宽一点显示更多信息
        bar_format='{l_bar}{bar:30}{r_bar}',
        colour='blue'
    )

    with torch.no_grad():
        for i, (image_stack, soft_gt, stack_size) in enumerate(progress_bar):
            image_stack, soft_gt = image_stack.to(device), soft_gt.to(device)
            B, N, H, W = soft_gt.shape

            # 推理模式下，Network 返回:
            # fused_image: [B, 1, H, W]
            # probs: [B, N, H, W] (已经经过 Softmax)
            fused_image, probs = model(image_stack)
            
            # === 1. 计算 Main Loss (Fidelity) ===
            main_loss = 0.0
            if loss_mode == 'ordered':
                # OT Loss (CDF L1)
                pred_cdf = torch.cumsum(probs, dim=1)
                gt_cdf = torch.cumsum(soft_gt, dim=1)
                main_loss = torch.mean(torch.abs(pred_cdf - gt_cdf)).item()
            else:
                # KL Divergence
                # probs 已经是 softmax 结果，取 log 得到 log_probs
                log_probs = torch.log(probs + 1e-8) 
                main_loss = F.kl_div(log_probs, soft_gt, reduction='batchmean').item()
            
            # === 2. 计算 Spatial Loss ===
            spatial_loss = 0.0
            if spatial_criterion is not None:
                # SpatialSmoothnessLoss 接受 probs
                spatial_loss = spatial_criterion(probs).item()
                
            # === 3. 计算 Total Loss ===
            total_batch_loss = main_loss + lambda_spatial * spatial_loss
            
            # 累加统计
            val_loss_accum += total_batch_loss
            val_main_loss_accum += main_loss
            val_spatial_loss_accum += spatial_loss
            
            # === 4. 计算期望深度误差 (Expected Depth Error) ===
            # 更适合 Soft GT 的评估指标
            # 期望深度 = sum(layer_index * probability)
            layer_indices = torch.arange(N, device=probs.device, dtype=torch.float32).view(1, N, 1, 1)
            
            pred_expected_depth = torch.sum(layer_indices * probs, dim=1)  # [B, H, W]
            gt_expected_depth = torch.sum(layer_indices * soft_gt, dim=1)  # [B, H, W]
            
            # 深度 MAE (Mean Absolute Error)
            batch_depth_mae = torch.mean(torch.abs(pred_expected_depth - gt_expected_depth)).item()
            depth_mae_accum += batch_depth_mae

            # 更新进度条
            progress_bar.set_postfix({
                "Loss": f"{total_batch_loss:.4f}",
                "Main": f"{main_loss:.4f}",
                "Spa": f"{spatial_loss:.4f}",
                "MAE": f"{batch_depth_mae:.3f}"
            })

            # === 5. 可视化 (只保存最后一个 batch) ===
            if i == len(val_loader) - 1:
                visualization_path = os.path.join(save_path, f'epoch_{epoch}')
                
                # 可视化期望深度图（更连续、更有意义）
                gt_depth_vis = gt_expected_depth.unsqueeze(1)  # [B, 1, H, W]
                pred_depth_vis = pred_expected_depth.unsqueeze(1)  # [B, 1, H, W]
                
                to_image(gt_depth_vis, epoch, 'depth_gt', visualization_path)
                to_image(pred_depth_vis, epoch, 'depth_pred', visualization_path)
                to_image(fused_image, epoch, 'fused_image', visualization_path)

    num_batches = len(val_loader)
    avg_total_loss = val_loss_accum / num_batches
    avg_main_loss = val_main_loss_accum / num_batches
    avg_depth_mae = depth_mae_accum / num_batches

    # 返回: Total Loss, Main Loss, Depth MAE
    return (avg_total_loss, avg_main_loss, avg_depth_mae)

def main():
    # 1. 解析参数
    args = parse_args()
    print_banner()
    
    # 2. 配置路径和日志
    # save_name 加上 mode 后缀，方便区分
    full_save_name = f"{args.save_name}_{args.loss_mode}"
    model_save_path = config_model_dir(resume=False, subdir_name=full_save_name)
    writer = SummaryWriter(log_dir=model_save_path)
    
    # 3. 创建 DataLoader
    train_loader, val_loaders = create_dataset_loaders(args)
    
    # 4. 初始化模型
    model = StackMFF_V5()
    num_params = count_parameters(model)
    print_model_info(model, num_params)
    
    # 5. 设备配置 (支持单卡/多卡)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.gpu_ids is not None:
         if len(args.gpu_ids) == 1:
             device = torch.device(f"cuda:{args.gpu_ids[0]}")
             model.to(device)
         else:
             model.to(device)
             model = nn.DataParallel(model, device_ids=args.gpu_ids)
    else:
        model.to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    print(f"🔧 Device: {device}")
    print(f"📉 Loss Mode: {args.loss_mode.upper()}")
    print(f"🌊 Spatial Lambda: {args.lambda_spatial}")
    print_dataset_info(train_loader, val_loaders, args)
    
    # 6. 初始化 Loss Function
    # 使用我们在 loss.py 中定义的 FusionLoss
    criterion = FusionLoss(mode=args.loss_mode, lambda_spatial=args.lambda_spatial).to(device)
    
    # 7. 优化器和调度器
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = ExponentialLR(optimizer, gamma=args.lr_decay)
    
    print_training_config(args, optimizer, scheduler)
    
    # 8. 训练循环
    best_val_loss = float('inf')
    best_epoch = -1
    start_time = time.time()
    val_results_data = []
    
    for epoch in range(args.num_epochs):
        train_loss = 0.0
        
        # --- Training ---
        if train_loader:
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args.num_epochs)
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
        
        # --- Validation ---
        val_results = [] # 存储每个数据集的结果
        epoch_val_data = {'epoch': epoch + 1, 'train_loss': train_loss}
        
        current_epoch_val_loss_sum = 0.0
        
        for i, val_loader in enumerate(val_loaders):
            dataset_name = args.val_datasets[i] if i < len(args.val_datasets) else f"dataset_{i+1}"
            
            # 传入 loss_mode 和 lambda_spatial，确保验证指标与训练一致
            results = validate_dataset(
                model, val_loader, device, epoch, 
                os.path.join(model_save_path, f'val_{dataset_name}'), 
                dataset_name,
                args.loss_mode,
                args.lambda_spatial # 传入参数
            )
            val_results.append(results)
            
            # results: (total_loss, main_loss, depth_mae)
            v_total_loss, v_main_loss, v_depth_mae = results
            current_epoch_val_loss_sum += v_total_loss
            
            writer.add_scalar(f'Loss/val/{dataset_name}/total', v_total_loss, epoch)
            writer.add_scalar(f'Loss/val/{dataset_name}/main', v_main_loss, epoch)
            writer.add_scalar(f'DepthMAE/val/{dataset_name}', v_depth_mae, epoch)
            
            epoch_val_data.update({
                f'val_{dataset_name}_total_loss': v_total_loss,
                f'val_{dataset_name}_main_loss': v_main_loss,
                f'val_{dataset_name}_depth_mae': v_depth_mae
            })
        
        val_results_data.append(epoch_val_data)
        
        # 保存 CSV 日志
        pd.DataFrame(val_results_data).to_csv(os.path.join(model_save_path, 'results.csv'), index=False)
        
        # 保存 Checkpoint
        save_dir = os.path.join(model_save_path, 'checkpoints')
        os.makedirs(save_dir, exist_ok=True)
        
        state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        if (epoch + 1) % 5 == 0:
            torch.save(state_dict, os.path.join(save_dir, f'epoch_{epoch}.pth'))
        
        # 保存 Best Model (基于验证集平均 Total Loss)
        if val_loaders:
            avg_val_loss = current_epoch_val_loss_sum / len(val_loaders)
            
            improved = False
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                improved = True
                torch.save(state_dict, os.path.join(model_save_path, 'best_model.pth'))
                
            print(f"Epoch {epoch+1} Summary | Train Loss: {train_loss:.4f} | Val Total Loss: {avg_val_loss:.4f} {'⭐' if improved else ''}")

        scheduler.step()
    
    print_training_complete(start_time, model_save_path)
    writer.close()

if __name__ == "__main__":
    main()