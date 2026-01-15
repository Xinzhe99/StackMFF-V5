import os
import argparse
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from Dataloader import get_updated_dataloader
from network import StackMFF_V5
import pandas as pd
from tools import (to_image, count_parameters, config_model_dir, 
                   print_banner, print_model_info, print_device_info, 
                   print_dataset_info, print_training_config, 
                   print_epoch_results, print_training_complete)
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    """
    Parse command line arguments for the training script.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    # Datasetset configuration
    parser = argparse.ArgumentParser(description="StackMFF V5 Training Script")
    parser.add_argument('--save_name', default='train_runs')
    parser.add_argument('--datasets_root', 
                        default='/home/ot/Students/xxz/datasets',
                        type=str, help='Root path to all datasets')

    parser.add_argument('--train_datasets', nargs='+', 
                        default=['DIODE-5000'],
                        help='List of datasets to use for training')
    parser.add_argument('--val_datasets', nargs='+',
                        default=['DIODE-5000'],
                        help='List of datasets to use for validation')
    parser.add_argument('--subset_fraction_train', type=float, default=1,
                        help='Fraction of training data to use, default is 0.01')
    parser.add_argument('--subset_fraction_val', type=float, default=0.5,
                        help='Fraction of validation data to use,default is 0.02')

    # Training and model configuration
    parser.add_argument('--training_image_size', type=int, default=256,
                        help='Target image size for training')
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='Batch size for training and validation')
    parser.add_argument('--num_epochs', type=int, default=30, 
                        help='Number of training epochs')
    parser.add_argument('--eval_interval', type=int, default=1, 
                        help='Interval of epochs between evaluations')
    parser.add_argument('--lr', type=float, default=1e-3, 
                        help='Initial learning rate, default is 1e-3')
    parser.add_argument('--lr_decay', type=float, default=0.9, 
                        help='Learning rate decay factor')
    parser.add_argument('--num_workers', type=int, default=8, 
                        help='Number of data loading workers')
    
    # Device configuration
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=None,
                        help='Specific GPU IDs to use (e.g., 0 1 for GPU 0 and 1). If not specified, use all available GPUs in parallel')
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()

def create_dataset_loaders(args):
    """
    Create training and validation data loaders based on simplified configuration.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        tuple: (train_loader, val_loaders)
    """
    # Create training dataset parameters
    train_dataset_params = []
    for dataset_name in args.train_datasets:
        dataset_path = os.path.join(args.datasets_root, dataset_name, 'TR')
        if os.path.exists(dataset_path):
            train_dataset_params.append({
                'root_dir': os.path.join(dataset_path, 'focus_stack'),
                'fused_gt': os.path.join(dataset_path, 'AiF'),
                'subset_fraction': args.subset_fraction_train
            })
        else:
            print(f"⚠️  Warning: Training dataset path not found: {dataset_path}")
    
    # Create training data loader
    train_loader = None
    if train_dataset_params:
        train_loader = get_updated_dataloader(
            train_dataset_params,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            augment=True,
            target_size=args.training_image_size
        )
    
    # Create validation dataset loaders (separate for each dataset)
    val_loaders = []
    for dataset_name in args.val_datasets:
        dataset_path = os.path.join(args.datasets_root, dataset_name, 'TE')
        if os.path.exists(dataset_path):
            val_loader = get_updated_dataloader(
                [{
                    'root_dir': os.path.join(dataset_path, 'focus_stack'),
                    'fused_gt': os.path.join(dataset_path, 'AiF'),
                    'subset_fraction': args.subset_fraction_val
                }],
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                augment=False,
                target_size=args.training_image_size
            )
            val_loaders.append(val_loader)
        else:
            print(f"⚠️  Warning: Validation dataset path not found: {dataset_path}")
    
    return train_loader, val_loaders

def train(model, train_loader, optimizer, device, epoch, total_epochs):
    """训练一个 epoch"""
    model.train()
    train_loss = 0.0
    
    # 创建进度条
    progress_bar = tqdm(
        train_loader, 
        desc=f"🔥 Epoch {epoch+1}/{total_epochs}",
        ncols=120,
        bar_format='{l_bar}{bar:20}{r_bar}',
    )
    
    # MSE损失函数
    mse_loss = nn.MSELoss()

    for batch_idx, (image_stack, fused_gt, stack_size) in enumerate(progress_bar):
        image_stack, fused_gt = image_stack.to(device), fused_gt.to(device)
        optimizer.zero_grad()
        
        # 网络输出融合图像
        fused_output = model(image_stack)  # [B, C, H, W]
        
        # 使用MSE损失
        total_loss = mse_loss(fused_output, fused_gt)

        # 使用accelerator进行反向传播
        total_loss.backward()

        optimizer.step()

        train_loss += total_loss.item()

        # 更新进度条
        progress_bar.set_postfix({
            "Loss": f"{total_loss.item():.6f}",
            "Avg": f"{train_loss/(batch_idx+1):.6f}",
            "Device": str(device).upper()
        })

    return train_loss / len(train_loader)

def validate_dataset(model, val_loader, device, epoch, save_path, dataset_name):
    """验证函数"""
    
    model.eval()  # 设置为评估模式
    val_loss = 0.0
    
    # 确保验证保存路径存在
    os.makedirs(save_path, exist_ok=True)
    
    # MSE损失函数
    mse_loss = nn.MSELoss()

    progress_bar = tqdm(
        val_loader, 
        desc=f"📊 Validating {dataset_name}",
        ncols=120,
        bar_format='{l_bar}{bar:30}{r_bar}',
        colour='blue',
    )

    with torch.no_grad():
        for i, (image_stack, fused_gt, stack_size) in enumerate(progress_bar):
            image_stack, fused_gt = image_stack.to(device), fused_gt.to(device)
            fused_output = model(image_stack)  # [B, C, H, W]
            
            # 计算MSE损失
            batch_loss = mse_loss(fused_output, fused_gt)
            val_loss += batch_loss.item()

            progress_bar.set_postfix({
                "📉 Loss": f"{batch_loss.item():.6f}",
                "Avg": f"{val_loss/(i+1):.6f}"
            })

            if i == len(val_loader) - 1:
                visualization_path = os.path.join(save_path, f'epoch_{epoch}')
                # VAE输出在[-1,1]范围，需要转换回[0,1]范围以显示
                to_image((fused_gt + 1.0) / 2.0, epoch, 'fused_gt', visualization_path)
                to_image((fused_output + 1.0) / 2.0, epoch, 'fused_output', visualization_path)

    num_batches = len(val_loader)

    return val_loss / num_batches

def main():
    # 解析参数
    args = parse_args()
    
    # 打印banner信息
    print_banner()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 初始化
    model_save_path = config_model_dir(resume=False, subdir_name=args.save_name)
    writer = SummaryWriter(log_dir=model_save_path)
    train_loader, val_loaders = create_dataset_loaders(args)
    
    # 创建模型
    model = StackMFF_V5()
    num_params = count_parameters(model)
    print_model_info(model, num_params)
    
    # 设备配置
    if torch.cuda.is_available():
        # 如果未指定gpu_ids，自动使用所有可用的GPU
        if args.gpu_ids is None:
            gpu_ids = list(range(torch.cuda.device_count()))
        else:
            gpu_ids = args.gpu_ids
        
        device = torch.device(f"cuda:{gpu_ids[0]}")
        model = model.to(device)
        if len(gpu_ids) > 1:
            model = nn.DataParallel(model, device_ids=gpu_ids)
            print_device_info(device, True, len(gpu_ids))
        else:
            print_device_info(device, False, 1)
    else:
        device = torch.device("cpu")
        print_device_info(device, False, 1)
    
    print_dataset_info(train_loader, val_loaders, args)
    
    # 优化器和调度器
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = ExponentialLR(optimizer, gamma=args.lr_decay)
    
    print_training_config(args, optimizer, scheduler)
    
    # 训练循环
    best_val_loss = float('inf')
    best_epoch = -1
    start_time = time.time()
    val_results_data = []
    
    for epoch in range(args.num_epochs):
        train_loss = None
        
        # 训练
        if train_loader:
            train_loss = train(model, train_loader, optimizer, device, epoch, args.num_epochs)
            writer.add_scalar('Loss/train', train_loss, epoch)
        
        # 验证
        val_results = []
        epoch_val_data = {'epoch': epoch + 1}
        
        for i, current_val_loader in enumerate(val_loaders):
            dataset_name = args.val_datasets[i] if i < len(args.val_datasets) else f"dataset_{i+1}"
            val_loss = validate_dataset(model, current_val_loader, device, epoch, 
                                     os.path.join(model_save_path, f'val_{dataset_name}'), 
                                     dataset_name)
            val_results.append(val_loss)
            
            writer.add_scalar(f'Loss/val_{dataset_name}', val_loss, epoch)
            
            epoch_val_data[f'val_{dataset_name}_loss'] = val_loss
        
        writer.add_scalar('LearningRate', scheduler.get_last_lr()[0])
        
        if train_loader:
            epoch_val_data['train_loss'] = train_loss
            epoch_val_data['learning_rate'] = scheduler.get_last_lr()[0]
        
        val_results_data.append(epoch_val_data)
        
        # 保存结果
        val_results_df = pd.DataFrame(val_results_data)
        val_results_df.to_csv(os.path.join(model_save_path, 'validation_results.csv'), index=False)
        
        # 保存模型
        os.makedirs(os.path.join(model_save_path, 'model_save'), exist_ok=True)
        
        # 如果使用DataParallel，获取原始模型
        unwrapped_model = model.module if isinstance(model, nn.DataParallel) else model
        
        # 提取 DepthTransformer 参数
        depth_transformer_state_dict = {}
        for name, param in unwrapped_model.state_dict().items():
            if name.startswith('depth_transformer.'):
                depth_transformer_state_dict[name] = param.cpu()
        
        print(f"   💾 Saving {len(depth_transformer_state_dict)} DepthTransformer parameters for epoch {epoch}")
        
        # 保存当前epoch的DepthTransformer参数
        torch.save(depth_transformer_state_dict, os.path.join(model_save_path, 'model_save', f'epoch_{epoch}.pth'))
        
        # 检查最佳模型
        improved = False
        if val_loaders:
            avg_val_loss = sum(val_results) / len(val_results)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                improved = True
                
                # 保存最佳DepthTransformer参数
                depth_transformer_state_dict = {}
                for name, param in unwrapped_model.state_dict().items():
                    if name.startswith('depth_transformer.'):
                        depth_transformer_state_dict[name] = param.cpu()
                torch.save(depth_transformer_state_dict, os.path.join(model_save_path, 'best_fusion_model.pth'))
        
        # 打印epoch结果
        print_epoch_results(epoch, args.num_epochs, train_loss, val_results, 
                           args.val_datasets[:len(val_loaders)], 
                           scheduler.get_last_lr()[0], best_val_loss, improved)
        
        scheduler.step()
    # 训练完成
    print_training_complete(start_time, model_save_path)
    
    # Print best epoch information
    if best_epoch >= 0:
        best_model_path = os.path.join(model_save_path, "best_fusion_model.pth")
        print(f"\n🏆 Best Model Information:")
        print(f"   📊 Best Epoch: {best_epoch + 1}/{args.num_epochs}")
        print(f"   📉 Best Validation Loss: {best_val_loss:.6f}")
        print(f"   💾 Best Model Path: {best_model_path}")
    else:
        print(f"\n⚠️  No validation performed, no best model selected.")
    
    writer.close()

if __name__ == "__main__":
    main()