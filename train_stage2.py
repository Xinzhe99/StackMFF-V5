"""
Stage2 训练脚本（使用预计算隐变量版本）

特点：
- 使用预计算的融合隐变量和GT隐变量
- 不需要加载FusionEncoder，大幅减少显存占用
- 训练速度更快，因为省去了实时VAE编码和融合过程

使用方法：
    1. 首先运行预处理脚本生成隐变量:
       python precompute_fused_latents.py --config configs/train/train_stage2.yaml
    
    2. 然后运行此训练脚本:
       python train_stage2_new.py --config configs/train/train_stage2.yaml --latents_path precomputed_latents/fused_latents.pth
"""

import os
from argparse import ArgumentParser
from omegaconf import OmegaConf
import torch
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from model import ControlLDM, Diffusion
from utils.common import instantiate_from_config, log_txt_as_img
from sampler import SpacedSampler
from Dataloader_stage2_new import create_dataloader


def main(args) -> None:
    # Setup accelerator:
    accelerator = Accelerator(split_batches=True)
    set_seed(231, device_specific=True)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)

    # Setup an experiment folder:
    if accelerator.is_main_process:
        exp_dir = cfg.train.exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Experiment directory created at {exp_dir}")

    # Create model:
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    sd = torch.load(cfg.train.sd_path, map_location="cpu", weights_only=False)["state_dict"]
    unused, missing = cldm.load_pretrained_sd(sd)
    if accelerator.is_main_process:
        print(
            f"strictly load pretrained SD weight from {cfg.train.sd_path}\n"
            f"unused weights: {unused}\n"
            f"missing weights: {missing}"
        )

    # 加载预训练的 ControlNet 权重
    if cfg.train.resume:
        cldm.load_controlnet_from_ckpt(torch.load(cfg.train.resume, map_location="cpu", weights_only=False))
        if accelerator.is_main_process:
            print(
                f"strictly load controlnet weight from checkpoint: {cfg.train.resume}"
            )
    else:
        init_with_new_zero, init_with_scratch = cldm.load_controlnet_from_unet()
        if accelerator.is_main_process:
            print(
                f"strictly load controlnet weight from pretrained SD\n"
                f"weights initialized with newly added zeros: {init_with_new_zero}\n"
                f"weights initialized from scratch: {init_with_scratch}"
            )

    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)

    # ==================== 不需要创建融合编码器！====================
    # 使用预计算的隐变量，所以不需要 FusionEncoder
    # 这大幅减少了显存占用
    if accelerator.is_main_process:
        print("✓ Using precomputed latents - FusionEncoder NOT loaded (saving VRAM)")

    # Setup optimizer (只优化 ControlNet)
    opt = torch.optim.AdamW(cldm.controlnet.parameters(), lr=cfg.train.learning_rate)

    # Setup data - 使用预计算隐变量的数据加载器
    precomputed_latents_path = args.latents_path
    if not os.path.exists(precomputed_latents_path):
        raise FileNotFoundError(f"Precomputed latents file not found: {precomputed_latents_path}")
    
    loader, dataset = create_dataloader(
        precomputed_latents_path=precomputed_latents_path,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        augment=cfg.dataset.train.params.get('augment', True),
        use_grouped_sampler=False  # 分布式训练时禁用分组采样器
    )
    if accelerator.is_main_process:
        print(f"Dataset contains {len(dataset):,} precomputed latent pairs")

    # Prepare models for training:
    cldm.train().to(device)
    diffusion.to(device)
    
    cldm, opt, loader = accelerator.prepare(cldm, opt, loader)
    pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)

    # Variables for monitoring/logging purposes:
    global_step = 0
    max_steps = cfg.train.train_steps
    step_loss = []
    epoch = 0
    epoch_loss = []
    sampler = SpacedSampler(
        diffusion.betas, diffusion.parameterization, rescale_cfg=False
    )
    if accelerator.is_main_process:
        writer = SummaryWriter(exp_dir)
        print(f"Training for {max_steps} steps...")
        print("=" * 60)
        print("Training Configuration (Precomputed Latents):")
        print(f"  - Only ControlNet is being trained")
        print(f"  - UNet, CLIP are frozen")
        print(f"  - VAE and DepthTransformer are NOT loaded (using precomputed)")
        print(f"  - Learning rate: {cfg.train.learning_rate}")
        print(f"  - Batch size: {cfg.train.batch_size}")
        print(f"  - Precomputed latents: {precomputed_latents_path}")
        print("=" * 60)

    while global_step < max_steps:
        pbar = tqdm(
            iterable=None,
            disable=not accelerator.is_main_process,
            unit="batch",
            total=len(loader),
        )
        for batch in loader:
            # 从预计算数据中直接提取隐变量
            fused_latent = batch["fused_latent"].to(device)  # [B, 4, H, W] - ControlNet条件
            z_0 = batch["gt_latent"].to(device)              # [B, 4, H, W] - diffusion目标
            prompt = batch["prompt"]

            with torch.no_grad():
                # 准备文本条件
                c_txt = pure_cldm.clip.encode(prompt)
                
                # 构建条件字典
                cond = {
                    "c_txt": c_txt,
                    "c_img": fused_latent  # 使用预计算的融合隐变量作为条件
                }

            t = torch.randint(
                0, diffusion.num_timesteps, (z_0.shape[0],), device=device
            )

            loss = diffusion.p_losses(cldm, z_0, t, cond)
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()

            accelerator.wait_for_everyone()

            global_step += 1
            step_loss.append(loss.item())
            epoch_loss.append(loss.item())
            pbar.update(1)
            pbar.set_description(
                f"Epoch: {epoch:04d}, Global Step: {global_step:07d}, Loss: {loss.item():.6f}"
            )

            # Log loss values:
            if global_step % cfg.train.log_every == 0 and global_step > 0:
                avg_loss = (
                    accelerator.gather(
                        torch.tensor(step_loss, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
                step_loss.clear()
                if accelerator.is_main_process:
                    writer.add_scalar("loss/loss_simple_step", avg_loss, global_step)

            # Save checkpoint:
            if global_step % cfg.train.ckpt_every == 0 and global_step > 0:
                if accelerator.is_main_process:
                    checkpoint = pure_cldm.controlnet.state_dict()
                    ckpt_path = f"{ckpt_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, ckpt_path)

            # 可视化采样（所有进程都执行采样，只有主进程记录日志）
            if global_step % cfg.train.image_every == 0 or global_step == 1:
                N = min(8, fused_latent.shape[0])
                log_fused_latent = fused_latent[:N]
                log_gt_latent = z_0[:N]
                log_prompt = prompt[:N]
                
                log_cond = {
                    "c_txt": pure_cldm.clip.encode(log_prompt),
                    "c_img": log_fused_latent
                }
                
                cldm.eval()
                with torch.no_grad():
                    z = sampler.sample(
                        model=cldm,
                        device=device,
                        steps=50,
                        x_size=(N, *log_gt_latent.shape[1:]),
                        cond=log_cond,
                        uncond=None,
                        cfg_scale=1.0,
                        progress=accelerator.is_main_process,
                    )
                    if accelerator.is_main_process:
                        # 记录隐变量的统计信息
                        writer.add_histogram("latent/sampled_z", z, global_step)
                        writer.add_histogram("latent/gt_z", log_gt_latent, global_step)
                        writer.add_histogram("latent/fused_condition", log_fused_latent, global_step)
                        
                        # 记录隐变量的可视化
                        def normalize_latent_for_vis(lat):
                            lat = lat.clone()
                            lat = (lat - lat.min()) / (lat.max() - lat.min() + 1e-8)
                            return lat
                        
                        # 只显示第一个通道作为灰度图
                        sampled_vis = normalize_latent_for_vis(z[:, 0:1, :, :].repeat(1, 3, 1, 1))
                        gt_vis = normalize_latent_for_vis(log_gt_latent[:, 0:1, :, :].repeat(1, 3, 1, 1))
                        cond_vis = normalize_latent_for_vis(log_fused_latent[:, 0:1, :, :].repeat(1, 3, 1, 1))
                        
                        writer.add_image("latent_vis/sampled", make_grid(sampled_vis, nrow=4), global_step)
                        writer.add_image("latent_vis/gt", make_grid(gt_vis, nrow=4), global_step)
                        writer.add_image("latent_vis/condition", make_grid(cond_vis, nrow=4), global_step)
                        
                        # prompt 可视化
                        prompt_img = (log_txt_as_img((512, 512), log_prompt) + 1) / 2
                        writer.add_image("image/prompt", make_grid(prompt_img, nrow=4), global_step)
                cldm.train()
            accelerator.wait_for_everyone()

            if global_step == max_steps:
                break

        pbar.close()
        epoch += 1
        avg_epoch_loss = (
            accelerator.gather(torch.tensor(epoch_loss, device=device).unsqueeze(0))
            .mean()
            .item()
        )
        epoch_loss.clear()
        if accelerator.is_main_process:
            writer.add_scalar("loss/loss_simple_epoch", avg_epoch_loss, global_step)

    if accelerator.is_main_process:
        print("done!")
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/train/train_stage2.yaml', 
                       help="训练配置文件路径")
    parser.add_argument("--latents_path", type=str, default='precomputed_latents/fused_latents.pth',
                       help="预计算隐变量文件路径")
    args = parser.parse_args()
    main(args)
