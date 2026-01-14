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
from Dataloader_stage2 import create_dataloader
from model.fusion_encoder import FusionEncoder

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

    # ==================== 创建融合编码器 ====================
    fusion_cfg = cfg.model.fusion_network
    fusion_encoder = FusionEncoder(
        vae_model_id=fusion_cfg.vae_model_id,
        vae_subfolder=fusion_cfg.vae_subfolder,
        depth_transformer_cfg=fusion_cfg.depth_transformer,
        fusion_weights_path=fusion_cfg.fusion_weights_path,
        latent_scale_factor=cldm.scale_factor
    )
    if accelerator.is_main_process:
        print("✓ Fusion encoder (VAE + DepthTransformer) created and frozen")

    # Setup optimizer (只优化 ControlNet)
    opt = torch.optim.AdamW(cldm.controlnet.parameters(), lr=cfg.train.learning_rate)

    # Setup data:
    training_image_size = cfg.dataset.train.params.get('training_image_size', 512)
    loader, dataset = create_dataloader(
        cfg, 
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        training_image_size=training_image_size
    )
    if accelerator.is_main_process:
        print(f"Dataset contains {len(dataset):,} image stacks")

    # Prepare models for training:
    cldm.train().to(device)
    diffusion.to(device)
    fusion_encoder.to(device)
    fusion_encoder.eval()  # 融合编码器始终保持 eval 模式
    
    cldm, opt, loader = accelerator.prepare(cldm, opt, loader)
    pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)
    noise_aug_timestep = cfg.train.noise_aug_timestep

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
        print("Training Configuration:")
        print(f"  - Only ControlNet is being trained")
        print(f"  - VAE, DepthTransformer, UNet, CLIP are frozen")
        print(f"  - Learning rate: {cfg.train.learning_rate}")
        print(f"  - Batch size: {cfg.train.batch_size}")
        print(f"  - Image size: {training_image_size}")
        print("=" * 60)

    while global_step < max_steps:
        pbar = tqdm(
            iterable=None,
            disable=not accelerator.is_main_process,
            unit="batch",
            total=len(loader),
        )
        for batch in loader:
            # 从字典中提取数据
            image_stack = batch["image_stack"].to(device)  # [B, N, C, H, W]
            aif = batch["aif"].to(device)  # [B, C, H, W]
            prompt = batch["prompt"]

            with torch.no_grad():
                # 1. 对GT图像进行VAE编码 (作为diffusion的目标)
                z_0 = fusion_encoder.encode_image(aif)  # [B, 4, H//8, W//8]
                
                # 2. 使用融合编码器处理图像栈，得到融合后的隐变量 (作为ControlNet的条件)
                fused_latent = fusion_encoder(image_stack)  # [B, 4, H//8, W//8]
                
                # 3. 准备文本条件
                c_txt = pure_cldm.clip.encode(prompt)
                
                # 4. 构建条件字典
                cond = {
                    "c_txt": c_txt,
                    "c_img": fused_latent  # 使用融合后的隐变量作为条件
                }
                cond_aug = cond  # 不使用noise augmentation

            t = torch.randint(
                0, diffusion.num_timesteps, (z_0.shape[0],), device=device
            )

            loss = diffusion.p_losses(cldm, z_0, t, cond_aug)
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()

            # ========== 显存优化：清理中间变量 ==========
            del z_0, fused_latent, c_txt, cond, cond_aug
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # ===========================================

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

            if global_step % cfg.train.image_every == 0 or global_step == 1:
                N = min(8, image_stack.shape[0])
                log_image_stack = image_stack[:N]
                log_cond = {k: v[:N] for k, v in cond.items()}
                log_aif = aif[:N]
                log_prompt = prompt[:N]
                
                cldm.eval()
                with torch.inference_mode():
                    z = sampler.sample(
                        model=cldm,
                        device=device,
                        steps=50,
                        x_size=(N, *z_0.shape[1:]),
                        cond=log_cond,
                        uncond=None,
                        cfg_scale=1.0,
                        progress=accelerator.is_main_process,
                    )
                    # 采样完成后立即清理采样产生的中间变量
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if accelerator.is_main_process:
                        # 获取图像栈中的第一张图作为条件图像展示
                        log_first_img = log_image_stack[:, 0, :, :, :]  # [N, C, H, W]
                        
                        for tag, image in [
                            ("image/samples", (fusion_encoder.decode_latent(z) + 1) / 2),
                            ("image/gt", (log_aif + 1) / 2),
                            ("image/stack_first", (log_first_img + 1) / 2),
                            (
                                "image/fused_condition_decoded",
                                (fusion_encoder.decode_latent(log_cond["c_img"]) + 1) / 2,
                            ),
                            (
                                "image/prompt",
                                (log_txt_as_img((512, 512), log_prompt) + 1) / 2,
                            ),
                        ]:
                            writer.add_image(tag, make_grid(image, nrow=4), global_step)
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
    parser.add_argument("--config", type=str, default='configs/train/train_stage2.yaml', help="训练配置文件路径")
    args = parser.parse_args()
    main(args)