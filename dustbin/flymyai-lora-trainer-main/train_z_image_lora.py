import argparse
import logging

import os

import shutil
import copy

import datasets
import torch

import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from tqdm.auto import tqdm
from diffusers.utils import convert_state_dict_to_diffusers
from peft.utils import get_peft_model_state_dict
from peft import TaskType

from omegaconf import OmegaConf
from copy import deepcopy
import diffusers
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from optimum.quanto import quantize, qfloat8, freeze, qint8
from diffusers.utils.torch_utils import is_compiled_module


from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    ZImageTransformer2DModel,
)

from peft import LoraConfig

from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from image_datasets.dataset import loader
logger = get_logger(__name__, log_level="INFO")
from diffusers import ZImagePipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config",
    )
    args = parser.parse_args()


    return args.config


def main():
    args = OmegaConf.load(parse_args())
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        #dynamo_plugin=dynamo_plugin
    )
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    text_encoding_pipeline = ZImagePipeline.from_pretrained(
        args.pretrained_model_name_or_path, transformer=None, vae=None, torch_dtype=weight_dtype
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
    )
    z_image_transformer = ZImageTransformer2DModel.from_pretrained(
        args.pretrained_transformer, subfolder="transformer"
    )


    quantize(z_image_transformer, weights=qfloat8)#qfloat8)
    freeze(z_image_transformer)
    z_image_transformer.to(accelerator.device)

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    z_image_transformer.add_adapter(lora_config)
    quantize(text_encoding_pipeline.text_encoder, weights=qfloat8)
    freeze(text_encoding_pipeline.text_encoder)
    text_encoding_pipeline.to(accelerator.device)
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
        
    vae.requires_grad_(False)
    z_image_transformer.requires_grad_(False)

    z_image_transformer.train()
    optimizer_cls = torch.optim.AdamW
    for n, param in z_image_transformer.named_parameters():
        if 'lora' not in n:
            param.requires_grad = False
            pass
        else:
            param.requires_grad = True
            print(n)
    print(sum([p.numel() for p in z_image_transformer.parameters() if p.requires_grad]) / 1000000, 'parameters')
    lora_layers = filter(lambda p: p.requires_grad, z_image_transformer.parameters())

    z_image_transformer.enable_gradient_checkpointing()
    optimizer = optimizer_cls(
        lora_layers,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataloader = loader(**args.data_config)    

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    global_step = 0
    vae.to(accelerator.device, dtype=weight_dtype)
    lora_layers, optimizer, _, lr_scheduler = accelerator.prepare(
        lora_layers, optimizer, deepcopy(train_dataloader), lr_scheduler
    )


    initial_global_step = 0

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, {"test": None})

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    train_loss = 0.0
    for steps, batch in enumerate(train_dataloader):
        with accelerator.accumulate(z_image_transformer):
            img, prompts = batch
            with torch.no_grad():
                pixel_values = img.to(dtype=weight_dtype).to(accelerator.device)
                pixel_latents = vae.encode(pixel_values).latent_dist.sample()

                pixel_latents = (pixel_latents - vae.config.shift_factor) * vae.config.scaling_factor
                bsz = pixel_latents.shape[0]
                noise = torch.randn_like(pixel_latents, device=accelerator.device, dtype=weight_dtype)
                u = compute_density_for_timestep_sampling(
                    weighting_scheme="none",
                    batch_size=bsz,
                    logit_mean=0.0,
                    logit_std=1.0,
                    mode_scale=1.29,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=pixel_latents.device)
            sigmas = get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
            noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise
            # Concatenate across channels.
            # Question: Should we concatenate before adding noise?

            # pack the latents.
            noisy_model_input2 = noisy_model_input.unsqueeze(2).to(weight_dtype)
            print("noisy_model_input", noisy_model_input2.shape)

            noisy_model_input = list(noisy_model_input2.unbind(dim=0))

            # latent image ids for RoPE.
            with torch.no_grad():
                prompt_embeds, _ = text_encoding_pipeline.encode_prompt(
                    list(prompts), do_classifier_free_guidance=False
                )
            model_pred = z_image_transformer(
                noisy_model_input,
                (1000 - timesteps) / 1000,
                prompt_embeds
            )[0]
            model_pred = torch.stack([t for t in model_pred], dim=0) * -1
            print("model_pred", model_pred.shape)

            weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
            target = noise - pixel_latents
            target = target.unsqueeze(2)

            print("target", target.shape)
            # flow-matching loss
            loss = torch.mean(
                (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                1,
            )
            
            loss = loss.mean()
            print('loss')
            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
            train_loss += avg_loss.item() / args.gradient_accumulation_steps

            # Backpropagate
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(z_image_transformer.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1
            accelerator.log({"train_loss": train_loss}, step=global_step)
            train_loss = 0.0

            if global_step % args.checkpointing_steps == 0:
                if accelerator.is_main_process:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")

                #accelerator.save_state(save_path)
                try:
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)
                except:
                    pass
                unwrapped_z_image_transformer = unwrap_model(z_image_transformer)
                z_image_transformer_lora_state_dict = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(unwrapped_z_image_transformer)
                )

                ZImagePipeline.save_lora_weights(
                    save_path,
                    z_image_transformer_lora_state_dict,
                    safe_serialization=True,
                )

                logger.info(f"Saved state to {save_path}")

        logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
