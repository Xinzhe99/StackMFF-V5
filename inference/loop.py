import os
from typing import Generator, List
from argparse import Namespace

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
import pandas as pd
import torch.nn as nn
import time

try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

from ..utils.common import (
    instantiate_from_config,
    VRAMPeakMonitor,
)
from .pipeline import StackMFFV5Pipeline
from model import ControlLDM, Diffusion


class EmptyCaptioner:
    def __init__(self, device):
        pass
    
    def __call__(self, image):
        return ""


class IdentityCleaner(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x


class InferenceLoop:

    def __init__(self, args: Namespace):
        self.args = args
        self.train_cfg = OmegaConf.load(args.train_cfg)
        self.loop_ctx = {}
        self.pipeline = None  # Will be initialized in load_pipeline
        
        # 初始化性能统计列表
        self.inference_times = []  # 存储每张图片的推理时间
        self.flops_list = []  # 存储每张图片的FLOPs
        
        with VRAMPeakMonitor("loading cleaner model"):
            self.load_cleaner()
        with VRAMPeakMonitor("loading cldm model"):
            self.load_cldm()
        
        # For GMFF, we don't use restoration guidance
        self.cond_fn = None
        
        self.load_pipeline()
        with VRAMPeakMonitor("loading captioner"):
            self.load_captioner()

    def load_cleaner(self) -> None:
        self.cleaner: nn.Module = IdentityCleaner()
        self.cleaner.eval().to(self.args.device)

    def load_cldm(self) -> None:
        self.cldm: ControlLDM = instantiate_from_config(self.train_cfg.model.cldm)

        # load pre-trained SD weight
        sd_weight = torch.load(self.train_cfg.train.sd_path, map_location="cpu")
        sd_weight = sd_weight["state_dict"]
        unused, missing = self.cldm.load_pretrained_sd(sd_weight)
        print(
            f"load pretrained stable diffusion, "
            f"unused weights: {unused}, missing weights: {missing}"
        )
        # load controlnet weight
        control_weight = torch.load(self.args.ckpt, map_location="cpu")
        self.cldm.load_controlnet_from_ckpt(control_weight)
        print(f"load controlnet weight")
        self.cldm.eval().to(self.args.device)
        cast_type = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.args.precision]
        self.cldm.cast_dtype(cast_type)

        # load diffusion
        self.diffusion: Diffusion = instantiate_from_config(
            self.train_cfg.model.diffusion
        )
        self.diffusion.to(self.args.device)

    def load_cond_fn(self) -> None:
        # For GMFF, we don't use restoration guidance, so this is a no-op
        pass

    def load_pipeline(self) -> None:
        self.pipeline = StackMFFV5Pipeline(
            self.cleaner,
            self.cldm,
            self.diffusion,
            self.cond_fn,
            self.args.device,
        )

    # GMFF doesn't use captioner, so use empty captioner
    def load_captioner(self) -> None:
        if self.args.captioner == "none":
            self.captioner = EmptyCaptioner(self.args.device)
        else:
            raise ValueError(f"unsupported captioner: {self.args.captioner}")

    def setup(self) -> None:
        self.save_dir = self.args.output
        os.makedirs(self.save_dir, exist_ok=True)

    def load_lq(self) -> Generator[Image.Image, None, None]:
        img_exts = [".png", ".jpg", ".jpeg"]
        assert os.path.isdir(
            self.args.input
        ), "Please put your low-quality images in a folder."
        for file_name in sorted(os.listdir(self.args.input)):
            stem, ext = os.path.splitext(file_name)
            if ext not in img_exts:
                print(f"{file_name} is not an image, continue")
                continue
            file_path = os.path.join(self.args.input, file_name)
            lq = Image.open(file_path).convert("RGB")
            print(f"load lq: {file_path}")
            self.loop_ctx["file_stem"] = stem
            yield lq

    def after_load_lq(self, lq: Image.Image) -> np.ndarray:
        # For GMFFPipeline with IdentityCleaner, no resizing is needed
        return np.array(lq)

    @torch.no_grad()
    def run(self) -> None:
        self.setup()
        auto_cast_type = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.args.precision]

        for lq in self.load_lq():
            # 记录开始时间
            start_time = time.time()
            
            # prepare prompt
            with VRAMPeakMonitor("applying captioner"):
                caption = self.captioner(lq)
            pos_prompt = ", ".join(
                [text for text in [caption, self.args.pos_prompt] if text]
            )
            neg_prompt = self.args.neg_prompt
            lq = self.after_load_lq(lq)

            # 计算当前图片的FLOPs
            if THOP_AVAILABLE:
                try:
                    # 准备示例输入用于FLOPs计算
                    lq_tensor = (
                        torch.tensor(lq[None], dtype=torch.float32, device=self.args.device)
                        .div(255)
                        .clamp(0, 1)
                        .permute(0, 3, 1, 2)
                        .contiguous()
                    )
                    # 计算cldm的FLOPs
                    h, w = lq_tensor.shape[2:]
                    
                    # 确保尺寸能被64整除（与实际推理中的padding保持一致）
                    h_padded = ((h + 63) // 64) * 64
                    w_padded = ((w + 63) // 64) * 64
                    
                    # 创建示例输入（latent空间尺寸为h/8 x w/8）
                    latent_h = h_padded // 8
                    latent_w = w_padded // 8
                    
                    x_noisy = torch.randn(1, 4, latent_h, latent_w, device=self.args.device, dtype=auto_cast_type)
                    t = torch.tensor([500], device=self.args.device, dtype=torch.long)
                    c_txt = torch.randn(1, 77, 1024, device=self.args.device, dtype=auto_cast_type)
                    c_img = torch.randn(1, 4, latent_h, latent_w, device=self.args.device, dtype=auto_cast_type)
                    cond = {"c_txt": c_txt, "c_img": c_img}
                    
                    # 使用thop计算FLOPs
                    with torch.no_grad():
                        flops, params = profile(self.cldm, inputs=(x_noisy, t, cond), verbose=False)
                        self.flops_list.append(flops)
                        print(f"FLOPs for current image (size {h}x{w}, padded to {h_padded}x{w_padded}): {flops / 1e9:.2f} G")
                except Exception as e:
                    print(f"FLOPs calculation failed for current image: {e}")
                    # 如果是第一张图片就失败，直接退出
                    if len(self.flops_list) == 0:
                        print("ERROR: FLOPs calculation failed on the first image. Exiting...")
                        raise RuntimeError(f"FLOPs calculation failed on the first image: {e}")
                    # 后续图片失败则添加0以保持列表长度一致
                    self.flops_list.append(0)

            # batch process
            n_samples = self.args.n_samples
            batch_size = self.args.batch_size
            num_batches = (n_samples + batch_size - 1) // batch_size
            samples = []
            for i in range(num_batches):
                n_inputs = min((i + 1) * batch_size, n_samples) - i * batch_size
                # Check if pipeline is properly initialized
                if self.pipeline is None:
                    raise RuntimeError("Pipeline not initialized. Make sure load_pipeline() is properly implemented.")
                    
                with torch.autocast(self.args.device, auto_cast_type):
                    batch_samples = self.pipeline.run(
                        np.tile(lq[None], (n_inputs, 1, 1, 1)),
                        self.args.steps,
                        self.args.strength,
                        self.args.cleaner_tiled,
                        self.args.cleaner_tile_size,
                        self.args.cleaner_tile_stride,
                        self.args.vae_encoder_tiled,
                        self.args.vae_encoder_tile_size,
                        self.args.vae_decoder_tiled,
                        self.args.vae_decoder_tile_size,
                        self.args.cldm_tiled,
                        self.args.cldm_tile_size,
                        self.args.cldm_tile_stride,
                        pos_prompt,
                        neg_prompt,
                        self.args.cfg_scale,
                        self.args.start_point_type,
                        self.args.sampler,
                        self.args.noise_aug,
                        self.args.rescale_cfg,
                        self.args.s_churn,
                        self.args.s_tmin,
                        self.args.s_tmax,
                        self.args.s_noise,
                        self.args.eta,
                        self.args.order,
                    )
                samples.extend(list(batch_samples))
            
            # 记录结束时间
            end_time = time.time()
            inference_time = end_time - start_time
            self.inference_times.append(inference_time)
            print(f"Inference time for current image: {inference_time:.4f} s")
            
            self.save(samples, pos_prompt, neg_prompt)

    def save(self, samples: List[np.ndarray], pos_prompt: str, neg_prompt: str) -> None:
        file_stem = self.loop_ctx["file_stem"]
        assert len(samples) == self.args.n_samples
        for i, sample in enumerate(samples):
            file_name = (
                f"{file_stem}_{i}.png"
                if self.args.n_samples > 1
                else f"{file_stem}.png"
            )
            save_path = os.path.join(self.save_dir, file_name)
            Image.fromarray(sample).save(save_path)
            print(f"save result to {save_path}")
        csv_path = os.path.join(self.save_dir, "prompt.csv")
        df = pd.DataFrame(
            {
                "file_name": [file_stem],
                "pos_prompt": [pos_prompt],
                "neg_prompt": [neg_prompt],
            }
        )
        if os.path.exists(csv_path):
            df.to_csv(csv_path, index=False, mode="a", header=False)
        else:
            df.to_csv(csv_path, index=False)