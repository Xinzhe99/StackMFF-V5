from argparse import ArgumentParser, Namespace
import torch
from accelerate.utils import set_seed
from gmff.inference.loop import InferenceLoop  # Changed from GMFFInferenceLoop to InferenceLoop
import time
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: thop not available. Install with: pip install thop")

def check_device(device: str) -> str:
    if device == "cuda":
        if not torch.cuda.is_available():
            print(
                "CUDA not available because the current PyTorch install was not "
                "built with CUDA enabled."
            )
            device = "cpu"
    else:
        if device == "mps":
            if not torch.backends.mps.is_available():
                if not torch.backends.mps.is_built():
                    print(
                        "MPS not available because the current PyTorch install was not "
                        "built with MPS enabled."
                    )
                    device = "cpu"
                else:
                    print(
                        "MPS not available because the current MacOS version is not 12.3+ "
                        "and/or you do not have an MPS-enabled device on this machine."
                    )
                    device = "cpu"
    print(f"using device {device}")
    return device


DEFAULT_POS_PROMPT = (
    "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, "
    "hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, "
    "skin pore detailing, hyper sharpness, perfect without deformations."
)

DEFAULT_NEG_PROMPT = (
    "painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, "
    "CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, "
    "signature, jpeg artifacts, deformed, lowres, over-smooth."
)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    # model parameters
    parser.add_argument(
        "--task",
        type=str,
        default="gmff",
        choices=["gmff"],
        help="Task you want to do. Ignore this option if you are using self-trained model.",
    )
    parser.add_argument(
        "--upscale", type=float, default=1, help="Upscale factor of output."
    )
    parser.add_argument(
        "--version",
        type=str,
        default="gmff",
        choices=["gmff"],
        help="GMFF model.",
    )
    parser.add_argument(
        "--train_cfg",
        type=str,
        default="configs/train/train_stage2.yaml",
        help="Path to training config. Only works when version is custom.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=r"/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/project_image_stack_fusion/GMFF/experiments/train_gmff_stage2/checkpoints/0010000.pt",
        help="Path to saved checkpoint. Only works when version is custom.",
    )
    # sampling parameters
    parser.add_argument(
        "--sampler",
        type=str,
        default="spaced",
        choices=[
            "dpm++_m2",
            "spaced",
            "ddim",
            "edm_euler",
            "edm_euler_a",
            "edm_heun",
            "edm_dpm_2",
            "edm_dpm_2_a",
            "edm_lms",
            "edm_dpm++_2s_a",
            "edm_dpm++_sde",
            "edm_dpm++_2m",
            "edm_dpm++_2m_sde",
            "edm_dpm++_3m_sde",
        ],
        help="Sampler type. Different samplers may produce very different samples.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Sampling steps. More steps, more details.",
    )
    parser.add_argument(
        "--start_point_type",
        type=str,
        choices=["noise", "cond"],
        default="noise",
        help=(
            "For DiffBIR v1 and v2, setting the start point types to 'cond' can make the results much more stable "
            "and ensure that the outcomes from ODE samplers like DDIM and DPMS are normal. "
            "However, this adjustment may lead to a decrease in sample quality."
        ),
    )
    parser.add_argument(
        "--cleaner_tiled",
        action="store_true",
        help="Enable tiled inference for stage-1 model, which reduces the GPU memory usage.",
    )
    parser.add_argument(
        "--cleaner_tile_size", type=int, default=512, help="Size of each tile."
    )
    parser.add_argument(
        "--cleaner_tile_stride", type=int, default=256, help="Stride between tiles."
    )
    parser.add_argument(
        "--vae_encoder_tiled",
        action="store_true",
        help="Enable tiled inference for AE encoder, which reduces the GPU memory usage.",
    )
    parser.add_argument(
        "--vae_encoder_tile_size", type=int, default=256, help="Size of each tile."
    )
    parser.add_argument(
        "--vae_decoder_tiled",
        action="store_true",
        help="Enable tiled inference for AE decoder, which reduces the GPU memory usage.",
    )
    parser.add_argument(
        "--vae_decoder_tile_size", type=int, default=256, help="Size of each tile."
    )
    parser.add_argument(
        "--cldm_tiled",
        action="store_true",
        help="Enable tiled sampling, which reduces the GPU memory usage.",
    )
    parser.add_argument(
        "--cldm_tile_size", type=int, default=512, help="Size of each tile."
    )
    parser.add_argument(
        "--cldm_tile_stride", type=int, default=256, help="Stride between tiles."
    )
    parser.add_argument(
        "--captioner",
        type=str,
        choices=["none", "llava", "ram"],
        default="none",
        help="No caption needed for GMFF.",
    )
    parser.add_argument(
        "--pos_prompt",
        type=str,
        default='',
        help=(
            "Descriptive words for 'good image quality'. "
            "It can also describe the things you WANT to appear in the image."
        ),
    )
    parser.add_argument(
        "--neg_prompt",
        type=str,
        default='low quality, blurry, low-resolution, noisy, unsharp, weird textures',
        help=(
            "Descriptive words for 'bad image quality'. "
            "It can also describe the things you DON'T WANT to appear in the image."
        ),
    )
    parser.add_argument(
        "--cfg_scale", type=float, default=4.0, help="Classifier-free guidance scale."
    )
    parser.add_argument(
        "--rescale_cfg",
        action="store_true",
        help="Gradually increase cfg scale from 1 to ('cfg_scale' + 1)",
    )
    parser.add_argument(
        "--noise_aug",
        type=int,
        default=0,
        help="Level of noise augmentation. More noise, more creative.",
    )
    parser.add_argument(
        "--s_churn",
        type=float,
        default=0,
        help="Randomness in sampling. Only works with some edm samplers.",
    )
    parser.add_argument(
        "--s_tmin",
        type=float,
        default=0,
        help="Minimum sigma for adding ramdomness to sampling. Only works with some edm samplers.",
    )
    parser.add_argument(
        "--s_tmax",
        type=float,
        default=300,
        help="Maximum  sigma for adding ramdomness to sampling. Only works with some edm samplers.",
    )
    parser.add_argument(
        "--s_noise",
        type=float,
        default=1,
        help="Randomness in sampling. Only works with some edm samplers.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1,
        help="I don't understand this parameter. Leave it as default.",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=1,
        help="Order of solver. Only works with edm_lms sampler.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1,
        help="Control strength from ControlNet. Less strength, more creative.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Nothing to say.")
    # guidance parameters
    parser.add_argument(
        "--guidance", action="store_true", help="Enable restoration guidance."
    )
    parser.add_argument(
        "--g_loss",
        type=str,
        default="w_mse",
        choices=["mse", "w_mse"],
        help="Loss function of restoration guidance.",
    )
    parser.add_argument(
        "--g_scale",
        type=float,
        default=0.0,
        help="Learning rate of optimizing the guidance loss function.",
    )
    # common parameters
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to folder that contains your low-quality images.",
    )
    parser.add_argument(
        "--n_samples", type=int, default=1, help="Number of samples for each image."
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save restored results."
    )
    parser.add_argument("--seed", type=int, default=231)
    # mps has not been tested
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"]
    )
    parser.add_argument(
        "--precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"]
    )
    parser.add_argument("--llava_bit", type=str, default="4", choices=["16", "8", "4"])

    return parser.parse_args()

def main():
    args = parse_args()
    args.device = check_device(args.device)
    set_seed(args.seed)

    if args.version == "gmff":
        loop = InferenceLoop(args)
        
        # 统计模型参数和大小
        print("\n" + "="*60)
        print("Model Statistics")
        print("="*60)
        
        # 计算参数量
        total_params = 0
        for param in loop.cldm.parameters():
            total_params += param.numel()
        
        params_m = total_params / 1e6
        print(f"Total Parameters (CLDM): {params_m:.2f} M")
        
        # 计算模型大小 (根据精度计算实际大小)
        bytes_per_param = {
            "fp32": 4,
            "fp16": 2,
            "bf16": 2,
        }[args.precision]
        model_size_mb = total_params * bytes_per_param / (1024 * 1024)
        print(f"Model Size ({args.precision}): {model_size_mb:.2f} MB")
        print("="*60 + "\n")
        
        # 运行推理并记录时间和FLOPs
        loop.run()
        
        # 打印统计信息
        if hasattr(loop, 'inference_times') and len(loop.inference_times) > 0:
            avg_time = sum(loop.inference_times) / len(loop.inference_times)
            print("\n" + "="*60)
            print("Inference Statistics")
            print("="*60)
            print(f"Total Images Processed: {len(loop.inference_times)}")
            print(f"Average Inference Time: {avg_time:.4f} s")
            
            if hasattr(loop, 'flops_list') and len(loop.flops_list) > 0:
                # 过滤掉失败的计算（值为0）
                valid_flops = [f for f in loop.flops_list if f > 0]
                if valid_flops:
                    avg_flops_g = sum(valid_flops) / len(valid_flops) / 1e9
                    print(f"Average FLOPs (per forward): {avg_flops_g:.2f} G")
                    print(f"Note: Total FLOPs ≈ {avg_flops_g * args.steps * 2:.2f} G (considering {args.steps} steps + CFG)")
                    print(f"Successfully calculated FLOPs for {len(valid_flops)}/{len(loop.flops_list)} images")
                else:
                    print("All FLOPs calculations failed")
            else:
                if not THOP_AVAILABLE:
                    print("FLOPs calculation unavailable (thop not installed)")
                    print("Install with: pip install thop")
                else:
                    print("FLOPs calculation failed")
            print("="*60 + "\n")
        
    print("done!")

if __name__ == "__main__":
    main()