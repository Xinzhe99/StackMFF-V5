from argparse import ArgumentParser, Namespace
import torch
from accelerate.utils import set_seed
from inference.stack_inference import StackInferenceLoop
import time

def check_device(device: str) -> str:
    if device == "cuda":
        if not torch.cuda.is_available():
            print("CUDA not available, using cpu")
            device = "cpu"
    return device

def parse_args() -> Namespace:
    parser = ArgumentParser()
    
    # Model Configs
    parser.add_argument(
        "--train_cfg",
        type=str,
        default="configs/train/train_stage2.yaml",
        help="Path to training config.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to saved Stage 2 ControlNet checkpoint.",
    )
    
    # Sampling Parameters
    parser.add_argument("--steps", type=int, default=50, help="Sampling steps.")
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="CFG scale.")
    parser.add_argument("--image_size", type=int, default=512, help="Image size for resizing stack inputs.")
    parser.add_argument("--rescale_cfg", action="store_true", help="Rescale CFG.")
    
    # Prompts
    parser.add_argument("--pos_prompt", type=str, default="", help="Positive prompt.")
    parser.add_argument(
        "--neg_prompt", 
        type=str, 
        default="low quality, blurry, low-resolution, noisy, unsharp, weird textures", 
        help="Negative prompt."
    )
    
    # I/O
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to single stack folder OR root folder containing multiple stack folders.",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save results."
    )
    
    # System
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])

    return parser.parse_args()

def main():
    args = parse_args()
    args.device = check_device(args.device)
    set_seed(args.seed)

    print("="*60)
    print("StackMFF V5 Inference")
    print("="*60)
    
    loop = StackInferenceLoop(args)
    loop.run()
    
    print("Done!")

if __name__ == "__main__":
    main()