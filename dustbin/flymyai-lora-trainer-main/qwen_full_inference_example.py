"""
Simple Example: Load Trained Qwen Image Model from Checkpoint
This is a minimal example for loading a trained model checkpoint
"""

from diffusers import QwenImagePipeline, QwenImageTransformer2DModel, AutoencoderKLQwenImage
import torch
from omegaconf import OmegaConf
import os


def load_trained_model(checkpoint_path):
    """Load trained model from checkpoint - simple version"""
    print(f"Loading trained model from: {checkpoint_path}")
    
    # Load config to get original model path
    config_path = os.path.join(checkpoint_path, "config.yaml")
    config = OmegaConf.load(config_path)
    original_model_path = config.pretrained_model_name_or_path
    
    # Load trained transformer
    transformer_path = os.path.join(checkpoint_path, "transformer")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        transformer_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    transformer.to("cuda")
    transformer.eval()
    
    # Load VAE from original model
    vae = AutoencoderKLQwenImage.from_pretrained(
        original_model_path,
        subfolder="vae",
        torch_dtype=torch.bfloat16
    )
    vae.to("cuda")
    vae.eval()
    
    # Create pipeline
    pipe = QwenImagePipeline.from_pretrained(
        original_model_path,
        transformer=transformer,
        vae=vae,
        torch_dtype=torch.bfloat16
    )
    pipe.to("cuda")
    
    print("Model loaded successfully!")
    return pipe


def main():
    """Simple example of loading and using trained model"""
    # 1. Load the trained model
    checkpoint_path = "output_full_training/checkpoint-500"
    pipe = load_trained_model(checkpoint_path)
    
    # 2. Generate image
    prompt = "A beautiful landscape with mountains and lake"
    
    image = pipe(
        prompt=prompt,
        width=1024,
        height=1024,
        num_inference_steps=40,
        true_cfg_scale=5,
        generator=torch.Generator(device="cuda").manual_seed(42)
    )
    
    # 3. Save the result
    output_image = image.images[0]
    output_image.save("generated_image.png")
    print("Image saved as: generated_image.png")


if __name__ == "__main__":
    main()
