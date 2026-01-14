import argparse
import torch
from diffusers import DiffusionPipeline
from optimum.quanto import quantize, qfloat8, freeze
from tqdm.auto import tqdm

def main(args):
    """
    Main function to load the model, apply quantization, and generate an image.
    """
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device = "cuda"
    else:
        torch_dtype = torch.float32
        device = "cpu"

    print(f"Using device: {device} with dtype: {torch_dtype}")

    # Load the base diffusion pipeline
    print(f"Loading model from: {args.model_name}")
    pipe = DiffusionPipeline.from_pretrained(args.model_name, torch_dtype=torch_dtype)

    # Load LoRA weights
    if args.lora_weights:
        print(f"Loading LoRA weights from: {args.lora_weights}")
        pipe.load_lora_weights(args.lora_weights, adapter_name="lora")

    # Set up quantization
    quantization_map = {
        "qfloat8": qfloat8,
    }
    quantization_type = quantization_map.get(args.quantization)
    if quantization_type is None:
        raise ValueError(f"Invalid quantization type: {args.quantization}. Choose from 'qfloat8'")

    print(f"Applying {args.quantization} quantization to the transformer...")

    all_blocks = list(pipe.transformer.transformer_blocks)
    for block in tqdm(all_blocks):
        block.to(device, dtype=torch_dtype)
        quantize(block, weights=quantization_type)
        freeze(block)
        block.to('cpu')
    pipe.transformer.to(device, dtype=torch_dtype)
    quantize(pipe.transformer, weights=quantization_type)
    freeze(pipe.transformer)
    print("Transformer quantization complete.")

    # Offload model to CPU to save VRAM, parts will be moved to GPU as needed
    pipe.enable_model_cpu_offload()

    # Set up the generator for reproducibility
    generator = torch.Generator(device=device).manual_seed(args.seed)

    print("Generating image...")
    # Generate the image
    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.num_inference_steps,
        true_cfg_scale=args.true_cfg_scale,
        generator=generator,
    ).images[0]

    # Save the output image
    image.save(args.output_image)
    print(f"Image successfully saved to {args.output_image}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images with a quantized Qwen-Image model using LoRA.")

    # Model and Weights Arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen-Image", help="Path or name of the base model.")
    parser.add_argument("--lora_weights", type=str, default="", help="Path to the LoRA weights.")
    parser.add_argument("--output_image", type=str, default="generated_image.png", help="Filename for the output image.")

    # Generation Arguments
    parser.add_argument("--prompt", type=str, default='''man in the city''', help="The prompt for image generation.")
    parser.add_argument("--negative_prompt", type=str, default=" ", help="The negative prompt for image generation.")
    parser.add_argument("--width", type=int, default=1024, help="Width of the generated image.")
    parser.add_argument("--height", type=int, default=1024, help="Height of the generated image.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--true_cfg_scale", type=float, default=5.0, help="Classifier-Free Guidance scale.")
    parser.add_argument("--seed", type=int, default=655, help="Random seed for the generator.")

    # Quantization Arguments
    parser.add_argument("--quantization", type=str, default="qfloat8", choices=["qfloat8"], help="The quantization type to apply.")

    args = parser.parse_args()
    main(args)