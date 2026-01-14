import torch
import os
# 设置 Hugging Face 镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_TOKEN"] = "hf_xxxxyyyzzzz"  # 替换为你的实际 Hugging Face 访问令牌
from diffusers import DiffusionPipeline
from diffusers.utils import load_image

# switch to "mps" for apple devices
pipe = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    dtype=torch.bfloat16,
    device_map="cuda",
    use_auth_token=os.environ["HF_TOKEN"]
)

prompt = "Turn this cat into a dog"
input_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")

image = pipe(image=input_image, prompt=prompt).images[0]