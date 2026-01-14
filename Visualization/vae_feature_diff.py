import os
# 必须在 import diffusers 之前设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from diffusers import AutoencoderKLQwenImage
from PIL import Image
import numpy as np
import argparse
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import OrderedDict

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================
# 0. SimCLR 类定义
# ============================
class SimCLR(nn.Module):
    def __init__(self, encoder, projection_dim, n_features):
        super(SimCLR, self).__init__()
        self.encoder = encoder
        self.n_features = n_features
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)
        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        return h_i, h_j, z_i, z_j

# ============================
# 1. 模型加载函数
# ============================

def load_vae_model(model_id, vae_subfolder, dtype, device, fallback_model):
    logger.info(f"正在加载 Wan VAE (来自 {model_id})...")
    
    if dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    try:
        vae = AutoencoderKLQwenImage.from_pretrained(
            model_id, 
            subfolder=vae_subfolder, 
            torch_dtype=torch_dtype,
            use_safetensors=True
        ).to(device)
    except Exception as e:
        logger.error(f"VAE 模型加载失败 ({e})")
        raise
    
    vae.eval()
    return vae, torch_dtype

def load_resnet_supervised(device):
    logger.info("正在加载 ResNet50 (Supervised ImageNet)...")
    weights = models.ResNet50_Weights.IMAGENET1K_V1
    model = models.resnet50(weights=weights)
    model.fc = nn.Identity() 
    model.to(device)
    model.eval()
    return model

def load_simclr_local(device, weight_path):
    """
    从本地加载 SimCLR 权重。
    """
    logger.info(f"正在加载 SimCLR (本地路径: {weight_path})...")
    
    # 1. 初始化骨架 (ResNet50 without FC)
    encoder = models.resnet50(weights=None)
    n_features = encoder.fc.in_features
    encoder.fc = nn.Identity()
    
    # 2. 初始化 SimCLR 包装器 (Projection Dim=64, 对应标准配置)
    model = SimCLR(encoder, projection_dim=64, n_features=n_features)
    
    if not os.path.exists(weight_path):
        logger.error(f"找不到权重文件: {weight_path}")
        logger.warning("!!! 将回退到随机初始化 SimCLR，实验无效 !!!")
        model.to(device)
        return model
    
    try:
        # 3. 加载权重
        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
        
        # 处理可能的 key 差异
        # SimCLR 训练代码保存的通常是整个 model 的 state_dict
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # 尝试直接加载
        model.load_state_dict(state_dict, strict=False)
        logger.info("SimCLR 权重加载成功！")
        
    except Exception as e:
        logger.error(f"SimCLR 权重加载出错: {e}")
        
    model.to(device)
    model.eval()
    return model

# ============================
# 2. 图像处理函数
# ============================

def preprocess_image_base(image_path):
    if not os.path.exists(image_path):
        logger.warning(f"文件不存在: {image_path}")
        return None
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    new_w = w - (w % 16)
    new_h = h - (h % 16)
    if new_w != w or new_h != h:
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return img

def apply_blur(pil_img, sigma):
    if sigma == 0:
        return pil_img
    k_size = int(6 * sigma) + 1
    if k_size % 2 == 0: k_size += 1
    blur_transform = T.GaussianBlur(kernel_size=(k_size, k_size), sigma=sigma)
    return blur_transform(pil_img)

def img_to_wan_tensor(pil_img, device, dtype):
    img_np = np.array(pil_img).astype(np.float32) / 255.0
    img_np = (img_np * 2.0) - 1.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(2)
    return img_tensor.to(device=device, dtype=dtype)

def img_to_resnet_tensor(pil_img, device):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(pil_img).unsqueeze(0)
    return tensor.to(device=device)

# ============================
# 3. 特征提取逻辑
# ============================

def get_features(model, tensor, model_type="resnet"):
    with torch.no_grad():
        if model_type == "resnet":
            feat = model(tensor)
            return torch.flatten(feat, 1)
        elif model_type == "simclr":
            # SimCLR forward 返回 (h_i, h_j, z_i, z_j)
            # 我们取 h (representation)，即 encoder 的输出
            h, _, _, _ = model(tensor, tensor)
            return torch.flatten(h, 1)
        elif model_type == "vae":
            latent = model.encode(tensor).latent_dist.mode()
            return torch.flatten(latent, 1).float()

# ============================
# 4. 批量实验主循环
# ============================

def run_batch_experiment(args):
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载三个模型
    vae, vae_dtype = load_vae_model(args.model_id, args.vae_subfolder, args.dtype, device, args.fallback_model)
    resnet_sup = load_resnet_supervised(device)
    
    # 加载 SimCLR (替代 MoCo)
    resnet_simclr = load_simclr_local(device, args.simclr_path)
    
    cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
    
    img_dir = args.input_dir
    sigmas = [0] + list(range(1, 21, 2)) 
    
    all_sup_sims = []
    all_simclr_sims = []
    all_vae_sims = []
    
    valid_img_count = 0
    
    logger.info(f"开始处理目录 {img_dir} 下的图片...")
    
    for i in tqdm(range(1, 21), desc="Processing Images"):
        img_name = f"{i}.jpg"
        img_path = os.path.join(img_dir, img_name)
        
        base_img = preprocess_image_base(img_path)
        if base_img is None:
            continue
            
        valid_img_count += 1
        
        t_wan_0 = img_to_wan_tensor(base_img, device, vae_dtype)
        t_res_0 = img_to_resnet_tensor(base_img, device)
        
        f_vae_0 = get_features(vae, t_wan_0, "vae")
        f_sup_0 = get_features(resnet_sup, t_res_0, "resnet")
        f_simclr_0 = get_features(resnet_simclr, t_res_0, "simclr")
        
        curr_sup = []
        curr_simclr = []
        curr_vae = []
        
        for sigma in sigmas:
            blurred_img = apply_blur(base_img, sigma)
            
            # --- Supervised ---
            t_res = img_to_resnet_tensor(blurred_img, device)
            f_sup = get_features(resnet_sup, t_res, "resnet")
            curr_sup.append(cosine_sim(f_sup_0, f_sup).item())
            
            # --- SimCLR ---
            f_simclr = get_features(resnet_simclr, t_res, "simclr")
            curr_simclr.append(cosine_sim(f_simclr_0, f_simclr).item())
            
            # --- VAE ---
            t_wan = img_to_wan_tensor(blurred_img, device, vae_dtype)
            f_vae = get_features(vae, t_wan, "vae")
            curr_vae.append(cosine_sim(f_vae_0, f_vae).item())
        
        all_sup_sims.append(curr_sup)
        all_simclr_sims.append(curr_simclr)
        all_vae_sims.append(curr_vae)

    if valid_img_count == 0:
        logger.error("未找到任何有效图片。")
        return

    avg_sup = np.mean(np.array(all_sup_sims), axis=0)
    avg_simclr = np.mean(np.array(all_simclr_sims), axis=0)
    avg_vae = np.mean(np.array(all_vae_sims), axis=0)

    logger.info("计算完成，开始绘图...")

    plt.figure(figsize=(10, 6))
    
    # 1. Supervised ResNet (Blue)
    plt.plot(sigmas, avg_sup, marker='o', label='ResNet-50 (Supervised)', 
             linewidth=2.5, color='#1f77b4') 
    
    # 2. SimCLR (Green)
    plt.plot(sigmas, avg_simclr, marker='^', label='SimCLR (Contrastive)', 
         linewidth=2.5, color='#2ca02c') 

    # 3. VAE (Orange)
    plt.plot(sigmas, avg_vae, marker='s', label='Qwen-Image-VAE (Generative)', 
             linewidth=2.5, color='#ff7f0e')
    
    plt.title("Feature Similarity Degradation", fontsize=14)
    plt.xlabel("Gaussian Blur Sigma ($\sigma$)", fontsize=12)
    plt.ylabel("Avg Cosine Similarity with Sharp Image", fontsize=12)
    plt.ylim(0.2, 1.05)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='lower right')
    
    # 设置x轴刻度为整数
    plt.xticks(sigmas, [int(x) for x in sigmas])
    
    save_path = "diff_feature_blur.pdf"
    plt.savefig(save_path, dpi=300, format='pdf')
    logger.info(f"最终对比图表已保存至: {save_path}")
    plt.show()

# ============================
# 参数设置
# ============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    default_dir = "/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/project_image_stack_fusion/StackMFFV5/Visualization/clear_images"
    # 这里设置为你之前的 SimCLR 权重路径
    default_simclr_path = "/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/project_image_stack_fusion/StackMFFV5/Visualization/checkpoint_100.tar"
    
    parser.add_argument("--input_dir", type=str, default=default_dir)
    parser.add_argument("--simclr_path", type=str, default=default_simclr_path, help="SimCLR 权重的本地路径")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen-Image-2512")
    parser.add_argument("--fallback_model", type=str, default="Qwen/Qwen-Image")
    parser.add_argument("--vae_subfolder", type=str, default="vae")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default=None)
    
    args = parser.parse_args()
    
    run_batch_experiment(args)