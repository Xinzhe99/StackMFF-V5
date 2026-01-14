import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random

def throw_one(probability: float) -> int:
    return 1 if random.random() < probability else 0


def image_resize(img, max_size=512):
    w, h = img.size
    if w >= h:
        new_w = max_size
        new_h = int((max_size / w) * h)
    else:
        new_h = max_size
        new_w = int((max_size / h) * w)
    return img.resize((new_w, new_h))

def c_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))

def crop_to_aspect_ratio(image, ratio="16:9"):
    width, height = image.size
    ratio_map = {
        "16:9": (16, 9),
        "4:3": (4, 3),
        "1:1": (1, 1)
    }
    target_w, target_h = ratio_map[ratio]
    target_ratio_value = target_w / target_h

    current_ratio = width / height

    if current_ratio > target_ratio_value:
        new_width = int(height * target_ratio_value)
        offset = (width - new_width) // 2
        crop_box = (offset, 0, offset + new_width, height)
    else:
        new_height = int(width / target_ratio_value)
        offset = (height - new_height) // 2
        crop_box = (0, offset, width, offset + new_height)

    cropped_img = image.crop(crop_box)
    return cropped_img


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, img_size=512, caption_type='txt',
                 random_ratio=False, caption_dropout_rate=0.1, cached_text_embeddings=None,
                 cached_image_embeddings=None, control_dir=None, cached_image_embeddings_control=None):
        self.images = [os.path.join(img_dir, i) for i in os.listdir(img_dir) if '.jpg' in i or '.png' in i]
        self.images.sort()
        # self.images = self.images[:5] # Limit to 5 images for testing
        self.img_size = img_size
        self.caption_type = caption_type
        self.random_ratio = random_ratio
        self.caption_dropout_rate = caption_dropout_rate
        self.control_dir = control_dir
        self.cached_text_embeddings = cached_text_embeddings
        self.cached_image_embeddings = cached_image_embeddings
        self.cached_control_image_embeddings = cached_image_embeddings_control
        print('cached_text_embeddings', type(cached_text_embeddings))
    def __len__(self):
        return 999999

    def __getitem__(self, idx):
        try:
            idx = random.randint(0, len(self.images) - 1)
            if self.cached_image_embeddings is None:
                img = Image.open(self.images[idx]).convert('RGB')
                if self.random_ratio:
                    ratio = random.choice(["16:9", "default", "1:1", "4:3"])
                    if ratio != "default":
                        img = crop_to_aspect_ratio(img, ratio)
                img = image_resize(img, self.img_size)
                w, h = img.size
                new_w = (w // 32) * 32
                new_h = (h // 32) * 32
                img = img.resize((new_w, new_h))
                img = torch.from_numpy((np.array(img) / 127.5) - 1)
                img = img.permute(2, 0, 1)
            else:
                img = self.cached_image_embeddings[os.path.basename(self.images[idx])]
            
            if self.cached_control_image_embeddings is None:
                # Load control image stack from folder
                target_img_name = os.path.basename(self.images[idx])
                base_name = target_img_name.rsplit('.', 1)[0]
                control_stack_dir = os.path.join(self.control_dir, base_name)
                
                stack_files = sorted([f for f in os.listdir(control_stack_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                
                stack_imgs = []
                for stack_file in stack_files:
                    c_img = Image.open(os.path.join(control_stack_dir, stack_file)).convert('RGB')
                    if self.random_ratio:
                         # Requires handling consistency across stack if applying random crop
                         # For now, assuming simply resize or handling consistency is out of scope unless specified
                         # But let's apply the same resize logic as target img
                         pass

                    c_img = image_resize(c_img, self.img_size)
                    w, h = c_img.size
                    new_w = (w // 32) * 32
                    new_h = (h // 32) * 32
                    c_img = c_img.resize((new_w, new_h))
                    c_img = torch.from_numpy((np.array(c_img) / 127.5) - 1)
                    c_img = c_img.permute(2, 0, 1) # (C, H, W)
                    stack_imgs.append(c_img)
                
                if stack_imgs:
                    control_img = torch.stack(stack_imgs) # (Stack, C, H, W)
                else:
                    # Fallback or error handling
                    print(f"Warning: No control images found in {control_stack_dir}")
                    control_img = torch.zeros_like(img).unsqueeze(0) # Dummy stack
            else:
                control_img = self.cached_control_image_embeddings[os.path.basename(self.images[idx])]
                
            txt_path = os.path.splitext(self.images[idx])[0] + '.' + self.caption_type
            if self.cached_text_embeddings is None:
                prompt = open(txt_path, encoding='utf-8').read()
                if throw_one(self.caption_dropout_rate):
                    return img, " ", control_img
                else:
                    return img, prompt, control_img
            else:
                txt = os.path.basename(txt_path)
                # Try multiple possible key formats to be robust
                try:
                     # Check if keys exist, if not, try to construct them properly or print debugging info
                    key_empty = txt + 'empty_embedding'
                    if key_empty not in self.cached_text_embeddings and txt in self.cached_text_embeddings:
                         # Fallback: maybe it was saved without the suffix in a different way or keys are mismatching
                         pass
                    
                    if throw_one(self.caption_dropout_rate):
                        return img, self.cached_text_embeddings[key_empty]['prompt_embeds'], self.cached_text_embeddings[key_empty]['prompt_embeds_mask'], control_img
                    else:
                        return img, self.cached_text_embeddings[txt]['prompt_embeds'], self.cached_text_embeddings[txt]['prompt_embeds_mask'], control_img
                except KeyError as e:
                     # Critical error debugging
                     print(f"KeyError for txt: '{txt}'. Available keys sample: {list(self.cached_text_embeddings.keys())[:5]}")
                     raise e 

        except Exception as e:
            # Prevent infinite recursion if all items fail
            # print(f"Error loading {self.images[idx]}: {e}") 
            # return self.__getitem__(random.randint(0, len(self.images) - 1))
             print(f"Critical Error loading {self.images[idx]}: {e}")
             raise e
        

def loader(train_batch_size, num_workers, **args):
    dataset = CustomImageDataset(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)
