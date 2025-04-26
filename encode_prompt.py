import argparse
import datetime
import json 
import itertools
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from einops import rearrange, repeat
from PIL import Image, ImageOps
from safetensors.torch import load_file
from torchvision.transforms import functional as F
from tqdm import tqdm 

import sampling
from modules.autoencoder import AutoEncoder
from modules.conditioner import Qwen25VL_7b_Embedder as Qwen2VLEmbedder
from modules.model_edit import Step1XParams, Step1XEdit
import gc


def input_process_image(img, img_size=512):
    # 1. 打开图片
    w, h = img.size
    r = w / h 

    if w > h:
        w_new = math.ceil(math.sqrt(img_size * img_size * r))
        h_new = math.ceil(w_new / r)
    else:
        h_new = math.ceil(math.sqrt(img_size * img_size / r))
        w_new = math.ceil(h_new * r)
    h_new = math.ceil(h_new) // 16 * 16
    w_new = math.ceil(w_new) // 16 * 16

    img_resized = img.resize((w_new, h_new))
    return img_resized, img.size

def encode_prompt(llm_encoder, prompt,negative_prompt,ref_images_raw, device,bs=1):
    txt, mask = llm_encoder([prompt,negative_prompt], ref_images_raw)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)
    llm_embedding = txt.to(device)
    txt_ids = txt_ids.to(device)
    
    return {
        "mask": mask,
        "llm_embedding": llm_embedding,
        "txt_ids": txt_ids,
    }
    

def load_image(image):
    from PIL import Image

    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image = image.unsqueeze(0)
        return image
    elif isinstance(image, Image.Image):
        image = F.to_tensor(image.convert("RGB"))
        image = image.unsqueeze(0)
        return image
    elif isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, str):
        image = F.to_tensor(Image.open(image).convert("RGB"))
        image = image.unsqueeze(0)
        return image
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    
if __name__ == "__main__":
    device = torch.device("cuda")
    qwen2vl_model_path = "F:/Step1X-Edit/models/qwenvl25_7b"
    ae_path = "F:/Step1X-Edit/models/vae.safetensors"
    dit_path = "F:/Step1X-Edit/models/step1x-edit-i1258.safetensors"
    image_path = "F:/ImageSet/ObjectRemoval/raw/1-Photoroom.png"
    cache_embedding_path = "cached_embedding.pt"
    max_length = 640
    dtype = torch.bfloat16
    
    if not os.path.exists(cache_embedding_path):
        qwen2vl_encoder = Qwen2VLEmbedder(
            qwen2vl_model_path,
            device=device,
            max_length=max_length,
            dtype=dtype,
        )
        ref_images = Image.open(image_path).convert("RGB")
        ref_images_raw, img_info = input_process_image(ref_images, img_size=512)
        
        width, height = ref_images_raw.width, ref_images_raw.height


        ref_images_raw = load_image(ref_images_raw)
        ref_images_raw = ref_images_raw.to(device)
        embedding = encode_prompt(qwen2vl_encoder, "remove the object from the image","",ref_images_raw, device,bs=1)
        
        # save embedding
        torch.save(embedding, cache_embedding_path)
    
    # load embedding
    embedding = torch.load(cache_embedding_path)
    
    llm_embedding = embedding['llm_embedding']
    print("llm_embedding.shape: ",llm_embedding.shape)