#!/usr/bin/env python3
# inference.py

import os
from collections import OrderedDict

import torch
import numpy as np
from PIL import Image

from config import cfg
from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner
from models.merger import Merger

import utils.binvox_visualization as vis
import utils.data_transforms as T
import utils.network_utils as NU

def strip_module_prefix(state_dict):
    """
    Remove the 'module.' prefix from keys in a DataParallel checkpoint.
    """
    new_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_dict[name] = v
    return new_dict

def main():
    # 1) Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) Build models (no DataParallel here)
    enc = Encoder(cfg).to(device)
    dec = Decoder(cfg).to(device)
    ref = Refiner(cfg).to(device) if cfg.NETWORK.USE_REFINER else None
    mer = Merger(cfg).to(device) if cfg.NETWORK.USE_MERGER else None

    # 3) Load the Pix2Vox-A checkpoint
    ckpt_path = "Pix2Vox-A-ShapeNet.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # 4) Strip 'module.' and load each sub-model
    enc.load_state_dict( strip_module_prefix(ckpt["encoder_state_dict"]) )
    dec.load_state_dict( strip_module_prefix(ckpt["decoder_state_dict"]) )
    if ref:
        ref.load_state_dict( strip_module_prefix(ckpt["refiner_state_dict"]) )
    if mer:
        mer.load_state_dict( strip_module_prefix(ckpt["merger_state_dict"]) )

    # 5) Eval mode
    enc.eval();  dec.eval()
    if ref: ref.eval()
    if mer: mer.eval()

    # 6) Load & preprocess your image as a 1-view batch
    img = Image.open("images.jpeg").convert("RGB")
    img_np = np.array(img)[None, ...]  # [V=1, H, W, 3]

    IMG_SIZE  = (cfg.CONST.IMG_H, cfg.CONST.IMG_W)
    CROP_SIZE = (cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W)
    preprocess = T.Compose([
        T.CenterCrop(IMG_SIZE, CROP_SIZE),
        T.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        T.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        T.ToTensor(),                 # → [V, C, H, W]
    ])

    img_tensor = preprocess(img_np)    # [V=1, C, H, W]
    img_t      = img_tensor.unsqueeze(0)  # [B=1, V=1, C, H, W]
    img_t      = NU.var_or_cuda(img_t)    # to GPU if available

    # 7) Forward
    with torch.no_grad():
        feats = enc(img_t)
        raw, vol = dec(feats)

        # merger or average
        if mer and ckpt["epoch_idx"] >= cfg.TRAIN.EPOCH_START_USE_MERGER:
            vol = mer(raw, vol)
        else:
            vol = vol.mean(1, keepdim=True)

        # refiner
        if ref and ckpt["epoch_idx"] >= cfg.TRAIN.EPOCH_START_USE_REFINER:
            vol = ref(vol)

    # 8) Save outputs
    vox = vol.squeeze().cpu().numpy()  # [D, H, W]
    os.makedirs("output", exist_ok=True)
    np.save("output/volume.npy", vox)

    # dump orthographic views (positional 0, not keyword)
    vis.get_volume_views(vox, os.path.join("output", "views"), 0)

    print("Saved voxel grid to output/volume.npy and views to output/views/")

if __name__ == "__main__":
    main()
