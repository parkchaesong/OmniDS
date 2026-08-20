#!/usr/bin/env python3
import argparse
import os
import glob
import torch
import sys
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from dataset import Dataset
from utils.common import Edict, argparse as apply_opts
from module.network import OmniDS


def find_latest_ckpt(ckpt_dir):
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    ckpts.sort(key=os.path.getmtime, reverse=True)
    return ckpts[0]


def build_model_from_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    net_opts = ckpt.get("net_opts", None)
    if net_opts is None:
        raise KeyError("Checkpoint does not contain net_opts")
    model = OmniDS(net_opts).to(device)
    model.load_state_dict(ckpt["net_state_dict"], strict=False)
    model.eval()
    return model, net_opts


@torch.no_grad()
def compute_geom_bias(model, grids_tensor, inv_depth_idx, device):
    # Compute reference points (ERP -> fisheye)
    reference_points = model.compute_reference_points_from_grids(grids_tensor, inv_depth_idx)

    # Find geom_bias_mlp in DINO cross-attention (v3 path)
    cross_attn = model.dino_extractor.erp_cross_attn.layers[0].cross_attn
    if not hasattr(cross_attn, "geom_bias_mlp"):
        raise AttributeError("Geometric bias MLP not found. use_geom_bias=False?")

    # Always center reference points: [0,1] -> [-1,1]
    # Invalid points can be <0 due to lookup table clipping; these become |coord|>1,
    # then clamped to 1.0 for a neutral/max distance.
    ref_pts = reference_points * 2.0 - 1.0
    dist = torch.sqrt(ref_pts[..., 0] ** 2 + ref_pts[..., 1] ** 2).clamp(0.0, 1.0)
    bias = cross_attn.geom_bias_mlp(dist.unsqueeze(-1))  # [B, Lq, num_cams, 1]
    return bias.squeeze(-1)


def save_bias_images(bias, H, W, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    # bias: [B, Lq, num_cams]
    bias = bias[0].cpu().numpy()  # [Lq, num_cams]
    for cam_idx in range(bias.shape[1]):
        b = bias[:, cam_idx].reshape(H, W)
        plt.figure(figsize=(6, 3))
        plt.imshow(b, cmap="viridis")
        plt.colorbar()
        plt.title(f"Geom Bias - cam {cam_idx}")
        out_path = os.path.join(out_dir, f"{prefix}_cam{cam_idx}.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    # also save average bias across cams
    b_avg = bias.mean(axis=1).reshape(H, W)
    plt.figure(figsize=(6, 3))
    plt.imshow(b_avg, cmap="viridis")
    plt.colorbar()
    plt.title("Geom Bias - avg cams")
    out_path = os.path.join(out_dir, f"{prefix}_avg.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="", help="checkpoint path")
    parser.add_argument("--ckpt_dir", type=str, default="/home/vdcl/RomniStereo/checkpoints/ROmniStereo_BEV")
    parser.add_argument("--dbname", type=str, default="sunny")
    parser.add_argument("--db_root", type=str, default="/home/vdcl/omnidata")
    parser.add_argument("--out_dir", type=str, default="/home/vdcl/RomniStereo/vis_bias")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.ckpt or find_latest_ckpt(args.ckpt_dir)
    print(f"[INFO] Using checkpoint: {ckpt_path}")

    model, net_opts = build_model_from_ckpt(ckpt_path, device)

    # build dataset to get grids
    data_opts = Edict()
    data_opts.equirect_size = [net_opts.equi_h, net_opts.equi_w]
    data_opts.num_invdepth = net_opts.num_invdepth
    data_opts.num_downsample = net_opts.num_downsample
    data_opts.phi_deg = net_opts.phi_deg
    data_opts.use_rgb = net_opts.use_rgb
    data = Dataset(args.dbname, data_opts, db_root=args.db_root, train=False)
    grids_tensor = [torch.tensor(g, device=device) for g in data.grids]

    # create a mid-depth index map
    H = net_opts.equi_h // (2 ** net_opts.num_downsample)
    W = net_opts.equi_w // (2 ** net_opts.num_downsample)
    inv_depth_idx = torch.full((1, 1, H, W), net_opts.num_invdepth / 2, device=device)

    bias = compute_geom_bias(model, grids_tensor, inv_depth_idx, device)
    save_bias_images(bias, H, W, args.out_dir, "geom_bias")
    print(f"[INFO] Saved bias images to {args.out_dir}")


if __name__ == "__main__":
    main()
