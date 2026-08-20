#!/usr/bin/env python3
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset import Dataset
from utils.common import Edict
from utils.geometry import applyTransform
from utils.image import pixelToGrid
from module.network import OmniDS


def find_latest_ckpt(ckpt_dir):
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    ckpts.sort(key=os.path.getmtime, reverse=True)
    return ckpts[0]


def to_torch_imgs(imgs, device):
    torch_imgs = []
    for img in imgs:
        if isinstance(img, np.ndarray):
            t = torch.from_numpy(img).float()
        else:
            t = img.float()
        if t.ndim == 2:
            t = t.unsqueeze(0)
        t = t.unsqueeze(0).to(device)
        torch_imgs.append(t)
    return torch_imgs


def normalize_map(arr, eps=1e-6):
    vmin = arr.min()
    vmax = arr.max()
    return (arr - vmin) / (vmax - vmin + eps)


def save_weight_map(weight, out_path, title):
    plt.figure(figsize=(6, 3))
    plt.imshow(weight, cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_feature_map(feat_map, out_path, title):
    plt.figure(figsize=(6, 3))
    plt.imshow(normalize_map(feat_map), cmap="magma")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_rgb_image(img, out_path, title):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    img = np.clip(img, 0, 1)
    plt.figure(figsize=(6, 4))
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def compute_grids_for_depth(grids, depth_idx):
    """Interpolate LUT grids using per-pixel depth indices.

    Args:
        grids: list of 4 tensors [H, W, D, 2]
        depth_idx: [B, 1, H, W] float
    Returns:
        list of 4 tensors [B, H, W, 2]
    """
    B, _, H, W = depth_idx.shape
    grids_out = []
    for grid in grids:
        D = grid.shape[2]
        idx_flat = depth_idx.view(B, -1)
        idx_floor = idx_flat.long().clamp(0, D - 1)
        idx_ceil = (idx_floor + 1).clamp(0, D - 1)
        weight = (idx_flat - idx_floor.float()).unsqueeze(-1)

        grid_flat = grid.view(-1, D, 2).unsqueeze(0).expand(B, -1, -1, -1)
        idx_floor_exp = idx_floor.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
        idx_ceil_exp = idx_ceil.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
        ref_floor = torch.gather(grid_flat, 2, idx_floor_exp).squeeze(2)
        ref_ceil = torch.gather(grid_flat, 2, idx_ceil_exp).squeeze(2)
        ref_pts = ref_floor * (1 - weight) + ref_ceil * weight
        grids_out.append(ref_pts.view(B, H, W, 2))
    return grids_out


def build_erp_from_gt_nearest_cam(raw_rgb, data, gt_invdepth_full, device):
    """Build ERP RGB by selecting the camera closest to principal point per pixel."""
    h, w = data.equirect_size
    depth = 1.0 / (gt_invdepth_full + 1e-8)
    depth = depth.reshape(1, -1)
    rays = data.rays  # [3, H*W]
    pts = rays * depth  # [3, H*W]

    cam_dists = []
    cam_grids = []
    for cam_idx, ocam in enumerate(data.ocams):
        P = applyTransform(ocam.rig2cam, pts)  # [3, H*W]
        p, theta = ocam.rayToPixel(P, out_theta=True)
        # p: [2, H*W], p[0]=y, p[1]=x
        dx = p[1, :] - ocam.xc
        dy = p[0, :] - ocam.yc
        dist = np.sqrt(dx * dx + dy * dy)
        invalid = np.logical_or(theta.squeeze() > ocam.max_theta, np.isnan(dist))
        dist[invalid] = np.inf
        cam_dists.append(dist.reshape(h, w))
        grid = pixelToGrid(p, (h, w), (ocam.height, ocam.width))
        cam_grids.append(
            torch.tensor(grid, device=device, dtype=torch.float32).unsqueeze(0)
        )  # [1, H, W, 2]

    cam_dists = np.stack(cam_dists, axis=0)  # [4, H, W]
    cam_sel = np.argmin(cam_dists, axis=0)  # [H, W]

    # Sample ERP image from each camera
    samples = []
    for cam_idx, img in enumerate(raw_rgb):
        img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
        samp = F.grid_sample(img_t, cam_grids[cam_idx], mode="bilinear",
                             padding_mode="zeros", align_corners=True)
        samples.append(samp)
    samples = torch.cat(samples, dim=0)  # [4, 3, H, W]

    cam_sel_t = torch.from_numpy(cam_sel).long().to(device)
    out = samples.permute(2, 3, 0, 1)  # [H, W, 4, 3]
    out = out[torch.arange(h, device=device)[:, None],
              torch.arange(w, device=device)[None, :],
              cam_sel_t]
    out = out.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    return out.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()


def build_erp_rgb_for_frame(data, fidx, device):
    imgs, _, _, raw_imgs = data.loadSample(fidx, read_input_image=True)
    gt_idx_full = data.loadGTInvdepthIndex(fidx, remove_gt_noise=True)
    gt_invdepth_full = data.indexToInvdepth(gt_idx_full)
    raw_rgb = []
    for img in raw_imgs:
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        raw_rgb.append(img)
    pano_rgb = build_erp_from_gt_nearest_cam(raw_rgb, data, gt_invdepth_full, device)
    pano_rgb = pano_rgb.astype(np.float32)
    if pano_rgb.max() > 1.0:
        pano_rgb = pano_rgb / 255.0
    return pano_rgb


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="", help="checkpoint path")
    parser.add_argument("--ckpt_dir", type=str, default="/home/vdcl/RomniStereo/checkpoints/ROmniStereo_BEV")
    parser.add_argument("--dbname", type=str, default="sunny")
    parser.add_argument("--db_root", type=str, default="/home/vdcl/omnidata")
    parser.add_argument("--out_dir", type=str, default="/home/vdcl/RomniStereo/vis_view_weights")
    parser.add_argument("--sample_idx", type=int, default=0, help="index within dataset split")
    parser.add_argument("--depth_idx", type=int, default=-1,
                        help="depth index in LUT (0..D-1). -1 uses middle depth")
    parser.add_argument("--use_gt_depth", action="store_true",
                        help="use GT inverse depth index map for ERP projection")
    parser.add_argument("--save_all", action="store_true",
                        help="save ERP GT RGB for all frames and exit")
    parser.add_argument("--use_augment", action="store_true",
                        help="apply dataset augmentation when loading images")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.ckpt or find_latest_ckpt(args.ckpt_dir)
    print(f"[INFO] Using checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    net_opts = ckpt.get("net_opts", None)
    if net_opts is None:
        raise KeyError("Checkpoint does not contain net_opts")

    state_dict = ckpt.get("net_state_dict", ckpt)
    model = OmniDS(net_opts).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    if not hasattr(model, "similarity_context"):
        raise AttributeError("Model does not have similarity_context. Use v2/v3.")

    data_opts = Edict()
    data_opts.equirect_size = [net_opts.equi_h, net_opts.equi_w]
    data_opts.num_invdepth = net_opts.num_invdepth
    data_opts.num_downsample = net_opts.num_downsample
    data_opts.phi_deg = net_opts.phi_deg
    data_opts.use_rgb = net_opts.use_rgb
    data = Dataset(args.dbname, data_opts, db_root=args.db_root, train=False)
    if not args.use_augment:
        data.train_idx = []
        def _no_aug(x):
            return x
        data.augmentor = _no_aug

    if args.save_all:
        os.makedirs(args.out_dir, exist_ok=True)
        frame_list = data.frame_idx
        print(f"[INFO] Saving ERP GT RGB for {len(frame_list)} frames to {args.out_dir}")
        for i, fidx in enumerate(frame_list):
            pano_rgb = build_erp_rgb_for_frame(data, fidx, device)
            out_path = os.path.join(args.out_dir, f"{int(fidx):05d}.png")
            save_rgb_image(pano_rgb, out_path, f"ERP RGB (GT depth) {int(fidx):05d}")
            if (i + 1) % 50 == 0:
                print(f"[INFO] Saved {i + 1}/{len(frame_list)}")
        return

    imgs, _, _, raw_imgs = data.loadTestSample(args.sample_idx, read_input_image=True)
    torch_imgs = to_torch_imgs(imgs, device)

    with torch.no_grad():
        fisheye_feats = model.encoder(torch_imgs)
    fisheye_feats = [feat.float() for feat in fisheye_feats]

    grids = [torch.tensor(g, device=device, dtype=fisheye_feats[0].dtype) for g in data.grids]
    D = grids[0].shape[2]
    if args.use_gt_depth:
        if not data.test_idx:
            raise RuntimeError("Dataset has no test_idx; cannot resolve GT frame index.")
        fidx = data.test_idx[args.sample_idx]
        gt_idx_full = data.loadGTInvdepthIndex(fidx, remove_gt_noise=True)
        gt_idx = torch.from_numpy(gt_idx_full).float().unsqueeze(0).unsqueeze(0).to(device)
        invalid_mask = gt_idx < 0
        # Resize GT to match ERP feature size (grids spatial size)
        target_h, target_w = grids[0].shape[0], grids[0].shape[1]
        if gt_idx.shape[-2:] != (target_h, target_w):
            gt_idx = F.interpolate(gt_idx, size=(target_h, target_w), mode="nearest")
            invalid_mask = F.interpolate(
                invalid_mask.float(), size=(target_h, target_w), mode="nearest"
            ).bool()
        depth_idx_map = torch.where(invalid_mask, torch.zeros_like(gt_idx), gt_idx)
        print("[INFO] Using GT invdepth index map for ERP projection.")
    else:
        depth_idx = args.depth_idx if args.depth_idx >= 0 else D // 2
        depth_idx = max(0, min(D - 1, depth_idx))
        depth_idx_map = torch.full(
            (1, 1, grids[0].shape[0], grids[0].shape[1]),
            float(depth_idx),
            device=device,
            dtype=fisheye_feats[0].dtype,
        )
        print(f"[INFO] Using depth_idx: {depth_idx} (D={D})")

    per_cam_grids = compute_grids_for_depth(grids, depth_idx_map)
    cam_feats = []
    for cam_idx in range(4):
        grid = per_cam_grids[cam_idx]
        feat = F.grid_sample(
            fisheye_feats[cam_idx],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True
        )
        cam_feats.append(feat)

    # ERP correlation map from CNN features at selected depth
    sim_map = model.similarity_context.encoder.compute_similarity(cam_feats)  # [1, H, W]
    sim_np = sim_map[0].detach().cpu().numpy()

    weights = model.similarity_context.encoder._predict_view_weights(cam_feats)  # [1, 4, H, W]
    weights_np = weights[0].detach().cpu().numpy()

    os.makedirs(args.out_dir, exist_ok=True)
    for cam_idx in range(weights_np.shape[0]):
        save_weight_map(
            weights_np[cam_idx],
            os.path.join(args.out_dir, f"view_weight_cam{cam_idx}.png"),
            f"View Weight cam{cam_idx}"
        )

    save_weight_map(
        weights_np.mean(axis=0),
        os.path.join(args.out_dir, "view_weight_avg.png"),
        "View Weight avg"
    )

    save_weight_map(
        sim_np,
        os.path.join(args.out_dir, "erp_correlation_map.png"),
        "ERP Correlation map"
    )

    # GT depth/invdepth maps (only when GT is used for projection)
    if args.use_gt_depth:
        gt_idx_np = depth_idx_map[0, 0].detach().cpu().numpy()
        gt_invdepth = data.indexToInvdepth(gt_idx_np)
        gt_depth = 1.0 / (gt_invdepth + 1e-6)
        gt_invdepth = np.where(gt_idx_np <= 0, 0.0, gt_invdepth)
        gt_depth = np.where(gt_idx_np <= 0, 0.0, gt_depth)
        save_feature_map(
            gt_invdepth,
            os.path.join(args.out_dir, "gt_invdepth.png"),
            "GT InvDepth"
        )
        save_feature_map(
            gt_depth,
            os.path.join(args.out_dir, "gt_depth.png"),
            "GT Depth"
        )

    # Input fisheye images (normalized)
    for cam_idx, img in enumerate(raw_imgs):
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                img = img.astype(np.float32)
                img = img / (img.max() + 1e-6)
            elif img.ndim == 3:
                img = img.astype(np.float32)
                if img.max() > 1.0:
                    img = img / 255.0
        save_rgb_image(
            img,
            os.path.join(args.out_dir, f"fisheye_input_cam{cam_idx}.png"),
            f"Fisheye Input cam{cam_idx}"
        )

    # ERP RGB image using GT depth
    if args.use_gt_depth:
        gt_invdepth_full = data.indexToInvdepth(gt_idx_full)
        raw_rgb = []
        for img in raw_imgs:
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            raw_rgb.append(img)
        pano_rgb = build_erp_from_gt_nearest_cam(raw_rgb, data, gt_invdepth_full, device)
        pano_rgb = pano_rgb.astype(np.float32)
        if pano_rgb.max() > 1.0:
            pano_rgb = pano_rgb / 255.0
        save_rgb_image(
            pano_rgb,
            os.path.join(args.out_dir, "erp_rgb_gt.png"),
            "ERP RGB (GT depth)"
        )
        pano_gray = np.mean(pano_rgb, axis=-1)
        save_rgb_image(
            pano_gray,
            os.path.join(args.out_dir, "erp_gray_gt.png"),
            "ERP Gray (GT depth)"
        )

    # Fisheye CNN feature maps
    for cam_idx, feat in enumerate(fisheye_feats):
        feat_map = torch.norm(feat[0], dim=0).detach().cpu().numpy()
        save_feature_map(
            feat_map,
            os.path.join(args.out_dir, f"fisheye_cnn_cam{cam_idx}.png"),
            f"Fisheye CNN cam{cam_idx}"
        )

    # ERP CNN feature maps (per-cam + avg)
    erp_cnn_maps = []
    for cam_idx, feat in enumerate(cam_feats):
        feat_map = torch.norm(feat[0], dim=0).detach().cpu().numpy()
        erp_cnn_maps.append(feat_map)
        save_feature_map(
            feat_map,
            os.path.join(args.out_dir, f"erp_cnn_cam{cam_idx}.png"),
            f"ERP CNN cam{cam_idx}"
        )
    erp_cnn_avg = np.mean(np.stack(erp_cnn_maps, axis=0), axis=0)
    save_feature_map(
        erp_cnn_avg,
        os.path.join(args.out_dir, "erp_cnn_avg.png"),
        "ERP CNN avg"
    )

    # Fisheye DINO + ERP DINO (v3 only)
    if hasattr(model, "dino_extractor"):
        dino_feats_stacked = model.dino_extractor.extract_features(torch_imgs)  # [1, 4, C, H, W]
        for cam_idx in range(4):
            feat_map = torch.norm(dino_feats_stacked[0, cam_idx], dim=0).detach().cpu().numpy()
            save_feature_map(
                feat_map,
                os.path.join(args.out_dir, f"fisheye_dino_cam{cam_idx}.png"),
                f"Fisheye DINO cam{cam_idx}"
            )

        # ERP DINO per-cam via grid sampling at selected depth
        dino_cam_feats = []
        for cam_idx in range(4):
            grid = per_cam_grids[cam_idx]
            feat = F.grid_sample(
                dino_feats_stacked[:, cam_idx],
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True
            )
            dino_cam_feats.append(feat)
            feat_map = torch.norm(feat[0], dim=0).detach().cpu().numpy()
            save_feature_map(
                feat_map,
                os.path.join(args.out_dir, f"erp_dino_cam{cam_idx}.png"),
                f"ERP DINO cam{cam_idx}"
            )

        # ERP correlation map from DINO features (plain cosine, ref/tgt = (1+3)/(2+4))
        ref = (dino_cam_feats[0] + dino_cam_feats[2]) / 2.0
        tgt = (dino_cam_feats[1] + dino_cam_feats[3]) / 2.0
        ref = ref / (ref.norm(dim=1, keepdim=True) + 1e-6)
        tgt = tgt / (tgt.norm(dim=1, keepdim=True) + 1e-6)
        dino_sim_map = (ref * tgt).sum(dim=1)  # [1, H, W]
        dino_sim_np = dino_sim_map[0].detach().cpu().numpy()
        save_weight_map(
            dino_sim_np,
            os.path.join(args.out_dir, "erp_correlation_map_dino.png"),
            "ERP Correlation map (DINO)"
        )

        reference_points = model.compute_reference_points_from_grids(grids, depth_idx_map)
        dino_erp = model.dino_extractor(dino_feats_stacked, reference_points, depth_idx_map)
        dino_erp_map = torch.norm(dino_erp[0], dim=0).detach().cpu().numpy()
        save_feature_map(
            dino_erp_map,
            os.path.join(args.out_dir, "erp_dino.png"),
            "ERP DINO"
        )
    else:
        print("[WARN] Model has no dino_extractor. Skipping DINO visualizations.")

    print(f"[INFO] Saved weight maps to {args.out_dir}")


if __name__ == "__main__":
    main()
