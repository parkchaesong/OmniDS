#!/usr/bin/env python3
"""
Visualize similarity_profile used in OmniDS.

This script:
1) Loads a BEV checkpoint (handles common key formats and DataParallel prefixes)
2) Loads one sample from Dataset (grids/LUT + 4 fisheye images)
3) Recomputes similarity_profile exactly like v3 does:
   cnn_feats -> spherical_sweep -> SimilarityContext.build_profile(...)
4) Saves:
   - mean/max maps
   - argmax depth-sample index map
   - a montage of depth slices
   - depth-profile plots for a few pixels
   - raw .npy arrays

Example:
  python tools/visualize_similarity_profile.py \
    --restore_ckpt /home/vdcl/RomniStereo/checkpoints/ROmniStereo_BEV/ROmniStereo_BEV_e60.pth \
    --db_root ../omnidata --dbname sunny --frame_idx 1
"""

from __future__ import annotations

import os
import os.path as osp
import sys
from argparse import ArgumentParser
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import matplotlib

# Headless-safe
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Ensure repo root is on sys.path so `import dataset` works even when launched from elsewhere.
_THIS_DIR = osp.dirname(osp.abspath(__file__))
_REPO_ROOT = osp.abspath(osp.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _strip_prefix_from_state_dict(state_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    if not state_dict:
        return state_dict
    if not all(isinstance(k, str) for k in state_dict.keys()):
        return state_dict
    if not all(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {k[len(prefix):]: v for k, v in state_dict.items()}


def _load_checkpoint(path: str) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Returns (state_dict, net_opts)
    """
    import torch

    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        # Common formats in this repo: {'net_opts': ..., 'net_state_dict': ...}
        net_opts = ckpt.get("net_opts", ckpt.get("opts", None))
        state_dict = ckpt.get("net_state_dict", ckpt.get("state_dict", None))
        if state_dict is None and all(isinstance(k, str) for k in ckpt.keys()):
            # Might be a raw state_dict saved directly
            state_dict = ckpt
        if not isinstance(state_dict, dict):
            raise ValueError(f"Unrecognized checkpoint format in {path}.")
        return state_dict, net_opts

    raise ValueError(f"Unrecognized checkpoint type: {type(ckpt)}")


def _auto_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _robust_minmax(x: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> Tuple[float, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(x, p_lo))
    hi = float(np.percentile(x, p_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(x))
        hi = float(np.nanmax(x))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def _save_heatmap(path: str, x2d: np.ndarray, title: str, cmap: str = "viridis") -> None:
    lo, hi = _robust_minmax(x2d)
    plt.figure(figsize=(12, 4), dpi=150)
    plt.imshow(x2d, cmap=cmap, vmin=lo, vmax=hi)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _save_vertical_strip(path: str, slices: List[np.ndarray], cmap: str = "gray", gap_inches: float = 0.2, dpi: int = 150) -> None:
    """Save images stacked vertically with no titles/labels and minimal gap."""
    n = len(slices)
    if n == 0:
        return
    # Infer per-image display size from first slice (preserve aspect ratio)
    s0 = slices[0]
    h_px, w_px = s0.shape[:2]
    img_w_in = 12.0
    img_h_in = img_w_in * h_px / max(1, w_px)
    fig_h = img_h_in * n + gap_inches * (n - 1)
    fig, axes = plt.subplots(n, 1, figsize=(img_w_in, fig_h), dpi=dpi)
    if n == 1:
        axes = [axes]
    for ax, sl in zip(axes, slices):
        if sl.ndim == 2:
            ax.imshow(sl, cmap=cmap, vmin=float(sl.min()), vmax=float(sl.max()))
        else:
            ax.imshow(np.clip(sl, 0.0, 1.0))
        ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=gap_inches / img_h_in)
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _save_montage(path: str, slices: List[np.ndarray], titles: List[str], ncols: int = 4, cmap: str = "viridis") -> None:
    assert len(slices) == len(titles)
    n = len(slices)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / ncols))
    plt.figure(figsize=(3.2 * ncols, 2.8 * nrows), dpi=150)
    for i, (sl, ti) in enumerate(zip(slices, titles), start=1):
        ax = plt.subplot(nrows, ncols, i)
        if sl.ndim == 2:
            lo, hi = _robust_minmax(sl)
            ax.imshow(sl, cmap=cmap, vmin=lo, vmax=hi)
        else:
            # RGB image (e.g., PCA projection) expected in [0,1] or [0,255]
            ax.imshow(sl)
        ax.set_title(ti, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _overlay_points(ax, xs: List[int], ys: List[int], labels: List[str], color_cycle=None):
    if color_cycle is None:
        color_cycle = ["#ff3b30", "#34c759", "#007aff", "#ffcc00"]
    for i, (x, y, lab) in enumerate(zip(xs, ys, labels)):
        c = color_cycle[i % len(color_cycle)]
        ax.scatter([x], [y], s=55, c=c, marker="x", linewidths=2.0)
        ax.text(
            x + 6,
            y + 6,
            lab,
            color=c,
            fontsize=9,
            bbox=dict(facecolor="black", alpha=0.35, pad=2, edgecolor="none"),
        )


def _interp_profile_y(prof_1d: np.ndarray, x: float) -> float:
    """
    Linear interpolation on a 1D profile array.
    Args:
        prof_1d: [Ds]
        x: float in [0, Ds-1]
    Returns:
        y at x (float)
    """
    if not np.isfinite(x):
        return float("nan")
    Ds = int(prof_1d.shape[0])
    if Ds <= 0:
        return float("nan")
    x = float(np.clip(x, 0.0, float(Ds - 1)))
    x0 = int(np.floor(x))
    x1 = min(Ds - 1, x0 + 1)
    w = x - float(x0)
    y0 = float(prof_1d[x0])
    y1 = float(prof_1d[x1])
    return (1.0 - w) * y0 + w * y1


def _compute_depth_sample_indices(num_invdepth: int, num_depth_samples: int, d_total: int) -> np.ndarray:
    """
    Mirrors SimilarityProfileEncoder's sparse sampling logic.
    Returns depth indices in [0, d_total-1] with length num_depth_samples.
    """
    depth_step = max(1, int(num_invdepth) // int(num_depth_samples))
    depth_indices = list(range(0, int(num_invdepth), depth_step))[: int(num_depth_samples)]
    while len(depth_indices) < int(num_depth_samples):
        depth_indices.append(depth_indices[-1])
    depth_idx = np.array(depth_indices, dtype=np.int64)
    depth_idx = np.clip(depth_idx, 0, int(d_total) - 1)
    return depth_idx


def _feat_to_vis_map(feat_chw: np.ndarray, mode: str = "l2") -> np.ndarray:
    """
    Convert a [C,H,W] feature map into a [H,W] visualization map.
    """
    if mode == "l2":
        return np.sqrt(np.maximum(0.0, (feat_chw ** 2).sum(axis=0)))
    if mode == "mean_abs":
        return np.abs(feat_chw).mean(axis=0)
    if mode == "mean":
        return feat_chw.mean(axis=0)
    raise ValueError(f"Unknown feat vis mode: {mode}")


def _normalize_to_01(x: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    """Robust normalize to [0,1] with percentile clipping."""
    x = x.astype(np.float32, copy=False)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    lo = float(np.percentile(finite, p_lo))
    hi = float(np.percentile(finite, p_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
    if hi <= lo:
        hi = lo + 1e-6
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def _pca_rgb_from_chw(
    feat_chw: np.ndarray,
    valid_mask_hw: Optional[np.ndarray] = None,
    p_lo: float = 1.0,
    p_hi: float = 99.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    PCA projection of a [C,H,W] feature map to an RGB image [H,W,3] in [0,1].
    Inspired by DINOv3 PCA notebook visualization:
    https://colab.research.google.com/github/facebookresearch/dinov3/blob/main/notebooks/pca.ipynb
    """
    feat = feat_chw.astype(np.float32, copy=False)
    C, H, W = feat.shape
    X = feat.reshape(C, -1).T  # [N, C]

    if valid_mask_hw is None:
        # Heuristic: ignore all-zero padded regions from grid_sample.
        valid_mask_hw = (np.abs(feat).sum(axis=0) > eps)
    m = valid_mask_hw.reshape(-1).astype(bool)
    if int(m.sum()) < max(10, C):
        return np.zeros((H, W, 3), dtype=np.float32)

    Xv = X[m]
    mu = Xv.mean(axis=0, keepdims=True)
    Xc = Xv - mu

    # Efficient PCA via CxC covariance (C is small, e.g. 32)
    cov = (Xc.T @ Xc) / max(1.0, float(Xc.shape[0] - 1))
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
    order = np.argsort(eigvals)[::-1]
    comps = eigvecs[:, order[:3]]  # [C,3]

    Yv = Xc @ comps  # [Nv,3]
    Y = np.zeros((X.shape[0], 3), dtype=np.float32)
    Y[m] = Yv.astype(np.float32)
    Y = Y.reshape(H, W, 3)

    out = np.zeros_like(Y, dtype=np.float32)
    for k in range(3):
        out[..., k] = _normalize_to_01(Y[..., k], p_lo=p_lo, p_hi=p_hi)
    return out


def _pca_rgb_vivid(feat_chw: np.ndarray, valid_mask_hw=None, sat_boost: float = 1.5) -> np.ndarray:
    """PCA with tighter percentile + saturation boost for vivid colors."""
    from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
    rgb = _pca_rgb_from_chw(feat_chw, valid_mask_hw=valid_mask_hw, p_lo=5.0, p_hi=95.0)
    hsv = rgb_to_hsv(rgb)
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_boost, 0.0, 1.0)
    return hsv_to_rgb(hsv).astype(np.float32)


def _pca_turbo(feat_chw: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """First PC → turbo colormap (blue→red)."""
    feat = feat_chw.astype(np.float32, copy=False)
    C, H, W = feat.shape
    X = feat.reshape(C, -1).T  # [N, C]
    valid = (np.abs(feat).sum(axis=0) > eps).reshape(-1)
    if int(valid.sum()) < max(10, C):
        return np.zeros((H, W, 3), dtype=np.float32)
    Xv = X[valid]
    mu = Xv.mean(axis=0, keepdims=True)
    Xc = Xv - mu
    cov = (Xc.T @ Xc) / max(1.0, float(Xc.shape[0] - 1))
    eigvals, eigvecs = np.linalg.eigh(cov)
    pc1 = eigvecs[:, np.argmax(eigvals)]
    proj = ((X - mu) @ pc1).reshape(H, W)
    pv = proj[valid.reshape(H, W)]
    lo, hi = float(np.percentile(pv, 2)), float(np.percentile(pv, 98))
    if hi <= lo:
        hi = lo + 1e-6
    normed = np.clip((proj - lo) / (hi - lo), 0.0, 1.0)
    normed[~valid.reshape(H, W)] = 0.0
    return plt.get_cmap("turbo")(normed)[..., :3].astype(np.float32)


def _feat_to_vis(feat_chw: np.ndarray, mode: str) -> np.ndarray:
    """
    Convert a [C,H,W] feature map to:
    - [H,W] if mode in {l2, mean_abs, mean}
    - [H,W,3] if mode == 'pca'
    """
    if mode == "pca":
        return _pca_rgb_from_chw(feat_chw)
    return _feat_to_vis_map(feat_chw, mode=mode)


def _downsample_idx_map_full_to_lut(gt_idx_full: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """
    Downsample a full-res GT invdepth index map to LUT/profile resolution.
    Handles invalid (<0) by mask-weighted interpolation.

    Args:
        gt_idx_full: [H_full, W_full] float/np
        target_hw: (H, W)
    Returns:
        gt_idx_ds: [H, W] float32, invalid stays <0 (set to -1)
    """
    import cv2

    Ht, Wt = int(target_hw[0]), int(target_hw[1])
    src = gt_idx_full.astype(np.float32)
    valid = (src >= 0).astype(np.float32)
    src0 = np.where(valid > 0, src, 0.0).astype(np.float32)
    # bilinear downsample both value and mask
    val_ds = cv2.resize(src0, (Wt, Ht), interpolation=cv2.INTER_LINEAR)
    m_ds = cv2.resize(valid, (Wt, Ht), interpolation=cv2.INTER_LINEAR)
    out = np.where(m_ds > 1e-3, val_ds / (m_ds + 1e-6), -1.0).astype(np.float32)
    return out


def _sample_cam_erp_at_idx(
    cam_feat: "np.ndarray",
    grid_lut: "np.ndarray",
    idx_map: "np.ndarray",
) -> "np.ndarray":
    """
    Sample a single camera fisheye feature map to ERP using a per-pixel (fractional) depth index.

    Args:
        cam_feat: torch Tensor [B, C, Hf, Wf]
        grid_lut: torch Tensor [H, W, D, 2] (grid_sample coords -1..1)
        idx_map:  torch Tensor [B, 1, H, W] (fractional depth index in LUT depth axis)
    Returns:
        erp_feat: torch Tensor [B, C, H, W]
    """
    import torch
    import torch.nn.functional as F

    B, C, Hf, Wf = cam_feat.shape
    H, W, D, _ = grid_lut.shape
    idx = idx_map.view(B, -1)  # [B, H*W]
    idx_floor = idx.long().clamp(0, D - 1)
    idx_ceil = (idx_floor + 1).clamp(0, D - 1)
    w = (idx - idx_floor.float()).unsqueeze(-1)  # [B, H*W, 1]

    grid_flat = grid_lut.view(-1, D, 2).unsqueeze(0).expand(B, -1, -1, -1)  # [B,H*W,D,2]
    idx_floor_exp = idx_floor.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
    idx_ceil_exp = idx_ceil.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
    g0 = torch.gather(grid_flat, 2, idx_floor_exp).squeeze(2)  # [B,H*W,2]
    g1 = torch.gather(grid_flat, 2, idx_ceil_exp).squeeze(2)
    grid = ((1.0 - w) * g0 + w * g1).view(B, H, W, 2)

    feat = F.grid_sample(
        cam_feat,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )  # [B,C,H,W]
    return feat


def main() -> None:
    parser = ArgumentParser("Visualize similarity_profile for OmniDS")
    parser.add_argument(
        "--restore_ckpt",
        default="/home/vdcl/RomniStereo/checkpoints/ROmniStereo_BEV/ROmniStereo_BEV_e60.pth",
        type=str,
        help="checkpoint path (.pth)",
    )
    parser.add_argument("--db_root", default="../omnidata", type=str, help="dataset root")
    parser.add_argument(
        "--dbname",
        default="itbt_sample",
        type=str,
        help="dataset name (see dataset.py / utils/dbhelper configs)",
    )
    parser.add_argument(
        "--frame_idx",
        default=None,
        type=int,
        help="frame index to visualize (if omitted, uses first test_idx or first frame_idx)",
    )
    parser.add_argument("--out_dir", default="./vis_similarity_profile", type=str, help="output directory")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="device")
    parser.add_argument("--iters", default=12, type=int, help="number of refinement iterations for model prediction")
    parser.add_argument("--num_slices", default=16, type=int, help="how many depth slices to dump in montage")
    parser.add_argument(
        "--num_cam_slices",
        default=8,
        type=int,
        help="how many depth slices to visualize for per-camera ERP feature volumes (cam0..3)",
    )
    parser.add_argument(
        "--cam_feat_vis",
        default="l2",
        choices=["l2", "mean_abs", "mean", "pca"],
        help="how to convert C-channel cam ERP features into a 2D map (or RGB if pca)",
    )
    parser.add_argument(
        "--num_pair_slices",
        default=8,
        type=int,
        help="how many depth slices to visualize for ref/tgt pair maps (correlation mode)",
    )
    parser.add_argument(
        "--pair_feat_vis",
        default="l2",
        choices=["l2", "mean_abs", "mean", "pca"],
        help="how to convert C-channel ref/tgt features into a 2D map (or RGB if pca)",
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--query_points",
        default=None,
        type=str,
        help="Comma-separated y,x pairs in full ERP coords for single-pixel similarity profile. "
             "e.g. '160,320' for one point or '160,320;80,400' for multiple.",
    )
    args = parser.parse_args()

    device = _auto_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    import torch
    from easydict import EasyDict as Edict

    from dataset import Dataset
    from module.network import OmniDS
    from module.sweep import spherical_sweep

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ----------------------------
    # Load checkpoint + model
    # ----------------------------
    if not osp.exists(args.restore_ckpt):
        raise FileNotFoundError(args.restore_ckpt)

    state_dict, net_opts = _load_checkpoint(args.restore_ckpt)

    # Instantiate model (net_opts may be EasyDict-like or plain dict)
    model = OmniDS(net_opts)
    model = model.to(device)

    # Handle DataParallel prefix mismatches robustly
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception:
        # try stripping 'module.'
        state2 = _strip_prefix_from_state_dict(state_dict, "module.")
        missing, unexpected = model.load_state_dict(state2, strict=False)
        print("[WARN] Loaded with strict=False")
        print("  missing:", missing[:20], "..." if len(missing) > 20 else "")
        print("  unexpected:", unexpected[:20], "..." if len(unexpected) > 20 else "")

    model.eval()

    # ----------------------------
    # Build dataset + get one sample
    # ----------------------------
    data_opts = Edict()
    data_opts.color_aug = False
    data_opts.phi_deg = float(getattr(model.opts, "phi_deg", 45.0))
    data_opts.equirect_size = [int(getattr(model.opts, "equi_h", 160)), int(getattr(model.opts, "equi_w", 640))]
    data_opts.use_rgb = bool(getattr(model.opts, "use_rgb", False))
    data_opts.num_invdepth = int(getattr(model.opts, "num_invdepth", 192))
    data_opts.num_downsample = int(getattr(model.opts, "num_downsample", 1))

    data = Dataset(args.dbname, data_opts, db_root=args.db_root, train=False)
    grids = [torch.tensor(g, requires_grad=False, device=device) for g in data.grids]

    if args.frame_idx is not None:
        fidx = int(args.frame_idx)
    else:
        if hasattr(data, "test_idx") and len(data.test_idx) > 0:
            fidx = int(data.test_idx[0])
        else:
            fidx = int(data.frame_idx[0])

    imgs, gt, valid, raw_imgs = data.loadSample(fidx)
    imgs_t = [torch.tensor(im, dtype=torch.float32, device=device).unsqueeze(0) for im in imgs]  # 4 x [1,C,H,W]

    # ----------------------------
    # Compute similarity_profile (v3-style)
    # ----------------------------
    with torch.no_grad():
        cnn_feats = model.encoder(imgs_t)  # list of 4 [B,C,Hf,Wf]
        cnn_feats = [f.float() for f in cnn_feats]
        cnn_feats_stacked = torch.stack(cnn_feats, dim=1)  # [B,4,C,Hf,Wf]

        # Build spherical volumes once and reuse (matches v3 forward)
        sph_feats = spherical_sweep(cnn_feats, grids)
        similarity_profile = model.similarity_context.build_profile(grids=grids, spherical_volumes=sph_feats)
        # Try to also get final prediction (invdepth index) from the model.
        pred_invdepth_idx = None
        pred_idx_map_full = None
        try:
            pred_invdepth_idx = model(imgs_t, grids, iters=int(args.iters), test_mode=True)
            try:
                pred_idx_map_full = pred_invdepth_idx[0, 0].detach().cpu().numpy().astype(np.float32)
            except Exception:
                pred_idx_map_full = None
        except Exception as e:
            print(f"[WARN] Failed to run model forward for prediction: {e}")

    prof = similarity_profile[0].detach().cpu().numpy()  # [Ds,H,W]
    Ds, H, W = prof.shape
    print(f"[INFO] similarity_profile shape: {prof.shape} (Ds,H,W)")
    if pred_invdepth_idx is not None:
        try:
            print(f"[INFO] pred_invdepth_idx shape: {tuple(pred_invdepth_idx.shape)}")
        except Exception:
            pass

    # Save raw arrays
    np.save(osp.join(args.out_dir, f"similarity_profile_f{fidx:05d}.npy"), prof)

    # Summaries
    prof_mean = prof.mean(axis=0)
    prof_max = prof.max(axis=0)
    prof_argmax = prof.argmax(axis=0).astype(np.float32) / max(1.0, float(Ds - 1))

    _save_heatmap(
        osp.join(args.out_dir, f"simprof_mean_f{fidx:05d}.png"),
        prof_mean,
        title=f"Similarity Profile (mean over Ds={Ds}) - fidx={fidx}",
        cmap="viridis",
    )
    _save_heatmap(
        osp.join(args.out_dir, f"simprof_max_f{fidx:05d}.png"),
        prof_max,
        title=f"Similarity Profile (max over Ds={Ds}) - fidx={fidx}",
        cmap="viridis",
    )
    _save_heatmap(
        osp.join(args.out_dir, f"simprof_argmax_f{fidx:05d}.png"),
        prof_argmax,
        title=f"Similarity Profile argmax depth-sample idx (normalized 0..1) - fidx={fidx}",
        cmap="turbo",
    )

    # ------------------------------------------------------------------
    # Save raw fisheye camera images (cam0..3)
    # ------------------------------------------------------------------
    try:
        cam_raw_slices = []
        cam_raw_titles = []
        for cam_i, raw_img in enumerate(raw_imgs):
            img = np.array(raw_img)
            # Normalize uint8 [0,255] → [0,1] float for imshow; handle grayscale/RGB
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            else:
                img = img.astype(np.float32, copy=False)
                lo, hi = float(img.min()), float(img.max())
                if hi > lo:
                    img = (img - lo) / (hi - lo)
            # Save individual cam image
            plt.figure(figsize=(5, 5), dpi=150)
            if img.ndim == 2:
                plt.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            else:
                plt.imshow(np.clip(img, 0.0, 1.0))
            plt.title(f"cam{cam_i} raw - fidx={fidx}")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(osp.join(args.out_dir, f"cam{cam_i}_raw_f{fidx:05d}.png"))
            plt.close()
            cam_raw_slices.append(img)
            cam_raw_titles.append(f"cam{cam_i} raw")
        # Also save as a 4-up montage
        _save_montage(
            osp.join(args.out_dir, f"cam0to3_raw_f{fidx:05d}.png"),
            slices=cam_raw_slices,
            titles=cam_raw_titles,
            ncols=4,
            cmap="gray",
        )
    except Exception as e:
        print(f"[WARN] Failed to save raw cam images: {e}")

    # Depth slice montage
    n_slices = int(max(1, min(args.num_slices, Ds)))
    idxs = np.linspace(0, Ds - 1, n_slices).round().astype(np.int64)
    idxs = np.unique(idxs)
    slices = [prof[i] for i in idxs.tolist()]
    titles = [f"Ds[{int(i)}]" for i in idxs.tolist()]
    _save_montage(
        osp.join(args.out_dir, f"simprof_slices_f{fidx:05d}.png"),
        slices=slices,
        titles=titles,
        ncols=4,
        cmap="viridis",
    )

    # ------------------------------------------------------------------
    # Visualize per-camera CNN features projected into ERP used for similarity_profile.
    # We reuse the spherical volumes from spherical_sweep (cam0..3): [B,C,H,W,D_total]
    # and select the same sparse depth indices used by SimilarityProfileEncoder.
    # ------------------------------------------------------------------
    try:
        num_invdepth = int(getattr(model.opts, "num_invdepth", 192))
        num_depth_samples = int(getattr(model.opts, "num_depth_samples", Ds))
    except Exception:
        num_invdepth = 192
        num_depth_samples = Ds
    d_total = int(grids[0].shape[2])
    depth_idx = _compute_depth_sample_indices(num_invdepth, num_depth_samples, d_total)  # [Ds]

    try:
        cam_vols = [
            sph_feats[cam_i].index_select(dim=4, index=torch.tensor(depth_idx, device=sph_feats[cam_i].device))
            for cam_i in range(4)
        ]  # each [B,C,H,W,Ds]
        cam_np = [v[0].detach().cpu().numpy() for v in cam_vols]  # each [C,H,W,Ds]

        n_cam = int(max(1, min(args.num_cam_slices, Ds)))
        cam_idxs = np.linspace(0, Ds - 1, n_cam).round().astype(np.int64)
        cam_idxs = np.unique(cam_idxs)

        cam_slices = []
        cam_titles = []
        for di in cam_idxs.tolist():
            lut_di = int(depth_idx[int(di)])
            for cam_i in range(4):
                m = _feat_to_vis(cam_np[cam_i][:, :, :, int(di)], mode=args.cam_feat_vis)
                cam_slices.append(m)
                cam_titles.append(f"cam{cam_i} Ds[{di}] (lut={lut_di})")

        _save_montage(
            osp.join(args.out_dir, f"simprof_cam0to3_erp_feats_f{fidx:05d}.png"),
            slices=cam_slices,
            titles=cam_titles,
            ncols=4,
            cmap="viridis",
        )

        # Also save per-cam average strength over Ds (average of per-slice 2D maps)
        # If PCA mode is selected (RGB), fall back to l2 for this "mean strength" summary.
        cam_mean_maps = []
        cam_mean_titles = []
        mean_mode = args.cam_feat_vis if args.cam_feat_vis != "pca" else "l2"
        for cam_i in range(4):
            # build [Ds,H,W] of 2D maps then average
            maps = np.stack([_feat_to_vis_map(cam_np[cam_i][:, :, :, d], mode=mean_mode) for d in range(Ds)], axis=0)
            cam_mean_maps.append(maps.mean(axis=0))
            cam_mean_titles.append(f"cam{cam_i} mean over Ds ({mean_mode})")

        _save_montage(
            osp.join(args.out_dir, f"simprof_cam0to3_erp_feats_mean_f{fidx:05d}.png"),
            slices=cam_mean_maps,
            titles=cam_mean_titles,
            ncols=4,
            cmap="viridis",
        )
    except Exception as e:
        print(f"[WARN] Failed to render per-camera ERP feature volumes: {e}")

    # ------------------------------------------------------------------
    # Visualize per-camera CNN->ERP features at *GT depth* (single slice per pixel).
    # This answers: "if we project each cam using the GT depth, what does the ERP feature look like?"
    # ------------------------------------------------------------------
    gt_idx_map = None
    if isinstance(gt, np.ndarray) and gt.ndim == 2 and gt.size > 0:
        gt_idx_map = gt
    elif hasattr(gt, "shape"):
        try:
            gt_idx_map = np.array(gt)
            if gt_idx_map.ndim != 2:
                gt_idx_map = None
        except Exception:
            gt_idx_map = None

    if gt_idx_map is not None:
        try:
            # LUT/profile resolution for grids is (H,W)
            gt_idx_ds = _downsample_idx_map_full_to_lut(gt_idx_map, (H, W))  # [H,W]
            # Convert to tensor [B,1,H,W]
            import torch

            gt_idx_ds_t = torch.from_numpy(gt_idx_ds).to(device=device, dtype=cnn_feats[0].dtype).unsqueeze(0).unsqueeze(0)

            # Sample each cam to ERP at GT index
            cam_gt_erp = []
            for cam_i in range(4):
                feat = _sample_cam_erp_at_idx(
                    cam_feat=cnn_feats[cam_i],
                    grid_lut=grids[cam_i].to(device=device, dtype=cnn_feats[cam_i].dtype),
                    idx_map=gt_idx_ds_t,
                )  # [B,C,H,W]
                cam_gt_erp.append(feat[0].detach().cpu().numpy())  # [C,H,W]

            # Make montage cam0..3
            gt_cam_maps = []
            gt_cam_titles = []
            for cam_i in range(4):
                m = _feat_to_vis(cam_gt_erp[cam_i], mode=args.cam_feat_vis)
                gt_cam_maps.append(m)
                gt_cam_titles.append(f"cam{cam_i} @ GT idx (proj)")

            _save_montage(
                osp.join(args.out_dir, f"simprof_cam0to3_erp_feats_at_gt_f{fidx:05d}.png"),
                slices=gt_cam_maps,
                titles=gt_cam_titles,
                ncols=4,
                cmap="viridis",
            )

            # Also save the downsampled GT idx map used for projection for sanity
            _save_heatmap(
                osp.join(args.out_dir, f"gt_idx_downsampled_for_lut_f{fidx:05d}.png"),
                np.where(gt_idx_ds >= 0, gt_idx_ds, np.nan),
                title=f"GT invdepth index downsampled to LUT res (H={H},W={W}) - fidx={fidx}",
                cmap="turbo",
            )

            # ------------------------------------------------------------------
            # Project raw fisheye images to ERP at GT depth (same warp as features)
            # ------------------------------------------------------------------
            raw_erp_slices = []
            raw_erp_titles = []
            for cam_i, raw_img in enumerate(raw_imgs):
                img = np.array(raw_img, dtype=np.float32)
                # Normalize to [0,1]
                if img.max() > 1.5:
                    img = img / 255.0
                # Ensure [C, H, W] tensor
                if img.ndim == 2:
                    img_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # [1,1,Hf,Wf]
                else:
                    img_t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)  # [1,C,Hf,Wf]
                img_t = img_t.to(device=device, dtype=torch.float32)

                erp_img = _sample_cam_erp_at_idx(
                    cam_feat=img_t,
                    grid_lut=grids[cam_i].to(device=device, dtype=torch.float32),
                    idx_map=gt_idx_ds_t.to(dtype=torch.float32),
                )  # [1,C,H,W]
                erp_np = erp_img[0].detach().cpu().numpy()  # [C,H,W]

                # Convert back to display-ready format
                if erp_np.shape[0] == 1:
                    erp_vis = erp_np[0]  # [H,W] grayscale
                else:
                    erp_vis = np.clip(erp_np.transpose(1, 2, 0), 0.0, 1.0)  # [H,W,3]

                # Save individual image
                plt.figure(figsize=(12, 4), dpi=150)
                if erp_vis.ndim == 2:
                    plt.imshow(erp_vis, cmap="gray", vmin=0.0, vmax=1.0)
                else:
                    plt.imshow(erp_vis)
                plt.title(f"cam{cam_i} raw @ GT depth (ERP) - fidx={fidx}")
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(osp.join(args.out_dir, f"cam{cam_i}_raw_erp_at_gt_f{fidx:05d}.png"))
                plt.close()

                raw_erp_slices.append(erp_vis)
                raw_erp_titles.append(f"cam{cam_i} raw @ GT")

            _save_montage(
                osp.join(args.out_dir, f"cam0to3_raw_erp_at_gt_f{fidx:05d}.png"),
                slices=raw_erp_slices,
                titles=raw_erp_titles,
                ncols=1,
                cmap="gray",
            )
            # Convert GT depth index map to RGB (turbo) to prepend to strip
            gt_vis = np.where(gt_idx_ds >= 0, gt_idx_ds, np.nan)
            gt_lo, gt_hi = _robust_minmax(gt_vis[np.isfinite(gt_vis)].ravel() if np.any(np.isfinite(gt_vis)) else np.array([0.0, 1.0]))
            gt_norm = np.clip((gt_vis - gt_lo) / max(gt_hi - gt_lo, 1e-6), 0.0, 1.0)
            gt_rgb = plt.get_cmap("turbo")(gt_norm)[..., :3].astype(np.float32)  # [H,W,3]
            gt_rgb[~np.isfinite(gt_vis)] = 0.0  # black for invalid

            _save_vertical_strip(
                osp.join(args.out_dir, f"cam0to3_raw_erp_at_gt_strip_f{fidx:05d}.png"),
                slices=[gt_rgb] + raw_erp_slices,
                cmap="gray",
                gap_inches=0.2,
            )

        except Exception as e:
            print(f"[WARN] Failed to render per-camera ERP features at GT depth: {e}")
    else:
        print("[INFO] No GT available; skipping cam ERP features at GT depth.")

    # If GT is not available, visualize the same projection using the model prediction.
    if ("gt_idx_map" not in locals() or gt_idx_map is None) and pred_idx_map_full is not None:
        try:
            gt_like_idx_ds = _downsample_idx_map_full_to_lut(pred_idx_map_full, (H, W))  # [H,W]
            import torch

            idx_ds_t = torch.from_numpy(gt_like_idx_ds).to(device=device, dtype=cnn_feats[0].dtype).unsqueeze(0).unsqueeze(0)
            cam_pred_erp = []
            for cam_i in range(4):
                feat = _sample_cam_erp_at_idx(
                    cam_feat=cnn_feats[cam_i],
                    grid_lut=grids[cam_i].to(device=device, dtype=cnn_feats[cam_i].dtype),
                    idx_map=idx_ds_t,
                )
                cam_pred_erp.append(feat[0].detach().cpu().numpy())

            pred_cam_maps = []
            pred_cam_titles = []
            for cam_i in range(4):
                m = _feat_to_vis(cam_pred_erp[cam_i], mode=args.cam_feat_vis)
                pred_cam_maps.append(m)
                pred_cam_titles.append(f"cam{cam_i} @ Pred idx (proj)")

            _save_montage(
                osp.join(args.out_dir, f"simprof_cam0to3_erp_feats_at_pred_f{fidx:05d}.png"),
                slices=pred_cam_maps,
                titles=pred_cam_titles,
                ncols=4,
                cmap="viridis",
            )

            _save_heatmap(
                osp.join(args.out_dir, f"pred_idx_downsampled_for_lut_f{fidx:05d}.png"),
                gt_like_idx_ds,
                title=f"Pred invdepth index downsampled to LUT res (H={H},W={W}) - fidx={fidx}",
                cmap="turbo",
            )
        except Exception as e:
            print(f"[WARN] Failed to render per-camera ERP features at Pred depth: {e}")

    # ------------------------------------------------------------------
    # Visualize the "pair" used to form correlation (cam1+cam3 vs cam2+cam4)
    # In codebase indexing this is typically (cam0+cam2) as ref and (cam1+cam3) as tgt.
    # We visualize ref/tgt feature-map strength in ERP for selected depth samples.
    # ------------------------------------------------------------------
    try:
        sim_type = str(getattr(model.opts, "similarity_type", getattr(model.similarity_context.encoder, "similarity_type", "")))
    except Exception:
        sim_type = ""

    if sim_type == "correlation":
        enc = model.similarity_context.encoder  # SimilarityProfileEncoder
        # Build per-cam ERP×depth volumes for the sampled depth indices (Ds)
        # Reuse depth_idx/cam_vols computed above (same sparse depth samples).
        f0_vol, f1_vol, f2_vol, f3_vol = cam_vols  # [B,C,H,W,Ds]
        # [B,C,H,W,Ds]

        # Compute ref/tgt with the same logic as SimilarityProfileEncoder.forward (correlation branch)
        use_view_weights = bool(getattr(enc, "use_view_weights", False)) and enc.reference_mapping is not None and enc.target_mapping is not None

        if use_view_weights:
            # Need selected grids (g*_sel) for MLP inputs
            g_sel = [g.index_select(dim=2, index=torch.tensor(depth_idx, device=g.device)) for g in grids]  # [H,W,Ds,2]
            B = int(f0_vol.shape[0])
            Ds_eff = int(f0_vol.shape[4])
            H_erp, W_erp = int(f0_vol.shape[2]), int(f0_vol.shape[3])

            def _flatten_depth(x):
                # [B,C,H,W,Ds] -> [B*Ds,C,H,W]
                return x.permute(0, 4, 1, 2, 3).reshape(B * Ds_eff, x.shape[1], H_erp, W_erp).contiguous()

            def _grid_flat(g):
                # [H,W,Ds,2] -> [B*Ds,2,H,W]
                gd = g.permute(2, 3, 0, 1).unsqueeze(0).expand(B, -1, -1, -1, -1)
                return gd.reshape(B * Ds_eff, 2, H_erp, W_erp).contiguous()

            f0_bd, f1_bd, f2_bd, f3_bd = map(_flatten_depth, (f0_vol, f1_vol, f2_vol, f3_vol))
            g0_bd, g1_bd, g2_bd, g3_bd = map(_grid_flat, g_sel)

            w_front = enc.reference_mapping(torch.cat([f0_bd, f2_bd, g0_bd, g2_bd], dim=1))  # [B*Ds,1,H,W]
            w_right = enc.target_mapping(torch.cat([f1_bd, f3_bd, g1_bd, g3_bd], dim=1))     # [B*Ds,1,H,W]

            w_front = w_front.view(B, Ds_eff, 1, H_erp, W_erp).permute(0, 2, 3, 4, 1).contiguous()
            w_right = w_right.view(B, Ds_eff, 1, H_erp, W_erp).permute(0, 2, 3, 4, 1).contiguous()

            ref = w_front * f0_vol + (1.0 - w_front) * f2_vol
            tgt = w_right * f1_vol + (1.0 - w_right) * f3_vol
        else:
            ref = 0.5 * (f0_vol + f2_vol)
            tgt = 0.5 * (f1_vol + f3_vol)

        # Visualize a subset of depth slices (side-by-side ref/tgt)
        n_pair = int(max(1, min(args.num_pair_slices, Ds)))
        pair_idxs = np.linspace(0, Ds - 1, n_pair).round().astype(np.int64)
        pair_idxs = np.unique(pair_idxs)

        ref_np = ref[0].detach().cpu().numpy()  # [C,H,W,Ds]
        tgt_np = tgt[0].detach().cpu().numpy()  # [C,H,W,Ds]

        pair_slices = []
        pair_titles = []
        for di in pair_idxs.tolist():
            ref_map = _feat_to_vis(ref_np[:, :, :, di], mode=args.pair_feat_vis)
            tgt_map = _feat_to_vis(tgt_np[:, :, :, di], mode=args.pair_feat_vis)
            pair_slices.extend([ref_map, tgt_map])
            pair_titles.extend([f"ref Ds[{di}]", f"tgt Ds[{di}]"])

        _save_montage(
            osp.join(args.out_dir, f"simprof_pair_ref_tgt_f{fidx:05d}.png"),
            slices=pair_slices,
            titles=pair_titles,
            ncols=4,  # 2 pairs per row by default
            cmap="viridis",
        )
    else:
        print(f"[INFO] similarity_type={sim_type!r}; skipping ref/tgt pair visualization (only for correlation).")

    # ------------------------------------------------------------------
    # Save ref/tgt images at GT depth: both raw image and CNN feature
    # Uses view weights when available, otherwise simple average.
    # ------------------------------------------------------------------
    if gt_idx_map is not None and "raw_erp_slices" in locals() and len(raw_erp_slices) == 4:
        try:
            has_vw = "w_front" in locals()

            # Map GT LUT depth index -> fractional Ds index (needed for both branches)
            ds_indices_np = np.arange(Ds, dtype=np.float64)
            depth_idx_f = depth_idx.astype(np.float64)
            gt_ds_idx = np.interp(
                gt_idx_ds.astype(np.float64).ravel(),
                depth_idx_f,
                ds_indices_np,
            ).reshape(H, W).astype(np.float32)
            gt_ds_idx = np.clip(gt_ds_idx, 0, Ds - 1)

            if has_vw:

                wf = w_front[0, 0].detach().cpu().numpy()  # [H,W,Ds]
                wr = w_right[0, 0].detach().cpu().numpy()  # [H,W,Ds]

                idx_fl = np.floor(gt_ds_idx).astype(np.int64)
                idx_ce = np.minimum(idx_fl + 1, Ds - 1)
                frac = gt_ds_idx - idx_fl.astype(np.float32)

                yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
                wf_gt = (1 - frac) * wf[yy, xx, idx_fl] + frac * wf[yy, xx, idx_ce]  # [H,W]
                wr_gt = (1 - frac) * wr[yy, xx, idx_fl] + frac * wr[yy, xx, idx_ce]

                wf_gt_img = wf_gt[..., np.newaxis] if raw_erp_slices[0].ndim == 3 else wf_gt
                wr_gt_img = wr_gt[..., np.newaxis] if raw_erp_slices[0].ndim == 3 else wr_gt

                # Raw image ref/tgt
                ref_img = wf_gt_img * raw_erp_slices[0].astype(np.float32) + (1 - wf_gt_img) * raw_erp_slices[2].astype(np.float32)
                tgt_img = wr_gt_img * raw_erp_slices[1].astype(np.float32) + (1 - wr_gt_img) * raw_erp_slices[3].astype(np.float32)

                # CNN feature ref/tgt (cam_gt_erp: list of [C,H,W])
                wf_gt_feat = wf_gt[np.newaxis, :, :]  # [1,H,W]
                wr_gt_feat = wr_gt[np.newaxis, :, :]
                ref_feat = wf_gt_feat * cam_gt_erp[0] + (1 - wf_gt_feat) * cam_gt_erp[2]  # [C,H,W]
                tgt_feat = wr_gt_feat * cam_gt_erp[1] + (1 - wr_gt_feat) * cam_gt_erp[3]
                blend_label = "view-weighted"
            else:
                ref_img = (raw_erp_slices[0].astype(np.float32) + raw_erp_slices[2].astype(np.float32)) / 2.0
                tgt_img = (raw_erp_slices[1].astype(np.float32) + raw_erp_slices[3].astype(np.float32)) / 2.0
                ref_feat = (cam_gt_erp[0] + cam_gt_erp[2]) / 2.0
                tgt_feat = (cam_gt_erp[1] + cam_gt_erp[3]) / 2.0
                blend_label = "avg"

            # Save raw image ref/tgt
            for name, img in [("ref", ref_img), ("tgt", tgt_img)]:
                plt.figure(figsize=(12, 4), dpi=150)
                if img.ndim == 2:
                    plt.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
                else:
                    plt.imshow(np.clip(img, 0.0, 1.0))
                plt.title(f"{name} raw @ GT depth ({blend_label}) - fidx={fidx}")
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(osp.join(args.out_dir, f"{name}_raw_erp_at_gt_f{fidx:05d}.png"))
                plt.close()

            _save_vertical_strip(
                osp.join(args.out_dir, f"ref_tgt_raw_erp_at_gt_strip_f{fidx:05d}.png"),
                slices=[ref_img, tgt_img],
                cmap="gray",
                gap_inches=0.2,
            )

            # Save CNN feature ref/tgt as vertical strip with GT depth on top
            ref_feat_vis = _feat_to_vis(ref_feat, mode=args.cam_feat_vis)
            tgt_feat_vis = _feat_to_vis(tgt_feat, mode=args.cam_feat_vis)

            # Normalize feat maps to [0,1] RGB for strip (apply viridis if 2D)
            feat_strip_slices = []
            for fv in [ref_feat_vis, tgt_feat_vis]:
                if fv.ndim == 2:
                    lo, hi = _robust_minmax(fv)
                    normed = np.clip((fv - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
                    feat_strip_slices.append(plt.get_cmap("viridis")(normed)[..., :3].astype(np.float32))
                else:
                    feat_strip_slices.append(np.clip(fv, 0.0, 1.0).astype(np.float32))

            _save_vertical_strip(
                osp.join(args.out_dir, f"ref_tgt_feat_at_gt_f{fidx:05d}.png"),
                slices=[gt_rgb] + feat_strip_slices,
                cmap="gray",
                gap_inches=0.2,
            )

            # ---- PCA visualization of ref/tgt CNN features ----
            ref_pca = _pca_rgb_from_chw(ref_feat)  # [H,W,3]
            tgt_pca = _pca_rgb_from_chw(tgt_feat)  # [H,W,3]
            _save_vertical_strip(
                osp.join(args.out_dir, f"ref_tgt_feat_pca_at_gt_f{fidx:05d}.png"),
                slices=[gt_rgb, ref_pca, tgt_pca],
                cmap="gray",
                gap_inches=0.2,
            )

            # ---- Context PCA + DINO avg PCA at GT depth ----
            try:
                import torch.nn.functional as F_nn

                gt_ds_idx_t = torch.from_numpy(gt_ds_idx).to(
                    device=device, dtype=similarity_profile.dtype
                ).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

                # 1) Context features (corr + GEV if available)
                with torch.no_grad():
                    ctx_feat = model.similarity_context.lookup_context(
                        similarity_profile, gt_ds_idx_t
                    )  # [B, ctx_dim, H, W]
                ctx_np = ctx_feat[0].detach().cpu().numpy()  # [ctx_dim, H, W]
                ctx_pca = _pca_rgb_from_chw(ctx_np)  # [H,W,3]

                # 2) DINO features: simple avg of 4 cams (no deformable)
                # Build valid mask from raw ERP images (where fisheye actually covers)
                raw_valid_masks = []  # [4] of [H,W] bool
                for cam_i in range(4):
                    rv = raw_erp_slices[cam_i]
                    if rv.ndim == 3:
                        raw_valid_masks.append(np.abs(rv).sum(axis=-1) > 1e-6)
                    else:
                        raw_valid_masks.append(np.abs(rv) > 1e-6)

                with torch.no_grad():
                    dino_feats_stacked = model.dino_extractor.extract_features(imgs_t)
                    dino_feats_stacked = dino_feats_stacked.float()
                    cnn_h, cnn_w = cnn_feats[0].shape[2], cnn_feats[0].shape[3]
                    dino_gt_erp = []
                    for cam_i in range(4):
                        dino_cam = dino_feats_stacked[:, cam_i]
                        dino_cam_up = F_nn.interpolate(
                            dino_cam, size=(cnn_h, cnn_w),
                            mode="bilinear", align_corners=False,
                        )
                        erp_feat = _sample_cam_erp_at_idx(
                            cam_feat=dino_cam_up,
                            grid_lut=grids[cam_i].to(device=device, dtype=dino_cam_up.dtype),
                            idx_map=gt_idx_ds_t,
                        )
                        feat_np = erp_feat[0].detach().cpu().numpy()  # [C_d,H,W]
                        # Zero out where raw image has no coverage
                        feat_np[:, ~raw_valid_masks[cam_i]] = 0.0
                        dino_gt_erp.append(feat_np)
                dino_stack = np.stack(dino_gt_erp, axis=0)  # [4, C_d, H, W]
                # Valid mask from raw images
                dino_valid = np.stack(raw_valid_masks, axis=0)[:, np.newaxis, :, :].astype(np.float32)  # [4,1,H,W]
                dino_count = dino_valid.sum(axis=0).clip(min=1.0)  # [1,H,W]
                dino_avg = (dino_stack * dino_valid).sum(axis=0) / dino_count  # [C_d,H,W]
                dino_avg_pca = _pca_rgb_from_chw(dino_avg)  # [H,W,3]

                # CNN features: valid-mask avg of 4 cams (reuse cam_gt_erp)
                cnn_stack = np.stack(cam_gt_erp, axis=0)  # [4, C, H, W]
                cnn_valid = np.stack(raw_valid_masks, axis=0)[:, np.newaxis, :, :].astype(np.float32)
                cnn_count = cnn_valid.sum(axis=0).clip(min=1.0)
                cnn_avg = (cnn_stack * cnn_valid).sum(axis=0) / cnn_count  # [C,H,W]
                cnn_avg_pca = _pca_rgb_from_chw(cnn_avg)

                # --- Helper: l2 → viridis RGB ---
                def _l2_viridis(feat_chw):
                    fv = _feat_to_vis(feat_chw, mode="l2")
                    lo, hi = _robust_minmax(fv)
                    normed = np.clip((fv - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
                    return plt.get_cmap("viridis")(normed)[..., :3].astype(np.float32)

                # DINO+CNN average
                if cnn_avg.shape == dino_avg.shape:
                    dino_cnn_avg = (dino_avg + cnn_avg) / 2.0
                else:
                    # Channel mismatch: concat then PCA will handle it
                    dino_cnn_avg = np.concatenate([dino_avg, cnn_avg], axis=0)
                dino_cnn_avg_pca = _pca_rgb_from_chw(dino_cnn_avg)

                # context_at_gt (vivid PCA): GT (viridis) / context / DINO+CNN avg / DINO / CNN
                gt_raw = gt_idx_ds.astype(np.float32)
                gt_lo_r, gt_hi_r = _robust_minmax(gt_raw)
                gt_norm_r = np.clip((gt_raw - gt_lo_r) / max(gt_hi_r - gt_lo_r, 1e-6), 0.0, 1.0)
                gt_rgb_viridis = plt.get_cmap("viridis")(gt_norm_r)[..., :3].astype(np.float32)

                _save_vertical_strip(
                    osp.join(args.out_dir, f"context_at_gt_f{fidx:05d}.png"),
                    slices=[gt_rgb_viridis,
                            _pca_rgb_vivid(ctx_np),
                            _pca_rgb_vivid(dino_cnn_avg),
                            _pca_rgb_vivid(dino_avg),
                            _pca_rgb_vivid(cnn_avg)],
                    cmap="gray",
                    gap_inches=0.2,
                )
                # context_at_gt (turbo): GT / context PC1 / DINO+CNN avg PC1
                _save_vertical_strip(
                    osp.join(args.out_dir, f"context_at_gt_turbo_f{fidx:05d}.png"),
                    slices=[gt_rgb_viridis,
                            _pca_turbo(ctx_np),
                            _pca_turbo(dino_cnn_avg),
                            _pca_turbo(dino_avg),
                            _pca_turbo(cnn_avg)],
                    cmap="gray",
                    gap_inches=0.2,
                )
                # context_at_gt (raw): GT / context l2 / DINO+CNN avg l2
                _save_vertical_strip(
                    osp.join(args.out_dir, f"context_at_gt_raw_f{fidx:05d}.png"),
                    slices=[gt_rgb_viridis,
                            _l2_viridis(ctx_np), _l2_viridis(dino_cnn_avg),
                            _l2_viridis(dino_avg), _l2_viridis(cnn_avg)],
                    cmap="gray",
                    gap_inches=0.2,
                )

                # dino_per_cam (PCA): cam0..3 / avg
                dino_cam_pcas = [_pca_rgb_from_chw(dino_gt_erp[ci]) for ci in range(4)]
                _save_vertical_strip(
                    osp.join(args.out_dir, f"dino_per_cam_at_gt_f{fidx:05d}.png"),
                    slices=dino_cam_pcas + [dino_avg_pca],
                    cmap="gray",
                    gap_inches=0.2,
                )
                # dino_per_cam (raw): cam0..3 / avg
                _save_vertical_strip(
                    osp.join(args.out_dir, f"dino_per_cam_at_gt_raw_f{fidx:05d}.png"),
                    slices=[_l2_viridis(dino_gt_erp[ci]) for ci in range(4)] + [_l2_viridis(dino_avg)],
                    cmap="gray",
                    gap_inches=0.2,
                )

                # cnn_per_cam (PCA): cam0..3 / avg
                cnn_cam_pcas = [_pca_rgb_from_chw(cam_gt_erp[ci]) for ci in range(4)]
                _save_vertical_strip(
                    osp.join(args.out_dir, f"cnn_per_cam_at_gt_f{fidx:05d}.png"),
                    slices=cnn_cam_pcas + [cnn_avg_pca],
                    cmap="gray",
                    gap_inches=0.2,
                )
                # cnn_per_cam (raw): cam0..3 / avg
                _save_vertical_strip(
                    osp.join(args.out_dir, f"cnn_per_cam_at_gt_raw_f{fidx:05d}.png"),
                    slices=[_l2_viridis(cam_gt_erp[ci]) for ci in range(4)] + [_l2_viridis(cnn_avg)],
                    cmap="gray",
                    gap_inches=0.2,
                )

                print(f"[INFO] Context shape: {ctx_np.shape}, DINO avg shape: {dino_avg.shape}, "
                      f"valid cam count range: [{int(dino_count.min())}, {int(dino_count.max())}]")
            except Exception as e:
                print(f"[WARN] Failed to compute/save context+DINO features: {e}")

        except Exception as e:
            print(f"[WARN] Failed to render ref/tgt images at GT depth: {e}")

    # Plot a few per-pixel depth profiles
    # IMPORTANT: Keep point identities consistent across:
    # - profile space (H,W) where similarity_profile lives
    # - full ERP space (H_full,W_full) where GT/pred idx maps live
    ds = int(getattr(model.opts, "num_downsample", 1))
    scale_hw = int(2 ** ds)

    # Prefer full-res sizes from GT/pred if available; otherwise fall back to model opts.
    # Prediction map (full-res) if we got it
    pred_idx_map = None
    if "pred_invdepth_idx" in locals() and pred_invdepth_idx is not None:
        try:
            pred_idx_map = pred_invdepth_idx[0, 0].detach().cpu().numpy().astype(np.float32)
        except Exception:
            pred_idx_map = None

    if pred_idx_map is not None:
        H_full, W_full = int(pred_idx_map.shape[0]), int(pred_idx_map.shape[1])
    elif gt_idx_map is not None:
        H_full, W_full = int(gt_idx_map.shape[0]), int(gt_idx_map.shape[1])
    else:
        H_full = int(getattr(model.opts, "equi_h", H * scale_hw))
        W_full = int(getattr(model.opts, "equi_w", W * scale_hw))

    # Choose points in full-res ERP coordinates (these define P0/P1/P2).
    ys_full = [H_full // 2, H_full // 2 - 7, (3 * H_full) // 4]
    xs_full = [W_full // 2, 5*W_full // 20, (3 * W_full) // 4]

    # Map to profile-res coordinates for indexing similarity_profile
    ys = [int(np.clip(y // scale_hw, 0, H - 1)) for y in ys_full]
    xs = [int(np.clip(x // scale_hw, 0, W - 1)) for x in xs_full]
    # Collect GT depth at those points (if available)
    gt_idx_vals = []
    gt_prof_idx_vals = []
    gt_valid_flags = []
    pred_idx_vals = []
    pred_prof_idx_vals = []

    if gt_idx_map is not None:
        invdepth_gt = data.indexToInvdepth(gt_idx_map)
        # Map GT invdepth index (0..num_invdepth) -> similarity profile index (0..Ds-1)
        num_invdepth = int(getattr(model.opts, "num_invdepth", 192))
        prof_scale = float(Ds) / float(max(1, num_invdepth))
        for (yF, xF) in zip(ys_full, xs_full):
            yF = int(np.clip(yF, 0, gt_idx_map.shape[0] - 1))
            xF = int(np.clip(xF, 0, gt_idx_map.shape[1] - 1))
            gt_idx_v = float(gt_idx_map[yF, xF])
            ok = bool(np.isfinite(gt_idx_v) and (gt_idx_v >= 0))
            gt_valid_flags.append(ok)
            gt_idx_vals.append(gt_idx_v if ok else float("nan"))
            gt_prof_idx_vals.append((gt_idx_v * prof_scale) if ok else float("nan"))
    else:
        gt_valid_flags = [False] * len(xs)
        gt_idx_vals = [float("nan")] * len(xs)
        gt_prof_idx_vals = [float("nan")] * len(xs)

    if pred_idx_map is not None:
        num_invdepth = int(getattr(model.opts, "num_invdepth", 192))
        prof_scale = float(Ds) / float(max(1, num_invdepth))
        for (yF, xF) in zip(ys_full, xs_full):
            yF = int(np.clip(yF, 0, pred_idx_map.shape[0] - 1))
            xF = int(np.clip(xF, 0, pred_idx_map.shape[1] - 1))
            pv = float(pred_idx_map[yF, xF])
            ok = bool(np.isfinite(pv))
            pred_idx_vals.append(pv if ok else float("nan"))
            pred_prof_idx_vals.append((pv * prof_scale) if ok else float("nan"))
    else:
        pred_idx_vals = [float("nan")] * len(xs)
        pred_prof_idx_vals = [float("nan")] * len(xs)

    plt.figure(figsize=(10, 4), dpi=150)
    color_cycle = ["#ff3b30", "#34c759", "#007aff"]  # P0/P1/P2 colors
    for i, (y, x) in enumerate(zip(ys, xs)):
        if gt_valid_flags[i]:
            lab = (
                f"P{i}: full(y={ys_full[i]},x={xs_full[i]}) "
                f"prof(y={y},x={x}) GT idx={gt_idx_vals[i]:.1f} (prof≈{gt_prof_idx_vals[i]:.1f})"
            )
        else:
            lab = f"P{i}: full(y={ys_full[i]},x={xs_full[i]}) prof(y={y},x={x}) GT idx=N/A"
        c = color_cycle[i % len(color_cycle)]
        curve = prof[:, y, x]
        plt.plot(np.arange(Ds), curve, label=lab, color=c, linewidth=1.6)

        # Mark GT location on the curve (in profile-index space) if available
        if gt_valid_flags[i]:
            gx = float(gt_prof_idx_vals[i])
            gy = _interp_profile_y(curve, gx)
            # vertical guide + marker
            plt.axvline(gx, color=c, alpha=0.15, linewidth=1.0)
            plt.scatter([gx], [gy], s=55, color=c, edgecolors="black", linewidths=0.6, zorder=5, marker="o")
            plt.text(
                gx + 1.5,
                gy,
                f"GT",
                color=c,
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.65, pad=1.5, edgecolor="none"),
            )

        # Mark Pred location on the curve (triangle marker) if available
        if np.isfinite(pred_prof_idx_vals[i]):
            px = float(pred_prof_idx_vals[i])
            py = _interp_profile_y(curve, px)
            plt.axvline(px, color=c, alpha=0.10, linewidth=1.0, linestyle="--")
            plt.scatter([px], [py], s=70, color=c, edgecolors="black", linewidths=0.6, zorder=6, marker="^")
            plt.text(
                px + 1.5,
                py,
                "Pred",
                color=c,
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.65, pad=1.5, edgecolor="none"),
            )
    plt.title(f"Similarity depth profiles (3 pixels) - fidx={fidx}")
    plt.xlabel("depth-sample index (0..Ds-1)")
    plt.ylabel("similarity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(osp.join(args.out_dir, f"simprof_profiles_f{fidx:05d}.png"))
    plt.close()

    # ------------------------------------------------------------------
    # Extra visual context: show where those pixels are on ERP + GT depth
    # ------------------------------------------------------------------
    # 1) ERP RGB panorama using GT depth (best-effort; only when GT exists)
    if gt_idx_map is not None:
        try:
            pano_rgb = data.getPanorama_rgb(raw_imgs, invdepth_gt)  # [H,W,3] uint8-ish
            fig = plt.figure(figsize=(12, 4), dpi=150)
            ax = plt.gca()
            ax.imshow(pano_rgb)
            ax.axis("off")
            _overlay_points(
                ax,
                xs=xs_full,
                ys=ys_full,
                labels=[f"P{i}" for i in range(len(xs))],
            )
            plt.title(f"ERP panorama (from GT depth reprojection) + selected pixels - fidx={fidx}")
            plt.tight_layout()
            plt.savefig(osp.join(args.out_dir, f"erp_rgb_points_f{fidx:05d}.png"))
            plt.close(fig)
        except Exception as e:
            print(f"[WARN] Failed to render ERP panorama from GT depth: {e}")

        # 2) GT invdepth index colormap + points + idx values
        try:
            fig = plt.figure(figsize=(12, 4), dpi=150)
            ax = plt.gca()
            gt_idx_vis = gt_idx_map.astype(np.float32).copy()
            # breakpoint()
            # gt_idx_vis[gt_idx_vis < 0] = np.nan
            num_invdepth = int(getattr(model.opts, "num_invdepth", 192))
            im = ax.imshow(gt_idx_vis, cmap="turbo", vmin=0.0, vmax=float(max(1, num_invdepth - 1)))
            ax.axis("off")
            plt.colorbar(im, fraction=0.046, pad=0.04, label="GT invdepth index")
            _overlay_points(
                ax,
                xs=xs_full,
                ys=ys_full,
                labels=[
                    (f"P{i}: idx={gt_idx_vals[i]:.1f}" if gt_valid_flags[i] else f"P{i}: N/A")
                    for i in range(len(xs))
                ],
            )
            plt.title(f"GT invdepth index + selected pixels - fidx={fidx}")
            plt.tight_layout()
            plt.savefig(osp.join(args.out_dir, f"gt_idx_points_f{fidx:05d}.png"))
            plt.close(fig)
        except Exception as e:
            print(f"[WARN] Failed to render GT idx map: {e}")
    else:
        print("[INFO] No GT available for this dataset/sample; skipping ERP/GT overlays.")

    # Pred invdepth index map + selected pixels
    if pred_idx_map is not None:
        try:
            fig = plt.figure(figsize=(12, 4), dpi=150)
            ax = plt.gca()
            num_invdepth = int(getattr(model.opts, "num_invdepth", 192))
            im = ax.imshow(pred_idx_map, cmap="turbo", vmin=0.0, vmax=float(max(1, num_invdepth - 1)))
            ax.axis("off")
            plt.colorbar(im, fraction=0.046, pad=0.04, label="Pred invdepth index")

            _overlay_points(
                ax,
                xs=xs_full,
                ys=ys_full,
                labels=[
                    (f"P{i}: idx={pred_idx_vals[i]:.1f}" if np.isfinite(pred_idx_vals[i]) else f"P{i}: N/A")
                    for i in range(len(xs))
                ],
            )
            plt.title(f"Pred invdepth index + selected pixels - fidx={fidx}")
            plt.tight_layout()
            plt.savefig(osp.join(args.out_dir, f"pred_idx_points_f{fidx:05d}.png"))
            plt.close(fig)
        except Exception as e:
            print(f"[WARN] Failed to render pred idx map: {e}")

    # ------------------------------------------------------------------
    # Query-point similarity profiles (--query_points)
    # ------------------------------------------------------------------
    if args.query_points is not None:
        qpoints = []
        for token in args.query_points.split(";"):
            parts = token.strip().split(",")
            if len(parts) == 2:
                qpoints.append((int(parts[0]), int(parts[1])))  # (y_full, x_full)

        num_invdepth = int(getattr(model.opts, "num_invdepth", 192))
        prof_scale = float(Ds) / float(max(1, num_invdepth))

        # Prepare GT depth image for top panel
        gt_depth_img = None
        if gt_idx_map is not None:
            gt_vis_q = gt_idx_map.astype(np.float32).copy()
            gt_vis_q[gt_vis_q < 0] = np.nan
            num_invdepth_q = int(getattr(model.opts, "num_invdepth", 192))
            lo_q = 0.0
            hi_q = float(max(1, num_invdepth_q - 1))
            gt_norm_q = np.clip((gt_vis_q - lo_q) / (hi_q - lo_q), 0.0, 1.0)
            gt_depth_img = plt.get_cmap("turbo")(gt_norm_q)[..., :3]
            gt_depth_img[~np.isfinite(gt_vis_q)] = 0.0

        for qi, (qy_full, qx_full) in enumerate(qpoints):
            # Map full-res ERP → profile-res
            qy = int(np.clip(qy_full // scale_hw, 0, H - 1))
            qx = int(np.clip(qx_full // scale_hw, 0, W - 1))
            curve = prof[:, qy, qx]  # [Ds]

            fig, (ax_img, ax_prof) = plt.subplots(
                2, 1, figsize=(12, 6), dpi=150,
                gridspec_kw={"height_ratios": [1, 2]},
            )

            # Top: GT depth image + X marker at query point
            if gt_depth_img is not None:
                ax_img.imshow(gt_depth_img)
            ax_img.plot(qx_full, qy_full, marker="x", color="lime",
                        markersize=14, markeredgewidth=3, zorder=10)
            ax_img.set_title(f"Query point ({qy_full}, {qx_full})")
            ax_img.axis("off")

            # Bottom: similarity profile
            ax_prof.plot(np.arange(Ds), curve, color="#007aff", linewidth=1.8,
                         label=f"similarity @ ({qy_full},{qx_full})")

            # GT depth index → green dashed line
            if gt_idx_map is not None:
                yF = int(np.clip(qy_full, 0, gt_idx_map.shape[0] - 1))
                xF = int(np.clip(qx_full, 0, gt_idx_map.shape[1] - 1))
                gt_v = float(gt_idx_map[yF, xF])
                if np.isfinite(gt_v) and gt_v >= 0:
                    gt_prof = gt_v * prof_scale
                    ax_prof.axvline(gt_prof, color="green", linestyle="--", linewidth=2.0,
                                    label=f"GT idx={gt_v:.1f} (prof≈{gt_prof:.1f})")
                    gy = _interp_profile_y(curve, gt_prof)
                    ax_prof.scatter([gt_prof], [gy], s=80, color="green",
                                    edgecolors="black", linewidths=0.8, zorder=5)

            ax_prof.set_xlabel("depth-sample index (0..Ds-1)")
            ax_prof.set_ylabel("similarity")
            ax_prof.grid(True, alpha=0.3)
            ax_prof.legend()
            fig.tight_layout()
            fig.savefig(osp.join(args.out_dir, f"query_sim_q{qi}_y{qy_full}_x{qx_full}_f{fidx:05d}.png"))
            plt.close(fig)
            print(f"[INFO] Query point {qi}: ({qy_full},{qx_full}) → profile saved.")

        # Combined comparison figure (when 2+ points)
        if len(qpoints) >= 2:
            point_colors = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4"]
            fig, (ax_img, ax_prof) = plt.subplots(
                2, 1, figsize=(14, 7), dpi=150,
                gridspec_kw={"height_ratios": [1, 2]},
            )

            # Top: GT depth + all query points
            if gt_depth_img is not None:
                ax_img.imshow(gt_depth_img)
            else:
                ax_img.set_facecolor("black")
            for qi, (qy_full, qx_full) in enumerate(qpoints):
                c = point_colors[qi % len(point_colors)]
                ax_img.plot(qx_full, qy_full, marker="x", color=c,
                            markersize=16, markeredgewidth=3, zorder=10)
                ax_img.annotate(f"Q{qi}", (qx_full + 8, qy_full - 8),
                                color=c, fontsize=11, fontweight="bold",
                                bbox=dict(facecolor="white", alpha=0.7, pad=1, edgecolor="none"))
            ax_img.set_title("Query points on GT depth")
            ax_img.axis("off")

            # Bottom: all profiles overlaid
            for qi, (qy_full, qx_full) in enumerate(qpoints):
                qy = int(np.clip(qy_full // scale_hw, 0, H - 1))
                qx = int(np.clip(qx_full // scale_hw, 0, W - 1))
                curve = prof[:, qy, qx]
                c = point_colors[qi % len(point_colors)]
                ax_prof.plot(np.arange(Ds), curve, color=c, linewidth=1.8,
                             label=f"Q{qi} ({qy_full},{qx_full})")

                # GT dashed line
                if gt_idx_map is not None:
                    yF = int(np.clip(qy_full, 0, gt_idx_map.shape[0] - 1))
                    xF = int(np.clip(qx_full, 0, gt_idx_map.shape[1] - 1))
                    gt_v = float(gt_idx_map[yF, xF])
                    if np.isfinite(gt_v) and gt_v >= 0:
                        gt_prof = gt_v * prof_scale
                        ax_prof.axvline(gt_prof, color=c, linestyle="--",
                                        linewidth=1.5, alpha=0.6)
                        gy = _interp_profile_y(curve, gt_prof)
                        ax_prof.scatter([gt_prof], [gy], s=70, color=c,
                                        edgecolors="black", linewidths=0.6, zorder=5)

            ax_prof.set_xlabel("depth-sample index (0..Ds-1)")
            ax_prof.set_ylabel("similarity")
            ax_prof.set_title("Similarity profiles comparison")
            ax_prof.grid(True, alpha=0.3)
            ax_prof.legend()
            fig.tight_layout()
            fig.savefig(osp.join(args.out_dir, f"query_sim_compare_f{fidx:05d}.png"))
            plt.close(fig)
            print(f"[INFO] Combined comparison figure saved.")

    print(f"[DONE] Wrote outputs to: {osp.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()

