# test.py
# Evaluation-only script for OmniDS with wandb logging

from __future__ import print_function, division

import os
import random
import time
import numpy as np
from argparse import ArgumentParser

import torch
import torch.nn as nn
import wandb

# Internal modules
from dataset import Dataset, MultiDataset
from utils.common import *
from utils.image import *
from module.network import OmniDS


def set_reproducibility(seed: int, deterministic: bool = False):
    """cuDNN autotuning is on by default; --deterministic trades ~8% speed for
    bit-exact runs (cuDNN otherwise picks non-deterministic algorithms for the
    GEV 3D convolutions, which compounds over the GRU refinement iterations)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)


# -------------------------------------------------------------
# Arguments
# -------------------------------------------------------------
parser = ArgumentParser(description='Evaluation for OmniDS')

parser.add_argument('--ckpt', required=True, help='checkpoint path to evaluate')
parser.add_argument('--name', default='ROmniStereo_BEV_eval', help='wandb run name')
parser.add_argument('--wandb', action='store_true', help='enable wandb logging')
parser.add_argument('--seed', type=int, default=0, help='random seed for reproducibility')
parser.add_argument('--deterministic', action='store_true',
                    help='enable deterministic inference (slower, less variance)')

parser.add_argument('--db_root', default='/media/vdcl/T7 Shield/RVLab/Paper/data', type=str)
parser.add_argument('--dbname', nargs='+', default=['omnithings'],
                    choices=['omnithings', 'omnihouse', 'sunny', 'cloudy', 'sunset'])

# data options
parser.add_argument('--phi_deg', type=float, default=45.0)
parser.add_argument('--num_invdepth', type=int, default=192)
parser.add_argument('--equirect_size', type=int, nargs='+', default=[160, 640])
parser.add_argument('--use_rgb', action='store_true')

# net options
parser.add_argument('--base_channel', type=int, default=32, help='base channel of the network')
parser.add_argument('--encoder_downsample_twice', action='store_true',
                    help='the feature extractor downsample the fisheye input twice instead once.')
parser.add_argument('--num_downsample', type=int, default=1, help="resolution of the disparity field (1/2^K)")

# ERP / cross-attention options
parser.add_argument('--num_heads', type=int, default=4, help="number of attention heads")
parser.add_argument('--num_points', type=int, default=4, help="number of sampling points per head")
parser.add_argument('--num_cross_attn_layers', type=int, default=1, help="number of cross-attention layers")
# Similarity context options
parser.add_argument('--num_depth_samples', type=int, default=96, help="number of depth samples for similarity profile")
parser.add_argument('--sim_radius', type=int, default=4, help="lookup radius for similarity context")
parser.add_argument('--sim_levels', type=int, default=4, help="number of pyramid levels for similarity lookup")
parser.add_argument('--similarity_type', type=str, default='correlation', 
                    choices=['variance', 'correlation', 'pairwise'], help="similarity computation method")
# Spatial aggregation options for similarity profile
parser.add_argument('--use_spatial_aggregation', action='store_true', default=True,
                    help='apply spatial aggregation on similarity profile (recommended)')
parser.add_argument('--spatial_agg_type', type=str, default='conv',
                    choices=['conv', 'deformable', 'multiscale'],
                    help='type of spatial aggregation: conv (simple), deformable (instance-aware), multiscale (ASPP)')
parser.add_argument('--spatial_hidden_dim', type=int, default=32,
                    help='hidden dimension for spatial aggregation conv')
parser.add_argument('--spatial_num_layers', type=int, default=2,
                    help='number of conv layers for spatial aggregation')

# DINO options
parser.add_argument('--use_dino_context', action='store_true',
                    help='deprecated no-op; DINO context is always used')
parser.add_argument('--dino_model', type=str, default='dinov3_vits16',
                    choices=['dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16'],
                    help="DINOv3 model variant (patch 16)")
parser.add_argument('--freeze_dino', action='store_true', default=True,
                    help='freeze DINO backbone weights')
parser.add_argument('--context_fusion_type', type=str, default='pointwise',
                    choices=['pointwise', 'gated', 'local', 'attention'],
                    help='DINO-CNN fusion type: pointwise (fast), gated, local, attention (slow)')

# GEV options (separate GEV with 3D UNet)
parser.add_argument('--use_gev', action='store_true',
                    help='deprecated no-op; the GEV branch is always used')
parser.add_argument('--gev_num_groups', type=int, default=8,
                    help='number of groups for group-wise correlation')
parser.add_argument('--gev_reg_channels', type=int, nargs='+', default=[16, 32, 48],
                    help='channels for 3D UNet regularization (3 stages)')
parser.add_argument('--gev_num_pyramid_levels', type=int, default=4,
                    help='number of depth-axis pyramid levels for GEV lookup')
parser.add_argument('--gev_radius', type=int, default=4,
                    help='lookup radius for GEV depth-axis sampling')
parser.add_argument('--gev_use_spatial_downsample', action='store_true', default=True,
                    help='spatial 2x downsample before 3D UNet')

parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--fix_bn', action='store_true', help='fix batch normalization')
parser.add_argument('--distilled', action='store_true', help='use distilled model')
parser.add_argument('--triton', action='store_true', help='use Triton fused kernels for GEV')

# training options
parser.add_argument('--total_epochs', type=int, default=100, help='total epochs of training')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--train_iters', type=int, default=12,
                    help="number of updates to the disparity field in each forward pass.")
parser.add_argument('--lr', type=float, default=0.0005, help="max learning rate.")
parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")

args = parser.parse_args()


# -------------------------------------------------------------
# Options (train.py와 동일한 구조)
# -------------------------------------------------------------
opts = Edict()
# Dataset & sweep arguments
opts.name = args.name
opts.model_dir = os.path.join('./checkpoints', args.name)

opts.dbname = args.dbname
opts.db_root = args.db_root

opts.data_opts = Edict()
opts.data_opts.phi_deg = args.phi_deg
opts.data_opts.num_invdepth = args.num_invdepth
opts.data_opts.equirect_size = args.equirect_size
opts.data_opts.num_downsample = args.num_downsample
opts.data_opts.use_rgb = args.use_rgb

opts.net_opts = Edict()
opts.net_opts.base_channel = args.base_channel
opts.net_opts.num_invdepth = opts.data_opts.num_invdepth
opts.net_opts.use_rgb = opts.data_opts.use_rgb
opts.net_opts.encoder_downsample_twice = args.encoder_downsample_twice
opts.net_opts.num_downsample = args.num_downsample
opts.net_opts.mixed_precision = args.mixed_precision
opts.net_opts.fix_bn = args.fix_bn
opts.distilled = args.distilled
# Build the student encoder only when the distilled path is requested
opts.net_opts.use_student = args.distilled
opts.net_opts.use_triton = args.triton

# ERP / cross-attention
opts.net_opts.equi_h = args.equirect_size[0]
opts.net_opts.equi_w = args.equirect_size[1]
opts.net_opts.phi_deg = args.phi_deg
opts.net_opts.num_heads = args.num_heads
opts.net_opts.num_points = args.num_points
opts.net_opts.num_cross_attn_layers = args.num_cross_attn_layers
# Similarity context options
opts.net_opts.num_depth_samples = args.num_depth_samples
opts.net_opts.sim_radius = args.sim_radius
opts.net_opts.sim_levels = args.sim_levels
opts.net_opts.similarity_type = args.similarity_type
# Spatial aggregation options
opts.net_opts.use_spatial_aggregation = args.use_spatial_aggregation
opts.net_opts.spatial_agg_type = args.spatial_agg_type
opts.net_opts.spatial_hidden_dim = args.spatial_hidden_dim
opts.net_opts.spatial_num_layers = args.spatial_num_layers

# DINO context options
opts.net_opts.dino_model = args.dino_model
opts.net_opts.freeze_dino = args.freeze_dino
opts.net_opts.context_fusion_type = args.context_fusion_type

# GEV options (v5: separate GEV with 3D UNet)
opts.net_opts.gev_num_groups = args.gev_num_groups
opts.net_opts.gev_reg_channels = tuple(args.gev_reg_channels)
opts.net_opts.gev_num_pyramid_levels = args.gev_num_pyramid_levels
opts.net_opts.gev_radius = args.gev_radius
opts.net_opts.gev_use_spatial_downsample = args.gev_use_spatial_downsample

opts.total_epochs = args.total_epochs
opts.batch_size = args.batch_size
opts.train_iters = args.train_iters
opts.lr = args.lr
opts.wdecay = args.wdecay


# -------------------------------------------------------------
# Main Eval
# -------------------------------------------------------------
def main():
    set_reproducibility(args.seed, args.deterministic)

    # Dataset
    if len(opts.dbname) > 1:
        data = MultiDataset(opts.dbname, opts.data_opts, db_root=opts.db_root)
    else:
        data = Dataset(opts.dbname[0], opts.data_opts, db_root=opts.db_root)

    # Network (v3 or v5)
    LOG_INFO("Using OmniDS (DINO context + separate GEV)")
    net = OmniDS(opts.net_opts)

    net = nn.DataParallel(net).cuda()
    net.eval()

    # Load checkpoint
    assert os.path.exists(args.ckpt), f"Checkpoint not found: {args.ckpt}"
    snapshot = torch.load(args.ckpt, map_location="cpu")
    net.load_state_dict(snapshot['net_state_dict'])
    LOG_INFO(f"Loaded checkpoint: {args.ckpt}")

    # WandB
    if args.wandb:
        wandb.init(
            entity="zhyeon-konkuk-university",
            project="Videodepth",
            name=args.name,
            config=vars(args),
            settings=wandb.Settings(start_method="fork")
        )

    # Grids
    grids = [torch.tensor(grid, requires_grad=False).cuda()
             for grid in data.grids]

    eval_list = data.opts.test_idx
    errors = np.zeros((len(eval_list), 5))

    net_eval = net.module if hasattr(net, "module") else net
    infer_ms = np.zeros(len(eval_list))
    for d, fidx in enumerate(eval_list):
        imgs, gt, valid, raw_imgs = data.loadSample(fidx)
        imgs = [torch.Tensor(img).unsqueeze(0).cuda() for img in imgs]

        # Time the network only. CUDA kernels are launched asynchronously, so
        # synchronise on both sides or we would just be timing the launch.
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            invdepth_idx = net_eval(imgs, grids, test_mode=True, iters=opts.train_iters,
                                    distilled=opts.distilled)
        torch.cuda.synchronize()
        infer_ms[d] = (time.time() - start) * 1000

        invdepth_idx_np = toNumpy(invdepth_idx[0, 0])

        # metrics
        errors[d, :] = data.evalError(invdepth_idx_np, gt, valid)
        mean_errors = errors[:d + 1].mean(axis=0)

        LOG_INFO(
            'Eval %d/%d | >1: %.3f, >3: %.3f, >5: %.3f, MAE: %.3f, RMS: %.3f | %.1fms'
            % (d + 1, len(eval_list),
               errors[d,0], errors[d,1], errors[d,2],
               errors[d,3], errors[d,4],
               infer_ms[d])
        )

        # ---------------- wandb logging ----------------
        if args.wandb:
            wandb.log({
                "Eval/>1":  errors[d,0],
                "Eval/>3":  errors[d,1],
                "Eval/>5":  errors[d,2],
                "Eval/MAE": errors[d,3],
                "Eval/RMS": errors[d,4],
            }, step=d)

            # Visualization (train 코드와 동일)
            raw_imgs_vis = [img for img in raw_imgs]
            pred_vis = data.indexToInvdepth(invdepth_idx_np)

            if torch.is_tensor(gt):
                gt_vis = data.indexToInvdepth(gt).detach().cpu() \
                         if gt.dim() == 3 else data.indexToInvdepth(gt).detach().cpu()
            else:
                gt_vis = data.indexToInvdepth(gt)

            vis_results = data.makeVisImage(
                raw_imgs_vis, pred_vis, gt=gt_vis, return_all=True
            )
            input_rgb, pred_rgb, gt_rgb, err_rgb = vis_results

            wandb.log({
                "Eval/Input": wandb.Image(input_rgb),
                "Eval/Prediction": wandb.Image(pred_rgb),
                "Eval/GT": wandb.Image(gt_rgb),
                "Eval/Error": wandb.Image(err_rgb),
            }, step=d)

    LOG_INFO("================== Final Evaluation ==================")
    LOG_INFO('>1: %.3f, >3: %.3f, >5: %.3f, MAE: %.3f, RMS: %.3f'
             % tuple(mean_errors))
    # Skip the first sample: it absorbs cuDNN autotuning and lazy allocations.
    steady = infer_ms[1:] if len(infer_ms) > 1 else infer_ms
    LOG_INFO('Inference: %.1fms mean, %.1fms median (%.1f FPS) | first sample %.1fms'
             % (steady.mean(), np.median(steady),
                1000.0 / steady.mean(), infer_ms[0]))

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
