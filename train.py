# train.py
# Training script for OmniDS

from __future__ import print_function, division

import time
from argparse import ArgumentParser
import wandb

# Torch libs
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="skimage")

try:
    from torch.cuda.amp import GradScaler
except:
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

# Internal modules
from dataset import Dataset, MultiDataset
from utils.common import *
from utils.image import *
from module.network import OmniDS
from module.loss_functions import sequence_loss

# Initialize
torch.backends.cudnn.benchmark = True
torch.backends.cuda.benchmark = True

parser = ArgumentParser(description='Training for OmniDS')

parser.add_argument('--name', default='ROmniStereo_BEV', help="name of your experiment")
parser.add_argument('--restore_ckpt', help="restore checkpoint")
parser.add_argument('--pretrain_ckpt', help="pretrained checkpoint for finetuning")

parser.add_argument('--db_root', default='/home/work/jh_code/data', type=str, help='path to dataset')
parser.add_argument('--dbname', nargs='+', default=['omnithings'], type=str,
                    choices=['omnithings', 'omnihouse', 'sunny', 'cloudy', 'sunset'], help='databases to train')

# data options
parser.add_argument('--phi_deg', type=float, default=45.0, help='phi_deg')
parser.add_argument('--num_invdepth', type=int, default=192, metavar='N', help='number of disparity')
parser.add_argument('--equirect_size', type=int, nargs='+', default=[160, 640], help="size of out ERP.")
parser.add_argument('--use_rgb', action='store_true', help='use 3-channel rgb color images as input')

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
                    choices=['variance', 'correlation', 'pairwise', 'adaptive'], help="similarity computation method")
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
parser.add_argument('--no_dino_cross_attn', action='store_true',
                    help='ablation: use fixed grid sampling instead of deformable cross-attention for DINO→ERP')
parser.add_argument('--use_dino_sim', action='store_true',
                    help='use DINO features (instead of CNN) for similarity volume construction')

# GEV options (separate GEV with 3D UNet)
parser.add_argument('--use_gev', action='store_true',
                    help='deprecated no-op; the GEV branch is always used')
parser.add_argument('--gev_num_groups', type=int, default=8,
                    help='number of groups for group-wise correlation')
parser.add_argument('--gev_reg_channels', type=int, nargs='+', default=[16, 32, 48],
                    help='channels for 3D UNet regularization (3 stages)')
parser.add_argument('--gev_num_pyramid_levels', type=int, default=2,
                    help='number of depth-axis pyramid levels for GEV lookup')
parser.add_argument('--gev_radius', type=int, default=4,
                    help='lookup radius for GEV depth-axis sampling')
parser.add_argument('--gev_use_spatial_downsample', action='store_true', default=True,
                    help='spatial 2x downsample before 3D UNet')

parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--fix_bn', action='store_true', help='fix batch normalization')
parser.add_argument('--distilled', action='store_true',
                    help='build and use the distilled student encoder')
parser.add_argument('--triton', action='store_true',
                    help='use Triton fused kernels for GEV/similarity')

# training options
parser.add_argument('--total_epochs', type=int, default=100, help='total epochs of training')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--train_iters', type=int, default=12,
                    help="number of updates to the disparity field in each forward pass.")
parser.add_argument('--lr', type=float, default=0.0005, help="max learning rate.")
parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")
parser.add_argument('--wandb', action='store_true', help='log training to wandb')
    
args = parser.parse_args()

opts = Edict()
# Dataset & sweep arguments
opts.name = args.name
opts.model_dir = os.path.join('./checkpoints', args.name)

opts.snapshot_path = args.restore_ckpt
opts.pretrain_path = args.pretrain_ckpt
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
opts.net_opts.no_dino_cross_attn = args.no_dino_cross_attn
opts.net_opts.use_dino_sim = args.use_dino_sim

# GEV options (v5: separate GEV with 3D UNet)
opts.net_opts.gev_num_groups = args.gev_num_groups
opts.net_opts.gev_reg_channels = tuple(args.gev_reg_channels)
opts.net_opts.gev_num_pyramid_levels = args.gev_num_pyramid_levels
opts.net_opts.gev_radius = args.gev_radius
opts.net_opts.gev_use_spatial_downsample = args.gev_use_spatial_downsample

# Distillation
opts.distilled = args.distilled
opts.net_opts.use_student = args.distilled
opts.net_opts.use_triton = args.triton

opts.total_epochs = args.total_epochs
opts.batch_size = args.batch_size
opts.train_iters = args.train_iters
opts.lr = args.lr
opts.wdecay = args.wdecay


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(model, num_steps):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=opts.lr, weight_decay=opts.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, opts.lr, num_steps+100,
                                              pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


def train(epoch_total, load_state):
    if len(opts.dbname) > 1:
        data = MultiDataset(opts.dbname, opts.data_opts, db_root=opts.db_root)
    else:
        data = Dataset(opts.dbname[0], opts.data_opts, db_root=opts.db_root)
    dbloader = torch.utils.data.DataLoader(data, batch_size=opts.batch_size,
                                           pin_memory=True, shuffle=True,
                                           num_workers=0, drop_last=True)
    total_num_steps = len(data)*opts.total_epochs//opts.batch_size

    LOG_INFO("Using OmniDS (DINO context + separate GEV)")
    LOG_INFO(f"  DINO model: {opts.net_opts.dino_model}, freeze: {opts.net_opts.freeze_dino}")
    LOG_INFO(f"  GEV groups: {opts.net_opts.gev_num_groups}, reg_channels: {opts.net_opts.gev_reg_channels}")
    LOG_INFO(f"  GEV pyramid: {opts.net_opts.gev_num_pyramid_levels} levels, radius {opts.net_opts.gev_radius}")
    net = nn.DataParallel(OmniDS(opts.net_opts)).cuda()
    if opts.net_opts.fix_bn:
        net.module.freeze_bn()
    LOG_INFO("Parameter Count: %d" % count_parameters(net))

    optimizer, scheduler = fetch_optimizer(net, total_num_steps)
    scaler = GradScaler(enabled=opts.net_opts.mixed_precision)

    start_epoch = 0
    if args.wandb:
        wandb.init(
            entity="entity",   
            project="Videodepth",                
            settings=wandb.Settings(start_method="fork")
        )

    if load_state:
        if opts.snapshot_path and osp.exists(opts.snapshot_path):
            snapshot = torch.load(opts.snapshot_path)
            if 'net_state_dict' in snapshot.keys():
                net.load_state_dict(snapshot['net_state_dict'])
                LOG_INFO('checkpoint %s is loaded' % (opts.snapshot_path))
            if 'epoch' in snapshot.keys():
                start_epoch = snapshot['epoch'] + 1
            if 'epoch_loss' in snapshot.keys():
                epoch_loss = snapshot['epoch_loss']
            if 'optimizer' in snapshot.keys():
                optimizer.load_state_dict(snapshot['optimizer'])
            if 'epoch' in snapshot.keys() and 'epoch_loss' in snapshot.keys():
                LOG_INFO('startepoch:%d epoch_loss:%f' % (start_epoch, epoch_loss))
        elif opts.pretrain_path is None:
            sys.exit('%s do not exsits' % (opts.snapshot_path))

        if opts.pretrain_path and osp.exists(opts.pretrain_path):
            snapshot = torch.load(opts.pretrain_path)
            if 'net_state_dict' in snapshot.keys():
                net.load_state_dict(snapshot['net_state_dict'])
                LOG_INFO('checkpoint %s is loaded' % (opts.pretrain_path))
        elif opts.snapshot_path is None:
            sys.exit('%s do not exsits' % (opts.snapshot_path))

    grids = [torch.tensor(grid, requires_grad=False).cuda() for grid in data.grids]
    if not osp.exists(opts.model_dir):
        os.makedirs(opts.model_dir, exist_ok=True)
        LOG_INFO('"%s" directory created' % (opts.model_dir))
    total_iters = len(data)*start_epoch//opts.batch_size

    for epoch in range(start_epoch, epoch_total):
        # training
        net.train()
        train_loss = 0
        epoch_loss = 0
        LOG_INFO('\nEpoch: %d' % epoch)
        if args.wandb:
            wandb.log({"Epoch": epoch}, step=total_iters)

        for step, data_blob in enumerate(dbloader):
            start_time = time.time()
            imgs, gt, valid, raw_imgs = data_blob

            imgs = [img.cuda() for img in imgs]
            valid = valid.cuda()
            gt = gt.cuda()

            optimizer.zero_grad()
            # Pass ocams for principal point bias (if available)
            ocams = getattr(data, 'ocams', None)
            predictions = net.module.forward(imgs, grids, ocams=ocams)

            loss = sequence_loss(predictions, gt.unsqueeze(1), valid.unsqueeze(1))
            
            train_loss += loss.data
            # train_loss += loss2.data *0.5
            # total_loss = loss + 0.5 * loss2

            epoch_loss = train_loss / (step + 1)

            if step % 200 == 0:
                LOG_INFO("Iter %d/%d training loss = %.3f, average training loss for every step = %.3f, time = %.2f" % (total_iters - epoch*len(data),len(data), loss, epoch_loss, time.time() - start_time))
            # writer.add_scalar("train/loss", loss, total_iters)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            total_iters += 1

            if args.wandb and total_iters % 300 == 0:
                wandb.log({
                    "LR": scheduler.get_last_lr()[0],
                    "Train/total_loss": loss.item(),
                }, step=total_iters)
                
                # === VIS inputs: 반드시 detach + cpu ===
                raw_imgs_vis = [img[0].detach().cpu() for img in raw_imgs]  # [768,800,3] x4
                pred_vis = data.indexToInvdepth(predictions[-1][0, 0]).detach().cpu()            # [160,640] (마지막 prediction 권장)

                if torch.is_tensor(gt):
                    gt_vis = data.indexToInvdepth(gt[0]).detach().cpu() if gt.dim() == 3 else data.indexToInvdepth(gt).detach().cpu()
                else:
                    gt_vis = data.indexToInvdepth(gt[0]).detach().cpu()  # gt가 list라면

                vis_results = data.makeVisImage(raw_imgs_vis, pred_vis, gt=gt_vis, return_all=True)
                wimages = {"Train/Predictions": []}
                input_rgb, pred_rgb, gt_rgb, err_rgb = vis_results # unpack each tuple/list

                images = {
                    f"Sample /Input": input_rgb,
                    f"Sample /Prediction": pred_rgb,
                    f"Sample /GT": gt_rgb,
                    f"Sample /Error": err_rgb
                }
                wimages["Train/Predictions"].extend(
                    [wandb.Image(img, caption=caption) for caption, img in images.items()]
                )
                wandb.log(wimages, step=total_iters)
    
        # evaluation
        net.eval()
        eval_list = data.opts.test_idx

        rank0 = (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)
        net_eval = net.module if hasattr(net, "module") else net  # DDP면 원본 모듈로 eval

        if rank0:
            errors = np.zeros((len(eval_list), 5))
            for d in range(len(eval_list)):
                fidx = eval_list[d]
                imgs, gt, valid, raw_imgs = data.loadSample(fidx)
                imgs = [torch.Tensor(img).unsqueeze(0).cuda() for img in imgs]
                # breakpoint()
                # Pass ocams for LoRA adaptation (same as training)
                ocams = getattr(data, 'ocams', None)
                with torch.no_grad():
                    invdepth_idx = net_eval(imgs, grids, test_mode=True)

                invdepth_idx = toNumpy(invdepth_idx[0, 0])

                # Compute errors
                errors[d, :] = data.evalError(invdepth_idx, gt, valid)
                mean_errors = errors[:d+1].mean(axis=0) 

                # LOG_INFO('Iter %d/%d >1: %.3f, >3: %.3f, >5: %.3f, MAE: %.3f, RMS: %.3f Time: %.2f' %
                #         (d, len(eval_list), mean_errors[0], mean_errors[1], mean_errors[2], mean_errors[3], mean_errors[4],time.time() - start_time_eval))
    
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        
        mean_errors = errors.mean(axis=0)
        LOG_INFO('>1: %.3f, >3: %.3f, >5: %.3f, MAE: %.3f, RMS: %.3f' %
            (mean_errors[0], mean_errors[1], mean_errors[2], mean_errors[3], mean_errors[4]))

        # logging
        if args.wandb:
            losses = dict()
            losses["val1"] = mean_errors[0]
            losses["val3"] = mean_errors[1]
            losses["val5"] = mean_errors[2]
            losses["val_mae"] = mean_errors[3]
            losses["val_rms"] = mean_errors[4]

            wandb.log({f"Eval/{name}": loss.item()
            for name, loss in losses.items()}, step=total_iters)

            # === VIS inputs: 반드시 detach + cpu ===
            raw_imgs_vis = [img[0].detach().cpu() for img in imgs]  # [768,800,3] x4
            # breakpoint()
            pred_vis = data.indexToInvdepth(invdepth_idx)           # [160,640] (마지막 prediction 권장)

            if torch.is_tensor(gt):
                gt_vis = data.indexToInvdepth(gt).detach().cpu() if gt.dim() == 3 else data.indexToInvdepth(gt).detach().cpu()
            else:
                gt_vis = data.indexToInvdepth(gt)  # gt가 list라면

            vis_results = data.makeVisImage(raw_imgs_vis, pred_vis, gt=gt_vis, return_all=True)
            wimages = {"Eval/Predictions": []}
            input_rgb, pred_rgb, gt_rgb, err_rgb = vis_results

            images = {
                f"Sample /Input": input_rgb,
                f"Sample /Prediction": pred_rgb,
                f"Sample /GT": gt_rgb,
                f"Sample /Error": err_rgb
            }
            wimages["Eval/Predictions"].extend(
                [wandb.Image(img, caption=caption) for caption, img in images.items()]
            )
            wandb.log(wimages, step=total_iters)
        
        # save
        savefilename = opts.model_dir + '/%s_e%d.pth' % (osp.basename(opts.name), epoch)
        torch.save({
                'net_state_dict': net.state_dict(),
                'net_opts': opts.net_opts,
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'epoch_loss': epoch_loss,
            }, savefilename)


def main():
    load_state = opts.snapshot_path is not None or opts.pretrain_path is not None
    train(opts.total_epochs, load_state)


if __name__ == "__main__":
    main()

