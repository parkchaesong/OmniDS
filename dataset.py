# database.py
#
# Author: Changhee Won (changhee.1.won@gmail.com)
#
# Modified by Hualie Jiang (jianghualie0@gmail.com)

import numpy as np
import random
import scipy.ndimage
from easydict import EasyDict as Edict
import torch
import torch.utils
from torchvision.transforms import ColorJitter, functional, Compose
from PIL import Image
import open3d as o3d

from utils.common import *
from utils.array_utils import *
from utils.geometry import *
from utils.log import *
from utils.ocam import *
from utils.image import *
import utils.dbhelper


def getEquirectCoordinate(pts: np.ndarray, equirect_size: (int, int),
                          phi_deg: float, phi2_deg=-1.0):
    h, w = equirect_size
    d = sqrt((pts**2).sum(0)).reshape((1, -1))
    nx = pts[0, :] / d
    ny = pts[1, :] / d
    nz = pts[2, :] / d
    phi = asin(ny)
    sign = cos(phi)
    sign[sign < 0] = -1.0
    sign[sign >= 0] = 1.0
    theta = atan2(nz * sign, -nx * sign)
    equi_x = ((theta - np.pi / 2) / np.pi + 1) * w / 2
    equi_x[equi_x < 0] += w
    equi_x[equi_x >= w] -= w
    if phi2_deg < 0:
        equi_y = (phi / np.deg2rad(phi_deg) + 1) * h / 2
    else:
        med = np.deg2rad((phi2_deg - phi_deg) / 2)
        med2 = np.deg2rad((phi2_deg - phi_deg) / 2)
        equi_y = ((phi + med2) / med + 1) * h / 2
    return concat([equi_x, equi_y], axis=0)


def makeSphericalRays(equirect_size: (int, int),
                      phi_deg: float, phi2_deg=-1.0) -> np.ndarray:
    h, w = equirect_size
    xs, ys = np.meshgrid(range(w), range(h))  # row major
    w_2, h_2 = w / 2.0, (h - 1) / 2.0
    xs = (xs - w_2) / w_2 * np.pi + (np.pi / 2.0)
    if phi2_deg > 0.0:
        med = np.deg2rad(sum(phi2_deg - phi_deg) / 2.0)
        med2 = np.deg2rad((phi2_deg + phi_deg) / 2.0)
        ys = (ys - h_2) / h_2 * med2 - med
    else:
        ys = (ys - h_2) / h_2 * np.deg2rad(phi_deg)
    
    X = -np.cos(ys) * np.cos(xs)
    Y = np.sin(ys) # sphere
    # Y = np.sin(ys) / np.cos(ys) # cylinder
    # Y = ys / np.deg2rad(phi_deg) # perspective cylinder
    Z = np.cos(ys) * np.sin(xs)
    rays = np.concatenate((np.reshape(X, [1, -1]),
                           np.reshape(Y, [1, -1]),
                           np.reshape(Z, [1, -1]))).astype(np.float64)
    return rays


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dbname: str, db_opts=None, load_lut=True, train=True, db_root='../omnidata'):
        super(torch.utils.data.Dataset, self).__init__()
        self.dbname = dbname.lower()
        self.db_path = osp.join(db_root, self.dbname)
        # Default arguments
        opts = Edict()
        opts.img_fmt = 'cam%d/%04d.png'  # [cam_idx, fidx]
        opts.gt_depth_fmt = 'omnidepth_gt_%d/%05d.tiff'  # [equi_w, fidx]
        opts.equirect_size, opts.num_invdepth = [160, 640], 192
        opts.num_downsample = 1
        opts.phi_deg, opts.phi2_deg = 45, -1.0
        opts.min_depth = 0.5  # meter scale
        opts.max_depth = 1 / EPS
        opts.max_fov = 220.0  # maximum FOV of input fisheye images
        opts.read_input_image = True  # for evaluation, False if read only GT
        opts.start, opts.step, opts.end = 1, 1, 1000  # frame
        opts.train_idx, opts.test_idx = [], []
        opts.gt_phi = 0.0
        opts.dtype = 'nogt'

        # first update opts using pre-defined config 
        # also load ocam parameters
        opts, self.ocams = utils.dbhelper.loadDBConfigs(
            self.dbname, self.db_path, opts)
        # update opts from the argument
        opts.lut_fmt = 'RoS_ds%d_lt_(%d,%d,%d).hwd'  # lookup table fmt [ds, equi_h, w, d]
        opts = argparse(opts, db_opts)

        # set member variables
        self.opts = opts
        self.img_fmt, self.lut_fmt = opts.img_fmt, opts.lut_fmt
        self.gt_depth_fmt = opts.gt_depth_fmt
        self.frame_idx = list(range(
            opts.start, opts.end + opts.step, opts.step))
        self.train_idx, self.test_idx = opts.train_idx, opts.test_idx
        self.gt_phi = opts.gt_phi
        self.dtype = opts.dtype
        self.use_rgb = opts.use_rgb

        self.equirect_size = opts.equirect_size
        self.min_depth, self.max_depth = opts.min_depth, opts.max_depth
        self.max_theta = np.deg2rad(opts.max_fov) / 2.0
        self.phi_deg, self.phi2_deg = opts.phi_deg, opts.phi2_deg
        self.num_invdepth = opts.num_invdepth
        self.num_downsample = opts.num_downsample
        self.read_input_image = opts.read_input_image
        self.data_size = len(self.frame_idx)
        self.train_size = len(self.train_idx)
        self.test_size = len(self.test_idx)
        self.__initSweep(load_lut)
        self.train = train
        self.augmentor = Augmentor()

    # end __init__
    
    def __initSweep(self, load_lut=True):
        self.rays = makeSphericalRays(self.equirect_size,
            self.phi_deg, self.phi2_deg)
        self.min_invdepth = 1.0 / self.max_depth
        self.max_invdepth = 1.0 / self.min_depth
        self.sample_step_invdepth = \
            (self.max_invdepth - self.min_invdepth) / (self.num_invdepth - 1.0)
        self.invdepths = np.arange(
            self.min_invdepth, self.max_invdepth + self.sample_step_invdepth,
            self.sample_step_invdepth, dtype=np.float64)        
        if load_lut: self.__loadOrBuildLookupTable()
    # end __initSweep

    def __loadOrBuildLookupTable(self) -> None:
        h, w = self.equirect_size
        path = osp.join(self.db_path, self.lut_fmt%(self.num_downsample, h, w, self.num_invdepth))
        h, w = h // 2**self.num_downsample, w // 2**self.num_downsample
        if not osp.exists(path):
            LOG_INFO('Lookup table not found: "%s"' % (path))
            LOG_INFO('Build lookup table...')
            self.grids = self.buildLookupTable()
            np.concatenate([toNumpy(g)[np.newaxis, ...] for g in self.grids],
                axis=0).tofile(path)
            LOG_INFO('Lookup table saved: "%s"' % (path))
        else:
            LOG_INFO('Load lookup table: "%s"' % path)
            grids = np.fromfile(path, dtype=np.float32).reshape(
                [4, h, w, int(self.num_invdepth / 2**self.num_downsample), 2])
            self.grids = [grids[i, ...].squeeze() for i in range(4)]
    # end __initSweep

    def __len__(self) -> int:
        if self.train:
            return len(self.train_idx)
        else:
            return len(self.test_idx)

    def __getitem__(self, i):
        if self.train:
            return self.loadTrainSample(i)
        else:
            return self.loadTestSample(i, self.read_input_image)

    def buildLookupTable(self, transform=None, phi_deg=None, phi2_deg=None, output_gpu_tensor=False) -> list:
        num_invdepth = int(self.num_invdepth / 2**self.num_downsample)
        h, w = self.equirect_size
        h, w = h // 2 ** self.num_downsample, w // 2 ** self.num_downsample
        equirect_size = [h, w]

        if phi_deg is None: phi_deg = self.phi_deg
        if phi2_deg is None: phi2_deg = self.phi2_deg
        rays = makeSphericalRays(equirect_size, phi_deg, phi2_deg)
        if output_gpu_tensor:
            grids = [torch.zeros((h, w, num_invdepth, 2),
                                 requires_grad=False).cuda() for _ in range(4)]
        else:
            grids = [np.zeros((h, w, num_invdepth, 2), dtype=np.float32) for _ in range(4)]
        for d in range(num_invdepth):
            depth = 1.0 / self.invdepths[2**self.num_downsample * d]
            pts = depth * rays
            if output_gpu_tensor:
                pts = torch.tensor(pts.astype(np.float32),
                                   requires_grad=False).cuda()
            if transform is not None:
                pts = applyTransform(transform, pts)
            for i in range(4):
                P = applyTransform(self.ocams[i].rig2cam, pts)
                p = self.ocams[i].rayToPixel(P)
                grid = pixelToGrid(p, equirect_size,
                                   (self.ocams[i].height, self.ocams[i].width))
                grid = np.clip(grid, -2, 1)
                if output_gpu_tensor:
                    grids[i][..., d, :] = grid
                else:
                    grids[i][..., d, :] = grid.astype(np.float32)
        return grids

    def loadImages(self, fidx, out_raw_imgs=False, use_rgb=False):
        imgs = []
        raw_imgs = []
        for i in range(4):
            file_path = osp.join(self.db_path, self.img_fmt % (i + 1, fidx))
            I = readImage(file_path)
            if out_raw_imgs: raw_imgs.append(I)
        if fidx in self.train_idx:
            raw_imgs = self.augmentor(raw_imgs)
        for I in raw_imgs:
            if not use_rgb and len(I.shape) == 3 and I.shape[2] == 3: 
                I = rgb2gray(I, channel_wise_mean=True)
            I = normalizeImage(I, self.ocams[i].invalid_mask)
            if len(I.shape) == 2:
                I = np.expand_dims(I, axis=0)  # make 1 x H x W
                if use_rgb: I = np.tile(I, (3, 1, 1))  # make 3 x H x W
            else: I = np.transpose(I, (2, 0, 1))  # make C x H x W
            imgs.append(I)

        if out_raw_imgs: return imgs, raw_imgs
        else: return imgs
    
    def readInvdepth(self, path: str) -> np.ndarray:
        _, ext = osp.splitext(path)
        if ext == '.png':
            step_invdepth = (self.max_invdepth - self.min_invdepth) / 65500.0
            quantized_inv_index = readImage(path).astype(np.float32)
            invdepth = self.min_invdepth + quantized_inv_index * step_invdepth
            return invdepth
        elif ext == '.tif' or ext == '.tiff':
            return readImageFloat(path)
        else:
            return np.fromfile(path, dtype=np.float32)
    
    def writeInvdepth(self, invdepth: np.ndarray, path: str) -> None:
        _, ext = osp.splitext(path)
        if ext == '.png':
            step_invdepth = (self.max_invdepth - self.min_invdepth) / 65500.0
            quantized_inv_index = (invdepth - self.min_invdepth) / step_invdepth
            writeImage(quantized_inv_index.round().astype(np.uint16), path)
        elif ext == '.tif' or ext == '.tiff':
            thumbnail = colorMap('oliver', invdepth,
                self.min_invdepth, self.max_invdepth)
            thumbnail = imrescale(thumbnail, 0.5)
            writeImageFloat(invdepth.astype(np.float32), path, thumbnail)
        else:
            invdepth.astype(np.float32).tofile(path)
    
    def indexToInvdepth(self, idx, start_index=0):
        return self.min_invdepth + \
            (idx - start_index) * self.sample_step_invdepth
    
    def invdepthToIndex(self, inv_depth, start_index=0):
        return (inv_depth - self.min_invdepth) / \
            self.sample_step_invdepth + start_index
    
    def loadGTInvdepthIndex(self, fidx, remove_gt_noise=True,
                            morph_win_size=5):
        h, w = self.equirect_size
        gt_depth_file = osp.join(self.db_path, self.gt_depth_fmt % (w, fidx))
        gt = self.readInvdepth(gt_depth_file)
        gt_h = gt.shape[0]
        # crop height
        if h < gt_h:
            sh = int(round((gt_h - h) / 2.0))
            gt = gt[sh:sh + h, :]

        gt_idx = self.invdepthToIndex(gt)
        if not remove_gt_noise:
            return gt_idx
        # make valid mask
        morph_filter = np.ones(
            (morph_win_size, morph_win_size), dtype=np.uint8)
        finite_depth = gt >= 1e-3 # <= 1000 m
        closed_depth = scipy.ndimage.binary_closing(
            finite_depth, morph_filter) 
        infinite_depth = np.logical_not(finite_depth)
        infinite_hole = np.logical_and(infinite_depth, closed_depth)
        gt_idx[infinite_hole] = -1
        return gt_idx

    def loadSample(self, fidx: int, read_input_image=True, varargin=None):
        opts = Edict()
        opts.remove_gt_noise = True
        opts.morph_win_size = 5
        opts = argparse(opts, varargin)
        imgs, raw_imgs = [], []
        if read_input_image:
            imgs, raw_imgs = self.loadImages(fidx, True, use_rgb=self.use_rgb)
        gt, valid = [], []
        if self.dtype == 'gt':
            gt = self.loadGTInvdepthIndex(fidx, 
                opts.remove_gt_noise, opts.morph_win_size)
            valid = np.logical_and(
                gt >= 0, gt <= self.num_invdepth).astype(np.bool)
        return imgs, gt, valid, raw_imgs

    def loadTrainSample(self, i: int, read_input_image=True, varargin=None):
        return self.loadSample(self.train_idx[i], read_input_image, varargin)
        
    def loadTestSample(self, i: int, read_input_image=True, varargin=None):
        return self.loadSample(self.test_idx[i], read_input_image, varargin)

    def getPanorama(self, imgs, invdepth, transform=None):
        invdepth = toNumpy(invdepth)
        depth = 1 / invdepth.reshape((1, -1))
        P = depth * self.rays
        if transform is not None:
            P = applyTransform(transform, P)
        pano_sum = np.zeros(self.equirect_size)
        valid_count = np.zeros(self.equirect_size, dtype=np.uint8)
        for i in range(4):
            P2 = applyTransform(self.ocams[i].rig2cam, P)
            p, theta = self.ocams[i].rayToPixel(P2, out_theta=True)
            grid = pixelToGrid(p, self.equirect_size,
                (self.ocams[i].height,self.ocams[i].width))
            equi_im = toNumpy(interp2D(imgs[i], grid))
            valid = (theta <= self.ocams[i].max_theta-0.05).reshape(
                self.equirect_size)
            pano_sum[valid] += equi_im[valid]
            valid_count[valid] += 1
        pano = np.round(pano_sum / valid_count).astype(np.uint8)
        return pano

    def getPanorama_rgb(self, imgs, invdepth, transform=None):
        invdepth = toNumpy(invdepth)
        depth = 1 / invdepth.reshape((1, -1))
        P = depth * self.rays
        if transform is not None:
            P = applyTransform(transform, P)
        pano_sum = np.zeros(self.equirect_size+[3])
        valid_count = np.zeros(self.equirect_size+[3], dtype=np.uint8)
        for i in range(4):
            P2 = applyTransform(self.ocams[i].rig2cam, P)
            p, theta = self.ocams[i].rayToPixel(P2, out_theta=True)
            grid = pixelToGrid(p, self.equirect_size,
                (self.ocams[i].height,self.ocams[i].width))
            equi_im = toNumpy(interp2D(imgs[i].transpose(2, 0, 1), grid)).transpose(1, 2, 0)
            valid = (theta <= self.ocams[i].max_theta-0.05).reshape(
                self.equirect_size)
            pano_sum[valid] += equi_im[valid]
            valid_count[valid] += 1
        pano = np.round(pano_sum / valid_count).astype(np.uint8)
        return pano

    def writePointCloud(self, pano_rgb: np.ndarray, invdepth: np.ndarray, path: str) -> None:
        invdepth = toNumpy(invdepth)
        depth = 1 / invdepth.reshape((1, -1))
        val = depth[0] < 50
        P = depth * self.rays
        P = P.transpose()
        color_raw = pano_rgb.reshape((-1, 3))/256
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(P[val, :])
        pcd.colors = o3d.utility.Vector3dVector(color_raw[val, :])
        o3d.io.write_point_cloud(path, pcd)
    
    def makeVisImage(
        self, imgs, invdepth: np.ndarray, gt=None, mask=None, transform=None, return_all=False, train=True
    ):
        import cv2
        import numpy as np
        import torch

        # ------------------------------
        # (768,800,3) 4장을 2x2로 붙이기
        # ------------------------------
        # imgs: list of 4 np.ndarray or torch.Tensor
        np_imgs = []
        for i in range(4):
            img = imgs[i]
            if torch.is_tensor(img):
                img = img.detach().cpu().numpy()
            if img.shape[0] == 1:           # (1,H,W) → (3,H,W)
                img = np.repeat(img, 3, axis=0)
            if img.shape[0] == 3:  # (3,H,W) → (H,W,3)
                img = np.transpose(img, (1, 2, 0))
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 1)
                img = (img * 255).astype(np.uint8)
            np_imgs.append(img)

        inputs_rgb = np.concatenate(
            [np.concatenate([np_imgs[0], np_imgs[1]], axis=1),
            np.concatenate([np_imgs[3], np_imgs[2]], axis=1)], axis=0
        )

        # ------------------------------
        # invdepth / gt 준비
        # ------------------------------
        if train and torch.is_tensor(invdepth):
            invdepth = invdepth.detach().cpu().numpy()
        if gt is not None and torch.is_tensor(gt):
            gt = gt.detach().cpu().numpy()

        invdepth_rgb = None
        gt_rgb = None
        err_rgb = None

        if invdepth is not None:
            invdepth_rgb = colorMap('oliver', invdepth, self.min_invdepth, self.max_invdepth)
        if gt is not None:
            gt_rgb = colorMap('oliver', gt, self.min_invdepth, self.max_invdepth)

        if invdepth is not None and gt is not None:
            err = np.abs(self.invdepthToIndex(invdepth) - self.invdepthToIndex(gt))
            err_rgb = colorMap('jet', err, 0, 10)

        # ------------------------------
        # 크기 맞추기
        # ------------------------------
        target_size = invdepth_rgb.shape[:2] if invdepth_rgb is not None else (160, 640)

        def resize_if_needed(arr, target_size):
            if arr is None:
                return None
            if arr.shape[:2] != target_size:
                return cv2.resize(arr, (target_size[1], target_size[0]))
            return arr

        invdepth_rgb = resize_if_needed(invdepth_rgb, target_size)
        gt_rgb       = resize_if_needed(gt_rgb, target_size)
        err_rgb      = resize_if_needed(err_rgb, target_size)

        # inputs_rgb도 2x2 pano라 크기가 달라서 줄여줌
        inputs_rgb   = resize_if_needed(inputs_rgb, target_size)

        # ------------------------------
        # Concatenate horizontally
        # ------------------------------
        vis_parts = [inputs_rgb]
        if invdepth_rgb is not None:
            vis_parts.append(invdepth_rgb)
        if gt_rgb is not None:
            vis_parts.append(gt_rgb)
        if err_rgb is not None:
            vis_parts.append(err_rgb)

        vis = np.concatenate(vis_parts, axis=1)

        if return_all:
            return inputs_rgb, invdepth_rgb, gt_rgb, err_rgb
        else:
            return vis

    def evalError(self, invdepth_idx, gt, valid) \
                -> (float, float, float, float, float):
        invdepth_idx = toNumpy(invdepth_idx).flatten()
        gt = toNumpy(gt).flatten()
        valid = toNumpy(valid).flatten().astype(bool)
        valid = np.logical_and(valid, np.logical_not(np.isnan(invdepth_idx)))
        nvalid = float(np.sum(valid))
        error = np.abs(invdepth_idx[valid] - gt[valid]) / \
            self.num_invdepth * 100
        e1 = np.sum(error > 1) / nvalid * 100
        e3 = np.sum(error > 3) / nvalid * 100
        e5 = np.sum(error > 5) / nvalid * 100
        mae = np.nanmean(error)
        rms = np.sqrt(np.nanmean(error**2))
        return e1, e3, e5, mae, rms

    def evalErrorThreshold(self, invdepth_idx, gt, valid, entropy, k) \
                -> (float, float, float, float, float, float):
        invdepth_idx = toNumpy(invdepth_idx).flatten()
        gt = toNumpy(gt).flatten()
        valid = toNumpy(valid).flatten().astype(bool)
        entropy = toNumpy(entropy).flatten()

        valid = np.logical_and(valid, np.logical_not(np.isnan(invdepth_idx)))
        nvalid = float(np.sum(valid))

        certain = np.logical_and(valid, entropy <= np.log(k))
        ncertain = float(np.sum(certain))
        completeness = ncertain / nvalid * 100

        error = np.abs(invdepth_idx[certain] - gt[certain]) / \
            self.num_invdepth * 100.0
        e1 = np.sum(error > 1) / ncertain * 100
        e3 = np.sum(error > 3) / ncertain * 100
        e5 = np.sum(error > 5) / ncertain * 100
        mae = np.nanmean(error)
        rms = np.sqrt(np.nanmean(error**2))
        return e1, e3, e5, mae, rms, completeness


class MultiDataset(Dataset):
    # !! share all opts among datasets
    def __init__(self, dbnames: list, db_opts=None, load_lut=True, train=True,
                 db_root = '../omnidata'):
        super().__init__(dbnames[0], db_opts, load_lut, train, db_root)
        self.dbnames = dbnames
        self.datasets = [Dataset(dbname, db_opts, False, train, db_root) \
                for dbname in dbnames[1:]]
        self.num_datasets = len(self.datasets) + 1
        self.frame_idx_offsets = np.zeros((self.num_datasets))
        self.train_idx_offsets = np.zeros((self.num_datasets))
        self.test_idx_offsets = np.zeros((self.num_datasets))
        for i, db in enumerate(self.datasets):
            self.frame_idx_offsets[i + 1] = self.data_size
            self.train_idx_offsets[i + 1] = self.train_size
            self.test_idx_offsets[i + 1] = self.test_size
            self.frame_idx += db.frame_idx
            self.train_idx += db.train_idx
            self.test_idx += db.test_idx
            self.train_size += db.train_size
            self.test_size += db.test_size
            self.data_size += db.data_size
        LOG_INFO('Multiple datasets are initialized (they share configs)')
    
    def __findDataIndex(self, sample_idx: int, offsets: list) -> (int, int):
        for i in range(self.num_datasets - 1):
            if sample_idx >= offsets[i] and sample_idx < offsets[i+1]:
                return i
        return self.num_datasets - 1 

    def __len__(self) -> int:
        if self.train:
            return len(self.train_idx)
        else:
            return len(self.test_idx)

    def __getitem__(self, i: int):
        if self.train: return self.loadTrainSample(i)
        else: return self.loadTestSample(i, self.read_input_image)

    def loadSample(self, i: int, read_input_image=True, varargin=None):
        didx = self.__findDataIndex(i, self.frame_idx_offsets)
        sample_idx = self.frame_idx[i]
        if didx == 0:
            return super().loadSample(sample_idx, read_input_image, varargin)
        else:
            return self.datasets[didx - 1].loadSample(
                sample_idx, read_input_image, varargin)

    def loadTrainSample(self, i: int, read_input_image=True, varargin=None):
        didx = self.__findDataIndex(i, self.train_idx_offsets)
        sample_idx = self.train_idx[i]
        if didx == 0:
            return super().loadSample(
                sample_idx, read_input_image, varargin)
        else:
            return self.datasets[didx - 1].loadSample(
                sample_idx, read_input_image, varargin)
        
    def loadTestSample(self, i: int, read_input_image=True, varargin=None):
        didx = self.__findDataIndex(i, self.test_idx_offsets)
        sample_idx = self.test_idx[i]
        if didx == 0:
            return super().loadSample(
                sample_idx, read_input_image, varargin)
        return self.datasets[didx - 1].loadSample(
            sample_idx, read_input_image, varargin)

    def splitDataset(self, dbname: str) -> Dataset:
        didx = self.dbnames.index(dbname.lower())
        if didx == 0:
            dataset = self
            dataset.__class__ = Dataset
            return dataset
        else:
            return self.datasets[didx - 1]


class AdjustGamma(object):

    def __init__(self, gamma_min, gamma_max, gain_min=1.0, gain_max=1.0):
        self.gamma_min, self.gamma_max, self.gain_min, self.gain_max = gamma_min, gamma_max, gain_min, gain_max
    def __call__(self, sample):
        gain = random.uniform(self.gain_min, self.gain_max)
        gamma = random.uniform(self.gamma_min, self.gamma_max)
        return functional.adjust_gamma(sample, gamma, gain)
    def __repr__(self):
        return f"Adjust Gamma {self.gamma_min}, ({self.gamma_max}) and Gain ({self.gain_min}, {self.gain_max})"


class Augmentor:
    def __init__(self, saturation_range=[0.6, 1.4], gamma=[1, 1, 1, 1]):

        # photometric augmentation params
        self.photo_aug = Compose(
            [ColorJitter(brightness=0.4, contrast=0.4, saturation=saturation_range, hue=0.5 / 3.14),
             AdjustGamma(*gamma)])
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.2

    def color_transform(self, raw_images):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            images = [np.array(self.photo_aug(Image.fromarray(img)), dtype=np.uint8) for img in raw_images]
        # symmetric
        else:
            image_stack = np.concatenate(raw_images, axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            images = list(np.split(image_stack, 4, axis=0))

        return images

    def eraser_transform(self, images, bounds=[50, 100]):
        """ Occlusion augmentation """

        ht, wd = images[0].shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            for i in [1, 3]:
                mean_color = np.mean(images[i].reshape(-1, 3), axis=0)
                for _ in range(np.random.randint(1, 3)):
                    x0 = np.random.randint(0, wd)
                    y0 = np.random.randint(0, ht)
                    dx = np.random.randint(bounds[0], bounds[1])
                    dy = np.random.randint(bounds[0], bounds[1])
                    images[i][y0:y0 + dy, x0:x0 + dx, :] = mean_color

        return images


    def __call__(self, raw_images):
        images = self.color_transform(raw_images)
        images = self.eraser_transform(images)
        return images
