# module/featurelayer_dual.py
# Dual-head CNN encoder for matching + context features
# Distills DINO knowledge into lightweight CNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.common import *

from module.foundation_stereo_blocks import (
    LayerNorm2d, BasicConv, Conv2x_IN, ResidualBlock,
)
import timm

from torchvision.models import mobilenet_v2

class FeatureStudent_mobilenetv2_deconv(nn.Module):
    """MobileNetV2-based feature extractor with deconvolution (like FeatureStudent)"""
    
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        
        # Load pretrained MobileNetV2
        mobilenet = mobilenet_v2(pretrained=True)
        
        # Extract features at different scales
        # MobileNetV2 structure:
        # features[0:2]: 1/2 scale -> 16 channels
        # features[2:4]: 1/4 scale -> 24 channels
        # features[4:7]: 1/8 scale -> 32 channels
        # features[7:14]: 1/16 scale -> 96 channels
        # features[14:18]: 1/32 scale -> 320 channels
        
        self.layer_x2 = nn.Sequential(*mobilenet.features[:2])   # 1/2 scale
        self.layer_x4 = nn.Sequential(*mobilenet.features[2:4])   # 1/4 scale
        self.layer_x8 = nn.Sequential(*mobilenet.features[4:7])  # 1/8 scale
        self.layer_x16 = nn.Sequential(*mobilenet.features[7:14]) # 1/16 scale
        self.layer_x32 = nn.Sequential(*mobilenet.features[14:18]) # 1/32 scale
        
        # MobileNetV2 channel sizes
        chans = [16, 24, 32, 96, 320]  # x2, x4, x8, x16, x32
        self.chans = chans
        
        # Deconvolution layers (like FeatureStudent)
        self.deconv32_16 = Conv2x_IN(chans[4], chans[3], deconv=True, concat=True)
        self.deconv16_8 = Conv2x_IN(chans[3]*2, chans[2], deconv=True, concat=True)
        self.deconv8_4 = Conv2x_IN(chans[2]*2, chans[1], deconv=True, concat=True)
        self.deconv4_2 = Conv2x_IN(chans[1]*2, chans[0], deconv=True, concat=True)
        
        # Channel information for output
        self.x2_channels = 32   # Output channels at x2 scale
        self.x16_channels = 32  # Output channels at x16 scale
        
        # Projection layers (1x1 convolutions) for x2 and x16
        self.proj_x2 = nn.Conv2d(chans[0]*2, self.x2_channels, kernel_size=1, bias=True)
        self.proj_x16 = nn.Conv2d(chans[3]*2, self.x16_channels, kernel_size=1, bias=True)
        
    def forward(self, x):
        cnn_feats = []
        dino_feats = []
        
        for img in x:
            if img.dim() == 4 and img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)

            # Extract multi-scale features
            x2 = self.layer_x2(img)       # 1/2 scale, 16 channels
            x4 = self.layer_x4(x2)      # 1/4 scale, 24 channels
            x8 = self.layer_x8(x4)      # 1/8 scale, 32 channels
            x16 = self.layer_x16(x8)    # 1/16 scale, 96 channels
            x32 = self.layer_x32(x16)   # 1/32 scale, 320 channels
            
            # Deconvolution with skip connections (like FeatureStudent)
            x16 = self.deconv32_16(x32, x16)  # 96*2 channels
            x8 = self.deconv16_8(x16, x8)     # 32*2 channels
            x4 = self.deconv8_4(x8, x4)       # 24*2 channels
            x2 = self.deconv4_2(x4, x2)       # 16*2 channels
            
            # Apply projections to target channels
            feat_x2 = self.proj_x2(x2)
            feat_x16 = self.proj_x16(x16)
            cnn_feats.append(feat_x2)
            dino_feats.append(feat_x16)

        dino_feats_stacked = torch.stack(dino_feats, dim=1)
        return cnn_feats, dino_feats_stacked

class FeatureStudent_mobilevit(nn.Module):
    """MobileViT-based feature extractor with deconvolution (like FeatureStudent)."""

    def __init__(self, args=None, variant='mobilevitv2_050'):
        super().__init__()
        self.args = args

        # MobileViT backbone with multi-scale features
        self.backbone = timm.create_model(
            variant,
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )

        feature_info = self.backbone.feature_info
        reduction_to_idx = {}
        for idx, info in enumerate(feature_info):
            reduction = info['reduction']
            if reduction in (2, 4, 8, 16, 32) and reduction not in reduction_to_idx:
                reduction_to_idx[reduction] = idx

        missing = [r for r in (2, 4, 8, 16, 32) if r not in reduction_to_idx]
        if missing:
            raise ValueError(f"MobileViT backbone missing reductions: {missing}")

        self.idx_x2 = reduction_to_idx[2]
        self.idx_x4 = reduction_to_idx[4]
        self.idx_x8 = reduction_to_idx[8]
        self.idx_x16 = reduction_to_idx[16]
        self.idx_x32 = reduction_to_idx[32]

        chans = [
            feature_info[self.idx_x2]['num_chs'],
            feature_info[self.idx_x4]['num_chs'],
            feature_info[self.idx_x8]['num_chs'],
            feature_info[self.idx_x16]['num_chs'],
            feature_info[self.idx_x32]['num_chs'],
        ]
        self.chans = chans

        # Deconvolution layers to upscale from x32 -> x16 -> x8 -> x4 -> x2
        self.deconv32_16 = Conv2x_IN(chans[4], chans[3], deconv=True, concat=True)
        self.deconv16_8 = Conv2x_IN(chans[3] * 2, chans[2], deconv=True, concat=True)
        self.deconv8_4 = Conv2x_IN(chans[2] * 2, chans[1], deconv=True, concat=True)
        self.deconv4_2 = Conv2x_IN(chans[1] * 2, chans[0], deconv=True, concat=True)

        # Channel information for output
        self.x2_channels = 32
        self.x16_channels = 32

        # Projection layers (1x1 convolutions) for x2 and x16
        self.proj_x2 = nn.Conv2d(chans[0] * 2, self.x2_channels, kernel_size=1, bias=True)
        self.proj_x16 = nn.Conv2d(chans[3] * 2, self.x16_channels, kernel_size=1, bias=True)

    def forward(self, x):
        cnn_feats = []
        dino_feats = []
        
        for img in x:
            if img.dim() == 4 and img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)

            feats = self.backbone(img)
            x2 = feats[self.idx_x2]
            x4 = feats[self.idx_x4]
            x8 = feats[self.idx_x8]
            x16 = feats[self.idx_x16]
            x32 = feats[self.idx_x32]

            # Deconvolution with skip connections (like FeatureStudent)
            x16 = self.deconv32_16(x32, x16)
            x8 = self.deconv16_8(x16, x8)
            x4 = self.deconv8_4(x8, x4)
            x2 = self.deconv4_2(x4, x2)

            # Apply projections to target channels
            feat_x2 = self.proj_x2(x2)
            feat_x16 = self.proj_x16(x16)

            cnn_feats.append(feat_x2)
            dino_feats.append(feat_x16)

        dino_feats_stacked = torch.stack(dino_feats, dim=1)
        return cnn_feats, dino_feats_stacked

class DistillationLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, student_cnn_feat, teacher_cnn_feat,
                student_dino_feat, teacher_dino_feat):
        
        # 1. CNN final feature distillation
        loss_cnn_final = F.mse_loss(student_cnn_feat, teacher_cnn_feat)
        
        # 2. DINO final feature distillation
        loss_dino = F.mse_loss(student_dino_feat, teacher_dino_feat)

        return loss_cnn_final, loss_dino
