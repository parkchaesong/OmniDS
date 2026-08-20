# module/network.py
# OmniDS: omnidirectional depth from 4 fisheye views.
# CNN matching + DINO context + a 3D-UNet Geometry Encoding Volume,
# aggregated to ERP by BEVFormer-style deformable cross-attention.
# Author: Adapted from ROmniStereo and BEVFormer

import torch
import torch.nn as nn
import torch.nn.functional as F
from module.featurelayer import FeatureLayers, Conv2D
from module.erp_cross_attention import IterativeERPExtractor
from module.update import ConvGRU, DepthHead
from module.similarity_context import SimilarityContext, GEVModule
from module.sweep import spherical_sweep
from utils.common import *

try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class DINOFeatureExtractor(nn.Module):
    """
    DINO feature extractor using DINOv3 via timm.
    
    Supported models:
    - DINOv3 (patch 16): 'dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16'
    
    Reference: https://huggingface.co/timm/vit_small_plus_patch16_dinov3.lvd1689m
    """
    def __init__(self, model_name='dinov3_vits16', output_dim=32, freeze=True):
        super().__init__()
        self.model_name = model_name
        self.output_dim = output_dim
        self.freeze = freeze
        
        try:
            import timm
        except ImportError:
            raise ImportError("Please install timm: pip install timm")
        
        # Map model names to timm model names
        timm_model_map = {
            'dinov3_vits16': 'vit_small_plus_patch16_dinov3.lvd1689m',
            'dinov3_vitb16': 'vit_base_patch16_dinov3.lvd1689m',
            'dinov3_vitl16': 'vit_large_patch16_dinov3.lvd1689m',
        }
        
        if model_name not in timm_model_map:
            raise ValueError(f"Unknown model: {model_name}. Supported: {list(timm_model_map.keys())}")
        
        timm_name = timm_model_map[model_name]
        print(f"Loading DINOv3 via timm: {timm_name}")
        
        # Get DINO output dimension
        if 'vits' in model_name:
            self.dino_dim = 384
        elif 'vitb' in model_name:
            self.dino_dim = 768
        elif 'vitl' in model_name:
            self.dino_dim = 1024
        else:
            self.dino_dim = 384
        
        # Create model with features_only=True for direct feature map output
        self.dino = timm.create_model(
            timm_name,
            pretrained=True,
            features_only=True,  # Returns feature maps directly [B, C, H, W]
        )
        print(f"Loaded DINOv3 model: {timm_name}")
        
        # Projection layer to match CNN feature dimension
        self.proj = nn.Sequential(
            nn.Conv2d(self.dino_dim, output_dim * 2, 1),
            nn.BatchNorm2d(output_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim * 2, output_dim, 3, padding=1),
            nn.BatchNorm2d(output_dim),
        )
        
        if freeze:
            self._freeze_dino()
    
    def _freeze_dino(self):
        """Freeze DINO backbone weights."""
        for param in self.dino.parameters():
            param.requires_grad = False
        self.dino.eval()
    
    def train(self, mode=True):
        """Override train to keep DINO frozen."""
        super().train(mode)
        if self.freeze:
            self.dino.eval()
        return self
    
    def forward(self, x, target_size=None):
        """
        Extract DINO features.
        
        Args:
            x: [B, C, H, W] - input image (RGB)
            target_size: (H, W) - target feature map size
        Returns:
            feat: [B, output_dim, H_out, W_out]
        """
        B, C, H, W = x.shape
        # Get feature maps (features_only=True returns list of feature maps)
        if target_size is None:
            H_out, W_out = H // 2, W // 2
        else:
            H_out, W_out = target_size

        with torch.no_grad() if self.freeze else torch.enable_grad():
            features = self.dino(x)  # List of [B, C, H, W] feature maps
            feat = features[-1]  # Use last feature map
        
        # Project to target dimension
        feat = self.proj(feat)
        
        # Resize to target size if specified
        feat = F.interpolate(feat, size=(H_out, W_out), mode='bilinear', align_corners=False)
        
        return feat


class DINOERPExtractor(nn.Module):
    """
    Extract DINO features from fisheye cameras and project to ERP space using cross-attention.
    Same structure as CNN feature extraction for consistency.
    Includes distortion-aware positional embedding for better texture handling at image edges.
    
    Usage:
        1. extract_features(imgs) - Run DINO backbone once
        2. forward(dino_feats, grids, inv_depth_idx) - Cross-attention (can be called multiple times)
    """
    def __init__(self, dino_model_name='dinov3_vits16', output_dim=32,
                 equi_h=80, equi_w=320, num_cams=4, freeze_dino=True,
                 num_heads=4, num_points=4, num_cross_attn_layers=1,
                 fish_h=384, fish_w=400, no_cross_attn=False):
        super().__init__()
        self.equi_h = equi_h
        self.equi_w = equi_w
        self.num_cams = num_cams
        self.output_dim = output_dim
        self.fish_h = fish_h
        self.fish_w = fish_w

        # DINO feature extractor (shared for all cameras)
        self.dino_backbone = DINOFeatureExtractor(
            model_name=dino_model_name,
            output_dim=output_dim,
            freeze=freeze_dino,
        )

        # DINO feature extractor is used directly without LoRA adaptation.

        # Distortion-aware positional embedding
        # Encodes distance from image center and distortion angle
        # self.distortion_embed = nn.Sequential(
        #     nn.Linear(2, output_dim // 4),  # (normalized_distance, normalized_theta)
        #     nn.ReLU(inplace=True),
        #     nn.Linear(output_dim // 4, output_dim // 2),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(output_dim // 2, output_dim)
        # )

        # Cross-attention to project DINO features to ERP space
        # When no_cross_attn=True, use fixed grid sampling instead (ablation baseline)
        if no_cross_attn:
            self.erp_cross_attn = None
        else:
            self.erp_cross_attn = IterativeERPExtractor(
                embed_dims=output_dim,
                equi_h=equi_h,
                equi_w=equi_w,
                num_cams=num_cams,
                num_heads=num_heads,
                num_points=num_points,
                num_layers=num_cross_attn_layers,
                use_geom_bias=True
            )

    def extract_features(self, imgs):
        """
        Extract DINO features from all cameras (run once).
        
        Args:
            imgs: list of 4 fisheye images [B, C, H, W]
        Returns:
            dino_feats_stacked: [B, num_cams, C, H, W]
        """
        dino_feats = []
        for img in imgs:
            # Ensure 3 channels for DINO
            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)
            
            # Extract features (no target_size, keep original resolution)
            feat = self.dino_backbone(img, target_size=None)
            dino_feats.append(feat)
        
        # Stack features: [B, num_cams, C, H, W]
        dino_feats_stacked = torch.stack(dino_feats, dim=1)
        return dino_feats_stacked
    
    def _sample_dino_grid(self, dino_feats_stacked, grids_tensor, inv_depth_idx):
        """Fixed grid sampling fallback for DINO→ERP projection (ablation baseline).

        Uses the same LUT-based approach as ``_sample_cnn_erp`` but with a simple
        average over cameras (no learned view weights) for a clean ablation.

        Args:
            dino_feats_stacked: [B, num_cams, C, H_dino, W_dino]
            grids_tensor: list of num_cams tensors, each [H, W, D, 2]
            inv_depth_idx: [B, 1, H, W]
        Returns:
            dino_erp: [B, C, H, W]
        """
        B, num_cams, C, _, _ = dino_feats_stacked.shape
        H, W = inv_depth_idx.shape[2], inv_depth_idx.shape[3]
        device = inv_depth_idx.device
        dtype = dino_feats_stacked.dtype
        idx_flat = inv_depth_idx.view(B, -1)  # [B, H*W]

        per_cam = []
        for cam_idx, grid_lut in enumerate(grids_tensor):
            grid_lut = grid_lut.to(device=device, dtype=dtype)  # [H, W, D, 2]
            D = grid_lut.shape[2]

            idx_floor = idx_flat.long().clamp(0, D - 1)
            idx_ceil = (idx_floor + 1).clamp(0, D - 1)
            w = (idx_flat - idx_floor.float()).unsqueeze(-1)  # [B, H*W, 1]

            grid_flat = grid_lut.view(-1, D, 2).unsqueeze(0).expand(B, -1, -1, -1)
            idx_floor_exp = idx_floor.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
            idx_ceil_exp = idx_ceil.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)

            g0 = torch.gather(grid_flat, 2, idx_floor_exp).squeeze(2)
            g1 = torch.gather(grid_flat, 2, idx_ceil_exp).squeeze(2)
            grid = ((1 - w) * g0 + w * g1).view(B, H, W, 2)

            feat = F.grid_sample(
                dino_feats_stacked[:, cam_idx], grid,
                mode='bilinear', padding_mode='zeros', align_corners=True,
            )
            per_cam.append(feat)

        return torch.stack(per_cam, dim=0).mean(dim=0)  # [B, C, H, W]

    def forward(self, dino_feats_stacked, reference_points, inv_depth_idx,
                grids_tensor=None):
        """
        Project DINO features to ERP using cross-attention (can be called every iteration).
        When ``self.erp_cross_attn is None`` (ablation), falls back to fixed grid sampling.

        Args:
            dino_feats_stacked: [B, num_cams, C, H, W] - pre-extracted DINO features
            reference_points: [B, H*W, num_cams, 2] - reference points from grids (normalized [0, 1])
            inv_depth_idx: [B, 1, H, W] - current depth estimate
            grids_tensor: (optional) list of num_cams LUT tensors [H, W, D, 2], required when
                          using grid sampling mode (no_cross_attn)
        Returns:
            dino_erp: [B, output_dim, H, W]
        """
        if self.erp_cross_attn is None:
            return self._sample_dino_grid(dino_feats_stacked, grids_tensor, inv_depth_idx)

        # Cross-attention to ERP space with distortion embedding added to value
        dino_erp = self.erp_cross_attn(
            dino_feats_stacked, reference_points, inv_depth_idx
        )
        return dino_erp


# --------------------------------------------------------------------
# Fuse DINO(context) + CNN(detail) in ERP space
# Multiple fusion strategies available for speed vs quality tradeoff
# --------------------------------------------------------------------

class ERPContextFusionPointwise(nn.Module):
    """
    Fast pointwise fusion: O(N) complexity.
    Same spatial location fused via 1x1 conv - semantically correct for ERP alignment.
    ~400x faster than full attention.
    """
    def __init__(self, embed_dims: int, hidden_dims: int = 64):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dims * 2, hidden_dims, 1),
            nn.GroupNorm(8, hidden_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dims, embed_dims, 1),
        )

    def forward(self, dino_erp: torch.Tensor, cnn_erp: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] concat -> [B, 2C, H, W] -> [B, C, H, W]
        combined = torch.cat([dino_erp, cnn_erp], dim=1)
        return dino_erp + self.fuse(combined)

class ERPContextFusionGated(nn.Module):
    """
    Gated fusion: Adaptive per-pixel weighting between DINO and CNN.
    O(N) complexity, ~400x faster than full attention.
    """
    def __init__(self, embed_dims: int, hidden_dims: int = 32):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(embed_dims * 2, hidden_dims, 3, padding=1),
            nn.GroupNorm(8, hidden_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dims, 1, 1),
            nn.Sigmoid(),
        )
        # Feature transform before gating
        self.dino_proj = nn.Conv2d(embed_dims, embed_dims, 1)
        self.cnn_proj = nn.Conv2d(embed_dims, embed_dims, 1)

    def forward(self, dino_erp: torch.Tensor, cnn_erp: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([dino_erp, cnn_erp], dim=1))  # [B, 1, H, W]
        dino_feat = self.dino_proj(dino_erp)
        cnn_feat = self.cnn_proj(cnn_erp)
        return gate * dino_feat + (1 - gate) * cnn_feat

class ERPContextFusionLocalConv(nn.Module):
    """
    Local conv fusion: Pointwise + local spatial context.
    O(N) complexity with 3x3 receptive field.
    """
    def __init__(self, embed_dims: int, hidden_dims: int = 64):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dims * 2, hidden_dims, 1),
            nn.GroupNorm(8, hidden_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dims, hidden_dims, 3, padding=1, groups=hidden_dims),  # depthwise
            nn.Conv2d(hidden_dims, embed_dims, 1),  # pointwise
        )

    def forward(self, dino_erp: torch.Tensor, cnn_erp: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([dino_erp, cnn_erp], dim=1)
        return dino_erp + self.fuse(combined)

class ERPContextFusionAttention(nn.Module):
    """
    Original full attention fusion: O(N²) complexity.
    Most powerful but SLOW - use only if quality is critical.
    """
    def __init__(self, embed_dims: int, num_heads: int = 4, ffn_dims: int = 128, dropout: float = 0.0):
        super().__init__()
        import inspect
        mha_kwargs = dict(embed_dim=embed_dims, num_heads=num_heads, dropout=dropout)
        if 'batch_first' in inspect.signature(nn.MultiheadAttention).parameters:
            mha_kwargs['batch_first'] = True
            self._batch_first = True
        else:
            self._batch_first = False
        self.attn = nn.MultiheadAttention(**mha_kwargs)
        self.norm1 = nn.LayerNorm(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, ffn_dims),
            nn.ReLU(inplace=True),
            nn.Linear(ffn_dims, embed_dims),
        )
        self.norm2 = nn.LayerNorm(embed_dims)

    def forward(self, dino_erp: torch.Tensor, cnn_erp: torch.Tensor) -> torch.Tensor:
        B, C, H, W = dino_erp.shape
        if self._batch_first:
            q = dino_erp.flatten(2).permute(0, 2, 1)
            k = cnn_erp.flatten(2).permute(0, 2, 1)
            v = k
            attn_out, _ = self.attn(q, k, v, need_weights=False)
            x = self.norm1(q + attn_out)
            x = self.norm2(x + self.ffn(x))
            return x.permute(0, 2, 1).view(B, C, H, W)

        q = dino_erp.flatten(2).permute(2, 0, 1)
        k = cnn_erp.flatten(2).permute(2, 0, 1)
        v = k
        attn_out, _ = self.attn(q, k, v, need_weights=False)
        x = self.norm1(q + attn_out)
        x = self.norm2(x + self.ffn(x))
        x = x.permute(1, 2, 0).contiguous().view(B, C, H, W)
        return x


class MotionEncoder(nn.Module):
    """
    Motion encoder with separate similarity, DINO and GEV inputs.
    sim(36) → 64, dino(32) → 64, gev(18) → 64, depth(1) → 32
    concat: 224 → 127 + depth(1) = 128
    """
    def __init__(self, sim_context_dims, dino_dims, gev_dims, hidden_channels=128):
        super().__init__()
        # Similarity context encoder (matching signal from corr branch)
        self.conv_sim1 = nn.Conv2d(sim_context_dims, 64, 1)
        self.conv_sim2 = nn.Conv2d(64, 64, 3, padding=1)

        # DINO feature encoder (scene context)
        self.conv_dino1 = nn.Conv2d(dino_dims, 64, 3, padding=1)
        self.conv_dino2 = nn.Conv2d(64, 64, 3, padding=1)

        # GEV feature encoder (geometry encoding)
        self.conv_gev1 = nn.Conv2d(gev_dims, 64, 1)
        self.conv_gev2 = nn.Conv2d(64, 64, 3, padding=1)

        # Depth encoder
        self.conv_depth1 = nn.Conv2d(1, 32, 7, padding=3)
        self.conv_depth2 = nn.Conv2d(32, 32, 3, padding=1)

        # Combine: sim(64) + dino(64) + gev(64) + depth(32) = 224 → hidden-1
        self.conv_out = nn.Conv2d(64 + 64 + 64 + 32, hidden_channels - 1, 3, padding=1)

    def forward(self, sim_context, dino_feat, gev_feat, depth):
        sim = F.relu(self.conv_sim1(sim_context))
        sim = F.relu(self.conv_sim2(sim))

        dino = F.relu(self.conv_dino1(dino_feat))
        dino = F.relu(self.conv_dino2(dino))

        gev = F.relu(self.conv_gev1(gev_feat))
        gev = F.relu(self.conv_gev2(gev))

        dep = F.relu(self.conv_depth1(depth))
        dep = F.relu(self.conv_depth2(dep))

        out = F.relu(self.conv_out(torch.cat([sim, dino, gev, dep], dim=1)))
        return torch.cat([out, depth], dim=1)


class UpdateBlock(nn.Module):
    """
    Update block with separate similarity context, DINO context and GEV features.
    Similarity profile is looked up internally at each iteration.
    """
    def __init__(self, sim_context_dims, dino_dims, gev_dims, hidden_dim,
                 num_downsample=1, similarity_context=None):
        super().__init__()
        self.num_downsample = num_downsample
        self.similarity_context = similarity_context  # For lookup

        # Motion encoder with separate sim, dino, gev inputs
        self.encoder = MotionEncoder(sim_context_dims, dino_dims, gev_dims,
                                           hidden_channels=128)
        encoder_output_dim = 128

        # GRU for hidden state update
        # Input: dino_feat (context) + motion_feat
        self.gru = ConvGRU(hidden_dim, encoder_output_dim + dino_dims)

        # Depth prediction head
        self.depth_head = DepthHead(hidden_dim, hidden_dim=128, output_dim=1)

        # Upsampling mask
        factor = 2 ** num_downsample
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, (factor ** 2) * 9, 1, padding=0)
        )

    def forward(self, net, similarity_profile, dino_feat, gev_feat, inv_depth,
                no_upsample=False):
        """
        Args:
            net: [B, hidden_dim, H, W] - hidden state
            similarity_profile: [B, Ds, H, W] - reused similarity profile
            dino_feat: [B, C_dino, H, W] - DINO context feature
            gev_feat: [B, gev_dims, H, W] - GEV lookup feature
            inv_depth: [B, 1, H, W] - current inverse depth estimate
        """
        # Lookup similarity context at current depth (corr branch)
        sim_context = self.similarity_context.lookup_context(similarity_profile, inv_depth)

        # Encode motion with all three signals
        motion_feat = self.encoder(sim_context, dino_feat, gev_feat, inv_depth)

        # Combine DINO context and motion features
        inp = torch.cat([dino_feat, motion_feat], dim=1)

        # GRU update
        net = self.gru(net, inp)

        # Predict depth update
        delta_depth = self.depth_head(net)

        if no_upsample:
            return net, delta_depth, None

        mask = 0.25 * self.mask(net)
        return net, delta_depth, mask


class OmniDS(nn.Module):
    """
    OmniDS: CNN features for matching, DINO features for context, and a
    separate Geometry Encoding Volume (GEV) for regularized geometry.

    Pipeline:
    1. CNN encoder over the 4 fisheye images -> spherical sweep -> CNN volumes
    2. SimilarityContext builds a depth-wise matching profile from those volumes
    3. GEVModule builds a 3D-UNet-regularized geometry volume from the same
       ref/tgt volumes and exposes a depth-axis pyramid for lookup
    4. DINO features are projected to ERP by deformable cross-attention and fused
       with sampled CNN features to form the context
    5. Iteratively: look up similarity + GEV at the current depth, update through
       the ConvGRU, and predict a depth residual
    """

    def __init__(self, varargin=None):
        super().__init__()
        opts = Edict()
        opts.use_rgb = False
        opts.base_channel = 32
        opts.num_downsample = 1
        opts.equi_h = 160
        opts.equi_w = 640
        opts.num_invdepth = 192
        opts.phi_deg = 45.0
        opts.num_heads = 4
        opts.num_points = 4
        opts.num_cross_attn_layers = 1
        opts.mixed_precision = False
        opts.fix_bn = False
        # DINO options
        opts.dino_model = 'dinov3_vits16'  # DINOv3 (patch 16)
        opts.freeze_dino = True
        opts.use_dino_sim = False
        # Similarity context options
        opts.num_depth_samples = 96  # Sparse sampling for similarity profile
        opts.sim_radius = 4
        opts.sim_levels = 4
        opts.similarity_type = 'correlation'
        # Distillation: build the lightweight student encoder (see --distilled)
        opts.use_student = False

        self.opts = argparse(opts, varargin)
        # CNN feature extractor (for similarity/matching)
        self.encoder = FeatureLayers(
            self.opts.base_channel, 
            self.opts.use_rgb,
            getattr(self.opts, 'encoder_downsample_twice', False)
        )
        
        # Compute downsampled ERP size
        factor = 2 ** self.opts.num_downsample
        self.erp_h = self.opts.equi_h // factor
        self.erp_w = self.opts.equi_w // factor
        
        cnn_dims = self.opts.base_channel
        dino_dims = self.opts.base_channel  # Project DINO to same dim
        hidden_dim = self.opts.base_channel * 2
        
        # Similarity context module (CNN features → matching signal)
        self.similarity_context = SimilarityContext(
            num_depth_samples=self.opts.num_depth_samples,
            num_invdepth=self.opts.num_invdepth,
            embed_dims=cnn_dims,
            radius=self.opts.sim_radius,
            num_levels=self.opts.sim_levels,
            similarity_type=self.opts.similarity_type,
            # Spatial aggregation options
            use_spatial_aggregation=getattr(self.opts, 'use_spatial_aggregation', True),
            spatial_agg_type=getattr(self.opts, 'spatial_agg_type', 'conv'),
            spatial_hidden_dim=getattr(self.opts, 'spatial_hidden_dim', 32),
            spatial_num_layers=getattr(self.opts, 'spatial_num_layers', 2),
        )
        sim_context_dims = self.similarity_context.output_dim
        
        # DINO feature extractor (for context) - uses cross-attention with distortion embedding
        # Get fisheye image dimensions from encoder output (will be set dynamically)
        self.dino_extractor = DINOERPExtractor(
            dino_model_name=self.opts.dino_model,
            output_dim=dino_dims,
            equi_h=self.erp_h,
            equi_w=self.erp_w,
            num_cams=4,
            freeze_dino=self.opts.freeze_dino,
            num_heads=self.opts.num_heads,
            num_points=self.opts.num_points,
            num_cross_attn_layers=self.opts.num_cross_attn_layers,
            fish_h=384,  # Default fisheye height (will be updated from actual image)
            fish_w=400,  # Default fisheye width (will be updated from actual image)
            no_cross_attn=getattr(self.opts, 'no_dino_cross_attn', False),
        )
        
        # Select fusion type based on options (default: fast pointwise)
        fusion_type = getattr(self.opts, 'context_fusion_type', 'pointwise')
        
        self.cnn_erp_proj = Conv2D(cnn_dims, dino_dims, 1, pad=0, relu=False) if cnn_dims != dino_dims else None
        
        if fusion_type == 'pointwise':
            self.context_fuser = ERPContextFusionPointwise(
                embed_dims=dino_dims,
                hidden_dims=dino_dims * 2,
            )
        elif fusion_type == 'gated':
            self.context_fuser = ERPContextFusionGated(
                embed_dims=dino_dims,
                hidden_dims=dino_dims,
            )
        elif fusion_type == 'local':
            self.context_fuser = ERPContextFusionLocalConv(
                embed_dims=dino_dims,
                hidden_dims=dino_dims * 2,
            )
        elif fusion_type == 'attention':
            self.context_fuser = ERPContextFusionAttention(
                embed_dims=dino_dims,
                num_heads=self.opts.num_heads,
                ffn_dims=4 * dino_dims,
                dropout=0.0,
            )
        else:
            raise ValueError(f"Unknown context_fusion_type: {fusion_type}")
        
        # Context/state initialization (use DINO only, similarity is in motion encoder)
        self.context_proj = Conv2D(dino_dims, dino_dims, 1, pad=0, relu=False)
        self.state_proj = Conv2D(dino_dims, hidden_dim, 1, pad=0, relu=False)
        
        # Independent GEV branch: 3D UNet (RegularizationNet3D) with guided
        # excitation over GWC + multi-view variance, exposed as a depth pyramid.
        self._gev_radius = getattr(self.opts, 'gev_radius', 4)
        self.gev_module = GEVModule(
            in_channels=cnn_dims,
            num_groups=getattr(self.opts, 'gev_num_groups', 8),
            reg_channels=getattr(self.opts, 'gev_reg_channels', (16, 32, 48)),
            num_pyramid_levels=getattr(self.opts, 'gev_num_pyramid_levels', 2),
            use_spatial_downsample=getattr(self.opts, 'gev_use_spatial_downsample', True),
        )

        # Update block; similarity_context is passed in for per-iteration lookup
        self.update_block = UpdateBlock(
            sim_context_dims=sim_context_dims,
            dino_dims=dino_dims,
            gev_dims=self.gev_module.lookup_dim(radius=self._gev_radius),
            hidden_dim=hidden_dim,
            num_downsample=self.opts.num_downsample,
            similarity_context=self.similarity_context,
        )

        # Learned view weights for aggregating the 4 CNN views into ERP
        self.view_weight_head = nn.Sequential(
            nn.Conv2d(cnn_dims * 4, cnn_dims, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnn_dims, 4, 1),
        )

        # Student encoder for DINO distillation. Built only when requested:
        # registering it unconditionally would add feature_student.* to
        # state_dict and break strict loading of checkpoints trained without
        # it, and would force the FoundationStereo import chain on every run.
        if self.opts.use_student:
            from module.network_student import FeatureStudent_mobilenetv2_deconv
            self.feature_student = FeatureStudent_mobilenetv2_deconv()
        else:
            self.feature_student = None

        self.debug_outputs = {}

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    
    def upsample_depth(self, depth, mask):
        """Upsample depth using convex combination."""
        B, C, H, W = depth.shape
        factor = 2 ** self.opts.num_downsample
        mask = mask.view(B, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)
        
        up_depth = F.unfold(factor * depth, [3, 3], padding=1)
        up_depth = up_depth.view(B, C, 9, 1, 1, H, W)
        
        up_depth = torch.sum(mask * up_depth, dim=2)
        up_depth = up_depth.permute(0, 1, 4, 2, 5, 3)
        return up_depth.reshape(B, C, factor * H, factor * W)
    
    def compute_reference_points_from_grids(self, grids, inv_depth_idx):
        """Compute reference points using pre-computed lookup tables."""
        B = inv_depth_idx.shape[0]
        device = inv_depth_idx.device
        dtype = inv_depth_idx.dtype

        reference_points = []

        for cam_idx, grid in enumerate(grids):
            grid = grid.to(device).to(dtype)
            D = grid.shape[2]
            
            idx_flat = inv_depth_idx.view(B, -1)
            
            idx_floor = idx_flat.long().clamp(0, D - 1)
            idx_ceil = (idx_floor + 1).clamp(0, D - 1)
            weight = (idx_flat - idx_floor.float()).unsqueeze(-1)
            
            grid_flat = grid.view(-1, D, 2)
            grid_flat = grid_flat.unsqueeze(0).expand(B, -1, -1, -1)
            
            idx_floor_exp = idx_floor.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
            idx_ceil_exp = idx_ceil.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
            
            ref_floor = torch.gather(grid_flat, 2, idx_floor_exp).squeeze(2)
            ref_ceil = torch.gather(grid_flat, 2, idx_ceil_exp).squeeze(2)
            
            ref_pts = ref_floor * (1 - weight) + ref_ceil * weight
            ref_pts = (ref_pts + 1) / 2
            
            reference_points.append(ref_pts)
        
        reference_points = torch.stack(reference_points, dim=2)
        return reference_points

    def _sample_cnn_erp(self, cnn_feats_stacked, grids_tensor, inv_depth_idx):
        """
        Sample CNN fisheye features to ERP with grid-validity masking and learned
        view weighting.

        Args:
            cnn_feats_stacked: [B, 4, C, Hf, Wf]
            grids_tensor: list of 4 tensors, each [H, W, D, 2] in grid_sample coords
            inv_depth_idx: [B, 1, H, W]
        Returns:
            cnn_erp: [B, C, H, W]
        """
        B, _, C, _, _ = cnn_feats_stacked.shape
        H, W = inv_depth_idx.shape[2], inv_depth_idx.shape[3]
        device = inv_depth_idx.device
        dtype = cnn_feats_stacked.dtype

        per_cam = []
        validity = []
        idx_flat = inv_depth_idx.view(B, -1)  # [B, H*W]

        for cam_idx, grid_lut in enumerate(grids_tensor):
            grid_lut = grid_lut.to(device=device, dtype=dtype)  # [H, W, D, 2]
            D = grid_lut.shape[2]

            idx_floor = idx_flat.long().clamp(0, D - 1)
            idx_ceil = (idx_floor + 1).clamp(0, D - 1)
            w = (idx_flat - idx_floor.float()).unsqueeze(-1)  # [B, H*W, 1]

            grid_flat = grid_lut.view(-1, D, 2).unsqueeze(0).expand(B, -1, -1, -1)
            idx_floor_exp = idx_floor.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
            idx_ceil_exp = idx_ceil.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)

            g0 = torch.gather(grid_flat, 2, idx_floor_exp).squeeze(2)
            g1 = torch.gather(grid_flat, 2, idx_ceil_exp).squeeze(2)
            grid = (1.0 - w) * g0 + w * g1  # [B, H*W, 2]
            grid = grid.view(B, H, W, 2)

            # Grid validity: coordinates inside [-1, 1] are within the camera FOV
            valid = ((grid[..., 0].abs() <= 1) & (grid[..., 1].abs() <= 1)).float()
            validity.append(valid)  # [B, H, W]

            feat = F.grid_sample(
                cnn_feats_stacked[:, cam_idx], grid,
                mode='bilinear', padding_mode='zeros', align_corners=True,
            )  # [B, C, H, W]
            per_cam.append(feat)

        # [B, 4, C, H, W]  and  [B, 4, H, W]
        per_cam = torch.stack(per_cam, dim=1)
        validity_mask = torch.stack(validity, dim=1)

        # Learned view weights conditioned on per-camera features
        feat_concat = per_cam.view(B, 4 * C, H, W)
        learned_logits = self.view_weight_head(feat_concat)  # [B, 4, H, W]

        # Mask out cameras outside FOV, then softmax
        _mask_val = -1e4 if learned_logits.dtype == torch.float16 else -1e9
        learned_logits = learned_logits.masked_fill(validity_mask == 0, _mask_val)
        weights = F.softmax(learned_logits, dim=1)  # [B, 4, H, W]

        # Weighted aggregation
        cnn_erp = (per_cam * weights.unsqueeze(2)).sum(dim=1)  # [B, C, H, W]
        return cnn_erp

    def forward(self, imgs, grids, iters=12, test_mode=False, ocams=None, distilled=False):
        """
        Forward pass with separate GEV + Similarity Context.
        """
        with autocast(enabled=self.opts.mixed_precision):
            # 1. Extract CNN features (for similarity/matching)
            # 2. Extract DINO features (for context only)
            dino_fish_h, dino_fish_w = imgs[0].shape[2], imgs[0].shape[3]
            self.dino_extractor.fish_h = dino_fish_h
            self.dino_extractor.fish_w = dino_fish_w

            if not distilled:
                cnn_feats = self.encoder(imgs)  # list of 4 [B, C, H, W]
                dino_feats_stacked = self.dino_extractor.extract_features(imgs)
            else:
                if self.feature_student is None:
                    raise RuntimeError(
                        "distilled=True but the student encoder was not built. "
                        "Construct the network with use_student=True (pass --distilled)."
                    )
                cnn_feats, dino_feats_stacked = self.feature_student(imgs)
                dino_feats_resized = [F.interpolate(dino_feat, size=(cnn_feats[0].shape[2], cnn_feats[0].shape[3]), mode='bilinear', align_corners=False) for dino_feat in dino_feats_stacked]
                dino_feats_stacked = torch.stack(dino_feats_resized, dim=0)

        if not self.opts.mixed_precision:
            cnn_feats = [feat.float() for feat in cnn_feats]
        B = cnn_feats[0].shape[0]

        cnn_feats_stacked = torch.stack(cnn_feats, dim=1)

        grids_tensor = [g if torch.is_tensor(g) else torch.tensor(g, device=cnn_feats[0].device)
                        for g in grids]

        if not self.opts.mixed_precision:
            dino_feats_stacked = dino_feats_stacked.float()

        # 3. Build spherical volumes for similarity
        use_dino_sim = getattr(self.opts, 'use_dino_sim', False)

        with autocast(enabled=self.opts.mixed_precision):
            if use_dino_sim:
                # DINO-only sim: use DINO features for spherical sweep
                dino_feats_list = [dino_feats_stacked[:, i] for i in range(4)]
                sim_sph = spherical_sweep(dino_feats_list, grids_tensor)
                del dino_feats_list
            else:
                # CNN-only sim (default)
                sim_sph = spherical_sweep(cnn_feats, grids_tensor)
            del cnn_feats

            # 4. Build similarity profile AND get ref/tgt/cam_vols for GEV
            similarity_profile, ref, tgt, cam_vols = \
                self.similarity_context.build_profile(
                    grids=grids_tensor, spherical_volumes=sim_sph,
                    return_volumes=True,
                    use_triton=self.opts.get('use_triton', False),
                )

            # 5. Build independent GEV from ref/tgt volumes
            self.gev_module.build(ref, tgt, cam_vols,
                                  use_triton=self.opts.get('use_triton', False))
            del ref, tgt, cam_vols, sim_sph

        # Initialize depth at D_grid / 2
        D_grid = grids_tensor[0].shape[2]
        inv_depth_idx = torch.zeros(B, 1, self.erp_h, self.erp_w,
                                    device=cnn_feats_stacked.device,
                                    dtype=cnn_feats_stacked.dtype)
        inv_depth_idx = inv_depth_idx + D_grid / 2

        # Initial context (DINO+CNN fusion at initial depth)
        reference_points = self.compute_reference_points_from_grids(
            grids_tensor, inv_depth_idx
        )

        with autocast(enabled=self.opts.mixed_precision):
            dino_erp = self.dino_extractor(
                dino_feats_stacked, reference_points, inv_depth_idx,
                grids_tensor=grids_tensor,
            )
            if not self.opts.mixed_precision:
                dino_erp = dino_erp.float()

            cnn_erp = self._sample_cnn_erp(
                cnn_feats_stacked, grids_tensor, inv_depth_idx
            )
            if not self.opts.mixed_precision:
                cnn_erp = cnn_erp.float()
            if self.cnn_erp_proj is not None:
                cnn_erp = self.cnn_erp_proj(cnn_erp)

            fused_erp = self.context_fuser(dino_erp, cnn_erp)

            context = torch.relu(self.context_proj(fused_erp))
            net = torch.tanh(self.state_proj(fused_erp))

        # Iterative refinement
        predictions = []

        for itr in range(iters):
            inv_depth_idx = inv_depth_idx.detach()

            if itr > 0:
                reference_points = self.compute_reference_points_from_grids(
                    grids_tensor, inv_depth_idx
                )

                with autocast(enabled=self.opts.mixed_precision):
                    dino_erp = self.dino_extractor(
                        dino_feats_stacked, reference_points, inv_depth_idx,
                        grids_tensor=grids_tensor,
                    )
                    if not self.opts.mixed_precision:
                        dino_erp = dino_erp.float()

                    cnn_erp = self._sample_cnn_erp(
                        cnn_feats_stacked, grids_tensor, inv_depth_idx
                    )
                    if not self.opts.mixed_precision:
                        cnn_erp = cnn_erp.float()
                    if self.cnn_erp_proj is not None:
                        cnn_erp = self.cnn_erp_proj(cnn_erp)

                    fused_erp = self.context_fuser(dino_erp, cnn_erp)
                    context = torch.relu(self.context_proj(fused_erp))

            # GEV lookup at current depth
            gev_feat = self.gev_module.lookup(inv_depth_idx, radius=self._gev_radius)

            with autocast(enabled=self.opts.mixed_precision):
                net, delta_depth, up_mask = self.update_block(
                    net, similarity_profile, context, gev_feat, inv_depth_idx,
                    no_upsample=(test_mode and itr < iters - 1)
                )

            inv_depth_idx = inv_depth_idx + delta_depth

            if up_mask is not None:
                inv_depth_up = self.upsample_depth(inv_depth_idx, up_mask)
                predictions.append(inv_depth_up)

        self.debug_outputs = {}

        # Clear all caches
        self.gev_module.clear_cache()

        if test_mode:
            return torch.clamp(predictions[-1], 0, self.opts.num_invdepth - 1)

        return predictions