# module/erp_cross_attention.py
# BEVFormer-style cross-attention for ERP feature extraction from fisheye cameras
# Uses CUDA Multi-Scale Deformable Attention when available

import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
import MultiScaleDeformableAttention as MSDA



class MSDeformAttnFunction(torch.autograd.Function):
    """CUDA Multi-Scale Deformable Attention Function."""
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, 
                sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index, 
            sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, 
                              sampling_locations, attention_weights)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, \
            sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index, 
                sampling_locations, attention_weights, grad_output, ctx.im2col_step)
        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_attn_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """
    PyTorch fallback for Multi-Scale Deformable Attention.
    
    Args:
        value: [B, sum(H*W), num_heads, head_dim]
        value_spatial_shapes: [num_levels, 2] - (H, W) for each level
        sampling_locations: [B, Lq, num_heads, num_levels, num_points, 2]
        attention_weights: [B, Lq, num_heads, num_levels, num_points]
    Returns:
        output: [B, Lq, num_heads * head_dim]
    """
    B, _, num_heads, head_dim = value.shape
    _, Lq, _, num_levels, num_points, _ = sampling_locations.shape
    
    # Split value by levels
    value_list = value.split([H * W for H, W in value_spatial_shapes], dim=1)
    
    sampling_grids = 2 * sampling_locations - 1  # Convert to [-1, 1]
    
    sampling_value_list = []
    for level_idx, (H, W) in enumerate(value_spatial_shapes):
        # value_l: [B, H*W, num_heads, head_dim] -> [B*num_heads, head_dim, H, W]
        value_l = value_list[level_idx].permute(0, 2, 3, 1).reshape(B * num_heads, head_dim, H, W)
        
        # sampling_grid_l: [B, Lq, num_heads, num_points, 2] -> [B*num_heads, Lq, num_points, 2]
        sampling_grid_l = sampling_grids[:, :, :, level_idx].permute(0, 2, 1, 3, 4).reshape(
            B * num_heads, Lq, num_points, 2)
        
        # Sample: [B*num_heads, head_dim, Lq, num_points]
        sampling_value_l = F.grid_sample(
            value_l, sampling_grid_l, mode='bilinear', padding_mode='zeros', align_corners=False)
        
        sampling_value_list.append(sampling_value_l)
    
    # Stack levels: [B*num_heads, head_dim, Lq, num_levels, num_points]
    sampling_values = torch.stack(sampling_value_list, dim=-2)
    
    # Reshape attention weights: [B, Lq, num_heads, num_levels, num_points] -> [B*num_heads, 1, Lq, num_levels, num_points]
    attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
        B * num_heads, 1, Lq, num_levels, num_points)
    
    # Weighted sum: [B*num_heads, head_dim, Lq]
    output = (sampling_values * attention_weights).sum(-1).sum(-1)
    
    # Reshape: [B, Lq, num_heads * head_dim]
    output = output.view(B, num_heads, head_dim, Lq).permute(0, 3, 1, 2).reshape(B, Lq, -1)
    
    return output


class PositionalEncoding2D(nn.Module):
    """2D positional encoding for ERP queries."""
    def __init__(self, embed_dims):
        super().__init__()
        self.embed_dims = embed_dims
        self.row_embed = nn.Embedding(512, embed_dims // 2)
        self.col_embed = nn.Embedding(512, embed_dims // 2)
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)
    
    def forward(self, h, w, device):
        i = torch.arange(w, device=device)
        j = torch.arange(h, device=device)
        x_emb = self.col_embed(i)  # [W, C/2]
        y_emb = self.row_embed(j)  # [H, C/2]
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),  # [H, W, C/2]
            y_emb.unsqueeze(1).repeat(1, w, 1),  # [H, W, C/2]
        ], dim=-1)  # [H, W, C]
        return pos.flatten(0, 1)  # [H*W, C]


class DepthEncoding(nn.Module):
    """Learnable depth level encoding."""
    def __init__(self, num_depth_bins, embed_dims):
        super().__init__()
        self.depth_embed = nn.Embedding(num_depth_bins, embed_dims)
    
    def forward(self, depth_idx):
        return self.depth_embed(depth_idx)


class MSDeformCrossAttention(nn.Module):
    """
    Multi-Scale Deformable Cross-Attention for fisheye to ERP feature extraction.
    Uses CUDA MSDA when available, falls back to PyTorch implementation.
    """
    def __init__(self, embed_dims, num_heads=4, num_points=4, num_cams=4, dropout=0.1,
                 use_radial_weighting=False, use_geom_bias=False):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f'd_model must be divisible by n_heads, got {embed_dims} and {num_heads}')
        
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_cams = num_cams  # num_levels in MSDA terminology
        self.head_dims = embed_dims // num_heads
        self.im2col_step = 64
        self.use_radial_weighting = use_radial_weighting
        self.use_geom_bias = use_geom_bias
        
        # Sampling offset prediction
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_cams * num_points * 2)
        
        # Attention weight prediction
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_cams * num_points)
        
        # Value projection
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        
        self.dropout = nn.Dropout(dropout)
        
        if self.use_radial_weighting:
            # Learnable radial weighting: [r_norm, theta_norm] -> weight in [0, 1]
            self.radial_mlp = nn.Sequential(
                nn.Conv2d(2, 16, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 1),
                nn.Sigmoid()
            )
            self._radial_cache = {}

        if self.use_geom_bias:
            # Geometric distortion bias from radial distance
            self.geom_bias_mlp = nn.Sequential(
                nn.Linear(1, 8),
                nn.ReLU(inplace=True),
                nn.Linear(8, 1)
            )

        self._reset_parameters()
    
    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        # Initialize offsets in a circular pattern
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
        grid_init = grid_init.view(self.num_heads, 1, 1, 2).repeat(1, self.num_cams, self.num_points, 1)
        
        # Scale by point index
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= (i + 1) * 0.5
        
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def _get_radial_coords(self, H, W, device, dtype):
        """Create normalized radial coords [1, 2, H, W] = (r_norm, theta_norm)."""
        key = (H, W, device.type, dtype)
        cached = self._radial_cache.get(key)
        if cached is not None:
            return cached
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing='ij'
        )
        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0
        dx = xx - cx
        dy = yy - cy
        r = torch.sqrt(dx * dx + dy * dy)
        max_r = torch.sqrt(torch.tensor(cx * cx + cy * cy, device=device, dtype=dtype))
        r_norm = r / (max_r + 1e-6)
        theta = torch.atan2(dy, dx)
        theta_norm = (theta + math.pi) / (2 * math.pi)
        coords = torch.stack([r_norm, theta_norm], dim=0).unsqueeze(0)  # [1, 2, H, W]
        self._radial_cache[key] = coords
        return coords
    
    def forward(self, query, fisheye_feats, reference_points, feat_spatial_shapes):
        """
        Args:
            query: [B, Lq, C] - ERP queries
            fisheye_feats: [B, num_cams, C, H_fish, W_fish] - fisheye features
            reference_points: [B, Lq, num_cams, 2] - projected points (normalized 0-1)
            feat_spatial_shapes: [num_cams, 2] - (H, W) of each fisheye feature
        Returns:
            output: [B, Lq, C]
        """
        B, Lq, C = query.shape
        num_cams = self.num_cams
        
        # Prepare value: flatten all camera features
        # [B, num_cams, C, H, W] -> [B, sum(H*W), num_heads, head_dim]
        value_list = []
        spatial_shapes_list = []
        for cam_idx in range(num_cams):
            feat = fisheye_feats[:, cam_idx]  # [B, C, H, W]
            H, W = feat.shape[2], feat.shape[3]
            
            # if self.use_radial_weighting:
            #     coords = self._get_radial_coords(H, W, feat.device, feat.dtype)
            #     radial_weight = self.radial_mlp(coords)  # [1, 1, H, W]
            #     feat = feat * radial_weight
            
            feat_flat = feat.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
            feat_flat = self.value_proj(feat_flat)  # [B, H*W, C]
            value_list.append(feat_flat)
            spatial_shapes_list.append([H, W])
        
        # Concatenate values from all cameras
        value = torch.cat(value_list, dim=1)  # [B, sum(H*W), C]
        value = value.view(B, -1, self.num_heads, self.head_dims)  # [B, sum(H*W), num_heads, head_dim]
        
        # Spatial shapes for MSDA
        spatial_shapes = torch.tensor(spatial_shapes_list, device=query.device, dtype=torch.long)
        level_start_index = torch.cat([
            torch.zeros(1, device=query.device, dtype=torch.long),
            torch.cumsum(spatial_shapes[:, 0] * spatial_shapes[:, 1], dim=0)[:-1]
        ])
        
        # Predict sampling offsets: [B, Lq, num_heads, num_cams, num_points, 2]
        offsets = self.sampling_offsets(query)
        offsets = offsets.view(B, Lq, self.num_heads, num_cams, self.num_points, 2)
        
        # Predict attention logits
        attn_logits = self.attention_weights(query)
        attn_logits = attn_logits.view(B, Lq, self.num_heads, num_cams, self.num_points)
        # Geometric distortion bias (per camera, per query)
        if self.use_geom_bias:
            ref_pts = reference_points
            ref_pts = ref_pts * 2.0 - 1.0
            # Radial distance from camera center (0,0) in normalized coords
            dist = torch.sqrt(ref_pts[..., 0] ** 2 + ref_pts[..., 1] ** 2)  # [B, Lq, num_cams]
            dist = dist.clamp(min=0.0, max=1.0)
            # MLP expects [N, 1]
            bias = self.geom_bias_mlp(dist.unsqueeze(-1))  # [B, Lq, num_cams, 1]
            # Expand to heads/points: [B, Lq, 1, num_cams, 1]
            bias = bias.unsqueeze(2)
            attn_logits = attn_logits + bias

        # Softmax over cameras and points
        attn_weights = F.softmax(attn_logits.view(B, Lq, self.num_heads, num_cams * self.num_points), dim=-1)
        attn_weights = attn_weights.view(B, Lq, self.num_heads, num_cams, self.num_points)
        
        # Compute sampling locations
        # reference_points: [B, Lq, num_cams, 2]
        # Normalize offsets by spatial shapes
        offset_normalizer = torch.stack([spatial_shapes[:, 1], spatial_shapes[:, 0]], -1).float()  # [num_cams, 2]
        
        sampling_locations = reference_points[:, :, None, :, None, :] + \
                            offsets / offset_normalizer[None, None, None, :, None, :]
        # [B, Lq, num_heads, num_cams, num_points, 2]
        
        # MSDA CUDA kernel requires fp32
        input_dtype = value.dtype
        output = MSDeformAttnFunction.apply(
            value.contiguous().float(),
            spatial_shapes,
            level_start_index,
            sampling_locations.contiguous().float(),
            attn_weights.contiguous().float(),
            self.im2col_step
        )
        output = output.to(input_dtype)


        output = self.output_proj(output)
        return self.dropout(output)


class ERPSelfAttention(nn.Module):
    """Self-attention in ERP space for context aggregation."""
    def __init__(self, embed_dims, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dims = embed_dims // num_heads
        
        self.q_proj = nn.Linear(embed_dims, embed_dims)
        self.k_proj = nn.Linear(embed_dims, embed_dims)
        self.v_proj = nn.Linear(embed_dims, embed_dims)
        self.out_proj = nn.Linear(embed_dims, embed_dims)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dims ** -0.5
    
    def forward(self, query, pos_embed=None):
        B, Lq, C = query.shape
        
        if pos_embed is not None:
            if pos_embed.dim() == 2:
                pos_embed = pos_embed.unsqueeze(0).expand(B, -1, -1)
            q = self.q_proj(query + pos_embed)
            k = self.k_proj(query + pos_embed)
        else:
            q = self.q_proj(query)
            k = self.k_proj(query)
        v = self.v_proj(query)
        
        # Multi-head reshape
        q = q.view(B, Lq, self.num_heads, self.head_dims).permute(0, 2, 1, 3)
        k = k.view(B, Lq, self.num_heads, self.head_dims).permute(0, 2, 1, 3)
        v = v.view(B, Lq, self.num_heads, self.head_dims).permute(0, 2, 1, 3)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, Lq, C)
        return self.out_proj(out)


class MSDeformSelfAttention(nn.Module):
    """
    Multi-Scale Deformable Self-Attention for ERP feature refinement.
    More efficient than full self-attention for large feature maps.
    """
    def __init__(self, embed_dims, num_heads=4, num_points=4, dropout=0.1):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dims = embed_dims // num_heads
        self.im2col_step = 64
        
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()
    
    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
        grid_init = grid_init.view(self.num_heads, 1, 2).repeat(1, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, i, :] *= (i + 1)
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)
    
    def forward(self, query, reference_points_2d, spatial_shape, pos_embed=None):
        """
        Args:
            query: [B, H*W, C]
            reference_points_2d: [B, H*W, 1, 2] - normalized coordinates
            spatial_shape: (H, W)
        """
        B, Lq, C = query.shape
        H, W = spatial_shape
        
        if pos_embed is not None:
            if pos_embed.dim() == 2:
                pos_embed = pos_embed.unsqueeze(0).expand(B, -1, -1)
            query_with_pos = query + pos_embed
        else:
            query_with_pos = query
        
        value = self.value_proj(query)
        value = value.view(B, Lq, self.num_heads, self.head_dims)
        
        offsets = self.sampling_offsets(query_with_pos)
        offsets = offsets.view(B, Lq, self.num_heads, 1, self.num_points, 2)
        
        attn_weights = self.attention_weights(query_with_pos)
        attn_weights = attn_weights.view(B, Lq, self.num_heads, self.num_points)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.view(B, Lq, self.num_heads, 1, self.num_points)
        
        offset_normalizer = torch.tensor([W, H], device=query.device, dtype=query.dtype)
        sampling_locations = reference_points_2d[:, :, None, :, None, :] + offsets / offset_normalizer
        
        spatial_shapes = torch.tensor([[H, W]], device=query.device, dtype=torch.long)
        level_start_index = torch.tensor([0], device=query.device, dtype=torch.long)
        
        # MSDA CUDA kernel requires fp32
        input_dtype = value.dtype
        output = MSDeformAttnFunction.apply(
            value.contiguous().float(),
            spatial_shapes,
            level_start_index,
            sampling_locations.contiguous().float(),
            attn_weights.contiguous().float(),
            self.im2col_step
        )
        output = output.to(input_dtype)


        output = self.output_proj(output)
        return self.dropout(output) + query


class ERPCrossAttentionLayer(nn.Module):
    """
    Single layer of ERP feature extraction:
    Self-Attention → Norm → Cross-Attention → Norm → FFN → Norm
    
    Uses CUDA MSDA for both self-attention and cross-attention.
    """
    def __init__(self, embed_dims, num_heads=4, num_points=4, num_cams=4, 
                 ffn_dims=256, dropout=0.1, use_deform_self_attn=True,
                 use_radial_weighting=False, use_geom_bias=False):
        super().__init__()
        self.use_deform_self_attn = use_deform_self_attn
        
        if use_deform_self_attn:
            self.self_attn = MSDeformSelfAttention(embed_dims, num_heads, num_points, dropout)
        else:
            self.self_attn = ERPSelfAttention(embed_dims, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dims)
        
        self.cross_attn = MSDeformCrossAttention(
            embed_dims, num_heads, num_points, num_cams, dropout,
            use_radial_weighting=use_radial_weighting,
            use_geom_bias=use_geom_bias
        )
        self.norm2 = nn.LayerNorm(embed_dims)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, ffn_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dims, embed_dims),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(embed_dims)
    
    def forward(self, query, fisheye_feats, reference_points, feat_spatial_shapes,
                pos_embed=None, ref_2d=None, spatial_shape=None):
        """
        Args:
            query: [B, Lq, C]
            fisheye_feats: [B, num_cams, C, H, W]
            reference_points: [B, Lq, num_cams, 2] - for cross-attention
            feat_spatial_shapes: [num_cams, 2]
            pos_embed: [Lq, C] or [B, Lq, C]
            ref_2d: [B, Lq, 1, 2] - for deformable self-attention
            spatial_shape: (H, W) - for deformable self-attention
        """
        # Self-attention
        identity = query
        if self.use_deform_self_attn and ref_2d is not None:
            query = self.self_attn(query, ref_2d, spatial_shape, pos_embed)
        else:
            query = self.self_attn(query, pos_embed)
        query = self.norm1(query + identity)
        
        # Cross-attention
        identity = query
        query = self.cross_attn(
            query, fisheye_feats, reference_points, feat_spatial_shapes,
        )
        query = self.norm2(query + identity)
        
        # FFN
        identity = query
        query = self.ffn(query)
        query = self.norm3(query + identity)
        
        return query


class IterativeERPExtractor(nn.Module):
    """
    Iterative ERP feature extraction with depth refinement.
    Uses CUDA MSDA for efficient cross-attention.
    """
    def __init__(self, embed_dims, equi_h, equi_w, num_cams=4,
                 num_heads=4, num_points=4, num_layers=1, use_deform_self_attn=True,
                 use_radial_weighting=False, use_geom_bias=False):
        super().__init__()
        self.embed_dims = embed_dims
        self.equi_h = equi_h
        self.equi_w = equi_w
        self.num_cams = num_cams
        self.use_deform_self_attn = use_deform_self_attn
        
        # ERP query embedding
        self.erp_embed = nn.Embedding(equi_h * equi_w, embed_dims)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding2D(embed_dims)
        
        # Depth encoding (continuous)
        self.depth_encoder = nn.Sequential(
            nn.Linear(1, embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims // 2, embed_dims),
        )
        
        # Cross-attention layers
        self.layers = nn.ModuleList([
            ERPCrossAttentionLayer(
                embed_dims, num_heads, num_points, num_cams,
                use_deform_self_attn=use_deform_self_attn,
                use_radial_weighting=use_radial_weighting,
                use_geom_bias=use_geom_bias
            )
            for _ in range(num_layers)
        ])
        
        # Pre-compute 2D reference points for self-attention
        self.register_buffer('ref_2d', self._init_ref_2d(equi_h, equi_w))
    
    def _init_ref_2d(self, H, W):
        """Initialize 2D reference points for deformable self-attention."""
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H),
            torch.linspace(0.5, W - 0.5, W),
            indexing='ij'
        )
        ref_y = ref_y.reshape(-1) / H
        ref_x = ref_x.reshape(-1) / W
        ref_2d = torch.stack([ref_x, ref_y], dim=-1)  # [H*W, 2]
        return ref_2d.view(1, -1, 1, 2)  # [1, H*W, 1, 2]
    
    def forward(self, fisheye_feats, reference_points, current_depth):
        """
        Args:
            fisheye_feats: [B, num_cams, C, H_fish, W_fish]
            reference_points: [B, Lq, num_cams, 2]
            current_depth: [B, 1, H, W]
        Returns:
            erp_feat: [B, C, H, W]
        """
        B = fisheye_feats.shape[0]
        device = fisheye_feats.device
        dtype = fisheye_feats.dtype
        # breakpoint()
        H, W = current_depth.shape[2], current_depth.shape[3]
        Lq = H * W
        
        # Get spatial shapes
        feat_spatial_shapes = torch.tensor(
            [[fisheye_feats.shape[3], fisheye_feats.shape[4]]] * self.num_cams,
            device=device
        )
        
        # Initialize ERP queries
        if Lq == self.equi_h * self.equi_w:
            erp_queries = self.erp_embed.weight.unsqueeze(0).expand(B, -1, -1).to(dtype)
        else:
            # Handle different resolution
            erp_queries = self.erp_embed.weight[:Lq].unsqueeze(0).expand(B, -1, -1).to(dtype)
        
        # Positional encoding
        pos_embed = self.pos_encoding(H, W, device).to(dtype)
        
        # Depth encoding
        depth_flat = current_depth.flatten(2).permute(0, 2, 1)  # [B, Lq, 1]
        depth_enc = self.depth_encoder(depth_flat)  # [B, Lq, C]
        
        # Add depth encoding to query
        query = erp_queries + depth_enc
        
        # 2D reference points for self-attention
        if self.use_deform_self_attn:
            if H == self.equi_h and W == self.equi_w:
                ref_2d = self.ref_2d.expand(B, -1, -1, -1).to(dtype)
            else:
                ref_2d = self._init_ref_2d(H, W).to(device).to(dtype).expand(B, -1, -1, -1)
        else:
            ref_2d = None
        
        # Apply cross-attention layers
        for layer in self.layers:
            query = layer(
                query, fisheye_feats, reference_points, feat_spatial_shapes,
                pos_embed, ref_2d, (H, W)
            )
        
        # Reshape to spatial
        erp_feat = query.permute(0, 2, 1).view(B, self.embed_dims, H, W)
        
        return erp_feat


# Keep backward compatibility
class ERPFeatureExtractor(nn.Module):
    """
    BEVFormer-style ERP feature extractor for multiple depth bins.
    """
    def __init__(self, embed_dims, equi_h, equi_w, num_cams=4, 
                 num_depth_bins=8, num_layers=2, num_heads=4, num_points=4,
                 use_geom_bias=False):
        super().__init__()
        self.embed_dims = embed_dims
        self.equi_h = equi_h
        self.equi_w = equi_w
        self.num_cams = num_cams
        self.num_depth_bins = num_depth_bins
        
        self.erp_embed = nn.Embedding(equi_h * equi_w, embed_dims)
        self.pos_encoding = PositionalEncoding2D(embed_dims)
        self.depth_encoding = DepthEncoding(num_depth_bins, embed_dims)
        
        self.layers = nn.ModuleList([
            ERPCrossAttentionLayer(
                embed_dims, num_heads, num_points, num_cams,
                use_geom_bias=use_geom_bias
            )
            for _ in range(num_layers)
        ])
        
        self.depth_combine = nn.Conv1d(num_depth_bins, 1, 1)
    
    def forward(self, fisheye_feats, reference_points_3d, depth_indices=None):
        B = fisheye_feats.shape[0]
        device = fisheye_feats.device
        dtype = fisheye_feats.dtype
        Lq = self.equi_h * self.equi_w
        
        feat_spatial_shapes = torch.tensor(
            [[fisheye_feats.shape[3], fisheye_feats.shape[4]]] * self.num_cams,
            device=device
        )
        
        erp_queries = self.erp_embed.weight.unsqueeze(0).expand(B, -1, -1).to(dtype)
        pos_embed = self.pos_encoding(self.equi_h, self.equi_w, device).to(dtype)
        
        depth_features = []
        if depth_indices is None:
            depth_indices = range(self.num_depth_bins)
        
        for d_idx in depth_indices:
            ref_points = reference_points_3d[:, d_idx]
            depth_enc = self.depth_encoding(torch.tensor(d_idx, device=device))
            query = erp_queries + depth_enc.unsqueeze(0).unsqueeze(0)
            
            for layer in self.layers:
                query = layer(query, fisheye_feats, ref_points, feat_spatial_shapes, pos_embed)
            
            depth_features.append(query)
        
        depth_features = torch.stack(depth_features, dim=1)
        depth_features = depth_features.permute(0, 3, 2, 1)
        depth_features = self.depth_combine(depth_features.flatten(0, 1))
        depth_features = depth_features.view(B, self.embed_dims, Lq).squeeze(-1)
        
        erp_feat = depth_features.view(B, self.embed_dims, self.equi_h, self.equi_w)
        return erp_feat

class AllDepthERPExtractor(nn.Module):
    """
    ERP feature extraction over *all* depth bins, but returns a SINGLE tensor:
      erp_feat: [B, embed_dims*D, H, W]

    Inputs:
      - fisheye_feats: [B, num_cams, C, Hf, Wf]
      - grids_tensor:  list(len=num_cams), each [H, W, D, 2]
    """
    def __init__(self, embed_dims, equi_h, equi_w, num_cams=4,
                 num_heads=4, num_points=4, num_layers=1,
                 use_deform_self_attn=True, normalize_depth=True,
                 use_geom_bias=False):
        super().__init__()
        self.embed_dims = embed_dims
        self.equi_h = equi_h
        self.equi_w = equi_w
        self.num_cams = num_cams
        self.use_deform_self_attn = use_deform_self_attn
        self.normalize_depth = normalize_depth

        # ERP base embedding (spatial only)
        self.erp_embed = nn.Embedding(equi_h * equi_w, embed_dims)

        self.pos_encoding = PositionalEncoding2D(embed_dims)

        # depth embedding (continuous)
        self.depth_encoder = nn.Sequential(
            nn.Linear(1, embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims // 2, embed_dims),
        )

        self.layers = nn.ModuleList([
            ERPCrossAttentionLayer(
                embed_dims, num_heads, num_points, num_cams,
                use_deform_self_attn=use_deform_self_attn,
                use_geom_bias=use_geom_bias
            )
            for _ in range(num_layers)
        ])

        self.register_buffer('ref_2d', self._init_ref_2d(equi_h, equi_w), persistent=True)
        self.max_depth_bins = 96
        self.depth_out_embed = nn.Embedding(self.max_depth_bins, self.embed_dims)
        nn.init.normal_(self.depth_out_embed.weight, mean=0.0, std=0.02)

    def _init_ref_2d(self, H, W):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H),
            torch.linspace(0.5, W - 0.5, W),
            indexing='ij'
        )
        ref_y = ref_y.reshape(-1) / H
        ref_x = ref_x.reshape(-1) / W
        ref_2d = torch.stack([ref_x, ref_y], dim=-1)  # [H*W, 2]
        return ref_2d.view(1, -1, 1, 2)              # [1, H*W, 1, 2]

    @torch.no_grad()
    def _build_ref_flat_from_grids(self, grids_tensor, B, device, dtype):
        """
        grids_tensor: list(num_cams), each [H,W,D,2]
        return:
          ref_flat: [B, H*W*D, num_cams, 2]
          H, W, D
        """
        assert isinstance(grids_tensor, (list, tuple)), "grids_tensor must be list/tuple"
        assert len(grids_tensor) == self.num_cams, f"Expected {self.num_cams} cams, got {len(grids_tensor)}"

        g0 = grids_tensor[0]
        assert torch.is_tensor(g0) and g0.dim() == 4 and g0.shape[-1] == 2
        H, W, D, _ = g0.shape
        HW = H * W

        grids = []
        for c in range(self.num_cams):
            g = grids_tensor[c]
            assert tuple(g.shape) == (H, W, D, 2), f"grid[{c}] shape mismatch: {g.shape}"
            grids.append(g.to(device=device, dtype=dtype))

        # [num_cams, H, W, D, 2] -> [H, W, D, num_cams, 2]
        grid_stack = torch.stack(grids, dim=0).permute(1, 2, 3, 0, 4).contiguous()
        # [HW, D, num_cams, 2]
        ref_hw_d = grid_stack.view(HW, D, self.num_cams, 2)
        # [B, HW*D, num_cams, 2]
        ref_flat = ref_hw_d.view(1, HW * D, self.num_cams, 2).expand(B, -1, -1, -1)
        return ref_flat, H, W, D

    def forward(self, fisheye_feats, grids_tensor):
        B = fisheye_feats.shape[0]
        device = fisheye_feats.device
        dtype = fisheye_feats.dtype

        # 1) build reference_points for ALL depths (flattened)
        ref_flat, H, W, D = self._build_ref_flat_from_grids(grids_tensor, B, device, dtype)
        Lq = H * W

        # 2) spatial base queries: [B, Lq, C]
        if Lq == self.equi_h * self.equi_w:
            erp_base = self.erp_embed.weight.unsqueeze(0).expand(B, -1, -1).to(dtype)
        else:
            erp_base = self.erp_embed.weight[:Lq].unsqueeze(0).expand(B, -1, -1).to(dtype)

        # 3) positional encoding: [B, Lq, C] 로 맞춰서 사용한다고 가정
        pos_embed = self.pos_encoding(H, W, device).to(dtype)
        if pos_embed.dim() == 2:  # [Lq, C]
            pos_embed = pos_embed.unsqueeze(0).expand(B, -1, -1)  # [B, Lq, C]
        elif pos_embed.dim() == 3 and pos_embed.shape[-1] == 1:   # [Lq, C, 1] 같은 경우
            pos_embed = pos_embed.squeeze(-1).unsqueeze(0).expand(B, -1, -1)  # [B, Lq, C]

        # 4) depth values: [D] -> [B, Lq, D, 1]
        depth_vals = torch.arange(D, device=device, dtype=dtype)
        if self.normalize_depth and D > 1:
            depth_vals = depth_vals / float(D - 1)
        depth_lqd1 = depth_vals.view(1, 1, D, 1).expand(B, Lq, D, 1)

        depth_enc = self.depth_encoder(depth_lqd1.reshape(B, Lq * D, 1))  # [B, Lq*D, C]
        depth_enc = depth_enc.view(B, Lq, D, self.embed_dims)

        # 5) expand spatial query to depth tokens: [B, Lq, D, C] -> [B, Lq*D, C]
        query = erp_base.unsqueeze(2).expand(B, Lq, D, self.embed_dims) + depth_enc
        query = query.reshape(B, Lq * D, self.embed_dims)

        # pos also expanded to tokens
        pos_tok = pos_embed.unsqueeze(2).expand(B, Lq, D, self.embed_dims).reshape(B, Lq * D, self.embed_dims)

        # 6) self-attn ref_2d expanded to Lq*D tokens if needed
        if self.use_deform_self_attn:
            if H == self.equi_h and W == self.equi_w:
                ref2d = self.ref_2d.expand(B, -1, -1, -1).to(dtype)  # [B,Lq,1,2]
            else:
                ref2d = self._init_ref_2d(H, W).to(device=device, dtype=dtype).expand(B, -1, -1, -1)

            ref2d_tok = ref2d.repeat_interleave(D, dim=1)  # [B, Lq*D, 1, 2]
        else:
            ref2d_tok = None

        # 7) MSDA spatial shapes (cam을 level로 취급)
        feat_spatial_shapes = torch.tensor(
            [[fisheye_feats.shape[3], fisheye_feats.shape[4]]] * self.num_cams,
            device=device
        )

        # 8) cross-attn layers
        for layer in self.layers:
            query = layer(
                query, fisheye_feats, ref_flat, feat_spatial_shapes,
                pos_tok, ref2d_tok, (H, W)
            )  # [B, Lq*D, C]

        # 9) reshape to ONE tensor: [B, C*D, H, W]
        # query: [B, Lq*D, C] -> [B, C, Lq, D] -> [B, C, H, W, D] -> [B, C*D, H, W]
        query = query.view(B, Lq, D, self.embed_dims)              # [B, Lq, D, C]
        query = query.permute(0, 3, 1, 2).contiguous()             # [B, C, Lq, D]
        query = query.view(B, self.embed_dims, H, W, D)            # [B, C, H, W, D]
        erp_feat = query.permute(0, 1, 4, 2, 3).contiguous()       # [B, C, D, H, W]

        ## 10) depth embedding
        depth_ids = torch.arange(D, device=device, dtype=torch.long)     # [D]
        depth_e = self.depth_out_embed(depth_ids).T                      # [C,D]
        depth_e = depth_e.view(1, self.embed_dims, D, 1, 1)              # [1,C,D,1,1]

        erp_feat = erp_feat + depth_e                                    # [B,C,D,H,W]
        erp_feat = erp_feat.view(B, self.embed_dims * D, H, W)

        return erp_feat

