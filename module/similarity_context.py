# module/similarity_context.py
# Similarity Profile Encoder and Context Lookup for depth estimation
# Provides explicit matching signal by computing feature consistency across depth levels

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import DeformConv2d


class DeformConv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, use_dcn=True):
        super().__init__()
        self.use_dcn = use_dcn
        if self.use_dcn:
            self.offset = nn.Conv2d(
                in_channels,
                2 * kernel_size * kernel_size,
                kernel_size=kernel_size,
                padding=padding
            )
            nn.init.zeros_(self.offset.weight)
            nn.init.zeros_(self.offset.bias)
            self.conv = DeformConv2d(
                in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True
            )
        else:
            self.offset = None
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True
            )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.use_dcn:
            offset = self.offset(x)
            x = self.conv(x, offset)
        else:
            x = self.conv(x)
        return self.act(x)


class BasicConv3d(nn.Module):
    """Conv3d + GroupNorm + LeakyReLU building block for 3D hourglass."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, deconv=False):
        super().__init__()
        if deconv:
            self.conv = nn.ConvTranspose3d(
                in_ch, out_ch, kernel_size=kernel_size, stride=2,
                padding=padding, output_padding=1, bias=False,
            )
        else:
            self.conv = nn.Conv3d(
                in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                padding=padding, bias=False,
            )
        num_groups = min(8, out_ch)
        self.norm = nn.GroupNorm(num_groups, out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Conv3dBnReLU(nn.Module):
    """Conv3d + BatchNorm3d + ReLU building block for 3D UNet."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DownBlock3D(nn.Module):
    """Stride-2 downsample + 2x Conv3d for 3D UNet encoder."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = Conv3dBnReLU(in_ch, out_ch, stride=2)
        self.conv1 = Conv3dBnReLU(out_ch, out_ch)
        self.conv2 = Conv3dBnReLU(out_ch, out_ch)

    def forward(self, x):
        x = self.down(x)
        x = self.conv1(x)
        return self.conv2(x)


class UpBlock3D(nn.Module):
    """ConvTranspose3d upsample + skip concat + 2x Conv3d for 3D UNet decoder."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=3, stride=2,
                                     padding=1, output_padding=1, bias=False)
        self.bn = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        # After skip concat: out_ch + out_ch = 2*out_ch
        self.conv1 = Conv3dBnReLU(out_ch * 2, out_ch)
        self.conv2 = Conv3dBnReLU(out_ch, out_ch)

    def forward(self, x, skip):
        x = self.relu(self.bn(self.up(x)))
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        return self.conv2(x)


class RegularizationNet3D(nn.Module):
    """
    3-stage 3D UNet with Guided Cost Volume Excitation.
    Input: [B, in_ch, D, H, W], Output: [B, 1, D, H, W].
    Guide features modulate each encoder level via channel-wise sigmoid gating.
    """
    def __init__(self, in_channels, channels=(16, 32, 48)):
        super().__init__()
        c1, c2, c3 = channels

        # Stem
        self.stem = nn.Sequential(
            Conv3dBnReLU(in_channels, c1),
            Conv3dBnReLU(c1, c1),
        )

        # Encoder
        self.down1 = DownBlock3D(c1, c2)
        self.down2 = DownBlock3D(c2, c3)

        # Decoder
        self.up2 = UpBlock3D(c3, c2)
        self.up1 = UpBlock3D(c2, c1)

        # Output
        self.out_conv = nn.Conv3d(c1, 1, 1, bias=True)

    def _excite(self, cost_vol, guide_feat):
        """Guided Cost Volume Excitation: cost_vol * sigmoid(guide_feat).unsqueeze(2)"""
        return cost_vol * torch.sigmoid(guide_feat).unsqueeze(2)

    def forward(self, x, guide_features=None):
        """
        Args:
            x: [B, in_ch, D, H, W]
            guide_features: list of 3 tensors matching each encoder level's channels
                and spatial resolution. If None, no excitation is applied.
        Returns:
            [B, 1, D, H, W]
        """
        s0 = self.stem(x)
        if guide_features is not None:
            s0 = self._excite(s0, guide_features[0])

        s1 = self.down1(s0)
        if guide_features is not None:
            s1 = self._excite(s1, guide_features[1])

        s2 = self.down2(s1)
        if guide_features is not None:
            s2 = self._excite(s2, guide_features[2])

        x = self.up2(s2, s1)
        x = self.up1(x, s0)

        return self.out_conv(x)


def groupwise_correlation_volume(ref, tgt, num_groups=8):
    """
    Group-wise correlation between ref and tgt feature volumes.
    Args:
        ref, tgt: [B, C, H, W, Ds]
        num_groups: number of groups for dot product
    Returns:
        gwc: [B, num_groups, H, W, Ds]
    """
    B, C, H, W, Ds = ref.shape
    assert C % num_groups == 0, f"C({C}) must be divisible by num_groups({num_groups})"
    channels_per_group = C // num_groups
    scale = (channels_per_group) ** 0.5

    ref = ref.view(B, num_groups, channels_per_group, H, W, Ds)
    tgt = tgt.view(B, num_groups, channels_per_group, H, W, Ds)
    gwc = (ref * tgt).sum(dim=2) / scale  # [B, num_groups, H, W, Ds]
    return gwc


class GEVModule(nn.Module):
    """
    Independent Geometry Encoding Volume module.

    Builds a GEV from ref/tgt volumes (GWC + multi-view variance) and regularizes
    it with a 3-stage 3D UNet (RegularizationNet3D) with guided cost volume excitation.
    Provides a multi-resolution pyramid for depth-axis lookup.
    """
    def __init__(self, in_channels=32, num_groups=8,
                 reg_channels=(16, 32, 48), num_pyramid_levels=2,
                 use_spatial_downsample=True):
        super().__init__()
        self.num_groups = num_groups
        self.num_pyramid_levels = num_pyramid_levels
        self.use_spatial_downsample = use_spatial_downsample

        hg_in = num_groups * 2  # GWC(num_groups) + variance(num_groups)
        reg_channels = tuple(reg_channels)
        self.regularization = RegularizationNet3D(hg_in, channels=reg_channels)

        # Guide feature projections (one per encoder level)
        self.guide_projs = nn.ModuleList([
            nn.Conv2d(in_channels, ch, 1) for ch in reg_channels
        ])

        self._gev_pyramid = None

    def build(self, ref, tgt, camera_volumes, use_triton=False):
        """
        Build GEV from ref/tgt volumes and individual camera volumes.

        Args:
            ref: [B, C, H, W, Ds] — view-mixed reference volume
            tgt: [B, C, H, W, Ds] — view-mixed target volume
            camera_volumes: tuple of 4 tensors [B, C, H, W, Ds] — per-camera volumes
            use_triton: if True, use Triton fused kernels to reduce memory
        """
        if use_triton:
            from module.triton_gev import fused_gwc, fused_group_variance
            # 1. GWC — Triton fused (no [B,G,CPG,H,W,Ds] intermediate)
            gwc = fused_gwc(ref, tgt, self.num_groups)
            # 2. Multi-view group variance — Triton fused (no 1.2GB stack)
            f0, f1, f2, f3 = camera_volumes
            group_var = fused_group_variance(f0, f1, f2, f3, self.num_groups)
        else:
            # 1. GWC (pairwise similarity)
            gwc = groupwise_correlation_volume(ref, tgt, self.num_groups)
            # 2. Multi-view group variance (4-camera consensus)
            B, C, H, W, Ds = camera_volumes[0].shape
            G = self.num_groups
            cpg = C // G
            stacked = torch.stack(list(camera_volumes), dim=0)
            stacked = stacked.view(4, B, G, cpg, H, W, Ds)
            group_var = stacked.var(dim=0).mean(dim=2)

        # 3. Concat → permute to [B, 2G, Ds, H, W]
        vol = torch.cat([gwc, group_var], dim=1)              # [B, 2G, H, W, Ds]
        vol = vol.permute(0, 1, 4, 2, 3).contiguous()        # [B, 2G, Ds, H, W]

        # 4. Optional spatial downsample
        if self.use_spatial_downsample:
            vol = F.avg_pool3d(vol, kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # 5. Guide features from ref (averaged over depth)
        ref_2d = ref.mean(dim=-1)  # [B, C, H, W]
        if self.use_spatial_downsample:
            ref_2d = F.avg_pool2d(ref_2d, kernel_size=2, stride=2)

        guide_features = []
        for i, proj in enumerate(self.guide_projs):
            g = proj(ref_2d)
            if i > 0:
                g = F.avg_pool2d(g, kernel_size=2 ** i, stride=2 ** i)
            guide_features.append(g)

        # 6. Regularization (3D UNet with guided excitation)
        gev = self.regularization(vol, guide_features)  # [B, 1, Ds, H', W']

        # 7. Optional spatial upsample
        if self.use_spatial_downsample:
            gev = F.interpolate(gev, scale_factor=(1, 2, 2),
                                mode='trilinear', align_corners=False)

        # 8. Build depth-axis pyramid
        self._gev_pyramid = [gev]
        cur = gev
        for lvl in range(1, self.num_pyramid_levels):
            cur = F.avg_pool3d(cur, kernel_size=(2, 1, 1), stride=(2, 1, 1))
            self._gev_pyramid.append(cur)

    def lookup(self, current_depth_idx, radius=4):
        """
        Multi-level pyramid lookup on GEV along depth axis.

        Args:
            current_depth_idx: [B, 1, H, W]
            radius: lookup radius around current depth
        Returns:
            gev_feat: [B, num_pyramid_levels * (2*radius+1), H, W]
        """
        assert self._gev_pyramid is not None, "Must call build() first"

        B, _, H, W = current_depth_idx.shape
        device = current_depth_idx.device
        dtype = current_depth_idx.dtype

        idx = current_depth_idx.squeeze(1)  # [B, H, W]

        out_levels = []
        for lvl, gev_vol in enumerate(self._gev_pyramid):
            # gev_vol: [B, 1, D_lvl, H, W]
            D_lvl = gev_vol.shape[2]

            # Scale index for this pyramid level
            scaled_idx = idx / (2 ** lvl)  # [B, H, W]

            # Sample points around current depth
            dx = torch.linspace(-radius, radius, 2 * radius + 1,
                                device=device, dtype=dtype)
            sample_idx = scaled_idx.unsqueeze(-1) + dx  # [B, H, W, 2r+1]
            sample_idx = sample_idx.clamp(0, D_lvl - 1)

            # Bilinear interpolation along depth
            idx_floor = sample_idx.long().clamp(0, D_lvl - 1)
            idx_ceil = (idx_floor + 1).clamp(0, D_lvl - 1)
            weight = (sample_idx - idx_floor.float()).clamp(0, 1)

            # [B, 1, D_lvl, H, W] → [B, H, W, D_lvl]
            gev_flat = gev_vol.squeeze(1).permute(0, 2, 3, 1)

            val_floor = torch.gather(gev_flat, 3, idx_floor)   # [B, H, W, 2r+1]
            val_ceil = torch.gather(gev_flat, 3, idx_ceil)
            sampled = val_floor * (1 - weight) + val_ceil * weight

            # [B, H, W, 2r+1] → [B, 2r+1, H, W]
            out_levels.append(sampled.permute(0, 3, 1, 2))

        return torch.cat(out_levels, dim=1)

    def lookup_dim(self, radius=4):
        """Return output dimension of lookup()."""
        return self.num_pyramid_levels * (2 * radius + 1)

    def clear_cache(self):
        """Clear cached pyramid volumes."""
        self._gev_pyramid = None


class MLP2D(nn.Module):
    """
    2D version of `module/volume_generator.py::MLP`:
    point-wise (1x1) conv MLP that predicts a sigmoid weight map in [0, 1].
    """
    def __init__(self, ch_in: int, ch_hid: int, ch_out: int = 1):
        super().__init__()
        self.linear1 = nn.Conv2d(ch_in, ch_hid, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv2d(ch_hid, ch_out, kernel_size=1)
        self.out_act = nn.Sigmoid()

    def forward(self, feat_a, feat_b, grid_a, grid_b):
        """Predict a per-pixel, per-depth blend weight for a camera pair.

        The convs are 1x1, so they are applied directly on the 5D volumes as
        1x1x1 conv3d instead of folding depth into the batch dimension. Folding
        needs `permute(...).contiguous()` on both feature volumes, which
        physically copies the whole [B,C,H,W,Ds] tensor twice per call and
        dominated this module's runtime. Math and weights are unchanged — the
        Conv2d params are kept (checkpoint compatibility) and only reshaped.

        Args:
            feat_a, feat_b: [B, C, H, W, Ds] - sampled feature volumes
            grid_a, grid_b: [H, W, Ds, 2]    - sampling grids of those cameras
        Returns:
            w: [B, 1, H, W, Ds] in [0, 1]
        """
        B = feat_a.shape[0]

        def _align_grid(g):                     # [H,W,Ds,2] -> [B,2,H,W,Ds] (view only)
            g = g.permute(3, 0, 1, 2)
            return g.unsqueeze(0).expand(B, -1, -1, -1, -1).to(feat_a.dtype)

        x = torch.cat([feat_a, feat_b, _align_grid(grid_a), _align_grid(grid_b)], dim=1)

        x = F.conv3d(x, self.linear1.weight.unsqueeze(-1), self.linear1.bias)
        x = self.relu(x)
        x = F.conv3d(x, self.linear2.weight.unsqueeze(-1), self.linear2.bias)
        return self.out_act(x)                  # [B, 1, H, W, Ds]


class SimilarityProfileEncoder(nn.Module):
    """
    각 ERP 픽셀에서 depth별 feature similarity를 계산하고 인코딩.
    RAFT의 Correlation Pyramid와 유사한 역할.
    
    핵심: 올바른 depth에서는 4개 카메라 feature가 일치 → similarity 높음
    """
    def __init__(
        self,
        num_depth_samples=32,
        embed_dims=32,
        num_cams=4,
        similarity_type='correlation',
    ):
        super().__init__()
        self.num_depth_samples = num_depth_samples
        self.num_cams = num_cams
        self.similarity_type = similarity_type

        # Learned view mixing (like `volume_generator.py`):
        # cam0+cam2 -> reference, cam1+cam3 -> target.
        #
        # We predict per-pixel weights using (feat_a, feat_b, grid_a(xy), grid_b(xy)).
        mlp_in = 2 * embed_dims + 4  # (feat0, feat2) + (grid0, grid2)
        self.reference_mapping = MLP2D(mlp_in, embed_dims)
        self.target_mapping = MLP2D(mlp_in, embed_dims)

        # NOTE: kept only for checkpoint compatibility - `forward` never calls it.
        # See the Tier 3 cleanup notes before removing (needs a ckpt migration).
        self.encoder = nn.Sequential(
            nn.Conv2d(num_depth_samples, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, embed_dims, 1),
        )


    def forward(self, fisheye_feats, grids, num_invdepth=192, spherical_volumes=None,
                return_volumes=False, use_triton=False):
        """
        Args:
            fisheye_feats: [B, 4, C, H_fish, W_fish] or list of [B, C, H_fish, W_fish]
            grids: list of 4 tensors, each [H_erp, W_erp, D, 2]
            num_invdepth: total number of depth levels (e.g., 192)
            spherical_volumes: optional list/tuple of 4 tensors, each [B, C, H_erp, W_erp, D]
                If provided, we will reuse these pre-swept volumes and avoid grid_sample.
            return_volumes: if True, return (similarity_profile, ref, tgt) tuple
        Returns:
            similarity_profile: [B, num_depth_samples, H_erp, W_erp]
            (if return_volumes): (similarity_profile, ref, tgt) where ref,tgt are [B,C,H,W,Ds]
        """
        if spherical_volumes is not None:
            assert isinstance(spherical_volumes, (list, tuple)) and len(spherical_volumes) == 4, \
                "spherical_volumes must be a list/tuple of 4 volumes [B,C,H,W,D]"
            B = spherical_volumes[0].shape[0]
            device = spherical_volumes[0].device
            dtype = spherical_volumes[0].dtype
        else:
            # Handle both stacked and list input
            if isinstance(fisheye_feats, (list, tuple)):
                fisheye_feats = torch.stack(fisheye_feats, dim=1)
            B = fisheye_feats.shape[0]
            device = fisheye_feats.device
            dtype = fisheye_feats.dtype

        H_erp, W_erp, D_total, _ = grids[0].shape
        # Sparse depth sampling: use actual grid depth (D_total) as basis
        depth_step = max(1, D_total // self.num_depth_samples)
        depth_indices = list(range(0, D_total, depth_step))[: self.num_depth_samples]
        while len(depth_indices) < self.num_depth_samples:
            depth_indices.append(depth_indices[-1])
        
        depth_idx = torch.tensor(depth_indices, device=device, dtype=torch.long)
        Ds = int(depth_idx.numel())

        def _select_grids(cam_idx: int) -> torch.Tensor:
            # [H, W, D, 2] -> [H, W, Ds, 2]
            g = grids[cam_idx].to(device=device, dtype=dtype)
            return g.index_select(dim=2, index=depth_idx)

        def _sample_volume(feat_4d: torch.Tensor, g_sel: torch.Tensor) -> torch.Tensor:
            """
            Vectorized 2D grid_sample across Ds depths by folding Ds into the batch dimension.

            Args:
                feat_4d: [B, C, H_fish, W_fish]
                g_sel:  [H_erp, W_erp, Ds, 2] (normalized coords)
            Returns:
                vol: [B, C, H_erp, W_erp, Ds]
            """
            # grids: [H, W, Ds, 2] -> [B*Ds, H, W, 2]
            g_bd = (
                g_sel.permute(2, 0, 1, 3)  # [Ds, H, W, 2]
                .unsqueeze(0)
                .expand(B, -1, -1, -1, -1)
                .reshape(B * Ds, H_erp, W_erp, 2)
                .contiguous()
            )
            # feats: [B, C, Hf, Wf] -> [B*Ds, C, Hf, Wf]
            feat_bd = (
                feat_4d.unsqueeze(1)
                .expand(B, Ds, -1, -1, -1)
                .reshape(B * Ds, feat_4d.shape[1], feat_4d.shape[2], feat_4d.shape[3])
                .contiguous()
            )
            sampled = F.grid_sample(
                feat_bd,
                g_bd,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )  # [B*Ds, C, H, W]
            vol = sampled.view(B, Ds, sampled.shape[1], H_erp, W_erp).permute(0, 2, 3, 4, 1).contiguous()
            return vol

        # Build per-camera sampled feature volumes: [B, C, H, W, Ds]
        g0_sel, g1_sel, g2_sel, g3_sel = (_select_grids(0), _select_grids(1), _select_grids(2), _select_grids(3))
        if spherical_volumes is not None:
            # Use stride slicing instead of index_select to avoid copying
            # when depth_step==1 (Ds==D_total), this is a zero-copy view
            depth_indices_list = depth_indices  # Python list from above
            if depth_step == 1 and Ds == D_total:
                # All depths selected — direct reference, no copy
                f0_vol = spherical_volumes[0].to(device=device, dtype=dtype)
                f1_vol = spherical_volumes[1].to(device=device, dtype=dtype)
                f2_vol = spherical_volumes[2].to(device=device, dtype=dtype)
                f3_vol = spherical_volumes[3].to(device=device, dtype=dtype)
            else:
                # Sparse sampling — use stride slicing for regular steps
                f0_vol = spherical_volumes[0].to(device=device, dtype=dtype)[:, :, :, :, ::depth_step][:, :, :, :, :Ds].contiguous()
                f1_vol = spherical_volumes[1].to(device=device, dtype=dtype)[:, :, :, :, ::depth_step][:, :, :, :, :Ds].contiguous()
                f2_vol = spherical_volumes[2].to(device=device, dtype=dtype)[:, :, :, :, ::depth_step][:, :, :, :, :Ds].contiguous()
                f3_vol = spherical_volumes[3].to(device=device, dtype=dtype)[:, :, :, :, ::depth_step][:, :, :, :, :Ds].contiguous()
        else:
            f0_vol = _sample_volume(fisheye_feats[:, 0], g0_sel)
            f1_vol = _sample_volume(fisheye_feats[:, 1], g1_sel)
            f2_vol = _sample_volume(fisheye_feats[:, 2], g2_sel)
            f3_vol = _sample_volume(fisheye_feats[:, 3], g3_sel)
            # Learned view mixing (mirrors `volume_generator.py`), fallback to 0.5 average.
            # Flatten depth into batch to apply 2D conv MLP per depth slice.
            # feats: [B, C, H, W, Ds] -> [B*Ds, C, H, W]

        # def _flatten_depth(x: torch.Tensor) -> torch.Tensor:
        #     return x.permute(0, 4, 1, 2, 3).reshape(B * Ds, x.shape[1], H_erp, W_erp).contiguous()

        # # grids: [H, W, Ds, 2] -> [B*Ds, 2, H, W]
        # def _grid_flat(g_sel: torch.Tensor) -> torch.Tensor:
        #     gd = g_sel.permute(2, 3, 0, 1).unsqueeze(0).expand(B, -1, -1, -1, -1)
        #     return gd.reshape(B * Ds, 2, H_erp, W_erp).contiguous()

        # f0_bd, f1_bd, f2_bd, f3_bd = map(_flatten_depth, (f0_vol, f1_vol, f2_vol, f3_vol))
        # g0_bd, g1_bd, g2_bd, g3_bd = map(_grid_flat, (g0_sel, g1_sel, g2_sel, g3_sel))

        # w_front = self.reference_mapping(torch.cat([f0_bd, f2_bd, g0_bd, g2_bd], dim=1))  # [B*Ds, 1, H, W]
        # w_right = self.target_mapping(torch.cat([f1_bd, f3_bd, g1_bd, g3_bd], dim=1))  # [B*Ds, 1, H, W]

        # w_front = w_front.view(B, Ds, 1, H_erp, W_erp).permute(0, 2, 3, 4, 1).contiguous()
        # w_right = w_right.view(B, Ds, 1, H_erp, W_erp).permute(0, 2, 3, 4, 1).contiguous()

        w_front = self.reference_mapping(f0_vol, f2_vol, g0_sel, g2_sel)
        w_right = self.target_mapping(f1_vol, f3_vol, g1_sel, g3_sel)

        ref = w_front * f0_vol + (1.0 - w_front) * f2_vol
        tgt = w_right * f1_vol + (1.0 - w_right) * f3_vol

        if use_triton:
            from module.triton_gev import fused_similarity
            sim_vol = fused_similarity(ref, tgt)  # [B, H, W, Ds] — no [B,C,H,W,Ds] intermediate
        else:
            denom = ref.new_tensor(ref.shape[1]).sqrt()
            sim_vol = (ref * tgt).sum(dim=1) / denom  # [B, H, W, Ds]

        # [B, H, W, Ds] -> [B, Ds, H, W]
        similarity_profile = sim_vol.permute(0, 3, 1, 2).contiguous()

        if return_volumes:
            return similarity_profile, ref, tgt, (f0_vol, f1_vol, f2_vol, f3_vol)
        return similarity_profile


class SimilarityContextLookup(nn.Module):
    """
    현재 depth 주변의 similarity 정보를 lookup.
    RAFT의 CorrBlock1D와 유사한 multi-scale pyramid 구조.
    """
    def __init__(self, num_depth_samples=32, num_invdepth=192, radius=4, num_levels=4):
        super().__init__()
        self.num_depth_samples = num_depth_samples
        self.num_invdepth = num_invdepth
        self.radius = radius
        self.num_levels = num_levels
        
        # Output dimension
        self.output_dim = num_levels * (2 * radius + 1)
    
    def forward(self, similarity_profile, current_depth_idx):
        """
        Args:
            similarity_profile: [B, num_depth_samples, H, W]
            current_depth_idx: [B, 1, H, W] - 현재 추정 depth index (0 ~ D-1),
                               where D = num_depth_samples (downsampled grid depth).
                               This directly corresponds to profile indices (no scale needed).
        Returns:
            context: [B, num_levels * (2*radius+1), H, W]
        """
        B, D, H_prof, W_prof = similarity_profile.shape
        _, _, H, W = current_depth_idx.shape
        device = similarity_profile.device
        dtype = similarity_profile.dtype
        # Profile bins map 1:1 to grid depth indices (D_total-based sampling),
        # so current_depth_idx directly indexes the profile — no scale needed.
        profile_idx = current_depth_idx.squeeze(1)  # [B, H, W]
        
        # Build multi-scale pyramid and sample
        out_pyramid = []
        
        # Reshape: [B, D, H*W] → [B, H*W, D] so pooling applies to D dimension
        sim_flat = similarity_profile.view(B, D, H * W).permute(0, 2, 1)  # [B, H*W, D]
        
        for level in range(self.num_levels):
            # Pooling along depth dimension (last dim)
            if level > 0:
                sim_flat = F.avg_pool1d(sim_flat, kernel_size=2, stride=2)  # [B, H*W, D/2]
            
            D_level = sim_flat.shape[2]  # depth dimension is now last
            
            # Scale current index for this level
            scaled_idx = profile_idx / (2 ** level)  # [B, H, W]
            scaled_idx = scaled_idx.view(B, -1)  # [B, H*W]
            
            # Sample points around current depth
            r = self.radius
            dx = torch.linspace(-r, r, 2 * r + 1, device=device, dtype=dtype)  # [2r+1]
            
            # Sample locations: [B, H*W, 2r+1]
            sample_idx = scaled_idx.unsqueeze(-1) + dx.unsqueeze(0).unsqueeze(0)
            sample_idx = sample_idx.clamp(0, D_level - 1)
            
            # Bilinear interpolation
            idx_floor = sample_idx.long().clamp(0, D_level - 1)
            idx_ceil = (idx_floor + 1).clamp(0, D_level - 1)
            weight = (sample_idx - idx_floor.float()).clamp(0, 1)
            
            # Gather values from sim_flat: [B, H*W, D_level]
            sampled_floor = torch.gather(sim_flat, 2, idx_floor)  # [B, H*W, 2r+1]
            sampled_ceil = torch.gather(sim_flat, 2, idx_ceil)  # [B, H*W, 2r+1]
            
            sampled = sampled_floor * (1 - weight) + sampled_ceil * weight  # [B, H*W, 2r+1]
            
            out_pyramid.append(sampled)
        
        # Concatenate all levels: [B, H*W, num_levels * (2r+1)]
        out = torch.cat(out_pyramid, dim=-1)
        
        # Reshape to spatial: [B, num_levels * (2r+1), H, W]
        out = out.permute(0, 2, 1).view(B, -1, H, W)
        
        return out


class SpatialProfileAggregator(nn.Module):
    """
    Spatial aggregation for similarity profile.
    
    Takes [B, Ds, H, W] profile (depth samples as channels) and applies
    2D spatial convolutions to incorporate neighboring pixel information.
    
    This helps with:
    - Textureless regions (neighboring pixels provide context)
    - Noise reduction (spatial averaging)
    - Instance-aware smoothing (learned receptive field)
    """
    def __init__(
        self,
        num_depth_samples: int,
        hidden_dim: int = 32,
        num_layers: int = 2,
        use_residual: bool = True,
        use_deformable: bool = False,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.use_deformable = use_deformable
        
        layers = []
        in_ch = num_depth_samples
        
        for i in range(num_layers):
            out_ch = hidden_dim if i < num_layers - 1 else num_depth_samples
            
            if use_deformable:
                # Deformable conv for instance-aware aggregation
                layers.append(DeformConv2dBlock(in_ch, out_ch, kernel_size=3, padding=1, use_dcn=True))
            else:
                # Standard conv
                layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
                if i < num_layers - 1:
                    layers.append(nn.GroupNorm(min(8, out_ch), out_ch))
                    layers.append(nn.ReLU(inplace=True))
            
            in_ch = out_ch
        
        self.conv = nn.Sequential(*layers)
        
        # Optional: learnable residual weight (start from 0 for stable training)
        if use_residual:
            self.residual_weight = nn.Parameter(torch.zeros(1))
    
    def forward(self, sim_profile: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sim_profile: [B, Ds, H, W] - depth samples as channels
        Returns:
            aggregated: [B, Ds, H, W] - spatially aggregated profile
        """
        out = self.conv(sim_profile)
        
        if self.use_residual:
            # Learnable residual: starts from original, gradually learns to aggregate
            alpha = torch.sigmoid(self.residual_weight)
            return sim_profile + alpha * out
        else:
            return out


class SimilarityContext(nn.Module):
    """
    Combined module: Profile Encoder + Spatial Aggregation + Context Lookup
    """
    def __init__(
        self,
        num_depth_samples=96,
        num_invdepth=192,
        embed_dims=32,
        radius=4,
        num_levels=4,
        similarity_type='correlation',
        # Spatial aggregation options
        use_spatial_aggregation: bool = True,
        spatial_agg_type: str = 'conv',  # 'conv', 'multiscale', 'deformable'
        spatial_hidden_dim: int = 32,
        spatial_num_layers: int = 2,
    ):
        super().__init__()

        self.encoder = SimilarityProfileEncoder(
            num_depth_samples=num_depth_samples,
            embed_dims=embed_dims,
            similarity_type=similarity_type,
        )

        # Spatial aggregation module
        self.use_spatial_aggregation = use_spatial_aggregation
        self.spatial_aggregator = None
        
        if use_spatial_aggregation:
            if spatial_agg_type == 'conv':
                self.spatial_aggregator = SpatialProfileAggregator(
                    num_depth_samples=num_depth_samples,
                    hidden_dim=spatial_hidden_dim,
                    num_layers=spatial_num_layers,
                    use_residual=True,
                    use_deformable=False,
                )
            elif spatial_agg_type == 'deformable':
                self.spatial_aggregator = SpatialProfileAggregator(
                    num_depth_samples=num_depth_samples,
                    hidden_dim=spatial_hidden_dim,
                    num_layers=spatial_num_layers,
                    use_residual=True,
                    use_deformable=True,
                )
            else:
                raise ValueError(f"Unknown spatial_agg_type: {spatial_agg_type}")
        
        self.lookup = SimilarityContextLookup(
            num_depth_samples=num_depth_samples,
            num_invdepth=num_invdepth,
            radius=radius,
            num_levels=num_levels
        )

        self.output_dim = self.lookup.output_dim
        self.num_invdepth = num_invdepth
    
    def build_profile(self, fisheye_feats=None, grids=None, spherical_volumes=None,
                       return_volumes=False, use_triton=False):
        """
        Build similarity profile (call once per forward).

        You can either provide:
        - fisheye_feats + grids (will grid_sample internally), or
        - spherical_volumes + grids (reuse pre-swept ERP×depth volumes; avoids grid_sample).

        Args:
            return_volumes: if True, also return (ref, tgt, cam_vols) from the encoder.
        Returns:
            similarity_profile: [B, num_depth_samples, H, W] - spatially aggregated if enabled
            (if return_volumes): (similarity_profile, ref, tgt, cam_vols)
        """
        assert grids is not None, "grids must be provided"

        # Build raw similarity profile
        result = self.encoder(
            fisheye_feats, grids, self.num_invdepth,
            spherical_volumes=spherical_volumes,
            return_volumes=return_volumes,
            use_triton=use_triton
        )

        if return_volumes:
            profile, ref, tgt, cam_vols = result
        else:
            profile = result

        # Apply spatial aggregation if enabled
        if self.use_spatial_aggregation and self.spatial_aggregator is not None:
            profile = self.spatial_aggregator(profile)

        if return_volumes:
            return profile, ref, tgt, cam_vols
        return profile
    
    def lookup_context(self, similarity_profile, current_depth_idx):
        """Lookup context at current depth (call at each iteration)"""
        return self.lookup(similarity_profile, current_depth_idx)
