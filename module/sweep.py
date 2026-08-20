import torch
import torch.nn.functional as F


def spherical_sweep(fisheye_feats, grids):
    """
    Build spherical (ERP) feature volumes by sampling fisheye features with precomputed grids,

    Args:
        fisheye_feats:
            - list/tuple of 4 tensors, each [B, C, H_fish, W_fish], or
            - tensor [B, 4, C, H_fish, W_fish]
        grids:
            list of 4 tensors, each [H_erp, W_erp, D, 2] in grid_sample normalized coords.

    Returns:
        sph_feats: list of 4 tensors [cam0_vol, cam1_vol, cam2_vol, cam3_vol],
        where cam*_vol is [B, C, H_erp, W_erp, D].
    """
    if isinstance(fisheye_feats, (list, tuple)):
        assert len(fisheye_feats) == 4, "fisheye_feats list/tuple must have length 4"
        feats = list(fisheye_feats)
    else:
        assert fisheye_feats.ndim == 5 and fisheye_feats.shape[1] == 4, "fisheye_feats must be [B,4,C,H,W]"
        feats = [fisheye_feats[:, i] for i in range(4)]

    bs = feats[0].shape[0]
    device = feats[0].device
    dtype = feats[0].dtype

    grids = [g.to(device=device, dtype=dtype) for g in grids]
    grids_pad = [torch.cat([torch.zeros_like(g[..., :1]), g], dim=-1) for g in grids]  # [H,W,D,3]

    sph_feats = []
    for feat, grid in zip(feats, grids_pad):
        # Use 5D grid_sample trick (treat last dim as dummy "W"=1) to sample all depths in one call.
        sph_feat = F.grid_sample(feat[..., None], grid.repeat(bs, 1, 1, 1, 1), align_corners=True)
        sph_feats.append(sph_feat)

    return sph_feats

