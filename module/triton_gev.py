"""
Triton fused kernels for GEV construction and similarity computation.

1. fused_group_variance: stack-free multi-view group variance (-1.2 GB)
2. fused_channel_dot: (A * B).sum(dim=channel) / scale without intermediate (-150~300 MB)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _group_variance_kernel(
    # Camera volumes: [B, C, H, W, Ds] contiguous
    f0_ptr, f1_ptr, f2_ptr, f3_ptr,
    # Output: [B, G, H, W, Ds] contiguous
    out_ptr,
    # Flattened spatial size = H * W * Ds
    spatial_size,
    # Strides for volumes [B, C, H, W, Ds]
    stride_vb, stride_vc,
    # Strides for output [B, G, H, W, Ds]
    stride_ob, stride_og,
    G: tl.constexpr,
    CPG: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    2D grid: (spatial_blocks, B * G).
    Each program processes BLOCK_N spatial elements for one (b, g) pair.
    Computes population variance across 4 cameras, averaged over CPG channels.
    """
    spatial_pid = tl.program_id(0)
    bg = tl.program_id(1)

    offsets = spatial_pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < spatial_size

    b = bg // G
    g = bg % G
    c_start = g * CPG

    vol_batch_base = b * stride_vb

    var_acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    for ci in range(CPG):
        c_offset = (c_start + ci) * stride_vc
        addr = vol_batch_base + c_offset + offsets

        v0 = tl.load(f0_ptr + addr, mask=mask, other=0.0).to(tl.float32)
        v1 = tl.load(f1_ptr + addr, mask=mask, other=0.0).to(tl.float32)
        v2 = tl.load(f2_ptr + addr, mask=mask, other=0.0).to(tl.float32)
        v3 = tl.load(f3_ptr + addr, mask=mask, other=0.0).to(tl.float32)

        mean_c = (v0 + v1 + v2 + v3) * 0.25
        d0 = v0 - mean_c
        d1 = v1 - mean_c
        d2 = v2 - mean_c
        d3 = v3 - mean_c
        var_acc += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3

    # Bessel-corrected variance (/ 3) to match torch.var(dim=0), then mean over CPG (/ CPG)
    result = var_acc * (1.0 / (3.0 * CPG))

    out_base = b * stride_ob + g * stride_og
    tl.store(out_ptr + out_base + offsets, result.to(out_ptr.dtype.element_ty), mask=mask)


def fused_group_variance(f0, f1, f2, f3, num_groups=8):
    """
    Multi-view group variance without torch.stack.

    Equivalent to:
        stacked = torch.stack([f0,f1,f2,f3], dim=0).view(4,B,G,CPG,H,W,Ds)
        group_var = stacked.var(dim=0).mean(dim=2)

    Args:
        f0, f1, f2, f3: [B, C, H, W, Ds] camera volumes
        num_groups: number of channel groups

    Returns:
        group_var: [B, G, H, W, Ds]
    """
    B, C, H, W, Ds = f0.shape
    G = num_groups
    CPG = C // G
    assert C % G == 0

    f0, f1, f2, f3 = [x.contiguous() for x in (f0, f1, f2, f3)]

    spatial_size = H * W * Ds
    group_var = torch.empty(B, G, H, W, Ds, device=f0.device, dtype=f0.dtype)

    BLOCK_N = 1024
    grid = (triton.cdiv(spatial_size, BLOCK_N), B * G)

    _group_variance_kernel[grid](
        f0, f1, f2, f3,
        group_var,
        spatial_size,
        f0.stride(0), f0.stride(1),
        group_var.stride(0), group_var.stride(1),
        G=G,
        CPG=CPG,
        BLOCK_N=BLOCK_N,
    )

    return group_var


# ---------------------------------------------------------------------------
# Fused channel-axis dot product: (A * B).sum(dim=1) / scale
# Avoids materializing the [B, C, H, W, Ds] elementwise product.
# Used for: GWC (groupwise correlation) and similarity profile.
# ---------------------------------------------------------------------------

@triton.jit
def _channel_dot_kernel(
    a_ptr, b_ptr, out_ptr,
    spatial_size,
    stride_ab, stride_ac,   # A strides: batch, channel
    stride_bb, stride_bc,   # B strides: batch, channel
    stride_ob, stride_og,   # out strides: batch, group
    G: tl.constexpr,
    CPG: tl.constexpr,
    scale,
    BLOCK_N: tl.constexpr,
):
    """
    2D grid: (spatial_blocks, B * G).
    Computes grouped dot product: out[b,g,spatial] = sum_c(A[b, g*CPG+c, sp] * B[b, g*CPG+c, sp]) / scale
    When G=1 and CPG=C, this is a full-channel dot product (for similarity).
    """
    spatial_pid = tl.program_id(0)
    bg = tl.program_id(1)

    offsets = spatial_pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < spatial_size

    b = bg // G
    g = bg % G
    c_start = g * CPG

    a_base = b * stride_ab
    b_base = b * stride_bb

    dot_acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    for ci in range(CPG):
        c_idx = c_start + ci
        a_addr = a_base + c_idx * stride_ac + offsets
        b_addr = b_base + c_idx * stride_bc + offsets

        a_val = tl.load(a_ptr + a_addr, mask=mask, other=0.0).to(tl.float32)
        b_val = tl.load(b_ptr + b_addr, mask=mask, other=0.0).to(tl.float32)
        dot_acc += a_val * b_val

    result = dot_acc / scale

    out_base = b * stride_ob + g * stride_og
    tl.store(out_ptr + out_base + offsets, result.to(out_ptr.dtype.element_ty), mask=mask)


def fused_gwc(ref, tgt, num_groups=8):
    """
    Group-wise correlation without [B, G, CPG, H, W, Ds] intermediate.

    Equivalent to:
        (ref.view(B,G,CPG,H,W,Ds) * tgt.view(B,G,CPG,H,W,Ds)).sum(dim=2) / sqrt(CPG)

    Args:
        ref, tgt: [B, C, H, W, Ds]
        num_groups: number of channel groups
    Returns:
        gwc: [B, G, H, W, Ds]
    """
    B, C, H, W, Ds = ref.shape
    G = num_groups
    CPG = C // G
    assert C % G == 0

    ref = ref.contiguous()
    tgt = tgt.contiguous()

    scale = CPG ** 0.5
    spatial_size = H * W * Ds

    gwc = torch.empty(B, G, H, W, Ds, device=ref.device, dtype=ref.dtype)

    BLOCK_N = 1024
    grid = (triton.cdiv(spatial_size, BLOCK_N), B * G)

    _channel_dot_kernel[grid](
        ref, tgt, gwc,
        spatial_size,
        ref.stride(0), ref.stride(1),
        tgt.stride(0), tgt.stride(1),
        gwc.stride(0), gwc.stride(1),
        G=G, CPG=CPG,
        scale=scale,
        BLOCK_N=BLOCK_N,
    )
    return gwc


def fused_similarity(ref, tgt):
    """
    Channel-wise dot product without [B, C, H, W, Ds] intermediate.

    Equivalent to:
        (ref * tgt).sum(dim=1) / sqrt(C)

    Args:
        ref, tgt: [B, C, H, W, Ds]
    Returns:
        sim: [B, H, W, Ds]
    """
    B, C, H, W, Ds = ref.shape

    ref = ref.contiguous()
    tgt = tgt.contiguous()

    scale = C ** 0.5
    spatial_size = H * W * Ds

    # Output: [B, 1, H, W, Ds] then squeeze
    sim = torch.empty(B, 1, H, W, Ds, device=ref.device, dtype=ref.dtype)

    BLOCK_N = 1024
    grid = (triton.cdiv(spatial_size, BLOCK_N), B * 1)

    _channel_dot_kernel[grid](
        ref, tgt, sim,
        spatial_size,
        ref.stride(0), ref.stride(1),
        tgt.stride(0), tgt.stride(1),
        sim.stride(0), sim.stride(1),
        G=1, CPG=C,
        scale=scale,
        BLOCK_N=BLOCK_N,
    )
    return sim.squeeze(1)  # [B, H, W, Ds]
