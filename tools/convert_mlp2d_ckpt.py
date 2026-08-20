#!/usr/bin/env python
"""Convert legacy MLP2D checkpoint keys to this tree's layout.

Older runs stored MLP2D as `nn.Sequential(Conv3d, ReLU, Conv3d, Sigmoid)`, so the
two 1x1 convs were named `net.0` / `net.2` with weights `[out, in, 1, 1, 1]`.
This tree names them `linear1` / `linear2` (Conv2d, `[out, in, 1, 1]`). Both are
1x1 kernels, so dropping the trailing singleton axis is an exact conversion —
the weights are unchanged.

Usage:
    python tools/convert_mlp2d_ckpt.py checkpoints/**/*.pth          # -> *_patched.pth
    python tools/convert_mlp2d_ckpt.py foo.pth --out bar.pth
    python tools/convert_mlp2d_ckpt.py foo.pth --inplace
"""
import argparse
import os
import sys

import torch

REMAP = {'net.0': 'linear1', 'net.2': 'linear2'}


def migrate(state_dict):
    out, n = {}, 0
    for k, v in state_dict.items():
        for old, new in REMAP.items():
            if f'mapping.{old}.' in k:
                k = k.replace(f'mapping.{old}.', f'mapping.{new}.')
                if v.dim() == 5:
                    v = v.squeeze(-1)
                n += 1
                break
        out[k] = v
    return out, n


def convert(path, out_path=None, inplace=False):
    snapshot = torch.load(path, map_location='cpu')
    if 'net_state_dict' not in snapshot:
        print(f"  skip (no net_state_dict): {path}")
        return False

    migrated, n = migrate(snapshot['net_state_dict'])
    if n == 0:
        print(f"  skip (already converted): {path}")
        return False

    snapshot['net_state_dict'] = migrated
    dst = path if inplace else (out_path or path.replace('.pth', '_patched.pth'))
    torch.save(snapshot, dst)
    print(f"  {os.path.basename(path)}: {n} keys -> {dst}")
    return True


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpts', nargs='+')
    ap.add_argument('--out', default=None, help='output path (single ckpt only)')
    ap.add_argument('--inplace', action='store_true', help='overwrite the input file')
    a = ap.parse_args()

    if a.out and len(a.ckpts) > 1:
        sys.exit('--out only works with a single checkpoint')

    done = sum(convert(p, a.out, a.inplace) for p in a.ckpts)
    print(f"converted {done}/{len(a.ckpts)}")
