from __future__ import annotations

from typing import Optional

import triton
import torch
from torch import Tensor

from ._kernels import _knn_segmented_kernel


def segmented_topk_search(
    x: Tensor,
    y: Tensor,
    k: int,
    batch_x: Optional[Tensor] = None,
    batch_y: Optional[Tensor] = None,
    cosine: bool = False,
    batch_size: Optional[int] = None,
    radius: Optional[float] = None,
    ignore_same_index: bool = False,
    match_cuda: bool = True,
) -> Tensor:
    r"""Compute top-k neighbor indices with an optional radius constraint."""
    use_batch = (batch_size or 1) > 1
    if use_batch:
        assert batch_x is not None
        assert batch_y is not None
        arange = torch.arange((batch_size or 1) + 1, device=x.device)
        ptr_x = torch.bucketize(arange, batch_x)
        ptr_y = torch.bucketize(arange, batch_y)
    else:
        ptr_x = torch.tensor(
            [0, x.size(0)],
            device=x.device,
            dtype=torch.int64,
        )
        ptr_y = torch.tensor(
            [0, y.size(0)],
            device=y.device,
            dtype=torch.int64,
        )

    if ptr_x.numel() != ptr_y.numel():
        raise ValueError(
            "ptr_x and ptr_y must have the same number of segments."
        )

    M, N = y.size(0), x.size(0)
    D = x.size(1)
    if use_batch:
        example_idx = batch_y.to(torch.int64).contiguous()
    else:
        y_idx = torch.arange(M, device=y.device, dtype=torch.int64)
        if ptr_y.numel() > 2:
            ptr_y_mid = ptr_y[1:-1].contiguous()
        else:
            ptr_y_mid = ptr_y.new_empty((0,))
        example_idx = torch.bucketize(y_idx, ptr_y_mid, right=False)

    seg_sizes = torch.diff(ptr_x).to(torch.int64)
    if seg_sizes.numel() > 0:
        max_candidates = int(seg_sizes.max().item())
    else:
        max_candidates = 0
    k_pad = triton.next_power_of_2(2 * k)
    eps = torch.finfo(torch.float32).eps

    row = torch.empty(M * k, device=y.device, dtype=torch.int64)
    col = torch.full((M * k,), -1, device=y.device, dtype=torch.int64)

    if max_candidates == 0:
        return col.view(M, k)

    use_radius = radius is not None
    r2 = float(radius) * float(radius) if use_radius else 0.0

    grid = (M,)
    _knn_segmented_kernel[grid](
        x,
        y,
        ptr_x,
        example_idx,
        row,
        col,
        M,
        N,
        D,
        max_candidates,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        r2,
        K=k,
        K_PAD=k_pad,
        COSINE=cosine,
        EPS=eps,
        USE_RADIUS=use_radius,
        IGNORE_SAME_INDEX=ignore_same_index,
        MATCH_CUDA=match_cuda,
    )
    return col.view(M, k)
