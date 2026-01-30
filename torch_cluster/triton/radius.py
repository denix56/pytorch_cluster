from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ._kernels import _radius_segmented_kernel


def radius(
    x: Tensor,
    y: Tensor,
    r: float,
    batch_x: Optional[Tensor] = None,
    batch_y: Optional[Tensor] = None,
    max_num_neighbors: int = 32,
    batch_size: Optional[int] = None,
    ignore_same_index: bool = False,
    match_cuda: bool = True,
) -> Tensor:
    r"""Find all neighbors within a radius using Triton kernels.

    Args:
        x (Tensor): Input features of shape [N, D].
        y (Tensor): Query features of shape [M, D].
        r (float): Search radius.
        batch_x (Tensor, optional): Batch vector for x.
        batch_y (Tensor, optional): Batch vector for y.
        max_num_neighbors (int, optional): Maximum neighbors per y.
        batch_size (int, optional): Number of batches.
        ignore_same_index (bool, optional): Exclude exact self matches.

    Returns:
        Tensor: Edge index with shape [2, num_edges].
    """
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

    row = torch.full(
        (M * max_num_neighbors,),
        -1,
        device=y.device,
        dtype=torch.int64,
    )
    col = torch.full(
        (M * max_num_neighbors,),
        -1,
        device=y.device,
        dtype=torch.int64,
    )

    if max_candidates == 0 or max_num_neighbors <= 0:
        mask = row != -1
        return torch.stack([row[mask], col[mask]], dim=0)

    grid = (M,)
    _radius_segmented_kernel[grid](
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
        float(r) * float(r),
        max_num_neighbors,
        IGNORE_SAME_INDEX=ignore_same_index,
        MATCH_CUDA=match_cuda,
    )

    mask = row != -1
    return torch.stack([row[mask], col[mask]], dim=0)
