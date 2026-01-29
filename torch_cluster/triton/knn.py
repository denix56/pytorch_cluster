from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor
import triton

from ._kernels import _knn_stage1_kernel, _knn_stage2_merge_kernel, _pairwise_distances


def knn(
    x: Tensor,
    y: Tensor,
    k: int,
    batch_x: Optional[Tensor] = None,
    batch_y: Optional[Tensor] = None,
    cosine: bool = False,
    batch_size: Optional[int] = None,
) -> Tensor:
    r"""Compute k-NN indices using Triton kernels.

    Args:
        x (Tensor): Input features of shape [N, D].
        y (Tensor): Query features of shape [M, D].
        k (int): Number of neighbors.
        batch_x (Tensor, optional): Batch vector for x.
        batch_y (Tensor, optional): Batch vector for y.
        cosine (bool, optional): Use cosine distance if True.
        num_workers (int, optional): Unused for Triton path.
        batch_size (int, optional): Number of batches.
        ignore_same_index (bool, optional): Exclude exact self matches.

    Returns:
        Tensor: Edge index with shape [2, M * k].
    """
    indices = knn_fused_splitN(x, y, k, batch_x, batch_y, cosine, batch_size)
    row = torch.arange(y.size(0), device=y.device).repeat_interleave(k)
    col = indices.reshape(-1)
    return torch.stack([row, col], dim=0)


def knn_fused_splitN(
    x: Tensor,
    y: Tensor,
    k: int,
    batch_x: Optional[Tensor] = None,
    batch_y: Optional[Tensor] = None,
    cosine: bool = False,
    batch_size: Optional[int] = None,
    split_n: int = 4,
) -> Tensor:
    r"""Compute k-NN indices using a split-N fused Triton implementation."""
    if k > 32:
        distances = _pairwise_distances(x, y, batch_size=batch_size, batch_x=batch_x,
                                        batch_y=batch_y, cosine=cosine)
        indices = torch.argsort(distances, stable=True)[..., :k]
        return indices

    use_batch = (batch_size or 1) > 1
    if use_batch:
        assert batch_x is not None
        assert batch_y is not None
        arange = torch.arange((batch_size or 1) + 1, device=x.device)
        ptr_x = torch.bucketize(arange, batch_x)
        batch_y = batch_y.contiguous()
    else:
        ptr_x = x
        batch_y = y

    M, N = y.size(0), x.size(0)
    part_i = torch.empty((split_n, M, k), device=x.device, dtype=torch.int32)
    part_d = torch.empty((split_n, M, k), device=x.device, dtype=torch.float32)
    out_i = torch.empty((M, k), device=x.device, dtype=torch.int32)
    eps = torch.finfo(part_d.dtype).eps

    def stage1_grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']), split_n)

    # Stage 1: split N into partitions and keep per-row partial top-k in registers.
    _knn_stage1_kernel[stage1_grid](
        x,
        y,
        part_i,
        part_d,
        M,
        N,
        x.size(1),
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        part_i.stride(0),
        part_i.stride(1),
        part_i.stride(2),
        ptr_x,
        batch_y,
        USE_BATCH=use_batch,
        COSINE=cosine,
        EPS=eps,
        K=k,
        SPLIT_N=split_n,
    )

    # Stage 2: merge partial candidates from all splits into the final top-k.
    def stage2_grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']),)

    _knn_stage2_merge_kernel[stage2_grid](
        part_i,
        part_d,
        out_i,
        M,
        part_i.stride(0),
        part_i.stride(1),
        part_i.stride(2),
        out_i.stride(0),
        out_i.stride(1),
        K=k,
        SPLIT_N=split_n,
    )
    return out_i
