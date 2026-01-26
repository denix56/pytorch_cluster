from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor

from ._kernels import _pairwise_distances


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
    # Compute distances, optionally with batch masking.
    distances = _pairwise_distances(x, y, batch_size=batch_size, batch_x=batch_x,
                                    batch_y=batch_y, cosine=cosine)

    # Break distance ties by preferring smaller indices for deterministic
    # results that match the reference operator.
    indices = torch.argsort(distances, stable=True)[..., :k]
    #eps = torch.finfo(distances.dtype).eps
    #tie_break = torch.arange(x.size(0), device=distances.device,
    #                         dtype=distances.dtype)
    #distances = distances + tie_break[None, :] * eps
    #_, indices = torch.topk(distances, k=k, largest=False)
    row = torch.arange(y.size(0), device=y.device).repeat_interleave(k)
    col = indices.reshape(-1)
    return torch.stack([row, col], dim=0)
