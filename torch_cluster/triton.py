from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
                      num_warps=4,
                      num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32},
                      num_warps=4,
                      num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32},
                      num_warps=4,
                      num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32},
                      num_warps=8,
                      num_stages=4),
    ],
    key=['M', 'N', 'D'],
)
@triton.heuristics({
    'EVEN_M': lambda args: args['M'] % args['BLOCK_M'] == 0,
    'EVEN_N': lambda args: args['N'] % args['BLOCK_N'] == 0,
})
@triton.jit
def _pairwise_distance_kernel(
    x_ptr,
    y_ptr,
    x_norm_ptr,
    y_norm_ptr,
    out_ptr,
    M,
    N,
    D,
    stride_xm,
    stride_xd,
    stride_ym,
    stride_yd,
    stride_outm,
    stride_outn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, D, BLOCK_K):
        k_offsets = k + offs_k
        mask_k = k_offsets < D

        y = tl.load(
            y_ptr + offs_m[:, None] * stride_ym + k_offsets[None, :] *
            stride_yd,
            mask=(offs_m[:, None] < M) & mask_k[None, :],
            other=0.0,
        )
        x = tl.load(
            x_ptr + k_offsets[:, None] * stride_xd + offs_n[None, :] *
            stride_xm,
            mask=mask_k[:, None] & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(y, x)

    y_norm = tl.load(y_norm_ptr + offs_m, mask=offs_m < M, other=0.0)
    x_norm = tl.load(x_norm_ptr + offs_n, mask=offs_n < N, other=0.0)

    dist = y_norm[:, None] + x_norm[None, :] - 2.0 * acc
    out_ptrs = (out_ptr + offs_m[:, None] * stride_outm +
                offs_n[None, :] * stride_outn)
    if EVEN_M and EVEN_N:
        tl.store(out_ptrs, dist)
    else:
        tl.store(out_ptrs, dist, mask=(offs_m[:, None] < M) &
                 (offs_n[None, :] < N))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
                      num_warps=4,
                      num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32},
                      num_warps=4,
                      num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32},
                      num_warps=4,
                      num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32},
                      num_warps=8,
                      num_stages=4),
    ],
    key=['M', 'N', 'D'],
)
@triton.heuristics({
    'EVEN_M': lambda args: args['M'] % args['BLOCK_M'] == 0,
    'EVEN_N': lambda args: args['N'] % args['BLOCK_N'] == 0,
})
@triton.jit
def _pairwise_dot_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    M,
    N,
    D,
    stride_xm,
    stride_xd,
    stride_ym,
    stride_yd,
    stride_outm,
    stride_outn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, D, BLOCK_K):
        k_offsets = k + offs_k
        mask_k = k_offsets < D

        y = tl.load(
            y_ptr + offs_m[:, None] * stride_ym + k_offsets[None, :] *
            stride_yd,
            mask=(offs_m[:, None] < M) & mask_k[None, :],
            other=0.0,
        )
        x = tl.load(
            x_ptr + k_offsets[:, None] * stride_xd + offs_n[None, :] *
            stride_xm,
            mask=mask_k[:, None] & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(y, x)

    out_ptrs = (out_ptr + offs_m[:, None] * stride_outm +
                offs_n[None, :] * stride_outn)
    if EVEN_M and EVEN_N:
        tl.store(out_ptrs, acc)
    else:
        tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) &
                 (offs_n[None, :] < N))


def _triton_pairwise_distances(
    x: Tensor,
    y: Tensor,
    cosine: bool = False,
) -> Tensor:
    x = x.contiguous()
    y = y.contiguous()
    M, N = y.size(0), x.size(0)

    if cosine:
        x = torch.nn.functional.normalize(x, dim=-1)
        y = torch.nn.functional.normalize(y, dim=-1)
        out = torch.empty((M, N), device=x.device, dtype=torch.float32)
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),
                             triton.cdiv(N, meta['BLOCK_N']))
        _pairwise_dot_kernel[grid](
            x,
            y,
            out,
            M,
            N,
            x.size(1),
            x.stride(0),
            x.stride(1),
            y.stride(0),
            y.stride(1),
            out.stride(0),
            out.stride(1),
        )
        return 1.0 - out

    x_norm = (x * x).sum(dim=1).float()
    y_norm = (y * y).sum(dim=1).float()
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),
                         triton.cdiv(N, meta['BLOCK_N']))
    _pairwise_distance_kernel[grid](
        x,
        y,
        x_norm,
        y_norm,
        out,
        M,
        N,
        x.size(1),
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        out.stride(0),
        out.stride(1),
    )
    return out


def _pairwise_distances(
    x: Tensor,
    y: Tensor,
    cosine: bool = False,
) -> Tensor:
    if x.is_cuda:
        return _triton_pairwise_distances(x, y, cosine=cosine)
    if cosine:
        x = torch.nn.functional.normalize(x, dim=-1)
        y = torch.nn.functional.normalize(y, dim=-1)
        return 1.0 - (y @ x.t())
    return torch.cdist(y, x, p=2).pow(2)


def knn__triton(
    x: Tensor,
    y: Tensor,
    k: int,
    batch_x: Optional[Tensor] = None,
    batch_y: Optional[Tensor] = None,
    cosine: bool = False,
    num_workers: int = 1,
    batch_size: Optional[int] = None,
) -> Tensor:
    if x.numel() == 0 or y.numel() == 0:
        return torch.empty(2, 0, dtype=torch.long, device=x.device)

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y
    x, y = x.contiguous(), y.contiguous()

    distances = _pairwise_distances(x, y, cosine=cosine)
    if batch_x is not None or batch_y is not None:
        if batch_x is None:
            batch_x = x.new_zeros(x.size(0), dtype=torch.long)
        if batch_y is None:
            batch_y = y.new_zeros(y.size(0), dtype=torch.long)
        batch_size = int(max(batch_x.max(), batch_y.max())) + 1
        arange = torch.arange(batch_size + 1, device=x.device)
        ptr_x = torch.bucketize(arange, batch_x)
        left = ptr_x[batch_y]
        right = ptr_x[batch_y + 1]
        idx_x = torch.arange(x.size(0), device=x.device)
        batch_mask = (idx_x >= left[:, None]) & (idx_x < right[:, None])
        distances = distances.masked_fill(~batch_mask, float('inf'))

    _, indices = torch.topk(distances, k=k, largest=False)
    row = torch.arange(y.size(0), device=y.device).repeat_interleave(k)
    col = indices.reshape(-1)
    return torch.stack([row, col], dim=0)


def knn_graph__triton(
    x: Tensor,
    k: int,
    batch: Optional[Tensor] = None,
    loop: bool = False,
    flow: str = 'source_to_target',
    cosine: bool = False,
    num_workers: int = 1,
    batch_size: Optional[int] = None,
) -> Tensor:
    if flow not in ['source_to_target', 'target_to_source']:
        raise ValueError("'flow' must be 'source_to_target' or 'target_to_source'")
    edge_index = knn__triton(x, x, k if loop else k + 1, batch, batch, cosine,
                             num_workers, batch_size)
    if flow == 'source_to_target':
        row, col = edge_index[1], edge_index[0]
    else:
        row, col = edge_index[0], edge_index[1]

    if not loop:
        mask = row != col
        row, col = row[mask], col[mask]

    return torch.stack([row, col], dim=0)


def radius__triton(
    x: Tensor,
    y: Tensor,
    r: float,
    batch_x: Optional[Tensor] = None,
    batch_y: Optional[Tensor] = None,
    max_num_neighbors: int = 32,
    num_workers: int = 1,
    batch_size: Optional[int] = None,
    ignore_same_index: bool = False,
) -> Tensor:
    if x.numel() == 0 or y.numel() == 0:
        return torch.empty(2, 0, dtype=torch.long, device=x.device)

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y
    x, y = x.contiguous(), y.contiguous()

    r2 = float(r) * float(r)
    distances = _pairwise_distances(x, y, cosine=False)
    if batch_x is not None or batch_y is not None:
        if batch_x is None:
            batch_x = x.new_zeros(x.size(0), dtype=torch.long)
        if batch_y is None:
            batch_y = y.new_zeros(y.size(0), dtype=torch.long)
        batch_size = int(max(batch_x.max(), batch_y.max())) + 1
        arange = torch.arange(batch_size + 1, device=x.device)
        ptr_x = torch.bucketize(arange, batch_x)
        left = ptr_x[batch_y]
        right = ptr_x[batch_y + 1]
        idx_x = torch.arange(x.size(0), device=x.device)
        batch_mask = (idx_x >= left[:, None]) & (idx_x < right[:, None])
        distances = distances.masked_fill(~batch_mask, r2 + 1.0)

    if ignore_same_index and x.size(0) == y.size(0) and torch.allclose(x, y):
        distances = distances.clone()
        distances.fill_diagonal_(r2 + 1.0)

    k = min(max_num_neighbors, x.size(0))
    values, indices = torch.topk(distances, k=k, largest=False)
    valid = values <= r2
    if not valid.any():
        return torch.empty(2, 0, dtype=torch.long, device=x.device)

    row = torch.arange(y.size(0), device=y.device).view(-1, 1).expand_as(values)
    row = row[valid]
    col = indices[valid]
    return torch.stack([row, col], dim=0)


def radius_graph__triton(
    x: Tensor,
    r: float,
    batch: Optional[Tensor] = None,
    loop: bool = False,
    max_num_neighbors: int = 32,
    flow: str = 'source_to_target',
    num_workers: int = 1,
    batch_size: Optional[int] = None,
) -> Tensor:
    if flow not in ['source_to_target', 'target_to_source']:
        raise ValueError("'flow' must be 'source_to_target' or 'target_to_source'")
    edge_index = radius__triton(x, x, r, batch, batch, max_num_neighbors,
                                num_workers, batch_size, not loop)
    if flow == 'source_to_target':
        row, col = edge_index[1], edge_index[0]
    else:
        row, col = edge_index[0], edge_index[1]
    return torch.stack([row, col], dim=0)


def nearest__triton(
    x: Tensor,
    y: Tensor,
    batch_x: Optional[Tensor] = None,
    batch_y: Optional[Tensor] = None,
) -> Tensor:
    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y
    if batch_x is not None and (batch_x[1:] - batch_x[:-1] < 0).any():
        raise ValueError("'batch_x' is not sorted")
    if batch_y is not None and (batch_y[1:] - batch_y[:-1] < 0).any():
        raise ValueError("'batch_y' is not sorted")

    if batch_x is None and batch_y is None:
        distances = _pairwise_distances(y, x, cosine=False)
        return distances.argmin(dim=0).to(torch.long)

    if batch_x is None:
        batch_x = x.new_zeros(x.size(0), dtype=torch.long)
    if batch_y is None:
        batch_y = y.new_zeros(y.size(0), dtype=torch.long)

    batch_size = int(max(batch_x.max(), batch_y.max())) + 1
    arange = torch.arange(batch_size + 1, device=x.device)
    ptr_x = torch.bucketize(arange, batch_x)
    ptr_y = torch.bucketize(arange, batch_y)
    nonempty_x = (ptr_x[1:] - ptr_x[:-1]) > 0
    nonempty_y = (ptr_y[1:] - ptr_y[:-1]) > 0
    if not torch.equal(nonempty_x, nonempty_y):
        raise ValueError("Some batch indices occur in 'batch_x' that do "
                         "not occur in 'batch_y'")

    distances = _pairwise_distances(y, x, cosine=False)
    left = ptr_y[batch_x]
    right = ptr_y[batch_x + 1]
    idx_y = torch.arange(y.size(0), device=y.device)
    batch_mask = (idx_y >= left[None, :]) & (idx_y < right[None, :])
    distances = distances.masked_fill(~batch_mask, float('inf'))
    return distances.argmin(dim=0).to(torch.long)


def grid_cluster__triton(
    pos: Tensor,
    size: Tensor,
    start: Optional[Tensor] = None,
    end: Optional[Tensor] = None,
) -> Tensor:
    pos = pos.view(pos.size(0), -1)
    if start is None:
        start = pos.min(dim=0).values
    if end is None:
        end = pos.max(dim=0).values

    pos = pos - start.unsqueeze(0)
    num_voxels = (end - start).div(size).to(torch.long) + 1
    num_voxels = torch.cumprod(num_voxels, dim=0)
    num_voxels = torch.cat(
        [torch.ones(1, device=pos.device, dtype=num_voxels.dtype), num_voxels])
    num_voxels = num_voxels[:-1]

    coords = pos.div(size.view(1, -1)).to(torch.long)
    return (coords * num_voxels.view(1, -1)).sum(dim=1)


def fps__triton(
    src: Tensor,
    batch: Optional[Tensor] = None,
    ratio: Optional[Union[Tensor, float]] = None,
    random_start: bool = True,
    batch_size: Optional[int] = None,
    ptr: Optional[Union[Tensor, List[int]]] = None,
) -> Tensor:
    src = src.view(src.size(0), -1).contiguous()
    if ratio is None:
        ratio_tensor = torch.tensor(0.5, dtype=src.dtype, device=src.device)
    elif isinstance(ratio, float):
        ratio_tensor = torch.tensor(ratio, dtype=src.dtype, device=src.device)
    else:
        ratio_tensor = ratio

    if ptr is not None:
        if isinstance(ptr, list):
            ptr_tensor = torch.tensor(ptr, device=src.device)
        else:
            ptr_tensor = ptr
    else:
        if batch is None:
            ptr_tensor = torch.tensor([0, src.size(0)], device=src.device)
        else:
            if batch_size is None:
                batch_size = int(batch.max()) + 1
            deg = src.new_zeros(batch_size, dtype=torch.long)
            deg.scatter_add_(0, batch, torch.ones_like(batch))
            ptr_tensor = deg.new_zeros(batch_size + 1)
            torch.cumsum(deg, 0, out=ptr_tensor[1:])

    ptr_tensor = ptr_tensor.to(torch.long)
    deg = ptr_tensor[1:] - ptr_tensor[:-1]
    out_ptr = (deg.to(src.dtype) * ratio_tensor).ceil().to(torch.long)
    out_ptr = torch.cumsum(out_ptr, 0)
    out = torch.empty(out_ptr[-1].item(), dtype=torch.long, device=src.device)

    for b in range(ptr_tensor.numel() - 1):
        start = ptr_tensor[b].item()
        end = ptr_tensor[b + 1].item()
        out_start = 0 if b == 0 else out_ptr[b - 1].item()
        out_end = out_ptr[b].item()
        if end <= start or out_end <= out_start:
            continue
        points = src[start:end]
        if random_start:
            start_idx = torch.randint(0, points.size(0), (1, ),
                                      device=src.device).item()
        else:
            start_idx = 0
        out[out_start] = start + start_idx
        dist = _pairwise_distances(points, points[start_idx:start_idx + 1],
                                   cosine=False).squeeze(0)
        for i in range(1, out_end - out_start):
            argmax = int(dist.argmax().item())
            out[out_start + i] = start + argmax
            dist = torch.minimum(
                dist,
                _pairwise_distances(points, points[argmax:argmax + 1],
                                    cosine=False).squeeze(0),
            )

    return out


def graclus_cluster__triton(
    row: Tensor,
    col: Tensor,
    weight: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> Tensor:
    if num_nodes is None:
        num_nodes = max(int(row.max()), int(col.max())) + 1

    mask = row != col
    row, col = row[mask], col[mask]
    if weight is not None:
        weight = weight[mask]

    perm = torch.argsort(row)
    row, col = row[perm], col[perm]
    if weight is not None:
        weight = weight[perm]

    deg = row.new_zeros(num_nodes)
    deg.scatter_add_(0, row, torch.ones_like(row))
    rowptr = row.new_zeros(num_nodes + 1)
    torch.cumsum(deg, 0, out=rowptr[1:])

    out = torch.full((num_nodes, ), -1, dtype=row.dtype, device=row.device)
    node_perm = torch.randperm(num_nodes, device=row.device)

    for u in node_perm.tolist():
        if out[u] >= 0:
            continue
        if weight is None:
            out[u] = u
            row_start = rowptr[u].item()
            row_end = rowptr[u + 1].item()
            for e in range(row_start, row_end):
                v = col[e].item()
                if out[v] >= 0:
                    continue
                cluster = min(u, v)
                out[u] = cluster
                out[v] = cluster
                break
        else:
            row_start = rowptr[u].item()
            row_end = rowptr[u + 1].item()
            if row_end <= row_start:
                out[u] = u
                continue
            weights = weight[row_start:row_end]
            candidates = col[row_start:row_end]
            if weights.numel() == 0:
                out[u] = u
                continue
            valid = out[candidates] < 0
            if not valid.any():
                out[u] = u
                continue
            weights = weights.clone()
            weights[~valid] = -float('inf')
            v = candidates[int(weights.argmax().item())].item()
            cluster = min(u, v)
            out[u] = cluster
            out[v] = cluster
    return out


def random_walk__triton(
    row: Tensor,
    col: Tensor,
    start: Tensor,
    walk_length: int,
    p: float = 1,
    q: float = 1,
    coalesced: bool = True,
    num_nodes: Optional[int] = None,
    return_edge_indices: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    if num_nodes is None:
        num_nodes = max(int(row.max()), int(col.max()), int(start.max())) + 1

    if coalesced:
        perm = torch.argsort(row * num_nodes + col)
        row, col = row[perm], col[perm]

    deg = row.new_zeros(num_nodes)
    deg.scatter_add_(0, row, torch.ones_like(row))
    rowptr = row.new_zeros(num_nodes + 1)
    torch.cumsum(deg, 0, out=rowptr[1:])

    if p != 1 or q != 1:
        raise NotImplementedError('Non-uniform node2vec sampling is not '
                                  'implemented for Triton yet.')

    numel = start.numel()
    node_seq = torch.empty((numel, walk_length + 1),
                           dtype=start.dtype,
                           device=start.device)
    edge_seq = torch.empty((numel, walk_length),
                           dtype=start.dtype,
                           device=start.device)
    rand = torch.rand((numel, walk_length), device=start.device)
    for n in range(numel):
        n_cur = int(start[n].item())
        node_seq[n, 0] = n_cur
        for l in range(walk_length):
            row_start = rowptr[n_cur].item()
            row_end = rowptr[n_cur + 1].item()
            if row_end - row_start == 0:
                edge_seq[n, l] = -1
                node_seq[n, l + 1] = n_cur
                continue
            idx = int(rand[n, l].item() * (row_end - row_start))
            e_cur = row_start + idx
            edge_seq[n, l] = e_cur
            n_cur = int(col[e_cur].item())
            node_seq[n, l + 1] = n_cur

    if return_edge_indices:
        return node_seq, edge_seq
    return node_seq


def neighbor_sampler__triton(
    start: Tensor,
    rowptr: Tensor,
    size: float,
) -> Tensor:
    if start.is_cuda:
        raise ValueError('neighbor_sampler__triton only supports CPU tensors.')

    factor = -1.0
    count = -1
    if size <= 1:
        factor = float(size)
    else:
        count = int(size)

    return torch.ops.torch_cluster.neighbor_sampler(start, rowptr, count,
                                                    factor)


__all__ = [
    'fps__triton',
    'graclus_cluster__triton',
    'grid_cluster__triton',
    'knn__triton',
    'knn_graph__triton',
    'nearest__triton',
    'radius__triton',
    'radius_graph__triton',
    'random_walk__triton',
    'neighbor_sampler__triton',
]
