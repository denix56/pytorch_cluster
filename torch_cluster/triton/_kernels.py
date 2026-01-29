from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
                      num_warps=4,
                      num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32},
                      num_warps=4,
                      num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32},
                      num_warps=4,
                      num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},
                      num_warps=8,
                      num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 16},
                      num_warps=8,
                      num_stages=2),
    ],
    key=['M', 'N', 'D'],
)
@triton.heuristics({
    'EVEN_N': lambda args: args['N'] % args['BLOCK_N'] == 0,
})
@triton.jit
def _nearest_kernel(
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
    ptr_y_ptr,
    batch_x_ptr,
    USE_BATCH: tl.constexpr,
    COSINE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    r"""Compute nearest neighbor indices for blocks of x.

    Args:
        x_ptr: Pointer to x matrix.
        y_ptr: Pointer to y matrix.
        out_ptr: Pointer to output indices.
        M (int): Number of y rows.
        N (int): Number of x rows.
        D (int): Feature dimension.
        stride_xm/stride_xd: Strides for x.
        stride_ym/stride_yd: Strides for y.
        ptr_y_ptr: Pointer to y batch prefix sums.
        batch_x_ptr: Pointer to x batch ids.
        USE_BATCH (constexpr bool): Whether to apply batch masking.
        COSINE (constexpr bool): Use cosine distance if True.
        EPS (constexpr float): Numerical epsilon.
        BLOCK_M/BLOCK_N/BLOCK_K (constexpr int): Tile sizes.
        EVEN_N (constexpr bool): Alignment hints.
    """
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_x = offs_n < N
    if EVEN_N:
        tl.multiple_of(offs_n, 8)
        tl.max_contiguous(offs_n, 8)

    best_dist = tl.full((BLOCK_N,), float('inf'), tl.float32)
    best_idx = tl.zeros((BLOCK_N,), dtype=tl.int32)

    if USE_BATCH:
        batch_id = tl.load(batch_x_ptr + offs_n, mask=mask_x, other=0)
        left = tl.load(ptr_y_ptr + batch_id, mask=mask_x, other=0)
        right = tl.load(ptr_y_ptr + batch_id + 1, mask=mask_x, other=0)
    else:
        left = 0
        right = M

    for y_start in range(0, M, BLOCK_M):
        offs_m = y_start + tl.arange(0, BLOCK_M)
        mask_y = offs_m < M

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        x_sq = tl.zeros((BLOCK_N,), dtype=tl.float32)
        y_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)

        x_block_ptr = tl.make_block_ptr(
            base=x_ptr,
            shape=(D, N),
            strides=(stride_xd, stride_xm),
            offsets=(0, pid * BLOCK_N),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(0, 1),
        )
        y_block_ptr = tl.make_block_ptr(
            base=y_ptr,
            shape=(M, D),
            strides=(stride_ym, stride_yd),
            offsets=(y_start, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )

        for _ in range(0, D, BLOCK_K):
            y = tl.load(y_block_ptr,
                        boundary_check=(0, 1),
                        padding_option="zero")
            x = tl.load(x_block_ptr,
                        boundary_check=(0, 1),
                        padding_option="zero")
            acc += tl.dot(y, x, input_precision="ieee")
            y_sq += tl.sum(y * y, axis=1)
            x_sq += tl.sum(x * x, axis=0)
            y_block_ptr = tl.advance(y_block_ptr, (0, BLOCK_K))
            x_block_ptr = tl.advance(x_block_ptr, (BLOCK_K, 0))

        if COSINE:
            inv_y = tl.rsqrt(y_sq + EPS)
            inv_x = tl.rsqrt(x_sq + EPS)
            dist = 1.0 - acc * (inv_y[:, None] * inv_x[None, :])
        else:
            dist = tl.fma(-2.0, acc, y_sq[:, None] + x_sq[None, :])

        valid = mask_y[:, None] & mask_x[None, :]
        if USE_BATCH:
            valid &= (offs_m[:, None] >= left[None, :]) & (offs_m[:, None] < right[None, :])
        dist = tl.where(valid, dist, float('inf'))

        block_min, block_arg = tl.min(dist,
                                      axis=0,
                                      return_indices=True,
                                      return_indices_tie_break_left=True)
        block_idx = y_start + block_arg
        better = block_min < best_dist
        best_dist = tl.where(better, block_min, best_dist)
        best_idx = tl.where(better, block_idx, best_idx)

    tl.store(out_ptr + offs_n, best_idx.to(tl.int64), mask=mask_x)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 32},
                      num_warps=2,
                      num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32},
                      num_warps=2,
                      num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 16, 'BLOCK_K': 32},
                      num_warps=2,
                      num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32},
                      num_warps=2,
                      num_stages=2),
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
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 16},
                      num_warps=8,
                      num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32},
                      num_warps=8,
                      num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},
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
def _pairwise_knn_kernel(
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
    ptr_x_ptr,
    batch_y_ptr,
    USE_BATCH: tl.constexpr,
    COSINE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    r"""Compute a block of pairwise distances for KNN.

    Args:
        x_ptr: Pointer to x matrix.
        y_ptr: Pointer to y matrix.
        out_ptr: Pointer to output distance matrix.
        M (int): Number of y rows.
        N (int): Number of x rows.
        D (int): Feature dimension.
        stride_xm/stride_xd: Strides for x.
        stride_ym/stride_yd: Strides for y.
        stride_outm/stride_outn: Strides for output.
        ptr_x_ptr: Pointer to x batch prefix sums.
        batch_y_ptr: Pointer to y batch ids.
        USE_BATCH (constexpr bool): Whether to apply batch masking.
        COSINE (constexpr bool): Use cosine distance if True.
        EPS (constexpr float): Numerical epsilon.
        BLOCK_M/BLOCK_N/BLOCK_K (constexpr int): Tile sizes.
        EVEN_M/EVEN_N (constexpr bool): Alignment hints.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute tile offsets.
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    if EVEN_M:
        # Hint: rows are aligned/contiguous, enabling better vectorized loads.
        tl.multiple_of(offs_m, 8)
        tl.max_contiguous(offs_m, 8)
    if EVEN_N:
        # Hint: columns are aligned/contiguous, improving memory coalescing.
        tl.multiple_of(offs_n, 8)
        tl.max_contiguous(offs_n, 8)

    tl.static_assert(x_ptr.dtype.element_ty == y_ptr.dtype.element_ty)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    x_sq = tl.zeros((BLOCK_N,), dtype=tl.float32)
    y_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)

    out_ptrs = (out_ptr + offs_m[:, None] * stride_outm +
                offs_n[None, :] * stride_outn)
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    if USE_BATCH:
        row_mask = offs_m < M
        batch_ids = tl.load(batch_y_ptr + offs_m, mask=row_mask, other=0)
        first = tl.load(batch_y_ptr + pid_m * BLOCK_M,
                        mask=pid_m * BLOCK_M < M,
                        other=0)
        same = tl.sum(batch_ids == first, axis=0) == tl.sum(row_mask, axis=0)
        left_scalar = tl.load(ptr_x_ptr + first)
        right_scalar = tl.load(ptr_x_ptr + first + 1)
        n_start = pid_n * BLOCK_N
        n_end = n_start + BLOCK_N - 1
        if same and ((n_end < left_scalar) or (n_start >= right_scalar)):
            tl.store(out_ptrs,
                     tl.full((BLOCK_M, BLOCK_N), float('inf'), tl.float32),
                     mask=mask_out)
            return
        left = tl.load(ptr_x_ptr + batch_ids, mask=row_mask, other=0)
        right = tl.load(ptr_x_ptr + batch_ids + 1, mask=row_mask, other=0)

    y_block_ptr = tl.make_block_ptr(
        base=y_ptr,
        shape=(M, D),
        strides=(stride_ym, stride_yd),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(D, N),
        strides=(stride_xd, stride_xm),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )

    # Accumulate dot products and norms over D.
    for _ in range(0, D, BLOCK_K):
        y = tl.load(y_block_ptr,
                    boundary_check=(0, 1),
                    padding_option="zero")
        x = tl.load(x_block_ptr,
                    boundary_check=(0, 1),
                    padding_option="zero")
        acc += tl.dot(y, x, input_precision="ieee")
        y_sq += tl.sum(y * y, axis=1)
        x_sq += tl.sum(x * x, axis=0)
        y_block_ptr = tl.advance(y_block_ptr, (0, BLOCK_K))
        x_block_ptr = tl.advance(x_block_ptr, (BLOCK_K, 0))

    if COSINE:
        inv_y = tl.rsqrt(y_sq + EPS)
        inv_x = tl.rsqrt(x_sq + EPS)
        dist = 1.0 - acc * (inv_y[:, None] * inv_x[None, :])
    else:
        dist = tl.fma(-2.0, acc, y_sq[:, None] + x_sq[None, :])

    dist += offs_n[None, :].to(tl.float32) * EPS

    if USE_BATCH:
        valid = (offs_n[None, :] >= left[:, None]) & (offs_n[None, :] <
                                                      right[:, None])
        dist = tl.where(valid, dist, float('inf'))

    if EVEN_M and EVEN_N:
        tl.store(out_ptrs, dist)
    else:
        tl.store(out_ptrs, dist, mask=mask_out)


def _pairwise_distances(
    x: Tensor,
    y: Tensor,
    batch_size: int = 1,
    batch_x: Optional[Tensor] = None,
    batch_y: Optional[Tensor] = None,
    cosine: bool = False,
) -> Tensor:
    r"""Compute pairwise distances using the Triton kernel.

    Args:
        x (Tensor): Input features of shape [N, D].
        y (Tensor): Query features of shape [M, D].
        cosine (bool, optional): Use cosine distance if True.

    Returns:
        Tensor: Pairwise distance matrix of shape [M, N].
    """
    use_batch = batch_size > 1
    if use_batch:
        assert batch_x is not None
        assert batch_y is not None
        arange = torch.arange(batch_size + 1, device=x.device)
        ptr_x = torch.bucketize(arange, batch_x)
        batch_y = batch_y.contiguous()
    else:
        ptr_x = x
        batch_y = y

    M, N = y.size(0), x.size(0)
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    eps = torch.finfo(out.dtype).eps
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),
                         triton.cdiv(N, meta['BLOCK_N']))
    _pairwise_knn_kernel[grid](x, y, out, M, N, x.size(1), x.stride(0), x.stride(1),
                               y.stride(0), y.stride(1), out.stride(0), out.stride(1),
                               ptr_x, batch_y, USE_BATCH=use_batch, COSINE=cosine, EPS=eps)
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 32},
                      num_warps=2,
                      num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32},
                      num_warps=2,
                      num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 16, 'BLOCK_K': 32},
                      num_warps=2,
                      num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32},
                      num_warps=2,
                      num_stages=2),
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
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 16},
                      num_warps=8,
                      num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32},
                      num_warps=8,
                      num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},
                      num_warps=8,
                      num_stages=4),
    ],
    key=['M', 'N', 'D'],
)
@triton.heuristics({
    'EVEN_M': lambda args: args['M'] % args['BLOCK_M'] == 0,
    'EVEN_N': lambda args: args['N'] % args['BLOCK_N'] == 0,
    'NUM_TILES': lambda args: triton.cdiv(args['N'], args['BLOCK_N']),
    'TILES_PER_SPLIT': lambda args: triton.cdiv(triton.cdiv(args['N'], args['BLOCK_N']), args['SPLIT_N']),
})
@triton.jit
def _knn_stage1_kernel(
    x_ptr,
    y_ptr,
    part_i_ptr,
    part_d_ptr,
    M,
    N,
    D,
    stride_xm,
    stride_xd,
    stride_ym,
    stride_yd,
    stride_part_s,
    stride_part_m,
    stride_part_k,
    ptr_x_ptr,
    batch_y_ptr,
    USE_BATCH: tl.constexpr,
    COSINE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    K: tl.constexpr,
    SPLIT_N: tl.constexpr,
    NUM_TILES: tl.constexpr,
    TILES_PER_SPLIT: tl.constexpr,
):
    r"""Split-N partial top-k kernel for fused KNN."""
    pid_m = tl.program_id(0)
    pid_s = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    if EVEN_M:
        tl.multiple_of(offs_m, 8)
        tl.max_contiguous(offs_m, 8)

    if USE_BATCH:
        batch_ids = tl.load(batch_y_ptr + offs_m, mask=mask_m, other=0)
        left = tl.load(ptr_x_ptr + batch_ids, mask=mask_m, other=0)
        right = tl.load(ptr_x_ptr + batch_ids + 1, mask=mask_m, other=0)

    best_d = tl.full((BLOCK_M, K), float('inf'), tl.float32)
    best_i = tl.full((BLOCK_M, K), -1, tl.int32)
    k_ids = tl.arange(0, K)

    # Split-N: each program processes a contiguous range of N-tiles.
    tile_start = pid_s * TILES_PER_SPLIT
    for t in tl.static_range(0, TILES_PER_SPLIT):
        tile_idx = tile_start + t
        in_range = tile_idx < NUM_TILES
        offs_n = tile_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = (offs_n < N) & in_range
        if EVEN_N:
            tl.multiple_of(offs_n, 8)
            tl.max_contiguous(offs_n, 8)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        x_sq = tl.zeros((BLOCK_N,), dtype=tl.float32)
        y_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)

        y_block_ptr = tl.make_block_ptr(
            base=y_ptr,
            shape=(M, D),
            strides=(stride_ym, stride_yd),
            offsets=(pid_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )
        x_block_ptr = tl.make_block_ptr(
            base=x_ptr,
            shape=(D, N),
            strides=(stride_xd, stride_xm),
            offsets=(0, tile_idx * BLOCK_N),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(0, 1),
        )

        for _ in range(0, D, BLOCK_K):
            y = tl.load(y_block_ptr,
                        boundary_check=(0, 1),
                        padding_option="zero")
            x = tl.load(x_block_ptr,
                        boundary_check=(0, 1),
                        padding_option="zero")
            acc += tl.dot(y, x, input_precision="ieee")
            y_sq += tl.sum(y * y, axis=1)
            x_sq += tl.sum(x * x, axis=0)
            y_block_ptr = tl.advance(y_block_ptr, (0, BLOCK_K))
            x_block_ptr = tl.advance(x_block_ptr, (BLOCK_K, 0))

        if COSINE:
            inv_y = tl.rsqrt(y_sq + EPS)
            inv_x = tl.rsqrt(x_sq + EPS)
            dist = 1.0 - acc * (inv_y[:, None] * inv_x[None, :])
        else:
            dist = tl.fma(-2.0, acc, y_sq[:, None] + x_sq[None, :])

        dist += offs_n[None, :].to(tl.float32) * EPS

        # Apply bounds and batch masking; invalid candidates become +inf.
        valid = mask_m[:, None] & mask_n[None, :]
        if USE_BATCH:
            valid &= (offs_n[None, :] >= left[:, None]) & (offs_n[None, :] < right[:, None])
        dist = tl.where(valid, dist, float('inf'))

        # Maintain per-row partial top-k by replacing the current worst entry.
        worst_d, worst_idx = tl.max(best_d,
                                    axis=1,
                                    return_indices=True,
                                    return_indices_tie_break_left=True)
        for n in tl.static_range(0, BLOCK_N):
            cand_d = dist[:, n]
            cand_i = offs_n[n]
            replace = cand_d < worst_d
            repl_mask = replace[:, None] & (k_ids[None, :] == worst_idx[:, None])
            best_d = tl.where(repl_mask, cand_d[:, None], best_d)
            best_i = tl.where(repl_mask, cand_i, best_i)
            worst_d, worst_idx = tl.max(best_d,
                                        axis=1,
                                        return_indices=True,
                                        return_indices_tie_break_left=True)

    offs_k = tl.arange(0, K)
    part_ptrs = (pid_s * stride_part_s +
                 offs_m[:, None] * stride_part_m +
                 offs_k[None, :] * stride_part_k)
    mask_out = mask_m[:, None]
    tl.store(part_d_ptr + part_ptrs, best_d, mask=mask_out)
    tl.store(part_i_ptr + part_ptrs, best_i, mask=mask_out)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=4),
    ],
    key=['M'],
)
@triton.jit
def _knn_stage2_merge_kernel(
    part_i_ptr,
    part_d_ptr,
    out_i_ptr,
    M,
    stride_part_s,
    stride_part_m,
    stride_part_k,
    stride_out_m,
    stride_out_k,
    BLOCK_M: tl.constexpr,
    K: tl.constexpr,
    SPLIT_N: tl.constexpr,
):
    r"""Merge split-N partial top-k buffers into final neighbors."""
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    best_d = tl.full((BLOCK_M, K), float('inf'), tl.float32)
    best_i = tl.full((BLOCK_M, K), -1, tl.int32)
    k_ids = tl.arange(0, K)

    # Merge all split-N partial candidates into the final top-k per row.
    for s in tl.static_range(0, SPLIT_N):
        part_base = (s * stride_part_s + offs_m[:, None] * stride_part_m +
                     k_ids[None, :] * stride_part_k)
        cand_d = tl.load(part_d_ptr + part_base, mask=mask_m[:, None], other=float('inf'))
        cand_i = tl.load(part_i_ptr + part_base, mask=mask_m[:, None], other=-1)
        for kk in tl.static_range(0, K):
            cand_dk = cand_d[:, kk]
            cand_ik = cand_i[:, kk]
            worst_d, worst_idx = tl.max(best_d,
                                        axis=1,
                                        return_indices=True,
                                        return_indices_tie_break_left=True)
            replace = cand_dk < worst_d
            repl_mask = replace[:, None] & (k_ids[None, :] == worst_idx[:, None])
            best_d = tl.where(repl_mask, cand_dk[:, None], best_d)
            best_i = tl.where(repl_mask, cand_ik, best_i)

    out_ptrs = (offs_m[:, None] * stride_out_m +
                k_ids[None, :] * stride_out_k)
    tl.store(out_i_ptr + out_ptrs, best_i, mask=mask_m[:, None])
