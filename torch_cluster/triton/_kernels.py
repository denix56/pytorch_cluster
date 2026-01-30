from __future__ import annotations


import triton
import triton.language as tl
from triton.language.extra import libdevice


def _prune_knn_configs(configs, args, **kwargs):
    d = int(args.get('D', 0))
    max_cand = int(args.get('MAX_CAND', 0))
    pruned = []
    for cfg in configs:
        block_n = cfg.kwargs.get('BLOCK_N', 0)
        block_d = cfg.kwargs.get('BLOCK_D', 0)
        if max_cand > 0 and block_n > max_cand:
            continue
        if d > 0 and block_d > d:
            continue
        pruned.append(cfg)
    if not pruned:
        pruned = [min(configs, key=lambda c: c.kwargs.get('BLOCK_N', 0))]
    return pruned


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
                      num_warps=4,
                      num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32},
                      num_warps=4,
                      num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32},
                      num_warps=4,
                      num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32},
                      num_warps=4,
                      num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},
                      num_warps=8,
                      num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32},
                      num_warps=8,
                      num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 16},
                      num_warps=8,
                      num_stages=2),
    ],
    key=['M', 'N', 'D'],
)
@triton.heuristics({
    'EVEN_N': lambda args: args['N'] % args['BLOCK_N'] == 0,
    'EVEN_K': lambda args: args['D'] % args['BLOCK_K'] == 0,
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
    EVEN_K: tl.constexpr,
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
    pid = tl.program_id(0)  # X block id.
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)  # X indices.
    mask_x = offs_n < N  # Valid x rows.
    if EVEN_N:
        tl.multiple_of(offs_n, 8)  # Hint contiguous access.
        tl.max_contiguous(offs_n, 8)  # Hint vectorization.

    best_dist = tl.full((BLOCK_N,), float('inf'), tl.float32)  # Best dist.
    best_idx = tl.zeros((BLOCK_N,), dtype=tl.int32)  # Best y index.

    if USE_BATCH:
        batch_id = tl.load(batch_x_ptr + offs_n, mask=mask_x, other=0)  # Batch id per x.
        left = tl.load(ptr_y_ptr + batch_id, mask=mask_x, other=0)  # y range start.
        right = tl.load(ptr_y_ptr + batch_id + 1, mask=mask_x, other=0)  # y range end.
    else:
        left = 0  # Full y range.
        right = M

    x_sq = tl.zeros((BLOCK_N,), dtype=tl.float32)  # ||x||^2.
    x_block_ptr_sq = tl.make_block_ptr(
        base=x_ptr,
        shape=(D, N),
        strides=(stride_xd, stride_xm),
        offsets=(0, pid * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )
    for _ in range(0, D, BLOCK_K):
        if EVEN_N and EVEN_K:
            x = tl.load(x_block_ptr_sq)  # Full tile.
        else:
            x = tl.load(
                x_block_ptr_sq,
                boundary_check=(0, 1),
                padding_option="zero",
            )
        x_sq += tl.sum(x * x, axis=0)
        x_block_ptr_sq = tl.advance(x_block_ptr_sq, (BLOCK_K, 0))
    if COSINE:
        inv_x = tl.rsqrt(x_sq + EPS)  # 1/||x||.

    x_block_ptr_base = tl.make_block_ptr(
        base=x_ptr,
        shape=(D, N),
        strides=(stride_xd, stride_xm),
        offsets=(0, pid * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )

    for y_start in range(0, M, BLOCK_M):
        offs_m = y_start + tl.arange(0, BLOCK_M)  # y indices.
        mask_y = offs_m < M  # Valid y rows.
        full_y = y_start + BLOCK_M <= M  # Full tile.

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)  # Dot acc.
        y_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)  # ||y||^2.

        x_block_ptr = x_block_ptr_base
        y_block_ptr = tl.make_block_ptr(
            base=y_ptr,
            shape=(M, D),
            strides=(stride_ym, stride_yd),
            offsets=(y_start, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )

        for _ in range(0, D, BLOCK_K):
            if full_y and EVEN_K:
                y = tl.load(y_block_ptr)  # Full tile.
                if EVEN_N:
                    x = tl.load(x_block_ptr)  # Full tile.
                else:
                    x = tl.load(
                        x_block_ptr,
                        boundary_check=(0, 1),
                        padding_option="zero",
                    )
            else:
                y = tl.load(
                    y_block_ptr,
                    boundary_check=(0, 1),
                    padding_option="zero",
                )
                x = tl.load(
                    x_block_ptr,
                    boundary_check=(0, 1),
                    padding_option="zero",
                )
            acc += tl.dot(y, x, input_precision="ieee")
            y_sq += tl.sum(y * y, axis=1)
            y_block_ptr = tl.advance(y_block_ptr, (0, BLOCK_K))
            x_block_ptr = tl.advance(x_block_ptr, (BLOCK_K, 0))

        if COSINE:
            inv_y = tl.rsqrt(y_sq + EPS)  # 1/||y||.
            dist = 1.0 - acc * (inv_y[:, None] * inv_x[None, :])
        else:
            dist = tl.fma(-2.0, acc, y_sq[:, None] + x_sq[None, :])  # L2^2 distance.

        if full_y:
            valid = tl.broadcast_to(mask_x[None, :], (BLOCK_M, BLOCK_N))
        else:
            valid = mask_y[:, None] & mask_x[None, :]
        if USE_BATCH:
            valid &= (
                (offs_m[:, None] >= left[None, :])
                & (offs_m[:, None] < right[None, :])
            )
        dist = tl.where(valid, dist, float('inf'))

        block_min, block_arg = tl.min(
            dist,
            axis=0,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        block_idx = y_start + block_arg
        better = block_min < best_dist
        best_dist = tl.where(better, block_min, best_dist)
        best_idx = tl.where(better, block_idx, best_idx)

    tl.store(out_ptr + offs_n, best_idx.to(tl.int64), mask=mask_x)  # Write output.


@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_N': 32, 'BLOCK_D': 16},
            num_warps=2,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_N': 32, 'BLOCK_D': 16},
            num_warps=2,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_N': 32, 'BLOCK_D': 32},
            num_warps=2,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_N': 64, 'BLOCK_D': 16},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_N': 64, 'BLOCK_D': 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_N': 64, 'BLOCK_D': 64},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_N': 128, 'BLOCK_D': 32},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_N': 128, 'BLOCK_D': 64},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {'BLOCK_N': 256, 'BLOCK_D': 32},
            num_warps=8,
            num_stages=4,
        ),
    ],
    key=['D', 'MAX_CAND'],
    # prune_configs_by={'early_config_prune': _prune_knn_configs},
)
@triton.heuristics({
    'NUM_D_BLOCKS': lambda args: triton.cdiv(args['D'], args['BLOCK_D']),
    'MAX_X_BLOCKS': lambda args: triton.cdiv(
        args['MAX_CAND'],
        args['BLOCK_N'],
    ),
    'EVEN_D': lambda args: args['D'] % args['BLOCK_D'] == 0,
})
@triton.jit
def _knn_segmented_kernel(
    x_ptr,
    y_ptr,
    ptr_x_ptr,
    example_idx_ptr,
    row_ptr,
    col_ptr,
    M,
    N,
    D,
    MAX_CAND,
    stride_xm,
    stride_xd,
    stride_ym,
    stride_yd,
    K: tl.constexpr,
    K_PAD: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_D_BLOCKS: tl.constexpr,
    MAX_X_BLOCKS: tl.constexpr,
    EVEN_D: tl.constexpr,
    COSINE: tl.constexpr,
    EPS: tl.constexpr,
):
    pid = tl.program_id(0)  # Program id over y rows.
    n_y = pid  # Current y index.
    mask_y = n_y < M  # Mask for valid y.

    k_offsets = tl.arange(0, K_PAD)  # Offsets for padded top-k buffer.
    k_mask = k_offsets < K  # Mask for real k.
    best_dist = tl.full((K_PAD,), float("inf"), tl.float32)  # Best distances.
    best_idx = tl.full((K_PAD,), -1, tl.int32)  # Best indices.

    example_idx = tl.load(
        example_idx_ptr + n_y,
        mask=mask_y,
        other=0,
    )  # Segment id.
    x_start = tl.load(
        ptr_x_ptr + example_idx,
        mask=mask_y,
        other=0,
    )  # Segment start.
    x_end = tl.load(
        ptr_x_ptr + example_idx + 1,
        mask=mask_y,
        other=0,
    )  # Segment end.
    y_sq = tl.zeros((1,), tl.float32)  # ||y||^2 accumulator.
    y_block_ptr = tl.make_block_ptr(
        base=y_ptr,
        shape=(D, M),
        strides=(stride_yd, stride_ym),
        offsets=(0, n_y),
        block_shape=(BLOCK_D, 1),
        order=(0, 1),
    )
    for yi in tl.static_range(NUM_D_BLOCKS):
        if EVEN_D:
            y = tl.load(y_block_ptr)  # Full load for y.
        else:
            y = tl.load(
                y_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
            )  # Tail D.
        y_sq += tl.sum(y * y, axis=0)  # Accumulate ||y||^2.
        y_block_ptr = tl.advance(y_block_ptr, (BLOCK_D, 0))  # Next D tile.
    if COSINE:
        y_rnorm = tl.rsqrt(y_sq + EPS)  # 1/||y|| for cosine.

    if stride_xm % 8 == 0:
        tl.multiple_of(stride_xm, 8)  # Hint alignment.
    if stride_xd % 8 == 0:
        tl.multiple_of(stride_xd, 8)
    if stride_ym % 8 == 0:
        tl.multiple_of(stride_ym, 8)
    if stride_yd % 8 == 0:
        tl.multiple_of(stride_yd, 8)

    for xb in range(MAX_X_BLOCKS):
        x_block_start = x_start + xb * BLOCK_N  # Start of x block.
        offs_n = tl.arange(0, BLOCK_N)  # Offsets within block.
        x_idx = x_block_start + offs_n  # Global x indices.
        full_block = (
            (x_block_start + BLOCK_N <= x_end)
            & (x_block_start + BLOCK_N <= N)
            & mask_y
        )
        mask_x = tl.where(
            full_block,
            mask_y,
            (x_idx < x_end) & (x_idx < N) & mask_y,
        )
        if MAX_X_BLOCKS > 1:
            tl.multiple_of(x_block_start, 8)  # Hint alignment for block start.
        tl.multiple_of(offs_n, 8)  # Hint vectorization.
        tl.max_contiguous(offs_n, 8)

        acc_dot = tl.zeros((BLOCK_N,), tl.float32)  # Dot accumulator.
        acc_x_sq = tl.zeros((BLOCK_N,), tl.float32)  # ||x||^2 accumulator.
        x_block_start_i32 = x_block_start.to(
            tl.int32
        )  # Block ptr needs int32 offsets.
        n_y_i32 = n_y.to(tl.int32)
        x_block_ptr = tl.make_block_ptr(
            base=x_ptr,
            shape=(N, D),
            strides=(stride_xm, stride_xd),
            offsets=(x_block_start_i32, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )
        y_block_ptr = tl.make_block_ptr(
            base=y_ptr,
            shape=(D, M),
            strides=(stride_yd, stride_ym),
            offsets=(0, n_y_i32),
            block_shape=(BLOCK_D, 1),
            order=(0, 1),
        )
        for nd in tl.static_range(NUM_D_BLOCKS):
            if EVEN_D:
                x = tl.load(x_block_ptr)  # Full load for x.
                y = tl.load(y_block_ptr)  # Full load for y.
            else:
                x = tl.load(
                    x_block_ptr,
                    boundary_check=(0, 1),
                    padding_option="zero",
                )  # Tail D.
                y = tl.load(
                    y_block_ptr,
                    boundary_check=(0, 1),
                    padding_option="zero",
                )
            prod = tl.dot(x, y, input_precision='ieee')  # Matrix-vector dot.
            acc_dot += tl.sum(prod, axis=1)  # Reduce over D.
            acc_x_sq += tl.sum(x * x, axis=1)  # ||x||^2.
            x_block_ptr = tl.advance(x_block_ptr, (0, BLOCK_D))  # Next D tile.
            y_block_ptr = tl.advance(y_block_ptr, (BLOCK_D, 0))

        if COSINE:
            x_rnorm = tl.rsqrt(acc_x_sq + EPS)  # 1/||x||.
            dist = 1.0 - acc_dot * (x_rnorm * y_rnorm)  # Cosine distance.
        else:
            dist = (acc_x_sq + y_sq) - 2.0 * acc_dot  # L2^2 distance.

        if not full_block:
            dist = tl.where(mask_x, dist, float("inf"))  # Mask invalid x.

        dist_bits = dist.to(tl.int32, bitcast=True).to(tl.int64)  # Pack dist.
        idx_bits = x_idx.to(tl.int32).to(tl.int64) & 0xFFFFFFFF  # Pack idx.
        key = (dist_bits << 32) | idx_bits  # Lexicographic key.
        sorted_key = tl.sort(key, descending=False)  # Full block sort.

        k_valid = k_offsets < K  # Only first k entries are valid.
        key_k = tl.gather(sorted_key, k_offsets, axis=0)  # Take first k keys.
        key_k = tl.where(k_valid, key_k, 0)
        dist_k = (
            (key_k >> 32)
            .to(tl.int32)
            .to(tl.float32, bitcast=True)
        )  # Unpack dist.
        idx_k = (key_k & 0xFFFFFFFF).to(tl.int32)  # Unpack idx.
        block_top_dist = tl.where(
            k_valid,
            dist_k,
            float("inf"),
        )  # Block top-k dist.
        block_top_idx = tl.where(k_valid, idx_k, -1)  # Block top-k idx.

        shift_idx = k_offsets - K  # Shift window for block top-k.
        shift_valid = (k_offsets >= K) & (k_offsets < (2 * K))
        shift_idx_safe = tl.where(shift_valid, shift_idx, 0)
        shift_dist = tl.gather(block_top_dist, shift_idx_safe, axis=0)
        block_shifted_dist = tl.where(shift_valid, shift_dist, float("inf"))
        shift_idx_val = tl.gather(block_top_idx, shift_idx_safe, axis=0)
        block_shifted_idx = tl.where(shift_valid, shift_idx_val, -1)
        combo_dist = tl.where(
            k_offsets < K,
            best_dist,
            block_shifted_dist,
        )  # Merge buffers.
        combo_idx = tl.where(k_offsets < K, best_idx, block_shifted_idx)
        dist_bits = (
            combo_dist.to(tl.int32, bitcast=True)
            .to(tl.int64)
        )  # Pack combo.
        idx_bits = combo_idx.to(tl.int32).to(tl.int64) & 0xFFFFFFFF
        combo_key = (dist_bits << 32) | idx_bits
        sorted_key = tl.sort(combo_key, descending=False)  # Sort 2K keys.
        best_key = tl.gather(sorted_key, k_offsets, axis=0)  # Keep best K.
        best_dist = (best_key >> 32).to(tl.int32).to(tl.float32, bitcast=True)
        best_idx = (best_key & 0xFFFFFFFF).to(tl.int32)

    best_idx = tl.where(
        libdevice.isinf(best_dist),
        -1,
        best_idx,
    )  # Mask invalid.

    out_offsets = n_y * K + tl.where(k_mask, k_offsets, 0)  # Output offsets.
    out_mask = mask_y & k_mask  # Valid output mask.
    tl.store(row_ptr + out_offsets, n_y, mask=out_mask)  # Write row ids.
    tl.store(
        col_ptr + out_offsets,
        best_idx.to(tl.int64),
        mask=out_mask,
    )  # Write col idx.
