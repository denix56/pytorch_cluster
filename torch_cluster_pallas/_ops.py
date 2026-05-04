from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _pad_rows(x, rows: int):
    if rows == x.shape[0]:
        return x
    return jnp.pad(x, ((0, rows - x.shape[0]), (0, 0)))


def _compact(edge_index, mask):
    return edge_index[:, mask.reshape(-1)]


def _batch_mask(
    batch_x,
    batch_y,
    num_x: int,
    num_y: int,
):
    if batch_x is None and batch_y is None:
        return jnp.ones((num_y, num_x), dtype=bool)
    if batch_x is None:
        batch_x = jnp.zeros((num_x, ), dtype=jnp.int32)
    if batch_y is None:
        batch_y = jnp.zeros((num_y, ), dtype=jnp.int32)
    return batch_y[:, None] == batch_x[None, :]


def _pairwise_distance_kernel(x_ref, y_ref, out_ref):
    x = x_ref[:, :].astype(jnp.float32)
    y = y_ref[:, :].astype(jnp.float32)
    diff = y[:, None, :] - x[None, :, :]
    out_ref[:, :] = jnp.sum(diff * diff, axis=-1)


def _pairwise_distance_pallas(
    x,
    y,
    *,
    block_m: int = 64,
    block_n: int = 128,
    interpret: bool = False,
):
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("'x' and 'y' must be two-dimensional arrays")
    if x.shape[1] != y.shape[1]:
        raise ValueError("'x' and 'y' must have the same feature dimension")
    if block_m <= 0 or block_n <= 0:
        raise ValueError("'block_m' and 'block_n' must be positive")

    n, dim = x.shape
    m = y.shape[0]
    if n == 0 or m == 0:
        return jnp.zeros((m, n), dtype=jnp.float32)

    padded_m = _ceil_div(m, block_m) * block_m
    padded_n = _ceil_div(n, block_n) * block_n
    x_pad = _pad_rows(x, padded_n)
    y_pad = _pad_rows(y, padded_m)

    out = pl.pallas_call(
        _pairwise_distance_kernel,
        out_shape=jax.ShapeDtypeStruct((padded_m, padded_n), jnp.float32),
        grid=(_ceil_div(padded_m, block_m), _ceil_div(padded_n, block_n)),
        in_specs=[
            pl.BlockSpec((block_n, dim), lambda i, j: (j, 0)),
            pl.BlockSpec((block_m, dim), lambda i, j: (i, 0)),
        ],
        out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
        interpret=interpret,
    )(x_pad, y_pad)
    return out[:m, :n]


def _should_use_pallas(backend: str) -> bool:
    if backend == 'pallas':
        return True
    if backend != 'auto':
        return False
    return any(device.platform == 'tpu' for device in jax.devices())


def pairwise_distance(
    x,
    y,
    *,
    backend: str = 'auto',
    block_m: int = 64,
    block_n: int = 128,
    interpret: bool = False,
):
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("'x' and 'y' must be two-dimensional arrays")
    if x.shape[1] != y.shape[1]:
        raise ValueError("'x' and 'y' must have the same feature dimension")
    if x.shape[0] == 0 or y.shape[0] == 0:
        return jnp.zeros((y.shape[0], x.shape[0]), dtype=jnp.float32)

    if _should_use_pallas(backend):
        return _pairwise_distance_pallas(x,
                                         y,
                                         block_m=block_m,
                                         block_n=block_n,
                                         interpret=interpret)
    if backend not in {'auto', 'jax'}:
        raise ValueError("'backend' must be one of 'auto', 'jax', or 'pallas'")
    diff = y[:, None, :].astype(jnp.float32) - x[None, :, :].astype(jnp.float32)
    return jnp.sum(diff * diff, axis=-1)


def grid_cluster(pos, size, start=None, end=None):
    pos = jnp.asarray(pos)
    size = jnp.asarray(size)
    pos = pos.reshape((pos.shape[0], -1))
    if start is None:
        start = jnp.min(pos, axis=0)
    if end is None:
        end = jnp.max(pos, axis=0)

    start = jnp.asarray(start)
    end = jnp.asarray(end)
    pos = pos - start[None, :]
    num_voxels = ((end - start) / size).astype(jnp.int32) + 1
    multipliers = jnp.cumprod(num_voxels)
    multipliers = jnp.concatenate(
        [jnp.ones((1, ), dtype=multipliers.dtype), multipliers[:-1]])
    coords = (pos / size.reshape((1, -1))).astype(jnp.int32)
    return jnp.sum(coords * multipliers.reshape((1, -1)), axis=1)


def knn(
    x,
    y,
    k: int,
    batch_x=None,
    batch_y=None,
    *,
    backend: str = 'auto',
    block_m: int = 64,
    block_n: int = 128,
    interpret: bool = False,
    compact: bool = True,
) -> Tuple[jax.Array, Optional[jax.Array]]:
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    x = x.reshape((-1, 1)) if x.ndim == 1 else x
    y = y.reshape((-1, 1)) if y.ndim == 1 else y
    k = min(k, x.shape[0])
    if x.shape[0] == 0 or y.shape[0] == 0 or k == 0:
        edge_index = jnp.empty((2, 0), dtype=jnp.int32)
        if compact:
            return edge_index, None
        return edge_index, jnp.zeros((y.shape[0], 0), dtype=bool)

    distances = pairwise_distance(x,
                                  y,
                                  backend=backend,
                                  block_m=block_m,
                                  block_n=block_n,
                                  interpret=interpret)
    mask = _batch_mask(batch_x, batch_y, x.shape[0], y.shape[0])
    distances = jnp.where(mask, distances, jnp.inf)
    values, indices = jax.lax.top_k(-distances, k)
    valid = jnp.isfinite(-values)

    row = jnp.broadcast_to(jnp.arange(y.shape[0])[:, None], indices.shape)
    col = jnp.where(valid, indices, -1)
    edge_index = jnp.stack([row.reshape(-1), col.reshape(-1)], axis=0)
    if compact:
        return _compact(edge_index, valid), None
    return edge_index, valid


def knn_graph(
    x,
    k: int,
    batch=None,
    *,
    loop: bool = False,
    flow: str = 'source_to_target',
    backend: str = 'auto',
    block_m: int = 64,
    block_n: int = 128,
    interpret: bool = False,
    compact: bool = True,
):
    if flow not in {'source_to_target', 'target_to_source'}:
        raise ValueError("'flow' must be 'source_to_target' or "
                         "'target_to_source'")
    edge_index, valid = knn(x,
                            x,
                            k if loop else k + 1,
                            batch,
                            batch,
                            backend=backend,
                            block_m=block_m,
                            block_n=block_n,
                            interpret=interpret,
                            compact=False)
    row, col = edge_index[1], edge_index[0]
    if flow == 'target_to_source':
        row, col = col, row

    keep = valid.reshape(-1)
    if not loop:
        keep = keep & (row != col)

    edge_index = jnp.stack([row, col], axis=0)
    if compact:
        return edge_index[:, keep], None
    return edge_index, keep.reshape(valid.shape)


def radius(
    x,
    y,
    r: float,
    batch_x=None,
    batch_y=None,
    *,
    max_num_neighbors: int = 32,
    ignore_same_index: bool = False,
    backend: str = 'auto',
    block_m: int = 64,
    block_n: int = 128,
    interpret: bool = False,
    compact: bool = True,
):
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    x = x.reshape((-1, 1)) if x.ndim == 1 else x
    y = y.reshape((-1, 1)) if y.ndim == 1 else y
    if x.shape[0] == 0 or y.shape[0] == 0 or max_num_neighbors == 0:
        edge_index = jnp.empty((2, 0), dtype=jnp.int32)
        if compact:
            return edge_index, None
        return edge_index, jnp.zeros((y.shape[0], 0), dtype=bool)

    distances = pairwise_distance(x,
                                  y,
                                  backend=backend,
                                  block_m=block_m,
                                  block_n=block_n,
                                  interpret=interpret)
    valid = distances < float(r) * float(r)
    valid = valid & _batch_mask(batch_x, batch_y, x.shape[0], y.shape[0])
    if ignore_same_index:
        valid = valid & (jnp.arange(x.shape[0])[None, :] !=
                         jnp.arange(y.shape[0])[:, None])

    order = jnp.arange(x.shape[0])[None, :]
    sort_key = jnp.where(valid, order, x.shape[0] + order)
    col = jnp.sort(sort_key, axis=1)[:, :max_num_neighbors]
    out_mask = col < x.shape[0]
    col = jnp.where(out_mask, col, -1)
    row = jnp.broadcast_to(jnp.arange(y.shape[0])[:, None], col.shape)
    edge_index = jnp.stack([row.reshape(-1), col.reshape(-1)], axis=0)
    if compact:
        return _compact(edge_index, out_mask), None
    return edge_index, out_mask


def radius_graph(
    x,
    r: float,
    batch=None,
    *,
    loop: bool = False,
    max_num_neighbors: int = 32,
    flow: str = 'source_to_target',
    backend: str = 'auto',
    block_m: int = 64,
    block_n: int = 128,
    interpret: bool = False,
    compact: bool = True,
):
    if flow not in {'source_to_target', 'target_to_source'}:
        raise ValueError("'flow' must be 'source_to_target' or "
                         "'target_to_source'")
    edge_index, valid = radius(x,
                               x,
                               r,
                               batch,
                               batch,
                               max_num_neighbors=max_num_neighbors,
                               ignore_same_index=not loop,
                               backend=backend,
                               block_m=block_m,
                               block_n=block_n,
                               interpret=interpret,
                               compact=False)
    row, col = edge_index[1], edge_index[0]
    if flow == 'target_to_source':
        row, col = col, row

    edge_index = jnp.stack([row, col], axis=0)
    if compact:
        return edge_index[:, valid.reshape(-1)], None
    return edge_index, valid


def nearest(
    x,
    y,
    batch_x=None,
    batch_y=None,
    *,
    backend: str = 'auto',
    block_m: int = 64,
    block_n: int = 128,
    interpret: bool = False,
):
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    x = x.reshape((-1, 1)) if x.ndim == 1 else x
    y = y.reshape((-1, 1)) if y.ndim == 1 else y
    if x.shape[0] == 0:
        return jnp.empty((0, ), dtype=jnp.int32)
    if y.shape[0] == 0:
        raise ValueError("'y' must contain at least one point")

    distances = pairwise_distance(y,
                                  x,
                                  backend=backend,
                                  block_m=block_m,
                                  block_n=block_n,
                                  interpret=interpret)
    mask = _batch_mask(batch_y, batch_x, y.shape[0], x.shape[0])
    distances = jnp.where(mask, distances, jnp.inf)
    return jnp.argmin(distances, axis=1).astype(jnp.int32)
