import importlib.util
import time

import pytest
import torch

from torch_cluster import (
    fps,
    graclus_cluster,
    grid_cluster,
    knn,
    knn_graph,
    nearest,
    radius,
    radius_graph,
    random_walk,
)

HAS_CUDA = torch.cuda.is_available()
HAS_TRITON = importlib.util.find_spec('triton') is not None

if HAS_TRITON:
    from torch_cluster import (
        fps__triton,
        graclus_cluster__triton,
        grid_cluster__triton,
        knn__triton,
        knn_graph__triton,
        nearest__triton,
        radius__triton,
        radius_graph__triton,
        random_walk__triton,
    )

pytestmark = pytest.mark.skipif(
    not (HAS_CUDA and HAS_TRITON),
    reason='CUDA and Triton are required for Triton parity tests.',
)


def _sort_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    key = edge_index[0] * num_nodes + edge_index[1]
    perm = key.argsort()
    return edge_index[:, perm]


def _benchmark(fn, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def test_knn_triton_matches_cuda():
    torch.manual_seed(42)
    x = torch.randn(128, 16, device='cuda')
    y = torch.randn(64, 16, device='cuda')
    batch_x = torch.zeros(x.size(0), dtype=torch.long, device='cuda')
    batch_y = torch.zeros(y.size(0), dtype=torch.long, device='cuda')

    out_cuda = knn(x, y, k=8, batch_x=batch_x, batch_y=batch_y)
    out_triton = knn__triton(x, y, k=8, batch_x=batch_x, batch_y=batch_y)
    assert torch.equal(
        _sort_edge_index(out_cuda, x.size(0)),
        _sort_edge_index(out_triton, x.size(0)),
    )

    out_cuda = knn(x, y, k=8, batch_x=batch_x, batch_y=batch_y, cosine=True)
    out_triton = knn__triton(x,
                             y,
                             k=8,
                             batch_x=batch_x,
                             batch_y=batch_y,
                             cosine=True)
    assert torch.equal(
        _sort_edge_index(out_cuda, x.size(0)),
        _sort_edge_index(out_triton, x.size(0)),
    )


def test_knn_graph_triton_matches_cuda():
    torch.manual_seed(1)
    x = torch.randn(64, 8, device='cuda')
    batch = torch.zeros(x.size(0), dtype=torch.long, device='cuda')

    out_cuda = knn_graph(x, k=4, batch=batch, loop=False)
    out_triton = knn_graph__triton(x, k=4, batch=batch, loop=False)
    assert torch.equal(
        _sort_edge_index(out_cuda, x.size(0)),
        _sort_edge_index(out_triton, x.size(0)),
    )


def test_radius_triton_matches_cuda():
    torch.manual_seed(7)
    x = torch.randn(128, 3, device='cuda')
    y = torch.randn(64, 3, device='cuda')
    batch_x = torch.zeros(x.size(0), dtype=torch.long, device='cuda')
    batch_y = torch.zeros(y.size(0), dtype=torch.long, device='cuda')

    out_cuda = radius(x,
                      y,
                      r=1.5,
                      batch_x=batch_x,
                      batch_y=batch_y,
                      max_num_neighbors=x.size(0))
    out_triton = radius__triton(x,
                                y,
                                r=1.5,
                                batch_x=batch_x,
                                batch_y=batch_y,
                                max_num_neighbors=x.size(0))
    assert torch.equal(
        _sort_edge_index(out_cuda, x.size(0)),
        _sort_edge_index(out_triton, x.size(0)),
    )

    out_cuda = radius(x,
                      y,
                      r=1.5,
                      batch_x=batch_x,
                      batch_y=batch_y,
                      max_num_neighbors=x.size(0),
                      ignore_same_index=True)
    out_triton = radius__triton(x,
                                y,
                                r=1.5,
                                batch_x=batch_x,
                                batch_y=batch_y,
                                max_num_neighbors=x.size(0),
                                ignore_same_index=True)
    assert torch.equal(
        _sort_edge_index(out_cuda, x.size(0)),
        _sort_edge_index(out_triton, x.size(0)),
    )


def test_radius_graph_triton_matches_cuda():
    torch.manual_seed(3)
    x = torch.randn(64, 4, device='cuda')
    batch = torch.zeros(x.size(0), dtype=torch.long, device='cuda')

    out_cuda = radius_graph(x, r=2.0, batch=batch, loop=False)
    out_triton = radius_graph__triton(x, r=2.0, batch=batch, loop=False)
    assert torch.equal(
        _sort_edge_index(out_cuda, x.size(0)),
        _sort_edge_index(out_triton, x.size(0)),
    )


def test_nearest_triton_matches_cuda():
    torch.manual_seed(123)
    x = torch.randn(128, 8, device='cuda')
    y = torch.randn(32, 8, device='cuda')
    batch_x = torch.zeros(x.size(0), dtype=torch.long, device='cuda')
    batch_y = torch.zeros(y.size(0), dtype=torch.long, device='cuda')

    out_cuda = nearest(x, y, batch_x, batch_y)
    out_triton = nearest__triton(x, y, batch_x, batch_y)
    assert torch.equal(out_cuda, out_triton)


def test_grid_cluster_triton_matches_cuda():
    torch.manual_seed(5)
    pos = torch.randn(128, 3, device='cuda')
    size = torch.tensor([0.5, 0.5, 0.5], device='cuda')

    out_cuda = grid_cluster(pos, size)
    out_triton = grid_cluster__triton(pos, size)
    assert torch.equal(out_cuda, out_triton)


def test_fps_triton_matches_cuda():
    torch.manual_seed(11)
    src = torch.randn(256, 3, device='cuda')
    batch = torch.zeros(src.size(0), dtype=torch.long, device='cuda')

    out_cuda = fps(src, batch=batch, ratio=0.25, random_start=False)
    out_triton = fps__triton(src, batch=batch, ratio=0.25, random_start=False)
    assert torch.equal(out_cuda, out_triton)


def test_graclus_triton_matches_cuda_on_empty_graph():
    row = torch.empty(0, dtype=torch.long, device='cuda')
    col = torch.empty(0, dtype=torch.long, device='cuda')
    out_cuda = graclus_cluster(row, col, num_nodes=4)
    out_triton = graclus_cluster__triton(row, col, num_nodes=4)
    assert torch.equal(out_cuda, out_triton)


def test_random_walk_triton_matches_cuda_on_deterministic_graph():
    row = torch.tensor([0, 1, 2, 3], device='cuda')
    col = torch.tensor([1, 2, 3, 0], device='cuda')
    start = torch.tensor([0, 1, 2, 3], device='cuda')

    node_cuda, edge_cuda = random_walk(row,
                                       col,
                                       start,
                                       walk_length=4,
                                       return_edge_indices=True)
    node_triton, edge_triton = random_walk__triton(row,
                                                   col,
                                                   start,
                                                   walk_length=4,
                                                   return_edge_indices=True)
    assert torch.equal(node_cuda, node_triton)
    assert torch.equal(edge_cuda, edge_triton)


def test_triton_edge_cases():
    empty = torch.empty(0, 2, device='cuda')
    out_cuda = knn(empty, empty, k=2)
    out_triton = knn__triton(empty, empty, k=2)
    assert torch.equal(out_cuda, out_triton)

    out_cuda = radius(empty, empty, r=1.0)
    out_triton = radius__triton(empty, empty, r=1.0)
    assert torch.equal(out_cuda, out_triton)

    x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    y = torch.tensor([1.5, 2.5], device='cuda')
    out_cuda = knn(x, y, k=1)
    out_triton = knn__triton(x, y, k=1)
    assert torch.equal(out_cuda, out_triton)


@pytest.mark.parametrize('num_x,num_y', [(256, 128), (1024, 512), (4096, 2048)])
def test_triton_knn_performance(num_x, num_y):
    torch.manual_seed(99)
    x = torch.randn(num_x, 16, device='cuda')
    y = torch.randn(num_y, 16, device='cuda')
    batch_x = torch.zeros(num_x, dtype=torch.long, device='cuda')
    batch_y = torch.zeros(num_y, dtype=torch.long, device='cuda')

    def cuda_fn():
        knn(x, y, k=16, batch_x=batch_x, batch_y=batch_y)

    def triton_fn():
        knn__triton(x, y, k=16, batch_x=batch_x, batch_y=batch_y)

    cuda_time = _benchmark(cuda_fn)
    triton_time = _benchmark(triton_fn)
    assert triton_time <= cuda_time * 20


@pytest.mark.parametrize('num_x,num_y', [(256, 128), (1024, 512), (4096, 2048)])
def test_triton_radius_performance(num_x, num_y):
    torch.manual_seed(199)
    x = torch.randn(num_x, 8, device='cuda')
    y = torch.randn(num_y, 8, device='cuda')
    batch_x = torch.zeros(num_x, dtype=torch.long, device='cuda')
    batch_y = torch.zeros(num_y, dtype=torch.long, device='cuda')

    def cuda_fn():
        radius(x,
               y,
               r=0.5,
               batch_x=batch_x,
               batch_y=batch_y,
               max_num_neighbors=num_x)

    def triton_fn():
        radius__triton(x,
                       y,
                       r=0.5,
                       batch_x=batch_x,
                       batch_y=batch_y,
                       max_num_neighbors=num_x)

    cuda_time = _benchmark(cuda_fn)
    triton_time = _benchmark(triton_fn)
    assert triton_time <= cuda_time * 20
