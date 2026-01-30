import importlib.util
from itertools import product

import pytest
import torch
import torch_cluster as tc

knn = tc.knn
knn_graph = tc.knn_graph


pytestmark = pytest.mark.skipif(
    not (
        torch.ops.torch_cluster.cuda_version() != -1
        and importlib.util.find_spec('triton') is not None
    ),
    reason='CUDA and Triton are required for Triton benchmark tests.',
)

KNN_SIZES = [
    (256, 128),
    (512, 256),
    (1024, 512),
    (2048, 1024),
    (4096, 2048),
    (8192, 4096),
    (8192, 8192),
    (8201, 4103),
    (255, 127),
    (256, 5),
    (1024, 5),
    (4096, 5),
    (255, 5),
]
KNN_GROUPS = [1, 2, 4, 8, 16, 32]

FEATURES = [8, 64, 200]


def to_set(edge_index):
    return set([(i, j) for i, j in edge_index.t().tolist()])


def _make_batch(
    num_nodes: int,
    num_groups: int,
    device: torch.device,
) -> torch.Tensor:
    groups = max(1, min(num_groups, num_nodes))
    counts = torch.full(
        (groups,),
        num_nodes // groups,
        device=device,
        dtype=torch.long,
    )
    remainder = num_nodes % groups
    if remainder:
        counts[:remainder] += 1
    return torch.repeat_interleave(
        torch.arange(groups, device=device),
        counts,
    )


def _knn_param_grid():
    return (
        (*p[0], p[1], p[2])
        for p in product(KNN_SIZES, KNN_GROUPS, FEATURES)
        if p[1] <= min(p[0])
    )


@pytest.mark.parametrize(
    'num_x,num_y,num_groups,num_features',
    _knn_param_grid(),
)
@pytest.mark.benchmark(group="knn")
def test_triton_knn_benchmark_cuda(
    benchmark,
    num_x,
    num_y,
    num_groups,
    num_features,
):
    torch.manual_seed(99)
    x = torch.randn(num_x, num_features, device='cuda')
    y = torch.randn(num_y, num_features, device='cuda')
    groups = min(num_groups, x.size(0), y.size(0))
    batch_x = _make_batch(num_x, groups, x.device)
    batch_y = _make_batch(num_y, groups, y.device)

    def cuda_fn():
        knn(
            x,
            y,
            k=16,
            batch_x=batch_x,
            batch_y=batch_y,
            cosine=False,
            use_triton=False,
        )

    for _ in range(5):
        cuda_fn()
        torch.cuda.synchronize()

    benchmark(cuda_fn)
    print(
        f"[knn][cuda] num_x={num_x} num_y={num_y} groups={groups}"
    )


@pytest.mark.parametrize(
    'num_x,num_y,num_groups,num_features',
    _knn_param_grid(),
)
@pytest.mark.benchmark(group="knn_cosine")
def test_triton_knn_benchmark_cuda_cosine(
    benchmark,
    num_x,
    num_y,
    num_groups,
    num_features,
):
    torch.manual_seed(99)
    x = torch.randn(num_x, num_features, device='cuda')
    y = torch.randn(num_y, num_features, device='cuda')
    groups = min(num_groups, x.size(0), y.size(0))
    batch_x = _make_batch(num_x, groups, x.device)
    batch_y = _make_batch(num_y, groups, y.device)

    def cuda_fn():
        knn(
            x,
            y,
            k=16,
            batch_x=batch_x,
            batch_y=batch_y,
            cosine=True,
            use_triton=False,
        )

    for _ in range(5):
        cuda_fn()
        torch.cuda.synchronize()

    benchmark(cuda_fn)
    print(
        f"[knn][cuda] num_x={num_x} num_y={num_y} groups={groups}"
    )


@pytest.mark.parametrize(
    'num_x,num_y,num_groups,num_features',
    _knn_param_grid(),
)
@pytest.mark.benchmark(group="knn_cosine")
def test_triton_knn_benchmark_triton_cosine(
    benchmark,
    num_x,
    num_y,
    num_groups,
    num_features,
):
    torch.manual_seed(99)
    x = torch.randn(num_x, num_features, device='cuda')
    y = torch.randn(num_y, num_features, device='cuda')
    groups = min(num_groups, x.size(0), y.size(0))
    batch_x = _make_batch(num_x, groups, x.device)
    batch_y = _make_batch(num_y, groups, y.device)

    def cuda_fn():
        return knn(
            x,
            y,
            k=16,
            batch_x=batch_x,
            batch_y=batch_y,
            cosine=True,
            use_triton=False,
        )

    def triton_fn():
        return knn(
            x,
            y,
            k=16,
            batch_x=batch_x,
            batch_y=batch_y,
            cosine=True,
            use_triton=True,
        )

    for i in range(5):
        if i == 0:
            out_cuda = cuda_fn()
            out_triton = triton_fn()
            for a, b in zip(
                sorted(list(to_set(out_cuda))),
                sorted(list(to_set(out_triton))),
            ):
                assert a == b
        else:
            triton_fn()
        torch.cuda.synchronize()

    benchmark(triton_fn)
    print(
        f"[knn][triton] num_x={num_x} num_y={num_y} groups={groups}"
    )


@pytest.mark.parametrize(
    'num_x,num_y,num_groups,num_features',
    _knn_param_grid(),
)
@pytest.mark.benchmark(group="knn")
def test_triton_knn_benchmark_triton(
    benchmark,
    num_x,
    num_y,
    num_groups,
    num_features,
):
    torch.manual_seed(99)
    x = torch.randn(num_x, num_features, device='cuda')
    y = torch.randn(num_y, num_features, device='cuda')
    groups = min(num_groups, x.size(0), y.size(0))
    batch_x = _make_batch(num_x, groups, x.device)
    batch_y = _make_batch(num_y, groups, y.device)

    def cuda_fn():
        return knn(
            x,
            y,
            k=16,
            batch_x=batch_x,
            batch_y=batch_y,
            cosine=False,
            use_triton=False,
        )

    def triton_fn():
        return knn(
            x,
            y,
            k=16,
            batch_x=batch_x,
            batch_y=batch_y,
            cosine=False,
            use_triton=True,
        )

    for i in range(5):
        if i == 0:
            out_cuda = cuda_fn()
            out_triton = triton_fn()
            for a, b in zip(
                sorted(list(to_set(out_cuda))),
                sorted(list(to_set(out_triton))),
            ):
                assert a == b
        else:
            triton_fn()
        torch.cuda.synchronize()

    benchmark(triton_fn)
    print(
        f"[knn][triton] num_x={num_x} num_y={num_y} groups={groups}"
    )


@pytest.mark.parametrize('num_x', [256, 1024, 4096, 8192, 255])
@pytest.mark.parametrize('num_groups', [1, 2, 4, 6, 8, 16, 24, 32])
@pytest.mark.benchmark(group="knn_graph")
def test_triton_knn_graph_benchmark_cuda(benchmark, num_x, num_groups):
    torch.manual_seed(199)
    x = torch.randn(num_x, 8, device='cuda')
    groups = min(num_groups, x.size(0))
    batch = _make_batch(num_x, groups, x.device)
    k = min(16, max(1, num_x - 1))

    def cuda_fn():
        knn_graph(x, k=k, batch=batch, loop=False, use_triton=False)

    for _ in range(5):
        cuda_fn()
        torch.cuda.synchronize()

    benchmark(cuda_fn)
    print(
        f"[knn_graph][cuda] num_x={num_x} groups={groups} k={k}"
    )


@pytest.mark.parametrize('num_x', [256, 1024, 4096, 8192, 255])
@pytest.mark.parametrize('num_groups', [1, 2, 4, 6, 8, 16, 24, 32])
@pytest.mark.benchmark(group="knn_graph")
def test_triton_knn_graph_benchmark_triton(benchmark, num_x, num_groups):
    torch.manual_seed(199)
    x = torch.randn(num_x, 8, device='cuda')
    groups = min(num_groups, x.size(0))
    batch = _make_batch(num_x, groups, x.device)
    k = min(16, max(1, num_x - 1))

    def cuda_fn():
        return knn_graph(
            x,
            k=k,
            batch=batch,
            loop=False,
            use_triton=False,
        )

    def triton_fn():
        return knn_graph(
            x,
            k=k,
            batch=batch,
            loop=False,
            use_triton=True,
        )

    for i in range(5):
        if i == 0:
            out_cuda = cuda_fn()
            out_triton = triton_fn()
            for a, b in zip(
                sorted(list(to_set(out_cuda))),
                sorted(list(to_set(out_triton))),
            ):
                assert a == b
        else:
            triton_fn()
        torch.cuda.synchronize()

    benchmark(triton_fn)
    print(f"[knn_graph][triton] num_x={num_x} groups={groups} k={k}")


@pytest.mark.parametrize('num_x', [256, 1024, 4096, 8192, 255])
@pytest.mark.parametrize('num_groups', [1, 2, 4, 6, 8, 16, 24, 32])
@pytest.mark.benchmark(group="knn_graph_cosine")
def test_triton_knn_graph_benchmark_cuda_cosine(benchmark, num_x, num_groups):
    torch.manual_seed(199)
    x = torch.randn(num_x, 8, device='cuda')
    groups = min(num_groups, x.size(0))
    batch = _make_batch(num_x, groups, x.device)
    k = min(16, max(1, num_x - 1))

    def cuda_fn():
        knn_graph(
            x,
            k=k,
            batch=batch,
            loop=False,
            use_triton=False,
            cosine=True,
        )

    for _ in range(5):
        cuda_fn()
        torch.cuda.synchronize()

    benchmark(cuda_fn)
    print(f"[knn_graph][cuda] num_x={num_x} groups={groups} k={k}")


@pytest.mark.parametrize('num_x', [256, 1024, 4096, 8192, 255])
@pytest.mark.parametrize('num_groups', [1, 2, 4, 6, 8, 16, 24, 32])
@pytest.mark.benchmark(group="knn_graph_cosine")
def test_triton_knn_graph_benchmark_triton_cosine(
    benchmark,
    num_x,
    num_groups,
):
    torch.manual_seed(199)
    x = torch.randn(num_x, 8, device='cuda')
    groups = min(num_groups, x.size(0))
    batch = _make_batch(num_x, groups, x.device)
    k = min(16, max(1, num_x - 1))

    def cuda_fn():
        return knn_graph(
            x,
            k=k,
            batch=batch,
            loop=False,
            use_triton=False,
            cosine=True,
        )

    def triton_fn():
        return knn_graph(
            x,
            k=k,
            batch=batch,
            loop=False,
            use_triton=True,
            cosine=True,
        )

    for i in range(5):
        if i == 0:
            out_cuda = cuda_fn()
            out_triton = triton_fn()
            for a, b in zip(
                sorted(list(to_set(out_cuda))),
                sorted(list(to_set(out_triton))),
            ):
                assert a == b
        else:
            triton_fn()
        torch.cuda.synchronize()

    benchmark(triton_fn)
    print(f"[knn_graph][triton] num_x={num_x} groups={groups} k={k}")
