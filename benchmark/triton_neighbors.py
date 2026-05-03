import argparse
import importlib.util
import statistics
import time

import torch

from torch_cluster import knn, radius
from torch_cluster import knn__triton, radius__triton


def _parse_ints(value):
    return [int(item) for item in value.split(',')]


def _parse_floats(value):
    return [float(item) for item in value.split(',')]


def _sort_edges(edge_index, num_x):
    key = edge_index[0] * num_x + edge_index[1]
    return edge_index[:, key.argsort()]


def _bench(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    return {
        'median_ms': statistics.median(times),
        'p90_ms': sorted(times)[int(0.9 * (len(times) - 1))],
        'min_ms': min(times),
    }


def _make_batch(numel, batches):
    if batches == 1:
        return torch.zeros(numel, dtype=torch.long, device='cuda')
    base = torch.arange(batches, device='cuda').repeat_interleave(
        numel // batches)
    tail = torch.full((numel - base.numel(), ),
                      batches - 1,
                      dtype=torch.long,
                      device='cuda')
    return torch.cat([base, tail])


def _make_inputs(num_x, num_y, dim, dtype, batches):
    x = torch.randn(num_x, dim, device='cuda', dtype=dtype)
    y = torch.randn(num_y, dim, device='cuda', dtype=dtype)
    batch_x = _make_batch(num_x, batches)
    batch_y = _make_batch(num_y, batches)
    return x, y, batch_x, batch_y


def _run_knn(args, dtype):
    for num_x in args.n:
        for num_y in args.m:
            for dim in args.d:
                for k in args.k:
                    x, y, batch_x, batch_y = _make_inputs(
                        num_x, num_y, dim, dtype, args.batches)
                    cuda_fn = lambda: knn(x, y, k, batch_x, batch_y)
                    triton_fn = lambda: knn__triton(x, y, k, batch_x, batch_y)
                    out_cuda = cuda_fn()
                    out_triton = triton_fn()
                    ok = torch.equal(_sort_edges(out_cuda, num_x),
                                     _sort_edges(out_triton, num_x))
                    cuda = _bench(cuda_fn, args.warmup, args.iters)
                    triton = _bench(triton_fn, args.warmup, args.iters)
                    speedup = cuda['median_ms'] / triton['median_ms']
                    print('knn',
                          f'dtype={dtype}',
                          f'n={num_x}',
                          f'm={num_y}',
                          f'd={dim}',
                          f'k={k}',
                          f'batches={args.batches}',
                          f'ok={ok}',
                          f'cuda_ms={cuda["median_ms"]:.4f}',
                          f'triton_ms={triton["median_ms"]:.4f}',
                          f'speedup={speedup:.3f}',
                          flush=True)


def _run_radius(args, dtype):
    for num_x in args.n:
        for num_y in args.m:
            for dim in args.d:
                for radius_value in args.radius:
                    for max_num_neighbors in args.max_num_neighbors:
                        x, y, batch_x, batch_y = _make_inputs(
                            num_x, num_y, dim, dtype, args.batches)
                        cuda_fn = lambda: radius(
                            x,
                            y,
                            radius_value,
                            batch_x,
                            batch_y,
                            max_num_neighbors=max_num_neighbors,
                        )
                        triton_fn = lambda: radius__triton(
                            x,
                            y,
                            radius_value,
                            batch_x,
                            batch_y,
                            max_num_neighbors=max_num_neighbors,
                        )
                        out_cuda = cuda_fn()
                        out_triton = triton_fn()
                        ok = torch.equal(_sort_edges(out_cuda, num_x),
                                         _sort_edges(out_triton, num_x))
                        cuda = _bench(cuda_fn, args.warmup, args.iters)
                        triton = _bench(triton_fn, args.warmup, args.iters)
                        speedup = cuda['median_ms'] / triton['median_ms']
                        print('radius',
                              f'dtype={dtype}',
                              f'n={num_x}',
                              f'm={num_y}',
                              f'd={dim}',
                              f'r={radius_value}',
                              f'max_neighbors={max_num_neighbors}',
                              f'batches={args.batches}',
                              f'ok={ok}',
                              f'cuda_ms={cuda["median_ms"]:.4f}',
                              f'triton_ms={triton["median_ms"]:.4f}',
                              f'speedup={speedup:.3f}',
                              flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--op', choices=['knn', 'radius'], required=True)
    parser.add_argument('--n', type=_parse_ints, default='1024,4096')
    parser.add_argument('--m', type=_parse_ints, default='1024,4096')
    parser.add_argument('--d', type=_parse_ints, default='3,16')
    parser.add_argument('--k', type=_parse_ints, default='8,32')
    parser.add_argument('--radius', type=_parse_floats, default='0.5,2.0')
    parser.add_argument('--max-num-neighbors',
                        type=_parse_ints,
                        default='16,64')
    parser.add_argument('--dtype',
                        choices=['float16', 'float32', 'float64', 'bfloat16'],
                        default='float32')
    parser.add_argument('--batches', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iters', type=int, default=30)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required')
    if importlib.util.find_spec('triton') is None:
        raise SystemExit('Triton is required')

    torch.manual_seed(12345)
    dtype = getattr(torch, args.dtype)
    if args.op == 'knn':
        _run_knn(args, dtype)
    else:
        _run_radius(args, dtype)


if __name__ == '__main__':
    main()
