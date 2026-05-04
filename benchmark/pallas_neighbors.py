import argparse
import os
import sys
import time
from itertools import product

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_cluster_pallas import knn, pairwise_distance, radius


def _parse_ints(value):
    return [int(item) for item in value.split(',') if item]


def _parse_floats(value):
    return [float(item) for item in value.split(',') if item]


def _block_until_ready(value):
    if isinstance(value, tuple):
        for item in value:
            if hasattr(item, 'block_until_ready'):
                item.block_until_ready()
    elif hasattr(value, 'block_until_ready'):
        value.block_until_ready()


def _time_call(fn, warmup, iters):
    start = time.perf_counter()
    out = fn()
    _block_until_ready(out)
    compile_ms = (time.perf_counter() - start) * 1000.0

    for _ in range(warmup):
        _block_until_ready(fn())

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        _block_until_ready(fn())
        times.append((time.perf_counter() - start) * 1000.0)

    times = jnp.asarray(times)
    return compile_ms, float(jnp.median(times)), float(jnp.percentile(times, 90))


def _same_output(lhs, rhs):
    if isinstance(lhs, tuple):
        return all(_same_output(left, right) for left, right in zip(lhs, rhs))
    if lhs.dtype in (jnp.float16, jnp.float32, jnp.bfloat16, jnp.float64):
        return jnp.allclose(lhs, rhs, rtol=0.0, atol=1e-4)
    return jnp.array_equal(lhs, rhs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--op',
                        choices=['pairwise', 'knn', 'radius'],
                        default='pairwise')
    parser.add_argument('--n', default='128,512')
    parser.add_argument('--m', default='128,512')
    parser.add_argument('--d', default='3,16')
    parser.add_argument('--k', default='8')
    parser.add_argument('--radius', default='1.0')
    parser.add_argument('--interpret',
                        action='store_true',
                        help='Run Pallas in interpret mode. This is required '
                        'for CPU-only smoke tests.')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--iters', type=int, default=10)
    args = parser.parse_args()

    key = jax.random.PRNGKey(12345)
    devices = jax.devices()
    interpret = args.interpret or not any(device.platform == 'tpu'
                                          for device in devices)
    print('jax_devices=' + ','.join(str(device) for device in devices))
    print(f'pallas_interpret={interpret}')

    for n, m, d in product(_parse_ints(args.n), _parse_ints(args.m),
                           _parse_ints(args.d)):
        key, x_key, y_key = jax.random.split(key, 3)
        x = jax.random.normal(x_key, (n, d), dtype=jnp.float32)
        y = jax.random.normal(y_key, (m, d), dtype=jnp.float32)

        if args.op == 'pairwise':
            fn_jax = jax.jit(lambda: pairwise_distance(x, y, backend='jax'))
            fn_pallas = jax.jit(lambda: pairwise_distance(
                x, y, backend='pallas', interpret=interpret))
            cj, mj, p90j = _time_call(fn_jax, args.warmup, args.iters)
            cp, mp, p90p = _time_call(fn_pallas, args.warmup, args.iters)
            ok = jnp.allclose(fn_jax(), fn_pallas(), rtol=0.0, atol=1e-4)
            print(f'pairwise n={n} m={m} d={d} ok={bool(ok)} '
                  f'jax_compile_ms={cj:.4f} pallas_compile_ms={cp:.4f} '
                  f'jax_ms={mj:.4f} pallas_ms={mp:.4f} '
                  f'jax_p90_ms={p90j:.4f} pallas_p90_ms={p90p:.4f} '
                  f'speedup={mj / mp:.3f}')

        elif args.op == 'knn':
            for k in _parse_ints(args.k):
                fn_jax = jax.jit(lambda k=k: knn(
                    x, y, k, backend='jax', compact=False))
                fn_pallas = jax.jit(lambda k=k: knn(
                    x,
                    y,
                    k,
                    backend='pallas',
                    interpret=interpret,
                    compact=False))
                cj, mj, p90j = _time_call(fn_jax, args.warmup, args.iters)
                cp, mp, p90p = _time_call(fn_pallas, args.warmup, args.iters)
                ok = _same_output(fn_jax(), fn_pallas())
                print(f'knn n={n} m={m} d={d} k={k} ok={bool(ok)} '
                      f'jax_compile_ms={cj:.4f} pallas_compile_ms={cp:.4f} '
                      f'jax_ms={mj:.4f} pallas_ms={mp:.4f} '
                      f'jax_p90_ms={p90j:.4f} pallas_p90_ms={p90p:.4f} '
                      f'speedup={mj / mp:.3f}')

        else:
            for r in _parse_floats(args.radius):
                fn_jax = jax.jit(lambda r=r: radius(
                    x, y, r, backend='jax', compact=False))
                fn_pallas = jax.jit(lambda r=r: radius(
                    x,
                    y,
                    r,
                    backend='pallas',
                    interpret=interpret,
                    compact=False))
                cj, mj, p90j = _time_call(fn_jax, args.warmup, args.iters)
                cp, mp, p90p = _time_call(fn_pallas, args.warmup, args.iters)
                ok = _same_output(fn_jax(), fn_pallas())
                print(f'radius n={n} m={m} d={d} r={r} ok={bool(ok)} '
                      f'jax_compile_ms={cj:.4f} pallas_compile_ms={cp:.4f} '
                      f'jax_ms={mj:.4f} pallas_ms={mp:.4f} '
                      f'jax_p90_ms={p90j:.4f} pallas_p90_ms={p90p:.4f} '
                      f'speedup={mj / mp:.3f}')


if __name__ == '__main__':
    main()
