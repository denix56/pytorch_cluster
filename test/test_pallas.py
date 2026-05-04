from types import SimpleNamespace

import pytest

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')

from torch_cluster_pallas import (  # noqa
    grid_cluster,
    knn,
    knn_graph,
    nearest,
    pairwise_distance,
    radius,
    radius_graph,
)

pytestmark = pytest.mark.pallas
HAS_TPU = any(device.platform == 'tpu' for device in jax.devices())
requires_tpu = pytest.mark.skipif(not HAS_TPU, reason='requires TPU hardware')


def _to_set(edge_index):
    return set(map(tuple, edge_index.T.tolist()))


def test_pallas_pairwise_distance_reference():
    x = jnp.array([[0.0], [2.0], [3.0]])
    y = jnp.array([[1.0], [4.0]])

    out = pairwise_distance(x, y, backend='jax')
    assert jnp.array_equal(out, jnp.array([[1.0, 1.0, 4.0],
                                          [16.0, 4.0, 1.0]]))


def test_pallas_pairwise_distance_auto_without_tpu_uses_jax(monkeypatch):
    monkeypatch.setattr(jax, 'devices',
                        lambda: [SimpleNamespace(platform='cpu')])

    x = jnp.array([[0.0, 1.0], [2.0, 4.0]])
    y = jnp.array([[1.0, 3.0]])

    out = pairwise_distance(x, y, backend='auto')
    assert jnp.array_equal(out, jnp.array([[5.0, 2.0]]))


def test_pallas_pairwise_distance_auto_with_tpu_uses_interpret_mode(
        monkeypatch):
    monkeypatch.setattr(jax, 'devices',
                        lambda: [SimpleNamespace(platform='tpu')])

    x = jnp.array([[0.0], [2.0], [4.0]])
    y = jnp.array([[1.0], [3.0]])

    out = pairwise_distance(x,
                            y,
                            backend='auto',
                            block_m=1,
                            block_n=2,
                            interpret=True)
    assert jnp.array_equal(out, jnp.array([[1.0, 1.0, 9.0],
                                          [9.0, 1.0, 1.0]]))


def test_pallas_pairwise_distance_rejects_unknown_backend():
    with pytest.raises(ValueError, match='backend'):
        pairwise_distance(jnp.array([[0.0]]),
                          jnp.array([[1.0]]),
                          backend='unknown')


def test_pallas_knn_reference():
    x = jnp.array([[0.0], [2.0], [3.0]])
    y = jnp.array([[1.0], [4.0]])

    edge_index, _ = knn(x, y, k=2, backend='jax')
    assert _to_set(edge_index) == {(0, 0), (0, 1), (1, 2), (1, 1)}


def test_pallas_knn_batched_compact_false_marks_invalid_neighbors():
    x = jnp.array([[0.0], [10.0], [12.0]])
    y = jnp.array([[0.5], [10.5], [100.0]])
    batch_x = jnp.array([0, 2, 2])
    batch_y = jnp.array([0, 2, 0])

    edge_index, valid = knn(x,
                            y,
                            k=2,
                            batch_x=batch_x,
                            batch_y=batch_y,
                            backend='jax',
                            compact=False)

    assert edge_index.tolist() == [[0, 0, 1, 1, 2, 2],
                                   [0, -1, 1, 2, 0, -1]]
    assert valid.tolist() == [[True, False], [True, True], [True, False]]


def test_pallas_knn_compact_false_is_jittable():
    x = jnp.array([[0.0], [2.0], [3.0]])
    y = jnp.array([[1.0], [4.0]])

    edge_index, valid = jax.jit(lambda: knn(
        x, y, k=2, backend='jax', compact=False))()
    assert edge_index.tolist() == [[0, 0, 1, 1], [0, 1, 2, 1]]
    assert valid.tolist() == [[True, True], [True, True]]


def test_pallas_knn_graph_flow_directions():
    x = jnp.array([[0.0], [2.0], [10.0]])

    edge_index, _ = knn_graph(x, k=1, backend='jax')
    assert edge_index.tolist() == [[1, 0, 1], [0, 1, 2]]

    edge_index, _ = knn_graph(x,
                              k=1,
                              flow='target_to_source',
                              backend='jax')
    assert edge_index.tolist() == [[0, 1, 2], [1, 0, 1]]


def test_pallas_radius_reference_uses_index_order_cap():
    x = jnp.array([[1.9], [0.1], [0.2]])
    y = jnp.array([[0.0]])

    edge_index, _ = radius(x,
                           y,
                           r=2.0,
                           max_num_neighbors=2,
                           backend='jax')
    assert edge_index.tolist() == [[0, 0], [0, 1]]


def test_pallas_radius_batched_ignore_same_index_compact_false():
    x = jnp.array([[0.0], [0.0], [3.0]])
    y = jnp.array([[0.0], [0.0], [3.0]])
    batch = jnp.array([0, 0, 1])

    edge_index, valid = radius(x,
                               y,
                               r=0.5,
                               batch_x=batch,
                               batch_y=batch,
                               max_num_neighbors=2,
                               ignore_same_index=True,
                               backend='jax',
                               compact=False)

    assert edge_index.tolist() == [[0, 0, 1, 1, 2, 2],
                                   [1, -1, 0, -1, -1, -1]]
    assert valid.tolist() == [[True, False], [True, False], [False, False]]


def test_pallas_radius_compact_false_is_jittable():
    x = jnp.array([[0.0], [0.25], [2.0]])
    y = jnp.array([[0.0], [2.0]])

    edge_index, valid = jax.jit(lambda: radius(
        x, y, r=0.5, max_num_neighbors=2, backend='jax', compact=False))()
    assert edge_index.tolist() == [[0, 0, 1, 1], [0, 1, 2, -1]]
    assert valid.tolist() == [[True, True], [True, False]]


def test_pallas_radius_graph_loop_and_flow_direction():
    x = jnp.array([[0.0], [0.25], [2.0]])

    edge_index, _ = radius_graph(x,
                                 r=0.5,
                                 loop=False,
                                 max_num_neighbors=2,
                                 backend='jax')
    assert edge_index.tolist() == [[1, 0], [0, 1]]

    edge_index, _ = radius_graph(x,
                                 r=0.5,
                                 loop=True,
                                 max_num_neighbors=2,
                                 flow='target_to_source',
                                 backend='jax')
    assert edge_index.tolist() == [[0, 0, 1, 1, 2], [0, 1, 0, 1, 2]]


def test_pallas_nearest_reference():
    x = jnp.array([[0.0], [10.0]])
    y = jnp.array([[1.0], [9.0], [11.0]])

    out = nearest(x, y, backend='jax')
    assert out.tolist() == [0, 1]


def test_pallas_grid_cluster_reference():
    pos = jnp.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    size = jnp.array([2.0, 2.0])

    out = grid_cluster(pos, size)
    assert out.tolist() == [0, 2, 1, 3]


def test_pallas_pairwise_distance_interpret_mode():
    x = jnp.array([[0.0], [2.0]])
    y = jnp.array([[1.0]])

    out = pairwise_distance(x,
                            y,
                            backend='pallas',
                            block_m=1,
                            block_n=2,
                            interpret=True)
    assert jnp.array_equal(out, jnp.array([[1.0, 1.0]]))


def test_pallas_pairwise_distance_default_block_handles_large_n_interpret():
    x = jnp.arange(130, dtype=jnp.float32).reshape((130, 1))
    y = jnp.array([[0.0], [129.0]])

    out = pairwise_distance(x, y, backend='pallas', interpret=True)
    expected = pairwise_distance(x, y, backend='jax')
    assert jnp.array_equal(out, expected)


def test_pallas_knn_empty_inputs():
    x = jnp.empty((0, 2), dtype=jnp.float32)
    y = jnp.array([[0.0, 1.0]], dtype=jnp.float32)

    edge_index, _ = knn(x, y, k=3, backend='jax')
    assert edge_index.shape == (2, 0)

    edge_index, valid = knn(x, y, k=3, backend='jax', compact=False)
    assert edge_index.shape == (2, 0)
    assert valid.shape == (1, 0)


def test_pallas_radius_empty_inputs():
    x = jnp.empty((0, 2), dtype=jnp.float32)
    y = jnp.array([[0.0, 1.0]], dtype=jnp.float32)

    edge_index, _ = radius(x, y, r=1.0, backend='jax')
    assert edge_index.shape == (2, 0)

    edge_index, valid = radius(x, y, r=1.0, backend='jax', compact=False)
    assert edge_index.shape == (2, 0)
    assert valid.shape == (1, 0)


def test_pallas_nearest_empty_x_and_empty_y_behavior():
    x = jnp.empty((0, 2), dtype=jnp.float32)
    y = jnp.array([[0.0, 1.0]], dtype=jnp.float32)

    out = nearest(x, y, backend='jax')
    assert out.shape == (0, )

    with pytest.raises(ValueError, match='at least one point'):
        nearest(y, x, backend='jax')


@pytest.mark.accelerator
@pytest.mark.tpu
@requires_tpu
def test_pallas_pairwise_distance_tpu_kernel_matches_jax():
    x = jnp.arange(130, dtype=jnp.float32).reshape((130, 1))
    y = jnp.array([[0.0], [129.0]])

    out = pairwise_distance(x, y, backend='pallas')
    expected = pairwise_distance(x, y, backend='jax')
    assert jnp.array_equal(out, expected)
