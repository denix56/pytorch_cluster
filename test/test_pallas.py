import pytest

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')

from torch_cluster_pallas import (  # noqa
    grid_cluster,
    knn,
    nearest,
    pairwise_distance,
    radius,
)

pytestmark = pytest.mark.pallas


def _to_set(edge_index):
    return set(map(tuple, edge_index.T.tolist()))


def test_pallas_pairwise_distance_reference():
    x = jnp.array([[0.0], [2.0], [3.0]])
    y = jnp.array([[1.0], [4.0]])

    out = pairwise_distance(x, y, backend='jax')
    assert jnp.array_equal(out, jnp.array([[1.0, 1.0, 4.0],
                                          [16.0, 4.0, 1.0]]))


def test_pallas_knn_reference():
    x = jnp.array([[0.0], [2.0], [3.0]])
    y = jnp.array([[1.0], [4.0]])

    edge_index, _ = knn(x, y, k=2, backend='jax')
    assert _to_set(edge_index) == {(0, 0), (0, 1), (1, 2), (1, 1)}


def test_pallas_radius_reference_uses_index_order_cap():
    x = jnp.array([[1.9], [0.1], [0.2]])
    y = jnp.array([[0.0]])

    edge_index, _ = radius(x,
                           y,
                           r=2.0,
                           max_num_neighbors=2,
                           backend='jax')
    assert edge_index.tolist() == [[0, 0], [0, 1]]


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
