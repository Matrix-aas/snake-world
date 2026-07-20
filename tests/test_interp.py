import numpy as np
from snake_rl.watch import _interp_body, _interp_chickens


def test_interp_body_same_length():
    prev = np.array([[0., 0.], [1., 0.]]); cur = np.array([[2., 0.], [3., 0.]])
    np.testing.assert_allclose(_interp_body(prev, cur, 0.5), [[1., 0.], [2., 0.]])


def test_interp_body_grows_no_stutter():
    prev = np.zeros((3, 2)); cur = np.ones((5, 2))
    out = _interp_body(prev, cur, 0.5)
    assert out.shape == (5, 2)
    np.testing.assert_allclose(out[:3], 0.5)      # head-side prefix blends smoothly
    np.testing.assert_allclose(out[3:], 1.0)      # new tail points snap in far from the head


def test_interp_chickens_crosses_seam():
    size = np.array([100., 100.])
    prev = {7: (np.array([99., 50.]), 0.0)}
    cur = {7: (np.array([1., 50.]), 0.0)}         # walked across the seam
    pos, _ = _interp_chickens(prev, cur, 0.5, size)
    assert min(pos[0][0], 100 - pos[0][0]) < 1e-6  # midpoint is the seam (0/100), not x=50


def test_interp_chickens_spawn_and_eat():
    size = np.array([80., 80.])
    prev = {1: (np.array([10., 10.]), 0.0)}
    cur = {2: (np.array([20., 20.]), 0.0)}        # id 1 eaten, id 2 spawned
    pos, _ = _interp_chickens(prev, cur, 0.5, size)
    assert len(pos) == 1                           # only current chickens are drawn
    np.testing.assert_allclose(pos[0], [20., 20.]) # a freshly spawned chicken snaps to its spot
