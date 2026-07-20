import numpy as np
from snake_rl.world import wrap, torus_delta, torus_dist, ray_circle_hit, segment_circle_hit

SIZE = np.array([60.0, 60.0])


def test_wrap():
    np.testing.assert_allclose(wrap(np.array([61.0, -1.0]), SIZE), [1.0, 59.0])


def test_delta_takes_nearest_image_across_seam():
    a = np.array([1.0, 1.0]); b = np.array([59.0, 59.0])
    d = torus_delta(a, b, SIZE)               # should be +2 across the seam, not -58
    np.testing.assert_allclose(d, [2.0, 2.0])
    assert abs(torus_dist(a, b, SIZE) - np.hypot(2, 2)) < 1e-9


def test_ray_hits_object_across_seam():
    # head near right edge, obstacle just across the left seam, ray pointing +x
    origin = np.array([59.0, 30.0]); u = np.array([1.0, 0.0])
    centers = np.array([[1.0, 30.0]]); radii = np.array([0.5])
    t = ray_circle_hit(origin, u, centers, radii, max_t=20.0, size=SIZE)
    assert abs(t[0] - 1.5) < 1e-6          # 2 units to center - 0.5 radius


def test_ray_misses_returns_inf():
    origin = np.array([30.0, 30.0]); u = np.array([1.0, 0.0])
    centers = np.array([[30.0, 50.0]]); radii = np.array([0.5])
    t = ray_circle_hit(origin, u, centers, radii, max_t=20.0, size=SIZE)
    assert np.isinf(t[0])


def test_swept_segment_catches_tunneling():
    # fast head jumps from x=58 to x=2 across seam through obstacle at x=0
    p0 = np.array([58.0, 30.0]); p1 = np.array([62.0, 30.0])  # p1 unwrapped just past seam
    centers = np.array([[0.0, 30.0]]); radii = np.array([0.5])
    hit = segment_circle_hit(p0, p1, centers, radii, SIZE)
    assert hit[0]
