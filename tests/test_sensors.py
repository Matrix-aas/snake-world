import numpy as np
from snake_rl.config import CFG
from snake_rl.world import World
from snake_rl.sensors import observe, sense_vision, smell, OBS_DIM


def straight_world():
    w = World(CFG, seed=0, size=(60, 60))
    w.head = np.array([30.0, 30.0]); w.head_uw = w.head.copy(); w.heading = 0.0  # facing +x
    return w


def test_obs_shape_and_bounds():
    w = straight_world(); w.maybe_spawn_forced()
    o = observe(w)
    assert o.shape == (OBS_DIM,) and o.dtype == np.float32
    assert np.isfinite(o).all()
    assert (o[:36].reshape(9, 4)[:, 0] >= 0).all() and (o[:36].reshape(9, 4)[:, 0] <= 1).all()


def test_center_ray_sees_obstacle_ahead():
    w = straight_world()
    w.obstacle_pos = np.array([[40.0, 30.0]]); w.obstacle_r = np.array([1.0]); w.obstacle_kind = np.array([0])
    v = sense_vision(w)
    center = v[CFG.n_rays // 2]
    assert center[1] == 1.0                       # obstacle channel
    # distance until the head EDGE touches: 40 - (obstacle_r + head_radius) - 30, normalized
    assert abs(center[0] - (40 - 1 - CFG.head_radius - 30) / CFG.ray_range) < 1e-2


def test_empty_ray_encoding():
    w = straight_world()
    v = sense_vision(w)
    assert (v == np.array([1.0, 0, 0, 0])).all()


def test_smell_stronger_when_closer_and_blocked_by_obstacle():
    w = straight_world()
    w.set_chickens([[35.0, 30.0]])
    near = smell(w)[0]
    w.set_chickens([[45.0, 30.0]])
    far = smell(w)[0]
    assert near > far
    # place an obstacle between head and the chicken -> intensity drops to 0
    w.set_chickens([[45.0, 30.0]])
    w.obstacle_pos = np.array([[38.0, 30.0]]); w.obstacle_r = np.array([2.0]); w.obstacle_kind = np.array([0])
    assert smell(w)[0] == 0.0


def test_smell_gradient_points_forward_to_chicken_ahead():
    w = straight_world()
    w.set_chickens([[40.0, 30.0]])
    g = smell(w)
    assert g[1] > abs(g[2])                       # forward component dominates
