import numpy as np
from snake_rl.config import CFG
from snake_rl.worldgen import generate_world
from snake_rl.world import torus_dist


def test_generate_is_deterministic_by_seed():
    a = generate_world(CFG, seed=42); b = generate_world(CFG, seed=42)
    np.testing.assert_allclose(a.size, b.size)
    np.testing.assert_allclose(a.obstacle_pos, b.obstacle_pos)


def test_has_chicken_and_obstacles_in_range():
    w = generate_world(CFG, seed=1)
    assert len(w.chicken_pos) >= 1
    assert CFG.n_obstacles_min <= len(w.obstacle_pos) <= CFG.n_obstacles_max
    assert (CFG.world_size_min <= w.size).all() and (w.size <= CFG.world_size_max).all()


def test_no_obstacle_on_snake_start():
    w = generate_world(CFG, seed=3)
    d = torus_dist(w.obstacle_pos, w.head, w.size)
    assert (d > w.obstacle_r + CFG.head_radius).all()
