"""Random world generation with rejection sampling."""
import numpy as np
from .world import World, torus_dist


def generate_world(cfg, seed=None):
    w = World(cfg, seed=seed)                         # random size + centered snake
    rng = w.rng
    n = int(rng.integers(cfg.n_obstacles_min, cfg.n_obstacles_max + 1))
    pos, rad, kind = [], [], []
    clear = cfg.r_flee                                # keep start area open
    attempts = 0
    while len(pos) < n and attempts < 2000:           # cap: never wedge if constants get dense
        attempts += 1
        p = rng.uniform([0, 0], w.size)
        r = rng.uniform(cfg.obstacle_radius_min, cfg.obstacle_radius_max)
        if torus_dist(np.array([w.head]), p, w.size)[0] < r + clear:
            continue
        if pos and (torus_dist(np.array(pos), p, w.size) < np.array(rad) + r + 1.0).any():
            continue
        pos.append(p); rad.append(r); kind.append(int(rng.integers(0, 2)))
    w.obstacle_pos = np.array(pos, float).reshape(-1, 2); w.obstacle_r = np.array(rad, float).reshape(-1)
    w.obstacle_kind = np.array(kind, dtype=int).reshape(-1)
    # initial chickens
    k = int(rng.integers(1, cfg.max_chickens + 1))
    for _ in range(k):
        w.maybe_spawn_forced()
    return w
