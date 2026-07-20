"""Random world generation with rejection sampling."""
import numpy as np
from .world import World, torus_dist


def generate_world(cfg, seed=None, size=None):
    w = World(cfg, seed=seed, size=size)                 # fixed size (e.g. screen-fit) or random
    rng = w.rng
    # scale obstacle count with area when an explicit size is given, to keep density ~constant
    ref = ((cfg.world_size_min + cfg.world_size_max) / 2) ** 2
    area_mult = (w.size[0] * w.size[1]) / ref if size is not None else 1.0
    n = int(np.clip(round(rng.integers(cfg.n_obstacles_min, cfg.n_obstacles_max + 1) * area_mult),
                    cfg.n_obstacles_min, 60))
    pos, rad, kind = [], [], []
    clear = cfg.r_flee                                   # keep start area open
    attempts = 0
    while len(pos) < n and attempts < 4000:              # cap: never wedge if constants get dense
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
