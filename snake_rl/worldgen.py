"""Random world generation with rejection sampling."""
import numpy as np
from .world import World, Snake, wrap, torus_dist


def generate_world(cfg, seed=None, size=None, n_snakes=1, arrivals=False):
    """`arrivals=True` (viewer / training world): non-ego snakes ARRIVE via a guaranteed egg that
    hatches a few steps in, and runtime chickens DROP FROM THE SKY (world.chicken_sky), rather than
    popping in (Goals 1 & 2). The ego (snake 0) is ALWAYS a live snake from step 0 -- SnakeEnv drives
    it and can't steer an inert egg. `arrivals=False` (default) keeps the plain instant spawn for unit
    fixtures. Initial chickens land instantly either way, so episode-start food is available at once."""
    w = World(cfg, seed=seed, size=size)                 # fixed size (e.g. screen-fit) or random
    w.chicken_sky = arrivals
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
    if n_snakes > 1:
        placed = []
        for i in range(n_snakes):
            for _ in range(200):
                p = w._free_point(cfg.head_radius)
                if not placed or (torus_dist(np.array(placed), p, w.size) > 2 * cfg.r_flee).all():
                    placed.append(p); break
            else:
                placed.append(w._free_point(cfg.head_radius))
        if arrivals:
            # ego is live from step 0 (SnakeEnv drives it); every OTHER snake ARRIVES via a
            # guaranteed egg (staggered hatch so they don't all pop at once), ids assigned at hatch.
            p0 = placed[0]
            w.snakes = [Snake(head_uw=p0.copy(), head=wrap(p0, w.size),
                              heading=float(rng.uniform(0, 2 * np.pi)), path_uw=[p0.copy()],
                              target_length=cfg.start_length, stamina=cfg.s_max, energy=cfg.energy_max,
                              _prev_head_uw=p0.copy(), id=0, color_seed=0)]
            w._next_snake_id = 1
            for p in placed[1:]:
                w.spawn_egg(p, timer=int(rng.integers(cfg.egg_timer // 2, cfg.egg_timer + 1)))
        else:
            w.snakes = [Snake(head_uw=p.copy(), head=wrap(p, w.size),
                              heading=float(rng.uniform(0, 2 * np.pi)), path_uw=[p.copy()],
                              target_length=cfg.start_length, stamina=cfg.s_max, energy=cfg.energy_max,
                              _prev_head_uw=p.copy(), id=i, color_seed=i) for i, p in enumerate(placed)]
            w._next_snake_id = n_snakes
    # initial chickens land instantly (arriving=False) so there's food at episode start regardless of
    # the sky-drop presentation, which applies to RUNTIME spawns (population ceiling, not the target rate)
    k = int(rng.integers(1, cfg.chicken_ceiling + 1))
    for _ in range(k):
        w.maybe_spawn_forced(arriving=False)
    return w
