"""Random world generation with rejection sampling."""
import numpy as np
from .world import World, wrap, torus_dist
from .genome import sample_genome


def generate_world(cfg, seed=None, size=None, n_snakes=1, arrivals=False, ego_live=True):
    """`arrivals=True` (viewer / training world): non-ego snakes ARRIVE via a guaranteed egg that
    hatches a few steps in, and runtime chickens DROP FROM THE SKY (world.chicken_sky), rather than
    popping in (Goals 1 & 2). `arrivals=False` (default) keeps the plain instant spawn for unit
    fixtures. Initial chickens land instantly either way, so episode-start food is available at once.

    `ego_live` (only meaningful with `arrivals=True`, needs `n_snakes > 1`): whether slot 0 is a LIVE
    privileged gradient-ego. `True` (training / SnakeEnv): snake 0 is live from step 0 -- SB3 drives
    it and can't steer an inert egg -- and every OTHER founder arrives as an egg. `False` (viewer):
    NO snake is special -- ZERO live snakes at step 0, all `n_snakes` founders arrive as eggs, and the
    world runs with `world.no_ego=True`. Snakes appear only by hatching."""
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
        if arrivals and not ego_live:
            # Viewer all-eggs world: NO snake is special. Zero live snakes; ALL founders arrive as
            # guaranteed eggs (staggered hatch). world.no_ego makes `step`/`_prune_dead` ego-free.
            w.snakes = []
            w._next_snake_id = 0
            w.no_ego = True
            for p in placed:
                w.spawn_egg(p, timer=int(rng.integers(cfg.egg_timer // 2, cfg.egg_timer + 1)))
        elif arrivals:
            # ego is live from step 0 (SnakeEnv drives it); every OTHER snake ARRIVES via a
            # guaranteed egg (staggered hatch so they don't all pop at once), ids assigned at hatch.
            p0 = placed[0]
            w.snakes = [w._make_snake(wrap(p0, w.size), float(rng.uniform(0, 2 * np.pi)),
                                      genome=sample_genome(rng), sex=int(rng.integers(0, 2)), lineage=0,
                                      id=0, color_seed=0, energy=cfg.energy_max,
                                      target_length=cfg.start_length, rng=rng)]
            w._next_snake_id = 1
            for p in placed[1:]:
                w.spawn_egg(p, timer=int(rng.integers(cfg.egg_timer // 2, cfg.egg_timer + 1)))
        else:
            w.snakes = [w._make_snake(wrap(p, w.size), float(rng.uniform(0, 2 * np.pi)),
                                      genome=sample_genome(rng), sex=int(rng.integers(0, 2)), lineage=i,
                                      id=i, color_seed=i, energy=cfg.energy_max,
                                      target_length=cfg.start_length, rng=rng)
                        for i, p in enumerate(placed)]
            w._next_snake_id = n_snakes
    # initial chickens land instantly (arriving=False) so there's food at episode start regardless of
    # the sky-drop presentation, which applies to RUNTIME spawns (population ceiling, not the target rate)
    k = int(rng.integers(1, cfg.chicken_ceiling + 1))
    for _ in range(k):
        w.maybe_spawn_forced(arriving=False)
    return w
