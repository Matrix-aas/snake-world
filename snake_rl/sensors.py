"""Sensory observation: vision raycasting + smell field -> a fixed 75-float per-snake vector."""
import numpy as np
from .world import torus_delta, torus_dist, segment_circle_hit

OBS_DIM = 75


def ray_dirs(cfg, heading):
    half = np.radians(cfg.fov_deg) / 2
    angles = heading + np.linspace(-half, half, cfg.n_rays)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def _all_targets(world, snake):
    """Returns (centers, radii, kind) where kind: 0=obstacle,1=chicken,2=self,3=other_body,4=egg."""
    c = world.cfg
    parts = []
    if len(world.obstacle_pos):
        parts.append((world.obstacle_pos, world.obstacle_r, np.zeros(len(world.obstacle_pos), int)))
    if len(world.chicken_pos):
        parts.append((world.chicken_pos, np.full(len(world.chicken_pos), c.chicken_radius),
                      np.ones(len(world.chicken_pos), int)))
    body = world._body_points(snake)
    if len(body):
        parts.append((body, np.full(len(body), c.body_radius), np.full(len(body), 2)))
    opts, orads = world._other_hazard(snake)                     # rival heads + full bodies (no neck-skip)
    if len(opts):
        parts.append((opts, orads, np.full(len(opts), 3)))
    eggs = world.eggs["pos"]
    if len(eggs):
        parts.append((eggs, np.full(len(eggs), c.egg_radius), np.full(len(eggs), 4)))
    if not parts:
        return np.zeros((0, 2)), np.zeros((0,)), np.zeros((0,), int)
    cen = np.vstack([p[0] for p in parts])
    rad = np.concatenate([p[1] for p in parts])
    kind = np.concatenate([p[2] for p in parts])
    return cen, rad, kind


def _scan(world, head, heading, snake=None):
    """Vectorized raycast of all rays at once. Returns (dirs (R,2), dist (R,), kind (R,))
    with kind 0=obstacle,1=chicken,2=self,3=other_body,4=egg,-1=none. `snake` (default ego)
    decides which body is "self" vs "other" -- independent of `head`/`heading` (an interpolated
    render pose need not match snake's exact stored pose)."""
    snake = snake if snake is not None else world.snakes[0]
    c = world.cfg
    cen, rad, kind = _all_targets(world, snake)
    rad = rad + c.head_radius                            # inflate by head radius (Minkowski): rays report
    dirs = ray_dirs(c, heading)                          # distance until the head EDGE touches -> the snake
    dist = np.full(c.n_rays, c.ray_range, float)         # perceives its own width, not just a center point
    kinds = np.full(c.n_rays, -1, int)
    if len(cen):
        m = torus_delta(cen, head, world.size)           # (K,2) head->center (nearest image)
        tca = dirs @ m.T                                 # (R,K) projection of each ray
        d2 = (m * m).sum(1)[None, :] - tca ** 2
        r2 = (rad ** 2)[None, :]
        thc = np.sqrt(np.clip(r2 - d2, 0.0, None))
        t0 = tca - thc; t1 = tca + thc
        t = np.where(t0 >= 0, t0, t1)                    # origin inside a circle -> exit point
        valid = (d2 <= r2) & (t >= 0) & (t <= c.ray_range)
        tt = np.where(valid, t, np.inf)                  # (R,K)
        j = np.argmin(tt, axis=1)
        best = tt[np.arange(c.n_rays), j]
        got = np.isfinite(best)
        dist[got] = best[got]
        kinds[got] = kind[j[got]]
    return dirs, dist, kinds


def sense_vision(world, snake=None):
    c = world.cfg
    snake = snake if snake is not None else world.snakes[0]
    _, dist, kinds = _scan(world, snake.head, snake.heading, snake)
    out = np.tile([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], (c.n_rays, 1))
    hit = kinds >= 0
    out[hit, 0] = dist[hit] / c.ray_range
    out[hit, 1 + kinds[hit]] = 1.0
    return out


def vision_distances(world, head, heading, snake=None):
    """For rendering: ray directions, hit distance, and hit kind, from an arbitrary (interpolated) pose."""
    return _scan(world, head, heading, snake)


def _smell_field(world, head, positions):
    """Sum of occluded 1/(1+r) intensity + gradient (pointing toward each position) from head."""
    if len(positions) == 0:
        return 0.0, np.zeros(2)
    rel = torus_delta(positions, head, world.size)
    dist = np.linalg.norm(rel, axis=1)
    intensity = 0.0
    grad = np.zeros(2)
    for i in range(len(positions)):
        if dist[i] < 1e-6:
            continue
        if len(world.obstacle_pos):                                 # line-of-sight occlusion
            blocked = segment_circle_hit(head, positions[i], world.obstacle_pos, world.obstacle_r,
                                         world.size).any()
            if blocked:
                continue
        r = dist[i]
        intensity += 1.0 / (1.0 + r)
        grad += (1.0 / (1.0 + r) ** 2) * (rel[i] / r)               # points toward the target
    return intensity, grad


def smell(world, snake=None):
    snake = snake if snake is not None else world.snakes[0]
    fwd = snake.heading_vec()
    left = np.array([-fwd[1], fwd[0]])
    ci, cg = _smell_field(world, snake.head, world.chicken_pos)
    rival_heads = np.array([o.head for o in world.snakes if o is not snake and o.alive])
    si, sg = _smell_field(world, snake.head, rival_heads)
    return np.array([ci, cg @ fwd, cg @ left, si, sg @ fwd, sg @ left], np.float32)


def _social(world, snake):
    c = world.cfg
    rivals = [o for o in world.snakes if o is not snake and o.alive]
    if not rivals:
        return np.zeros(7, np.float32)
    d = torus_dist(np.array([o.head for o in rivals]), snake.head, world.size)
    r = rivals[int(np.argmin(d))]
    fwd = snake.heading_vec()
    left = np.array([-fwd[1], fwd[0]])
    rel = torus_delta(r.head, snake.head, world.size)
    rel_fwd = float(np.clip((rel @ fwd) / c.ray_range, -1.0, 1.0))
    rel_left = float(np.clip((rel @ left) / c.ray_range, -1.0, 1.0))
    rh = r.heading_vec()
    rh_fwd = float(np.clip(rh @ fwd, -1.0, 1.0))
    rh_left = float(np.clip(rh @ left, -1.0, 1.0))
    size_ratio = float(np.clip(r.target_length / c.length_cap, 0.0, 1.0))
    return np.array([1.0, rel_fwd, rel_left, rh_fwd, rh_left, size_ratio, float(r.dashed)], np.float32)


def _egg_channel(world, snake):
    c = world.cfg
    e = world.eggs
    if not len(e["pos"]):
        return np.zeros(4, np.float32)
    d = torus_dist(e["pos"], snake.head, world.size)
    i = int(np.argmin(d))
    fwd = snake.heading_vec()
    left = np.array([-fwd[1], fwd[0]])
    rel = torus_delta(e["pos"][i], snake.head, world.size)
    rel_fwd = float(np.clip((rel @ fwd) / c.ray_range, -1.0, 1.0))
    rel_left = float(np.clip((rel @ left) / c.ray_range, -1.0, 1.0))
    is_mine = 1.0 if snake.id in e["owner"][i] else 0.0
    return np.array([1.0, rel_fwd, rel_left, is_mine], np.float32)


def _repro_ready(cfg, snake):
    return 1.0 if (snake.energy > cfg.repro_energy_frac * cfg.energy_max and
                    snake.target_length > cfg.repro_length_min and
                    snake.repro_cooldown == 0) else 0.0


def observe(world, snake=None):
    c = world.cfg
    snake = snake if snake is not None else world.snakes[0]
    vision = sense_vision(world, snake).astype(np.float32).flatten()      # 54
    social = _social(world, snake)                                        # 7
    egg = _egg_channel(world, snake)                                      # 4
    sm = smell(world, snake)                                              # 6
    proprio = np.array([
        np.clip(snake.energy / c.energy_max, 0.0, 1.0),
        np.clip(snake.target_length / c.length_cap, 0.0, 1.0),
        np.clip(snake.stamina / c.s_max, 0.0, 1.0),
        _repro_ready(c, snake),
    ], np.float32)                                                        # 4
    return np.concatenate([vision, social, egg, sm, proprio]).astype(np.float32)
