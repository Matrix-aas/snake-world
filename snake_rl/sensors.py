"""Sensory observation: vision raycasting + smell field -> a fixed 42-float vector."""
import numpy as np
from .world import torus_delta, segment_circle_hit

OBS_DIM = 42


def ray_dirs(cfg, heading):
    half = np.radians(cfg.fov_deg) / 2
    angles = heading + np.linspace(-half, half, cfg.n_rays)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def _all_targets(world):
    """Returns (centers, radii, kind) where kind: 0=obstacle,1=chicken,2=self."""
    c = world.cfg
    parts = []
    if len(world.obstacle_pos):
        parts.append((world.obstacle_pos, world.obstacle_r, np.zeros(len(world.obstacle_pos), int)))
    if len(world.chicken_pos):
        parts.append((world.chicken_pos, np.full(len(world.chicken_pos), c.chicken_radius),
                      np.ones(len(world.chicken_pos), int)))
    body = world.body_points()
    if len(body):
        parts.append((body, np.full(len(body), c.body_radius), np.full(len(body), 2)))
    if not parts:
        return np.zeros((0, 2)), np.zeros((0,)), np.zeros((0,), int)
    cen = np.vstack([p[0] for p in parts])
    rad = np.concatenate([p[1] for p in parts])
    kind = np.concatenate([p[2] for p in parts])
    return cen, rad, kind


def _scan(world, head, heading):
    """Vectorized raycast of all rays at once. Returns (dirs (R,2), dist (R,), kind (R,))
    with kind 0=obstacle,1=chicken,2=self,-1=none. Same result as a per-ray loop, faster."""
    c = world.cfg
    cen, rad, kind = _all_targets(world)
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


def sense_vision(world):
    c = world.cfg
    _, dist, kinds = _scan(world, world.head, world.heading)
    out = np.tile([1.0, 0.0, 0.0, 0.0], (c.n_rays, 1))
    hit = kinds >= 0
    out[hit, 0] = dist[hit] / c.ray_range
    out[hit, 1 + kinds[hit]] = 1.0
    return out


def vision_distances(world, head, heading):
    """For rendering: ray directions, hit distance, and hit kind, from an arbitrary (interpolated) pose."""
    return _scan(world, head, heading)


def smell(world):
    c = world.cfg
    if len(world.chicken_pos) == 0:
        return np.zeros(3, np.float32)
    rel = torus_delta(world.chicken_pos, world.head, world.size)     # head->chicken
    dist = np.linalg.norm(rel, axis=1)
    intensity = 0.0
    grad = np.zeros(2)
    for i in range(len(world.chicken_pos)):
        if dist[i] < 1e-6:
            continue
        if len(world.obstacle_pos):                                 # line-of-sight occlusion
            blocked = segment_circle_hit(world.head, world.chicken_pos[i],
                                         world.obstacle_pos, world.obstacle_r, world.size).any()
            if blocked:
                continue
        r = dist[i]
        intensity += 1.0 / (1.0 + r)
        grad += (1.0 / (1.0 + r) ** 2) * (rel[i] / r)               # points toward chicken
    fwd = world.heading_vec()
    left = np.array([-fwd[1], fwd[0]])
    return np.array([intensity, grad @ fwd, grad @ left], np.float32)


def observe(world):
    c = world.cfg
    vision = sense_vision(world).astype(np.float32).flatten()        # 36
    sm = smell(world)                                                # 3
    proprio = np.array([world.energy / c.energy_max,
                        world.target_length / c.length_cap,
                        world.stamina / c.s_max], np.float32)        # 3
    return np.concatenate([vision, sm, proprio]).astype(np.float32)
