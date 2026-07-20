"""Sensory observation: vision raycasting + smell field -> a fixed 42-float vector."""
import numpy as np
from .world import torus_delta, ray_circle_hit, segment_circle_hit

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


def sense_vision(world):
    c = world.cfg
    cen, rad, kind = _all_targets(world)
    dirs = ray_dirs(c, world.heading)
    out = np.tile([1.0, 0.0, 0.0, 0.0], (c.n_rays, 1))
    if len(cen) == 0:
        return out
    for i, u in enumerate(dirs):
        t = ray_circle_hit(world.head, u, cen, rad, c.ray_range, world.size)
        j = int(np.argmin(t))
        if np.isfinite(t[j]):
            out[i, 0] = t[j] / c.ray_range
            out[i, 1 + kind[j]] = 1.0
    return out


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
