"""Sensory observation: vision raycasting + smell field -> a fixed 113-float per-snake vector.

Layout (see env._make_observation_space for the matching bounds):
  vision   0:88   RAY_COUNT(=11) x [dist, is_obstacle, is_chicken, is_self, is_other_body,
                  is_egg, is_corpse, obstacle_clearance]   (all /ray_range or one-hot -> [0,1])
  social  88:95   nearest live rival, egocentric
  egg     95:99   nearest EATABLE (owner>=0) egg, egocentric
  smell   99:108  chicken / snake / corpse intensity+gradient fields
  proprio 108:113 [energy, length, stamina, repro_ready, speed]
"""
import numpy as np
from .world import torus_delta, torus_dist

OBS_DIM = 113


def ray_dirs(cfg, heading):
    """9 uniform rays over ±fov/2 PLUS n_fwd_rays forward at ± half the 9-ray spacing (16.875° for
    fov=270/9). RAY_COUNT = n_rays + n_fwd_rays = 11. Forward offset derives from the 9-ray
    spacing, never a mutated n_rays."""
    half = np.radians(cfg.fov_deg) / 2
    angles = heading + np.linspace(-half, half, cfg.n_rays)
    if cfg.n_fwd_rays:
        fwd_off = np.radians(cfg.fov_deg / (cfg.n_rays - 1)) / 2      # half the uniform 9-ray spacing
        angles = np.concatenate([angles, heading + np.array([-fwd_off, fwd_off])])
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def _all_targets(world, snake):
    """Returns (centers, radii, kind) where kind: 0=obstacle,1=chicken,2=self,3=other_body,4=egg,5=corpse."""
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
    corpses = world.corpses["pos"]
    if len(corpses):                                              # eaten at eat_radius = head_radius+chicken_radius
        parts.append((corpses, np.full(len(corpses), c.chicken_radius), np.full(len(corpses), 5)))
    if not parts:
        return np.zeros((0, 2)), np.zeros((0,)), np.zeros((0,), int)
    cen = np.vstack([p[0] for p in parts])
    rad = np.concatenate([p[1] for p in parts])
    kind = np.concatenate([p[2] for p in parts])
    return cen, rad, kind


def _cast(dirs, head, size, cen, rad, ray_range, n_obs=0):
    """Nearest-hit distance per ray against circles (cen, rad already Minkowski-inflated).
    Returns (dist (R,), idx (R,), obs (R,)): dist=ray_range & idx=-1 on a ray that hits nothing.
    `obs` = nearest hit among the FIRST n_obs columns (obstacles, the un-mask channel) computed from
    the SAME cast, so vision needs only one pass instead of a second obstacle-only scan."""
    dist = np.full(len(dirs), ray_range, float)
    idx = np.full(len(dirs), -1, int)
    obs = np.full(len(dirs), ray_range, float)
    if len(cen):
        m = torus_delta(cen, head, size)                 # (K,2) head->center (nearest image)
        tca = dirs @ m.T                                 # (R,K) projection of each ray
        d2 = (m * m).sum(1)[None, :] - tca ** 2
        r2 = (rad ** 2)[None, :]
        thc = np.sqrt(np.clip(r2 - d2, 0.0, None))
        t0 = tca - thc; t1 = tca + thc
        t = np.where(t0 >= 0, t0, t1)                    # origin inside a circle -> exit point
        valid = (d2 <= r2) & (t >= 0) & (t <= ray_range)
        tt = np.where(valid, t, np.inf)                  # (R,K)
        j = np.argmin(tt, axis=1)
        best = tt[np.arange(len(dirs)), j]
        got = np.isfinite(best)
        dist[got] = best[got]
        idx[got] = j[got]
        if n_obs:                                        # nearest OBSTACLE (cols 0..n_obs) from same tt
            bo = tt[:, :n_obs].min(axis=1)
            go = np.isfinite(bo)
            obs[go] = bo[go]
    return dist, idx, obs


def _scan(world, head, heading, snake=None):
    """Vectorized raycast of all rays at once. Returns (dirs (R,2), dist (R,), kind (R,), clear (R,))
    with kind 0=obstacle,1=chicken,2=self,3=other_body,4=egg,5=corpse,-1=none, and `clear` = the
    per-ray nearest-OBSTACLE distance (the un-mask channel) from the SAME cast -- no second pass.
    `snake` (default ego) decides which body is "self" vs "other" -- independent of `head`/`heading`
    (an interpolated render pose need not match snake's exact stored pose)."""
    snake = snake if snake is not None else world.snakes[0]
    c = world.cfg
    cen, rad, kind = _all_targets(world, snake)
    rad = rad + c.head_radius                            # inflate by head radius (Minkowski): rays report
    dirs = ray_dirs(c, heading)                          # distance until the head EDGE touches -> the snake
    n_obs = len(world.obstacle_pos)                      # obstacles are cols 0..n_obs of _all_targets
    dist, idx, clear = _cast(dirs, head, world.size, cen, rad, c.ray_range, n_obs)  # perceives own width
    kinds = np.full(len(dirs), -1, int)
    got = idx >= 0
    kinds[got] = kind[idx[got]]
    return dirs, dist, kinds, clear


def sense_vision(world, snake=None):
    c = world.cfg
    snake = snake if snake is not None else world.snakes[0]
    dirs, dist, kinds, clear = _scan(world, snake.head, snake.heading, snake)
    out = np.tile([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], (len(dirs), 1))   # 8 features/ray
    hit = kinds >= 0
    out[hit, 0] = dist[hit] / c.ray_range
    out[hit, 1 + kinds[hit]] = 1.0
    out[:, 7] = clear / c.ray_range                      # un-mask: nearest obstacle on this bearing
    return out


def vision_distances(world, head, heading, snake=None):
    """For rendering: ray directions, hit distance, and hit kind, from an arbitrary (interpolated) pose."""
    return _scan(world, head, heading, snake)[:3]


def _smell_field(world, head, positions):
    """Sum of occluded 1/(1+r) intensity + gradient (pointing toward each position) from head.
    Vectorized: per-position rel/dist, then a batched head->position vs all-obstacles occlusion test
    that mirrors segment_circle_hit's arithmetic op-for-op (so the occlusion booleans are bit-identical),
    then a masked sum over unoccluded positions (numpy sums <128 terms left-to-right = the old loop)."""
    if len(positions) == 0:
        return 0.0, np.zeros(2)
    rel = torus_delta(positions, head, world.size)
    dist = np.sqrt((rel * rel).sum(1))
    keep = dist >= 1e-6                                              # skip a target sitting on the head
    if len(world.obstacle_pos) and keep.any():
        m = torus_delta(world.obstacle_pos, head, world.size)       # (O,2) head->obstacle (shared p0)
        seg_len2 = (rel * rel).sum(1)                               # (P,) == seg @ seg per position
        safe = np.where(keep, seg_len2, 1.0)                        # avoid 0/0 on the skipped positions
        t = np.clip((rel @ m.T) / safe[:, None], 0.0, 1.0)          # (P,O) clamp to the segment
        proj = m[None] - t[..., None] * rel[:, None, :]            # (P,O,2) obstacle offset from segment
        closest2 = (proj * proj).sum(2)                            # (P,O)
        blocked = (closest2 <= (world.obstacle_r ** 2)[None]).any(1)
        keep &= ~blocked
    if not keep.any():
        return 0.0, np.zeros(2)
    r = dist[keep]
    relv = rel[keep]
    intensity = float((1.0 / (1.0 + r)).sum())
    grad = ((1.0 / (1.0 + r) ** 2)[:, None] * (relv / r[:, None])).sum(0)
    return intensity, grad


def smell(world, snake=None):
    c = world.cfg
    snake = snake if snake is not None else world.snakes[0]
    fwd = snake.heading_vec()
    left = np.array([-fwd[1], fwd[0]])
    ci, cg = _smell_field(world, snake.head, world.chicken_pos)
    rival_heads = np.array([o.head for o in world.snakes if o is not snake and o.alive])
    si, sg = _smell_field(world, snake.head, rival_heads)
    ki, kg = _smell_field(world, snake.head, world.corpses["pos"])
    # ponytail: chicken/rival counts are structurally capped elsewhere (chicken_ceiling, n_max), so
    # their raw intensity is already provably within the obs bound. Corpses have no such cap (they
    # persist until eaten) -- clip to chicken_ceiling so observation_space.contains always holds
    # regardless of how many uneaten corpses pile up.
    ceil = float(c.chicken_ceiling)
    ki = float(np.clip(ki, 0.0, ceil))
    kf = float(np.clip(kg @ fwd, -ceil, ceil))
    kl = float(np.clip(kg @ left, -ceil, ceil))
    return np.array([ci, cg @ fwd, cg @ left, si, sg @ fwd, sg @ left, ki, kf, kl], np.float32)


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
    eatable = e["owner"][:, 0] >= 0 if len(e["pos"]) else np.zeros(0, bool)   # arrival eggs (owner -1) uneatable
    if not eatable.any():
        return np.zeros(4, np.float32)
    pos, owner = e["pos"][eatable], e["owner"][eatable]
    d = torus_dist(pos, snake.head, world.size)
    i = int(np.argmin(d))
    fwd = snake.heading_vec()
    left = np.array([-fwd[1], fwd[0]])
    rel = torus_delta(pos[i], snake.head, world.size)
    rel_fwd = float(np.clip((rel @ fwd) / c.ray_range, -1.0, 1.0))
    rel_left = float(np.clip((rel @ left) / c.ray_range, -1.0, 1.0))
    is_mine = 1.0 if snake.id in owner[i] else 0.0
    return np.array([1.0, rel_fwd, rel_left, is_mine], np.float32)


def _repro_ready(cfg, snake):
    return 1.0 if (snake.energy > cfg.repro_energy_frac * cfg.energy_max and
                    snake.target_length > cfg.repro_length_min and
                    snake.repro_cooldown == 0) else 0.0


def observe(world, snake=None):
    c = world.cfg
    snake = snake if snake is not None else world.snakes[0]
    vision = sense_vision(world, snake).astype(np.float32).flatten()      # 88 (11 rays x 8)
    social = _social(world, snake)                                        # 7
    egg = _egg_channel(world, snake)                                      # 4
    sm = smell(world, snake)                                              # 9
    proprio = np.array([
        np.clip(snake.energy / c.energy_max, 0.0, 1.0),
        np.clip(snake.target_length / c.length_cap, 0.0, 1.0),
        np.clip(snake.stamina / c.s_max, 0.0, 1.0),
        _repro_ready(c, snake),
        np.clip(snake.speed / c.v_dash, 0.0, 1.0),                       # last-move speed (prey-sense mirror)
    ], np.float32)                                                        # 5
    return np.concatenate([vision, social, egg, sm, proprio]).astype(np.float32)
