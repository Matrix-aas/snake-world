"""Sensory observation: vision raycasting + smell/vibration fields -> a fixed 143-float per-snake vector.

Layout (see env._make_observation_space for the matching bounds):
  vision    0:99   RAY_COUNT(=11) x [dist, is_obstacle, is_chicken, is_self, is_other_body,
                   is_egg, is_corpse, obstacle_clearance, target_motion]  (all /norm or one-hot -> [0,1])
  social   99:110  nearest live rival, egocentric: [has_rival, rel_pos_fwd, rel_pos_left,
                   rival_heading_fwd, rival_heading_left, size_ratio, rival_is_dashing,
                   relatedness, rival_energy, rival_repro_ready, rival_sex]
  egg     110:114  nearest EATABLE (owner>=0) egg, egocentric [has_egg, rel_fwd, rel_left, is_mine]
  smell   114:123  chicken / snake / corpse intensity+gradient fields
  vibration 123:126 omni motion field over rivals + fleeing chickens (UN-occluded)
  proprio 126:143  [energy, length, stamina, repro_ready, speed, sex, age_frac, stun_frac, genome x9]
"""
import numpy as np
from .world import torus_delta, torus_dist
from .genome import relatedness, GENE_COUNT

OBS_DIM = 143


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
    """Returns (centers, radii, kind, motion) where kind: 0=obstacle,1=chicken,2=self,3=other_body,
    4=egg,5=corpse. `motion` is a PARALLEL per-row array (review I3): the target's STATE-NOMINAL speed
    (chicken -> {peck:0, walk:v_wander, flee/startle:v_flee}; rival head/body -> that rival's speed;
    obstacle/self/egg/corpse -> 0). Threaded through _cast so per-ray motion survives the flatten in
    _other_hazard (where idx alone can't recover which rival a body point belongs to)."""
    c = world.cfg
    parts = []                                                   # each: (centers, radii, kind, motion)
    if len(world.obstacle_pos):
        n = len(world.obstacle_pos)
        parts.append((world.obstacle_pos, world.obstacle_r, np.zeros(n, int), np.zeros(n)))
    if len(world.chicken_pos):
        n = len(world.chicken_pos)
        nominal = np.array([0.0, c.v_wander, c.v_flee])[world.chicken_state]   # state-nominal, not instantaneous
        parts.append((world.chicken_pos, np.full(n, c.chicken_radius), np.ones(n, int), nominal))
    body = world._body_points(snake)
    if len(body):
        parts.append((body, np.full(len(body), c.body_radius), np.full(len(body), 2), np.zeros(len(body))))
    opts, orads, omot = world._other_hazard(snake)               # rival heads + full bodies (no neck-skip) + speeds
    if len(opts):
        parts.append((opts, orads, np.full(len(opts), 3), omot))
    eggs = world.eggs["pos"]
    if len(eggs):
        parts.append((eggs, np.full(len(eggs), c.egg_radius), np.full(len(eggs), 4), np.zeros(len(eggs))))
    corpses = world.corpses["pos"]
    if len(corpses):                                              # eaten at eat_radius = head_radius+chicken_radius
        parts.append((corpses, np.full(len(corpses), c.chicken_radius), np.full(len(corpses), 5), np.zeros(len(corpses))))
    if not parts:
        return np.zeros((0, 2)), np.zeros((0,)), np.zeros((0,), int), np.zeros((0,))
    cen = np.vstack([p[0] for p in parts])
    rad = np.concatenate([p[1] for p in parts])
    kind = np.concatenate([p[2] for p in parts])
    motion = np.concatenate([p[3] for p in parts])
    return cen, rad, kind, motion


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
    """Vectorized raycast of all rays at once. Returns (dirs (R,2), dist (R,), kind (R,), clear (R,),
    motion (R,)) with kind 0=obstacle,1=chicken,2=self,3=other_body,4=egg,5=corpse,-1=none, `clear` =
    the per-ray nearest-OBSTACLE distance (the un-mask channel) from the SAME cast, and `motion` = the
    hit target's state-nominal speed (0 on a no-hit ray). `snake` (default ego) decides which body is
    "self" vs "other" -- independent of `head`/`heading` (an interpolated render pose need not match)."""
    snake = snake if snake is not None else world.snakes[0]
    c = world.cfg
    cen, rad, kind, motion = _all_targets(world, snake)
    rad = rad + c.head_radius                            # inflate by head radius (Minkowski): rays report
    dirs = ray_dirs(c, heading)                          # distance until the head EDGE touches -> the snake
    n_obs = len(world.obstacle_pos)                      # obstacles are cols 0..n_obs of _all_targets
    ray_range = snake.phenotype.ray_range                # per-snake sight (genome SENSES gene)
    dist, idx, clear = _cast(dirs, head, world.size, cen, rad, ray_range, n_obs)  # perceives own width
    kinds = np.full(len(dirs), -1, int)
    mot = np.zeros(len(dirs))
    got = idx >= 0
    kinds[got] = kind[idx[got]]
    mot[got] = motion[idx[got]]                          # per-ray motion of the winning target (like kind)
    return dirs, dist, kinds, clear, mot


def sense_vision(world, snake=None):
    c = world.cfg
    snake = snake if snake is not None else world.snakes[0]
    ray_range = snake.phenotype.ray_range                # per-snake sight (genome SENSES gene)
    dirs, dist, kinds, clear, mot = _scan(world, snake.head, snake.heading, snake)
    out = np.tile([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], (len(dirs), 1))   # 9 features/ray
    hit = kinds >= 0
    out[hit, 0] = dist[hit] / ray_range
    out[hit, 1 + kinds[hit]] = 1.0
    out[:, 7] = clear / ray_range                         # un-mask: nearest obstacle on this bearing
    # feature 8: hit target's motion, /(v_dash*gene_speed_hi) = observer-independent max speed (a max-
    # speed-gene dash or a fleeing hen both read high, a pecking hen 0); 0 on no-hit rays (tile default).
    out[hit, 8] = np.clip(mot[hit] / (c.v_dash * c.gene_speed_hi), 0.0, 1.0)
    return out


def vision_distances(world, head, heading, snake=None):
    """For rendering: ray directions, hit distance, and hit kind, from an arbitrary (interpolated) pose."""
    return _scan(world, head, heading, snake)[:3]


def _smell_field(world, head, positions, weights=None, occlude=True):
    """Sum of 1/(1+r) intensity + gradient (pointing toward each position) from head. `occlude=True`
    (smell) drops positions behind an obstacle via a batched head->position vs all-obstacles test that
    mirrors segment_circle_hit op-for-op (bit-identical booleans); `occlude=False` (vibration -- motion
    is FELT through cover) skips it. `weights` (default all-ones) multiplies each term's intensity AND
    gradient -- vibration weights by normalized speed. weights=None is bit-identical to the old smell."""
    if len(positions) == 0:
        return 0.0, np.zeros(2)
    rel = torus_delta(positions, head, world.size)
    dist = np.sqrt((rel * rel).sum(1))
    keep = dist >= 1e-6                                              # skip a target sitting on the head
    if occlude and len(world.obstacle_pos) and keep.any():
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
    w = (np.ones(len(positions)) if weights is None else np.asarray(weights, float))[keep]
    # ponytail: numpy sums <128 terms left-to-right (== the old per-target loop, bit-for-bit); above
    # 128 it switches to pairwise summation -> at most a ULP (~1e-16) shift, still 7 orders under the
    # 1e-9 parity gate and imperceptible to the policy. Chickens (chicken_ceiling) and rivals (n_max)
    # are structurally capped far below 128; only an unbounded corpse pile could ever reach it.
    intensity = float((w / (1.0 + r)).sum())
    grad = ((w / (1.0 + r) ** 2)[:, None] * (relv / r[:, None])).sum(0)
    return intensity, grad


def smell(world, snake=None):
    c = world.cfg
    snake = snake if snake is not None else world.snakes[0]
    reach = snake.phenotype.smell_reach                  # per-snake smell strength (genome SENSES gene)
    fwd = snake.heading_vec()
    left = np.array([-fwd[1], fwd[0]])
    ci, cg = _smell_field(world, snake.head, world.chicken_pos)
    rival_heads = np.array([o.head for o in world.snakes if o is not snake and o.alive])
    si, sg = _smell_field(world, snake.head, rival_heads)
    ki, kg = _smell_field(world, snake.head, world.corpses["pos"])
    # ponytail: chicken/rival counts are structurally capped elsewhere (chicken_ceiling, n_max), so
    # their raw intensity was previously provably within the obs bound with no clip needed. A per-snake
    # smell_reach up to 1.4x (genome SENSES gene) can now push a scaled intensity past that structural
    # cap, so every field is clipped the same way corpses already were (corpses have no population cap
    # at all -- persist until eaten -- so they needed it regardless of reach).
    ceil = float(c.chicken_ceiling)
    ci = float(np.clip(ci * reach, 0.0, ceil))
    cf = float(np.clip((cg @ fwd) * reach, -ceil, ceil))
    cl = float(np.clip((cg @ left) * reach, -ceil, ceil))
    si = float(np.clip(si * reach, 0.0, ceil))
    sf = float(np.clip((sg @ fwd) * reach, -ceil, ceil))
    sl = float(np.clip((sg @ left) * reach, -ceil, ceil))
    ki = float(np.clip(ki * reach, 0.0, ceil))
    kf = float(np.clip((kg @ fwd) * reach, -ceil, ceil))
    kl = float(np.clip((kg @ left) * reach, -ceil, ceil))
    return np.array([ci, cf, cl, si, sf, sl, ki, kf, kl], np.float32)


def sense_vibration(world, snake):
    """Omni motion 'vibration' field (3): live rivals + FLEEING chickens, each weighted by normalized
    speed, felt UN-occluded (through cover -- its point). Returns [intensity, grad_fwd, grad_left],
    clipped to +/-(n_max + chicken_ceiling) DERIVED from cfg (never hardcoded). A stopped rival (speed
    0) and a calm/pecking hen contribute nothing -> the field only lights up for real motion nearby."""
    c = world.cfg
    fwd = snake.heading_vec()
    left = np.array([-fwd[1], fwd[0]])
    norm = c.v_dash * c.gene_speed_hi                    # same observer-independent max speed as vision feat-8
    positions, weights = [], []
    for o in world.snakes:
        if o is snake or not o.alive:
            continue
        positions.append(o.head); weights.append(o.speed / norm)
    if len(world.chicken_pos):
        fleeing = world.chicken_state == 2               # a fleeing hen vibrates at its state-nominal v_flee
        for p in world.chicken_pos[fleeing]:
            positions.append(p); weights.append(c.v_flee / norm)
    if not positions:
        return np.array([0.0, 0.0, 0.0], np.float32)
    intensity, grad = _smell_field(world, snake.head, np.array(positions),
                                   weights=np.array(weights), occlude=False)
    ceil = float(c.n_max + c.chicken_ceiling)
    return np.array([float(np.clip(intensity, 0.0, ceil)),
                     float(np.clip(grad @ fwd, -ceil, ceil)),
                     float(np.clip(grad @ left, -ceil, ceil))], np.float32)


def _social(world, snake):
    c = world.cfg
    rivals = [o for o in world.snakes if o is not snake and o.alive]
    if not rivals:
        return np.zeros(11, np.float32)
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
    # + relatedness (genome kinship -> guard/raid & cooperation), rival energy, rival repro_ready, rival sex
    rel_kin = relatedness(snake.genome, r.genome)
    r_energy = float(np.clip(r.energy / c.energy_max, 0.0, 1.0))
    r_ready = _repro_ready(c, r)
    return np.array([1.0, rel_fwd, rel_left, rh_fwd, rh_left, size_ratio, float(r.dashed),
                     rel_kin, r_energy, r_ready, float(r.sex)], np.float32)


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
    # length gate is size-RELATIVE: a fraction (cfg.repro_length_frac, curriculum-swept) of this
    # snake's OWN max_length -> a small genome qualifies at its own full length, not an absolute bar.
    # Reads the SAME cfg field the real mating gate does (world._resolve_mating), so the observed
    # repro_ready bit and actual eligibility stay in lockstep through the curriculum (review I1).
    return 1.0 if (snake.energy > cfg.repro_energy_frac * cfg.energy_max and
                    snake.target_length > cfg.repro_length_frac * snake.phenotype.max_length and
                    snake.repro_cooldown == 0) else 0.0


def observe(world, snake=None):
    c = world.cfg
    snake = snake if snake is not None else world.snakes[0]
    ph = snake.phenotype                                                  # PER-SNAKE normalizers (genome)
    vision = sense_vision(world, snake).astype(np.float32).flatten()      # 99 (11 rays x 9)
    social = _social(world, snake)                                        # 11
    egg = _egg_channel(world, snake)                                      # 4
    sm = smell(world, snake)                                              # 9
    vib = sense_vibration(world, snake)                                   # 3
    proprio = np.concatenate([
        np.array([
            np.clip(snake.energy / c.energy_max, 0.0, 1.0),
            np.clip(snake.target_length / ph.max_length, 0.0, 1.0),       # own max_length (size gene)
            np.clip(snake.stamina / ph.s_max, 0.0, 1.0),                  # own s_max (stamina gene)
            _repro_ready(c, snake),
            np.clip(snake.speed / ph.v_dash, 0.0, 1.0),                   # own v_dash (last-move speed)
            float(snake.sex),
            float(np.clip(snake.age / snake.max_lifespan, 0.0, 1.0)),     # age fraction of own lifespan
            float(np.clip(snake.stun / c.stun_steps, 0.0, 1.0)),         # dizzy fraction
        ], np.float32),
        snake.genome.astype(np.float32),                                 # own genome tail (9)
    ])                                                                    # 17
    return np.concatenate([vision, social, egg, sm, vib, proprio]).astype(np.float32)
