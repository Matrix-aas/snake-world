"""Continuous-torus simulation: torus geometry helpers + the World state machine."""
from __future__ import annotations
import numpy as np


# --- torus geometry (nearest-image everywhere) ---

def wrap(p, size):
    return np.mod(p, size)


def torus_delta(a, b, size):
    """Nearest-image vector a-b on a torus of given size (per-axis)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    return (d + size / 2) % size - size / 2


def torus_dist(a, b, size):
    return np.linalg.norm(torus_delta(a, b, size), axis=-1)


def ray_circle_hit(origin, u, centers, radii, max_t, size):
    """First hit distance of ray origin+t*u (t in [0,max_t]) against each circle,
    using each circle's nearest image. inf where no hit. u must be unit length."""
    origin = np.asarray(origin, float); u = np.asarray(u, float)
    centers = np.asarray(centers, float).reshape(-1, 2)
    radii = np.asarray(radii, float).reshape(-1)
    m = torus_delta(centers, origin, size)          # (K,2) head->center (nearest image)
    tca = m @ u                                      # projection onto ray
    d2 = np.einsum("ij,ij->i", m, m) - tca ** 2
    out = np.full(len(centers), np.inf)
    ok = (d2 <= radii ** 2)
    thc = np.sqrt(np.clip(radii ** 2 - d2, 0, None))
    t0 = tca - thc
    t1 = tca + thc
    t = np.where(t0 >= 0, t0, t1)                    # origin inside circle -> use exit point t1
    valid = ok & (t >= 0) & (t <= max_t)
    out[valid] = t[valid]
    return out


def segment_circle_hit(p0, p1, centers, radii, size):
    """Swept test: does segment p0->p1 (nearest-image) pass within radius of each center."""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    centers = np.asarray(centers, float).reshape(-1, 2)
    radii = np.asarray(radii, float).reshape(-1)
    seg = torus_delta(p1, p0, size)                 # displacement of the swept step
    seg_len2 = seg @ seg
    m = torus_delta(centers, p0, size)              # p0->center per circle
    if seg_len2 < 1e-12:
        closest2 = np.einsum("ij,ij->i", m, m)
    else:
        t = np.clip((m @ seg) / seg_len2, 0.0, 1.0)
        proj = m - np.outer(t, seg)
        closest2 = np.einsum("ij,ij->i", proj, proj)
    return closest2 <= radii ** 2


# --- world state machine ---

class World:
    def __init__(self, cfg, seed=None, size=None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        if size is None:
            s = self.rng.uniform(cfg.world_size_min, cfg.world_size_max, size=2)
        else:
            s = np.asarray(size, float)
        self.size = s
        self.head_uw = s / 2.0
        self.head = wrap(self.head_uw, s)
        self.heading = float(self.rng.uniform(0, 2 * np.pi))
        self.path_uw = [self.head_uw.copy()]
        self.target_length = cfg.start_length
        self.stamina = cfg.s_max
        self.energy = cfg.energy_max
        self.alive = True
        self.dashed = False
        self.steps = 0
        self._prev_head_uw = self.head_uw.copy()
        # chickens / obstacles filled by worldgen; default empty
        self.chicken_pos = np.zeros((0, 2)); self.chicken_dir = np.zeros((0,))
        self.chicken_id = np.zeros((0,), dtype=int)      # stable id per chicken
        self._next_chicken_id = 0
        self.obstacle_pos = np.zeros((0, 2)); self.obstacle_r = np.zeros((0,))
        self.obstacle_kind = np.zeros((0,), dtype=int)   # 0=rock,1=tree (render only)

    def heading_vec(self):
        return np.array([np.cos(self.heading), np.sin(self.heading)])

    # --- motion ---
    def move(self, steering, dash):
        c = self.cfg
        if steering == 0:
            self.heading -= np.radians(c.turn_deg)
        elif steering == 2:
            self.heading += np.radians(c.turn_deg)
        self.heading %= 2 * np.pi
        dashing = bool(dash) and self.stamina >= c.dash_min_stamina  # reserve gate (curriculum-tunable)
        speed = c.v_dash if dashing else c.v_snake
        prev_uw = self.head_uw.copy()
        self.head_uw = prev_uw + speed * self.heading_vec()
        self.head = wrap(self.head_uw, self.size)
        self.path_uw.append(self.head_uw.copy())
        self._prune_path()
        if dashing:
            self.stamina = max(0.0, self.stamina - c.stamina_drain)
        else:
            self.stamina = min(c.s_max, self.stamina + c.stamina_regen)
        self.steps += 1
        self._prev_head_uw = prev_uw
        self.dashed = dashing
        return dashing

    def _prune_path(self):
        pts = np.array(self.path_uw)
        if len(pts) < 3:
            return
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.cumsum(seg[::-1])[::-1]              # dist from head to each point's forward neighbor
        # slack = max motion step (v_dash), so body_points_uw never truncates its tail after a dash
        keep = np.concatenate([cum <= (self.target_length + self.cfg.v_dash), [True]])
        self.path_uw = [p.copy() for p in pts[keep]]

    def body_points_uw(self):
        c = self.cfg
        pts = np.array(self.path_uw)
        if len(pts) < 2:
            return np.zeros((0, 2))
        # Skip the head-adjacent "neck", then a body point every segment_spacing.
        # The skip must clear the whole swept collision segment [prev_head -> head] (up to v_dash long),
        # not just the head, or a straight snake collides with its own neck. Needs
        # skip - v_dash > body_radius + head_radius; this leaves a comfortable margin.
        skip = c.head_radius + c.body_radius + c.v_dash + c.segment_spacing
        targets = []
        t = skip
        while t <= self.target_length:
            targets.append(t); t += c.segment_spacing
        if not targets:
            return np.zeros((0, 2))
        # walk from head (last vertex) backward, INTERPOLATING to the exact arc position
        out, acc, ti = [], 0.0, 0
        for i in range(len(pts) - 1, 0, -1):
            a = pts[i]; b = pts[i - 1]
            step = float(np.linalg.norm(a - b))
            if step < 1e-12:
                continue
            while ti < len(targets) and acc + step >= targets[ti]:
                frac = (targets[ti] - acc) / step
                out.append(a + (b - a) * frac)
                ti += 1
            acc += step
            if ti >= len(targets):
                break
        return np.array(out) if out else np.zeros((0, 2))

    def body_points(self):
        b = self.body_points_uw()
        return wrap(b, self.size) if len(b) else b

    def body_render_path_uw(self, spacing=None):
        """Dense UNWRAPPED body polyline from the head (index 0) back to target_length.
        For rendering only — NO neck skip, so the drawn body connects to the head (no gap)."""
        c = self.cfg
        spacing = spacing if spacing is not None else max(0.25, c.body_radius * 0.5)
        pts = np.array(self.path_uw)
        if len(pts) < 2:
            return self.head_uw[None].copy()
        targets = np.arange(0.0, self.target_length + 1e-9, spacing)
        out = [pts[-1].copy()]                           # index 0 = head (arc 0)
        ti, acc = 1, 0.0
        for i in range(len(pts) - 1, 0, -1):
            a = pts[i]; b = pts[i - 1]
            step = float(np.linalg.norm(a - b))
            if step < 1e-12:
                continue
            while ti < len(targets) and acc + step >= targets[ti]:
                frac = (targets[ti] - acc) / step
                out.append(a + (b - a) * frac)
                ti += 1
            acc += step
            if ti >= len(targets):
                break
        return np.array(out)

    # --- chickens & energy ---
    def nearest_chicken(self):
        if len(self.chicken_pos) == 0:
            return -1, np.inf
        d = torus_dist(self.chicken_pos, self.head, self.size)
        i = int(np.argmin(d))
        return i, float(d[i])

    def nearest_chicken_id(self):
        """Stable id of the nearest chicken (survives array reindexing on eat/spawn); -1 if none."""
        i, _ = self.nearest_chicken()
        return -1 if i < 0 else int(self.chicken_id[i])

    def _add_chicken(self, p):
        self.chicken_pos = np.vstack([self.chicken_pos, p]) if len(self.chicken_pos) else p[None]
        self.chicken_dir = np.append(self.chicken_dir, self.rng.uniform(0, 2 * np.pi))
        self.chicken_id = np.append(self.chicken_id, self._next_chicken_id)
        self._next_chicken_id += 1

    def set_chickens(self, positions):
        """Place chickens at explicit positions with fresh stable ids (setup/debug helper)."""
        positions = np.asarray(positions, float).reshape(-1, 2)
        self.chicken_pos = positions.copy()
        self.chicken_dir = np.zeros(len(positions))
        self.chicken_id = np.arange(self._next_chicken_id, self._next_chicken_id + len(positions))
        self._next_chicken_id += len(positions)

    def _blocked(self, old, new):
        """True if `new` moves deeper into an obstacle than `old` (so tangential/outward moves pass)."""
        if not len(self.obstacle_pos):
            return False
        nc = float((torus_dist(self.obstacle_pos, new, self.size) - self.obstacle_r).min())
        if nc >= self.cfg.chicken_radius:
            return False
        oc = float((torus_dist(self.obstacle_pos, old, self.size) - self.obstacle_r).min())
        return nc < oc

    def update_chickens(self):
        c = self.cfg
        if len(self.chicken_pos) == 0:
            return
        to_head = torus_delta(self.head, self.chicken_pos, self.size)   # chicken->head
        dist = np.linalg.norm(to_head, axis=1)
        for i in range(len(self.chicken_pos)):
            if dist[i] < c.r_flee and dist[i] > 1e-6:
                base = np.arctan2(-to_head[i][1], -to_head[i][0])       # away from snake
                speed = c.v_flee
            else:
                self.chicken_dir[i] += self.rng.normal(0, 0.3)          # slow wander drift
                base = self.chicken_dir[i]; speed = c.v_wander
            old = self.chicken_pos[i]
            for da in (0.0, 0.5, -0.5, 1.0, -1.0, 1.6, -1.6):           # go straight, else steer around the rock
                a = base + da
                new = wrap(old + speed * np.array([np.cos(a), np.sin(a)]), self.size)
                if not self._blocked(old, new):
                    self.chicken_pos[i] = new
                    self.chicken_dir[i] = a                             # keep heading consistent for wander
                    break

    def try_eat(self):
        if len(self.chicken_pos) == 0:
            return 0
        d = torus_dist(self.chicken_pos, self.head, self.size)
        eaten = d <= self.cfg.eat_radius
        n = int(eaten.sum())
        if n:
            keep = ~eaten
            self.chicken_pos = self.chicken_pos[keep]
            self.chicken_dir = self.chicken_dir[keep]
            self.chicken_id = self.chicken_id[keep]
            self.target_length = min(self.cfg.length_cap,
                                     self.target_length + n * self.cfg.grow_per_chicken)
            self.energy = min(self.cfg.energy_max, self.energy + n * self.cfg.energy_refill)
        return n

    def decay_energy(self):
        self.energy = max(0.0, self.energy - self.cfg.energy_decay)

    def _free_point(self, radius):
        best = None; best_clear = -np.inf
        for _ in range(50):
            p = self.rng.uniform([0, 0], self.size)
            clear = np.inf if not len(self.obstacle_pos) else \
                float((torus_dist(self.obstacle_pos, p, self.size) - self.obstacle_r).min())
            if clear > best_clear:                    # remember the least-bad candidate
                best_clear, best = clear, p
            if clear < radius:
                continue
            if torus_dist(self.head, p, self.size) < self.cfg.r_flee:
                continue
            body = self.body_points()
            if len(body) and (torus_dist(body, p, self.size) < radius + self.cfg.body_radius).any():
                continue
            return p
        # exhausted (only in a near-packed world worldgen never makes): furthest-from-any-obstacle point.
        # It may skip the head/body checks, but a chicken spawned next to the snake is harmless — it gets eaten at once.
        return best

    def maybe_spawn(self):
        c = self.cfg
        n = len(self.chicken_pos)
        if n >= c.max_chickens:
            return
        p = 0.06 if n < c.min_chickens else 1.0 / c.spawn_period        # fast refill to min, then random up to max
        if n == 0 or self.rng.random() < p:
            self._add_chicken(self._free_point(c.chicken_radius))

    def maybe_spawn_forced(self):
        self._add_chicken(self._free_point(self.cfg.chicken_radius))

    # --- collisions & full step ---
    def check_death(self):
        # swept head segment [prev_head -> head]; body_points_uw already skips the neck (see body_points_uw)
        p0 = self._prev_head_uw
        p1 = self.head_uw
        hr = self.cfg.head_radius
        if len(self.obstacle_pos):
            hit = segment_circle_hit(wrap(p0, self.size), wrap(p1, self.size),
                                     self.obstacle_pos, self.obstacle_r + hr, self.size)
            if hit.any():
                self.alive = False
                return True
        body = self.body_points_uw()
        if len(body):
            hit = segment_circle_hit(p0, p1, body,
                                     np.full(len(body), self.cfg.body_radius + hr), self.size)
            if hit.any():
                self.alive = False
                return True
        return False

    def step(self, steering, dash):
        dashed = self.move(steering, dash)
        self.update_chickens()
        ate = self.try_eat()
        self.decay_energy()
        self.maybe_spawn()
        died = self.check_death()
        return {"ate": ate, "died": died, "dashed": dashed}
