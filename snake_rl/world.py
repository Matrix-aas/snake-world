"""Continuous-torus simulation: torus geometry helpers + the World state machine."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(eq=False)
class Snake:
    head_uw: np.ndarray
    head: np.ndarray
    heading: float
    path_uw: list
    target_length: float
    stamina: float
    energy: float
    _prev_head_uw: np.ndarray
    id: int = 0
    color_seed: int = 0
    alive: bool = True
    dashed: bool = False
    death_cause: object = None
    steps: int = 0
    repro_cooldown: int = 0

    def heading_vec(self):
        return np.array([np.cos(self.heading), np.sin(self.heading)])


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
        self.snakes = [Snake(
            head_uw=s / 2.0, head=wrap(s / 2.0, s), heading=float(self.rng.uniform(0, 2 * np.pi)),
            path_uw=[(s / 2.0).copy()], target_length=cfg.start_length,
            stamina=cfg.s_max, energy=cfg.energy_max, _prev_head_uw=(s / 2.0).copy(),
            id=0, color_seed=0,
        )]
        self._next_snake_id = 1
        # chickens / obstacles filled by worldgen; default empty
        self.chicken_pos = np.zeros((0, 2)); self.chicken_dir = np.zeros((0,))
        self.chicken_id = np.zeros((0,), dtype=int)      # stable id per chicken
        # behavior FSM, parallel to chicken_pos: state 0=peck 1=walk 2=flee, timer = steps left
        # in the current peck/walk, startle = flee steps still fluttering fast. Kept the SAME length
        # as chicken_pos everywhere (created in _add_chicken/set_chickens, filtered in _snake_eat).
        self.chicken_state = np.zeros((0,), dtype=int)
        self.chicken_timer = np.zeros((0,), dtype=int)
        self.chicken_startle = np.zeros((0,), dtype=int)
        self._next_chicken_id = 0
        self.obstacle_pos = np.zeros((0, 2)); self.obstacle_r = np.zeros((0,))
        self.obstacle_kind = np.zeros((0,), dtype=int)   # 0=rock,1=tree (render only)
        self.corpses = {"pos": np.zeros((0, 2)), "food": np.zeros((0,))}
        self.eggs = {"pos": np.zeros((0, 2)), "timer": np.zeros((0,)), "owner": np.zeros((0, 2), int)}
        self._mate_streak = {}
        self.auto_lay_warmup = False                     # set by env.set_hardness; read by B4's _auto_lay fallback

    # --- motion (per-snake workers) ---
    def _move_snake(self, s, steering, dash):
        c = self.cfg
        if steering == 0:
            s.heading -= np.radians(c.turn_deg)
        elif steering == 2:
            s.heading += np.radians(c.turn_deg)
        s.heading %= 2 * np.pi
        dashing = bool(dash) and s.stamina >= c.dash_min_stamina  # reserve gate (curriculum-tunable)
        speed = c.v_dash if dashing else c.v_snake
        prev_uw = s.head_uw.copy()
        s.head_uw = prev_uw + speed * s.heading_vec()
        s.head = wrap(s.head_uw, self.size)
        s.path_uw.append(s.head_uw.copy())
        self._prune_path(s)
        if dashing:
            s.stamina = max(0.0, s.stamina - c.stamina_drain)
        else:
            s.stamina = min(c.s_max, s.stamina + c.stamina_regen)
        s.steps += 1
        s._prev_head_uw = prev_uw
        s.dashed = dashing
        return dashing

    def _prune_path(self, s):
        pts = np.array(s.path_uw)
        if len(pts) < 3:
            return
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.cumsum(seg[::-1])[::-1]              # dist from head to each point's forward neighbor
        # slack = max motion step (v_dash), so body_points_uw never truncates its tail after a dash
        keep = np.concatenate([cum <= (s.target_length + self.cfg.v_dash), [True]])
        s.path_uw = [p.copy() for p in pts[keep]]

    def _body_points_uw(self, s):
        c = self.cfg
        pts = np.array(s.path_uw)
        if len(pts) < 2:
            return np.zeros((0, 2))
        # Skip the head-adjacent "neck", then a body point every segment_spacing.
        # The skip must clear the whole swept collision segment [prev_head -> head] (up to v_dash long),
        # not just the head, or a straight snake collides with its own neck. Needs
        # skip - v_dash > body_radius + head_radius; this leaves a comfortable margin.
        skip = c.head_radius + c.body_radius + c.v_dash + c.segment_spacing
        targets = []
        t = skip
        while t <= s.target_length:
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

    def _body_points(self, s):
        b = self._body_points_uw(s)
        return wrap(b, self.size) if len(b) else b

    def _body_render_path_uw(self, s, spacing=None):
        """Dense UNWRAPPED body polyline from the head (index 0) back to target_length.
        For rendering only — NO neck skip, so the drawn body connects to the head (no gap)."""
        c = self.cfg
        spacing = spacing if spacing is not None else max(0.25, c.body_radius * 0.5)
        pts = np.array(s.path_uw)
        if len(pts) < 2:
            return s.head_uw[None].copy()
        targets = np.arange(0.0, s.target_length + 1e-9, spacing)
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

    def _random_state_timer(self):
        """Fresh chicken starts pecking or wandering with a staggered random timer (rng, reproducible)."""
        c = self.cfg
        if self.rng.random() < 0.5:
            return 0, int(self.rng.integers(c.chicken_peck_min, c.chicken_peck_max + 1))
        return 1, int(self.rng.integers(c.chicken_walk_min, c.chicken_walk_max + 1))

    def _add_chicken(self, p):
        self.chicken_pos = np.vstack([self.chicken_pos, p]) if len(self.chicken_pos) else p[None]
        self.chicken_dir = np.append(self.chicken_dir, self.rng.uniform(0, 2 * np.pi))
        self.chicken_id = np.append(self.chicken_id, self._next_chicken_id)
        st, tm = self._random_state_timer()
        self.chicken_state = np.append(self.chicken_state, st)
        self.chicken_timer = np.append(self.chicken_timer, tm)
        self.chicken_startle = np.append(self.chicken_startle, 0)
        self._next_chicken_id += 1

    def set_chickens(self, positions):
        """Place chickens at explicit positions with fresh stable ids (setup/debug helper)."""
        positions = np.asarray(positions, float).reshape(-1, 2)
        n = len(positions)
        self.chicken_pos = positions.copy()
        self.chicken_dir = np.zeros(n)
        self.chicken_id = np.arange(self._next_chicken_id, self._next_chicken_id + n)
        sts = np.zeros(n, int); tms = np.zeros(n, int)
        for k in range(n):
            sts[k], tms[k] = self._random_state_timer()
        self.chicken_state = sts
        self.chicken_timer = tms
        self.chicken_startle = np.zeros(n, int)
        self._next_chicken_id += n

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
        """Advance each chicken's peck/walk/flee FSM one step (see _chicken_step)."""
        if len(self.chicken_pos) == 0:
            return

        def steer(i, base, speed):
            old = self.chicken_pos[i]
            for da in (0.0, 0.5, -0.5, 1.0, -1.0, 1.6, -1.6):           # go straight, else steer around the rock
                a = base + da
                new = wrap(old + speed * np.array([np.cos(a), np.sin(a)]), self.size)
                if not self._blocked(old, new):
                    self.chicken_pos[i] = new
                    self.chicken_dir[i] = a                             # keep heading consistent for wander
                    break

        # Live heads only (the ego proxy is kept in snakes[0] even when dead, so filter on .alive):
        # a chicken must flee the actual LIVE snake, never a frozen dead-ego ghost.
        heads = np.array([s.head for s in self.snakes if s.alive]) if any(
            s.alive for s in self.snakes) else np.zeros((0, 2))
        for i in range(len(self.chicken_pos)):
            base, speed = self._chicken_step(i, heads)
            if speed > 0.0:
                steer(i, base, speed)                                   # peck (speed 0) just stands still

    def _chicken_step(self, i, heads):
        """FSM transition for chicken i against live snake `heads`; returns (heading, speed).
        Mutates chicken_state/timer/startle in place. Flee overrides the peck/walk timer; when no
        snake is near a fleeing chicken it settles to WALK (not straight back to pecking under a nose)."""
        c = self.cfg
        # --- threat: flee the repulsion resultant of every snake within the alert radius (never bolt
        #     from one snake straight into another; opposing snakes that cancel -> flee the nearest).
        #     A head-down PECKING chicken is distracted: it only notices a snake within the tight
        #     r_flee_peck (the stalk-and-pounce window). WALK / already-fleeing use the full r_flee. ---
        alert = c.r_flee_peck if self.chicken_state[i] == 0 else c.r_flee
        if len(heads):
            to_heads = torus_delta(heads, self.chicken_pos[i], self.size)   # (K,2) chicken->each head
            dist = np.linalg.norm(to_heads, axis=1)
            near = (dist < alert) & (dist > 1e-6)
        else:
            near = np.zeros(0, bool)
        if near.any():
            if self.chicken_state[i] != 2:                                  # startle-FREEZE on ENTERING flee
                self.chicken_state[i] = 2
                self.chicken_startle[i] = c.chicken_startle_steps
            weight = c.r_flee - dist[near]                                  # linear falloff, ->0 at r_flee
            away = -to_heads[near] / dist[near, None]
            repulsion = (weight[:, None] * away).sum(axis=0)
            if np.linalg.norm(repulsion) > 1e-6:
                base = np.arctan2(repulsion[1], repulsion[0])
            else:                                                          # degenerate cancellation ->
                j = int(np.argmin(np.where(near, dist, np.inf)))           # flee the nearest NEAR snake
                base = np.arctan2(-to_heads[j][1], -to_heads[j][0])
            if self.chicken_startle[i] > 0:
                self.chicken_startle[i] -= 1
                return base, 0.0                                            # startle FREEZE: a beat of surprise
            return base, c.v_flee                                          # then bolt away

        # --- safe: a chicken that WAS fleeing resumes wandering (settles to peck later, via timer) ---
        if self.chicken_state[i] == 2:
            self.chicken_state[i] = 1
            self.chicken_timer[i] = int(self.rng.integers(c.chicken_walk_min, c.chicken_walk_max + 1))
            self.chicken_startle[i] = 0
        # --- peck <-> walk on the timer ---
        self.chicken_timer[i] -= 1
        if self.chicken_timer[i] <= 0:
            if self.chicken_state[i] == 0:                                  # peck -> walk
                self.chicken_state[i] = 1
                self.chicken_timer[i] = int(self.rng.integers(c.chicken_walk_min, c.chicken_walk_max + 1))
            else:                                                          # walk -> peck
                self.chicken_state[i] = 0
                self.chicken_timer[i] = int(self.rng.integers(c.chicken_peck_min, c.chicken_peck_max + 1))
        if self.chicken_state[i] == 0:                                      # PECK: prime catch window, no move
            return self.chicken_dir[i], 0.0
        self.chicken_dir[i] += self.rng.normal(0, 0.3)                      # WALK: gentle wander drift
        return self.chicken_dir[i], c.v_wander

    def _spawn_corpse(self, s):
        food = self.cfg.corpse_food_per_length * s.target_length
        pos = s.head[None].copy()
        self.corpses["pos"] = np.vstack([self.corpses["pos"], pos]) if len(self.corpses["pos"]) else pos
        self.corpses["food"] = np.append(self.corpses["food"], food)

    def _snake_eat(self, s):
        """Eat any chicken OR corpse OR foreign egg within eat_radius of s.head (nearest-image);
        each item counts once into n, growth/energy applied to s, egg ownership keyed on s.id."""
        n = 0
        energy_gain = 0.0
        if len(self.chicken_pos):
            d = torus_dist(self.chicken_pos, s.head, self.size)
            eaten = d <= self.cfg.eat_radius
            nc = int(eaten.sum())
            if nc:
                keep = ~eaten
                self.chicken_pos = self.chicken_pos[keep]
                self.chicken_dir = self.chicken_dir[keep]
                self.chicken_id = self.chicken_id[keep]
                self.chicken_state = self.chicken_state[keep]     # keep FSM arrays in lock-step
                self.chicken_timer = self.chicken_timer[keep]
                self.chicken_startle = self.chicken_startle[keep]
                n += nc
                energy_gain += nc * self.cfg.energy_refill
        if len(self.corpses["pos"]):
            d = torus_dist(self.corpses["pos"], s.head, self.size)
            eaten = d <= self.cfg.eat_radius
            nk = int(eaten.sum())
            if nk:
                keep = ~eaten
                energy_gain += float(self.corpses["food"][eaten].sum())
                self.corpses["pos"] = self.corpses["pos"][keep]
                self.corpses["food"] = self.corpses["food"][keep]
                n += nk
        if len(self.eggs["pos"]):
            eater_id = s.id
            owner = self.eggs["owner"]
            foreign = (owner[:, 0] != eater_id) & (owner[:, 1] != eater_id)
            d = torus_dist(self.eggs["pos"], s.head, self.size)
            eaten = foreign & (d <= self.cfg.eat_radius)
            ne = int(eaten.sum())
            if ne:
                keep = ~eaten
                self.eggs["pos"] = self.eggs["pos"][keep]
                self.eggs["timer"] = self.eggs["timer"][keep]
                self.eggs["owner"] = self.eggs["owner"][keep]
                n += ne
                energy_gain += ne * self.cfg.egg_food
        if n:
            s.target_length = min(self.cfg.length_cap,
                                   s.target_length + n * self.cfg.grow_per_chicken)
            s.energy = min(self.cfg.energy_max, s.energy + energy_gain)
        return n

    def try_eat(self):
        """Every live snake eats independently; returns the EGO's count from this single pass
        (never re-runs _snake_eat on the ego, which would double-consume already-cleared arrays)."""
        ate_ego = 0
        for s in self.snakes:
            if not s.alive:
                continue
            n = self._snake_eat(s)
            if s is self.snakes[0]:
                ate_ego = n
        return ate_ego

    def decay_energy(self):
        for s in self.snakes:
            if s.alive:
                s.energy = max(0.0, s.energy - self.cfg.energy_decay)

    def _prune_dead(self):
        """Drop dead non-ego opponents (ego kept in slot 0 even when dead)."""
        self.snakes = [self.snakes[0]] + [s for s in self.snakes[1:] if s.alive]

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
        n_alive = max(1, sum(1 for s in self.snakes if s.alive))
        max_target = int(np.clip(round(c.chickens_per_snake_max * n_alive), 1, c.chicken_ceiling))
        min_target = int(np.clip(round(c.chickens_per_snake_min * n_alive), 1, max_target))
        n = len(self.chicken_pos)
        if n >= max_target:
            return
        p = 0.06 if n < min_target else 1.0 / c.spawn_period   # fast refill to min, then random to max
        if n == 0 or self.rng.random() < p:
            self._add_chicken(self._free_point(c.chicken_radius))

    def maybe_spawn_forced(self):
        self._add_chicken(self._free_point(self.cfg.chicken_radius))

    # --- reproduction ---
    def _lay_egg(self, pos, id_a, id_b):
        e = self.eggs
        e["pos"] = np.vstack([e["pos"], pos[None]]) if len(e["pos"]) else pos[None].copy()
        e["timer"] = np.append(e["timer"], self.cfg.egg_timer)
        row = np.array([[id_a, id_b]])
        e["owner"] = np.vstack([e["owner"], row]) if len(e["owner"]) else row

    def _hatch_eggs(self):
        """Returns owner-sets (frozenset of parent ids) of eggs that produced a hatchling this step."""
        c = self.cfg
        e = self.eggs
        hatched_owners = []
        if not len(e["pos"]):
            return hatched_owners
        e["timer"] = e["timer"] - 1
        hatch = e["timer"] <= 0
        if hatch.any():
            n_alive = sum(1 for s in self.snakes if s.alive)
            for i in np.nonzero(hatch)[0]:
                if n_alive >= c.n_max:
                    continue
                pos = e["pos"][i].copy()
                sid = self._next_snake_id
                self._next_snake_id += 1
                self.snakes.append(Snake(
                    head_uw=pos, head=wrap(pos, self.size), heading=float(self.rng.uniform(0, 2 * np.pi)),
                    path_uw=[pos.copy()], target_length=c.start_length,
                    stamina=c.s_max, energy=c.hatch_energy_frac * c.energy_max, _prev_head_uw=pos.copy(),
                    id=sid, color_seed=sid,
                ))
                n_alive += 1
                hatched_owners.append(frozenset(int(x) for x in e["owner"][i]))
            keep = ~hatch
            e["pos"] = e["pos"][keep]; e["timer"] = e["timer"][keep]; e["owner"] = e["owner"][keep]
        return hatched_owners

    def _resolve_mating(self):
        c = self.cfg
        for s in self.snakes:
            if s.repro_cooldown > 0:
                s.repro_cooldown -= 1
        live = [s for s in self.snakes if s.alive]
        seen = set()
        for i in range(len(live)):
            for j in range(i + 1, len(live)):
                a, b = live[i], live[j]
                key = frozenset((a.id, b.id)); seen.add(key)
                ready = (a.energy > c.repro_energy_frac * c.energy_max and
                         b.energy > c.repro_energy_frac * c.energy_max and
                         a.target_length > c.repro_length_min and b.target_length > c.repro_length_min and
                         a.repro_cooldown == 0 and b.repro_cooldown == 0)
                close = torus_dist(a.head_uw[None], b.head_uw, self.size)[0] <= c.r_mate
                if ready and close:
                    self._mate_streak[key] = self._mate_streak.get(key, 0) + 1
                    if self._mate_streak[key] >= c.mate_steps:
                        mid = wrap(a.head_uw + torus_delta(b.head_uw, a.head_uw, self.size) / 2, self.size)
                        self._lay_egg(mid, a.id, b.id)
                        a.energy -= c.repro_cost; b.energy -= c.repro_cost
                        a.repro_cooldown = c.repro_cooldown; b.repro_cooldown = c.repro_cooldown
                        self._mate_streak[key] = 0
                else:
                    self._mate_streak.pop(key, None)
        for key in list(self._mate_streak):          # forget pairs that no longer both live
            if key not in seen:
                del self._mate_streak[key]

    # --- collisions & full step ---
    def _other_hazard(self, s):
        pts, rads = [], []
        for o in self.snakes:
            if o is s or not o.alive:
                continue
            pts.append(o.head_uw); rads.append(self.cfg.head_radius)   # head (head_radius)
            body = self._body_render_path_uw(o)[1:]    # dense body, NO neck skip; skip idx-0 head (added above)
            if len(body):
                pts.extend(body); rads.extend([self.cfg.body_radius] * len(body))
        if not pts:
            return np.zeros((0, 2)), np.zeros((0,))
        return np.array(pts), np.array(rads)

    def _death_cause(self, s):
        """Pure — returns 'obstacle'|'self'|'snake'|None for s vs post-move state. No mutation."""
        c = self.cfg; hr = c.head_radius
        p0, p1 = s._prev_head_uw, s.head_uw
        if len(self.obstacle_pos) and segment_circle_hit(
                wrap(p0, self.size), wrap(p1, self.size), self.obstacle_pos, self.obstacle_r + hr,
                self.size).any():
            return "obstacle"
        body = self._body_points_uw(s)                 # self set KEEPS the neck-skip
        if len(body) and segment_circle_hit(
                p0, p1, body, np.full(len(body), c.body_radius + hr), self.size).any():
            return "self"
        opts, orads = self._other_hazard(s)
        if len(opts) and segment_circle_hit(
                s._prev_head_uw, s.head_uw, opts, orads + hr, self.size).any():
            return "snake"
        return None

    def _check_death(self, s):                         # wrapper: decide + apply (single-snake / proxy use)
        cause = self._death_cause(s)
        if cause:
            s.alive = False; s.death_cause = cause
            return True
        return False

    # --- ego proxies (temporary Milestone-A bridge; reworked in Milestone B) ---
    _EGO_ATTRS = ("head_uw", "head", "heading", "target_length", "stamina", "energy",
                  "alive", "dashed", "death_cause", "steps", "path_uw", "_prev_head_uw")

    def heading_vec(self):            return self.snakes[0].heading_vec()
    def move(self, steering, dash):   return self._move_snake(self.snakes[0], steering, dash)
    def check_death(self):            return self._check_death(self.snakes[0])
    def body_points_uw(self):         return self._body_points_uw(self.snakes[0])
    def body_points(self):            return self._body_points(self.snakes[0])
    def body_render_path_uw(self, spacing=None):
        return self._body_render_path_uw(self.snakes[0], spacing)

    def step(self, steering, dash, opponent_fn=None):
        opponent_fn = opponent_fn or (lambda world, s: (1, 0))
        # phase 1: EVERY snake (ego + opponents) acts on the PRE-MOVE world, so collect all opponent
        # actions BEFORE moving anyone [M-2], then move ego, then move opponents with those actions.
        ego = self.snakes[0]
        opp_actions = {o.id: opponent_fn(self, o) for o in self.snakes[1:] if o.alive}
        ego_dashed = self._move_snake(ego, steering, dash) if ego.alive else False
        for o in self.snakes[1:]:
            if o.alive:
                st, da = opp_actions[o.id]
                self._move_snake(o, st, da)
        # phase 2: DECIDE all deaths against frozen post-move state, THEN apply (order-independent, C2)
        dying = [(s, cause) for s in self.snakes if s.alive and (cause := self._death_cause(s))]
        for s, cause in dying:
            s.alive = False; s.death_cause = cause
            self._spawn_corpse(s)
        deaths = [s.id for s, _ in dying]
        deaths_detailed = [(s.id, cause) for s, cause in dying]
        # phase 3: chickens, eat, energy decay, starvation, spawn, mating, hatching (every live snake)
        self.update_chickens()
        ate = self.try_eat()
        self.decay_energy()
        for s in self.snakes:
            if s.alive and s.energy <= 0:
                s.alive = False; s.death_cause = "starve"
                self._spawn_corpse(s)
                deaths_detailed.append((s.id, "starve"))
        self.maybe_spawn()
        self._resolve_mating()
        hatched_owners = self._hatch_eggs()
        self._prune_dead()
        return {"ate": ate, "died": not ego.alive, "dashed": ego_dashed, "deaths": deaths,
                "deaths_detailed": deaths_detailed, "hatched_owners": hatched_owners}


def _ego_prop(name):
    return property(lambda self: getattr(self.snakes[0], name),
                    lambda self, v: setattr(self.snakes[0], name, v))
for _n in World._EGO_ATTRS:
    setattr(World, _n, _ego_prop(_n))
