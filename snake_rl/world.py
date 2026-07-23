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
    stun: int = 0          # >0 => frozen (dizzy) after dashing into a solid; counts down in _move_snake
    speed: float = 0.0     # actual translation speed of the LAST move (0 while stopped/stunned) -> prey sense
    genome: np.ndarray = None            # (9,) float32; None -> filled to a mid genome by __post_init__
    sex: int = 0                         # 0=female, 1=male
    age: int = 0
    max_lifespan: float = 1e9
    lineage: int = 0
    phenotype: object = None             # resolved namedtuple; filled by __post_init__/_make_snake

    def __post_init__(self):
        # Self-heal raw Snake(...) construction (tests, ad-hoc): a None genome becomes the mid
        # genome and its phenotype resolves against the default CFG. _make_snake passes both
        # explicitly, so this is a no-op on the real spawn path.
        if self.genome is None:
            from .config import CFG
            from .genome import resolve_phenotype, GENE_COUNT
            self.genome = np.full(GENE_COUNT, 0.5, np.float32)
            self.phenotype = resolve_phenotype(self.genome, CFG)
            if self.max_lifespan == 1e9:
                self.max_lifespan = self.phenotype.max_lifespan_base
        elif self.phenotype is None:
            from .config import CFG
            from .genome import resolve_phenotype
            self.phenotype = resolve_phenotype(self.genome, CFG)

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
    d = torus_delta(a, b, size)
    return np.sqrt((d * d).sum(-1))      # == linalg.norm(axis=-1), bit-identical, ~2x faster


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
        from .genome import sample_genome
        self.snakes = [self._make_snake(
            wrap(s / 2.0, s), float(self.rng.uniform(0, 2 * np.pi)),
            genome=sample_genome(self.rng), sex=int(self.rng.integers(0, 2)), lineage=0,
            id=0, color_seed=0, energy=cfg.energy_max, target_length=cfg.start_length, rng=self.rng,
        )]
        self._next_snake_id = 1
        # chickens / obstacles filled by worldgen; default empty
        self.chicken_pos = np.zeros((0, 2)); self.chicken_dir = np.zeros((0,))
        self.chicken_id = np.zeros((0,), dtype=int)      # stable id per chicken
        # behavior FSM, parallel to chicken_pos: state 0=peck 1=walk 2=flee, timer = steps left
        # in the current peck/walk, startle = flee steps still fluttering fast, flee = FEAR-PERSISTENCE
        # countdown (a scared hen keeps bolting this many steps even after the snake stops/leaves, so a
        # snake can't "un-spook" it by freezing). All kept the SAME length as chicken_pos everywhere
        # (created in _add_chicken/set_chickens, filtered in _snake_eat).
        self.chicken_state = np.zeros((0,), dtype=int)
        self.chicken_timer = np.zeros((0,), dtype=int)
        self.chicken_startle = np.zeros((0,), dtype=int)
        self.chicken_flee = np.zeros((0,), dtype=int)
        self._next_chicken_id = 0
        # chickens DROP FROM THE SKY (Goal 2): a spawned chicken first spends chicken_arrive_steps
        # here, falling, then lands into the real chicken_* arrays. Kept SEPARATE from chicken_pos so
        # sensors/eat (which read chicken_pos) never see an in-flight bird. sky=True in the viewer /
        # training world (worldgen `arrivals=True`); False keeps the plain instant-spawn for unit fixtures.
        # "head" carries a stable random heading per in-flight bird (set at spawn), handed to the
        # landed chicken's chicken_dir so the fall's facing matches where it walks off (render only).
        self.arriving = {"pos": np.zeros((0, 2)), "timer": np.zeros((0,), dtype=int),
                         "head": np.zeros((0,))}
        self.chicken_sky = False
        self.obstacle_pos = np.zeros((0, 2)); self.obstacle_r = np.zeros((0,))
        self.obstacle_kind = np.zeros((0,), dtype=int)   # 0=rock,1=tree (render only)
        self.corpses = {"pos": np.zeros((0, 2)), "food": np.zeros((0,))}
        self.eggs = {"pos": np.zeros((0, 2)), "timer": np.zeros((0,)), "owner": np.zeros((0, 2), int)}
        self._mate_streak = {}
        # bumped whenever ANY snake's body geometry changes (move/grow); keys the per-snake dense-body
        # cache reused across observers within a step (see _cached_render_body). Move bumps it N times
        # in phase 1, then it's stable through phase-2 death checks and (post-eat) through observe.
        self._body_gen = 0

    def _make_snake(self, head, heading, *, genome, sex, lineage, id, color_seed,
                    energy, target_length, rng):
        """Single construction path for a live Snake: resolves the genome's phenotype once and
        caches it on the instance (World.__init__, worldgen founders, _hatch_eggs all route here)."""
        from .genome import resolve_phenotype
        ph = resolve_phenotype(genome, self.cfg)
        jitter = 1.0 + rng.uniform(-self.cfg.lifespan_jitter, self.cfg.lifespan_jitter)
        return Snake(
            head_uw=head.copy(), head=head.copy(), heading=heading, path_uw=[head.copy()],
            target_length=target_length, stamina=ph.s_max, energy=energy,
            _prev_head_uw=head.copy(), id=id, color_seed=color_seed,
            genome=genome, sex=int(sex), age=0, max_lifespan=ph.max_lifespan_base * jitter,
            lineage=lineage, phenotype=ph,
        )

    def _phenotype_of(self, snake):
        return snake.phenotype

    # --- motion (per-snake workers) ---
    def _slide(self, p0, disp, centers, radii):
        """Move p0 by disp; if it would enter any solid circle (center, radius), slide along the first
        one hit (project the remaining motion onto the tangent). Returns (new_pos_uw, hit: bool).
        centers compared via torus_delta; radii already include head_radius."""
        if not len(centers):
            return p0 + disp, False
        d = disp
        m = torus_delta(centers, p0, self.size)              # (K,2) p0->center
        a = float(d @ d)
        best_t, best_i = 1.0, -1
        if a > 1e-12:
            b = -2.0 * (m @ d)
            c = np.einsum("ij,ij->i", m, m) - radii ** 2
            disc = b * b - 4 * a * c
            ok = disc >= 0
            t0 = np.where(ok, (-b - np.sqrt(np.clip(disc, 0, None))) / (2 * a), np.inf)
            t0 = np.where(ok & (t0 >= -1e-9) & (t0 <= 1.0), t0, np.inf)
            i = int(np.argmin(t0))
            if np.isfinite(t0[i]):
                best_t, best_i = max(0.0, float(t0[i])), i
        if best_i < 0:
            return p0 + d, False
        contact = p0 + best_t * d
        n = torus_delta(contact[None], centers[best_i][None], self.size)[0]   # center->contact (outward)
        ln = np.sqrt((n * n).sum())
        if ln < 1e-9:
            return contact, True
        n = n / ln
        rem = (1.0 - best_t) * d
        tang = rem - (rem @ n) * n
        out = contact + tang
        md = torus_delta(centers, out, self.size)
        if (np.einsum("ij,ij->i", md, md) < radii ** 2 - 1e-6).any():
            out = contact + 1e-3 * n
        return out, True

    def _move_snake(self, s, speed_idx, steer, dash):
        c = self.cfg
        self._body_gen += 1                              # body geometry about to change -> invalidate cache
        if s.stun > 0:                                   # dizzy from a dash into a solid: frozen (steering too)
            s.stun -= 1
            s.dashed = False
            s.speed = 0.0
            s.stamina = min(c.s_max, s.stamina + c.stamina_regen)   # stunned == a dead stop -> full (speed-0) regen
            s._prev_head_uw = s.head_uw.copy()
            s.path_uw.append(s.head_uw.copy())
            self._prune_path(s)
            s.steps += 1
            return False
        if steer == 0:
            s.heading -= np.radians(c.turn_deg)
        elif steer == 2:
            s.heading += np.radians(c.turn_deg)
        s.heading %= 2 * np.pi
        dashing = bool(dash) and s.stamina >= c.dash_min_stamina  # reserve gate (curriculum-tunable)
        speed = c.v_dash if dashing else c.speed_levels[speed_idx] * c.v_snake
        s.speed = speed
        prev_uw = s.head_uw.copy()
        disp = speed * s.heading_vec()
        # solids the head slides along (Minkowski + head_radius): obstacles + own body (neck-skipped)
        cen, rad = [], []
        if len(self.obstacle_pos):
            cen.append(self.obstacle_pos); rad.append(self.obstacle_r + c.head_radius)
        body = self._body_points_uw(s)                   # keeps the Pitfall-5 neck-skip
        if len(body):
            cen.append(body); rad.append(np.full(len(body), c.body_radius + c.head_radius))
        solid_centers = np.concatenate(cen, axis=0) if cen else np.zeros((0, 2))
        solid_radii = np.concatenate(rad, axis=0) if rad else np.zeros((0,))
        new, hit = self._slide(prev_uw, disp, solid_centers, solid_radii)
        if hit and dashing:                              # dashed into a solid -> head-spinning stun
            s.stun = c.stun_steps
        s.head_uw = new
        s.head = wrap(new, self.size)
        s.path_uw.append(s.head_uw.copy())
        self._prune_path(s)
        if dashing:
            s.stamina = max(0.0, s.stamina - c.stamina_drain)
        else:
            # regen scales with the CHOSEN cruise speed: full at a dead stop (speed_idx 0), zero at
            # full cruise (speed_idx 3). Standing still both ambushes AND recharges the dash reserve
            # fastest -> the net must learn to pause to bank stamina (v2.1, resume-trained).
            s.stamina = min(c.s_max, s.stamina + c.stamina_regen * (1.0 - c.speed_levels[speed_idx]))
        s.steps += 1
        s._prev_head_uw = prev_uw
        s.dashed = dashing
        return dashing

    def _prune_path(self, s):
        pts = np.array(s.path_uw)
        if len(pts) < 3:
            return
        dp = np.diff(pts, axis=0)
        seg = np.sqrt((dp * dp).sum(1))
        cum = np.cumsum(seg[::-1])[::-1]              # dist from head to each point's forward neighbor
        # slack = max motion step (v_dash), so body_points_uw never truncates its tail after a dash
        keep = np.concatenate([cum <= (s.target_length + self.cfg.v_dash), [True]])
        s.path_uw = [p.copy() for p in pts[keep]]

    def _arc_points(self, pts, targets):
        """Place a point at each arc distance in `targets`, measured from the head (pts[-1]) walking
        BACKWARD along the polyline. Vectorized replacement for the old per-vertex Python loop, and
        BIT-IDENTICAL to it: same per-segment length (squared-sum sqrt), same left-to-right cumulative
        arc (cumsum == the loop's `acc`; cumstart via concat, NOT cumend-seglen, to keep the exact
        accumulation), same "first segment whose cumulative end >= target" assignment (searchsorted
        'left' == the loop's `acc+step >= target`), same a + (b-a)*frac. Zero-length segments are
        dropped (the loop's `continue`); targets past the path's total length are dropped (loop ends)."""
        targets = np.asarray(targets, float)
        rev = pts[::-1]
        seg = rev[1:] - rev[:-1]                          # (M,2) b-a per segment (a = head-side vertex)
        seglen = np.sqrt((seg * seg).sum(1))
        keep = seglen >= 1e-12
        if not keep.any():
            return np.zeros((0, 2))
        a = rev[:-1][keep]; seg = seg[keep]; seglen = seglen[keep]
        cumend = np.cumsum(seglen)                        # arc at each segment tail (== loop acc+step)
        cumstart = np.concatenate([[0.0], cumend[:-1]])   # arc at each segment head (== loop `acc`)
        k = np.searchsorted(cumend, targets, side="left")
        valid = k < len(seglen)
        if not valid.any():
            return np.zeros((0, 2))
        k = k[valid]
        frac = (targets[valid] - cumstart[k]) / seglen[k]
        return a[k] + seg[k] * frac[:, None]

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
        while t <= s.target_length:                       # kept as-is: bit-identical target arc values
            targets.append(t); t += c.segment_spacing
        if not targets:
            return np.zeros((0, 2))
        return self._arc_points(pts, targets)

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
        arc = self._arc_points(pts, targets[1:])          # targets[0]=0 is the head, prepended as idx 0
        return np.vstack([pts[-1][None], arc]) if len(arc) else pts[-1][None].copy()

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

    def _add_chicken(self, p, arriving=False, head=None):
        p = np.asarray(p, float)
        h = float(self.rng.uniform(0, 2 * np.pi)) if head is None else float(head)
        if arriving:                                     # queue a sky-drop; lands into the real arrays later
            self.arriving["pos"] = (np.vstack([self.arriving["pos"], p[None]])
                                    if len(self.arriving["pos"]) else p[None].copy())
            self.arriving["timer"] = np.append(self.arriving["timer"], self.cfg.chicken_arrive_steps)
            self.arriving["head"] = np.append(self.arriving["head"], h)   # lockstep with pos/timer (Pitfall 17)
            return
        self.chicken_pos = np.vstack([self.chicken_pos, p]) if len(self.chicken_pos) else p[None]
        self.chicken_dir = np.append(self.chicken_dir, h)   # landed birds keep their in-flight facing
        self.chicken_id = np.append(self.chicken_id, self._next_chicken_id)
        st, tm = self._random_state_timer()
        self.chicken_state = np.append(self.chicken_state, st)
        self.chicken_timer = np.append(self.chicken_timer, tm)
        self.chicken_startle = np.append(self.chicken_startle, 0)
        self.chicken_flee = np.append(self.chicken_flee, 0)
        self._next_chicken_id += 1

    def _land_arrivals(self):
        """Tick every in-flight chicken's fall; any that reached the ground lands into the real
        chicken arrays (a normal huntable/sensed chicken from that step on). Two-phase-safe: this
        touches no snake and resolves no death, it just turns a sky-drop into a real chicken."""
        a = self.arriving
        if not len(a["pos"]):
            return
        a["timer"] = a["timer"] - 1
        landed = a["timer"] <= 0
        if landed.any():
            for p, h in zip(a["pos"][landed], a["head"][landed]):
                self._add_chicken(p, arriving=False, head=h)     # keep the in-flight facing on landing
            keep = ~landed
            a["pos"] = a["pos"][keep]; a["timer"] = a["timer"][keep]; a["head"] = a["head"][keep]

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
        self.chicken_flee = np.zeros(n, int)
        self.arriving = {"pos": np.zeros((0, 2)), "timer": np.zeros((0,), dtype=int),
                         "head": np.zeros((0,))}                                         # full chicken reset
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
        # a chicken must flee the actual LIVE snake, never a frozen dead-ego ghost. Carry each live
        # snake's SPEED too -> prey senses motion (a stopped snake is invisible; Task 3).
        live = [s for s in self.snakes if s.alive]
        heads = np.array([s.head for s in live]) if live else np.zeros((0, 2))
        speeds = np.array([s.speed for s in live]) if live else np.zeros((0,))
        for i in range(len(self.chicken_pos)):
            base, speed = self._chicken_step(i, heads, speeds)
            if speed > 0.0:
                steer(i, base, speed)                                   # peck (speed 0) just stands still

    def _chicken_step(self, i, heads, speeds):
        """FSM transition for chicken i against live snake `heads` (moving at `speeds`); returns
        (heading, speed). Mutates chicken_state/timer/startle in place. Flee overrides the peck/walk
        timer; when no snake is near a fleeing chicken it settles to WALK (not straight back to pecking)."""
        c = self.cfg
        # --- threat: flee the repulsion resultant of every snake within its alert radius (never bolt
        #     from one snake straight into another; opposing snakes that cancel -> flee the nearest).
        #     A head-down PECKING chicken is distracted: base alert is the tight r_flee_peck (the
        #     stalk-and-pounce window); WALK / already-fleeing use the full r_flee. Prey senses MOTION:
        #     each snake's effective alert scales with its speed, CAPPED at 1x base -> a stopped snake
        #     never alerts, a dash is no scarier than full cruise (keeps max alert = today's r_flee). ---
        alert = c.r_flee_peck if self.chicken_state[i] == 0 else c.r_flee
        if len(heads):
            to_heads = torus_delta(heads, self.chicken_pos[i], self.size)   # (K,2) chicken->each head
            dist = np.sqrt((to_heads * to_heads).sum(1))
            eff = alert * np.clip(speeds / c.v_snake, 0.0, 1.0)             # per-snake speed-scaled reach
            near = (dist < eff) & (dist > 1e-6)
        else:
            near = np.zeros(0, bool)
        if near.any():                                                     # a MOVING snake is within reach
            if self.chicken_state[i] != 2:                                  # startle-FREEZE on ENTERING flee
                self.chicken_state[i] = 2
                self.chicken_startle[i] = c.chicken_startle_steps
            self.chicken_flee[i] = c.chicken_flee_persist                   # (re)arm fear-persistence
            # linear falloff, ->0 at r_flee (the WALK/flee alert radius; for a PECK-triggered startle
            # the alert is the tighter r_flee_peck, so weight there bottoms out at ~r_flee-r_flee_peck)
            weight = c.r_flee - dist[near]
            away = -to_heads[near] / dist[near, None]
            repulsion = (weight[:, None] * away).sum(axis=0)
            if np.sqrt((repulsion * repulsion).sum()) > 1e-6:
                base = float(np.arctan2(repulsion[1], repulsion[0]))
            else:                                                          # degenerate cancellation ->
                j = int(np.argmin(np.where(near, dist, np.inf)))           # flee the nearest NEAR snake
                base = float(np.arctan2(-to_heads[j][1], -to_heads[j][0]))
            self.chicken_dir[i] = base                                      # remember the flee heading
            if self.chicken_startle[i] > 0:
                self.chicken_startle[i] -= 1
                return base, 0.0                                            # startle FREEZE: a beat of surprise
            return base, c.v_flee                                          # then bolt away

        # --- FEAR PERSISTENCE: no snake is within reach right now (it stopped, slowed, or backed off),
        #     but a hen that was just scared keeps BOLTING in its last flee direction until its panic
        #     timer runs out -- it does NOT re-settle the instant the predator freezes. This is what
        #     kills the "spook the chicken, then stop dead so it calms, then grab it" exploit. ---
        if self.chicken_state[i] == 2 and self.chicken_flee[i] > 0:
            self.chicken_flee[i] -= 1
            if self.chicken_startle[i] > 0:                                 # finish a pending startle freeze
                self.chicken_startle[i] -= 1
                return self.chicken_dir[i], 0.0
            return self.chicken_dir[i], c.v_flee                            # keep bolting the last flee heading

        # --- calm at last: panic expired (or never scared) -> a fleeing chicken settles to WALK ---
        if self.chicken_state[i] == 2:
            self.chicken_state[i] = 1
            self.chicken_timer[i] = int(self.rng.integers(c.chicken_walk_min, c.chicken_walk_max + 1))
            self.chicken_startle[i] = 0
            self.chicken_flee[i] = 0
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
                self.chicken_flee = self.chicken_flee[keep]
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
            owned = owner[:, 0] >= 0        # spawn/arrival eggs (owner -1) belong to nobody: uneatable,
            foreign &= owned                # a GUARANTEED arrival, never scavenged before it hatches
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
            self._body_gen += 1                          # growth changes body length -> invalidate cache
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
            live = [o for o in self.snakes if o.alive]      # clear EVERY live snake's body, not just the ego
            body = np.concatenate([self._body_points(o) for o in live], axis=0) if live else np.zeros((0, 2))
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
        n = len(self.chicken_pos) + len(self.arriving["pos"])  # in-flight birds count toward the target
        if n >= max_target:                                    # (else we'd over-spawn while they fall)
            return
        p = 0.06 if n < min_target else 1.0 / c.spawn_period   # fast refill to min, then random to max
        if n == 0 or self.rng.random() < p:
            self._add_chicken(self._free_point(c.chicken_radius), arriving=self.chicken_sky)

    def maybe_spawn_forced(self, arriving=False):
        self._add_chicken(self._free_point(self.cfg.chicken_radius), arriving=arriving)

    # --- reproduction ---
    def _lay_egg(self, pos, id_a, id_b, timer=None):
        e = self.eggs
        e["pos"] = np.vstack([e["pos"], pos[None]]) if len(e["pos"]) else pos[None].copy()
        e["timer"] = np.append(e["timer"], self.cfg.egg_timer if timer is None else timer)
        row = np.array([[id_a, id_b]])
        e["owner"] = np.vstack([e["owner"], row]) if len(e["owner"]) else row

    def spawn_egg(self, pos, timer=None):
        """Lay a GUARANTEED arrival egg (owner -1 = nobody's): every new NON-ego snake ARRIVES via
        one of these instead of popping in (Goal 1) -- used by worldgen for opponents and by the
        viewer reseed floor. Reuses the repro egg/hatch machinery, but an owner -1 makes it
        uneatable (_snake_eat) and n_max-cap-exempt (_hatch_eggs), and it pays no repro reward /
        counts as no 'birth' (excluded from hatched_owners). It still renders + hatches (shell
        crack) exactly like a normal egg."""
        self._lay_egg(np.asarray(pos, float), -1, -1, timer=timer)

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
                is_spawn = e["owner"][i][0] < 0        # guaranteed arrival egg: bypasses the n_max cap
                if n_alive >= c.n_max and not is_spawn:
                    continue
                pos = e["pos"][i].copy()
                sid = self._next_snake_id
                self._next_snake_id += 1
                # M1 (Task 3): eggs don't carry a genome yet (Task 8) -- a hatchling samples a fresh
                # one + random sex; lineage = its own id (always fresh, no separate counter needed).
                from .genome import sample_genome
                self.snakes.append(self._make_snake(
                    wrap(pos, self.size), float(self.rng.uniform(0, 2 * np.pi)),
                    genome=sample_genome(self.rng), sex=int(self.rng.integers(0, 2)), lineage=sid,
                    id=sid, color_seed=sid, energy=c.hatch_energy_frac * c.energy_max,
                    target_length=c.start_length, rng=self.rng,
                ))
                n_alive += 1
                if not is_spawn:                       # only real matings pay repro reward / count as births
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
    def _cached_render_body(self, o):
        """`_body_render_path_uw(o)` memoized on the current _body_gen. _other_hazard recomputes each
        rival's dense body once per observer (N-1 death checks + N-1 sensor scans per step); the body
        is fixed between geometry changes, so cache it per snake and reuse. Bit-identical (returns the
        very array a fresh call would build). Read-only by convention -- _other_hazard only reads/copies."""
        cache = getattr(o, "_render_cache", None)
        if cache is not None and cache[0] == self._body_gen:
            return cache[1]
        body = self._body_render_path_uw(o)
        o._render_cache = (self._body_gen, body)
        return body

    def _other_hazard(self, s):
        pts, rads = [], []
        for o in self.snakes:
            if o is s or not o.alive:
                continue
            pts.append(o.head_uw); rads.append(self.cfg.head_radius)   # head (head_radius)
            body = self._cached_render_body(o)[1:]     # dense body, NO neck skip; skip idx-0 head (added above)
            if len(body):
                pts.extend(body); rads.extend([self.cfg.body_radius] * len(body))
        if not pts:
            return np.zeros((0, 2)), np.zeros((0,))
        return np.array(pts), np.array(rads)

    def _death_cause(self, s):
        """Pure — returns 'snake'|None for s vs post-move state. No mutation. Obstacles and the snake's
        OWN body are now solid (slid along in _move_snake), never lethal; only a RIVAL's body/head
        kills (cut-off), swept over s._prev_head_uw -> s.head_uw (the slid path)."""
        c = self.cfg; hr = c.head_radius
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

    def heading_vec(self):                     return self.snakes[0].heading_vec()
    def move(self, speed_idx, steer, dash):    return self._move_snake(self.snakes[0], speed_idx, steer, dash)
    def check_death(self):                     return self._check_death(self.snakes[0])
    def body_points_uw(self):         return self._body_points_uw(self.snakes[0])
    def body_points(self):            return self._body_points(self.snakes[0])
    def body_render_path_uw(self, spacing=None):
        return self._body_render_path_uw(self.snakes[0], spacing)

    def step(self, speed_idx, steer, dash, opponent_fn=None):
        opponent_fn = opponent_fn or (lambda world, s: (1, 1, 0))    # default opp: ⅓-cruise straight
        # phase 1: EVERY snake (ego + opponents) acts on the PRE-MOVE world, so collect all opponent
        # actions BEFORE moving anyone [M-2], then move ego, then move opponents with those actions.
        ego = self.snakes[0]
        opp_actions = {o.id: opponent_fn(self, o) for o in self.snakes[1:] if o.alive}
        ego_dashed = self._move_snake(ego, speed_idx, steer, dash) if ego.alive else False
        for o in self.snakes[1:]:
            if o.alive:
                sp, st, da = opp_actions[o.id]
                self._move_snake(o, sp, st, da)
        # phase 2: DECIDE all deaths against frozen post-move state, THEN apply (order-independent, C2)
        dying = [(s, cause) for s in self.snakes if s.alive and (cause := self._death_cause(s))]
        for s, cause in dying:
            s.alive = False; s.death_cause = cause
            self._spawn_corpse(s)
        deaths = [s.id for s, _ in dying]
        deaths_detailed = [(s.id, cause) for s, cause in dying]
        # phase 3: chickens, eat, energy decay, starvation, spawn, mating, hatching (every live snake)
        self._land_arrivals()                 # sky-dropped chickens that reached the ground become real
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
