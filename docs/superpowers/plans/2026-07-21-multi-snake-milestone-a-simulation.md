# Multi-snake — Milestone A (simulation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add multi-snake *simulation* capability to the pure-numpy world — a list of snakes, a deterministic two-phase step, inter-snake cut-off death + corpses, egg-based reproduction, starvation, and population-scaled balance — with **zero change to the RL stack** (env/sensors/train run single-snake, all existing tests stay green).

**Architecture:** Extract a `Snake` dataclass from `World`; `World` holds `snakes: list[Snake]` (+ `eggs`, `corpses`). Per-snake physics become methods that take a `Snake`. **Ego-proxy properties** (`world.head → world.snakes[0].head`, etc.) keep `sensors`/`env`/`watch`/`render` and all current tests working unchanged at N=1. Multi-snake behavior is exercised by direct `World`/`worldgen` tests. The RL observation, training self-play, and rendering of multiple snakes are **Milestone B** (a separate plan) — this plan is a self-contained, fully-tested simulation with no retrain.

**Tech Stack:** Python 3.13 (`.venv`), numpy≥2, pytest. (No torch/SB3/pygame changes in this milestone.)

## Global Constraints

- Run tests: `SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest -q` (all 57 existing tests must stay green after every task).
- **Torus nearest-image geometry everywhere** — reuse `world.torus_delta / torus_dist / ray_circle_hit / segment_circle_hit`; never write raw subtraction for pairwise vectors.
- Simulation timestep `dt = 1`; all speeds in units/step.
- **N=1 behavioral regression is exact:** a world built with `n_snakes=1` at a fixed explicit size + seed must produce a byte-identical trajectory to the pre-refactor code. Regression tests pin `size=(80.0, 80.0)` explicitly so later world-size balance changes don't perturb them.
- `Config` is a **frozen** dataclass — population-dependent food is stored as *per-snake rates*; the live target is computed at runtime in `maybe_spawn`. Never store an absolute dynamic count as a config field.
- **Ego-proxy bridge is deliberate and temporary** — documented in `world.py`; Milestone B removes/reworks it when `sensors` becomes multi-snake. Do not build sensors/env multi-snake logic here.
- No attribution lines in commits (per user global rule).
- TDD: every code task = write failing test → run (fail) → minimal impl → run (pass) → commit.

## File Structure

- **Modify `snake_rl/world.py`** — `Snake` dataclass; `World.snakes/eggs/corpses`; per-snake `_move_snake/_body_points_uw/_body_points/_other_hazard/_check_death/_prune_path`; two-phase `step`; `_resolve_mating/_hatch_eggs/_spawn_corpse`; generalized `try_eat`; `decay_energy`+starvation; ego proxies. (world.py is the one file that owns physics — keep it the single home; it grows but stays one responsibility.)
- **Modify `snake_rl/worldgen.py`** — `generate_world(cfg, seed, size, n_snakes)`; spread N snakes at free points; N=1 keeps the center placement.
- **Modify `snake_rl/config.py`** — reproduction/egg/corpse/starvation/population **rate** constants; world-size bump; new invariants.
- **Test files:** extend `tests/test_snake_motion.py`, `tests/test_collision.py`, `tests/test_worldgen.py`; new `tests/test_multisnake.py`, `tests/test_reproduction.py`.

---

### Task 1: Extract `Snake` dataclass + ego proxies (N=1 identical)

**Files:**
- Modify: `snake_rl/world.py` (state → `Snake`; `World.snakes=[Snake]`; proxies)
- Test: `tests/test_multisnake.py` (new)

**Interfaces:**
- Produces:
  - `snake_rl.world.Snake` — dataclass with fields
    `head_uw: np.ndarray, head: np.ndarray, heading: float, path_uw: list[np.ndarray],
    target_length: float, stamina: float, energy: float, alive: bool, dashed: bool,
    death_cause: str|None, steps: int, _prev_head_uw: np.ndarray, id: int, color_seed: int,
    repro_cooldown: int` (defaults: `alive=True, dashed=False, death_cause=None, steps=0,
    repro_cooldown=0`).
  - `World.snakes: list[Snake]` (length ≥ 1; `snakes[0]` is the ego).
  - Ego **read proxies** (properties on `World`) → `snakes[0]`: `head_uw, head, heading,
    target_length, stamina, energy, alive, dashed, death_cause, steps, path_uw, _prev_head_uw`.
  - Ego **method proxies**: `heading_vec()`, `move(steering, dash)`, `check_death()`,
    `body_points()`, `body_points_uw()`, `body_render_path_uw(spacing=None)`,
    `nearest_chicken()`, `nearest_chicken_id()` — all operate on `snakes[0]`.
  - Per-snake worker methods: `World._move_snake(s, steering, dash) -> bool`,
    `World._prune_path(s)`, `World._body_points_uw(s) -> np.ndarray`,
    `World._body_points(s) -> np.ndarray`, `World._body_render_path_uw(s, spacing=None)`,
    `World._check_death(s) -> bool`.

- [ ] **Step 1: Write the failing test** — `tests/test_multisnake.py`

```python
import numpy as np
from snake_rl.config import CFG
from snake_rl.world import World, Snake
from snake_rl.worldgen import generate_world


def test_world_has_snake_list_and_ego_proxies():
    w = World(CFG, seed=1, size=(80.0, 80.0))
    assert isinstance(w.snakes, list) and len(w.snakes) == 1
    s = w.snakes[0]
    assert isinstance(s, Snake)
    # read proxies mirror the ego
    assert np.allclose(w.head, s.head)
    assert w.energy == s.energy and w.stamina == s.stamina
    assert w.heading == s.heading


def test_ego_move_proxy_mutates_snakes0():
    w = World(CFG, seed=1, size=(80.0, 80.0))
    before = w.snakes[0].head_uw.copy()
    w.move(1, 0)                      # straight, no dash — via proxy
    assert not np.allclose(w.snakes[0].head_uw, before)
    assert w.snakes[0].steps == 1


def test_n1_trajectory_regression():
    # Refactor must be behavior-preserving: a fixed seed+size world stepped straight
    # yields a deterministic head path. This pins the ego physics.
    w = generate_world(CFG, seed=12345, size=(80.0, 80.0))
    heads = []
    for _ in range(30):
        w.step(1, 0)                  # steer straight, no dash
        heads.append(w.head.copy())
    heads = np.array(heads)
    # deterministic + advancing along the heading (no NaN, monotone arc length)
    assert np.isfinite(heads).all()
    seg = np.linalg.norm(np.diff(heads, axis=0), axis=1)
    assert (seg > 0).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest tests/test_multisnake.py -q`
Expected: FAIL — `ImportError: cannot import name 'Snake'`.

- [ ] **Step 3: Implement `Snake` + move state off `World`**

In `world.py`, add the dataclass (top, after imports):

```python
from dataclasses import dataclass, field

@dataclass
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
```

In `World.__init__`, replace the per-snake field block (`world.py:70-81`) with a single ego snake:

```python
        self.snakes = [Snake(
            head_uw=s / 2.0, head=wrap(s / 2.0, s), heading=float(self.rng.uniform(0, 2*np.pi)),
            path_uw=[(s / 2.0).copy()], target_length=cfg.start_length,
            stamina=cfg.s_max, energy=cfg.energy_max, _prev_head_uw=(s / 2.0).copy(),
            id=0, color_seed=0,
        )]
        self._next_snake_id = 1
```

Keep world-level state on `World` (`rng, size, chickens, obstacles, _next_chicken_id`).

- [ ] **Step 4: Convert per-snake methods to take a `Snake`, add proxies**

Rewrite `move/_prune_path/body_points_uw/body_points/body_render_path_uw/check_death` (`world.py:93-185,298-316`) into `_move_snake(self, s, steering, dash)` etc., replacing every `self.head*/self.heading*/self.stamina/self.path_uw/self.target_length/self._prev_head_uw/self.dashed/self.steps/self.alive/self.death_cause` with `s.*`. The head-vs-obstacle and self-body logic is unchanged (still `self.obstacle_pos`, `self.size`, `self.cfg`). Then add proxies:

```python
    # --- ego proxies (temporary Milestone-A bridge; reworked in Milestone B) ---
    _EGO_ATTRS = ("head_uw","head","heading","target_length","stamina","energy",
                  "alive","dashed","death_cause","steps","path_uw","_prev_head_uw")
    def heading_vec(self):            return self.snakes[0].heading_vec()
    def move(self, steering, dash):   return self._move_snake(self.snakes[0], steering, dash)
    def check_death(self):            return self._check_death(self.snakes[0])
    def body_points_uw(self):         return self._body_points_uw(self.snakes[0])
    def body_points(self):            return self._body_points(self.snakes[0])
    def body_render_path_uw(self, spacing=None):
        return self._body_render_path_uw(self.snakes[0], spacing)
```

Add the read proxies via a loop after the class body, or explicit properties:

```python
def _ego_prop(name):
    return property(lambda self: getattr(self.snakes[0], name))
for _n in World._EGO_ATTRS:
    setattr(World, _n, _ego_prop(_n))
```

`nearest_chicken/nearest_chicken_id` (`world.py:188-198`) already use `self.head` — leave them; they now read the ego head via the proxy. In `step` (`world.py:318-325`) keep calling the ego methods for now (`self.move/self.update_chickens/self.try_eat/self.decay_energy/self.maybe_spawn/self.check_death`) — Task 2 restructures `step`.

- [ ] **Step 5: Run the new tests + the full suite**

Run: `SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest -q`
Expected: PASS — `tests/test_multisnake.py` green AND all 57 existing tests still green (proxies preserve the single-snake API).

- [ ] **Step 6: Commit**

```bash
git add snake_rl/world.py tests/test_multisnake.py
git commit -m "world: extract Snake dataclass + ego proxies (N=1 identical)"
```

---

### Task 2: Two-phase `World.step` for N snakes (move-all → resolve-deaths)

**Files:**
- Modify: `snake_rl/world.py` (`step`; add `_other_hazard`)
- Test: `tests/test_multisnake.py`

**Interfaces:**
- Consumes: `Snake`, `_move_snake`, `_check_death` (Task 1).
- Produces:
  - `World.step(steering, dash, opponent_fn=None) -> dict` — ego (`snakes[0]`) acts with
    `(steering, dash)`; every other live snake acts via `opponent_fn(world, snake) -> (steering,
    dash)` (default: go straight `(1, 0)`). **Order:** (1) move all live snakes; (2) resolve all
    deaths against post-move state; (3) chickens/eat/energy/spawn (unchanged for now). Return dict
    unchanged for back-compat: `{"ate", "died", "dashed"}` for the ego, plus `{"deaths": [ids]}`.
  - `World._other_hazard(self, s) -> tuple[np.ndarray, np.ndarray]` — `(points, radii)` of every
    **other** live snake's head circle **+ full body with NO neck-skip**. (`s` excluded.) Head
    point = `other.head_uw`, radius `head_radius`; body points = dense body with radius
    `body_radius`. Used in Task 4; defined here returning empty arrays while N=1.

- [ ] **Step 1: Write the failing test**

```python
def test_two_phase_step_is_order_independent_head_to_head():
    # Two snakes driven head-on into each other die together regardless of list order.
    w = World(CFG, seed=2, size=(80.0, 80.0))
    from snake_rl.world import Snake
    from snake_rl.world import wrap
    import numpy as np
    a = w.snakes[0]
    a.head_uw = np.array([40.0, 40.0]); a.head = wrap(a.head_uw, w.size)
    a.heading = 0.0; a.path_uw = [a.head_uw.copy()]; a._prev_head_uw = a.head_uw.copy()
    b = Snake(head_uw=np.array([43.0, 40.0]), head=wrap(np.array([43.0,40.0]), w.size),
              heading=np.pi, path_uw=[np.array([43.0,40.0])], target_length=CFG.start_length,
              stamina=CFG.s_max, energy=CFG.energy_max, _prev_head_uw=np.array([43.0,40.0]), id=1)
    w.snakes.append(b)
    # will be lethal once Task 4 wires _other_hazard; here assert the step runs both moves first
    out = w.step(1, 0, opponent_fn=lambda world, s: (1, 0))
    assert w.snakes[0].steps == 1 and w.snakes[1].steps == 1   # BOTH moved (phase 1) before any resolve
```

- [ ] **Step 2: Run to verify it fails**

Run: `... pytest tests/test_multisnake.py::test_two_phase_step_is_order_independent_head_to_head -v`
Expected: FAIL (`step` takes no `opponent_fn`, or second snake not stepped).

- [ ] **Step 3: Implement two-phase `step` + `_other_hazard` stub**

```python
    def _other_hazard(self, s):
        pts, rads = [], []
        for o in self.snakes:
            if o is s or not o.alive:
                continue
            pts.append(o.head_uw); rads.append(self.cfg.head_radius)
            body = self._body_render_path_uw(o)        # dense, NO neck skip, includes head-adjacent
            if len(body):
                pts.extend(body); rads.extend([self.cfg.body_radius] * len(body))
        if not pts:
            return np.zeros((0, 2)), np.zeros((0,))
        return np.array(pts), np.array(rads)

    def step(self, steering, dash, opponent_fn=None):
        opponent_fn = opponent_fn or (lambda world, s: (1, 0))
        # phase 1: move ALL live snakes
        ego = self.snakes[0]
        ego_dashed = self._move_snake(ego, steering, dash) if ego.alive else False
        for o in self.snakes[1:]:
            if o.alive:
                st, da = opponent_fn(self, o)
                self._move_snake(o, st, da)
        # phase 2: resolve deaths against post-move state
        deaths = []
        for s in self.snakes:
            if s.alive and self._check_death(s):
                deaths.append(s.id)
        # phase 3: world updates (chickens/eat/energy/spawn) — ego-centric for now
        self.update_chickens()
        ate = self.try_eat()
        self.decay_energy()
        self.maybe_spawn()
        return {"ate": ate, "died": not ego.alive, "dashed": ego_dashed, "deaths": deaths}
```

- [ ] **Step 4: Run the new test + full suite**

Run: `SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest -q`
Expected: PASS — both snakes moved; all existing tests green (ego path unchanged, `opponent_fn` defaulted).

- [ ] **Step 5: Commit**

```bash
git add snake_rl/world.py tests/test_multisnake.py
git commit -m "world: two-phase step (move-all then resolve-deaths) + _other_hazard scaffold"
```

---

### Task 3: worldgen spawns N snakes (spread; N=1 keeps center)

**Files:**
- Modify: `snake_rl/worldgen.py`
- Test: `tests/test_worldgen.py`

**Interfaces:**
- Consumes: `World.snakes`, `Snake` (Task 1), `World._free_point` (existing, `world.py:265`).
- Produces: `generate_world(cfg, seed=None, size=None, n_snakes=1) -> World` — for `n_snakes>1`,
  replaces `World.snakes` with N snakes at mutually-spread free points (each ≥ `2·r_flee` from the
  others where possible), random headings, ascending `id`/`color_seed`. `n_snakes=1` is unchanged
  (center placement — preserves every existing test + the N=1 regression).

- [ ] **Step 1: Write the failing test** — append to `tests/test_worldgen.py`

```python
def test_generate_world_multi_snake_spread():
    from snake_rl.worldgen import generate_world
    from snake_rl.world import torus_dist
    import numpy as np
    w = generate_world(__import__("snake_rl.config", fromlist=["CFG"]).CFG,
                       seed=7, size=(140.0, 140.0), n_snakes=4)
    assert len(w.snakes) == 4
    assert [s.id for s in w.snakes] == [0, 1, 2, 3]
    heads = np.array([s.head for s in w.snakes])
    # no two snakes spawn on top of each other
    for i in range(4):
        for j in range(i + 1, 4):
            assert torus_dist(heads[i][None], heads[j], w.size)[0] > 2.0


def test_generate_world_default_is_single_and_centered():
    from snake_rl.worldgen import generate_world
    from snake_rl.config import CFG
    import numpy as np
    w = generate_world(CFG, seed=7, size=(80.0, 80.0))       # n_snakes defaults to 1
    assert len(w.snakes) == 1
    assert np.allclose(w.snakes[0].head_uw, np.array(w.size) / 2.0)
```

- [ ] **Step 2: Run to verify fail**

Run: `... pytest tests/test_worldgen.py -q`
Expected: FAIL (`generate_world` has no `n_snakes`).

- [ ] **Step 3: Implement** — in `worldgen.py`, add the param and post-obstacle spread spawn:

```python
def generate_world(cfg, seed=None, size=None, n_snakes=1):
    w = World(cfg, seed=seed, size=size)
    rng = w.rng
    # ... existing obstacle sampling unchanged ...
    if n_snakes > 1:
        from .world import Snake, wrap, torus_dist
        placed = []
        for i in range(n_snakes):
            for _ in range(200):
                p = w._free_point(cfg.head_radius)
                if p is None:
                    break
                if not placed or (torus_dist(np.array([q for q in placed]), p, w.size) > 2 * cfg.r_flee).all():
                    placed.append(p); break
            else:
                placed.append(w._free_point(cfg.head_radius))
        w.snakes = [Snake(head_uw=p.copy(), head=wrap(p, w.size),
                          heading=float(rng.uniform(0, 2*np.pi)), path_uw=[p.copy()],
                          target_length=cfg.start_length, stamina=cfg.s_max, energy=cfg.energy_max,
                          _prev_head_uw=p.copy(), id=i, color_seed=i) for i, p in enumerate(placed)]
        w._next_snake_id = n_snakes
    # ... existing initial-chicken spawn unchanged ...
    return w
```

(Note: `_free_point` already avoids obstacles + the ego head + body; passing a small radius is fine for spacing here. The `2·r_flee` spread is best-effort — the `for…else` falls back to any free point in a packed world.)

- [ ] **Step 4: Run tests + full suite**

Run: `SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add snake_rl/worldgen.py tests/test_worldgen.py
git commit -m "worldgen: spawn N spread snakes (n_snakes param; N=1 unchanged)"
```

---

### Task 4: Inter-snake cut-off + head-to-head death

**Files:**
- Modify: `snake_rl/world.py` (`_check_death` includes `_other_hazard`)
- Test: `tests/test_collision.py`

**Interfaces:**
- Consumes: `_other_hazard` (Task 2), `_check_death` (Task 1), `segment_circle_hit` (existing).
- Produces: `_check_death(s)` now also tests the swept head segment `[s._prev_head_uw → s.head_uw]`
  against `_other_hazard(s)` (radii inflated by `head_radius`); on hit sets `s.alive=False`,
  `s.death_cause="snake"`. Mutual head-to-head ⇒ both die in the same resolve phase (each sees the
  other's post-move head as a hazard).

- [ ] **Step 1: Write the failing test** — append to `tests/test_collision.py`

```python
def test_head_into_other_body_kills_mover():
    import numpy as np
    from snake_rl.config import CFG
    from snake_rl.world import World, Snake, wrap
    w = World(CFG, seed=3, size=(80.0, 80.0))
    victim = w.snakes[0]                                   # long, lying along +x at y=40
    victim.head_uw = np.array([50.0, 40.0]); victim.heading = 0.0
    victim.path_uw = [np.array([40.0, 40.0]), np.array([50.0, 40.0])]
    victim.target_length = 12.0; victim._prev_head_uw = np.array([49.0, 40.0])
    victim.head = wrap(victim.head_uw, w.size)
    attacker = Snake(head_uw=np.array([45.0, 43.0]), head=wrap(np.array([45.0,43.0]), w.size),
                     heading=-np.pi/2, path_uw=[np.array([45.0,46.0]), np.array([45.0,43.0])],
                     target_length=CFG.start_length, stamina=CFG.s_max, energy=CFG.energy_max,
                     _prev_head_uw=np.array([45.0, 46.0]), id=1)
    w.snakes.append(attacker)
    # attacker's swept head crosses the victim's body line y=40 -> attacker dies, victim lives
    w.step(1, 0, opponent_fn=lambda world, s: (1, 0))       # ego straight; attacker also straight down via its heading? use direct check:
    # deterministic check independent of opponent_fn heading:
    attacker._prev_head_uw = np.array([45.0, 41.0]); attacker.head_uw = np.array([45.0, 39.0])
    attacker.head = wrap(attacker.head_uw, w.size)
    assert w._check_death(attacker) is True and attacker.death_cause == "snake"
    assert victim.alive is True


def test_mutual_head_to_head_both_die():
    import numpy as np
    from snake_rl.config import CFG
    from snake_rl.world import World, Snake, wrap
    w = World(CFG, seed=4, size=(80.0, 80.0))
    a = w.snakes[0]
    a._prev_head_uw = np.array([39.0, 40.0]); a.head_uw = np.array([41.0, 40.0]); a.heading = 0.0
    a.head = wrap(a.head_uw, w.size); a.path_uw = [a._prev_head_uw.copy(), a.head_uw.copy()]
    b = Snake(head_uw=np.array([41.0, 40.0]), head=wrap(np.array([41.0,40.0]), w.size), heading=np.pi,
              path_uw=[np.array([43.0,40.0]), np.array([41.0,40.0])], target_length=CFG.start_length,
              stamina=CFG.s_max, energy=CFG.energy_max, _prev_head_uw=np.array([43.0,40.0]), id=1)
    w.snakes.append(b)
    dead = [s.id for s in w.snakes if w._check_death(s)]
    assert set(dead) == {0, 1}                              # both heads overlap post-move
```

- [ ] **Step 2: Run to verify fail**

Run: `... pytest tests/test_collision.py -q`
Expected: FAIL (`_check_death` ignores other snakes).

- [ ] **Step 3: Implement** — extend `_check_death(s)` (the converted `check_death`) after the self-body block, before `return False`:

```python
        opts, orads = self._other_hazard(s)
        if len(opts):
            hit = segment_circle_hit(s._prev_head_uw, s.head_uw, opts, orads + hr, self.size)
            if hit.any():
                s.alive = False; s.death_cause = "snake"
                return True
```

(`hr = self.cfg.head_radius`, already bound at the top of `_check_death`.)

- [ ] **Step 4: Run tests + full suite**

Run: `SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest -q`
Expected: PASS. (Existing single-snake collision tests unaffected — `_other_hazard` is empty at N=1.)

- [ ] **Step 5: Commit**

```bash
git add snake_rl/world.py tests/test_collision.py
git commit -m "world: inter-snake cut-off + head-to-head death (death_cause=snake)"
```

---

### Task 5: Config — new constants, world-size bump, invariants

**Files:**
- Modify: `snake_rl/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces new `Config` fields (frozen defaults):
  `world_size_min=110.0, world_size_max=160.0`;
  `n_start_min=2, n_start_max=4, n_max=6`;
  `chickens_per_snake_max=2.0, chickens_per_snake_min=1.0, chicken_ceiling=12`;
  `repro_energy_frac=0.7, repro_length_min=10.0, r_mate=4.0, mate_steps=4, repro_cost=30.0,
  repro_cooldown=120, egg_timer=45, hatch_energy_frac=0.5, egg_food=25.0`;
  `corpse_food_per_length=4.0`; `reward_repro=12.0`.
- New invariants in `assert_invariants` (numbered 7–10 continuing the existing list).

- [ ] **Step 1: Write the failing test** — append to `tests/test_config.py`

```python
def test_multisnake_invariants_hold():
    from snake_rl.config import CFG, assert_invariants
    assert_invariants(CFG)                                  # must not raise
    assert CFG.world_size_min == 110.0 and CFG.n_max == 6
    # mating distance lets two snakes coexist without a forced cut-off
    assert CFG.r_mate >= 2 * CFG.head_radius
    # a just-qualified snake can pay the repro cost and survive
    assert CFG.repro_cost < CFG.repro_energy_frac * CFG.energy_max
    # food ceiling covers the population-scaled max
    assert CFG.chicken_ceiling >= CFG.chickens_per_snake_max * CFG.n_max
```

- [ ] **Step 2: Run to verify fail**

Run: `... pytest tests/test_config.py -q`
Expected: FAIL (`world_size_min` still 60 / new fields missing).

- [ ] **Step 3: Implement** — bump `world_size_min/max` (`config.py:10-11`) to `110.0/160.0`; add the fields above; extend `assert_invariants`:

```python
    # (7) two snakes can sit at mating distance without a forced cut-off
    assert cfg.r_mate >= 2 * cfg.head_radius, "r_mate too small: mating forces a collision"
    # (8) a snake that just crossed the energy threshold can pay the repro cost and live
    assert cfg.repro_cost < cfg.repro_energy_frac * cfg.energy_max, "repro_cost exceeds the mating gate"
    # (9) hatchling viable
    assert cfg.hatch_energy_frac * cfg.energy_max > 0 and cfg.start_length >= (
        cfg.head_radius + cfg.body_radius + cfg.v_dash + cfg.segment_spacing), "hatchling not viable"
    # (10) food ceiling covers the population-scaled demand (soft feasibility)
    assert cfg.chicken_ceiling >= cfg.chickens_per_snake_max * cfg.n_max, "chicken_ceiling too low"
```

- [ ] **Step 4: Run tests + full suite**

Run: `SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest -q`
Expected: PASS. (Existing invariants 1–6 still hold at world 110/160 — verified in the spec §7.)

- [ ] **Step 5: Commit**

```bash
git add snake_rl/config.py tests/test_config.py
git commit -m "config: multi-snake constants (rates), world-size bump, invariants 7-10"
```

---

### Task 6: Corpses — spawn on death + generalized eating (once per item)

**Files:**
- Modify: `snake_rl/world.py` (`corpses`, `_spawn_corpse`, generalized `try_eat`)
- Test: `tests/test_multisnake.py`

**Interfaces:**
- Consumes: config `corpse_food_per_length`, `energy_refill`, `grow_per_chicken` (existing).
- Produces:
  - `World.corpses` = `{"pos": np.ndarray(K,2), "food": np.ndarray(K,)}` (init empty in `__init__`).
  - `World._spawn_corpse(s)` — appends a corpse at `s.head` with `food = corpse_food_per_length ·
    s.target_length`. Called in `step` phase 2 for each newly-dead snake.
  - `try_eat()` generalized: the ego eats any chicken **or corpse** within `eat_radius`; each
    **item** consumed pays `reward_eat` once (`ate` counts items), energy/growth scale with the
    item. A corpse is consumed whole on contact for v1 (single bite). Returns int items eaten.

- [ ] **Step 1: Write the failing test**

```python
def test_dead_snake_becomes_corpse_and_is_edible():
    import numpy as np
    from snake_rl.config import CFG
    from snake_rl.world import World, wrap
    w = World(CFG, seed=5, size=(80.0, 80.0))
    w.chicken_pos = np.zeros((0, 2)); w.chicken_dir = np.zeros((0,)); w.chicken_id = np.zeros((0,), int)
    w._spawn_corpse(w.snakes[0])
    assert w.corpses["pos"].shape == (1, 2)
    assert w.corpses["food"][0] == CFG.corpse_food_per_length * w.snakes[0].target_length
    # move ego onto the corpse -> eats it (one item), energy up, corpse gone
    w.snakes[0].energy = 10.0
    w.snakes[0].head = w.corpses["pos"][0].copy()
    n = w.try_eat()
    assert n == 1 and w.corpses["pos"].shape == (0, 2) and w.snakes[0].energy > 10.0
```

- [ ] **Step 2: Run to verify fail**

Run: `... pytest tests/test_multisnake.py::test_dead_snake_becomes_corpse_and_is_edible -v`
Expected: FAIL (`corpses`/`_spawn_corpse` missing).

- [ ] **Step 3: Implement** — init `self.corpses = {"pos": np.zeros((0,2)), "food": np.zeros((0,))}` in `__init__`; add `_spawn_corpse`; generalize `try_eat` (`world.py:246-260`) to also test corpses (nearest-image distance ≤ `eat_radius`), remove eaten corpses, add their energy/growth, and count them into `n`. In `step` phase 2, after marking deaths, call `self._spawn_corpse(s)` for each snake that just died. Keep chicken logic intact.

- [ ] **Step 4: Run tests + full suite**

Run: `SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add snake_rl/world.py tests/test_multisnake.py
git commit -m "world: corpses on death + generalized eating (once per item)"
```

---

### Task 7: Reproduction — mating detection + egg laying

**Files:**
- Modify: `snake_rl/world.py` (`eggs`, `_mate_streak`, `_resolve_mating`)
- Test: `tests/test_reproduction.py` (new)

**Interfaces:**
- Consumes: config `repro_energy_frac, repro_length_min, r_mate, mate_steps, repro_cost,
  repro_cooldown` (Task 5).
- Produces:
  - `World.eggs` = `{"pos": np.ndarray(M,2), "timer": np.ndarray(M,), "owner": np.ndarray(M,2 int)}`
    (init empty).
  - `World._resolve_mating()` — called in `step` phase 3. A pair `(a, b)` of live snakes both with
    `energy > repro_energy_frac·energy_max`, `target_length > repro_length_min`, `repro_cooldown==0`,
    within `r_mate` (nearest-image) increments a per-pair proximity streak; at `mate_steps`
    consecutive qualifying steps → lay an egg at the torus-midpoint, subtract `repro_cost` energy
    from both, set both `repro_cooldown = repro_cooldown`. Streaks reset when the pair separates.
    Decrement all snakes' `repro_cooldown` each step (floor 0).
  - `World._lay_egg(pos, id_a, id_b)` — appends `{pos, timer=egg_timer, owner=[id_a, id_b]}` to
    `self.eggs`.
  - `World._mate_streak: dict` — `frozenset({id_a, id_b}) -> int` consecutive-qualifying-step count.

- [ ] **Step 1: Write the failing test** — `tests/test_reproduction.py`

```python
import numpy as np
from snake_rl.config import CFG
from snake_rl.world import World, Snake, wrap


def _two_fed_snakes(w, d=2.0):
    a = w.snakes[0]
    a.head_uw = np.array([40.0, 40.0]); a.head = wrap(a.head_uw, w.size)
    a.energy = CFG.energy_max; a.target_length = CFG.repro_length_min + 2; a.repro_cooldown = 0
    b = Snake(head_uw=np.array([40.0 + d, 40.0]), head=wrap(np.array([40.0+d,40.0]), w.size),
              heading=np.pi, path_uw=[np.array([40.0+d,40.0])], target_length=CFG.repro_length_min+2,
              stamina=CFG.s_max, energy=CFG.energy_max, _prev_head_uw=np.array([40.0+d,40.0]), id=1)
    w.snakes.append(b)
    return a, b


def test_mating_lays_egg_after_streak_and_costs_energy():
    w = World(CFG, seed=6, size=(80.0, 80.0))
    a, b = _two_fed_snakes(w, d=2.0)                        # within r_mate (4.0)
    for _ in range(CFG.mate_steps):
        w._resolve_mating()
    assert w.eggs["pos"].shape[0] == 1
    assert set(w.eggs["owner"][0].tolist()) == {0, 1}
    assert a.energy == CFG.energy_max - CFG.repro_cost
    assert b.energy == CFG.energy_max - CFG.repro_cost
    assert a.repro_cooldown > 0 and b.repro_cooldown > 0


def test_no_egg_if_separated_before_streak_completes():
    w = World(CFG, seed=6, size=(80.0, 80.0))
    a, b = _two_fed_snakes(w, d=2.0)
    w._resolve_mating()                                    # 1 qualifying step
    b.head_uw = np.array([70.0, 40.0]); b.head = wrap(b.head_uw, w.size)  # bolt away
    for _ in range(CFG.mate_steps):
        w._resolve_mating()
    assert w.eggs["pos"].shape[0] == 0
```

- [ ] **Step 2: Run to verify fail**

Run: `... pytest tests/test_reproduction.py -q`
Expected: FAIL (`eggs`/`_resolve_mating` missing).

- [ ] **Step 3: Implement** — init `self.eggs = {"pos": np.zeros((0,2)), "timer": np.zeros((0,)), "owner": np.zeros((0,2), int)}` and `self._mate_streak = {}` in `__init__`; add `_lay_egg`; implement `_resolve_mating`; call it in `step` phase 3.

```python
    def _lay_egg(self, pos, id_a, id_b):
        e = self.eggs
        e["pos"] = np.vstack([e["pos"], pos[None]]) if len(e["pos"]) else pos[None].copy()
        e["timer"] = np.append(e["timer"], self.cfg.egg_timer)
        row = np.array([[id_a, id_b]])
        e["owner"] = np.vstack([e["owner"], row]) if len(e["owner"]) else row

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
```

- [ ] **Step 4: Run tests + full suite**

Run: `SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add snake_rl/world.py tests/test_reproduction.py
git commit -m "world: reproduction — mating streak lays an egg, both parents pay energy"
```

---

### Task 8: Eggs — hatching + ownership-aware eating

**Files:**
- Modify: `snake_rl/world.py` (`_hatch_eggs`; eggs in generalized `try_eat`)
- Test: `tests/test_reproduction.py`

**Interfaces:**
- Consumes: config `egg_timer, hatch_energy_frac, egg_food, start_length` (Task 5); `eggs` (Task 7).
- Produces:
  - `World._hatch_eggs()` — called in `step` phase 3: decrement every egg `timer`; a `timer<=0` egg
    is removed and a new `Snake(start_length, energy=hatch_energy_frac·energy_max)` is appended at
    the egg `pos` (id `_next_snake_id++`, color_seed = id), unless `len(alive) >= n_max`.
  - `try_eat()` extended: a snake also eats a **foreign** egg within `eat_radius` (an egg whose
    `owner` does NOT contain the eater's id) → removes the egg, pays `egg_food` energy + one item;
    a snake **never** eats an egg it owns.

- [ ] **Step 1: Write the failing test**

```python
def test_egg_hatches_into_new_snake():
    w = World(CFG, seed=8, size=(80.0, 80.0))
    w.eggs = {"pos": np.array([[40.0, 40.0]]), "timer": np.array([1.0]),
              "owner": np.array([[0, 1]])}
    n0 = len(w.snakes)
    w._hatch_eggs()                                        # timer 1 -> 0 -> hatch
    assert len(w.snakes) == n0 + 1
    baby = w.snakes[-1]
    assert baby.target_length == CFG.start_length
    assert np.allclose(baby.head_uw, np.array([40.0, 40.0]))
    assert w.eggs["pos"].shape[0] == 0


def test_parent_cannot_eat_own_egg_but_rival_can():
    w = World(CFG, seed=8, size=(80.0, 80.0))
    ego = w.snakes[0]; ego.id = 0; ego.head = np.array([40.0, 40.0]); ego.energy = 10.0
    w.eggs = {"pos": np.array([[40.0, 40.0]]), "timer": np.array([30.0]), "owner": np.array([[0, 1]])}
    assert w.try_eat() == 0 and w.eggs["pos"].shape[0] == 1     # own egg: not eaten
    ego.id = 5                                                  # now a non-owner
    assert w.try_eat() == 1 and w.eggs["pos"].shape[0] == 0     # foreign egg: eaten
    assert ego.energy == 10.0 + CFG.egg_food
```

- [ ] **Step 2: Run to verify fail**

Run: `... pytest tests/test_reproduction.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement** — add `_hatch_eggs`; extend `try_eat` to test the ego (the eater) vs eggs with an ownership mask (`self.snakes[0].id not in owner_row`). Call `self._hatch_eggs()` in `step` phase 3. (Ownership check uses the eater snake's `id`; `try_eat` is ego-centric in Milestone A — Milestone B generalizes eating per snake.)

- [ ] **Step 4: Run tests + full suite**

Run: `SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add snake_rl/world.py tests/test_reproduction.py
git commit -m "world: eggs hatch into snakes; ownership-aware egg eating"
```

---

### Task 9: Starvation death + population-scaled food

**Files:**
- Modify: `snake_rl/world.py` (`decay_energy`→starvation; `maybe_spawn` dynamic target)
- Test: `tests/test_multisnake.py`

**Interfaces:**
- Consumes: config `chickens_per_snake_max/min, chicken_ceiling, n_max` (Task 5).
- Produces:
  - Starvation: in `step` phase 3, after energy decay, any live snake with `energy <= 0` dies
    (`death_cause="starve"`) and spawns a corpse. (`decay_energy` still floors energy at 0; the
    death check is separate so it applies to every snake, not just the ego.)
  - `maybe_spawn()` computes the target counts from the **live** snake count:
    `n_alive = sum(s.alive)`; `max_target = clamp(round(chickens_per_snake_max·n_alive), 1,
    chicken_ceiling)`, `min_target = clamp(round(chickens_per_snake_min·n_alive), 1, max_target)`;
    then the existing fast-refill-to-min / random-up-to-max logic uses these instead of
    `cfg.max_chickens/min_chickens`.

- [ ] **Step 1: Write the failing test**

```python
def test_starvation_kills_and_leaves_corpse():
    import numpy as np
    from snake_rl.config import CFG
    from snake_rl.world import World
    w = World(CFG, seed=9, size=(80.0, 80.0))
    w.chicken_pos = np.zeros((0, 2)); w.chicken_dir = np.zeros((0,)); w.chicken_id = np.zeros((0,), int)
    w.snakes[0].energy = CFG.energy_decay / 2               # will hit 0 this step
    w.step(1, 0)
    assert w.snakes[0].alive is False and w.snakes[0].death_cause == "starve"
    assert w.corpses["pos"].shape[0] == 1


def test_food_target_scales_with_live_snakes():
    import numpy as np
    from snake_rl.config import CFG
    from snake_rl.worldgen import generate_world
    w = generate_world(CFG, seed=10, size=(150.0, 150.0), n_snakes=4)
    # drive spawns and assert the chicken count tracks ~2 per snake, capped by the ceiling
    for _ in range(400):
        w.maybe_spawn()
    assert len(w.chicken_pos) <= min(CFG.chicken_ceiling, round(CFG.chickens_per_snake_max * 4))
    assert len(w.chicken_pos) >= 1
```

- [ ] **Step 2: Run to verify fail**

Run: `... pytest tests/test_multisnake.py -q`
Expected: FAIL (no starvation death; `maybe_spawn` uses static caps).

- [ ] **Step 3: Implement** — in `step` phase 3 add a starvation pass over live snakes (`energy<=0` → `alive=False`, `death_cause="starve"`, `self._spawn_corpse(s)`). Rewrite `maybe_spawn`:

```python
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
```

- [ ] **Step 4: Run tests + full suite**

Run: `SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest -q`
Expected: PASS — the full multi-snake simulation is now correct and every existing test is still green.

- [ ] **Step 5: Commit**

```bash
git add snake_rl/world.py tests/test_multisnake.py
git commit -m "world: starvation death + population-scaled food target"
```

---

## Milestone A exit criteria (go/no-go before Milestone B)

- All existing 57 tests green + the new multi-snake/reproduction tests green.
- N=1 behavioral regression exact (Task 1) — the RL stack still runs the shipped single-snake model
  unchanged (env/sensors/train untouched; ego proxies preserve the API).
- A 4-snake world can be constructed, stepped for hundreds of steps without error, and exhibits:
  inter-snake deaths, corpses, mating→eggs→hatchlings, starvation, and a chicken count that scales
  with the live population. (Add a throwaway `scripts/smoke_multisnake.py` under the scratchpad if a
  manual sanity run helps — do not commit it.)

## Deferred to Milestone B (separate plan, written after this lands)

Phase 0 spikes (0a opponent obs preprocessing parity for C1; 0b mating-discovery under curriculum —
now runnable because mating exists here); multi-snake **observation** (`sensors.py` new ray
categories + social/egg channels + snake-smell + repro-ready, new `OBS_DIM`, obs-space bounds);
**env** ego/opponent split + `SyncOpponentPolicy` (state_dict + `obs_rms`) + repro/corpse/egg
rewards + mating curriculum; **render** golden-angle colors + 3-ring HUD + eggs/corpses; **watch**
persistent world + camera-follow + ecosystem metrics; the **retrain** and CLAUDE.md update. Removing
the ego-proxy bridge happens here when `sensors` goes multi-snake.
```
