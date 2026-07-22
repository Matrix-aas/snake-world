# Thinking-Snake v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development to implement
> this plan task-by-task with a code review after each task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Snakes navigate by perception + physics, not by dying to scenery — obstacles become solid
(slide + dash-stun, non-lethal), snakes gain graduated speed (emergent ambush/stalk/threading),
finer forward sight, and prey that reacts to a snake's speed. One from-scratch retrain.

**Architecture:** Physics move to `world.py` (solid-slide collision response, stun, speed-scaled prey
alert); the whole Pitfall-16 obstacle-reward machinery is deleted; `sensors.py` grows forward rays +
an un-masked obstacle-clearance channel + a proprio speed bit (OBS_DIM 87→113); the action space
gains a speed dimension (`MultiDiscrete([4,3,2])`). Egg-arrivals + sky-chickens already shipped.

**Tech Stack:** SB3 PPO (MlpPolicy, CPU, MultiDiscrete), numpy, pygame, Gymnasium, pytest.

## Global Constraints (verbatim from spec + review fixes)

- Death causes after this change: **`starve`, `snake` only** (no `obstacle`, no `self`).
- Model **A**: obstacles + own body = solid-slide non-lethal; **rival body/head = lethal** (cut-off).
- No hardcoded action gates — stun is a *consequence*, the snake stays free to dash anywhere.
- `OBS_DIM 87 → 113`; `action_space = MultiDiscrete([4,3,2])` = speed×steer×dash. Old "full-speed
  straight" ⇒ **speed_idx = 3**.
- **Ray count is fixed at `RAY_COUNT = n_rays + n_fwd_rays = 9 + 2 = 11`** — 9 uniform over ±135°
  plus 2 forward at ±(9-ray half-spacing = 33.75/2 = 16.875°). Derive the forward offset from the
  **9-ray** spacing, NEVER a mutated `n_rays`. Size every vision array by `RAY_COUNT` (review I1).
- **Prey alert scales with speed, CAPPED at 1× base:** `base_alert · clip(speed/v_snake, 0, 1)` — a
  dash is as alarming as full cruise, never more (keeps max alert = today's `r_flee`, so
  hunting-bootstrap + invariant #3 hold; review I3).
- Run tests: `SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest -q`. Always
  `.venv/bin/python`. No commit-attribution lines. Update `CLAUDE.md` in the same change.
- Coupling: obs-shape + action-shape changes ripple across `sensors.py`, `env.py`, `selfplay.py`,
  `watch.py`, and many tests atomically — the suite only returns green after Tasks 2+4+5+6 land.
  A subagent per task may commit an intermediate red state; the suite must be green by end of Task 7.

---

### Task 1: Config — motion/collision constants in, Pitfall-16 reward constants out

**Files:** Modify `snake_rl/config.py`; Test `tests/test_config.py`.

**Produces:** `cfg.speed_levels: tuple`, `cfg.stun_steps: int`, `cfg.n_fwd_rays: int`. **Removes:**
`reward_death_obstacle`, `obs_avoid_weight`, `obs_avoid_range` (currently on disk from an earlier change).

- [ ] Under snake-motion: `speed_levels: tuple = (0.0, 1/3, 2/3, 1.0)  # cruise fractions of v_snake; dash overrides`.
- [ ] Under snake-motion: `stun_steps: int = 10  # dash into a solid -> frozen this many steps ("head spinning")`.
- [ ] Under sensing (near `n_rays`): `n_fwd_rays: int = 2  # extra forward rays; RAY_COUNT = n_rays + n_fwd_rays = 11`.
- [ ] Delete the three Pitfall-16 constants + their comment block.
- [ ] `assert_invariants`: add `assert cfg.stun_steps >= 1`;
  `assert cfg.speed_levels[0] == 0.0 and cfg.speed_levels[-1] == 1.0`; `assert cfg.n_fwd_rays >= 0`.
  Invariant #3 (catch feasibility) is UNCHANGED and still valid (alert cap keeps max alert = `r_flee`).
- [ ] `test_config.py`: assert the three new constants exist + sane; assert removed ones gone
  (`not hasattr(CFG, "obs_avoid_weight")`). Run `pytest tests/test_config.py -q`.

---

### Task 2: World motion & collision rework (`world.py`) — the core

**Files:** Modify `snake_rl/world.py`; Test `tests/test_collision.py`, `test_snake_motion.py`,
`test_multisnake.py`.

**Produces:** `Snake.stun:int`, `Snake.speed:float`; `_move_snake(s, speed_idx, steer, dash)->bool`;
`World.step(speed_idx, steer, dash, opponent_fn)` with `opponent_fn(world,s)->(speed_idx,steer,dash)`;
`_death_cause(s)` returns `"snake"|None`; `_slide(p0, disp, centers, radii)->(pos, hit)`.
**Consumes:** `cfg.speed_levels`, `cfg.stun_steps`.

**Slide algorithm (verified correct by review — implement exactly):**

```python
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
    ln = np.linalg.norm(n)
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
```

- [ ] Add `stun: int = 0` and `speed: float = 0.0` to `Snake`.
- [ ] Rewrite `_move_snake(self, s, speed_idx, steer, dash)`:
  - **Stunned:** `if s.stun > 0:` → `s.stun -= 1; s.dashed = False; s.speed = 0.0;
    s.stamina = min(c.s_max, s.stamina + c.stamina_regen)` (regen while dizzy — review M1);
    `s._prev_head_uw = s.head_uw.copy(); s.path_uw.append(s.head_uw.copy()); self._prune_path(s);
    s.steps += 1; return False`  (steering frozen too).
  - Steering as today. `dashing = bool(dash) and s.stamina >= c.dash_min_stamina`.
    `speed = c.v_dash if dashing else c.speed_levels[speed_idx] * c.v_snake`. `s.speed = speed`.
  - `disp = speed * s.heading_vec()`. Solids = obstacles (`obstacle_pos`, `obstacle_r + head_radius`)
    **+ own body** (`_body_points_uw(s)` — carries the Pitfall-5 neck-skip, radii `body_radius +
    head_radius`). `new, hit = self._slide(prev_uw, disp, solid_centers, solid_radii)`.
  - `if hit and dashing: s.stun = c.stun_steps`.
  - `s.head_uw = new; s.head = wrap(new, self.size); path append; prune; stamina drain(if dashing)/
    regen(else) as today; steps+=1; s._prev_head_uw = prev_uw; s.dashed = dashing`. Return `dashing`.
- [ ] `_death_cause(s)`: **remove obstacle + self branches**; keep ONLY the rival branch (`"snake"`),
  swept over `s._prev_head_uw -> s.head_uw` (the slid path).
- [ ] `World.step(self, speed_idx, steer, dash, opponent_fn=None)`: default `opponent_fn` →
  `(1, 1, 0)` (⅓-cruise straight); collect opp actions as 3-tuples; call `_move_snake` with them.
- [ ] `move(self, speed_idx, steer, dash)` proxy updated.
- [ ] **Delete contradicted tests** (review I4) in `test_collision.py`: `test_head_into_obstacle_kills`,
  `test_self_collision_when_curled`, `test_constant_turning_eventually_self_collides`; **rewrite**
  `test_dash_tunneling_still_kills` → dash into an obstacle sets `stun == stun_steps` (no death).
- [ ] **New tests** (`test_collision.py`): (a) walk into an obstacle → **not dead**, ends adjacent to
  the surface **and `snake.stun == 0`** (review M2); (b) dash into an obstacle → `stun == stun_steps`,
  not dead; (c) a stunned snake does not translate for the next `stun_steps` `_move_snake` calls;
  (d) head curled onto its **own** body → not dead; (e) head into a **rival** body still dies (`"snake"`).
  (f) **Pitfall-5 regression** (review M3): a full-length snake going straight (speed_idx=3) advances
  its head by exactly `v_snake·heading` with no tangential deflection off its own neck.
- [ ] Update `test_snake_motion.py` / `test_multisnake.py` to the new signatures — **map old
  full-speed straight to `speed_idx=3`** (review I2): `move(3, 1, 0)` advances `v_snake`;
  `test_dash_ignored_when_stamina_empty` must pass `speed_idx=3` so the stamina-gated-off dash falls
  back to `v_snake` (not a lower cruise). Run these three test files.

---

### Task 3: Prey senses motion (`world.py` chicken FSM)

**Files:** Modify `snake_rl/world.py` (`update_chickens`, `_chicken_step`); Test `tests/test_chickens.py`.

**Consumes:** `Snake.speed`.

- [ ] `update_chickens`: gather per-live-snake `(head, speed)`; pass speeds into `_chicken_step`.
- [ ] `_chicken_step`: a near snake alerts the chicken only if `d < base_alert *
  clip(snake.speed / c.v_snake, 0.0, 1.0)`, `base_alert = r_flee_peck if pecking else r_flee` (review
  I3 cap). Speed 0 ⇒ 0 ⇒ never alerts. Keep the repulsion-resultant over the snakes that DO alert.
- [ ] **Fix layout-dependent test** (review I4): `test_chickens.py:191` slice
  `observe(w)[:63].reshape(9,7)[:,2]` → `[:88].reshape(11,8)[:,2]` (new vision block).
- [ ] New test: a **stopped** (`speed=0`) snake 1 unit from a chicken does NOT trigger flee; the same
  snake at full cruise within `r_flee` does. Run `pytest tests/test_chickens.py -q`.

---

### Task 4: Sensors — forward rays + un-mask + proprio speed + spawn-egg fix (`sensors.py`)

**Files:** Modify `snake_rl/sensors.py`; Test `tests/test_sensors.py`.

**Produces:** `OBS_DIM = 113`; a `RAY_COUNT`-wide × 8 ray block; 5-float proprio. **Consumes:** `Snake.speed`.

- [ ] Define `RAY_COUNT` from cfg (`c.n_rays + c.n_fwd_rays` = 11). `ray_dirs(cfg, heading)`: build 9
  uniform angles over `±fov/2` **plus** 2 forward at `heading ± radians((fov_deg/(n_rays-1))/2)` (the
  9-ray half-spacing, 16.875°). Return the 11-direction array. **Do not mutate `cfg.n_rays`.**
- [ ] Size every vision array by `RAY_COUNT`, not `c.n_rays` (fix `sensors.py:53,54,77` `np.full` /
  `np.tile` and the `dist`/`kinds` arrays in `_scan`).
- [ ] Add `_obstacle_clearance(world, head, heading) -> (RAY_COUNT,)`: an obstacle-ONLY raycast
  (`world.obstacle_pos`, radii `+head_radius`), returning each ray's nearest-obstacle distance
  (`ray_range` if none). Reuse the `_scan` projection math.
- [ ] `sense_vision`: append `obstacle_clearance / ray_range` as the 8th feature per ray → `(11, 8)`.
- [ ] `_egg_channel`: filter to `owner[:,0] >= 0` before nearest-egg selection (arrival eggs uneatable).
- [ ] `observe`: proprio 4→5 — append `np.clip(snake.speed / c.v_dash, 0.0, 1.0)`.
- [ ] `OBS_DIM = 113`; update the module docstring layout comment.
- [ ] **Fix tests** (review I4) `test_sensors.py`: `OBS_DIM==87`→`113`; `reshape(9,7)`→`reshape(11,8)`;
  the empty-ray literal `[1,0,0,0,0,0,0]`→8-wide `[1,0,0,0,0,0,0,0]` (or adapt the assertion).
- [ ] **New tests:** (a) `len(ray_dirs(CFG,0.0))==11` and the 2 forward angles ≈ ±16.875°;
  (b) `observe(...).shape==(113,)`; (c) **un-mask**: a chicken directly ahead + a rock just beyond it
  on the same forward ray → that ray `is_chicken==1` (nearest) BUT `obstacle_clearance` reports the
  rock (< ray_range); (d) an arrival egg (owner `[-1,-1]`) nearest → `has_egg==0`; a real foreign egg
  → `has_egg==1`. Run `pytest tests/test_sensors.py -q`.

---

### Task 5: Env — action space, obs bounds, reward simplification (`env.py`)

**Files:** Modify `snake_rl/env.py`; Test `tests/test_env.py`.

**Consumes:** `OBS_DIM=113`, `World.step(speed,steer,dash,...)`.

- [ ] `action_space = spaces.MultiDiscrete([4, 3, 2])`.
- [ ] `_make_observation_space`: rewrite for the 113 layout — vision `0:88` (11×8, `[0,1]`);
  social `88:95` (signed rel/heading `[89:93]`→±1); egg `95:99` (rel `[96:98]`→±1);
  smell `99:108` (ceil/nmax bounds at `99..107` per the existing pattern);
  proprio `108:113` (`[0,1]`). `assert OBS_DIM == 113`.
- [ ] `step(action)`: `speed, steer, dash = int(action[0]), int(action[1]), int(action[2])`;
  `self.world.step(speed, steer, dash, opponent_fn=lambda w,s: self._opp.act(w,s))`.
- [ ] **Remove** `_phi_obstacle`, `_last_phi_obs` (both `__init__` + `reset`), the obstacle term in
  `_shaping`, and the now-unused `from .world import torus_dist` import (currently on disk). Reward on
  `terminated`: flat `reward += c.reward_death` (no cause branch — `reward_death_obstacle` is gone).
- [ ] **Delete contradicted tests** (review I4): `test_obstacle_pbrs_well_and_steer_away`,
  `test_obstacle_death_costs_more_than_other_deaths`, `test_pbrs_closed_loop_...`/
  `test_pbrs_zeroes_on_set_change...` if they reference removed obstacle plumbing (keep the
  chicken-PBRS ones, retargeted). Update `test_eat_gives_positive_reward` + any `env.step([...])`
  calls to the **3-dim** action. Keep `check_env` (single + multi). Run `pytest tests/test_env.py -q`.

---

### Task 6: Selfplay + watch action wiring — 3-dim action, 113 obs (`selfplay.py`, `watch.py`)

**Files:** Modify `snake_rl/selfplay.py`, `snake_rl/watch.py`; Test `tests/test_selfplay.py`.

- [ ] `selfplay.py`: the policy skeleton's action space `MultiDiscrete([3,2])` → `MultiDiscrete([4,3,2])`
  (`selfplay.py:26` — 5→9 action logits, else `load_state_dict` mismatches the trained action net).
  `act` returns a 3-tuple `(speed, steer, dash)`. Frame-ring width follows `OBS_DIM` (verify no
  hardcoded 87/75).
- [ ] **`watch.py::_step_world` (review C1 — critical):** `snake_rl/watch.py:102-103` →
  `speed, steer, dash = controller.act(world, ego) if ego.alive else (1, 1, 0)` and
  `world.step(speed, steer, dash, opponent_fn=lambda w, s: controller.act(w, s))`. (Sole tick path for
  the viewer AND `run_headless` — Task 8 depends on it.)
- [ ] **Fix tests** (review I4) `test_selfplay.py:79,114,130,132,145`: `env.step([1,0])`→`env.step([1,1,0])`
  (or `[3,1,0]` where full-speed matters); update the parity test's action arity + obs width.
- [ ] Run `pytest tests/test_selfplay.py -q`.

---

### Task 7: Full-suite sweep + CLAUDE.md

**Files:** any remaining tests; `snake_rl/watch.py` (cosmetic); `CLAUDE.md`.

- [ ] Run the **full** suite; fix remaining obs-dim/action-shape breakage. Known: `test_watch_smoke.py:86`
  stub `spaces.MultiDiscrete([3,2])`→`[4,3,2]`; `test_train_smoke`, `test_render_smoke`, `test_interp`.
  Target: all green.
- [ ] (review M4, cosmetic) `watch.py:~206` headless-eval `deaths` dict: drop `"obstacle"`/`"self"`
  keys (always 0 now) or leave the guarded `if cause in deaths`. Trim for clarity.
- [ ] CLAUDE.md: architecture rows (world collision/motion, action, obs), RL-design bullets (obs 113,
  `MultiDiscrete([4,3,2])`, solid-slide+stun, graduated speed, prey-senses-motion + the alert cap),
  config + invariants, judging (deaths `starve`/`snake` only), Pitfalls — **rewrite Pitfall 16**
  (obstacles no longer lethal; the reward saga retired; one paragraph on *why physics replaced reward*
  — the 71%-dash-commit diagnostic), add pitfalls for solid-slide+stun and graduated-speed/prey-motion
  cap. Note OBS_DIM history 87→113.
- [ ] Commit (green suite): `git commit -m "feat: solid-slide obstacles + graduated speed + forward/un-masked sight (thinking-snake v2)"`.

---

### Task 8: Retrain from scratch + judge

- [ ] `cp -r models models_v1_backup`.
- [ ] `./snake train --steps 8000000 --envs 16 --reset` (bigger action/obs may want more; judge by
  metrics). **The env has killed long background trainings twice — run foreground via `!` or accept
  eval-of-latest-checkpoint.**
- [ ] **Bootstrap monitor (review I3):** confirm `snake/eaten_per_window` climbs within the first
  ~50–100k steps. If flat near 0, the speed-scaled prey alert is blocking discovery — abort and
  reconsider the cap/curriculum before burning the full run.
- [ ] Headless eval: deaths `starve`/`snake` only (no obstacle/self); catch 10–14/1000; ambush
  visible (snakes at speed 0 near prey); population 2–4; births+kills present.
- [ ] Watch: chicken between rocks → snake slows/threads or gives up, does NOT crash; dash into a wall
  → brief dizzy stun, then continues.

## Self-Review (post-fix)

- **Review C1** (watch.py) → Task 6. **I1** (ray count) → Task 1 (`n_fwd_rays`) + Task 4 (RAY_COUNT
  construction + sizing). **I2** (speed→idx3) → Task 2. **I3** (alert cap + invariant) → Global +
  Task 1 + Task 3 + Task 8 monitor. **I4** (contradicted tests) → enumerated in Tasks 2/3/4/5/6/7.
  **M1** stun regen → Task 2. **M2** stun==0 → Task 2. **M3** straight-advance → Task 2. **M4** deaths
  dict → Task 7.
- **Type consistency:** `_move_snake(s, speed_idx, steer, dash)`, `World.step(speed_idx, steer, dash,
  opponent_fn)`, `opponent_fn/act -> (speed, steer, dash)`, `OBS_DIM=113`, `RAY_COUNT=11` consistent
  across tasks.
- **Placeholder scan:** slide algorithm is full code; every test/edit names exact files/lines. OK.
