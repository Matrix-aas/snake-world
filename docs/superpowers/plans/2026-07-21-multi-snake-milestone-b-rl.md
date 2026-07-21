# Multi-snake — Milestone B (RL: obs + self-play + render + retrain) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Milestone-A multi-snake *simulation* into a trained, watchable multi-snake *ecosystem* — snakes perceive each other + eggs, one shared PPO brain is trained by self-play (opponents driven by a policy snapshot), and a persistent viewer shows N distinctly-colored snakes with compact ring HUDs, eggs, and corpses.

**Architecture:** Per-snake world dynamics (eating/energy/starvation for every snake, not just the ego); a redesigned 75-float egocentric observation (`OBS_DIM 42→75`); a self-play env where the ego is the SB3 learner and opponents are stepped in-env from a policy snapshot with the **exact** `VecNormalize`+`VecFrameStack` preprocessing proven in spike 0a; reward for reproduction/corpse/egg; a mating curriculum with an auto-lay fallback; render + persistent viewer; a from-scratch retrain.

**Tech Stack:** Python 3.13 (`.venv`), numpy≥2, PyTorch (CPU), stable-baselines3, gymnasium, pygame.

## Global Constraints

- Run tests: `SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest -q` (all Milestone-A tests + new ones green after each task).
- **This milestone REQUIRES a from-scratch retrain** (`--reset`) — obs, reward, and physics all change. The shipped `models/` snake is already a placeholder (see CLAUDE.md WIP note). Back up nothing new; `models_good_backup/` holds the original single-snake model.
- **Torus nearest-image geometry everywhere** — reuse `world.torus_delta / torus_dist / ray_circle_hit / segment_circle_hit`.
- **Opponent obs preprocessing (spike 0a, PROVEN — reproduce EXACTLY):** an opponent's action = `policy(preprocess(ring_buffer))` where `preprocess = clip((stack(frames) - obs_rms.mean) / sqrt(obs_rms.var + epsilon), -clip_obs, clip_obs)`, `epsilon=1e-8`, `clip_obs=10.0` (read from the pushed `VecNormalize`, don't hardcode). `obs_rms.mean/var` are over the **stacked** vector (shape `(OBS_DIM*4,)`). Frame-stack: newest frame **last** (`[-OBS_DIM:]`), oldest first; a normal step does `roll(buf, -OBS_DIM)` then overwrites the last `OBS_DIM`; **on a snake's death/respawn the ring buffer is ZEROED, not rolled** (a stale pre-death frame must not leak). `normalize_obs` is pure, so a pushed `(mean, var)` snapshot per rollout is sufficient.
- **All obs channels normalized/clipped** so obs is size-agnostic across the 110–160 world range (rel-positions `/ray_range` clipped to `[-1,1]`; presence bits; smell bounds from `chicken_ceiling`/`n_max`). See spec §4.
- **Reward stays sparse** (Pitfalls in CLAUDE.md): `+reward_eat` per item (chicken/corpse/foreign-egg), `+reward_repro` on hatch of your egg, `+reward_death`, unchanged PBRS to nearest chicken. No dense cooperation shaping.
- **PPO hyperparameters unchanged from the single-snake recipe** (constant lr `3e-4`, `n_steps 1024`, `batch_size 256`, `n_epochs 10`, `gae_lambda 0.95`, `ent_coef 0.01`, `target_kl 0.03`, net `[128,128]`, `gamma` from CFG) — already tuned for the non-stationary task self-play is. See spec §8 + CLAUDE.md Pitfalls 3/5/6.
- No Claude attribution in commits.
- TDD: every code task = failing test → run (fail) → minimal impl → run (pass) → commit. RL-plumbing tasks that can't be unit-TDD'd get a runnable smoke assert instead (stated per task).

## File Structure

- **Modify `snake_rl/world.py`** — per-snake `try_eat`/`decay_energy` (all snakes eat/decay, not just ego); prune non-ego dead snakes for the persistent world; keep ego = `snakes[0]`.
- **Modify `snake_rl/sensors.py`** — `observe(world, snake=None)` per-snake; new ray categories, social/egg channels, snake-smell, repro-ready; `OBS_DIM 42→75`.
- **Create `snake_rl/selfplay.py`** — the opponent policy runner: frame-ring preprocessing (0a recipe) + `predict`, decoupled from SB3 wrappers. One clear responsibility, kept out of `env.py`.
- **Modify `snake_rl/env.py`** — obs-space bounds for `OBS_DIM=75`; ego/opponent step split (opponents act via `selfplay`); reward for repro/corpse/egg; `set_opponent_policy(state_dict, obs_rms)`; extend `set_hardness` with the mating curriculum.
- **Modify `snake_rl/train.py`** — `SyncOpponentPolicy` callback (push `state_dict`+`obs_rms` each rollout); retrain recipe; keep `AnnealHardness`.
- **Modify `snake_rl/config.py`** — mating-curriculum easy values; any new reward/obs constants.
- **Modify `snake_rl/render.py`** — golden-angle colors, per-snake 3-ring HUD, eggs, corpses.
- **Modify `snake_rl/watch.py`** — persistent world decoupled from ego death; camera-follow; multi-snake render; headless ecosystem metrics (births/kills/starvations/population + all death causes).

---

### Task B1: Per-snake world dynamics (eating, energy, starvation, pruning)

Make the ecosystem close: every snake eats/grows/starves, not just the ego, so opponents can grow (→ reach `repro_length_min`, mate) and die. Prune non-ego dead snakes so the persistent world (B5) doesn't accumulate them (spec C2; final-review Important #2).

**Files:**
- Modify: `snake_rl/world.py`
- Test: `tests/test_multisnake.py`

**Interfaces:**
- Consumes: `World.snakes`, `try_eat` (ego-only from Milestone A), `decay_energy` (ego-only), `_spawn_corpse`.
- Produces:
  - `World.try_eat() -> int` — now processes eating for EVERY live snake (each eats chickens/corpses/foreign-eggs near ITS OWN head via `torus_dist(items, s.head, size) <= eat_radius`, applying that snake's growth/energy). **Returns the EGO's (`snakes[0]`) eaten-item count** (env reward reads this; back-compat). Ownership for eggs uses each eater's own `id`.
  - `World.decay_energy()` — decays EVERY live snake's energy (was ego-only).
  - Starvation pass in `step` phase 3 already loops all live snakes (Milestone A) — now fires for opponents too since they decay.
  - `World._prune_dead()` — called at the end of `step`: remove dead snakes EXCEPT the ego (`snakes[0]` must persist for the proxy + env terminated logic). Keeps `snakes[0]` even when `alive=False`.

- [ ] **Step 1: Write failing tests** — `tests/test_multisnake.py`

```python
def test_all_snakes_eat_and_decay_not_just_ego():
    import numpy as np
    from snake_rl.config import CFG
    from snake_rl.worldgen import generate_world
    w = generate_world(CFG, seed=11, size=(140.0, 140.0), n_snakes=3)
    w.chicken_pos = np.zeros((0,2)); w.chicken_dir = np.zeros((0,)); w.chicken_id = np.zeros((0,), int)
    opp = w.snakes[1]
    # place a chicken on the OPPONENT's head; opponent must eat it (grow), not just the ego
    opp.energy = 10.0; before_len = opp.target_length
    w.chicken_pos = opp.head[None].copy(); w.chicken_dir = np.zeros(1); w.chicken_id = np.array([999])
    w.try_eat()
    assert opp.energy > 10.0 and opp.target_length >= before_len
    # decay hits opponents too
    e = opp.energy; w.decay_energy(); assert opp.energy == max(0.0, e - CFG.energy_decay)


def test_dead_opponent_is_pruned_ego_kept():
    import numpy as np
    from snake_rl.config import CFG
    from snake_rl.worldgen import generate_world
    w = generate_world(CFG, seed=12, size=(140.0,140.0), n_snakes=3)
    w.snakes[2].alive = False; w.snakes[2].death_cause = "snake"
    w._prune_dead()
    assert all(s.alive for s in w.snakes[1:]) and len(w.snakes) == 2
    # ego is kept even when dead (proxy invariant)
    w.snakes[0].alive = False; w._prune_dead()
    assert w.snakes[0] is not None and len(w.snakes) == 2 and w.snakes[0].alive is False
```

- [ ] **Step 2: Run to verify fail** — `... pytest tests/test_multisnake.py -q` → FAIL (`try_eat`/`decay_energy` ego-only; no `_prune_dead`).

- [ ] **Step 3: Implement** — generalize `try_eat` to loop over live snakes (factor the current single-snake eat body into a `_snake_eat(s) -> int` that eats items near `s.head` with `s`'s id for egg-ownership; sum is discarded, return `_snake_eat(self.snakes[0])`). Generalize `decay_energy` to loop live snakes. Add `_prune_dead` (`self.snakes = [self.snakes[0]] + [s for s in self.snakes[1:] if s.alive]`); call it at the end of `step`. Keep the corpse/egg/chicken array-consistency discipline.

- [ ] **Step 4: Run tests + full suite** → all green (existing single-snake tests: at N=1, `try_eat`/`decay_energy` still act on `snakes[0]` identically; `_prune_dead` on a 1-snake world is a no-op).

- [ ] **Step 5: Commit** — `world: per-snake eating/energy/starvation + prune dead opponents`

---

### Task B2: Multi-snake observation (`sensors.py`, `OBS_DIM 42→75`)

Per-snake egocentric obs so a snake reads rivals + eggs. **Retrain-critical.** Layout per spec §4.

**Files:**
- Modify: `snake_rl/sensors.py`, `snake_rl/env.py` (obs-space bounds)
- Test: `tests/test_sensors.py`

**Interfaces:**
- Produces: `sensors.observe(world, snake=None) -> np.ndarray(75)` (default `snake = world.snakes[0]`). `sensors.OBS_DIM = 75`. Layout (all channels normalized/clipped, spec §4):
  - rays `9 × 6` = 54: `[dist, is_obstacle, is_chicken, is_self, is_other_body, is_egg]` (Minkowski-inflated by `head_radius`).
  - social `7`: `[has_rival, rel_pos_fwd, rel_pos_left, rival_heading_fwd, rival_heading_left, size_ratio, rival_is_dashing]` (rel-pos `/ray_range` clipped `[-1,1]`; egocentric to `snake`).
  - egg `4`: `[has_egg, rel_pos_fwd, rel_pos_left, is_mine]` (nearest egg; `is_mine` from `owner` vs `snake.id`).
  - smell `6`: `[chicken_intensity, chicken_grad_fwd, chicken_grad_left, snake_intensity, snake_grad_fwd, snake_grad_left]`.
  - proprio `4`: `[energy/energy_max, length/length_cap, stamina/s_max, repro_ready]`.
- `env.observation_space` low/high updated: indices in `[0,1]` except rel-pos in `[-1,1]` and smell (intensity `[0, ceiling]`, gradients `[-ceiling, ceiling]` where ceiling = `chicken_ceiling` for chicken-smell, `n_max` for snake-smell).

- [ ] **Step 1: Write failing tests** — `tests/test_sensors.py` (append). Assert: `OBS_DIM == 75`; `observe(world)` returns shape `(75,)` and lies within the declared bounds; a rival placed dead-ahead sets `has_rival==1` and `rel_pos_fwd>0`, `|rel_pos|<=1`; a self-owned egg ahead sets `has_egg==1, is_mine==1`; a foreign egg sets `is_mine==0`; `is_other_body` ray fires when a rival body crosses a ray and `is_self`/`is_chicken` stay correct; `repro_ready` is 1 only when above both thresholds and off cooldown. (Write concrete coordinate-based cases like the existing `test_sensors.py`.)

- [ ] **Step 2: Run to verify fail** → FAIL (OBS_DIM 42, no channels).

- [ ] **Step 3: Implement** — extend `sensors._scan` to classify other-snake bodies + eggs as ray hit-types (add them to the raycast target set, Minkowski `+head_radius`); add `_social(world, snake)`, `_egg_channel(world, snake)`, extend `smell(world, snake)` with a snake field, add `repro_ready` to proprio; parametrize everything by `snake`. Bump `OBS_DIM`. Update `env.observation_space` bounds. Keep single-image torus geometry.

- [ ] **Step 4: Run tests + full suite** → new sensors tests green; NOTE existing `test_env.py::check_env` and any obs-shape assertions will need updating to 75 (this is intended — the obs changed; update them to the new dim, don't weaken). If a test asserted the old 42, change it to 75 and keep the real check.

- [ ] **Step 5: Commit** — `sensors: multi-snake 75-float obs (rival/egg rays, social/egg channels, snake-smell, repro-ready)`

---

### Task B3: Self-play env + opponent policy runner + rewards + curriculum

The core RL task. Ego = SB3 learner; opponents stepped in-env from a policy snapshot using the **proven 0a preprocessing**. New sparse rewards. Mating curriculum.

**Files:**
- Create: `snake_rl/selfplay.py`
- Modify: `snake_rl/env.py`, `snake_rl/train.py`, `snake_rl/config.py`
- Test: `tests/test_selfplay.py` (new), `tests/test_env.py`

**Interfaces:**
- `selfplay.OpponentController` — holds a torch policy clone + `(mean, var, clip_obs, epsilon)` and a per-opponent `dict[snake_id -> ring_buffer(4, OBS_DIM)]`.
  - `.sync(state_dict, obs_rms)` — load weights + normalization snapshot (called by the env when the callback pushes).
  - `.act(world, snake) -> (steering, dash)` — `obs = observe(world, snake)`; update that snake's ring buffer (roll+write, or **zero+write if the snake is newly (re)spawned**); `x = preprocess(ring)` (0a recipe, verbatim); `action = policy.predict(x, deterministic=False)`; return decoded MultiDiscrete.
  - `.reset_snake(snake_id)` — zero that snake's ring buffer (call on hatch/death-respawn).
- `env.SnakeEnv`:
  - `set_opponent_policy(state_dict, obs_rms)` — forwards to the `OpponentController`.
  - `step`: ego acts from the SB3 action; opponents act via `OpponentController.act`; reward = `reward_eat*ego_ate + reward_repro*ego_hatched_this_step + (death) + PBRS + step_penalty` (spec §6). `ego_hatched_this_step` = an egg the ego owns hatched this step (track owner→reward).
  - `set_hardness(h)` extended: also interpolate the **mating curriculum** — easy `r_mate_easy`(large)/`mate_steps_easy`(1)/`repro_length_min_easy`(low) → hard real values.
- `train.SyncOpponentPolicy(BaseCallback)` — every rollout end, `env_method("set_opponent_policy", policy.state_dict()-as-numpy, venv.obs_rms)` (mirrors `AnnealHardness`).
- `config` — `r_mate_easy, mate_steps_easy, repro_length_min_easy`, plus `reward_repro` (exists), `auto_lay_warmup` flag default off.

- [ ] **Step 1 (TDD-able parts): Write failing tests** — `tests/test_selfplay.py`:
  - `test_opponent_preprocess_matches_recipe`: build a known ring buffer + a fake `obs_rms` (mean/var), assert `OpponentController`'s preprocess equals `clip((stack-mean)/sqrt(var+eps), -clip, clip)` computed independently (the 0a recipe).
  - `test_ring_buffer_zeroed_on_reset_snake`: after `reset_snake`, the buffer is all zeros; a subsequent `act` writes only the newest slot.
  - `test_env_repro_reward_on_ego_egg_hatch`: force an ego-owned egg to hatch in a stepped env; assert the step's reward includes `reward_repro` exactly once, and a NON-ego egg hatching pays nothing.
  - `check_env(SnakeEnv())` still passes at `OBS_DIM=75` with a default (identity/random) opponent policy.

- [ ] **Step 2: Run to verify fail** → FAIL (`selfplay` missing).

- [ ] **Step 3: Implement `selfplay.py`** — the preprocessing is the spike-0a recipe VERBATIM (see Global Constraints). Default opponent behavior before the first `sync` = go straight (or random) so a fresh env is valid. Then wire `env` (opponent stepping, rewards, curriculum) and `train` (callback). Reward: track eggs' `owner`; when `_hatch_eggs` hatches an egg whose owner contains the ego id, add `reward_repro` to that step's reward (return the count from `_hatch_eggs`, or have the env diff egg-sets — pick one and test it).

- [ ] **Step 4: Run tests + full suite + a smoke** → tests green. Then a 2000-step smoke: `SnakeEnv` with a random opponent policy runs without error, opponents move, an ego reward is finite each step.

- [ ] **Step 5: Commit** — `env: self-play (in-env opponents via 0a preprocessing) + repro/corpse/egg rewards + mating curriculum`

---

### Task B4: 0b gate — short training run confirms mating is discovered

Before the full retrain, verify `+reward_repro` actually fires under the mating curriculum (spec §6 discovery risk). This is a **gate**, not production code.

**Files:** none committed (scratch training run + a metrics read).

- [ ] **Step 1** — run a short train: `./snake train --steps 800000 --envs 16` (fresh) with the mating curriculum warmup generous (`r_mate_easy` large, `mate_steps_easy=1`, low length threshold). Log hatch/repro-reward events (add a temporary counter to the env info dict or a callback).
- [ ] **Step 2 — GATE:** inspect whether `+reward_repro` events occur during warmup (any hatches of ego-owned eggs). 
  - **If yes** → mating is discoverable; proceed to B5/B6 with the curriculum as-is. Record the observed rate.
  - **If no** → enable the `auto_lay_warmup` fallback (world auto-lays an egg for any qualifying nearby pair during the earliest warmup, then withdraws), re-run the short train, confirm reproduction is then learned. If STILL no reproduction behavior, STOP and escalate — the reproduction pillar needs a design revisit (do not burn the full retrain).
- [ ] **Step 3** — record the outcome + chosen curriculum settings in the progress ledger and CLAUDE.md Pitfalls (new "mating discovery" pitfall).

---

### Task B5: Render + persistent viewer + ecosystem metrics (no retrain)

**Files:**
- Modify: `snake_rl/render.py`, `snake_rl/watch.py`
- Test: `tests/test_render_smoke.py`, `tests/test_watch_smoke.py`

**Interfaces:**
- `render`: golden-angle colors (`hue = (snake.color_seed * 137.5) % 360`, fixed S/V; deterministic); draw all `world.snakes` (body/head gradient from the snake's hue); per-snake **3-ring concentric HUD** near each head (outer=energy, middle=stamina, inner=length→cap); eggs (pulse + hatch countdown); corpses (shrinking piles). Ego gets a larger HUD.
- `watch`: the viewer/eval world is **persistent** — it does NOT reset on ego death; the camera follows a chosen live snake (or overview). Headless eval (`run_headless`) reports per-snake catch rate, dash usage, births, kills (cut-off/head-to-head), starvations, population-over-time, and ALL death causes (`obstacle/self/snake/starve`) — replace the hardcoded `deaths={"obstacle":0,"self":0}`.

- [ ] **Step 1: Write failing tests** — extend the smoke tests: `render` draws an N-snake world (with eggs+corpses) to a `dummy` surface without error and colors are deterministic per `color_seed`; `run_headless` on a multi-snake world returns a metrics dict containing all four death causes + a population series, and does not reset on a non-ego death.
- [ ] **Step 2: Run to verify fail.**
- [ ] **Step 3: Implement** — golden-angle palette helper; per-snake draw loop; ring-badge geometry (3 arcs); egg/corpse sprites; persistent-world viewer loop + camera-follow; extended headless metrics.
- [ ] **Step 4: Run tests + full suite** → green.
- [ ] **Step 5: Commit** — `render/watch: N distinct-colored snakes, 3-ring HUD, eggs/corpses, persistent viewer + ecosystem metrics`

---

### Task B6: Full retrain, behavioral judging, tuning, CLAUDE.md

**Files:** `models/` (not committed), `CLAUDE.md`, `docs/superpowers/specs/2026-07-21-multi-snake-world-design.md` (align §5 phase-order note).

- [ ] **Step 1: Retrain** — `./snake train --steps 8000000 --envs 16 --reset`. Watch `ep_rew_mean`, `snake/hardness` → 1.0, and the opponent-sync + mating curriculum. Expect the ramp dip then recovery (Pitfall 4); judge by the END.
- [ ] **Step 2: Judge behaviorally** — `./snake watch --headless --episodes 20`. Healthy (spec §9): population oscillates around a carrying capacity below `n_max`; per-snake catch rate in a sane band; some births + some kills (kills may be rare — I7); self-deaths ≈ 0. Tune §7 knobs + `reward_repro` if off (measure kill rate — I7; if ~0 and more predation wanted, add a tiny gated kill reward, else accept).
- [ ] **Step 3: Update CLAUDE.md** — remove/replace the WIP note with the real multi-snake description; update the architecture table, the RL-design section (self-play, opponent preprocessing, obs 75), the **judging band** for the multi-snake world, and add new Pitfalls: reproduction-reward balance, mating discovery + the chosen fallback, opponent obs preprocessing (0a recipe), the ego/opponent split, dead-snake pruning. Align spec §5 phase-order note to the implemented order.
- [ ] **Step 4: Commit** — `docs: multi-snake retrain results, judging band, new pitfalls (self-play, obs, reproduction, preprocessing)`

---

## Milestone B exit criteria

- All tests green (Milestone A + B).
- 0b gate passed (reproduction discovered, curriculum or fallback recorded).
- Retrain reaches full hardness with a healthy §9 behavioral profile (judged, not assumed).
- `./snake watch` shows N distinctly-colored snakes with ring HUDs, eggs, corpses, in a persistent world.
- CLAUDE.md updated to describe the shipped multi-snake model + new pitfalls.

## Carried-over Milestone-A items resolved here

Dead-snake pruning (B1); per-snake eating/energy (B1); watch `starve`/`snake` death metrics (B5); spec §5 phase-order alignment (B6). Deferred-and-still-deferred (tune during B6 retrain): corpse flat-growth; hatchling `repro_cooldown=0` (revisit with mating rules in B3); test coverage nits (`test_food_target` discrimination, round/clip order in `test_chickens`) — fold into B2/B5 test updates if touched, else note.

## Note on the final review

Per the user's workflow: after B6, run the superpowers final whole-branch review over the ENTIRE multi-snake branch and fix everything including suggestions, then finish the branch.
