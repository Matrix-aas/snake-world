# Multi-snake — Milestone B (RL: obs + self-play + render + retrain) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Milestone-A multi-snake *simulation* into a trained, watchable multi-snake *ecosystem* — snakes perceive each other + eggs, one shared PPO brain is trained by self-play (opponents driven by a policy snapshot), and a persistent viewer shows N distinctly-colored snakes with compact ring HUDs, eggs, and corpses.

**Architecture:** Per-snake world dynamics; a redesigned 75-float egocentric observation (`OBS_DIM 42→75`); a self-play env where the ego is the SB3 learner and opponents are stepped in-env from a policy snapshot with the **exact** `VecNormalize`+`VecFrameStack` preprocessing proven in spike 0a; sparse reward for reproduction/corpse/egg; a mating curriculum with an auto-lay fallback; render + persistent viewer; a from-scratch retrain.

**Tech Stack:** Python 3.13 (`.venv`), numpy≥2, PyTorch (CPU), stable-baselines3, gymnasium, pygame.

> **Revised after a plan review** that caught a fatal omission (the env never spawned opponents) plus reward-corruption traps and under-specified plumbing. Fixes are folded in and tagged `[C-1]`/`[I-n]`/`[M-n]` where they land.

## Global Constraints

- Run tests: `SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest -q` (all Milestone-A tests + new ones green after each task).
- **This milestone REQUIRES a from-scratch retrain** (`--reset`). The shipped `models/` snake is already a placeholder (CLAUDE.md WIP note); `models_good_backup/` holds the original.
- **Torus nearest-image geometry everywhere** — reuse `world.torus_delta / torus_dist / ray_circle_hit / segment_circle_hit`.
- **Opponent obs preprocessing (spike 0a, PROVEN — reproduce EXACTLY).** An opponent's action = `policy.predict(preprocess(ring))` where
  `preprocess(ring) = clip((stack(ring) - mean) / sqrt(var + epsilon), -clip_obs, clip_obs)`.
  `stack(ring)` is the `(OBS_DIM*4,)` vector with the **newest frame LAST**; `mean`/`var` are the `VecNormalize.obs_rms.mean/var` over that **stacked** vector (shape `(OBS_DIM*4,)`) — so **stack THEN normalize**, never per-frame. A normal step does `ring = roll(ring, -OBS_DIM)` then writes the new frame into the last `OBS_DIM` slots; **on a snake's (re)spawn or a full env reset the ring is ZEROED, not rolled** (no stale frame may leak). `epsilon` and `clip_obs` are the loaded `VecNormalize`'s attributes (defaults `1e-8` / `10.0`) — they are **pushed alongside** `obs_rms` by the sync callback `[I-5]`, not hardcoded. `normalize_obs` is a pure function of the snapshot, so pushing `(mean, var, clip_obs, epsilon)` once per rollout is sufficient.
- **Every snake — ego AND opponents — chooses its action from the PRE-STEP world state** (the same state the ego's training obs was taken from). `world.step` therefore collects all opponent actions from the current world BEFORE moving anyone `[M-2]`.
- **All obs channels normalized/clipped** so obs is size-agnostic across the 110–160 world range (spec §4). Every signed channel is bounded explicitly in `observation_space` `[I-4]`.
- **Reward stays sparse** (CLAUDE.md Pitfalls): `+reward_eat` per item, `+reward_repro` on hatch of an ego-owned egg **that actually produced a hatchling** (raided or cap-dropped eggs pay nothing) `[I-2]`, `+reward_death`, unchanged PBRS to nearest chicken. No dense cooperation shaping.
- **PPO hyperparameters unchanged** from the single-snake recipe (constant lr `3e-4`, `n_steps 1024`, `batch_size 256`, `n_epochs 10`, `gae_lambda 0.95`, `ent_coef 0.01`, `target_kl 0.03`, net `[128,128]`, `gamma` from CFG).
- No Claude attribution in commits.
- TDD: failing test → run (fail) → minimal impl → run (pass) → commit. RL-plumbing that can't be unit-TDD'd gets a runnable smoke assert (stated per task).

## File Structure

- **Modify `snake_rl/world.py`** — per-snake `try_eat`/`decay_energy`; `_prune_dead`; `step` collects opponent actions pre-move `[M-2]` and returns detailed deaths `[(id,cause)]` incl. starvation + a hatch owner-list `[I-8]`; `_hatch_eggs` returns the owners of eggs that actually hatched `[I-2]`; `_auto_lay` fallback `[I-9]`.
- **Modify `snake_rl/sensors.py`** — `observe(world, snake=None)` per-snake; new ray categories, social/egg channels, snake-smell, repro-ready; `OBS_DIM 42→75`.
- **Create `snake_rl/selfplay.py`** — `OpponentController`: per-snake frame ring + 0a preprocessing + a policy skeleton loaded from a pushed `state_dict`.
- **Modify `snake_rl/env.py`** — obs-space bounds for `OBS_DIM=75` `[I-4]`; **`reset` draws `n_snakes` from `n_start_min/max` and passes it to `generate_world`** `[C-1]`, and clears the opponent controller `[I-1]`; ego/opponent step split; repro/corpse/egg reward; `set_opponent_policy(state_dict, obs_rms, clip_obs, epsilon)` `[I-5]`; mating curriculum + `auto_lay_warmup` flag in `set_hardness`.
- **Modify `snake_rl/train.py`** — `SyncOpponentPolicy` callback (push `state_dict` + `obs_rms` + `clip_obs` + `epsilon` each rollout); keep `AnnealHardness`.
- **Modify `snake_rl/config.py`** — mating-curriculum easy values, `egg_radius`, and use `chicken_ceiling` for initial spawn `[M-4]`.
- **Modify `snake_rl/render.py`** — golden-angle colors, per-snake 3-ring HUD, eggs, corpses.
- **Modify `snake_rl/watch.py`** — persistent world driving ALL snakes via `OpponentController` against a directly-stepped `World` (no SB3 autoreset) `[I-7]`; camera-follow; `run_headless` RETURNS an ecosystem metrics dict with all death causes `[I-7,I-8]`.

---

### Task B1: Per-snake world dynamics + richer step reporting

Close the ecosystem: every snake eats/grows/starves (so opponents can reach `repro_length_min` and mate, or die); prune dead opponents (persistent world); and surface per-cause deaths + hatches so the viewer can report them.

**Files:** Modify `snake_rl/world.py`; Test `tests/test_multisnake.py`.

**Interfaces:**
- `World.try_eat() -> int` — a private `_snake_eat(s) -> int` eats items near `s.head` applying `s`'s growth/energy and `s.id` for egg-ownership. `try_eat` loops ALL live snakes, **capturing the ego's count in that single pass** (`if s is self.snakes[0]: ate_ego = n`) and returning it — NEVER calling `_snake_eat(ego)` twice `[I-3]`.
- `World.decay_energy()` — decays every live snake.
- `World._prune_dead()` — `self.snakes = [self.snakes[0]] + [s for s in self.snakes[1:] if s.alive]` (ego kept even if dead). Called at end of `step`.
- `World.step(...) -> dict` — return dict gains `"deaths_detailed": list[(id, cause)]` covering BOTH phase-2 (obstacle/self/snake) AND phase-3 starvation deaths, and `"hatched_owners": list[frozenset]` (owner-sets of eggs that produced a hatchling this step) `[I-8]`. Keep `"ate"/"died"/"dashed"/"deaths"` for back-compat.

- [ ] **Step 1: Write failing tests** — `tests/test_multisnake.py` (append):

```python
def test_all_snakes_eat_and_decay_and_ego_count():
    import numpy as np
    from snake_rl.config import CFG
    from snake_rl.worldgen import generate_world
    w = generate_world(CFG, seed=11, size=(140.0,140.0), n_snakes=3)
    w.chicken_pos=np.zeros((0,2)); w.chicken_dir=np.zeros((0,)); w.chicken_id=np.zeros((0,),int)
    opp = w.snakes[1]; opp.energy=10.0
    w.chicken_pos=opp.head[None].copy(); w.chicken_dir=np.zeros(1); w.chicken_id=np.array([999])
    assert w.try_eat() == 0            # ego ate nothing this call...
    assert opp.energy > 10.0           # ...but the opponent did
    # ego count is returned (place a chicken on the ego head)
    ego=w.snakes[0]; w.chicken_pos=ego.head[None].copy(); w.chicken_dir=np.zeros(1); w.chicken_id=np.array([7])
    assert w.try_eat() == 1
    e=opp.energy; w.decay_energy(); assert opp.energy == max(0.0, e-CFG.energy_decay)


def test_prune_keeps_ego_removes_dead_opponents():
    from snake_rl.config import CFG
    from snake_rl.worldgen import generate_world
    w = generate_world(CFG, seed=12, size=(140.0,140.0), n_snakes=3)
    w.snakes[2].alive=False; w._prune_dead()
    assert len(w.snakes)==2 and all(s.alive for s in w.snakes[1:])
    w.snakes[0].alive=False; w._prune_dead()
    assert len(w.snakes)==2 and w.snakes[0].alive is False   # ego kept though dead


def test_step_reports_detailed_deaths_and_hatches():
    import numpy as np
    from snake_rl.config import CFG
    from snake_rl.worldgen import generate_world
    w = generate_world(CFG, seed=13, size=(140.0,140.0), n_snakes=2)
    w.snakes[1].energy = CFG.energy_decay/2    # opponent starves this step
    out = w.step(1,0)
    assert any(cause=="starve" for _id,cause in out["deaths_detailed"])
```

- [ ] **Step 2: Run to verify fail** — `... pytest tests/test_multisnake.py -q`.
- [ ] **Step 3: Implement** — factor `_snake_eat(s)`; loop `try_eat`/`decay_energy` over live snakes; add `_prune_dead`; enrich the `step` return (collect `(id,cause)` in phase-2 AND the phase-3 starvation pass; have `_hatch_eggs` return hatched owner-sets, surfaced as `"hatched_owners"`); call `_prune_dead()` at the end of `step`. Preserve array-consistency.
- [ ] **Step 4: Run tests + full suite** — green (N=1: `try_eat`/`decay_energy` act on `snakes[0]` identically; `_prune_dead` a no-op; the extra return keys are additive so `env`/existing tests are unaffected).
- [ ] **Step 5: Commit** — `world: per-snake eat/energy/starvation, prune dead, detailed death+hatch reporting`

---

### Task B2: Multi-snake observation (`OBS_DIM 42→75`) + bounds

Per-snake egocentric obs (spec §4). **Retrain-critical.** Every signed channel bounded so `check_env` passes `[I-4]`.

**Files:** Modify `snake_rl/sensors.py`, `snake_rl/env.py`, `snake_rl/config.py`; Test `tests/test_sensors.py`.

**Interfaces:**
- `sensors.observe(world, snake=None) -> np.ndarray(75)` (default `snake=world.snakes[0]`). `sensors.OBS_DIM=75`. Layout (all normalized/clipped):
  - rays `9×6=54`: `[dist, is_obstacle, is_chicken, is_self, is_other_body, is_egg]` (Minkowski `+head_radius`; eggs use `egg_radius`).
  - social `7`: `[has_rival, rel_pos_fwd, rel_pos_left, rival_heading_fwd, rival_heading_left, size_ratio, rival_is_dashing]` — `rel_pos_*`=`torus_delta(rival.head,snake.head)` projected on snake's fwd/left, `/ray_range`, clipped `[-1,1]`; `rival_heading_*`=rival heading vec projected on snake fwd/left ∈`[-1,1]`; `size_ratio = clip(rival.target_length / length_cap, 0, 1)` `[I-4]`.
  - egg `4`: `[has_egg, rel_pos_fwd, rel_pos_left, is_mine]` (nearest egg; rel-pos as above; `is_mine`= snake.id in that egg's owner row).
  - smell `6`: `[chicken_intensity, chicken_grad_fwd, chicken_grad_left, snake_intensity, snake_grad_fwd, snake_grad_left]`.
  - proprio `4`: `[energy/energy_max, target_length/length_cap, stamina/s_max, repro_ready]` (`repro_ready`=1 iff energy>repro_energy_frac*energy_max AND target_length>repro_length_min AND repro_cooldown==0).
- `env.observation_space` low/high — enumerate EVERY index `[I-4]`: rays/one-hots/proprio ∈`[0,1]`; **signed → `[-1,1]`**: social idx `rel_pos_fwd,rel_pos_left,rival_heading_fwd,rival_heading_left` and egg `rel_pos_fwd,rel_pos_left`; smell intensities ∈`[0, chicken_ceiling]`/`[0, n_max]`, smell **gradients** ∈`[-chicken_ceiling, chicken_ceiling]`/`[-n_max, n_max]`.
- `config`: add `egg_radius: float = 1.0` `[M-4]`.

- [ ] **Step 1: Write failing tests** — `tests/test_sensors.py` (concrete coordinate cases like the existing file): `OBS_DIM==75`; `observation_space.contains(observe(world))` for a generated multi-snake world (guards all bounds `[I-4]`); rival dead-ahead → `has_rival==1, rel_pos_fwd>0, all |channels|<=1`; self-owned egg ahead → `has_egg==1, is_mine==1`; foreign egg → `is_mine==0`; a rival body across a ray sets `is_other_body` while `is_self`/`is_chicken` stay correct; `repro_ready` toggles exactly on the three-way gate; `size_ratio<=1` for a max-length rival.
- [ ] **Step 2: Run to verify fail.**
- [ ] **Step 3: Implement** — extend `_scan` to classify other-snake bodies (`_other_hazard`-style point set) + eggs as ray hit-types (Minkowski-inflated); add `_social`/`_egg_channel`; extend `smell` with a snake field; add `repro_ready`; parametrize by `snake`. Bump `OBS_DIM`. Rewrite `env.observation_space` bounds index-by-index. Also `[M-4]`: `worldgen.py` initial-chicken draw uses `chicken_ceiling` (not `max_chickens`); add soft invariant §10.6 (`n_max` bodies ≪ world area, log-warn).
- [ ] **Step 4: Run tests + full suite** — new sensors tests green. **Update** `test_env.py::check_env` and any obs-shape/dim assertions from 42→75 (intended obs change; update to the real new dim, don't weaken).
- [ ] **Step 5: Commit** — `sensors: 75-float multi-snake obs + full observation_space bounds`

---

### Task B3: Self-play env + `OpponentController` + rewards + curriculum

The core RL task. Ego = SB3 learner; opponents stepped in-env from a policy snapshot via the proven 0a preprocessing.

**Files:** Create `snake_rl/selfplay.py`; Modify `snake_rl/env.py`, `snake_rl/train.py`, `snake_rl/config.py`; Test `tests/test_selfplay.py`, `tests/test_env.py`.

**Interfaces:**
- `selfplay.OpponentController`:
  - state: a torch **policy skeleton** + `(mean, var, clip_obs, epsilon)` + `dict[snake_id -> np.ndarray(4, OBS_DIM)]` ring buffers.
  - **Policy skeleton construction `[I-6]`:** `from stable_baselines3.common.policies import ActorCriticPolicy; policy = ActorCriticPolicy(observation_space=Box(-inf,inf,(OBS_DIM*4,),float32), action_space=MultiDiscrete([3,2]), lr_schedule=lambda _: 0.0, net_arch=dict(pi=[128,128], vf=[128,128]))`. `.predict(x)` consumes the ALREADY-normalized `(OBS_DIM*4,)` vector (no internal normalization) — that's exactly what `model.policy` sees at train time.
  - `.sync(state_dict, obs_rms, clip_obs, epsilon)` — `policy.load_state_dict({k: torch.as_tensor(v) for k,v in state_dict.items()})`; store `obs_rms.mean/var`, `clip_obs`, `epsilon`. Idempotent, pure.
  - `.act(world, snake) -> (steering, dash)` — `raw = observe(world, snake)`; roll+write the snake's ring (create+zero it if absent); `x = clip((stack-mean)/sqrt(var+eps), -clip_obs, clip_obs)`; `a,_ = policy.predict(x[None], deterministic=False)`; return `(int(a[0][0]), int(a[0][1]))`. Before the first `.sync`, act = go straight `(1,0)` so a fresh env is valid `[I-1 bootstrap]`.
  - `.reset_snake(snake_id)` — zero that ring (call on hatch + on a snake's death).
  - `.reset_all()` — drop all rings (call on env reset) `[I-1]`.
- `env.SnakeEnv`:
  - **`reset` `[C-1]`:** `n = int(self.np_random.integers(cfg.n_start_min, cfg.n_start_max+1)); self.world = generate_world(self.cfg, seed=world_seed, size=self._world_size, n_snakes=n)`; then `self._opp.reset_all()` `[I-1]`.
  - `set_opponent_policy(state_dict, obs_rms, clip_obs, epsilon)` → `self._opp.sync(...)` `[I-5]`.
  - `step`: pass `opponent_fn = lambda world, s: self._opp.act(world, s)` to `world.step`. Reward = `reward_eat*out["ate"] + reward_repro*(# of out["hatched_owners"] sets containing the ego id) + (out["died"] → reward_death) + PBRS + step_penalty` (spec §6). A raided/cap-dropped ego egg contributes nothing because it is not in `hatched_owners` `[I-2]`. On any snake in `out["deaths_detailed"]`, call `self._opp.reset_snake(id)`; on each hatch, the new snake's ring is created-zeroed lazily on first `.act`.
  - `set_hardness(h)` — also interpolate the mating curriculum: `r_mate = lerp(r_mate_easy, r_mate, h)`, `mate_steps = round(lerp(mate_steps_easy, mate_steps, h))`, `repro_length_min = lerp(repro_length_min_easy, repro_length_min, h)`; set `world`-visible `auto_lay_warmup = (h < auto_lay_until)` when the fallback is enabled `[I-9]`.
- `world.step` change `[M-2]`: collect `opp_actions = {s.id: opponent_fn(self, s) for s in self.snakes[1:] if s.alive}` from the PRE-MOVE world, THEN move ego, THEN move each opponent with its pre-collected action. (Ego and opponents thus both act on pre-step state.)
- `train.SyncOpponentPolicy(BaseCallback)` — on `_on_rollout_end`: `sd = {k: v.detach().cpu().numpy() for k,v in self.model.policy.state_dict().items()}`; `vn = self.model.get_vec_normalize_env()`; `self.training_env.env_method("set_opponent_policy", sd, vn.obs_rms, vn.clip_obs, vn.epsilon)`. Registered alongside `AnnealHardness`.
- `config`: `r_mate_easy=12.0, mate_steps_easy=1, repro_length_min_easy=6.0, auto_lay_until=0.15, auto_lay_warmup_enabled=False` (default off; B4 flips it if needed).

- [ ] **Step 1: Write failing tests** — `tests/test_selfplay.py`:
  - `test_preprocess_matches_0a_recipe`: given a hand-built ring + fake `(mean,var,clip,eps)`, `OpponentController.act`'s internal preprocess equals `clip((stack-mean)/sqrt(var+eps),-clip,clip)` computed independently; newest frame is at `[-OBS_DIM:]`.
  - `test_reset_all_and_reset_snake_zero_rings`: after `reset_all`/`reset_snake`, the buffer(s) are zeros; a subsequent `.act` writes only the last `OBS_DIM` slots.
  - `test_env_spawns_multiple_snakes` `[C-1]`: `SnakeEnv().reset()` then `len(world.snakes)` ∈ `[n_start_min, n_start_max]` across several resets.
  - `test_repro_reward_only_on_ego_hatch` `[I-2]`: an ego-owned egg that HATCHES pays `reward_repro` once; an ego-owned egg that is EATEN (raided) pays nothing; a non-ego egg hatching pays nothing.
  - `check_env(SnakeEnv())` passes at `OBS_DIM=75` (opponents default-straight pre-sync).
- [ ] **Step 2: Run to verify fail.**
- [ ] **Step 3: Implement** `selfplay.py` (preprocess VERBATIM per Global Constraints + `[I-6]` skeleton), then `env` (opponent stepping, `[C-1]` spawn, `[I-1]` reset_all, `[I-2]` reward, `[I-5]` sync signature, curriculum), `world.step` `[M-2]` reorder, `train` callback. **Note `[M-1]`:** opponents sample from torch's global RNG (episode-level opponent-action reproducibility is not preserved — acceptable for training; documented, not a correctness issue).
- [ ] **Step 4: Run tests + full suite + smoke** — tests green; then a 2000-step smoke: a `SnakeEnv` with the default (pre-sync) controller runs without error, opponents move, per-step ego reward is finite, and after a manual `set_opponent_policy` with the current model's `state_dict`/`obs_rms` the opponents still step without error.
- [ ] **Step 5: Commit** — `env: self-play (in-env opponents, 0a preprocessing) + spawn N + repro/corpse/egg reward + mating curriculum`

---

### Task B4: 0b gate — short run confirms mating is discovered

Verify `+reward_repro` fires under the curriculum BEFORE the 8M retrain (spec §6). A **gate**, not committed code.

- [ ] **Step 1** — short train `./snake train --steps 800000 --envs 16` (fresh; generous warmup: `r_mate_easy=12`, `mate_steps_easy=1`, `repro_length_min_easy=6`). Add a temporary hatch/repro-reward counter (env `info` or a callback) logged to stdout.
- [ ] **Step 2 — GATE:** did ego-owned eggs hatch (repro reward fired) during warmup?
  - **Yes** → mating discoverable; record the rate; proceed to B5/B6.
  - **No** → enable the auto-lay fallback (`[I-9]`, below), re-run 800k, confirm reproduction is then learned. **Still no** → STOP + escalate (reproduction pillar needs redesign); do NOT run the 8M retrain.
- **Auto-lay fallback spec `[I-9]`** (implement only if the gate fails): `World._auto_lay()` — for every unordered live pair both `repro_ready` and within `r_mate`, lay an egg immediately (no streak), same cost/cooldown as `_resolve_mating`. `World.auto_lay_warmup` (bool, default False) is set by `env.set_hardness` to `h < auto_lay_until` (`0.15`). `world.step` phase-3 calls `_auto_lay()` when `self.auto_lay_warmup`. This seeds `+reward_repro` early, then withdraws as `h` ramps.
- [ ] **Step 3** — record the outcome + chosen curriculum/fallback in the ledger and a new CLAUDE.md "mating discovery" pitfall.

---

### Task B5: Render + persistent viewer + ecosystem metrics (no retrain)

**Files:** Modify `snake_rl/render.py`, `snake_rl/watch.py`; Test `tests/test_render_smoke.py`, `tests/test_watch_smoke.py`.

**Interfaces:**
- `render`: golden-angle colors (`hue=(snake.color_seed*137.5)%360`, fixed S/V, deterministic); draw all `world.snakes` (body/head gradient from the snake's hue); per-snake **3-ring concentric HUD** near each head (outer=energy, middle=stamina, inner=target_length→cap); eggs (pulse + hatch countdown from `timer`); corpses (piles scaled by `food`). Ego HUD larger.
- `watch` persistent viewer `[I-7]`: load the model + `VecNormalize` from `models/` to obtain `obs_rms`/`clip_obs`/`epsilon`; build ONE `selfplay.OpponentController`, `sync` it with the loaded policy `state_dict` + norm stats; step a plain `World` (from `generate_world(..., n_snakes=...)`) DIRECTLY (no SB3 VecEnv, no autoreset), driving **every** snake (including the "ego" slot) via `controller.act`; the world is **persistent** — never reset on any death; the camera follows a chosen live snake (fallback: overview) and re-targets when it dies.
- `watch.run_headless(...) -> dict` `[I-7,I-8]`: RETURN (not just print) a metrics dict — per-snake catch rate, dash usage; and ecosystem series from `step`'s `deaths_detailed`/`hatched_owners`: `births`, `kills` (cause∈{snake}), `starvations` (cause=="starve"), `obstacle`/`self` deaths, and population-over-time. Replace the hardcoded `deaths={"obstacle":0,"self":0}`.
- `watch` `[M-3]`: on load, if `model.observation_space.shape != (OBS_DIM*4,)`, raise a clear error (don't run an old-dim model against new obs).

- [ ] **Step 1: Write failing tests** — extend smokes: `render` draws an N-snake world with eggs+corpses to a `dummy` surface without error; `color_for(seed)` is deterministic and distinct for `0..5`; `run_headless` on a multi-snake world RETURNS a dict containing all four death causes + a `population` series and does NOT reset on a non-ego death (assert population can drop below the start count without a reset).
- [ ] **Step 2: Run to verify fail.**
- [ ] **Step 3: Implement** — golden-angle palette helper; per-snake draw loop; 3-arc ring badge geometry; egg/corpse sprites; the persistent-world controller-driven viewer loop + camera-follow; `run_headless` returns metrics; the dim-mismatch guard.
- [ ] **Step 4: Run tests + full suite** — green.
- [ ] **Step 5: Commit** — `render/watch: N-color snakes, 3-ring HUD, eggs/corpses, persistent controller-driven viewer + ecosystem metrics`

---

### Task B6: Full retrain, judging, tuning, docs

**Files:** `models/` (not committed), `CLAUDE.md`, spec §5 note.

- [ ] **Step 1: Retrain** — `./snake train --steps 8000000 --envs 16 --reset`. Watch `ep_rew_mean`, `snake/hardness`→1.0, opponent-sync + mating curriculum. Expect the ramp dip then recovery (Pitfall 4); judge by the END. **`[M-5]`** if fps is too low, reduce the `_other_hazard` body-point density (coarser spacing for the hazard set only) and re-time before committing to the full run.
- [ ] **Step 2: Judge behaviorally** — `./snake watch --headless --episodes 20`. Healthy (spec §9): population oscillates below `n_max`; per-snake catch rate sane; some births + some kills (kills may be rare — spec I7; measure, and only if ~0 & more predation wanted add a tiny gated kill reward); self-deaths ≈ 0. Tune §7 knobs + `reward_repro` as needed.
- [ ] **Step 3: Update CLAUDE.md** — replace the WIP note with the real multi-snake description; update the architecture table, RL-design section (self-play, opponent 0a preprocessing, obs 75, ego/opponent split, `_prune_dead`), the **judging band** for the multi-snake world, and new Pitfalls: reproduction-reward balance, mating discovery + chosen fallback, opponent obs preprocessing, per-snake vs ego-only history. Align spec §5 phase-order note to the implemented order.
- [ ] **Step 4: Commit** — `docs: multi-snake retrain results, judging band, new pitfalls`

---

## Milestone B exit criteria
- All tests green (A + B).
- 0b gate passed (reproduction discovered; curriculum/fallback recorded).
- Retrain reaches full hardness with a healthy §9 profile (judged, not assumed).
- `./snake watch` shows a persistent world of N distinctly-colored snakes with ring HUDs, eggs, corpses.
- CLAUDE.md updated.

## Carried-over Milestone-A items resolved here
Dead-snake pruning (B1); per-snake eating/energy (B1); watch `starve`/`snake` metrics (B5); spec §5 phase-order alignment (B6); `worldgen` uses `chicken_ceiling`, soft invariant §10.6 (B2). Tune during B6: corpse flat-growth; hatchling `repro_cooldown=0` (mating rules — B3 curriculum). Fold the `test_food_target`/round-clip nits into B2 test updates if touched.

## Note on the final review
After B6, run the superpowers final whole-branch review over the ENTIRE multi-snake branch and fix everything including suggestions, then finish the branch.
