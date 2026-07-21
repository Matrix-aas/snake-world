# CLAUDE.md — Snake-RL project guide

A PPO-trained multi-snake predator **ecosystem** in a continuous 2D torus world. 2–4 snakes
(population flux up to `n_max=6`) share **one PPO brain** via self-play: they hunt fleeing/pecking
chickens by egocentric **sight** (9 vision rays) + **smell**, sprint with a stamina-limited
**dash**, kill each other by **cut-off** (leaving a corpse to scavenge), and **reproduce** via
eggs laid by two well-fed snakes — a living, self-sustaining population meant to run forever as a
screensaver. Trained headless (SB3 PPO, CPU), watched in a fullscreen pygame viewer. **The
behavior is meant to look alive** — stalk, deliberate pounce, guard/raid eggs, avoid clutter — not
to be optimal.

> **This file is the living memory of the project. Keep it current.** When you change the
> reward, the observation, the physics, the training recipe, or the hyperparameters — update
> the relevant section here *in the same change*, especially the **Pitfalls** and
> **What needs a retrain** lists. Future-you will rely on it to avoid re-learning the hard way.

---

## Run it

```bash
./snake watch                 # fullscreen viewer (a trained model ships in models/)
./snake watch --windowed      # windowed
./snake watch --headless --episodes 20   # persistent-ecosystem eval, prints + returns metrics
./snake train --steps 8000000 --envs 16 --reset   # train from scratch (~40–75 min)
```

`./snake` is a launcher that creates `.venv` and installs deps on first run — never touch pip
directly. Watch keys: `SPACE` pause · `N` new persistent world · `S` sensors · `↑/↓` (or `+/-`)
sim speed · `ESC` quit. (No deterministic toggle: the viewer drives **every** snake, including
the legacy slot-0 "ego", through the same stochastic self-play controller — see below.)

**Env:** macOS, Python 3.13 venv at `.venv/`. Always run Python via `.venv/bin/python` (or
`./snake`). For any headless script that imports `snake_rl`, set `PYTHONPATH="$PWD"` and
`SDL_VIDEODRIVER=dummy` (pygame needs a driver; `dummy` = no window). `SubprocVecEnv` cannot
be created from a `python - <<EOF` heredoc (multiprocessing needs a real module file) — write
the script to a file and run it.

Design docs (spec + step plans) live in `docs/superpowers/specs/` and `docs/superpowers/plans/`;
session-by-session build reports live in `.superpowers/sdd/*-report.md` and the ledger
`.superpowers/sdd/progress.md`.

---

## Architecture (one responsibility per file)

| File | Does | Retrain if changed? |
|---|---|---|
| `snake_rl/config.py` | ALL constants (frozen `Config`) + `assert_invariants` | usually yes |
| `snake_rl/world.py` | torus geometry, `Snake` dataclass, `World.snakes[]`, per-snake motion/dash/stamina, **two-phase `step()`** (move all → resolve all deaths → chickens/eat/energy/starvation/spawn/mating/hatching), inter-snake cut-off + corpses, peck/walk/flee chicken FSM, egg-based reproduction | yes (dynamics) |
| `snake_rl/worldgen.py` | random world (size, obstacles, N spread-out spawned snakes, initial chickens up to the population ceiling) | yes |
| `snake_rl/sensors.py` | vectorized raycast (7 categories) + social + egg + smell (chicken/rival/corpse) + proprio → 87-float per-snake observation | **yes** (obs) |
| `snake_rl/selfplay.py` | `OpponentController` — drives in-env opponents from a synced policy snapshot, replicating the exact `VecFrameStack`+`VecNormalize` preprocessing the SB3 learner sees | **yes** (self-play plumbing, mirrors sensors) |
| `snake_rl/env.py` | Gymnasium env: snake 0 = the learner ("ego"); opponents stepped via `OpponentController`; reward (eat/reproduce/death + PBRS); dual curriculum (`set_hardness`: stamina **and** mating); 87-float obs-space bounds | yes (reward/obs) |
| `snake_rl/train.py` | SB3 PPO, VecEnv stack, `AnnealHardness` + `SyncOpponentPolicy` callbacks, checkpoints | recipe only |
| `snake_rl/render.py` | pygame drawing: sprites, per-snake ring HUD, eggs/corpses, blood/gore, sprite-sheet animations | **no** (visual only) |
| `snake_rl/watch.py` | load model, build a **persistent** multi-snake world (never resets on any single death), reseed floor, interpolated viewer, headless ecosystem eval | no (except obs plumbing) |
| `snake_rl/__main__.py` | CLI dispatch | no |

Data flow (training): `worldgen → World` (N snakes, physics) → `sensors.observe(world, snake)` per
snake → `env` (ego obs/reward/done; opponents driven by `selfplay.OpponentController`) → PPO.
Data flow (viewer/headless): `watch` loads the checkpoint, builds **one** persistent `World` +
**one** `OpponentController` synced to it, and drives **every** snake — including the nominal
slot-0 "ego" left over from the single-snake days — through that same controller; nothing resets
on death, only individual snakes die/hatch/reseed.

**Vec-wrapper stack** (still applies, matters for accessors): `VecNormalize( VecFrameStack(
DummyVecEnv([ Monitor(SnakeEnv) ]) ) )`. The underlying world is
`vec.venv.venv.envs[0].unwrapped.world` (see `watch._world_of`).

---

## The RL design (and why each piece is the way it is)

- **Algorithm:** SB3 `PPO`, `MlpPolicy`, **CPU**, 16 parallel random worlds (`n_steps=1024`,
  `batch_size=256` → 64 minibatches over a 16384 buffer — unchanged from the single-snake days).
- **Self-play, one shared brain, no genetics.** `SnakeEnv` controls **snake 0 ("ego")** — the
  actual PPO learner. `World` holds `n_start∈[2,4]` snakes at reset, growing by hatching up to
  `n_max=6`. Every other snake (`snakes[1:]`) is stepped **in-env** by
  `selfplay.OpponentController`, driven by a **snapshot** of the ego's own policy — every snake
  "thinks" the same way; the ecosystem is population flux (birth/death), not evolving intelligence.
- **Opponent obs preprocessing — do not improvise this, it's exact-parity-critical.** The ego's
  obs is normalized (`VecNormalize`) and frame-stacked (`VecFrameStack`) **outside** the env, in
  the main process. In-env opponents only have the raw single-frame `sensors.observe(world,
  snake)`, so `OpponentController` must reproduce the SB3 preprocessing by hand, verified to
  `1.7e-7` parity in a dedicated spike:
  1. Each opponent keeps a **4-frame ring buffer** keyed by snake id, newest frame **last**.
  2. **Stack THEN normalize** — `clip((stack - mean) / sqrt(var + eps), ±clip_obs)` over the
     whole stacked vector, never per-frame.
  3. On a snake's death or hatch (new id) or a full env reset, its ring is **ZEROED, not rolled**
     — a stale frame from a different occupant of that slot must never leak in.
  4. `train.SyncOpponentPolicy` pushes `state_dict` + `obs_rms` (mean/var) + `clip_obs` + `epsilon`
     into every env's `OpponentController` once per rollout (`env_method("set_opponent_policy",
     …)`), mirroring `AnnealHardness`'s pattern. Unlike `AnnealHardness` (fresh-run only), this
     callback runs for **both a fresh run and a resume** — a resumed opponent must also track the
     current policy.
  5. Opponent actions sample **stochastically** (`deterministic=False`) for behavioral diversity.
- **Observation — 87 floats, egocentric, frame-stacked ×4 = 348** (grew from the single-snake
  game's 42 in two steps: 42→75 adding social/egg/rival-sensing, then 75→87 adding corpse
  sensing). Layout (`sensors.observe`, bounds hand-enumerated in `env._make_observation_space`):
  - **Vision, 9 rays × 7 = 63:** `[dist, is_obstacle, is_chicken, is_self, is_other_body, is_egg,
    is_corpse]`. Every target (obstacles, chickens, own body, **rival heads+bodies, eggs,
    corpses**) is inflated by `head_radius` in the same shared `_scan` step (Pitfall 4).
  - **Social, 7:** nearest *other* live snake, egocentric — `[has_rival, rel_pos_fwd, rel_pos_left,
    their_heading_fwd, their_heading_left, size_ratio, is_dashing]`. `has_rival` disambiguates "no
    rival" from "rival at range 0"; positions normalized by `ray_range`, clipped to `[-1,1]`.
  - **Egg, 4:** nearest egg, egocentric — `[has_egg, rel_pos_fwd, rel_pos_left, is_mine]`.
    `is_mine` (egg carries `owner` ids) is what lets guarding vs. raiding diverge.
  - **Smell, 9 = 3 fields × 3:** `[chicken_intensity/grad_fwd/grad_left, snake_…, corpse_…]` —
    omnidirectional sense-around-a-rock/beyond-the-vision-cone channel. Chicken/snake fields are
    provably bounded by `chicken_ceiling`/`n_max` (structural caps in `world.py`); corpses have no
    such cap (they persist until eaten), so `sensors.smell` explicitly clips the corpse field to
    `chicken_ceiling` to keep `observation_space.contains()` true unconditionally.
  - **Proprioception, 4:** `[energy, length, stamina, repro_ready]`. `repro_ready` = above the
    mating thresholds (energy, length) and off `repro_cooldown` — **this reads `repro_length_min`
    directly**, which matters for Pitfall 12 below.
- **Action:** `MultiDiscrete([3,2])` = steer `{left,straight,right}` × dash `{no,yes}` — unchanged.
- **Reward (sparse — never dense "cooperation" shaping):** `+reward_eat` once per item consumed
  (chicken, corpse, or a foreign egg — never per-bite of a big corpse); `+reward_repro` **only on
  the real hatch of an egg the ego co-owns** (deferred to hatch, not laying — a raided or
  population-cap-dropped egg pays nothing, so guarding matters); `reward_death` on any death cause;
  PBRS toward the nearest chicken (unchanged, Pitfall 3). Cooperation emerges only because
  `+reward_repro` is unreachable alone; "avoid bigger snakes" emerges because their body is lethal
  — no proximity/friendship bonus.
- **Memory:** `VecFrameStack(4)`, NOT an LSTM — unchanged reasoning (self-collision memory, 0
  self-deaths). Opponents get the same 4-frame ring (above), so this applies symmetrically.
- **Deliberate dash = MECHANICAL rationing, never a reward penalty** (Pitfall 1) — unchanged.
- **Dual curriculum, one ramp, one callback.** `env.set_hardness(h)` interpolates BOTH: (a) the
  stamina gate/regen (`dash_min_stamina`, `stamina_regen`: easy always-on dash → the real reserve)
  and (b) the mating gate (`r_mate`, `mate_steps`, `repro_length_min`: loose/instant → the tight
  hard values) on the same `hardness_warmup`→`hardness_full` ramp (0.42→0.85 of total steps).
  `train.AnnealHardness` pushes `h` into every env each rollout, same as before Milestone B — now
  it anneals reproduction discovery too: an abrupt switch straight to the tight hard mating gate
  would collapse a learned pair-bonding behavior the same way an abrupt stamina switch collapses a
  learned hunter, so both ramp together, gradually, on one callback.
- **Reproduction / eggs / corpses / starvation (world mechanics, not reward hacks).** Two live
  snakes both above `repro_energy_frac·energy_max` energy and `repro_length_min` length, both off
  cooldown, that stay within `r_mate` for `mate_steps` consecutive steps lay one egg at their
  midpoint; **both** parents pay `repro_cost` energy and enter `repro_cooldown`. The egg lies
  `egg_timer` steps — edible by any non-owner (`egg_food` energy) — then hatches into a fresh
  `Snake` (same shared brain, `hatch_energy_frac·energy_max` starting energy) unless the population
  is already at `n_max` (the egg just expires). A snake whose `energy` hits 0 dies of
  `"starve"`. Any death (`obstacle`/`self`/`snake`/`starve`) spawns a corpse
  (`corpse_food_per_length · length` food) that anyone can scavenge.
- **Chicken behavior FSM (peck / walk / flee).** Each chicken cycles **PECK** (stands still,
  `chicken_peck_min..max` steps — the prime catch window) ⇄ **WALK** (`v_wander` amble,
  `chicken_walk_min..max` steps) on a timer; any live snake within the **alert radius** triggers
  **FLEE**: a `chicken_startle_steps`-step **freeze** (speed 0, a beat of surprise), then bolt at
  `v_flee` along the **repulsion resultant** of every near snake (never bolts from one snake
  straight into a second; a degenerate cancellation falls back to fleeing the single nearest). The
  alert radius is **state-dependent**: `r_flee_peck` (2.5, tight) while pecking — a head-down
  chicken is *distracted*, catchable by stalking up and pouncing — vs. the full `r_flee` (12)
  while walking or already fleeing. This (Pitfall 10) is what makes hunting discoverable at all
  with 2–4 snakes competing for the same chickens.
- **Persistent-viewer reseed floor.** The training env resets on ego death (short episodes); the
  viewer/headless-eval world does not (Pitfall 11). `watch._reseed_floor`, called every
  `_step_world` tick, spawns a fresh full-health snake (via `World._free_point`, same placement
  worldgen uses) whenever the live count drops below `cfg.n_start_min` (2) — a hard "the
  screensaver never truly dies" floor, a no-op whenever natural dynamics keep the population
  healthy.

---

## PITFALLS — the hard-won lessons (read before touching reward/stamina/obs/hyperparams)

Every one of these cost real training runs (some cost more than one). Do not rediscover them.

**Still true from the single-snake era:**

1. **A per-step reward penalty on dash COLLAPSES hunting.** The agent learns "never dash, just
   survive" before it discovers dashing → chickens → `+reward_eat`. *Ration the dash mechanically
   (stamina), not with reward.* (`dash_penalty` exists but defaults to `0`.)
2. **A decaying learning rate makes the curriculum FRAGILE — now doubly true.** The task is
   *non-stationary* twice over: stamina/mating harden mid-training (curriculum) **and** the
   self-play opponent keeps improving as the ego does. With `lr → 0` the policy can't adapt once
   either source of drift bites and collapses to survive-only. **Constant `learning_rate=3e-4`**
   with `target_kl=0.03` as the stability guard.
3. **PBRS: zero shaping only on a chicken-SET change (eat/spawn), NOT on a nearest-identity
   switch.** `Φ = −dist_to_nearest` is *continuous* as which chicken is nearest switches, so
   zeroing there throws away valid guidance. Zero only when the id-set changes
   (`env._shaping`, tracked by `frozenset(chicken_id)`).
4. **Rays fire from the head CENTER — inflate every target by `head_radius`** (Minkowski) or the
   snake clips things with the edge of its head where no ray points. Still one code path
   (`sensors._scan`), now shared by obstacles, chickens, own body, **rival heads/bodies, eggs, and
   corpses** — get this right once and every new sensed category inherits it for free.
5. **The self-collision neck-skip must clear the whole swept step**
   (`head_radius + body_radius + v_dash + segment_spacing`), not just the head, or a snake moving
   *straight* collides with its own neck. **Corollary for multi-snake:** the *rival* hazard set
   (`world._other_hazard`) must **NOT** reuse this skip — a rival's full body + head, no neck-skip,
   because the self neck-skip is a self-collision concession; applying it to a rival would make the
   area right behind their head falsely non-lethal and break cut-off kills.
6. **Over-tuning PPO SLOWS hunting *discovery* → survive-only.** `gamma=0.99` (not `0.995`) and
   `batch_size=256` (not `512`, → 64 minibatches over the 16384 buffer) reliably discover hunting;
   research-style configs optimize *final* quality, not *discovery*. Get hunting working first.
7. **pygame/macOS rendering:** SRCALPHA sprites need `.convert_alpha()` or they render as black
   squares on a real display (fine under `dummy`); `smoothscale` *into* the window surface renders
   black — smoothscale to a temp then blit; fullscreen needs `FULLSCREEN|SCALED` and you must read
   back `display.get_size()` (Retina). Clear the sprite cache on `set_mode`.
8. **Monitoring steals CPU.** Running evals / frame renders *while* training drops training fps.
   Monitor with cheap `grep`s, not heavy scripts. Kill stray runs with `pkill -f "snake_rl
   train"`; orphaned `forkserver` workers linger and slow everything (`pgrep -f forkserver`).

**New from the multi-snake build:**

9. **Prey that flee EVERY nearby snake collapse hunting-discovery.** With chickens reacting to the
   repulsion of all N snakes, they're almost always sprinting (`v_flee > v_snake`) somewhere near
   *someone*, so a fresh (near-random) policy can never catch one — the first full 8M retrain
   aborted at ~2M, `eaten≈0`, `ep_rew_mean` stuck around −13. A training diagnostic where chickens
   reacted to only ONE snake bootstrapped fine — proof the multi-snake reactivity, not the task
   size, was the blocker.
10. **Peck-distraction is THE fix that made multi-snake hunting bootstrap.** A realistic chicken
    FSM alone (peck/walk/flee) wasn't enough — a *pecking* chicken still fled at the full
    `r_flee=12`, so the "catch window" gave no real catch (a second retrain attempt still aborted,
    survive-only at 1.73M). The real fix: a pecking chicken is **distracted**, alert only within a
    tight `r_flee_peck≈2.5` — stalk up and pounce, no precise dash needed. This made hunting
    bootstrap **instantly** (eaten climbing from step 30k) and the whole hunt→reproduce→hatch cycle
    followed within under 1M steps. Startle is a brief **freeze**, not a fast burst — a fast burst
    at engagement range makes the catch *harder*, not more exciting. Bonus: walk-catchable pecking
    chickens make hunting robust through the stamina-hardening ramp (no dash strictly required), so
    the ramp dip that plagued the single-snake days is much milder here.
11. **The training env RESETS on ego death (short episodes), so the policy learns to HUNT but not
    to SUSTAIN a persistent population.** Dropped into the never-resetting viewer world, a strong
    hunter's population still collapsed to 0: at full stamina/mating hardness, organic reproduction
    is rare (0–2 matings per training window) while deaths (obstacles, cut-off) keep happening, so
    births < deaths. Two runtime-only fixes (no retrain — see Pitfall 12 for exactly why that's
    safe): (a) **ease the hard-endpoint mating constants** (`r_mate`, `mate_steps`, `repro_cost`,
    `repro_cooldown`, `repro_length_min`, plus a softer `stamina_regen`) so organic births keep
    pace; (b) a viewer **reseed floor** (`watch._reseed_floor`) as a guaranteed liveliness net —
    spawn a fresh snake whenever live count < `n_start_min`. Verified end to end: before the fixes,
    population decayed to ~0 almost immediately; after, population never drops below the floor
    (min 2) **and** organic births are consistently > 0 (11–30 per 10k–20k-step run across seeds) —
    a genuinely living, cycling ecosystem, not just floor-padding.
12. **Before easing a "hard endpoint" constant post-training, check whether `sensors.observe` reads
    it — if so, the new value must stay within the range the curriculum already swept.** Of the six
    constants eased in Pitfall 11, five (`r_mate`, `mate_steps`, `repro_cost`, `repro_cooldown`,
    `stamina_regen`) are genuinely **not** in the observation — the policy only feels their effect
    indirectly (a slightly less starved stamina reserve, a looser mating gate) and trivially
    generalizes. But `repro_length_min` **is** observed — `sensors._repro_ready` reads it directly
    to compute the `repro_ready` proprioception bit (obs index 86). That's safe here **only**
    because `set_hardness`'s mating curriculum already sweeps `repro_length_min` continuously from
    `repro_length_min_easy=6.0` up to the hard value (`10.0` during the training run that produced
    the shipped model) as `h: 0→1` — the post-training eased value (`8.0`) sits *inside* that
    already-experienced `[6, 10]` range, so the model isn't seeing a novel observation, just
    flipping `repro_ready` slightly earlier. Landing an eased value **outside** the curriculum's
    swept range would be a real (silent) obs-distribution shift and would need a retrain.
13. **Two-phase step is required for multi-snake determinism.** `World.step` moves **every** live
    snake first (ego + all opponents, opponent actions sampled from the pre-move world), **then**
    decides all deaths together against the frozen post-move state, **then** applies them — never
    move-and-resolve one snake at a time. A sequential order would make results order-dependent and
    break "mutual head-to-head ⇒ both die" (whoever moves second would hit an already-stale body).
14. **Corpses were food (`try_eat`) but UNSENSED** (no ray category, no smell field) — scavenging
    literally could not emerge, because nothing in the observation pointed a snake at a corpse.
    Fixed by adding an `is_corpse` ray one-hot + a corpse smell field (`OBS_DIM` 75→87), caught
    late (right before the final 8M retrain) by a spec-gap review, not by the original design.
15. *(Minor, worth knowing)* **macOS sleep pauses training** — the machine sleeping mid-run makes
    SB3's reported fps read absurdly low afterward (it divides steps by a wall-clock time inflated
    by the sleep, not by actual compute time). Use `caffeinate` for long unattended runs and judge
    real throughput from step-timestamps, not the printed fps alone.

---

## Config & invariants (`config.py`)

All tunable numbers live in one frozen `Config`. `assert_invariants(cfg)` runs at env init and
**fails fast** if a change breaks a guarantee — respect these when tuning:

1. `v_dash > v_flee` — a dash can out-run a fleeing chicken.
2. `r_flee_peck < r_flee` — a pecking chicken must be more distracted than a walking one.
3. `(s_max/drain)·(v_dash−v_flee) ≥ catch_slack_k·r_flee` — a full reserve closes the flee radius.
4. `turn_deg/2 < atan(eat_radius/r_flee)` — the snake can aim precisely enough to catch.
5. `length_cap < world_size_min/2` — the body never wraps the torus onto its own head.
6. `2π·v_snake/rad(turn_deg) < length_cap` — self-collision is physically reachable.
7. `ray_range + obstacle_radius_max + head_radius < world_size_min/2` — nearest-image raycast
   (and every Minkowski-inflated ray target: obstacles, chickens, rivals, eggs, corpses) is valid.
8. `r_mate ≥ 2·head_radius` — two snakes can sit at mating distance without a forced cut-off.
9. `repro_cost < repro_energy_frac·energy_max` — a snake that just qualified can pay and survive.
10. Hatchling viable: `hatch_energy_frac·energy_max > 0` and `start_length ≥` the neck-skip sum.
11. `chicken_ceiling ≥ chickens_per_snake_max · n_max` — the food ceiling covers max demand.
12. *(soft, logs a warning, doesn't fail)* `n_max` full-length bodies stay well under the smallest
    world's area.

**World / population:** `world_size 110–160` (up from the single-snake game's 60–100),
`n_start_min/max 2/4`, `n_max 6`. **Food scales with population** (rates, not absolutes, since
`Config` is frozen): `chickens_per_snake_max/min 2.0/1.0`, hard cap `chicken_ceiling 12`.

**Stamina/dash** (unchanged from the single-snake game, except `stamina_regen` — see "eased"
below): `s_max 30`, `stamina_drain 1.0`, hard `dash_min_stamina 1.0`, easy-curriculum
`dash_min_stamina_easy 0.05` / `stamina_regen_easy 0.6`, `v_dash 2.0`, `r_flee 12` (walk alert).

**Chicken FSM:** `chicken_peck_min/max 12/35`, `chicken_walk_min/max 18/45`,
`chicken_startle_steps 4`, `r_flee 12` (walk/flee alert), `r_flee_peck 2.5` (peck alert — the
stalk-and-pounce window, Pitfall 10).

**Reproduction / eggs / corpses (values that TRAINED the shipped model, i.e. the curriculum's hard
endpoint during training):** `repro_energy_frac 0.7`, `repro_length_min 10.0`, `r_mate 4.0`,
`mate_steps 4`, `repro_cost 30.0`, `repro_cooldown 120`, `egg_timer 45`, `hatch_energy_frac 0.5`,
`egg_food 25.0`, `corpse_food_per_length 4.0`. Mating curriculum easy endpoints (swept toward the
above as `hardness: 0→1`): `r_mate_easy 12.0`, `mate_steps_easy 1`, `repro_length_min_easy 6.0`.

**Eased AFTER training, runtime-only, for the persistent viewer's ecosystem sustainability**
(Pitfalls 11–12 — safe without a retrain, `repro_length_min` specifically because the eased value
stays inside the curriculum's already-swept range): `stamina_regen 0.3→0.42`, `r_mate 4.0→7.0`,
`mate_steps 4→2`, `repro_cost 30.0→18.0`, `repro_length_min 10.0→8.0`, `repro_cooldown 120→80`.
`config.py` documents both the pre- and post-easing values inline; **don't retune these further
without re-checking Pitfall 12's reasoning.**

Curriculum ramp: `hardness_warmup 0.42`, `hardness_full 0.85` (of total training steps) — governs
BOTH the stamina gate/regen and the mating gate together (`env.set_hardness`).

---

## Making changes & retraining

**Decide first: does this need a retrain?** Anything that changes what the policy senses or is
rewarded for, or the physics it acts in, DOES.

- **Needs retrain:** `sensors.py` (obs values or `OBS_DIM`), `env.py` reward/action/obs-bounds,
  `world.py` physics/stamina/reproduction/chicken-FSM/collision rules, `worldgen.py`,
  stamina/reward/`gamma`/curriculum constants — **and any constant that `sensors.observe` reads
  directly** (Pitfall 12), even if it's a "balance" constant like `repro_length_min`.
- **No retrain:** `render.py`, `watch.py` viewer loop/HUD/effects/reseed-floor, `__main__` CLI,
  fullscreen, and any config constant that is genuinely never read inside `sensors.observe` (check
  before assuming — Pitfall 12).

**Retrain recipe (reliable, ~40–75 min — slower than the single-snake game's ~15–20 min because
more entities step each frame):**
```bash
./snake train --steps 8000000 --envs 16 --reset
```
- Back up the current good model first if it's precious: `cp -r models models_good_backup`
  (gitignored). Models are NOT committed (`.gitignore`).
- Expected trajectory (the shipped model's actual run): hunting **and** reproduction bootstrap
  FAST on the easy warmup (peck-distraction chickens give an immediate catch window — eaten counts
  climbing by ~30k steps, hatchlings following within under 1M) → stamina **and** mating harden
  together from `hardness_warmup` to `hardness_full` (3.36M→6.8M at 8M total steps) with only a
  mild dip (walk-catchable pecking chickens don't strictly need a dash, so the ramp is gentler than
  the single-snake days) → ends **`ep_rew_mean` ≈ +127** at full hardness. Judge by the *end*, not
  the mid-ramp dip.
- If it collapses: check the chicken alert radii first (Pitfalls 9–10 — reactive-to-everyone or
  non-distracted-while-pecking chickens are the #1 cause of a hunting-discovery collapse in this
  world, not a PPO hyperparameter). Otherwise: lengthen `hardness_warmup`, keep lr constant, soften
  the final `stamina_regen`, or change `--seed` (PPO is high-variance).
- To keep training a good model further: `./snake train --steps 3000000` (no `--reset` → resume,
  starts fully hard — both stamina and mating — and `SyncOpponentPolicy` still runs so opponents
  track the resumed policy).

**PPO hyperparameters live in `train.py`** (`PPO(...)` for a fresh run): constant `lr 3e-4`,
`n_steps 1024`, `batch_size 256`, `n_epochs 10`, `gamma` from CFG (`0.99`), `gae_lambda 0.95`,
`ent_coef 0.01`, `target_kl 0.03`, `net_arch pi/vf [128,128]`. See Pitfalls 2, 6 before touching
these; the same net architecture is reused verbatim for `OpponentController`'s policy skeleton.

---

## Judging a model (behavioral, not just reward)

`./snake watch --headless --episodes 20` runs a **persistent** multi-snake ecosystem (never resets
on any single snake's death — same world the real viewer uses) and prints + returns a metrics
dict. A healthy multi-snake model:
- **per-snake catch rate** ≈ 10–14 items / 1000 snake-steps (confirmed ~11.9 on the shipped
  model) — computed per-snake-step, not population-summed, or it'd scale with headcount;
- **dash usage** ≈ 25–36% of live snake-steps (deliberate bursts, not constant);
- **population** sustains around 2–4 (never below `n_start_min=2`, guaranteed by the reseed floor;
  organic births on top of that, 11–30 per 10k–20k-step run — a genuinely living cycle, not just
  floor-padding);
- **deaths**: mostly `obstacle`, a real number of `snake` (cut-off kills — predation is genuinely
  emergent, not just avoidance), `self` ≈ 0 (frame-stack memory still works), `starve` rare/few;
- **births + kills + (occasional) starvations all present at once** = the full hunt → grow →
  reproduce → hatch → die → scavenge cycle is running, not just one piece of it.
Also glance at the SB3 table during training: `snake/eaten_per_window`, `snake/repro_per_window`,
`snake/hatched_per_window` climbing together (not stuck near 0) is the earliest healthy signal,
well before `ep_rew_mean` fully recovers from the mid-ramp dip. `snake/hardness` should reach 1.0.

---

## Testing

`SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest -q` (126 tests, ~8s). One
runnable `assert` per non-trivial mechanic: torus nearest-image, raycast (incl. rival/egg/corpse
ray kinds), swept/tunneling collision, self-collision reachability + straight-motion regression,
two-phase-step order-independence + head-to-head, inter-snake cut-off, corpse spawn/eat-once,
mate→egg→hatch + parent-can't-eat-own-egg + egg raid, starvation, population cap, population-scaled
food, chicken peck/walk/flee FSM + startle-freeze + peck-distraction, PBRS telescoping + set-change
zeroing, stamina gate, opponent-obs preprocessing parity, `check_env` (single- and multi-snake),
interpolation, color/ring-HUD determinism. Spread across `tests/test_torus.py`,
`test_collision.py`, `test_snake_motion.py`, `test_multisnake.py`, `test_reproduction.py`,
`test_chickens.py`, `test_worldgen.py`, `test_sensors.py`, `test_selfplay.py`, `test_env.py`,
`test_config.py`, `test_interp.py`, `test_render_smoke.py`, `test_watch_smoke.py`,
`test_train_smoke.py`. Add a test with any new mechanic; update the vision/geometry tests if you
change `head_radius`, `ray_range`, the neck-skip, or the rival hazard set. No commit attribution
lines (per repo rule).
