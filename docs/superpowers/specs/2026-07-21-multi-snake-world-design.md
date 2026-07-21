# Multi-snake world — design spec

**Date:** 2026-07-21
**Branch:** `multi-snake` (model backed up to `models_good_backup/`, gitignored)
**Status:** approved design, revised after design review (2026-07-21), pre-implementation

Turn the single-predator sim into a small **living ecosystem**: 2–4 snakes on one torus map,
sharing **one PPO brain** (no genetics yet), competing for chickens, killing by cut-off,
scavenging corpses, and **reproducing via eggs that need two cooperating parents**. Behavior
should look alive — pack up when it pays, split when it doesn't — and emerge from world
mechanics, not from reward hacks.

> Revised after a design review that caught two design-breakers (opponent obs preprocessing;
> ego-death world resets making the ecosystem unobservable) and several gaps. Their resolutions
> are folded into §5–§9 and the phase plan (§11). See §13 for the review-resolution log.

---

## 1. North star & scope

**Hybrid (chosen):** one shared PPO policy drives every snake (self-play); population dynamics
(birth via eggs, death via cut-off/starvation) are **world rules**, not genetics. All snakes
think alike; the "ecosystem" is population flux, not evolving intelligence.

**In v1:**
- 2–4 starting snakes, dynamic population up to a hard cap (~6).
- Snakes perceive other snakes + eggs (vision + smell) → **observation changes → one retrain**.
- Conflict = **cut-off death** (head into another body = you die); corpse → food.
- Reproduction = **two well-fed snakes mate → egg**; egg lies N steps → hatches (same brain).
- Eggs are **edible/raidable** → guarding (one feeds, one guards) + raids on rival clutches.
- **Starvation death** (energy→0) + food scaled to population → self-regulating carrying capacity.
- Per-snake compact ring HUD + distinct auto-generated colors (render only).

**Deferred (explicitly NOT in v1):** genetic evolution / heredity; recurrent memory for
persistent alliances, reputation, grudges (v1 cooperation is *conditional on observable state*,
not contractual); scent-trail territories; pack-hunting of special "big prey"; all snakes
learning simultaneously (v1 trains one ego snake, opponents run a policy snapshot).

### The ecosystem loop

```
hunt chickens ──► energy/growth ──► meet a fed partner ──► EGG (both pay energy)
      ▲                                                        │
      │                                              lies N steps, VULNERABLE
  starvation / death                                ┌──────────┴──────────┐
  regulate population                          parents guard         rivals raid (egg = food)
      │                                             └──────────┬──────────┘
      └──── corpse = food ◄── cut-off (head into a body) ◄──── hatchling (same brain)
```

**Two honest constraints that shaped everything (do not forget):**
1. **Cooperation is not free.** With one shared brain and individual reward, selfishness is the
   default gradient. Teaming up emerges only because a *reward that requires a partner*
   (reproduction) is unreachable alone. We add opportunity via mechanics; a **sparse** reward
   makes the policy take it — never dense "cooperation bonus" shaping (see Pitfalls in CLAUDE.md).
2. **No per-individual memory.** One shared brain + frame-stack (≈4 steps) means a snake cannot
   remember specific individuals. "Who to fight/befriend" is a function of *observable-now* state
   (rival size, heading, whether it's dashing at me, our relative energy). Persistent named
   alliances need `RecurrentPPO` — deferred.

---

## 2. Architecture changes (per file)

| File | Change | Retrain? |
|---|---|---|
| `config.py` | New constants (mating, eggs, corpses, starvation, population, balance **rates**) + new invariants | yes |
| `world.py` | `World` holds a **list of `Snake`** + `eggs` + `corpses`; **two-phase step**; inter-snake collision; mating; hatching; starvation; corpse/egg eat | yes |
| `worldgen.py` | Spawn N snakes at spread-out free points; scale world size + food with population | yes |
| `sensors.py` | New ray categories (`other-body`, `egg`), social channel (nearest rival + presence bit), egg channel (+ presence bit), snake-smell, repro-ready bit → new `OBS_DIM`; all channels normalized/clipped | **yes** |
| `env.py` | Ego = the learner; opponents stepped by a **preprocessed policy snapshot**; reward for repro/corpse/egg; ego-death terminates the training episode; obs-space bounds reworked | yes |
| `train.py` | New callback pushes policy `state_dict` **+ `obs_rms`** into envs each rollout (mirror `AnnealHardness`); mating curriculum; retrain recipe | recipe only |
| `render.py` | Draw N snakes (distinct colors), eggs, corpses; **per-snake concentric-ring HUD** | **no** |
| `watch.py` | **Persistent world** decoupled from ego death; camera follows any live snake; multi-snake render; new headless ecosystem metrics | no (obs plumbing only) |

**Design principle:** refactor `World` so **N=1 behaves identically to today** (same seed → same
trajectory) — verified by a behavioral regression, not by leaving test files untouched (see I3/§11).

---

## 3. Entities & state

Extract a `Snake` dataclass from the current per-instance `World` fields (`world.py:70-81`):
`head_uw, head, heading, path_uw, target_length, stamina, energy, alive, dashed, death_cause,
steps, _prev_head_uw` + new: `id, color_seed, repro_cooldown, frame_hist` (opponent obs history,
§8). World-level state (`rng, size`, chickens, obstacles) stays on `World`.

`World` gains:
- `snakes: list[Snake]` (per-snake `move`, `body_points`, hazard set become methods on/over `Snake`).
- `eggs`: `pos (M,2)`, `timer (M,)`, `owner_ids (M,2 int)` (the two parents; parents can't eat it).
- `corpses`: `pos (K,2)`, `food (K,)` (energy/growth left; shrinks as eaten).
- chickens / obstacles: unchanged.

Torus geometry (`torus_delta`, `ray_circle_hit`, `segment_circle_hit`) is reused 1:1 — it is
already vectorized over "centers", so other snakes' body points and eggs plug in as more centers.

---

## 4. Perception (obs) — retrain-critical

A snake reads neighbors using only observable-now state. **Every new channel is normalized and
clipped so the obs means the same thing across the random 110–160 world sizes** (CLAUDE.md's
egocentric/size-agnostic property; today everything is `/ray_range` or `/cap`).

**Vision rays** (`sensors._scan`): categories grow from
`[dist, is-obstacle, is-chicken, is-self]` → `[dist, is-obstacle, is-chicken, is-self,
is-other-body, is-egg]` (6 per ray × 9 = **54**). Other snakes' bodies and eggs become visible,
Minkowski-inflated by `head_radius` like every other target (Pitfall 8).

**Social channel** — nearest *other* snake, egocentric, **7** floats:
`[has_rival, rel_pos_fwd, rel_pos_left, their_heading_fwd, their_heading_left, size_ratio,
is_dashing]`. `rel_pos_*` normalized by `ray_range` and clipped to `[-1,1]`. `has_rival` presence
bit disambiguates "no rival" from "rival at range 0". Enough for "bigger and aimed at me → flee" /
"fed and near → close in to mate".

**Egg channel** — nearest egg, egocentric, **4** floats:
`[has_egg, rel_pos_fwd, rel_pos_left, is_mine]` (positions normalized as above). `is_mine` (egg
carries `owner_ids`) lets guarding vs raiding diverge and stops a snake eating its own clutch.

**Smell** repurposed — **6** floats, two fields:
`[chicken_intensity, chicken_grad_fwd, chicken_grad_left, snake_intensity, snake_grad_fwd,
snake_grad_left]`. Snake-smell is the omnidirectional "sense a rival beyond the vision cone /
around a rock" channel — where smell becomes useful again. Line-of-sight occlusion as today
(diffusion deferred). New intensity/gradient bounds added to `observation_space` (§7).

**Proprioception:** `[energy, length, stamina, repro_ready]` (**4**) — `repro_ready` = above the
mating thresholds and off cooldown.

**`OBS_DIM` = 54 + 7 + 4 + 6 + 4 = 75** (×4 frame-stack = 300). `env.observation_space` low/high
updated per channel (rays/one-hots/proprio in `[0,1]`; rel-pos in `[-1,1]`; smell bounds from the
population ceiling, §7). Bounded extension → **one retrain from scratch** (`--reset`).

---

## 5. Mechanics & rules

**Two-phase step (required for determinism).** Today `world.step` is move→eat→death for one snake
(`world.py:318-325`). With N snakes a sequential move-and-resolve is order-dependent and breaks
"head-to-head ⇒ both die" (whoever moves second hits a stale body). New order per world step:
1. **Move all** live snakes (steer + dash + path update).
2. **Resolve deaths** for all snakes against post-move state (obstacles, self, other snakes) —
   simultaneously, so results are order-independent.
3. Update chickens; resolve eating (chickens/corpses/eggs); mating; egg timers/hatching; energy
   decay + starvation; spawn. RNG for opponent action sampling and chicken motion is per-world
   seeded for reproducible episodes.

**Conflict — cut-off.** Each snake's **other-hazard set** = every *other* live snake's **head
circle + full body with NO neck-skip**. (The self neck-skip in `body_points_uw:132-135` is a
self-collision concession; applying it to rivals would make the region behind a rival's head
non-lethal — you must NOT reuse the self set for rivals. Head-to-head needs the head as a target,
which `body_points` never returns — include it explicitly.) A snake is excluded from its own other
set. Head into another body/head on the swept segment ⇒ that snake dies (`death_cause = "snake"`);
mutual overlap ⇒ **both die**. No new geometry — just more hazard circles in `segment_circle_hit`.

**Food — one mechanism.** Generalize `try_eat`: a snake eats any *edible point within `eat_radius`*
— chickens (as today), **corpses**, **foreign eggs**. Energy/growth scale with the food consumed;
the `+reward_eat` bonus fires **once per item consumed, not per bite** (I5 — else a big corpse pays
`reward_eat` several times and out-values chasing chickens). A corpse holds `corpse_food_per_length
· length` food, drained over one/few bites; foreign egg = `egg_food`.

**Reproduction.** Each step, for every unordered pair of live snakes both with
`energy > repro_energy_frac·energy_max` **and** `target_length > repro_length_min` **and** both off
`repro_cooldown`, if they stay within `r_mate` for `mate_steps` consecutive steps → lay one egg at
their midpoint; **both parents pay `repro_cost` energy** and enter `repro_cooldown`. Egg lives
`egg_timer` steps → hatches a `Snake(start_length, energy = hatch_energy_frac·energy_max)` at the
egg position with the shared brain (an opponent during training; §8). Cap `n_max` blocks laying.

**Death & population.** Add **starvation**: `energy == 0 ⇒ death` (`death_cause = "starve"`; today
energy just floors at 0). A dead snake becomes a corpse (food ∝ length). Food scaled to **live**
snake count (§7) ⇒ more snakes ⇒ less food each ⇒ carrying capacity below `n_max`.

---

## 6. Reward (sparse — respects the Pitfalls)

Cooperation is a mechanic, but the policy must *want* the payoff, so reproduction gets a **sparse**
reward like eating — never dense shaping toward other snakes.

- `+reward_eat` **once per item** (chicken / corpse / foreign egg) via the shared eat path.
- `+reward_repro` **on hatch of your egg** (deferred to hatch, not laying → a raided egg pays
  nothing → guarding matters).
- `+reward_death` on death (cut-off, head-to-head, starvation, self, obstacle).
- PBRS toward nearest chicken — **unchanged** (set-change zeroing per Pitfall 7).

Cooperation emerges because `+reward_repro` is **unreachable alone**; "avoid bigger snakes" emerges
because their body is lethal. No friendship bonus, no proximity shaping.

**Known asymmetry (I7):** being cut off is a hard `−10`, but cutting a rival off yields only the
weak, downstream corpse reward → with only the ego training, the policy learns strong cut-off
**avoidance** and near-zero offensive intent, so **kills may be rare-emergent**. v1 stance: accept
rare kills and set §9 expectations accordingly; **measure** kill rate, and only if it's ~0 and we
want more predation, add a *tiny, gated* direct reward for a rival dying to your body (risks
suicidal ramming — keep it minimal). Do not add it preemptively.

**New pitfalls to record in CLAUDE.md after tuning:**
- `reward_repro` too large ⇒ snakes neglect hunting to breed; too small ⇒ never discovered.
  Guarding's opportunity cost is tight (`reward_repro 12` vs ~45 idle steps ≈ 5–10 of hunting).
- **Mating discovery problem (highest risk — see §11 spike):** at cold start ego *and* opponent are
  the same near-random net, so the joint event (both fed, off cooldown, within `r_mate` for
  `mate_steps` consecutive steps) essentially never fires, and `+reward_repro` is discounted ~0.63
  over the 45-step hatch delay plus forgone hunting. A random-walking opponent won't linger, and
  lingering-to-mate is exactly when cut-off risk peaks with no memory to tell "approach to mate"
  from "approach that gets me killed." **Mitigation:** a mating curriculum on the same hardness
  ramp — early warmup uses `mate_steps=1` + very large `r_mate` + low thresholds, tightened to the
  real values by `hardness_full`. Self-play symmetry helps (behavior bootstraps on both sides at
  once). **Fallback if the curriculum can't bootstrap it:** an auto-lay heuristic during the
  earliest warmup (world lays an egg for any qualifying nearby pair) to seed the `+reward_repro`
  signal, then withdraw it. Constant lr + `target_kl` (Pitfall 3) handle opponents improving.

---

## 7. Balance & scaling (don't let it get cramped or starve)

All values are **starting guesses → calibrated on the retrain** via §9 metrics. Leave them as knobs.

**Frozen config holds RATES, not absolutes (I4).** `Config` is a frozen dataclass, so the
population-dependent food target can't be a constant. Store per-snake *rates*; compute the live
target inside `maybe_spawn` from `n_alive`.

**World size** — scale so per-snake area ≈ today's single-snake feel (~6400 avg area). Bump
`world_size_min/max` from `60/100` to **`110/160`** (holds every invariant: `length_cap 24 <
world_size_min/2 = 55`; `ray_range+obstacle_r_max+head 25 < 55`). Option (decide in plan): size the
world from `n_start` as `side ∝ base·sqrt(n_start/ref)` for constant density; else accept mild
variance from the larger fixed range.

**Food regulation** — chickens track the **live** population so competition is real but not
starvation. `max_target = clamp(round(chickens_per_snake_max · n_alive), floor, ceiling)`,
`min_target = clamp(round(chickens_per_snake_min · n_alive), 1, ceiling)`, computed in `maybe_spawn`.
Starting rates: `chickens_per_snake_max ≈ 2.0`, `chickens_per_snake_min ≈ 1.0`; shorten
`spawn_period`. Tune so equilibrium population sits **below `n_max`** (food is the regulator).
The **obs smell bound** uses the fixed `chicken_ceiling` (not the dynamic target), and a new bound
covers snake-smell intensity (≤ `n_max`). `worldgen.py:29` initial-chicken draw uses the ceiling.

**Obstacles** — keep today's area-scaled density; also apply area scaling to random-size *training*
worlds (today `area_mult=1.0` when `size is None`), so bigger training maps aren't sparse.

**Population** — `n_start ∈ [2,4]` (random per episode), hard cap `n_max = 6` (compute/render).

**Starting constants (indicative, all tunable):**
`repro_energy_frac 0.7`, `repro_length_min 10.0`, `r_mate 4.0`, `mate_steps 4`, `repro_cost 30.0`,
`repro_cooldown 120`, `egg_timer 45` (≈ a few seconds at viewer speed), `hatch_energy_frac 0.5`,
`corpse_food_per_length 4.0`, `egg_food 25.0`, `reward_repro 12.0`, `chicken_ceiling 12`.

---

## 8. Training (lazy self-play — reuse the stack; the obs pipeline is the tricky part)

- `SnakeEnv` controls **snake 0 (ego)** — the learner. `World` holds all snakes.
- **Opponents step via a preprocessed policy snapshot (C1 — the design-breaker the review caught).**
  The ego's obs is normalized by `VecNormalize` and stacked by `VecFrameStack`, both **outside** the
  env in the main process (`train.py:24-25`). Opponents run *inside* the env, where only the raw
  single-frame `observe(world)` exists. So the env MUST reproduce the preprocessing per opponent:
  1. Each opponent keeps a **4-frame ring buffer** (`Snake.frame_hist`), cold-padded on hatch,
     dropped on death.
  2. The sync callback pushes **both** the policy `state_dict` **and the current `obs_rms`
     (mean/var)** into every env each rollout (`env_method("set_opponent_policy", state_dict,
     obs_rms)`), mirroring `AnnealHardness`. (Using the ego's `obs_rms` for opponents is fine — same
     `observe`, ~same distribution.)
  3. Per opponent per step: `action = policy(normalize(stack(raw_obs)))`, sampled (stochastic) for
     diversity, from a per-world-seeded RNG. This is ~30–40 lines of bookkeeping — Phase 5 is scoped
     for it, NOT "just push the state_dict."
- **Training episode boundary:** ego death ⇒ `terminated` ⇒ world reset (reuse today's logic).
  Horizon truncation unchanged. Only ego transitions train the policy (sample-inefficient but
  simple; "all snakes learn" MARL upgrade deferred). Ego's hatchlings are opponents.
- **Viewer/eval worlds are PERSISTENT and decoupled from ego death (C2 — §9).** Training resets on
  ego death; the *viewer and headless eval do not*, or the ecosystem (egg_timer 45, cooldown 120,
  carrying capacity over thousands of steps) is never observable.
- **Curriculum:** `AnnealHardness` (stamina) unchanged; add the mating curriculum (§6) on the same
  ramp. Cold start (random opponents early) matches the existing "learn to hunt first" warmup.
- **Obs changed ⇒ retrain from scratch** (`--reset`); `frame_stack 4` kept. PPO hyperparameters
  (constant lr `3e-4`, `n_steps 1024`, `batch_size 256`, `target_kl 0.03`, net `[128,128]`)
  unchanged — already tuned for a non-stationary task, which self-play is.

CPU cost: per step runs N snakes' Python-level physics + (N−1) normalize+stack+forward passes.
Budget **2–3×** slower stepping (retrain ~40–60 min vs ~15–20 today) — not blocking.

---

## 9. Rendering, viewer & judging (render = no retrain)

- **Distinct colors:** golden-angle hue palette — snake `i` hue `= (i·137.5°) mod 360`, fixed S/V
  for the dark background; head/body a gradient from that hue. Deterministic from `snake.color_seed`
  so a snake keeps its color across frames.
- **Per-snake compact HUD:** a small **3-ring concentric badge** floating near each snake's head —
  outer = energy/food, middle = stamina/dash, inner = length→`length_cap`, each a filling arc,
  tinted the snake's color. Ego gets a larger version. Replaces the single-snake bar HUD (doesn't
  scale to 6). Eggs: pulse + hatch-countdown; corpses: food piles that shrink as eaten.
- **Persistent world (C2):** the viewer and `watch --headless` run a world that does **not** reset
  on any single snake's death; the camera follows a chosen live snake (or a detached overview). The
  ego-centric env exists only for training. Interpolation applies per snake.
- **Headless eval metrics (`watch.py:run_headless`)** extended: per-snake catch rate, dash usage;
  ecosystem — births (hatches), kills (cut-off / head-to-head), starvations, and population over
  time. Replace the hardcoded `deaths = {"obstacle":0,"self":0}` (`watch.py:90`) with all causes
  (`obstacle, self, snake, starve`). Healthy world: population oscillates around a carrying capacity
  below `n_max`; hunting still works (per-snake catch rate in a sane band); self-deaths ≈ 0; kills
  present but possibly rare (I7).

---

## 10. New invariants (`assert_invariants`)

1. All existing invariants still hold with the bigger world / same `length_cap` (verified: §7).
2. `r_mate ≥ 2·head_radius + margin` — two snakes can sit at mating distance without a forced
   cut-off (heads near, not head-into-body).
3. `repro_cost < repro_energy_frac·energy_max` — a snake that just qualified can pay and survive.
4. Hatchling `start_length ≥ neck-skip` (reuse existing) and `hatch_energy_frac·energy_max > 0`.
5. `chicken_ceiling ≥ chickens_per_snake_max · n_max` won't over-pack `_free_point` sampling (soft).
6. `n_max` bodies occupy ≪ world area (soft, log a warning) — trivially true at the starting values.

---

## 11. Implementation phases (spike the unknowns FIRST — review recommendation)

**Phase 0 — De-risk spikes (before the invasive refactor).** Two throwaway spikes; their results
can change the design, so run them first:
- **0a Opponent obs plumbing:** prove an in-env opponent driven by a snapshot with correct
  `normalize(stack(raw))` produces sane actions (load the *current* single-snake model, drive a
  second dummy snake with it in a scratch world, eyeball it hunts). Validates C1's fix end to end.
- **0b Mating discovery:** in a minimal 2-snake toy, check the mating curriculum (or the auto-lay
  fallback) actually produces `+reward_repro` events early in training. If neither bootstraps it,
  revisit the reproduction design before building it.

**Milestone A — Simulation (no RL retrain; fully testable at N=1 and multi-snake):**
1. **World refactor:** extract `Snake`; `World.snakes[]` (+ empty `eggs`/`corpses`); two-phase step;
   per-snake `move`/`body_points`/hazard-set. Update `env`/`sensors`/`watch` callers and **rewrite
   affected tests to the `snakes[0]` API** (I3 — don't claim "tests untouched"; guarantee is a
   *behavioral regression*: same seed → identical N=1 trajectory/metrics).
2. **Inter-snake + scavenging:** cut-off vs other-hazard set (head + full body, no neck-skip, self
   excluded), head-to-head both-die, corpse spawn + eat (once-per-item reward).
3. **Reproduction & population:** mating→egg→hatch, egg eat/ownership, starvation death, `n_max`,
   population-scaled food, bigger world (config rates + worldgen + invariants).

**Milestone B — RL (gated on Phase 0 results):**
4. **Sensors:** new ray categories + social/egg channels (normalized + presence bits) + snake-smell
   + repro-ready; new `OBS_DIM`; env obs-space bounds reworked (smell bound from `chicken_ceiling`).
5. **Env + training:** ego/opponent split with **full opponent obs preprocessing** (frame ring +
   pushed `obs_rms` + normalize/stack), `SyncOpponentPolicy` callback, repro/corpse/egg rewards,
   mating curriculum.
6. **Render + viewer:** golden-angle colors, 3-ring badge, eggs/corpses; **persistent** `watch`
   world + camera-follow + new headless metrics.
7. **Tests + invariants:** one runnable assert per new mechanic (two-phase determinism, cut-off,
   head-to-head, corpse eat once-per-item, mate→egg→hatch, parent-can't-eat-own-egg, egg raid,
   starvation, population cap, food scaling, obs normalization/bounds, color determinism, ring-badge
   geometry); update vision tests for the new categories.
8. **Retrain, judge, tune, document:** retrain, judge behaviorally (§9) at full hardness (Pitfall 4),
   calibrate §7 knobs + `reward_repro`, measure kill rate (I7), and **update CLAUDE.md**
   (architecture table; new pitfalls: reproduction-reward balance, mating discovery, opponent obs
   preprocessing, self-play plumbing, obs dim, persistent-viewer decoupling).

**Go/no-go gates:** after Phase 0 (design still valid?), after Milestone A (sim correct at N=1 +
multi-snake, all tests green?), and after the retrain (behavior healthy per §9?).

---

## 12. Retrain requirement

**Full retrain from scratch** (obs, reward, physics all change). Model already backed up to
`models_good_backup/` (gitignored). `watch.py` must handle loading a model whose obs dim changed
(no silent shape mismatch — fail loudly if an old model is loaded against the new obs). Recipe
(indicative): `./snake train --steps 8000000 --envs 16 --reset` (more steps — harder task: more to
sense, opponents, mating discovery). Judge by §9 metrics at full hardness, not mid-ramp (Pitfall 4).

---

## 13. Review-resolution log (2026-07-21 design review)

- **C1 (opponents saw raw obs):** resolved — §8 opponent preprocessing (frame ring + pushed
  `obs_rms` + normalize/stack); Phase 0a spike; Phase 5 re-scoped.
- **C2 (ego death resets ecosystem):** resolved — §8/§9 persistent viewer/eval world decoupled from
  ego death; training keeps ego-centric episodes.
- **I1 (step ordering / simultaneous cut-off):** resolved — §5 two-phase step + per-world RNG.
- **I2 (rival hazard set):** resolved — §5 head + full body, NO neck-skip, self excluded.
- **I3 (N=1 "tests untouched" claim):** resolved — §2/§11 guarantee is behavioral regression;
  affected tests rewritten to the new API.
- **I4 (obs bounds/normalization/presence + `max_chickens` bound):** resolved — §4 normalized+clipped
  channels + presence bits; §7 rates-in-config, dynamic target, smell bound from `chicken_ceiling`.
- **I5 (corpse multi-bite vs count reward):** resolved — §5/§6 reward once per item, energy per food.
- **I6 (mating discovery):** resolved — §6 curriculum + auto-lay fallback; Phase 0b spike + gate.
- **I7 (kill incentive asymmetry):** resolved — §6 accept rare-emergent + measure; §9 expectation
  softened; optional tiny gated kill reward only if needed.
- **M1–M4:** CPU budget 2–3× (§8); deaths-dict causes (§9); soft invariant (§10.6); guarding
  opportunity-cost pitfall (§6).
