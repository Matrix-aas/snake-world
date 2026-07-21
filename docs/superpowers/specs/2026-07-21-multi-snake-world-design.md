# Multi-snake world — design spec

**Date:** 2026-07-21
**Branch:** `multi-snake` (model backed up to `models_good_backup/`, gitignored)
**Status:** approved design, pre-implementation

Turn the single-predator sim into a small **living ecosystem**: 2–4 snakes on one torus map,
sharing **one PPO brain** (no genetics yet), competing for chickens, killing by cut-off,
scavenging corpses, and **reproducing via eggs that need two cooperating parents**. Behavior
should look alive — pack up when it pays, split when it doesn't — and emerge from world
mechanics, not from reward hacks.

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
| `config.py` | New constants (mating, eggs, corpses, starvation, population, balance) + new invariants | yes |
| `world.py` | `World` holds a **list of snakes** + `eggs` + `corpses`; inter-snake collision; mating; hatching; starvation; corpse/egg eat. Extract per-snake state into a `Snake` dataclass. | yes |
| `worldgen.py` | Spawn N snakes at spread-out free points; scale world size + food with population | yes |
| `sensors.py` | New ray categories (`other-body`, `egg`), social channel (nearest rival), egg channel, snake-smell, repro-ready bit → new `OBS_DIM` | **yes** |
| `env.py` | Ego snake = the learner; opponents stepped by a **policy snapshot**; reward for repro/corpse/egg; terminated on ego death; obs-space bounds | yes |
| `train.py` | New callback: push policy snapshot into envs each rollout (mirror `AnnealHardness`); optional mating-curriculum; retrain recipe | recipe only |
| `render.py` | Draw N snakes (distinct colors), eggs, corpses; **per-snake concentric-ring HUD** | **no** |
| `watch.py` | `_world_of` → iterate snakes; headless eval reports population/births/kills | no (obs plumbing only) |

**Design principle:** refactor `World` so **N=1 reproduces today's single-snake behavior exactly**
(existing tests must still pass). Multi-snake is additive; the single-snake path is the N=1 case.

---

## 3. Entities & state

Extract a `Snake` dataclass from the current `World` fields:
`head_uw, head, heading, path_uw, target_length, stamina, energy, alive, dashed, death_cause,
steps, _prev_head_uw, id, color_seed, repro_cooldown`.

`World` gains:
- `snakes: list[Snake]` (replaces the single-snake fields; helpers like `body_points`,
  `move`, `check_death` become per-snake).
- `eggs`: `pos (M,2)`, `timer (M,)`, `owner_ids (M,2 int)` (the two parents; parents can't eat it).
- `corpses`: `pos (K,2)`, `food (K,)` (amount of energy/growth left; shrinks as eaten).
- chickens / obstacles: unchanged.

Torus geometry (`torus_delta`, `ray_circle_hit`, `segment_circle_hit`) is reused 1:1 — it is
already vectorized over "centers", so other snakes' body points and eggs plug in as more centers.

---

## 4. Perception (obs) — retrain-critical

A snake must read neighbors to decide fight/flee/mate, using only observable-now state.

**Vision rays** (`sensors._scan`): categories grow from
`[dist, is-obstacle, is-chicken, is-self]` → `[dist, is-obstacle, is-chicken, is-self,
is-other-body, is-egg]` (6 per ray × 9 = 54). Other snakes' bodies and eggs become visible,
Minkowski-inflated by `head_radius` like every other target (Pitfall 8).

**Social channel** — nearest *other* snake, egocentric:
`[rel_pos_fwd, rel_pos_left, their_heading_fwd, their_heading_left, size_ratio, is_dashing]` (6).
Enough for "bigger and aimed at me → flee" / "fed and near → close in to mate".

**Egg channel** — nearest egg, egocentric: `[rel_pos_fwd, rel_pos_left, is_mine]` (3). The
`is_mine` bit (egg carries `owner_ids`) is what lets guarding vs raiding diverge and stops a snake
eating its own clutch.

**Smell** repurposed — now two fields:
`[chicken_intensity, chicken_grad_fwd, chicken_grad_left, snake_intensity, snake_grad_fwd,
snake_grad_left]` (6). Snake-smell is the omnidirectional "sense a rival beyond the vision cone /
around a rock" channel — this is where smell becomes useful again. (Occlusion by obstacles as
today; line-of-sight, not diffusion — deferred.)

**Proprioception:** `[energy, length, stamina, repro_ready]` (4) — add a bit for "I'm above the
mating thresholds and off cooldown".

**Proposed `OBS_DIM` = 54 + 6 + 3 + 6 + 4 = 73** (×4 frame-stack = 292). Exact layout/bounds
finalized in the plan; `env.observation_space` low/high updated accordingly. Bounded extension →
**one retrain from scratch** (`--reset`).

---

## 5. Mechanics & rules

**Conflict — cut-off (reuse `check_death`):** every *other* snake's body points join the set of
lethal circles in the swept head test. Head into another body ⇒ that snake dies (`death_cause =
"snake"`). Head-to-head (both heads within `head_radius`+`body_radius` on the swept segment) ⇒
**both die**. No new physics — bodies are just more hazard circles. Attacking = maneuvering so a
rival's head runs into your body.

**Food — one mechanism (`try_eat` generalized):** a snake eats any *edible point within
`eat_radius`*: chickens (as today), **corpses**, **foreign eggs**. Each yields energy + growth.
Corpses hold more food than a chicken (a bigger meal, drained over one or few bites); eggs a
moderate amount. Starting values in §7, tuned on retrain.

**Reproduction:** on each step, for every unordered pair of live snakes both with
`energy > repro_energy_frac·energy_max` **and** `target_length > repro_length_min` **and** both off
`repro_cooldown`, if they stay within `r_mate` for `mate_steps` consecutive steps → lay one egg at
their midpoint; **both parents pay `repro_cost` energy** and enter `repro_cooldown`. Egg:
`egg_timer` steps, then hatches a `Snake(start_length, energy = hatch_energy)` at the egg
position with the shared brain (an opponent during training; §6). Population cap `n_max` blocks
laying if reached.

**Death & population:** add **starvation** — `energy == 0 ⇒ death` (`death_cause = "starve"`;
today energy just floors at 0). A dead snake becomes a corpse (food ∝ its length). Food supply is
scaled to the **live** snake count (§7), so more snakes ⇒ less food each ⇒ a natural carrying
capacity below `n_max`.

---

## 6. Reward (sparse — respects the Pitfalls)

Cooperation is a mechanic, but the policy must *want* the payoff, so reproduction gets a **sparse**
reward like eating — never dense shaping toward other snakes.

- `+reward_eat` for chicken / corpse / foreign egg (via the shared eat path).
- `+reward_repro` **on hatch of your egg** (deferred to hatch, not to laying, so a raided egg pays
  nothing — guarding matters).
- `+reward_death` on death (cut-off, head-to-head, starvation, self, obstacle).
- PBRS toward nearest chicken — **unchanged** (set-change zeroing per Pitfall 7).

Cooperation emerges because `+reward_repro` is **unreachable alone**; "avoid bigger snakes" emerges
because their body is lethal. No friendship bonus, no proximity shaping.

**New pitfall to watch (record in CLAUDE.md after tuning):** `reward_repro` too large ⇒ snakes
neglect hunting to breed; too small ⇒ never discovered. Plus a **discovery problem**: early
self-play opponents are near-random and won't position to mate, so `+reward_repro` never fires.
Mitigation — a **mating curriculum** (generous `r_mate` / lower thresholds during warmup, tightened
by the same hardness ramp), analogous to the stamina curriculum. Self-play symmetry helps: because
both snakes share the brain, the mating behavior bootstraps on both sides at once. Constant lr +
`target_kl` (Pitfall 3) already handle the non-stationarity of opponents improving mid-training.

---

## 7. Balance & scaling (the user's ask: don't let it get cramped or starve)

All values below are **starting guesses → calibrated on the retrain** via §9 metrics. The physical
(RL) world needs empirical tuning a paper model can't predict — leave these as knobs.

**World size** — scale so per-snake area stays roughly like today's single-snake feel (~6400 avg
area). Bump `world_size_min/max` from `60/100` to about **`110/160`** (holds all invariants:
`length_cap 24 < world_size_min/2`; `ray_range+obs_max+head 25 < world_size_min/2`). Option
(finalized in plan): size the world from `n_start` as `side ∝ base·sqrt(n_start/ref)` for constant
density; else accept mild density variance from the fixed larger range.

**Food regulation** — chickens must track the **live** population so competition is real but not
starvation. Replace the fixed `max/min_chickens` with population-scaled targets:
`max_chickens ≈ round(chickens_per_snake_max · n_alive)`,
`min_chickens ≈ round(chickens_per_snake_min · n_alive)`, clamped to a floor/ceiling.
Starting: `chickens_per_snake_max ≈ 2.0`, `chickens_per_snake_min ≈ 1.0`, and shorten
`spawn_period` (more mouths) — tune so the equilibrium population sits **below `n_max`** (food is
the population regulator). Corpses/eggs are bonus food that briefly relaxes competition.

**Obstacles** — keep today's area-scaled density. Also apply the area scaling to random-size
training worlds (today `area_mult=1.0` when `size is None`), so bigger training maps aren't sparse.

**Population** — `n_start ∈ [2,4]` (random per episode), hard cap `n_max = 6` (compute/render). Cap
chosen so N snakes' bodies occupy a small fraction of world area (no gridlock).

**Starting constants (indicative, all tunable):**
`repro_energy_frac 0.7`, `repro_length_min 10.0`, `r_mate 4.0`, `mate_steps 4`, `repro_cost 30.0`,
`repro_cooldown 120`, `egg_timer 45` (≈ a few seconds at viewer speed), `hatch_energy_frac 0.5`,
`corpse_food_per_length 4.0`, `egg_food 25.0`, `reward_repro 12.0`.

---

## 8. Training (lazy self-play — the existing stack is reused, not rewritten)

- `SnakeEnv` controls **snake 0 (ego)** — the learner. The `World` holds all snakes.
- **Opponents** step via a **policy snapshot** the env holds: a torch clone of the policy whose
  `state_dict` is pushed into every env **each rollout** through a new callback
  (`SyncOpponentPolicy`), exactly mirroring how `AnnealHardness` pushes `set_hardness`. Opponents
  act **stochastically** (sampled) for behavioral diversity. Ego's hatchlings are opponents too.
- **Episode boundary:** ego death ⇒ `terminated` (reuse today's logic) ⇒ world reset. Horizon
  truncation unchanged. Only the ego's transitions train the policy (sample-inefficient but simple
  and robust; the "all snakes learn" MARL upgrade is deferred).
- **Curriculum:** `AnnealHardness` (stamina) unchanged; add the mating curriculum (§6) on the same
  ramp. Cold start (random opponents early) is fine and matches the existing "learn to hunt first"
  warmup philosophy.
- **Obs changed ⇒ retrain from scratch** (`--reset`); `frame_stack 4` kept. PPO hyperparameters
  (constant lr `3e-4`, `n_steps 1024`, `batch_size 256`, `target_kl 0.03`, net `[128,128]`)
  unchanged — they are already tuned for a non-stationary task, which self-play is.

CPU cost: each step runs N snakes' physics + (N−1) tiny-MLP forward passes; net is `[128,128]`,
N ≤ 6 → roughly 1.5–2× slower stepping. Acceptable (retrain ~30–40 min vs ~15–20 today).

---

## 9. Rendering, viewer & judging (render = no retrain)

- **Distinct colors:** golden-angle hue palette — snake `i` hue `= (i·137.5°) mod 360`, fixed S/V
  tuned for the dark background; head/body drawn as a gradient from that hue. Deterministic from
  `snake.id`/`color_seed` so a snake keeps its color across frames.
- **Per-snake compact HUD:** a small **3-ring concentric badge** floating near each snake's head —
  outer ring = energy/food, middle = stamina/dash, inner = length→`length_cap`, each a filling arc,
  tinted the snake's color. Ego snake gets a larger version. Replaces the current single-snake bar
  HUD (which doesn't scale to 6 snakes). Eggs rendered with a pulse + hatch-countdown; corpses as
  food piles.
- **Viewer plumbing:** `watch._world_of` iterates `world.snakes`; the viewer still drives the ego
  env but renders all snakes. Interpolation applies per snake.
- **Headless eval (`watch --headless`)** reports new ecosystem metrics: per-snake catch rate, dash
  usage, births (hatches), kills (cut-off / head-to-head), starvations, and population over time.
  A healthy world: population oscillates around a carrying capacity below `n_max`, some births and
  some kills, hunting still works (catch rate per snake in a sane band), self-deaths ≈ 0.

---

## 10. New invariants (`assert_invariants`)

1. All existing invariants still hold with the bigger world / same `length_cap`.
2. `r_mate ≥ 2·head_radius + margin` — two snakes can sit at mating distance without a forced
   cut-off (heads near, not head-into-body).
3. `repro_cost < repro_energy_frac·energy_max` — a snake that just qualified can pay and survive.
4. `hatch_energy_frac·energy_max > 0` and hatchling `start_length ≥ neck-skip` (reuse existing).
5. `n_max` bodies fit: `n_max · (length_cap · body_radius·2) ≪ world_area` (soft, log a warning).
6. Population-scaled `max_chickens` clamp keeps food density feasible for `_free_point` sampling.

---

## 11. Implementation phases (for the plan)

1. **World refactor:** extract `Snake`; `World` holds `snakes[]` (+ empty `eggs`/`corpses`); make
   per-snake `move`/`body_points`/`check_death`; **N=1 regression: all current tests green.**
2. **Inter-snake + scavenging:** cut-off death vs other bodies, head-to-head, corpse spawn + eat.
3. **Reproduction & population:** mating→egg→hatch, egg eat/ownership, starvation death, `n_max`,
   population-scaled food, bigger world (config + worldgen + invariants).
4. **Sensors:** new ray categories + social/egg channels + snake-smell + repro-ready; new
   `OBS_DIM`; env obs-space bounds.
5. **Env + training:** ego/opponent split, `SyncOpponentPolicy` callback, repro/corpse/egg rewards,
   mating curriculum; retrain recipe.
6. **Render + viewer:** golden-angle colors, 3-ring badge, eggs/corpses; `watch` multi-snake + new
   headless metrics.
7. **Tests + invariants:** one runnable assert per new mechanic (cut-off, head-to-head, corpse
   eat, mate→egg→hatch, parent-can't-eat-own-egg, egg raid, starvation, population cap, food
   scaling, color determinism, ring-badge geometry); update vision tests for new categories.
8. **Retrain, judge, tune, document:** run the retrain, judge behaviorally (§9), calibrate §7
   knobs, and **update CLAUDE.md** (architecture table, new pitfalls: reproduction reward
   balance + mating discovery, self-play plumbing, obs dim).

---

## 12. Retrain requirement

**This needs a full retrain from scratch** (obs, reward, physics all change). Model already backed
up to `models_good_backup/` (gitignored). Recipe (indicative): `./snake train --steps 8000000
--envs 16 --reset` (more steps than today — the task is harder: more to sense, opponents, mating
discovery). Judge by the §9 metrics at full hardness, not mid-ramp (Pitfall 4).
