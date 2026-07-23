# CLAUDE.md — Snake-RL project guide

A PPO-trained multi-snake predator **genetic ecosystem** (**v3**) in a continuous 2D torus world.
3–6 founder snakes (population flux up to `n_max=12`) share **one PPO brain**, but every snake
carries a heritable **9-gene genome** (size, metabolism, speed, stamina, senses, lifespan +
behavioral aggression/kin_care/boldness), a **sex**, an **age** and a genetically-set **lifespan** —
so the *same* brain, **conditioned on each snake's own genome**, drives predators, ambushers,
sprinters and kin-breeders differently. They hunt fleeing/pecking chickens by egocentric **sight**
(11 vision rays, per-snake range) + **smell** + **vibration**, modulate their **cruise speed** (stop
→ full) and burst with a stamina-limited **dash**, **stalk/ambush** (prey fear a snake in proportion
to its speed — a motionless snake is invisible to them), thread around **solid** obstacles
(rocks/trees are impassable, NOT lethal — dashing into one just *stuns*), kill each other by
**cut-off** (leaving a corpse to scavenge), age and die, and **reproduce** through a proper
**courtship → female-lays → guard** cycle whose egg carries the **crossover+mutation child genome +
maternal lineage** — a living, self-sustaining, *evolving* population meant to run forever as a
screensaver. **Evolution is a runtime process** (genome inheritance + natural selection over the
never-resetting world), NOT something PPO optimizes: PPO only produces a brain competent *for any
genome*. Trained headless (SB3 PPO, CPU), watched in a fullscreen pygame viewer. **The behavior is
meant to look alive** — stalk, deliberate pounce, court/guard/raid eggs, weave through clutter — and
to let you *watch lineages diverge*, not to be optimal.

> **This file is the living memory of the project. Keep it current.** When you change the
> reward, the observation, the physics, the training recipe, or the hyperparameters — update
> the relevant section here *in the same change*, especially the **Pitfalls** and
> **What needs a retrain** lists. Future-you will rely on it to avoid re-learning the hard way.

---

## Run it

```bash
./snake watch                 # fullscreen viewer (a v3 model must be RETRAINED first — none ships yet)
./snake watch --windowed      # windowed
./snake watch --headless --episodes 20   # persistent-ecosystem eval, prints + returns metrics
./snake train --steps 8000000 --envs 16 --reset   # train from scratch (v3: budget MORE wall-clock — 12 snakes + bigger obs/world; time PENDING the retrain)
```

> **v3 has NOT been retrained yet** — Phase A (genome/sex/aging/courtship/no-ego/obs-143/n_max=12)
> is code-complete and feeds a **from-scratch** retrain. Any `models/snake.zip` on disk is a stale
> v2 checkpoint and is **not loadable** (obs 113→143, action-context, physics all changed).

`./snake` is a launcher that creates `.venv` and installs deps on first run — never touch pip
directly. `watch` picks a **random map every launch** (`--seed N` for a fixed one) and fits the world
to the screen (`_screen_fit_world_size(short=86.4)`); the v3 world is bigger (`world_size 180–260`),
so a future Phase-B camera (pan/zoom/follow) is planned. Watch keys: `SPACE` pause · `N` new
persistent world · `S` vision rays · `H` ring HUD (vision rays + ring HUD are **OFF by default** —
S/H toggle them on) · `↑/↓` (or `+/-`) sim speed · `ESC` quit. (No deterministic toggle: the viewer
is a **no-ego, all-eggs** world — there is no privileged snake; it drives **every** snake through the
same stochastic self-play controller, and the camera falls back to overview until an egg hatches.)

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
| `snake_rl/config.py` | ALL constants (frozen `Config`) incl. **gene→stat interpolation ranges** + evolution constants; `assert_invariants` **and** `assert_invariants_over_genome` (HARD precision+raycast gates across the whole gene box; stamina/self-collision now SOFT) | usually yes |
| `snake_rl/genome.py` | **(v3, pure — no sim state)** the 9-gene genome: `sample_genome`/`crossover`/`mutate`/`relatedness` + `resolve_phenotype(genome, cfg) → Phenotype` namedtuple (per-snake `max_length, turn_deg, v_snake, v_dash, s_max, stamina_regen, ray_range, smell_reach, energy_decay, max_lifespan_base`) | **yes** (genome is observed + drives physics) |
| `snake_rl/world.py` | torus geometry, `Snake` dataclass (**+genome/sex/age/max_lifespan/lineage/phenotype**), `World.snakes[]`, **per-snake (phenotype-driven) graduated-speed** motion + dash/stamina + **stun**, **solid-slide collision** (`_slide`: obstacles + own body impassable non-lethal; dash into a solid ⇒ stun), **`no_ego`-bifurcated two-phase `step()`** (move-with-slide all → resolve **rival-only** deaths → land sky-drops/chickens/eat/energy/**starvation+aging**/spawn/**sex-gated mating**/hatching), inter-snake cut-off + corpses, speed-scaled peck/walk/flee chicken FSM, **courtship→female-lays→guard reproduction** (egg carries child `genome`+`lineage`, `eaten_eggs` returned) + **guaranteed arrival eggs** (`spawn_egg`, fresh founder genome), sky-drop chicken arrivals | yes (dynamics) |
| `snake_rl/worldgen.py` | random world (size, obstacles, N spread-out founders each with a **freshly-sampled genome + random sex**, initial chickens); **`arrivals=True`**: founders ARRIVE via `spawn_egg`, chickens drop from the sky; **`ego_live`**: `True` (training) keeps one live gradient-ego, `False` (viewer) starts with **zero live snakes, all eggs** (`no_ego`) | yes |
| `snake_rl/sensors.py` | vectorized raycast (**11 rays × 9**: 7 hit-kind one-hots + un-masked **obstacle-clearance** + **per-ray target-motion**) + **social 11** (relatedness/rival-energy/repro/sex) + egg + smell + **vibration 3** + **proprio 17** (per-snake normalizers + sex/age_frac/stun + **own genome ×9**) → **143-float** per-snake observation; **all normalizers per-snake** (own `s_max`/`max_length`/`v_dash`/`ray_range`) | **yes** (obs) |
| `snake_rl/selfplay.py` | `OpponentController` — drives in-env opponents from a synced policy snapshot, replicating the exact `VecFrameStack`+`VecNormalize` preprocessing the SB3 learner sees (obs width + `MultiDiscrete([4,3,2])` action follow `OBS_DIM`, no hardcodes) | **yes** (self-play plumbing, mirrors sensors) |
| `snake_rl/env.py` | Gymnasium env: **one live gradient-ego** (snake 0, zero special powers) + opponents via `OpponentController`; **`MultiDiscrete([4,3,2])`** action; **sparse** reward (eat/reproduce-at-hatch/flat-death + chicken-PBRS + **egg-lost hook**, default 0); dual curriculum (`set_hardness`: stamina **and** size-relative mating); **143-float** obs bounds; **per-reset genome+sex domain-randomization** (via worldgen) | yes (reward/obs) |
| `snake_rl/train.py` | SB3 PPO, VecEnv stack, `AnnealHardness` + `SyncOpponentPolicy` callbacks, checkpoints | recipe only |
| `snake_rl/render.py` | pygame drawing: sprites, per-snake ring HUD, eggs/corpses, blood/gore, sprite-sheet animations, sky-drop chicken arrival animation (`_draw_arrivals`) | **no** (visual only; Phase-B camera/genome-color/inspector pending) |
| `snake_rl/watch.py` | load model, build a **persistent no-ego all-eggs** world (never resets on any single death, `ego_live=False`), **egg-based reseed floor**, interpolated viewer (overview-until-hatch fallback), headless ecosystem eval | no (except obs plumbing) |
| `snake_rl/__main__.py` | CLI dispatch | no |

Data flow (training): `worldgen → World` (N founders, per-snake physics from genome) →
`sensors.observe(world, snake)` per snake → `env` (gradient-ego obs/reward/done; opponents driven by
`selfplay.OpponentController`) → PPO. Data flow (viewer/headless): `watch` builds **one** persistent
**no-ego** `World` (starts all-eggs) + **one** `OpponentController` synced to it, and drives **every**
snake through that controller; nothing resets on death, only individual snakes hatch/die/reseed.
**Evolution runs here at runtime** — eggs inherit `crossover+mutation`, selection = who survives and
breeds. PPO never sees it; it only trains a genome-competent brain.

**Vec-wrapper stack** (still applies, matters for accessors): `VecNormalize( VecFrameStack(
DummyVecEnv([ Monitor(SnakeEnv) ]) ) )`. The underlying world is
`vec.venv.venv.envs[0].unwrapped.world` (see `watch._world_of`).

---

## The RL design (and why each piece is the way it is)

- **Algorithm:** SB3 `PPO`, `MlpPolicy`, **CPU**, 16 parallel random worlds (`n_steps=1024`,
  `batch_size=256` → 64 minibatches over a 16384 buffer — unchanged from the single-snake days).
- **Self-play, one brain — GENOME-CONDITIONED, evolution at runtime (v3).** `SnakeEnv` controls
  **snake 0**, a live **gradient-ego** with *zero* special powers (it ages, mates, dies, is sensed
  like any other — it is merely where gradients are collected, and is never rendered/watched). `World`
  holds `n_start∈[3,6]` founders at reset, growing by hatching up to `n_max=12`. Every other snake
  (`snakes[1:]`) is stepped **in-env** by `selfplay.OpponentController`, driven by a **snapshot** of
  the ego's own policy. **One shared brain, but each snake carries its own genome** (below) that the
  brain reads and conditions on — so a slow-ambush genome and a fast-dash genome act differently under
  the *same* weights. **Evolution is a RUNTIME process** over the persistent world (egg inheritance +
  differential survival/breeding), NOT trained: PPO only makes the brain competent for the whole gene
  box; there is no neuroevolution and no zoo of models.
- **Genome — 9 genes ∈[0,1], heritable, sensed (v3).** `Snake.genome` (`np.float32(9,)`):
  `[size, metabolism, speed, stamina, senses, lifespan, aggression, kin_care, boldness]`. Genes 0–5
  are **physical trade-offs** interpolated to per-snake stats by `genome.resolve_phenotype(genome,
  cfg)` → a `Phenotype` namedtuple (`max_length, turn_deg, v_snake, v_dash, s_max, stamina_regen,
  ray_range, smell_reach, energy_decay, max_lifespan_base`), cached on `Snake.phenotype`; genes 6–8
  are **behavioral** — no direct physics, they enter only as observation the brain conditions on, and
  pay off through world interaction (cut-off risk/reward, kin selection via relatedness). Trade-offs:
  bigger `size` = more reach/cut-off strength but coarser turns + more hunger; faster `metabolism` =
  grows/acts longer but starves quicker; `senses` trades sight range against smell reach on one
  budget; short `lifespan` = live-fast r-strategist. **Inheritance:** an egg carries `child =
  crossover(A,B)+mutation` (per-gene uniform pick, then `+N(0, mutation_sigma)` clipped to [0,1]),
  computed **at laying** (parents may die before hatch). **Relatedness** = `1 − ‖gₐ−g_b‖/‖ones(9)‖`
  (a genome-similarity proxy, not pedigree), sensed in the social channel for kin recognition.
  Founders (world start / reseed / training) get a **uniformly-sampled** genome — training
  **randomizes it per reset AND per hatch** (domain randomization over the whole box) so the brain
  learns to read the full gene range.
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
- **Observation — 143 floats, egocentric, frame-stacked ×4 ≈ 572** (grew 42→75→87→113→**143**: v3
  added a per-ray target-motion channel, 4 social fields, a 3-float vibration field, and a 12-float
  proprio tail incl. the whole genome).
  Layout (`sensors.observe`, bounds hand-enumerated in `env._make_observation_space`; offsets vision
  `0:99`, social `99:110`, egg `110:114`, smell `114:123`, vibration `123:126`, proprio `126:143`):
  - **Vision, 11 rays × 9 = 99:** `[dist, is_obstacle, is_chicken, is_self, is_other_body, is_egg,
    is_corpse, obstacle_clearance, target_motion]`. **11 rays** = 9 uniform over ±135° + **2 forward**
    at ±16.875° (½ the uniform spacing). Every target (obstacles, chickens, own body, rival
    heads+bodies, eggs, corpses) inflated by `head_radius` in the shared `_cast`/`_scan` step
    (Pitfall 4). `obstacle_clearance` (feat 7) is a **second, obstacle-ONLY** cast so nearer prey can't
    **mask** a rock in the gap (Pitfall 19). **`target_motion` (feat 8, v3)** = the hit's
    **STATE-NOMINAL** speed (pecking hen→0, walking→low, fleeing/**startle-freeze**→high, rival→its
    speed), normalized by the observer-independent `v_dash·gene_speed_hi`. State-nominal (not
    instantaneous) is deliberate: a hen in its startle freeze reads "flee," not "peck" (Pitfall 26).
    **Ray range is per-snake** (`phenotype.ray_range`, `senses` gene) and is also the normalizer.
  - **Social, 11 (v3):** nearest *other* live snake, egocentric — `[has_rival, rel_pos_fwd,
    rel_pos_left, their_heading_fwd, their_heading_left, size_ratio, is_dashing, **relatedness**,
    **rival_energy**, **rival_repro_ready**, **rival_sex**]`. The four new fields drive kin
    recognition (guard/raid, cooperation), weakness-cueing (low-energy rival), and courtship
    (find a repro-ready opposite-sex partner).
  - **Egg, 4:** nearest **eatable** (owner≥0) egg — `[has_egg, rel_pos_fwd, rel_pos_left, is_mine]`.
    `is_mine` (owner ids) is what lets guarding vs. raiding diverge.
  - **Smell, 9 = 3 fields × 3:** `[chicken_…, snake_…, corpse_…]` intensity+gradient — omnidirectional
    sense-around-cover channel. **Reach is now per-snake** (`phenotype.smell_reach`, up to 1.4×); to
    keep `observation_space.contains()` true for a high-`senses` genome, **all three fields are now
    clipped to `chicken_ceiling`** (previously only corpses were — Pitfall 27).
  - **Vibration, 3 (v3):** `[intensity, grad_fwd, grad_left]` — an **un-occluded** omni motion field
    over live rivals + fleeing chickens, weighted by normalized speed: *feel a dasher/fleer through a
    rock or beyond the vision cone*. A stopped rival / calm hen contributes nothing (pairs with the
    speed mechanics). Bound = `n_max + chicken_ceiling` (cfg-derived). **The most cuttable sensor** —
    first to drop if the whole-box bootstrap struggles (Pitfall 9–10 risk, §11 of the spec).
  - **Proprioception, 17 (v3):** `[energy, length, stamina, repro_ready, speed, sex, age_frac,
    stun_frac, genome×9]` (indices 126–142). **Normalizers are PER-SNAKE** (`length/own max_length`,
    `stamina/own s_max`, `speed/own v_dash`) so the signal spans [0,1] for every genome instead of
    saturating (Pitfall 28). `repro_ready` (idx 129) = energy + **size-relative length**
    (`repro_length_frac × own max_length`) + off cooldown — reads the curriculum-swept
    `repro_length_frac`, the ONE observed mating constant (Pitfall 12). `age_frac` = `age/own
    max_lifespan` (feels its life fraction, not absolute age); `stun_frac` = `stun/stun_steps`
    (credit "dash into solid → wasted time"). The **genome tail (9)** is what makes the brain
    genome-conditioned — it senses its own gene box every frame.
- **Action:** `MultiDiscrete([4,3,2])` = **speed** `{0, ⅓, ⅔, 1}×v_snake` × steer
  `{left,straight,right}` × dash `{no,yes}`. Graduated cruise speed is new in v2 (was `[3,2]`); dash
  is still the separate sharp burst `v_dash`, overriding the chosen cruise and stamina-gated. Speed 0
  = rotate in place (steer turns heading, no translation). Lower speed ⇒ tighter turn radius
  (`R = speed/turn_deg`) ⇒ the snake can *choose* to slow down to thread a gap or ambush — Pitfall 20.
- **Reward (sparse — never dense "cooperation" shaping):** `+reward_eat` once per item consumed
  (chicken, corpse, or a foreign egg — never per-bite of a big corpse); `+reward_repro` **only on
  the real hatch of an egg the ego co-owns** (deferred to hatch, not laying — a raided or
  population-cap-dropped egg pays nothing, so guarding matters); an **egg-lost hook (v3)** —
  `+reward_egg_lost` when a REAL egg the ego co-owns is *eaten* (`env._egg_lost_reward(out["eaten_eggs"])`;
  flat, not per-egg), **DEFAULT 0.0** for the discovery retrain (a downside on laying before guarding
  is learned suppresses laying, Pitfall-1's cousin — Pitfall 24); flat `reward_death` on **any** death
  (now `snake`/`starve`/`age` — obstacles aren't lethal); **one** PBRS potential pulling toward the
  nearest chicken (Pitfall 3). **No obstacle-reward machinery** (the Pitfall-16 saga was deleted in
  v2 — obstacles are solid+stun physics, avoidance learned from sight + speed + the stun's opportunity
  cost, Pitfall 19). Personality is **selected, not rewarded**: the reward is genome-blind, so
  aggression/kin-care/boldness pay off only through world interaction under differential survival.
- **Memory:** `VecFrameStack(4)`, NOT an LSTM — unchanged reasoning (self-collision memory, 0
  self-deaths). Opponents get the same 4-frame ring (above), so this applies symmetrically.
- **Deliberate dash = MECHANICAL rationing, never a reward penalty** (Pitfall 1) — unchanged.
- **Solid obstacles + stun, NOT death (v2, Pitfall 19).** Rocks/trees and a snake's **own body** are
  **impassable but non-lethal**: `world._slide` projects the blocked head motion onto the obstacle
  tangent (the head *slides* around it). A **dash** into a solid ⇒ `stun_steps` frozen (steering too
  — "head spinning"); a walk just slides. Only a **rival's** body/head stays lethal (cut-off
  predation, Model A). So the death causes are now `snake` / `starve` / **`age`** (v3 aging, below) —
  never `obstacle`/`self`. This replaces the reward-shaping approach to obstacle avoidance (retired
  Pitfall 16): the stun is a physical *consequence*, not a reward gate — the snake stays free to dash
  anywhere and *learns* (via sight + speed) that ramming a wall wastes time. Because solids no longer
  kill, the world is deliberately **denser** (`n_obstacles_min/max 24/64` — rescaled for v3's bigger
  world/population) — richer to watch and it invites tactics (herd prey into clutter, weave through
  cover) instead of just thinning the population.
- **Graduated speed → emergent stalk / ambush / gap-threading (v2, Pitfall 20).** The speed action
  dimension lets a snake pick any cruise 0→`v_snake` (dash on top). **Prey fear motion:** a chicken's
  flee-alert from each snake scales with that snake's speed, **capped at 1× base** —
  `base_alert·clip(speed/v_snake, 0, 1)`. A **motionless** snake (speed 0) alerts nobody → true
  ambush; a slow stalk lets it close before spooking; a full-cruise or dashing snake is at the full
  `r_flee`/`r_flee_peck` (never *more* — capping at 1× keeps max alert = the value hunting already
  bootstraps under, Pitfalls 9–10). None of this is scripted — stop/stalk/pounce and slow-to-thread
  all *emerge* because they pay off.
- **Dual curriculum, one ramp, one callback.** `env.set_hardness(h)` interpolates BOTH: (a) the
  stamina gate/regen (`dash_min_stamina`, `stamina_regen`: easy always-on dash → the real reserve)
  and (b) the mating gate (`r_mate`, `mate_steps`, and the **size-relative `repro_length_frac`** —
  v3 replaces the old absolute `repro_length_min` sweep, because a small-`size` genome could never
  reach a fixed absolute length and would be sterile by construction) on the same
  `hardness_warmup`→`hardness_full` ramp (0.42→0.85 of total steps). `repro_length_frac` is read
  **identically** by `world._resolve_mating` and `sensors._repro_ready`, so the observed `repro_ready`
  bit and the real eligibility ramp together (the one observed mating constant — Pitfall 12).
  `train.AnnealHardness` pushes `h` into every env each rollout; both discovery ramps move together so
  an abrupt tight-gate switch can't collapse a learned pair-bond or hunter. **Genome ranges are FIXED,
  not annealed** — the brain handles the whole gene box from step 0 (domain randomization).
- **Sex (v3).** `Snake.sex ∈ {0=female, 1=male}`, assigned **randomly ~50/50 at every hatch and
  founder spawn** — NOT heritable (a fresh draw per birth, biologically correct, avoids sex-ratio
  drift). Sensed: own sex (proprio) + nearest rival's sex (social). Mating requires **one male + one
  female**; the **female lays** (maternal lineage).
- **Aging & death by age (v3).** `Snake.age` increments every step; `Snake.max_lifespan =
  lerp(lifespan_min, lifespan_max, gene[lifespan]) × (1 ± lifespan_jitter)`, the jitter resolved **at
  birth** so even same-genome siblings live different spans. `age ≥ max_lifespan` ⇒ death cause
  `"age"` (spawns a corpse like any death; mutually exclusive with `starve` in the phase-3 per-snake
  pass). Sensed as `age_frac = age/own max_lifespan` — the snake feels its *life fraction*, not an
  absolute clock.
- **Reproduction: courtship → female-lays → guard (v3 — world mechanics, not reward hacks).** An
  eligible pair (both alive, **opposite sex**, both `repro_ready` = above `repro_energy_frac·energy_max`
  energy AND above `repro_length_frac × own max_length` (size-relative), both off cooldown) that holds
  within `r_mate` for `mate_steps` consecutive steps **courts**; on completion the **female lays ONE
  egg at the pair midpoint** (within `r_mate/2`, collision-safe — *not* `_free_point`, which would
  place it far and defeat guarding). Both parents pay `repro_cost` and enter `repro_cooldown`. The egg
  **carries the child genome** (`crossover(female,male)+mutation`, computed at lay), `owner =
  [femaleId, maleId]`, and a **maternal `lineage`**. It lies `egg_timer` steps — edible by any
  non-owner (`egg_food`), never by an owner (guarding emerges since reward comes only at hatch) — then
  hatches into a fresh `Snake` inheriting the egg's **genome + lineage + a random sex**,
  `hatch_energy_frac·energy_max` energy, and its own jittered `max_lifespan`, unless the population is
  already at `n_max` (the egg expires — pays neither reward nor egg-lost). **Egg arrays thread
  `genome (N×9)` + `lineage` in LOCKSTEP with `pos/timer/owner`** through every `keep=~eaten`/`~hatched`
  filter — one wrong filter desyncs a genome from its egg (Pitfall 22). Approach/courtship are
  **emergent** (the social channel exposes rel-pos/sex/repro_ready/relatedness; the brain closes in
  because it pays at hatch) — nothing scripts the pair-bond.
- **Corpses / starvation.** A snake whose `energy` hits 0 dies `"starve"`. Any death (`snake`
  cut-off, `starve`, or `age`) spawns a corpse (`corpse_food_per_length · length` food) that anyone
  can scavenge.
- **No ego / spawn-only-from-eggs (v3, Goal 1, Pitfalls 17–18).** Snakes ARRIVE via a **guaranteed
  arrival egg** (`world.spawn_egg`) — reusing the repro egg/hatch machinery, marked by **sentinel owner
  `[-1,-1]`** which makes it **uneatable** (`_snake_eat` gates on `owner[:,0] >= 0`),
  **`n_max`-cap-exempt** in `_hatch_eggs`, and **not a birth** (excluded from `hatched_owners`, pays no
  `reward_repro`); it carries a **freshly-sampled founder genome** + its own new lineage, and its
  hatchling gets a random sex. **`world.no_ego` bifurcates `step`:** the **VIEWER** world (`ego_live=False`)
  has **NO privileged snake — ZERO live snakes at t=0, all founders as eggs**; `step`/`_prune_dead`/`_free_point`
  tolerate an empty live set (camera falls back to overview until an egg hatches). **TRAINING**
  (`ego_live=True`) keeps **one** live gradient-ego in slot 0 (SB3 can't steer an inert egg) with the
  rest as eggs — the only asymmetry, and it has zero gameplay powers. `_free_point`'s ego-area
  clearance is **gated on `not no_ego and self.snakes`** so it still protects the training ego's spawn
  (byte-identical) but skips it in the no-ego viewer (Pitfall 23).
- **Chicken behavior FSM (peck / walk / flee).** Each chicken cycles **PECK** (stands still,
  `chicken_peck_min..max` steps — the prime catch window) ⇄ **WALK** (`v_wander` amble,
  `chicken_walk_min..max` steps) on a timer; any live snake within the **alert radius** triggers
  **FLEE**: a `chicken_startle_steps`-step **freeze** (speed 0, a beat of surprise), then bolt at
  `v_flee` along the **repulsion resultant** of every near snake (never bolts from one snake
  straight into a second; a degenerate cancellation falls back to fleeing the single nearest). The
  alert radius is **state-dependent** *and* **speed-scaled** (v2): the base is `r_flee_peck` (2.5,
  tight) while pecking — a head-down chicken is *distracted*, catchable by stalking up and pouncing —
  vs. the full `r_flee` (12) while walking or already fleeing; each near snake's contribution is then
  scaled by its own speed (`·clip(speed/v_snake, 0, 1)`), so a slow/stopped snake is less/not
  alarming (Pitfall 20). The state-dependent base (Pitfall 10) is what makes hunting discoverable at
  all with 2–4 snakes competing for the same chickens. **Fear persistence (v2, Pitfall 20):** once a
  hen is fleeing it keeps bolting (in its last flee heading) for `chicken_flee_persist` steps *even
  after the snake stops or leaves* — re-armed every step a snake is within reach. A scared hen does
  NOT re-settle the instant the predator freezes, which closes the "spook it, then stop dead so it
  calms, then grab it" exploit the speed-scaled alert would otherwise open. The **peck window was
  halved** (`chicken_peck_min/max 6/18`) so the stationary catch window isn't too generous once snakes
  wield the full speed/stop toolkit.
- **Chickens DROP FROM THE SKY, not pop in (Goal 2, Pitfall 17).** In a production world
  (`world.chicken_sky`, set by `worldgen(arrivals=True)`), a runtime-spawned chicken first spends
  `chicken_arrive_steps` **falling** — a growing ground shadow + the hen descending — living in a
  **separate `world.arriving` array** (NOT `chicken_pos`), so it is **unsensed and uneatable** while
  in the air; `_land_arrivals` (phase 3 of `step`) drops it into the real chicken arrays once it
  lands. **Initial worldgen chickens land instantly** (`arriving=False`) so episode-start food is
  immediate. `maybe_spawn` counts in-flight birds toward the population target so it doesn't
  over-spawn while they fall. Purely a spawn presentation — no reward/obs change. The *render* of the
  fall (a dedicated `chicken_fall` top-down flap sheet, gravity ease-in, shrink, growing shadow,
  damped-smooth glide, random heading) lives entirely in `render._draw_arrivals`. Heading continuity
  is wired: `world.arriving` carries a `head` array (random at spawn in `_add_chicken`, passed into
  `chicken_dir` on landing in `_land_arrivals`), and `render._draw_arrivals` reads
  `world.arriving.get("head")`, so the hen keeps the exact heading it fell with through touchdown.
- **Persistent-viewer reseed floor (egg-based, Goal 1).** The training env resets on ego death
  (short episodes); the viewer/headless-eval world does not (Pitfall 11). `watch._reseed_floor`,
  called every `_step_world` tick, lays **guaranteed arrival egg(s)** (`spawn_egg`, placed via
  `World._free_point`) whenever **live snakes PLUS pending arrival eggs** drop below `cfg.n_start_min`
  (**3** in v3) — so even a reseed *arrives via an egg* rather than popping in. It guarantees *arrivals
  in flight* (live + incubating eggs ≥ 3), NOT *live ≥ 3 instantly*: the **live** count can dip
  briefly (even to 0, showing only rattling eggs) while a reseed egg incubates `egg_timer` steps, then
  recovers. A no-op whenever natural dynamics keep the population healthy.

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
   area right behind their head falsely non-lethal and break cut-off kills. **(v3 extension:** `v_dash`
   is now **per-snake** — `phenotype.v_dash`. The neck-skip AND the prune slack (`target_length +
   v_dash`) must read the SAME per-snake `v_dash`, or a fast-`speed`-gene snake false-collides with
   its own neck; `skip − v_dash` is invariant, so it stays self-consistent iff one value feeds both.)
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
    it — if so, the new value must stay within the range the curriculum already swept.** **(v3
    extension:** the observed mating constant is now the size-relative **`repro_length_frac`** —
    `repro_length_min`/`_easy` are LEGACY, no gate reads them. The genome makes this trap mostly moot:
    the brain observes the *gene* ∈[0,1], never the derived stat, so retuning any interpolation range
    can't shift the observed distribution — see the new **Pitfall 21**.) Of the six
    constants eased in Pitfall 11, five (`r_mate`, `mate_steps`, `repro_cost`, `repro_cooldown`,
    `stamina_regen`) are genuinely **not** in the observation — the policy only feels their effect
    indirectly (a slightly less starved stamina reserve, a looser mating gate) and trivially
    generalizes. But `repro_length_min` **is** observed — `sensors._repro_ready` reads it directly
    to compute the `repro_ready` proprioception bit (obs index 111 in the v2 layout). That's safe here **only**
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
    **(v3 extension:** `step` is now **`no_ego`-bifurcated** — the viewer branch drives EVERY snake via
    `opponent_fn` and reports no ego death, tolerating an empty live set; the training branch keeps the
    live slot-0 gradient-ego. The two-phase order-independence is identical in both branches. The
    training (`no_ego=False`) branch is byte-identical to v2.)
14. **Corpses were food (`try_eat`) but UNSENSED** (no ray category, no smell field) — scavenging
    literally could not emerge, because nothing in the observation pointed a snake at a corpse.
    Fixed by adding an `is_corpse` ray one-hot + a corpse smell field (`OBS_DIM` 75→87), caught
    late (right before the final 8M retrain) by a spec-gap review, not by the original design.
15. *(Minor, worth knowing)* **macOS sleep pauses training** — the machine sleeping mid-run makes
    SB3's reported fps read absurdly low afterward (it divides steps by a wall-clock time inflated
    by the sleep, not by actual compute time). Use `caffeinate` for long unattended runs and judge
    real throughput from step-timestamps, not the printed fps alone.
16. **[RETIRED in v2 — kept as a hard-won lesson] Obstacle deaths were once a REWARD problem; the
    real fix turned out to be PHYSICS.** History: the shipped model plowed into rocks ~37–64% of
    deaths, ~**71% of them mid-dash**. Never a sensing bug (`self 0`, swept collision, ray kind 0
    present) — obstacles were simply *lethal on contact* while nothing rewarded avoiding them, and
    the 71%-are-dashes diagnostic revealed the true cause: at dash speed the turn radius (~7.2) makes
    a rock physically **unavoidable once committed** — a *physics-commitment* problem no reward could
    fully fix. Each failed reward attempt taught something: (a) an obstacle-avoidance **PBRS well** is
    policy-invariant → teaches the avoidance *skill* faster but in the limit can't stop a
    reward-optimal crash; (b) chasing a chicken flanked by rocks is a genuine **+EV gamble**, so a
    heavier `reward_death_obstacle` was needed to flip it negative — yet a fully-trained model with
    *both* still didn't cut the aggregate hazard, because the crash is geometric, not incentive.
    **v2's answer: stop making obstacles lethal at all** — make them **solid (slide) + dash-stun** and
    give the snake **un-masked forward sight + speed control** so it *decides* to slow and thread.
    `reward_death_obstacle` and the PBRS well are both **deleted**. The lesson that outlives the code:
    *when a "reward gap" resists reward fixes, check whether it's really a physics/capability limit —
    and prefer a world-physics consequence (stun) over a reward gate (Pitfall 1).* See Pitfalls 19–20.
17. **Staged arrivals must live OUTSIDE `chicken_pos` — because you can't edit `sensors.py` to
    exclude them.** For chickens to "arrive from the sky" (Goal 2) they must be **unsensed and
    uneatable while falling**, but `sensors._all_targets`/`smell`/`env._phi` all read
    `world.chicken_pos` directly and (per the constraint) sensors is off-limits. The fix is *where the
    data lives*, not a sensor filter: in-flight birds sit in a **separate `world.arriving` array** and
    only get appended to `chicken_pos` when they land (`_land_arrivals`, phase 3 of `step`). So
    nothing that reads `chicken_pos` ever sees them — no sensors/eat/PBRS change needed. Same trick
    for arrival **snakes** (Goal 1): a guaranteed egg is just an egg with **sentinel owner `[-1,-1]`**,
    which the *existing* egg code already renders/hatches; only three tiny `world.py` gates make it
    special (uneatable via `owner>=0`, cap-exempt in `_hatch_eggs`, excluded from `hatched_owners`).
    Reuse the machinery, mark the exception — don't fork a parallel system.
18. **`worldgen(arrivals=True)` is TRAINING-active, and the ego must stay a live snake.** `env.reset`
    passes `arrivals=True`, so during training opponents START as incubating eggs (hatch in
    ~`egg_timer//2..egg_timer` steps) and runtime chickens drop from the sky — a real dynamics change
    the policy experiences (retrain-relevant, and fine: a from-scratch retrain is planned). Two
    guardrails: (a) **snake 0 (the ego) is ALWAYS spawned live** — `SnakeEnv` drives it from step 0 and
    cannot steer an inert egg, so only `snakes[1:]` become eggs. (b) The `arrivals` flag **defaults
    `False`** so unit-test fixtures that build a multi-snake `generate_world(n_snakes=K)` still get K
    *live* snakes; only the production paths (`env.reset`, `watch`) opt in. Consequence: any test that
    reads `env.world.snakes[1]` right after `reset()` must instead account for eggs (a few
    `test_selfplay` tests were updated to count pending arrival eggs / add an explicit opponent).
19. **Solid-slide + dash-stun beats reward-shaping for obstacle avoidance — and un-masking is the
    perception half.** Rocks/trees + own body are impassable-not-lethal (`world._slide` slides the
    head along the tangent; a dash into one ⇒ `stun_steps` frozen — steering too). Two subtleties:
    (a) the slide's own-body solids **must** use the neck-skip body points (Pitfall 5) or a
    straight-moving snake false-collides with its own neck; the phase-2 rival-lethality check then
    sweeps the **post-slide** head (two-phase order-independence, Pitfall 13, intact). (b) A denser
    forward ray fan alone would NOT fix the "chicken between two rocks" crash, because each ray reports
    only its *nearest* hit — a chicken in the gap **masks** the rock behind it. The fix is the per-ray
    **obstacle-only** `obstacle_clearance` channel (a second scan ignoring prey), so the policy always
    sees the rock in its path even while chasing. Sight + speed + the stun's opportunity cost teach
    avoidance with **zero reward shaping**.
20. **Graduated speed is the keystone: ambush, stalking, AND gap-threading fall out of one action
    dim — but cap prey-alert scaling at 1× or you re-open the Pitfall-9/10 bootstrap trap.** A
    0→`v_snake` cruise choice (dash separate) means low speed ⇒ tight turn radius ⇒ the snake can
    *choose* to slow and thread a gap (emergent anti-crash), and speed-scaled prey-alert
    (`base·clip(speed/v_snake,0,1)`) makes a motionless snake invisible to prey ⇒ emergent
    freeze-and-strike. **The 1× cap is load-bearing:** an early design let a dash scale alert to 2×
    (walking-chicken flight at 24) — exactly the reactive-prey bootstrap collapse of Pitfalls 9–10 on
    a fresh retrain. Capping at 1× keeps max alert = today's `r_flee`, so hunting still bootstraps and
    catch-invariant #3 stays valid. The snake also senses its **own speed** (proprio idx 130 in the
    v3 layout, normalized by its OWN `v_dash`) to reason about its turn radius.
    **v2.1 — speed also drives STAMINA regen:** `regen = stamina_regen·(1 − speed_fraction)` — full at
    a dead stop (speed_idx 0), zero at full cruise (speed_idx 3); dash still drains. So standing still
    *both* ambushes *and* recharges the dash reserve fastest — one choice pays off twice, turning ambush
    into a real resource strategy (pause to bank stamina, then dash-hunt). A dynamics change (not
    observed — Pitfall 12), re-tuned by a **resume**: perception/motor/hunting/breeding skills transfer
    unchanged; only *when-to-dash-vs-pause* is re-learned. Expect a transient dip → recovery (the
    peck-distraction path needs no dash, so it can't fully collapse). Paired with a faster hunger
    (`energy_decay 0.20`) in the same resume.
    **The speed-scaled alert opens a degenerate exploit — close it with FEAR PERSISTENCE.** If alert
    tracks the snake's *instantaneous* speed, a snake can spook a hen, then **stop dead** (speed 0 ⇒
    alert 0) so the hen instantly "calms" and freezes, then grab it. The policy discovers this from
    the very first steps of training. Fix: a scared hen keeps bolting for `chicken_flee_persist` (15)
    steps *regardless of the snake's current speed* — the panic timer is re-armed while a snake is in
    reach and only runs down once it isn't (`world.chicken_flee`, a 4th FSM array kept lockstep with
    the others, Pitfall 17). A real hen doesn't re-settle the instant the predator freezes. Ambush
    (a stopped snake never alarms a *calm* hen) is untouched — persistence only governs an
    *already-scared* one. Pair it with a shorter peck window (`chicken_peck_min/max 6/18`, halved) so
    the stationary catch window isn't a free lunch once snakes wield stop/stalk/dash together.

**New from the genetic-ecosystem v3 build:**

21. **Observe the GENE, not the derived stat (Pitfall-12-clean by construction).** The brain senses
    every genome float ∈[0,1] (proprio tail), NOT the interpolated physics value. Domain randomization
    sweeps the whole [0,1] box during training, so **retuning any `gene_*` interpolation range
    post-training never shifts the observed distribution** — it only changes physics the brain already
    generalizes over. This retires the Pitfall-12 easing trap for genes. **Corollary — the *derived*
    stats used as normalization denominators are the thing to watch**, not the genes (see Pitfall 28).
22. **Egg arrays thread `genome`+`lineage` in LOCKSTEP with `pos`/`timer`/`owner`.** Five parallel
    arrays; a hatch/eat filter (`keep = ~eaten` / `~hatch`) that masks four but forgets one **desyncs a
    genome from its egg** — a hatchling would inherit some *other* egg's genes (Pitfall-17-style
    "keep the parallel arrays aligned"). `_lay_egg` centralizes the append; every keep-filter masks
    **all five**. Test any new egg field the same way.
23. **`_free_point`'s ego-area clearance must be gated on `not no_ego and self.snakes`, not stripped.**
    `_free_point` rejects candidates within `r_flee` of the ego head to keep its start area open. The
    no-ego viewer has no ego head to protect, so an early refactor **stripped the check outright** — but
    `maybe_spawn` also calls `_free_point`, so that silently changed **training-time chicken placement
    and the per-episode RNG stream**. Gate it: training (`no_ego=False`, live ego) keeps the clearance
    byte-identical; the viewer skips it. RED-on-revert covered by a worldgen test.
24. **`reward_egg_lost` DEFAULTS to 0.0 for the discovery retrain (Pitfall-1's cousin).** Reproduction
    is already the hardest thing to bootstrap (Pitfalls 9–11 = multiple aborted runs); attaching a
    downside to *laying* before guarding is learned suppresses laying the way `dash_penalty` suppressed
    dashing. So the egg-lost penalty ships **off**, enabled only as a **post-competence sharpening
    knob** (e.g. `−4.0`) once guarding exists — and it's re-tunable at runtime since it isn't observed.
25. **`assert_invariants_over_genome` keeps only the true correctness gates HARD across the gene box;
    invariants 2 & 4 are now SOFT.** Requiring EVERY joint genome to satisfy EVERY invariant boxes the
    genes so tightly the snakes end up cosmetically identical — defeating the headline goal. **HARD**
    (must hold for all genomes): catch **precision** at max-`size` coarsest turn (caps `gene_size_turn_hi
    ≤ 1.15`) and **nearest-image raycast** at max-`senses` longest `ray_range`. **SOFT (log, never
    fail):** invariant 2 (a full dash closes a fleeing chicken) — a slow/low-stamina genome that can't
    run a hen down is a valid **ambush/scavenge niche** now that peck-hunting needs no dash; invariant 4
    (self-collision reachable) — own body is non-lethal solid-slide (Pitfall 19), so an un-curl-able
    genome is fine. Downgrading these is deliberate, not an oversight.
26. **Per-ray target-motion is STATE-NOMINAL, not instantaneous.** A hen in its `chicken_startle_steps`
    freeze has instantaneous speed 0 but is about-to-bolt — it must read "flee" (high), not collide with
    a pecking hen's 0. So the motion channel uses each target's *state-nominal* speed (peck 0 / walk
    `v_wander` / flee `v_flee`; rival → its actual speed), normalized by the **observer-independent**
    `v_dash·gene_speed_hi` (so a max-speed-gene dasher reads ≤1, never >1). Same normalizer feeds the
    vibration field. Motion rides a **parallel array threaded through `_other_hazard`/`_cast`** because
    the flatten there otherwise loses which rival a body point belongs to (`_death_cause` ignores it).
27. **Per-snake `smell_reach` means ALL three smell fields need the clip, not just corpses.** A
    high-`senses` genome scales smell intensity by up to 1.4×, which can push chicken/snake intensity
    past `chicken_ceiling` → `observation_space.contains()` fails → `check_env` fails. v2 only clipped
    the (uncapped) corpse field; v3 clips **all three** to `chicken_ceiling`, and the env smell bounds
    were widened `n_max → chicken_ceiling` to match the clip exactly.
28. **Normalizers must be PER-SNAKE or a strong genome saturates and loses the granularity it needs.**
    `stamina/s_max`, `length/max_length`, `speed/v_dash` were global-`CFG`-normalized; a high-`stamina`
    or high-`size` genome would then read `1.0` well below its *own* reserve/length, blinding the brain
    to how much dash budget or growth room it actually has. Use the snake's **own phenotype** as the
    denominator so every genome's signal spans [0,1]. (This is the derived-stat corollary of Pitfall 21
    — the genes are safe, their *normalizers* are the trap.)

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

**Genome invariant gate (v3): `assert_invariants_over_genome(cfg)`** runs at import alongside
`assert_invariants` and re-checks the correctness gates across the WHOLE gene box (worst-corner
genomes), because per-snake physiology means one CFG check no longer covers every snake:
- **HARD (fail fast):** aim **precision** at the max-`size` coarsest turn (this is what caps
  `gene_size_turn_hi ≤ 1.15`), and nearest-image **raycast** at the max-`senses` longest `ray_range`.
- **SOFT (log, never fail) — two invariants are downgraded per-genome** (Pitfall 25): the
  stamina/dash budget (**#3** above, "a full dash closes a fleeing chicken") and self-collision
  reachability (**#6** above) are single-strategy relics — a slow ambush/scavenge genome and a genome
  that can't curl onto its (now non-lethal) own body are both valid. **Aim precision (#4) stays
  HARD** — it is never softened. Logged so we *know* which genomes are ambush-only.

**World / population (v3):** `world_size 180–260` (bigger — room for 12 snakes, exceeds the screen so
a Phase-B camera can pan/zoom), `n_start_min/max 3/6`, **`n_max 12`**. **Food scales with population**
(rates, not absolutes): `chickens_per_snake_max/min 2.0/1.0`, hard cap **`chicken_ceiling 24`**
(`= 2·n_max` exactly). `n_obstacles_min/max 24/64` (density rescaled for the bigger map).

**Motion / speed — now PER-SNAKE via `phenotype` (v3):** base `v_snake 1.0`, `turn_deg 16`,
`v_dash 2.0` are the interpolation *midpoints*; each snake's actual `turn_deg`/`v_snake`/`v_dash`/
`s_max`/`stamina_regen`/`ray_range`/`smell_reach`/`energy_decay`/`max_lifespan` come from
`resolve_phenotype(genome, cfg)`. Graduated **`speed_levels (0, ⅓, ⅔, 1)`** × that snake's `v_snake`
(the speed action dim); dash = separate burst at its `v_dash`; **`stun_steps 10`** (dash into a solid
⇒ frozen — Pitfall 19).

**Sensing (v3):** `n_rays 9` uniform + **`n_fwd_rays 2`** forward (RAY_COUNT 11), `fov_deg 270`,
per-snake `ray_range` (base `ray_range 20`, gene sweeps 14–26), `frame_stack 4` → **`OBS_DIM 143`**
(Pitfalls 21–28).

**Stamina/dash:** base `s_max 30`, `stamina_drain 1.0`, hard `dash_min_stamina 1.0`, easy
`dash_min_stamina_easy 0.05`, base `v_dash 2.0`, `r_flee 12`. **`stamina_regen 0.7`** = the
**speed-0 (dead-stop)** regen rate, scaled `×(1 − speed_levels[idx])` to zero at full cruise
(Pitfall 20); `stamina_regen_easy 0.9`. `s_max` and `stamina_regen` are further scaled per-snake by
the `stamina` gene (×0.7–1.4).

**Energy / hunger:** `energy_max 100`, base **`energy_decay 0.20`** (life-without-food ≈ 500 steps),
per-snake scaled by the `metabolism` gene (×0.65–1.5) AND a `size` hunger surcharge (bigger body
costs up to +0.4×). `energy_refill 40` (per chicken/corpse/egg). The *rate* is not observed (Pitfall
12) — obs carries only `energy/energy_max`.

**Chicken FSM:** `chicken_peck_min/max 6/18`, `chicken_walk_min/max 18/45`, `chicken_startle_steps 4`,
**`chicken_flee_persist 15`** (fear-persistence — kills the spook-then-stop exploit, Pitfall 20),
`r_flee 12` (walk/flee alert), `r_flee_peck 2.5` (peck alert — stalk-and-pounce, Pitfall 10),
`chicken_arrive_steps 12` (sky-drop fall).

**Reproduction / eggs / corpses / aging (v3 hard endpoints — the curriculum sweeps to these, trains
from scratch ON them):** `repro_energy_frac 0.7`, **`repro_length_frac 0.55`** (size-relative length
gate = fraction of own max_length — THE observed mating constant, replaces the legacy absolute
`repro_length_min 8.0` which no gate reads), `r_mate 7.0`, `mate_steps 2`, `repro_cost 18.0`,
`repro_cooldown 80`, `egg_timer 45`, `hatch_energy_frac 0.5`, `egg_food 25.0`,
`corpse_food_per_length 4.0`. Mating curriculum easy endpoints (swept as `hardness: 0→1`):
`r_mate_easy 12.0`, `mate_steps_easy 1`, **`repro_length_frac_easy 0.4`**, `stamina_regen_easy 0.9`.

**Genome / evolution (v3, new):** gene→stat ranges — `gene_size_len 0.65–1.35`,
`gene_size_turn 0.85–1.15` (precision-gate cap), `gene_size_hunger_hi 0.4`, `gene_metab 0.65–1.5`,
`gene_speed 0.85–1.2`, `gene_stamina 0.7–1.4`, `gene_stamina_regen 0.7–1.4`, `gene_rayrange 14–26`,
`gene_smell 1.4→0.7` (inverse of sight — one senses budget), `gene_lifespan 900–3200` steps.
`mutation_sigma 0.05` (evolutionary temperature), `lifespan_jitter 0.15` (±fraction on max_lifespan
at birth). **`reward_egg_lost 0.0`** (DEFAULT OFF — Pitfall 24).

**Reward (v3 — sparse):** `reward_eat 10.0`, `reward_repro 12.0`, `reward_death −10.0` (flat, fires on
`snake`/`starve`/`age`), `step_penalty 0.01`, `dash_penalty 0.0` (rationed by stamina — Pitfall 1),
**`reward_egg_lost 0.0`** (egg-lost hook, off for bootstrap — Pitfall 24). A single chicken-PBRS
potential (`env._phi`/`_shaping`, Pitfall 3); **no** obstacle-reward machinery (physics + sight +
speed, Pitfalls 16, 19–20).

Curriculum ramp: `hardness_warmup 0.42`, `hardness_full 0.85` (of total training steps) — governs
BOTH the stamina gate/regen and the mating gate together (`env.set_hardness`).

---

## Making changes & retraining

**Decide first: does this need a retrain?** Anything that changes what the policy senses or is
rewarded for, or the physics it acts in, DOES.

- **Needs retrain:** `sensors.py` (obs values or `OBS_DIM`), **`genome.py`** (gene meaning/phenotype —
  it's both observed AND drives physics), `env.py` reward/action/obs-bounds, `world.py`
  physics/stamina/reproduction/aging/sex/chicken-FSM/collision rules, `worldgen.py`,
  stamina/reward/`gamma`/curriculum constants — **and any constant `sensors.observe` reads directly**
  (Pitfall 12, e.g. `repro_length_frac`). **Gene interpolation ranges are the exception (Pitfall 21):**
  retuning a `gene_*` range changes *physics* but NOT the observed distribution (the brain sees the
  gene ∈[0,1], not the derived stat), so it does *not* strictly need a retrain — but it *does* change
  the world the brain was trained to be competent in, so re-judge behavior after.
- **No retrain:** `render.py`, `watch.py` viewer loop/HUD/effects/reseed-floor/camera, `__main__` CLI,
  fullscreen, and any config constant genuinely never read inside `sensors.observe` (check first).

**Retrain recipe (v3 — FROM SCRATCH, wall-clock PENDING — budget MORE than v2's ~40–75 min: 12 snakes
+ 143-float obs + bigger world = more entities/step):**
```bash
./snake train --steps 8000000 --envs 16 --reset
```
- **v3 requires a FRESH from-scratch retrain** — obs (`OBS_DIM 113→143`), the genome-conditioned
  action context, physics (per-snake physiology, sex-gated courtship, aging), and n_max all changed,
  so no v2 checkpoint can be resumed. **The whole-box bootstrap is the sharpest risk (spec §11/§14):**
  a fresh policy must hunt across the *entire randomized gene box* AND discover courtship. Watch
  `snake/eaten_per_window` climb in the first ~50–100k steps across the randomized population; if flat
  near 0, the weakest genomes may not bootstrap — **narrow the harshest gene extremes** or **drop the
  vibration channel** (the most cuttable sensor, Pitfall 27's field) and retry. Abort early if
  collapsed (Pitfalls 9–10).
- Curriculum: keep the dual stamina + size-relative-mating hardness ramp (`hardness_warmup 0.42` →
  `hardness_full 0.85`). **Genome ranges are FIXED, not annealed** — domain randomization from step 0.
- **Expected end reward / judging bands / trajectory are PENDING the v3 retrain** — do NOT assume v2's
  `ep_rew_mean ≈ +127` or its catch/dash/population numbers carry over (bigger world, per-snake
  physiology, aging deaths all shift them). Judge by the *end* and by **behavioral diversity** in the
  persistent eval (lineages visibly diverging), not the mid-ramp dip.
- If it collapses: check chicken alert radii first (Pitfalls 9–10), then narrow gene ranges / drop
  vibration, then lengthen `hardness_warmup`, keep lr constant, or change `--seed` (PPO is high-variance).

**PPO hyperparameters live in `train.py`** (`PPO(...)` for a fresh run): constant `lr 3e-4`,
`n_steps 1024`, `batch_size 256`, `n_epochs 10`, `gamma` from CFG (`0.99`), `gae_lambda 0.95`,
`ent_coef 0.01`, `target_kl 0.03`, `net_arch pi/vf [128,128]`. See Pitfalls 2, 6 before touching
these; the same net architecture is reused verbatim for `OpponentController`'s policy skeleton.

---

## Judging a model (behavioral, not just reward)

`./snake watch --headless --episodes 20` runs a **persistent** no-ego ecosystem (never resets on any
single death — same world the real viewer uses) and prints + returns a metrics dict. **The specific
catch/dash/population BANDS below are v2 numbers, PENDING re-measurement after the v3 retrain** — the
bigger world, per-snake physiology and aging deaths will shift them; use them as *shape*, not targets.
A healthy v3 model:
- **per-snake catch rate** (v2 was ≈ 10–14 items / 1000 snake-steps) — computed per-snake-step, not
  population-summed, or it'd scale with headcount;
- **dash usage** (v2 was ≈ 25–36% of live snake-steps) — deliberate bursts, not constant; expect a
  WIDER spread in v3 (a low-`stamina`/slow genome barely dashes; a high one dashes often);
- **population** sustains around **3–6** (the egg-based reseed floor guarantees *live + incubating
  arrival eggs* ≥ `n_start_min=3`, so the **live** count can dip briefly while a reseed egg hatches —
  a transient dip is normal); organic births on top of that = a living cycle, not floor-padding;
- **deaths (v3 — three causes)**: real `snake` (cut-off predation), some `starve`, and now `age`
  (old snakes die). **No `obstacle`/`self`** — obstacles + own body are solid-not-lethal, so those
  leaking back means the old lethal-collision code returned;
- **ambush is visible**: some snakes sit at speed 0 near prey then strike (graduated speed working);
- **NEW v3 — behavioral/lineage DIVERSITY is the headline signal**: snakes should *visibly differ*
  (sizes, speeds, dash frequency, ambush vs. run-down hunting) and lineages should diverge over a long
  run — predator lines, ambush-scavengers, fast sprinters. A population that all behaves identically
  means the genome isn't being read (check the proprio genome tail + domain randomization);
- **full cycle at once**: hunt → grow → **court** → lay → guard → hatch → age/die → scavenge.
Also glance at the SB3 table during training: `snake/eaten_per_window`, `snake/repro_per_window`,
`snake/hatched_per_window` climbing together (not stuck near 0) is the earliest healthy signal.
`snake/hardness` should reach 1.0.

---

## Testing

`SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest -q` (**169 tests**, ~11s). One
runnable `assert` per non-trivial mechanic. **v2 mechanics:** torus nearest-image, raycast (11 rays
incl. rival/egg/corpse ray kinds + forward-ray angles + the un-mask obstacle-clearance channel),
solid-slide collision (walk-slide-no-death, dash-into-solid⇒stun, own-body non-lethal, rival-body
lethal, straight-motion-no-neck-deflection), graduated speed + prey-senses-motion, two-phase-step
order-independence + head-to-head, inter-snake cut-off, corpse spawn/eat-once,
mate→egg→hatch + parent-can't-eat-own-egg + egg raid, sky-drop chicken, starvation, population cap,
population-scaled food, chicken FSM + startle-freeze + peck-distraction, PBRS telescoping, stamina
gate, opponent-obs parity, `check_env`, interpolation, color/ring-HUD determinism. **v3 mechanics:**
genome ops (`crossover` per-gene from a parent, `mutate` stays in [0,1], `relatedness` metric) +
`resolve_phenotype` (distinct genomes → distinct stats); `assert_invariants_over_genome` passes at the
gene-box extremes; **sex-gated mating** (same-sex pair never lays, female is the layer); **aging**
(death at `age ≥ max_lifespan`, `age_frac`∈[0,1], sibling lifespan jitter differs); **courtship→lay**
with crossover-genome egg + both owners + egg-lost penalty on eat; **egg genome+lineage lockstep**;
sensors (relatedness/rival-state in social, vibration responds to a moving-not-stopped rival, per-ray
motion peck≈0/flee≈high, stun sensor); **no-ego/all-eggs** (viewer starts 0 live + ≥1 egg, snake
appears after `egg_timer`; training keeps one live ego); `_free_point` ego-clearance gated on live-ego;
obs bounds / `check_env` for **extreme genomes**; **genome domain-randomization across founders+eggs on
reset**; **egg-lost reward only for ego-co-owned eggs**; **obstacle-count clamp reaches `n_obstacles_max`**.
Spread across `tests/test_torus.py`, `test_collision.py`, `test_snake_motion.py`, `test_multisnake.py`,
`test_reproduction.py`, `test_chickens.py`, `test_worldgen.py`, `test_sensors.py`, `test_selfplay.py`,
`test_env.py`, `test_config.py`, **`test_genome.py`**, **`test_phenotype.py`**, `test_interp.py`,
`test_render_smoke.py`, `test_watch_smoke.py`, `test_train_smoke.py`. Add a test with any new mechanic;
update the vision/geometry tests if you change `head_radius`, `ray_range`, the neck-skip, or the rival
hazard set. No commit attribution lines (per repo rule).
