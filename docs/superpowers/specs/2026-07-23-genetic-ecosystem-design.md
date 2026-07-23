# Genetic Ecosystem (v3) — Design

**Branch:** `genetic-ecosystem` · **Date:** 2026-07-23 · **Status:** approved, pre-plan

A from-scratch reworking of the Snake-RL world from *one shared brain, cosmetically identical
snakes* into a **living genetic ecosystem**: every snake carries a heritable **genome** of
physiological trade-offs and behavioral tendencies, has a **sex**, an **age** and a genetically
determined **lifespan**, and passes its genes to offspring through a proper **courtship → laying →
guarding** reproduction cycle. One PPO brain, **conditioned on the genome**, drives every snake;
**evolution is a runtime process** over the never-resetting world (genome inheritance + natural
selection by who-survives-and-breeds), not something PPO optimizes. The goal is a world where you
can *watch* lineages diverge — predator lines, cooperative kin-breeders, sprinters, ambushers — and
*see* they are different.

This supersedes the v2 "thinking-snake" design. It requires a **fresh from-scratch retrain** (obs,
physics, reward, action-context all change).

---

## 1. Design philosophy (unchanged core, extended)

- **Emergence over scripting (Philosophy B).** Genes are **physical/physiological trade-offs**, not
  reward knobs. The reward stays **sparse** (eat / reproduce-at-hatch / flat-death / small
  egg-lost / chicken-PBRS). Different genomes win in different niches under the *same* reward, so
  personality is *selected*, never imposed. Behavioral genes (aggression / kin_care / boldness)
  matter only through world interaction (cut-off gives food but risks death; helping kin spreads
  shared genes via relatives) — the brain reads them and learns genome-appropriate behavior.
- **One brain, genome-conditioned.** The genome is part of every snake's observation. The brain
  learns "given *this* body and disposition, act *this* way." Because `sensors.observe(world,
  snake)` is the single chokepoint used by training (`env`), self-play (`selfplay`), and the viewer
  (`watch`), appending the genome there makes it flow everywhere for free.
- **Evolution is runtime, not trained.** PPO produces a brain competent *for any genome*. The
  genetic algorithm lives in the persistent world: eggs inherit `crossover(parents)+mutation`,
  selection = differential survival/reproduction. No neuroevolution over network weights, no zoo of
  models.
- **Respect every existing Pitfall (1–20).** Especially: no per-step dash/behavior reward penalties
  (Pitfall 1); constant LR + `target_kl` for a doubly-non-stationary task (Pitfall 2); peck-
  distraction curriculum so hunting bootstraps under reactive prey (Pitfalls 9–10); solid-slide +
  stun physics for obstacles (Pitfalls 19–20); the 1× prey-alert cap (Pitfall 20).

---

## 2. Genome

A fixed-length vector of **9 genes**, each a float in **[0, 1]**, stored on `Snake.genome`
(`np.ndarray(9,)`). Genes map to per-snake stats by interpolating a config-defined `[low, high]`
range. **The brain senses its own genome** (all 9 floats in proprioception).

### 2.1 Genes and their effects

| # | Gene | Physical effect (low → high) | Trade-off |
|---|------|------------------------------|-----------|
| 0 | `size` | small max length → large max length | reach/cut-off strength ↑, but hunger ↑ and turn radius ↑ (coarser `turn_deg`) |
| 1 | `metabolism` | thrifty `energy_decay` → fast | fast = grows faster/acts longer between meals stress, but starves quicker |
| 2 | `speed` | economical top cruise/dash → fast | fast closes prey/rivals, but see stamina/hunger cost |
| 3 | `stamina` | small reserve → large `s_max` + faster regen | more/longer dashes vs. cheaper upkeep |
| 4 | `senses` | eyes↔nose allocation: long `ray_range`/short smell → short sight/long smell | open-ground hunter vs. clutter/scent tracker (one budget, traded) |
| 5 | `lifespan` | short `max_lifespan` → long | long-lived slow breeder vs. fast "live-fast" r-strategist |
| 6 | `aggression` | disposition toward cut-off predation (behavioral, brain-read) | kills yield corpse food but risk death → frequency-dependent selection |
| 7 | `kin_care` | tolerance/cooperation toward kin (behavioral) | helping relatives spreads shared genes (kin selection) — needs relatedness sensor |
| 8 | `boldness` | risk tolerance near obstacles/rivals/prey (behavioral) | more dashing/committing vs. caution (stun cost, cut-off exposure) |

Concrete initial interpolation ranges (in `Config`; the HARD gates in §2.4 cap them, the rest is
tuned during retrain):

- `size`: `length_cap × [0.65, 1.35]`; `turn_deg × [0.85, 1.15]` (bigger = coarser turns; **1.15 cap
  set by the precision gate, §2.4**); `energy_decay` gets a size surcharge (bigger body costs more).
- `metabolism`: `energy_decay × [0.65, 1.5]`.
- `speed`: `v_snake` & `v_dash × [0.85, 1.2]`.
- `stamina`: `s_max × [0.7, 1.4]`, `stamina_regen × [0.7, 1.4]`.
- `senses`: `ray_range = lerp(14, 26)`; smell "reach" scaled inversely (a smell range/strength
  multiplier `lerp(1.4, 0.7)`), so the two trade against one shared budget.
- `lifespan`: `max_lifespan = lerp(900, 3200)` steps (see §5).

Behavioral genes (6–8) have **no direct physics**; they enter the world only as observation the
brain conditions on. They evolve because the brain's genome-conditioned policy makes them pay off
differently in different population states.

> **Pitfall-12-clean by construction.** The brain observes the **gene** (always ∈[0,1]), not the
> derived stat. Domain randomization sweeps the whole [0,1] box during training, so **retuning any
> interpolation range post-training never shifts the observed distribution** — it only changes
> physics. This makes the genome immune to the Pitfall-12 trap that `repro_length_min` had to be
> argued into. (Corollary: the *derived stats* used as normalization denominators are the thing to
> watch — see §7.)

### 2.2 Inheritance

Reproduction produces an egg that **carries the child genome, computed at laying time** (parents may
die before hatch, so we never look them up later):

```
child = crossover(parentA.genome, parentB.genome) + mutation
```

- **Crossover:** per-gene uniform pick from either parent (`p=0.5`) — simple, standard, preserves
  gene identity. (Blend/BLX considered; uniform is the lazy correct default and keeps discrete gene
  meaning.)
- **Mutation:** add `N(0, mutation_sigma)` per gene, then **clip to [0, 1]**. `mutation_sigma`
  (config, e.g. `0.05`) is the evolutionary "temperature" — small enough to preserve competence,
  large enough to explore.
- **Founder genomes** (world start / viewer reseed / training randomization): sampled uniformly
  in [0, 1] per gene (see §9, §11).

### 2.3 Relatedness (for kin recognition)

`relatedness(a, b) = 1 − ‖a.genome − b.genome‖ / ‖ones(9)‖` clipped to [0, 1] — a **genome-similarity
proxy** (phenotype/greenbeard matching), *not* pedigree. Lazy and biologically defensible; true
ancestry-coefficient tracking is a future upgrade if the proxy proves too noisy. Sensed in the
social channel (§7).

### 2.4 Invariants across the gene box (revised — hard vs. soft)

The base `Config` still satisfies **all** of `assert_invariants` unchanged (the interpolation
endpoints are sane). But requiring **every** joint genome to satisfy **every** invariant boxes the
genes so tightly that snakes end up cosmetically similar — defeating the headline goal. Two of the
invariants are **single-strategy-era relics** and must be **downgraded to soft warnings** for the
per-genome gate:

- **Invariant 4 (self-collision reachable / turn-circumference)** existed only because own-body was
  *lethal* in the single-snake era. Own body is now **non-lethal solid-slide** (Pitfall 19), so a
  genome that physically can't curl onto itself is fine — the self-solid simply never engages. **→
  soft.**
- **Invariant 2 (a full dash guarantees closing a fleeing chicken)** guaranteed *one* hunting
  strategy for *one* snake type. With peck-distraction (Pitfall 10) hunting a *pecking* hen needs no
  dash at all, so a slow/low-stamina genome that can't run down a sprinting hen is a **valid niche**
  (ambush pecking hens, scavenge corpses/eggs), not a broken snake. **→ soft.**

Add `assert_invariants_over_genome(cfg)` that keeps only the **true correctness gates HARD** across
the box, and logs the soft ones:

- **HARD — Catch precision** (`turn_deg/2 < atan(eat_radius/r_flee)`): the brain must be *able* to
  aim. Stresses at **max size** (coarsest turn). Limit: `turn_deg < 18.92°` at base `r_flee=12`,
  `eat_radius=2` → **caps `size`'s `turn_deg` multiplier at ≤ 1.15** (16×1.15 = 18.4° < 18.92°).
- **HARD — Nearest-image raycast** (`ray_range + obstacle_radius_max + head_radius < world_size_min/2`):
  correctness of the torus raycast. Stresses at **max `senses`** (longest `ray_range`). With the
  enlarged world (`world_size_min ≥ 180`) and `ray_range ≤ 26`, comfortably satisfied.
- **SOFT (log, don't fail)** — Invariants 2 and 4 evaluated at their worst joint corners, reported so
  we *know* which genomes are ambush-only vs. dash-capable, but never blocking.

So the practical constraints on the §2.1 ranges are just the two HARD gates; `size`'s turn multiplier
is narrowed to `[0.85, 1.15]` accordingly. This gate keeps every genome *aimable and correctly
sensed*, while allowing the strategic diversity (fast dash-hunters vs. slow ambush-scavengers) the
whole redesign exists to produce.

---

## 3. Per-snake physiology

The above means **`world.py`/`sensors.py` must read per-snake stats, not global `CFG`**, for:
max length/size, `energy_decay`, top cruise & dash speed, `turn_deg`, `s_max` + regen, `ray_range`,
smell reach, `max_lifespan`. Implement as a small resolved-stats accessor derived once from
`snake.genome` (e.g. `snake.phenotype` cached, or a `world._stat(snake, name)` helper).

**This is the largest single piece of the build.** Ponytail scoping: introduce the accessor and
route the physics/sensing sites through it; each gene is a scalar interpolation over an existing
quantity, so no new physics *equations*, only per-snake *values*. Keep the global `CFG` values as
the interpolation endpoints / defaults.

**Concrete `CFG.`-read sites that must become per-snake** (enumerate in the plan; the accessor step
gets its own review):
- `_move_snake` motion: `turn_deg`, cruise `v_snake`, `v_dash`, stun; stamina regen/drain, `s_max`.
- **Pitfall-5 pair (both must use the SAME per-snake `v_dash`, or a fast snake false-collides with
  its own neck):** the neck-skip `skip = head_radius + body_radius + v_dash + segment_spacing`
  (`world.py:269`, `_body_points_uw`) AND the prune slack `target_length + v_dash` (`world.py:231`,
  `_prune_path`). `skip − v_dash` is invariant, so this is self-consistent for any `v_dash` **iff
  the same value feeds both**.
- Energy: `energy_decay` (+ size surcharge) in the per-step hunger; growth per chicken.
- Sensing: `ray_range` (per-snake `ray_dirs` uses global `fov_deg`/`n_rays` — those stay global,
  only range scales) and smell reach/strength in `sensors.smell` / `_smell_field`.
- Lifespan → `max_lifespan` at birth (§5).
- **Return channel for the egg-lost penalty:** `_snake_eat` currently returns only a count
  (`world.py:530`); it must also surface the **owner-sets of eaten eggs** (analogous to
  `hatched_owners`) so `world.step` can hand `env` the penalty targets (§6).

---

## 4. Sex

- **Assigned randomly ~50/50 at hatch** (and at founder spawn). **Not heritable** — a random draw
  per birth (biologically correct; avoids sex-ratio drift).
- Stored `Snake.sex ∈ {0=female, 1=male}` (or a bool).
- **Sensed:** own sex (1 bit, proprio); nearest rival's sex (social channel).
- **Mating requires one male + one female** (see §6). The **female lays** the egg.

---

## 5. Aging & death by age

- `Snake.age` — increments each step.
- `Snake.max_lifespan = lerp(lifespan_min, lifespan_max, gene[lifespan]) × (1 + jitter)`, where
  `jitter ~ U(−lifespan_jitter, +lifespan_jitter)` (e.g. ±0.15) resolved **at birth** — so even
  same-genome siblings live different spans ("lives less than average" happens within a genome, not
  only across genomes).
- Death when `age ≥ max_lifespan`, new `death_cause = "age"` (spawns a corpse like any death).
- **Sensed:** `age / max_lifespan ∈ [0,1]` (proprio) — the snake feels its *life fraction*, not an
  absolute age (denominator is per-snake, exactly as requested).
- Death causes are now **`snake` (cut-off) / `starve` / `age`**. Still no `obstacle`/`self`
  (solid-slide non-lethal, Pitfall 19).

---

## 6. Reproduction: courtship → laying → guarding

Replaces "egg pops in at the midpoint." ~70% reframe + visuals of existing machinery, ~30% new
(courtship FSM, sex gating, female-lays, egg-lost penalty).

**FSM (world.py):**

1. **Approach** — *emergent*, unscripted. A repro-ready opposite-sex partner is visible in the
   social channel (rel pos, sex, repro_ready, relatedness); the brain closes in because it pays off
   at hatch. No forcing.
2. **Courtship** — an eligible pair (both alive, opposite sex, both `repro_ready`, cooldown 0) that
   **holds within `r_mate` for `mate_steps` consecutive steps** is *courting*. This reuses the
   existing hold-distance counter, now sex-gated. (Visual: slow circling / pause / heart particles —
   Phase B.)
   **`repro_ready` becomes size-relative:** the length gate is now a **fraction of the snake's own
   max length** (`length ≥ repro_length_frac × own_max_length`), not a global absolute — otherwise a
   small-`size` genome could never reach a fixed `repro_length_min` and would be sterile by
   construction. `repro_ready` is observed (Pitfall 12), so `sensors._repro_ready` reads this
   per-snake fraction; the value stays in [0,1] and the mating curriculum sweeps the *fraction*
   endpoints instead of an absolute length.
3. **Lay** — on courtship completion, the **female** lays **one** egg **adjacent to herself** — a
   small fixed offset behind her head (collision-safe), **NOT `_free_point`**: `_free_point`
   (`world.py:553`) is a *global* spawner that explicitly rejects points *near* snakes (it keeps the
   ego's start area clear), so it would drop the egg *far* from the female and defeat guarding. The
   existing parents-midpoint placement (`world.py:656`) already sits within `r_mate/2` and is
   collision-safe — keep that, or a small offset from her head. The egg carries `child =
   crossover+mutation` (§2.2), `owner = [femaleId, maleId]`, and a **lineage id = the female's**
   (maternal inheritance, simple and stable for the family-color, §10). Both parents pay `repro_cost`
   energy and enter `repro_cooldown`. **Egg arrays now thread `genome (N×9)` + `lineage` in lockstep
   with `pos/timer/owner` through every `keep=~eaten`/`keep=~hatched` filter** (`world.py:519-522,634`)
   — name this lockstep surface in the plan (Pitfall-17-style: one wrong filter desyncs a genome from
   its egg).
4. **Incubate / guard** — egg lies `egg_timer` steps. **Eatable by any non-owner** (`egg_food`).
   Owners cannot eat their own egg (existing gate) and do not trample it (eggs aren't destroyed by
   contact). Reward to owners comes **only at hatch** (existing) → guarding emerges. (Visual: egg
   wobble/pulse → hatch crack — mostly existing.)
5. **Hatch** — egg → fresh `Snake`, **random sex**, inherited genome, `hatch_energy_frac` energy,
   fresh `max_lifespan` (own jitter). Pays `reward_repro` to surviving owners.

**Egg-lost penalty (new, small, tunable — but DEFAULT OFF for the discovery retrain).** When an egg
is **eaten** (not at hatch, not on population-cap-drop — a cap-dropped egg pays neither reward nor
penalty), each surviving owner receives `reward_egg_lost` (negative, strictly `< reward_repro` so
breeding stays net-positive; applied to the ego iff it co-owns the eaten egg). **Because reproduction
is already the hardest thing to bootstrap (Pitfalls 9–11 = multiple aborted runs), attaching a
downside to *laying* before guarding is learned can suppress laying the way `dash_penalty` suppressed
dashing (Pitfall 1's cousin). So `reward_egg_lost` DEFAULTS to `0.0` for the from-scratch retrain**
(mirroring `dash_penalty = 0.0`, `config.py:126`); it is enabled only as a **post-competence
sharpening knob** (e.g. `−4.0`) once guarding behavior exists, and re-evaluated (a short resume or
just viewer-runtime, since it doesn't change the observation). Requires the eaten-egg owner-set
return channel from `world.step` (§3).

Guaranteed **arrival eggs** (owner `[-1,-1]`, world start + viewer reseed, §9) keep their existing
special-casing: uneatable, `n_max`-cap-exempt, not a "birth", pay no reward. Their carried genome is
a fresh founder genome, their hatchling gets a random sex.

---

## 7. Observation redesign

All additions are per-frame; frame-stacked ×4 as before. Constant-per-life fields (genome, sex,
lifespan-derived) are stacked redundantly — harmless, keeps the pipeline simple.

| Block | Was | Now | New content |
|-------|-----|-----|-------------|
| **Vision** | 11 rays × 8 | 11 rays × **9** | + per-ray **target motion/activity** = the hit's **state-nominal** speed (NOT raw instantaneous): pecking hen → 0, walking → low, **fleeing/startling → high**, dashing snake → high. State-nominal (not instantaneous) is deliberate — a hen in its startle-*freeze* (speed 0, about to bolt) must read "flee," not collide with a pecking hen's 0. One scalar then cleanly encodes walk/flee/peck. |
| **Social** | 7 | **11** | existing 7 + `relatedness` + rival `energy` (weakness cue) + rival `repro_ready` + rival `sex` |
| **Egg** | 4 | 4 | unchanged |
| **Smell** | 9 | 9 | unchanged (chicken/snake/corpse); *reach* now per-snake (`senses` gene) |
| **Vibration** | — | **3** | omni "ground-sense" motion field: intensity + gradient(fwd,left), magnitude ∝ others' speed — feel a dasher/fleer you can't see (pairs with speed mechanics; a stopped stalker is "silent") |
| **Proprio** | 5 | **17** | existing 5 (energy/length/stamina/repro_ready/speed) + own `sex` + `age_frac` + `stun` (normalized) + own **genome (9)** |

**Resulting `OBS_DIM ≈ 143** (11×9 + 11 + 4 + 9 + 3 + 17 = 143). The plan will pin the exact number
and hand-enumerate the observation-space bounds in `env._make_observation_space`, mirroring the
current layout discipline. `frame_stack=4` → policy input ≈ 572.

**Vibration field** is a smell-like field over live rivals + fleeing chickens weighted by their
speed, **un-occluded** (its one genuinely-new signal over the occluded smell field is "feel a dasher
behind a rock / beyond the vision cone"). Its ±bound is enumerated at `n_max + chicken_ceiling`
(= 12 + 12 = 24) and clipped for `observation_space.contains`. **It is the most cuttable new sensor**
(rivals are already in the smell-snake field, their visible speed in per-ray motion, the nearest
rival's dash in social `is_dashing`) — kept in v1 per the user's "all sensors in", but it is the
**first thing to drop** if the whole-box bootstrap (§11) struggles, since fewer obs dims = less
bootstrap burden.

**Bounds & normalization discipline (Pitfall-12 corollary — the derived stats bite, not the genes):**
several current normalizers are global `CFG` and break or lose granularity once stats are per-snake —
name each in the plan:
- **`smell()` chicken/snake intensity is NOT clipped** (`sensors.py:161-173`; only corpse is). A
  per-snake smell reach ×1.4 pushes intensity to `1.4×chicken_ceiling` → `observation_space.contains`
  fails → `check_env` fails for high-`senses` genomes. **Fix:** clip chicken/snake intensity to
  `chicken_ceiling` like corpse already is (or widen the smell bound by the max reach multiplier).
- **Per-ray motion** normalized by global `v_dash`: a rival dashing at `v_dash×1.2` reads 1.2 → clip
  to 1, or normalize by the **observer-independent global max** `v_dash × max_speed_gene`.
- **Proprio `speed`/`stamina`/`length`** are already `np.clip`'d (`sensors.py:228-231`) so bounds
  hold, but a high-`stamina`/`size` genome **saturates at 1.0 well below its own reserve/length**,
  losing the granularity the brain needs to time dashes. **Prefer per-snake denominators**
  (`stamina / own_s_max`, `length / own_max_length`) so the signal spans [0,1] for every genome.

**Stun sensor:** `Snake.stun` (already a countdown field) normalized into proprio, so the brain can
credit "dash into solid → stun → wasted time" and learn to avoid it (Pitfall 19's opportunity cost
becomes directly observable).

---

## 8. No ego / spawn-only-from-eggs

- **Viewer & world semantics:** there is **no special snake**. `worldgen` for the viewer starts
  with **only arrival eggs** — no live snake at all at t=0. Snakes exist only by hatching. **This is
  a cross-cutting refactor of the `snakes[0]`-is-the-ego assumption, not a one-line drop — it gets
  its own Phase-A step with its own review.** Concrete sites that hard-assume a live slot-0 ego and
  must tolerate *zero live snakes*:
  - `world.step` (`world.py:727-729`): `ego = self.snakes[0]; not ego.alive; ego_dashed` →
    `IndexError` on an empty list. `step` is shared with training (which *needs* the ego return), so
    it **bifurcates**: return `died=None`/no-ego info when there is no privileged/live ego (viewer),
    keep the ego return for the training env.
  - `world._free_point` (`world.py:563,565`) reads `self.head` (→ `snakes[0]` via the `_ego_prop`
    descriptors) to reject the ego's start area — but it's what worldgen (`worldgen.py:37`) and the
    reseed floor (`watch.py:91`) call to place the founder eggs, so the all-eggs world can't place
    its own founders without an ego. **Strip the ego-area rejection** (it only ever protected the
    ego's spawn; irrelevant for placing eggs).
  - `watch._step_world` (`watch.py:101-102`), `run_watch` `follow_id = world.snakes[0].id`
    (`watch.py:286`), `World.__init__`/`_ego_prop` (`world.py:96-101`), `_prune_dead`
    (`world.py:551`): guard every "slot-0 exists" read for an empty live set (follow falls back to
    "no target / overview" until an egg hatches).
- **Training plumbing:** SB3 PPO requires exactly one learner agent with a body every step.
  `env.reset` therefore spawns **one live gradient-source snake** (the rest as eggs). It has **zero
  special powers** — ages, mates, dies, is sensed like any other; it is merely where gradients are
  collected, and it is **never rendered/watched**. When it dies the training episode ends (short
  episodes are fine: evolution is a runtime phenomenon, §1, so PPO only needs a genome-competent
  brain).
- **Genome during training:** the gradient-ego's genome (and opponents') is **randomized per
  reset/hatch** (domain randomization over the gene box) so the brain learns to read the genome
  across its full range.
- The "purist" alternative (multi-agent experience collection → no ego even in training) is
  **explicitly out of scope for v3** — more `train.py` risk, and it does not enable evolution
  (which is runtime). Noted as a possible future.

---

## 9. World, population, balance

- `n_max = 12` (was 6); `n_start_min/max` scaled up (e.g. 3/6 as *egg* arrivals).
- `world_size` enlarged (e.g. `180–260`) so 12 snakes have room and the map exceeds the screen
  (enables camera pan/zoom, §10). Density (obstacles, food rates) rescaled to the larger area.
- **Caps that key on `n_max`** (smell fields, vibration field, `chicken_ceiling ≥
  chickens_per_snake_max·n_max`) updated for `n_max=12`.
- Retrain cost rises (12 snakes + larger obs + bigger world ⇒ more entities/step). Budget more
  wall-clock; keep `--envs 16`.
- Guaranteed arrival eggs (owner `[-1,-1]`) are the **only** spawn mechanism at world start; the
  viewer reseed floor (`watch._reseed_floor`) already lays arrival eggs and needs only the
  all-eggs-start adjustment.

---

## 10. Viewer (Phase B — no retrain)

- **Camera modes:**
  - **Free:** arrow keys pan; mouse-wheel / `+`/`-` zoom.
  - **Follow:** locked to a snake; `[` / `]` cycle prev/next live snake; `Tab` toggles free↔follow.
  - **Death handling:** when the followed snake dies, the camera **lingers 3 s at the death spot**,
    then advances to the next live snake.
  - Sim-speed control moves off the arrows to `,` / `.` (arrows are now pan).
  - Render transform (`render._p/_circle/_blit_world/_scale`) gains a **camera offset + zoom**; the
    world no longer fits-to-screen.
- **Genome as phenotype (legibility):** hue = **lineage/founder** (a stable "family" color carried
  down a line), saturation/markings = trait mix, body **size** = `size` gene. So families and rough
  "build" read at a glance. (Founder lineage id assigned at each founder egg; inherited by
  offspring.)
- **Genome inspector:** for the followed snake, an overlay with gene bars, sex, age/lifespan,
  lineage, and life stats (kills, offspring). This is where "who became a killer vs. a cooperator"
  becomes readable.
- **Visual FX:** stun "dizzy" (spinning stars) while `stun > 0`; courtship (hearts/pulse) during the
  hold; egg wobble during incubation (hatch crack already exists).

---

## 11. Retrain

- **From scratch** (`--reset`); no v2 checkpoint is resumable (obs/physics/reward/context all
  changed).
- Watch the Pitfall 9–10 bootstrap: `snake/eaten_per_window` must climb in the first ~50–100k
  steps across the *randomized genome* population; if flat near 0, revisit prey-alert (1× cap) or
  narrow the harshest gene extremes. Abort early if collapsed.
- Curriculum: keep the dual stamina+mating hardness ramp; genome ranges are **fixed** (not annealed)
  — the brain must handle the whole box from the start (domain randomization), while
  stamina/mating harden as before.
- Judge by the *end*, and by behavioral diversity in the persistent eval, not the mid-ramp dip.

---

## 12. Testing (extend the suite; one runnable assert per new mechanic)

New/changed tests:
- **Genome:** crossover picks per-gene from a parent; mutation stays in [0,1]; relatedness metric
  (identical genomes → 1, opposite → 0).
- **Per-snake physiology:** two genomes yield different resolved stats (speed/size/hunger/ray_range);
  `assert_invariants_over_genome` passes at all extremes.
- **Sex:** ~50/50 at hatch (statistical); mating gated to M+F (same-sex pair never lays); female is
  the layer.
- **Aging:** death at `age ≥ max_lifespan`; `age_frac` sensor in [0,1]; sibling lifespan jitter
  differs.
- **Reproduction FSM:** courtship hold → female lays adjacent egg with crossover genome + both
  owners; egg eatable by non-owner, not by owner; reward at hatch; **egg-lost penalty** on eat.
- **Sensors:** relatedness / rival-state in social; vibration field responds to a moving rival and
  not a stopped one; per-ray motion channel distinguishes pecking (≈0) vs fleeing (≈high) hen; stun
  sensor reflects `stun`.
- **No-ego / all-eggs:** viewer worldgen starts with 0 live snakes and ≥1 arrival egg; a snake
  appears only after `egg_timer`.
- **Obs bounds / check_env:** `observation_space.contains(observe(...))` for extreme genomes;
  `check_env` single- and (egg-based) multi-snake.
- Keep all existing invariants/tests that still apply (torus, raycast, solid-slide, two-phase step,
  cut-off, corpse, PBRS, opponent-obs parity).

---

## 13. Build phasing

- **Phase A (feeds the retrain)** — heavier than one line each; the plan splits these into
  independently-reviewed steps, roughly in this order:
  1. **Genome + inheritance + relatedness** (dataclass field, crossover/mutation, relatedness metric).
  2. **Gene-range resolution + invariant-over-genome gate** (§2.4 — resolve the actual HARD-safe
     ranges FIRST; downstream physics depends on final ranges).
  3. **Per-snake physiology accessor** + routing the §3 sites (incl. the Pitfall-5 `v_dash` pair).
  4. **Sex**; **aging/death-by-age**.
  5. **Reproduction FSM** (courtship→lay→guard, sex-gated, egg carries genome+lineage; egg-lost
     penalty plumbed but **defaulted 0** for the retrain).
  6. **Observation redesign** (all new channels + the §7 bounds/normalization fixes; enumerate exact
     `OBS_DIM` and env bounds; opponent-obs parity check).
  7. **No-ego / all-eggs refactor** (its own step — the §8 `snakes[0]` sites); training plumbing
     keeps one live gradient-ego.
  8. **World/population scaling** (`n_max=12`, bigger world, `n_max`-keyed caps, food/obstacle
     density).
  Every step lands with its tests; steps 3, 6, 7 get their own reviews (highest-risk).
- **Retrain** (from scratch).
- **Phase B (no retrain):** camera (free/follow/cycle/3-s-linger, pan+zoom); genome→color/pattern/
  size; genome inspector overlay; stun & courtship & egg-wobble FX.

Each phase gets a superpowers code review; important steps within Phase A (physiology accessor,
obs redesign, reproduction FSM) get reviews too; a final review at the end.

---

## 14. Risks & open items

- **Physiology refactor surface** (§3) is the main risk — many `CFG.x` reads become
  `stat(snake, x)`. Mitigate: single accessor, route incrementally, test each.
- **Gene box vs. invariants** (§2.4): some ranges may need narrowing to keep every genome playable.
- **Bootstrap under randomized genomes** (§11): a fresh policy must hunt across the *whole* gene box;
  if the weakest genomes never bootstrap, the average signal may stall. Fallback: start with
  narrower gene ranges, widen after competence.
- **Obs growth** (113→~143, ×4≈572): larger net input, slightly slower; acceptable.
- **Bootstrap under randomized genomes is the sharpest risk** (§11): a fresh policy must hunt across
  the WHOLE gene box from scratch, and reproduction (the hardest thing to bootstrap, Pitfalls 9–11)
  now also depends on courtship + sex. Mitigations baked in: peck-distraction hunting needs no dash
  (so weak genomes still catch), `reward_egg_lost` defaults off, and the fallback is to **start with
  narrower gene ranges and widen after competence**. Watch `eaten_per_window` in the first ~50–100k
  steps; abort early if flat.
- **Effort is front-loaded and heavier than a bullet list reads:** the physiology accessor (§3), the
  no-ego refactor (§8), and the egg-array genome/lineage lockstep (§6) are each real multi-site
  changes, not one-liners. The phasing (§13) reflects this; don't under-budget Phase A.
- **CLAUDE.md**: must be updated in the same work (new pitfalls, retrain notes, config table) per the
  project's living-memory rule.
