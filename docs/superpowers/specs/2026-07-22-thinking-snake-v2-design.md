# Thinking-Snake v2 — design (2026-07-22)

One **from-scratch retrain**. Goal: snakes navigate by their own **perception + world physics**,
not by dying to scenery. Obstacles stop being lethal; snakes get finer forward sight, graduated
speed (→ emergent ambush/stalk/obstacle-threading), and prey that reacts to a snake's *speed*.
Presentation (egg-arrivals, sky-chickens) already landed via a subagent.

Design philosophy (user): **give the agent richer sensing + realistic physics and let it decide;
do NOT hardcode action gates.** A stun after ramming a wall is a *consequence*, not a gate — the
snake stays free to dash anywhere.

## 1. Collision model — NEW physics (`world.py`) [retrain]

Replace "obstacle/self collision ⇒ death" with solids:

- **Obstacles (rocks + trees) and a snake's OWN body → SOLID, non-lethal.** When the swept head
  step `p0→p1` would penetrate a solid, the head **slides**: clamp to the contact point along the
  motion normal, then project the remaining motion onto the surface tangent (one cheap iteration).
  No death.
- **Dash into a solid → STUN.** If the *blocked* move was a dash, the snake enters `stun` for
  `stun_steps` (~10 ≈ 1 s). While stunned: no translation, **steering frozen** (heading held),
  stamina regens, `stun -= 1` each step ("head spinning"). A *walk* into a solid just slides, no stun.
- **Rival body/head → LETHAL (unchanged cut-off).** A head contacting a *rival's* body or head still
  dies, cause `"snake"`. Predation/herding tactic preserved (Model A).
- **Death causes now: `starve`, `snake` only.** `obstacle` and `self` deaths are gone.

Notes:
- Own body still uses `_body_points_uw` **with the neck-skip** (Pitfall 5) for the slide test, so
  straight motion never self-blocks on its own neck.
- Two-phase step (Pitfall 13) preserved: slide + stun are per-snake, resolved in **phase 1**
  (`_move_snake`), against static obstacles / the snake's own frozen body; **rival lethality stays
  phase 2** against the frozen post-move state (slid heads are final by then).
- `Snake.stun: int = 0` new field.

## 2. Sensors (`sensors.py`, `env._make_observation_space`) [retrain]

- **`n_rays 9 → 11`:** keep the 9 uniform rays over ±135°, add **2 forward** rays at ±½·(uniform
  spacing) ≈ ±16.9° (bisect the two central gaps). `ray_dirs` builds an explicit angle array.
- **Un-mask obstacles:** each ray gains an `obstacle_clearance` channel = distance to the nearest
  **obstacle** along that ray (obstacles only — rocks AND trees; ignores chickens/bodies/eggs so a
  chicken-in-a-gap can't hide the rock behind it), `/ray_range`. Ray = `[dist, is_obstacle,
  is_chicken, is_self, is_other_body, is_egg, is_corpse, obs_clearance]` (8). Vision = 11×8 = **88**.
- **Spawn-egg fix** (flagged by the arrivals subagent): exclude owner<0 (arrival) eggs from
  `_egg_channel` — they're uneatable, so don't point the policy at them as food.
- **Proprio +1:** add current speed fraction `speed / v_snake`. Proprio 4 → 5.
- **`OBS_DIM 87 → 113`** = vision 88 + social 7 + egg 4 + smell 9 + proprio 5. Frame-stack ×4 = 452.
  Update the hand-enumerated bounds; new channels are `[0,1]`. `OpponentController` mirrors via
  `OBS_DIM` (verify no hardcoded 87).

## 3. Action / graduated speed (`env.py`, `world.py`, `config.py`) [retrain]

- `action_space = MultiDiscrete([4, 3, 2])` = **speed** `{0, ⅓, ⅔, 1}×v_snake` × **steer** `{L,S,R}`
  × **dash** `{0,1}`. Dash is the separate sharp burst `v_dash` (overrides cruise speed,
  stamina-gated as today).
- `_move_snake(s, speed_idx, steer, dash)`: cruise = `speed_levels[speed_idx]·v_snake`; `speed 0`
  ⇒ rotate in place (steer turns heading, no translation). Lower speed ⇒ tighter turn radius
  (`R = speed/turn_deg`) ⇒ emergent obstacle-threading, no gate.
- `config.speed_levels = (0.0, 1/3, 2/3, 1.0)`; `Snake.speed` field (set each move).

## 4. Prey senses motion (`world.py` chicken FSM) [retrain]

- A snake's contribution to a chicken's flee scales with the snake's **speed, CAPPED at 1× base**:
  effective alert = `base_alert · clip(snake.speed / v_snake, 0, 1)`. `base_alert` stays
  state-dependent (`r_flee_peck` if the chicken pecks, else `r_flee`). Full cruise **or dash** → base
  alert (unchanged from today); slow → proportionally less; **speed 0 ⇒ alert 0 ⇒ full ambush**.
  We deliberately do NOT let a dash exceed base alert — a 2× would push walking-chicken flight to 24
  and risk the Pitfall-9/10 hunting-discovery collapse on the fresh retrain; capping at 1× keeps the
  max alert = today's `r_flee`, so hunting-bootstrap and catch-invariant #3 are preserved. Enables
  stalk-slow / freeze-and-strike, emergent.

## 5. Reward — SIMPLIFIED back to clean (`env.py`, `config.py`) [retrain]

Obstacles are no longer lethal, so **remove the entire Pitfall-16 machinery**: delete
`reward_death_obstacle`, `obs_avoid_weight`, `obs_avoid_range`, `_phi_obstacle`, the obstacle term
in `_shaping`, `_last_phi_obs`. Reward returns to: `reward_eat`, `reward_repro`, flat `reward_death`
(for the remaining `starve`/`snake`), `step_penalty`, chicken-PBRS. Physics (stun) + sensors + speed
now carry obstacle-avoidance, not reward.

## 6. Arrivals — DONE (subagent) + animation polish (subagent, in progress)

Egg-based snake arrivals + sky-drop chickens already implemented (`world.arriving`,
`spawn_egg` owner `[-1,-1]`, `chicken_arrive_steps`, Pitfalls 17–18). A follow-up polishes the
sky-drop **animation** (render only, no retrain).

## Retrain

`./snake train --steps 8000000 --envs 16 --reset` (bigger action/obs may want more steps; judge by
the metrics). Curriculum unchanged. **The environment has killed long background trainings twice** —
likely run this one foreground via `!` or expect to eval the latest checkpoint.

`assert_invariants`: recheck — `stun_steps ≥ 1`; `speed_levels` max = `v_snake` (turn-radius /
self-reachability invariants #4/#6 use `v_snake`, unchanged); catch invariant #3 uses `v_dash`,
unchanged. Obstacle-lethality-related asserts (if any) revisited.

## Tests (add/adjust — repo does one runnable assert per mechanic)

- Collision: slide along an obstacle without dying; dash-into-solid ⇒ stun (frozen N steps);
  walk-into-solid ⇒ slide, no stun; **own body non-lethal** (curl onto self ⇒ no death); **rival
  body still lethal**; stunned snake can't move/turn.
- Sensors: 11 rays; **un-mask** (chicken directly in front of a rock ⇒ `obs_clearance` still reports
  the rock); spawn-egg (owner<0) excluded from `_egg_channel`; `OBS_DIM==113`; `check_env`.
- Speed: 4 levels map correctly; `speed 0` ⇒ no translation but heading rotates; **prey alert
  scales with speed** (a stopped snake beside a chicken doesn't spook it; a full-cruise one does).
- Update every obs-dim / action-shape dependent test + selfplay parity.

## CLAUDE.md

Update in the same change: architecture table (world collision, action, obs rows), the RL design
(obs 113, `MultiDiscrete([4,3,2])`, solid-slide+stun collision, graduated speed, prey-senses-motion),
config + invariants, judging (deaths are now `starve`/`snake` only), and the Pitfalls — **retire/rewrite
Pitfall 16** (obstacles no longer lethal; the reward saga is obsolete — keep a short note on *why*
the physics approach replaced it), add pitfalls for solid-slide+stun and graduated-speed/prey-motion.
