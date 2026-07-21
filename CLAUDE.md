# CLAUDE.md — Snake-RL project guide

A PPO-trained predator snake in a continuous 2D torus world. It hunts fleeing chickens by
egocentric **sight** (9 vision rays) + **smell**, sprints with a stamina-limited **dash**,
and dies on collision with rocks/trees or itself. Trained headless (SB3 PPO, CPU), watched
in a fullscreen pygame viewer. **The behavior is meant to look alive** — stalk, deliberate
pounce, avoid clutter — not to be optimal.

> **⚠️ Multi-snake WIP (branch `multi-snake`, Milestone A complete).** The simulation (`world.py`/`worldgen.py`/`config.py`) now supports N snakes — inter-snake cut-off death, corpses, egg-based reproduction, starvation, and population-scaled food — and the world was rescaled (`world_size` 60/100 → 110/160, food now ~2 chickens per live snake, a new `starve` death cause). **`env.py`/`sensors.py` still run single-snake, so the shipped `models/` snake is now a PLACEHOLDER:** it was trained on the old 60/100 world with 3–5 chickens, so its live behavior and the judging band below no longer describe this world until the Milestone B multi-snake retrain. Design + step plan: `docs/superpowers/specs/2026-07-21-multi-snake-world-design.md` and `docs/superpowers/plans/2026-07-21-multi-snake-milestone-a-simulation.md`.

> **This file is the living memory of the project. Keep it current.** When you change the
> reward, the observation, the physics, the training recipe, or the hyperparameters — update
> the relevant section here *in the same change*, especially the **Pitfalls** and
> **What needs a retrain** lists. Future-you will rely on it to avoid re-learning the hard way.

---

## Run it

```bash
./snake watch                 # fullscreen viewer (a trained model ships in models/)
./snake watch --windowed      # windowed
./snake watch --headless --episodes 20   # behavioral eval, prints metrics (no window)
./snake train --steps 6000000 --envs 16 --reset   # train from scratch (~15–20 min)
```

`./snake` is a launcher that creates `.venv` and installs deps on first run — never touch pip
directly. Watch keys: `SPACE` pause · `N` new world · `S` sensors · `D` deterministic ·
`↑/↓` speed · `ESC` quit.

**Env:** macOS, Python 3.13 venv at `.venv/`. Always run Python via `.venv/bin/python` (or
`./snake`). For any headless script that imports `snake_rl`, set `PYTHONPATH="$PWD"` and
`SDL_VIDEODRIVER=dummy` (pygame needs a driver; `dummy` = no window). `SubprocVecEnv` cannot
be created from a `python - <<EOF` heredoc (multiprocessing needs a real module file) — write
the script to a file and run it.

Design docs (spec + step plan) live in `docs/superpowers/specs/` and `docs/superpowers/plans/`.

---

## Architecture (one responsibility per file)

| File | Does | Retrain if changed? |
|---|---|---|
| `snake_rl/config.py` | ALL constants (frozen `Config`) + `assert_invariants` | usually yes |
| `snake_rl/world.py` | torus geometry (nearest-image), snake motion, dash/stamina, chickens, swept collision, `step()` | yes (dynamics) |
| `snake_rl/worldgen.py` | random world (size, obstacles, chickens), rejection sampling | yes |
| `snake_rl/sensors.py` | vectorized raycast + smell → 42-float observation | **yes** (obs) |
| `snake_rl/env.py` | Gymnasium env: obs, MultiDiscrete action, PBRS reward, truncation, `set_hardness` | yes (reward/obs) |
| `snake_rl/train.py` | SB3 PPO, VecEnv stack, curriculum callback, checkpoints | recipe only |
| `snake_rl/render.py` | pygame drawing (AA, HUD, effects) | **no** (visual only) |
| `snake_rl/watch.py` | load model, interpolated viewer, headless eval | no (except obs plumbing) |
| `snake_rl/__main__.py` | CLI dispatch | no |

Data flow: `worldgen → World` (physics) → `sensors.observe(world)` → `env` (reward/done) →
PPO. Viewer: `watch` loads the checkpoint, steps one env, `render` draws it with interpolation.

**Vec-wrapper stack** (matters for accessors): `VecNormalize( VecFrameStack( DummyVecEnv([ Monitor(SnakeEnv) ]) ) )`.
The underlying world is `vec.venv.venv.envs[0].unwrapped.world` (see `watch._world_of`).

---

## The RL design (and why each piece is the way it is)

- **Algorithm:** SB3 `PPO`, `MlpPolicy`, **CPU** (SB3 doesn't support MPS; the net is tiny so
  CPU + `SubprocVecEnv` beats GPU/MPS). 16 parallel random worlds.
- **Observation (42 floats, egocentric, frame-stacked ×4 = 168):** 9 vision rays
  `[dist, is-obstacle, is-chicken, is-self]` + smell `[intensity, grad-fwd, grad-left]` +
  proprioception `[energy, length, stamina]`. Egocentric + fixed-size is what lets a `VecEnv`
  of **different-sized** worlds share one obs space. **Vision inflates every target by
  `head_radius`** (Minkowski) so a ray reports *distance until the head EDGE touches* — the
  snake perceives its own width and stops clipping rocks.
- **Action:** `MultiDiscrete([3,2])` = steer `{left,straight,right}` × dash `{no,yes}`.
- **Reward:** `+10` eat, potential-based shaping toward the nearest chicken (PBRS), `−10`
  death, tiny hunger-scaled step penalty. Keep sparse rewards big vs the dense step penalty.
- **Memory:** `VecFrameStack(4)`, NOT an LSTM. The only real memory need is remembering the
  body for self-collision, and frame-stack handles it — trained models show **0 self-deaths**.
  Only reach for `sb3-contrib RecurrentPPO` if "eats own tail" ever reappears (drop FrameStack
  then; ~2–4× slower on CPU).
- **Deliberate dash = MECHANICAL rationing, never a reward penalty.** Dash needs a full
  stamina reserve (`dash_min_stamina`) to fire and the reserve refills slowly (`stamina_regen`).
  So the snake earns a dash by walking and spends it in a burst → stalk-and-pounce emerges,
  with the reward left as a clean "catch chickens." See Pitfalls for why penalties fail.
- **Curriculum (automatic, one run):** stamina difficulty is **annealed** — `hardness=0` (easy
  free dash) for the first `hardness_warmup` of training so it learns to hunt, then linearly
  ramped to `1.0` by `hardness_full`. `train.AnnealHardness` pushes `hardness` into every env
  each rollout via `env_method("set_hardness", h)`; `env.set_hardness` interpolates the gate +
  regen. A fresh run starts at 0 and anneals; a **resume** starts fully hard (no callback).

---

## PITFALLS — the hard-won lessons (read before touching reward/stamina/hyperparams)

Every one of these cost real training runs. Do not rediscover them.

1. **A per-step reward penalty on dash COLLAPSES hunting.** The agent learns "never dash, just
   survive" (a lazy local optimum) before it discovers that dashing → chickens → `+10`. A
   penalty big enough to curb reflexive dashing is big enough to kill hunting. *Ration the dash
   mechanically (stamina), not with reward.* (`dash_penalty` exists but defaults to `0`.)
2. **An abrupt easy→hard stamina switch COLLAPSES the learned hunter.** Resuming a good hunter
   straight into the hard reserve makes reward crater (42 → 3 in ~80k steps) and it never
   recovers. *Anneal the hardness gradually* (`AnnealHardness`), never flip it.
3. **A decaying learning rate makes the curriculum FRAGILE.** The task is *non-stationary*
   (stamina hardens mid-training); with `lr → 0`, by the time the reserve is tight the policy
   can no longer adapt and collapses to survive-only. **Use a CONSTANT `learning_rate=3e-4`**
   with `target_kl=0.03` as the stability guard. (This was THE fix that made the ramp reliably
   recover to +50–77 instead of collapsing.)
4. **The ramp dip is NORMAL — watch for RECOVERY, not the dip.** During hardening, reward dips
   (even negative) as the snake relearns to hunt with the tighter reserve, then climbs back.
   Judge a run by its *end* (full hardness), not the middle. A run that stays negative /
   `eaten≈0` through consolidation has truly collapsed — then revisit lr / warmup length /
   final `stamina_regen`.
5. **Over-tuning PPO SLOWED hunting *discovery* → survive-only.** `gamma=0.995` (slower value
   learning) and `batch_size=512` (halves gradient steps: 32 vs 64 minibatches over the 16384
   buffer) both delayed discovery so the policy committed to surviving. **`gamma=0.99`,
   `batch_size=256` reliably discover hunting.** Research configs optimize *final* quality, not
   *discovery* — get hunting working first, refine second.
6. **More envs ≠ automatically faster learning.** Rollout buffer = `n_steps × n_envs`. Bumping
   `--envs` with the same `n_steps` gives a bigger buffer → *fewer* PPO updates per env-step →
   slower learning for the same step budget. Keep the buffer ≈16384 (`n_steps=1024 × 16 envs`)
   and `batch_size=256` (→64 minibatches). If you change `n_envs`, re-derive `n_steps`.
7. **PBRS: zero shaping only on a chicken-SET change (eat/spawn), NOT on a nearest-identity
   switch.** `Φ = −dist_to_nearest` is *continuous* as which chicken is nearest switches (the
   two distances are equal at the crossover), so zeroing there throws away valid guidance and
   cripples hunting when there are many chickens. Zero only when the id-set changes (see
   `env._shaping`, tracked by `frozenset(chicken_id)`).
8. **Rays fire from the head CENTER — inflate targets by `head_radius`** or the snake clips
   obstacles with the edge of its head where no ray points. This one change ~halved obstacle
   deaths. (`sensors._scan`: `rad = rad + c.head_radius`.)
9. **The self-collision neck-skip must clear the whole swept step**, not just the head:
   `skip = head_radius + body_radius + v_dash + segment_spacing`. Otherwise a snake moving
   *straight* collides with its own neck (the swept segment's tail reaches into the skipped
   region).
10. **Small `r_flee` (chickens bolt only when very close) HURTS hunting.** Tried `r_flee=5` for
    "short pounces" — the snake can't turn fast enough (`turn_deg`) to track a bolting chicken
    at close range and stops eating. Keep `r_flee=12`.
11. **pygame/macOS rendering:** SRCALPHA sprites need `.convert_alpha()` or they render as black
    squares on a real display (fine under `dummy`); `smoothscale` *into* the window surface
    renders black — smoothscale to a temp then blit; fullscreen needs `FULLSCREEN|SCALED` and
    you must read back `display.get_size()` (Retina). Clear the sprite cache on `set_mode`.
12. **Monitoring steals CPU.** Running evals / frame renders *while* training drops training fps
    (env stepping and the eval compete for the 8 perf cores). Monitor with cheap `grep`s, not
    heavy scripts, or expect a slower run. Kill stray runs with `pkill -f "snake_rl train"`;
    orphaned `forkserver` workers linger and slow everything (`pgrep -f forkserver`).

---

## Config & invariants (`config.py`)

All tunable numbers live in one frozen `Config`. `assert_invariants(cfg)` runs at env init and
**fails fast** if a change breaks a guarantee — respect these when tuning:

1. `v_dash > v_flee` — a dash can out-run a fleeing chicken.
2. `(s_max/drain)·(v_dash−v_flee) ≥ k·r_flee` — a full reserve closes the flee radius.
3. `turn_deg/2 < atan(eat_radius/r_flee)` — the snake can aim precisely enough to catch.
4. `2π·v_snake/rad(turn_deg) < length_cap` — self-collision is physically reachable.
5. `ray_range + obstacle_radius_max + head_radius < world_size_min/2` — nearest-image raycast is valid.
6. `length_cap < world_size_min/2` — the body never wraps the torus onto its own head.

Current tuned values that produce the shipped model: `gamma 0.99`, `r_flee 12`, `stamina_regen
0.3`, `dash_min_stamina 1.0`, `min_chickens 3 / max 5`, `hardness_warmup 0.42 / full 0.85`,
`dash_penalty 0`. Easy (warmup) stamina: `dash_min_stamina_easy 0.05`, `stamina_regen_easy 0.6`.

---

## Making changes & retraining

**Decide first: does this need a retrain?** Anything that changes what the policy senses or is
rewarded for, or the physics it acts in, DOES. Pure render/viewer changes do NOT.

- **Needs retrain:** `sensors.py` (obs values or dim), `env.py` reward/action/obs-bounds,
  `world.py` physics/stamina/chickens, `worldgen.py`, stamina/reward/`gamma` constants.
- **No retrain:** `render.py`, `watch.py` viewer loop, HUD, effects, `__main__` CLI, fullscreen.

**Retrain recipe (reliable, ~15–20 min):**
```bash
./snake train --steps 6000000 --envs 16 --reset
```
- Back up the current good model first if it's precious: `cp -r models models_good_backup`
  (gitignored). Models are NOT committed (`.gitignore`).
- Expected trajectory: `ep_rew_mean` ≈ −10 early (learns to survive) → climbs to +40–50 by the
  end of warmup (hunting on easy dash) → **dips during the ramp** (relearning with the reserve)
  → recovers to **+50–77** at full hardness. Watch `hardness` in the SB3 table ramp 0→1.
- If it collapses (see Pitfall 4): lengthen `hardness_warmup`, keep lr constant, or soften the
  final `stamina_regen` (e.g. 0.3 → 0.4 makes hunting easier to maintain, economy slightly less
  pronounced). Change the `--seed` — PPO is high-variance; a bad seed can collapse a good recipe.
- To keep training a good model further: `./snake train --steps 3000000` (no `--reset` → resume,
  starts fully hard, restarts the SB3 lr/optimizer for the new call).

**PPO hyperparameters live in `train.py`** (`PPO(...)` for a fresh run): constant `lr 3e-4`,
`n_steps 1024`, `batch_size 256`, `n_epochs 10`, `gamma` from CFG, `gae_lambda 0.95`,
`ent_coef 0.01`, `target_kl 0.03`, `net_arch pi/vf [128,128]`. See Pitfalls 3, 5, 6 before
touching these.

---

## Judging a model (behavioral, not just reward)

`./snake watch --headless --episodes 20` prints the metrics that matter. A good model:
- **catch rate** ≈ 10–14 chickens / 1000 steps (hunting works);
- **dash usage** ≈ 25–30% of steps (deliberate bursts, not constant);
- **stamina** reserve builds > 10 for ~35–48% of the time (the economy cycles, not pinned at 0);
- **deaths** almost all `obstacle`, `self` ≈ 0 (memory works);
- **episode len** ≈ 500–700 steps.
Also glance at the SB3 table during training: `explained_variance` high + `entropy_loss` low +
reward flat-negative = converged to survive-only (bad). `snake/hardness` should reach 1.0.

---

## Testing

`SDL_VIDEODRIVER=dummy .venv/bin/python -m pytest -q` (57 tests, ~3s). One runnable `assert`
per non-trivial mechanic (torus nearest-image, raycast, swept/tunneling collision,
self-collision reachability + the straight-motion regression, PBRS telescoping + set-change
zeroing, stamina gate, chicken flee/spawn/obstacle-avoidance, interpolation, `check_env`). Add
a test with any new mechanic; update the vision/geometry tests if you change `head_radius`,
`ray_range`, or the neck-skip. No commit attribution lines (per repo rule).
