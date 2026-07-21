# Snake-RL

A 2D continuous-torus **multi-snake ecosystem**, PPO-trained (self-play, one shared brain).
2–4 snakes spawn per world (population flux up to 6 via reproduction), all driven by the
*same* policy — there's no genetics, just births and deaths. Snakes hunt fleeing/pecking
chickens by **sight** (9 vision rays) and **smell**, dodge rocks/trees/each other/their own
body, sprint with a stamina-limited **dash**, kill rivals by **cut-off** (leaving a corpse
to scavenge), and **reproduce**: two well-fed snakes that hold mating distance for a few
steps lay an egg that hatches into a fresh hatchling. Worlds are randomly generated every
episode, so the policy learns to survive anywhere — not to memorize one map.

## Run it (one command, no Python setup)

The `./snake` launcher creates the venv and installs everything on first run, then
just runs — you never touch pip or activate anything.

```bash
./snake watch          # watch the ecosystem (a trained model is included)
./snake train          # train from the last checkpoint (or from scratch: ./snake train --reset)
```

(First `./snake` call does a one-time environment setup. Prefer `python3.13`.)

## Watch (pygame)

```bash
./snake watch                       # fullscreen; the map fits your screen
./snake watch --windowed            # run in a window instead
# keys:  SPACE pause · N new world · S sensors · ↑/↓ (or +/-) sim speed · ESC quit

# no window? headless eval prints ecosystem stats instead:
./snake watch --headless --episodes 10
```

The viewer is a **persistent world**: it never resets on a death. Every snake in it —
including the legacy slot-0 "ego" left over from the single-snake days — is driven by
the same self-play controller, so there's no separate "deterministic" mode to toggle; it's
one continuous, stochastic screensaver. A **reseed floor** keeps it alive forever: if the
live population ever drops below the sustain floor, a fresh snake spawns to bring it back
up, so a bad run of deaths can't empty the world for good.

The snake is **size-agnostic** — it senses only egocentrically (rays + smell), never the
map size, and trained on random sizes — so the world is generated to match your screen's
aspect and it plays any size just fine.

Renders at 60 FPS with the whole scene interpolated between sim steps. Anti-aliased via
supersampling: glossy tapering snakes with eyes, sprite-sheet chickens that animate
peck/walk/run by their own behavior state, shaded rocks, swaying trees, eggs, corpses,
blood/gore bursts on a kill or catch, an egg-hatch effect, soft shadows, a vignette, and
vision rays tinted by what they hit (red = obstacle, yellow = chicken, purple = own body,
other tints for rivals/eggs/corpses).

## Train (headless, fast — loads the M1 fully)

```bash
./snake train --steps 8000000 --envs 16    # set --envs to your performance-core count
# resume from the last checkpoint: same command
# start over from scratch:          add --reset
```

Training runs on CPU with `SubprocVecEnv` (many random worlds in parallel) — for a
tiny MLP this beats GPU/MPS. `--envs` scales with your CPU: ~one env per performance
core saturates it (on an M1 Pro's 8 performance cores, `--envs 16` is the sweet spot;
more just oversubscribes). The raycast is vectorized, so each env step is cheap. Only
snake 0 in each parallel world is the actual PPO learner ("ego"); every other snake in
that world is driven in-env by a synced snapshot of the ego's own policy (self-play), so
training a single network produces an entire ecosystem of snakes that "think" the same
way. It prints running stats (catches, reproductions, hatches, deaths) and saves
`models/snake.zip` + `models/vecnormalize.pkl` periodically, so you can stop any time
(Ctrl-C) and `./snake watch` the current checkpoint.

### How the "deliberate dash" is learned (automatic curriculum)

The dash is rationed **mechanically, not by a reward penalty**: it needs a full stamina
reserve to fire and the reserve refills slowly, so a snake must *earn* a dash by walking
and then spend it in a burst — a stalk-and-pounce rhythm emerges. But a fresh snake has to
learn *to hunt* (and *to mate*) before it can learn thrift or patience: dropping the hard
stamina/mating constraints on it from step 0 traps it in "never dash, just survive". So a
single training run **anneals** both curricula together — easy always-on dash and a loose,
fast mating gate for the first ~42% of training, then both linearly ramp to their real,
tight values. No manual phases; just:

```bash
./snake train --steps 8000000 --envs 16 --reset
```

(This mirrors reward-shaping best practice: warm up on the easy task, then anneal in the
constraint — an abrupt switch collapses the learned behavior.)

## What to expect

Fast on an M1 (CPU, vectorized raycast). A full `8M`-step run (~40–75 min, slower than a
single-snake run since more entities step each frame) produces a policy that, measured over
a persistent multi-snake ecosystem (`./snake watch --headless --episodes 20`):

- **hunts hard** — ~10–14 chickens / 1000 snake-steps (per snake, not population-summed),
  `ep_rew_mean` ≈ **+127**, stays well-fed;
- **dashes deliberately** — a sprint only ~25–36% of live snake-steps, in bursts to run a
  chicken down or close a cut-off; stamina visibly builds while walking and drains in the
  pounce;
- **fights and reproduces** — a real (not accidental) number of `snake`-cause deaths (cut-off
  predation between rivals), balanced by organic births from egg-laying/hatching, so the
  population sustains itself around 2–4 snakes without ever dropping below the reseed floor;
- **rarely clips terrain** — it perceives its own body width (vision is inflated by the head
  radius), so obstacle deaths are the majority cause but still rare per snake-step;
- **never eats its own tail** — frame-stacked memory tracks where the body went (`self`
  deaths ≈ 0).

Random worlds every episode mean it generalizes rather than memorizing a map. Tune any
constant in `snake_rl/config.py` (the invariants there fail fast if a change breaks a
feasibility guarantee, e.g. "a dash always catches a fleeing chicken").

## How it works

- **Senses (87 floats, egocentric, frame-stacked ×4 = 348):**
  - 9 vision rays × 7 channels `[dist, is_obstacle, is_chicken, is_self, is_other_body,
    is_egg, is_corpse]` — every target is inflated by the head radius so each ray reports
    *distance until the head edge would touch*, giving the snake awareness of its own width;
  - social, 7 floats: the nearest rival snake's relative position, heading, size ratio, and
    whether it's dashing (`has_rival` disambiguates "no rival" from "rival at range 0");
  - egg, 4 floats: the nearest egg's relative position and whether it's this snake's own
    (guarding vs. raiding can only diverge once the policy can tell);
  - smell, 9 floats: three omnidirectional scent fields (chicken / rival snake / corpse),
    each `[intensity, gradient-forward, gradient-left]`;
  - proprioception, 4 floats: `[energy, length, stamina, repro_ready]`.
  - Temporal memory via frame-stacking (no rear vision, so the snake must *remember* where
    its own body went).
- **Actions:** `MultiDiscrete([3, 2])` — steer `{left, straight, right}` × dash `{no, yes}`.
- **Reward:** `+reward_eat` per item eaten (chicken, corpse, or a raided foreign egg),
  `+reward_repro` only when an egg the snake co-owns actually hatches (not on laying — a
  raided or population-capped egg pays nothing, so guarding matters), potential-based
  shaping toward the nearest chicken (provably un-farmable), `reward_death` on any death
  cause, a small hunger-scaled step cost.
- **World mechanics, not reward hacks.** Reproduction (mating distance + energy + length
  gates → egg → hatch), corpses (any death drops scavengeable food), and starvation (energy
  hitting 0 kills) are physics of the world, not bonuses wired into the reward — cooperation
  and rivalry both emerge from what's achievable, not from a "be nice"/"be mean" signal.
- **Balance is guaranteed by construction:** a fleeing chicken is faster than a walking
  snake, but a full-stamina dash always closes the gap; two snakes can always reach mating
  distance without a forced collision — asserted as invariants in `config.py`.

All tunable constants live in `snake_rl/config.py`. See `CLAUDE.md` for the full design
rationale, the hard-won pitfalls, and the retrain recipe.

## Tests

```bash
SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest -q
```
