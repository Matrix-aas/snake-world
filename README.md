# Snake-RL

A 2D continuous-torus world where an RL-trained (PPO) snake hunts fleeing chickens
by **sight** (9 vision rays) and **smell** (a `1/(1+r)` scent field blocked by obstacles),
dodges rocks/trees and its own body, and catches runaway chickens with a
stamina-limited **dash**. Worlds are randomly generated every episode, so the snake
learns to survive anywhere — not to memorize one map.

## Setup

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Train (headless, fast — loads the M1 fully)

```bash
python -m snake_rl train --steps 2000000 --envs 8
# resume from the last checkpoint: same command
# start over from scratch:          add --reset
```

Training runs on CPU with `SubprocVecEnv` (many random worlds in parallel) — for a
tiny MLP this beats GPU/MPS. It prints running stats (chickens eaten, deaths) and
saves `models/snake.zip` + `models/vecnormalize.pkl`.

## Watch (pygame)

```bash
python -m snake_rl watch
# keys:  SPACE pause · N new world · S toggle sensors · D toggle deterministic · ESC quit

# no window? headless eval prints per-episode stats instead:
python -m snake_rl watch --headless --episodes 10
```

The viewer runs the current checkpoint in a fresh random world and draws the vision
rays and a smell readout. Play is stochastic by default (looks more alive and, on a
partly-trained policy, survives a touch longer than greedy argmax — press `D` to
compare). Stop training any time and watch how far the snake has come.

## What to expect

Learning is fast on an M1 (CPU, ~4–5k steps/s with 8 envs):

- **~0–300k steps:** learns not to die — episode length climbs from ~3 to ~200 steps
  as it dodges obstacles and its own body.
- **~300k–1M steps:** starts hunting — `ep_rew_mean` crosses from −10 into positive
  territory (~+20) as it follows smell/sight to chickens and dashes down the runners.
- **~3M steps (~15 min):** `ep_rew_mean` ≈ +50, ~4–5 chickens per episode, best runs
  700–900 steps. It explores, tracks scent around rocks, and saves stamina for the chase.

Random worlds every episode mean it generalizes rather than memorizing a map. Tune any
constant in `snake_rl/config.py` (the invariants there fail fast if a change breaks the
"a dash always catches a fleeing chicken" guarantee).

## How it works

- **Senses (≈34 floats, egocentric):** 9 vision rays `[distance, is-obstacle, is-chicken, is-self]`;
  smell `[intensity, gradient-forward, gradient-left]`; proprioception `[energy, length, stamina]`.
  Temporal memory via frame-stacking (the snake has no rear vision, so it must *remember* where its body went).
- **Actions:** `MultiDiscrete([3, 2])` — steer `{left, straight, right}` × dash `{no, yes}`.
- **Reward:** `+10` per chicken, potential-based shaping toward the nearest scent
  (provably un-farmable), `-10` on death, a small hunger-scaled step cost. Hunger
  motivates but never kills.
- **Balance is guaranteed by construction:** a fleeing chicken is faster than a walking
  snake, but a full-stamina dash always closes the gap — asserted as an invariant in
  `config.py` (dash > flee, stamina budget ≥ flee radius, aim precision, curl reachability).

All tunable constants live in `snake_rl/config.py`.

## Tests

```bash
SDL_VIDEODRIVER=dummy pytest -q
```
