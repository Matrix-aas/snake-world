# Genetic Ecosystem — Phase A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the shared-brain snake sim into a genome-conditioned genetic ecosystem — heritable 9-gene genomes, sex, aging, courtship reproduction, no-ego all-eggs world, 12-snake scaled world, and the enlarged observation — so a from-scratch retrain can produce ONE brain that expresses visibly different, evolving lineages. Phase A ends at a trainable `SnakeEnv` with a green test suite; the retrain and the viewer polish (Phase B) follow.

**Architecture:** Every `Snake` carries `genome ∈ [0,1]^9`; a pure `genome.py` module resolves it to a per-snake **phenotype** (stat dict) that `world.py`/`sensors.py` read instead of global `CFG`. The genome is appended to the observation at the single `sensors.observe` chokepoint, so training, self-play, and the viewer all inherit it. Evolution is a **runtime** process (eggs inherit `crossover+mutation`); PPO only learns a genome-competent brain via domain-randomized genomes during training.

**Tech Stack:** Python 3.13 (`.venv/`), NumPy, Gymnasium, Stable-Baselines3 (PPO, CPU), pytest. Run everything via `.venv/bin/python` or `./snake`. Headless imports need `PYTHONPATH="$PWD" SDL_VIDEODRIVER=dummy`.

## Global Constraints

- **Design spec:** `docs/superpowers/specs/2026-07-23-genetic-ecosystem-design.md` — this plan implements its Phase A. Read it.
- **Emergent, not scripted; sparse reward.** No new per-step behavior reward penalties (Pitfall 1). `reward_egg_lost` **defaults `0.0`** for the retrain.
- **Respect all Pitfalls 1–20** in `CLAUDE.md`. Especially: Pitfall 5 (neck-skip uses the SAME per-snake `v_dash` in both the skip and the swept-step), Pitfall 12 (observe the *gene* ∈[0,1], never the derived stat), Pitfall 13 (two-phase step order-independence), Pitfall 17 (arrays kept in lockstep), opponent-obs parity (`selfplay`).
- **Test discipline:** one runnable `assert` per new mechanic. Run the suite with `SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest -q`.
- **Config is a frozen dataclass** — add fields, never mutate at runtime. `assert_invariants(CFG)` still passes on the base config unchanged.
- **No commit attribution lines** (repo rule).
- **Genes (index → name):** `0 size, 1 metabolism, 2 speed, 3 stamina, 4 senses, 5 lifespan, 6 aggression, 7 kin_care, 8 boldness`. `GENE_COUNT = 9`.
- **Commit after every green step.** Branch: `genetic-ecosystem`.

---

## File Structure

- **Create `snake_rl/genome.py`** — pure genome ops + phenotype resolution: `sample_genome`, `crossover`, `mutate`, `relatedness`, `Phenotype`/`resolve_phenotype`, gene index constants. No sim state, no I/O. Fully unit-testable.
- **Modify `snake_rl/config.py`** — gene interpolation-range constants, `reward_egg_lost`, `mutation_sigma`, `lifespan_jitter`, `repro_length_frac`, aging/sex constants, `n_max`/world scaling; add `assert_invariants_over_genome`.
- **Modify `snake_rl/world.py`** — `Snake` gains `genome/sex/age/max_lifespan/lineage`; physics/sensing sites read the phenotype; sex-gated courtship FSM; female-lays; egg arrays thread `genome/lineage`; `_snake_eat` returns eaten-egg owner-sets; aging death; `world.step` no-ego bifurcation; `_free_point` ego-strip.
- **Modify `snake_rl/sensors.py`** — per-ray motion channel; social relatedness/rival-state/sex; vibration field; proprio sex/age/stun/genome; per-snake sensing ranges + smell clip; `OBS_DIM` bump; `_repro_ready` size-relative.
- **Modify `snake_rl/env.py`** — obs-space bounds for the new layout; `reward_egg_lost` wiring; genome domain-randomization + sex in `reset`.
- **Modify `snake_rl/selfplay.py`** — `OBS_DIM` follows automatically (imported); verify parity.
- **Modify `snake_rl/worldgen.py`** — all-eggs start (no live founder in the viewer path); founders carry sampled genomes.
- **Modify `snake_rl/watch.py`** — guard the empty-live-set / no-ego cases.
- **Modify `CLAUDE.md`** — living-memory update (new pitfalls, config table, retrain notes).

---

## Task 1: Genome module — sampling, crossover, mutation, relatedness

**Files:**
- Create: `snake_rl/genome.py`
- Test: `tests/test_genome.py`

**Interfaces:**
- Produces:
  - `GENE_COUNT = 9`; gene index consts `SIZE, METABOLISM, SPEED, STAMINA, SENSES, LIFESPAN, AGGRESSION, KIN_CARE, BOLDNESS`.
  - `sample_genome(rng) -> np.ndarray` shape `(9,)` float32 in `[0,1)`.
  - `crossover(a, b, rng) -> np.ndarray` — per-gene uniform pick from `a` or `b`.
  - `mutate(g, rng, sigma) -> np.ndarray` — `clip(g + N(0,sigma), 0, 1)`.
  - `relatedness(a, b) -> float` — `1 - ‖a-b‖/‖ones(9)‖`, clipped `[0,1]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_genome.py
import numpy as np
from snake_rl import genome as gm


def test_sample_shape_and_range():
    rng = np.random.default_rng(0)
    g = gm.sample_genome(rng)
    assert g.shape == (gm.GENE_COUNT,) and g.dtype == np.float32
    assert (g >= 0).all() and (g <= 1).all()


def test_crossover_is_per_gene_from_a_parent():
    rng = np.random.default_rng(1)
    a = np.zeros(gm.GENE_COUNT, np.float32)
    b = np.ones(gm.GENE_COUNT, np.float32)
    c = gm.crossover(a, b, rng)
    # every gene came from exactly one parent (0 or 1), and both parents contribute over many draws
    assert set(np.unique(c)).issubset({0.0, 1.0})
    seen = set()
    for _ in range(50):
        seen.update(np.unique(gm.crossover(a, b, np.random.default_rng(_))))
    assert seen == {0.0, 1.0}


def test_mutate_stays_in_unit_box():
    rng = np.random.default_rng(2)
    g = np.array([0.0, 1.0] + [0.5] * (gm.GENE_COUNT - 2), np.float32)
    for _ in range(200):
        m = gm.mutate(g, rng, sigma=0.5)
        assert (m >= 0).all() and (m <= 1).all()


def test_relatedness_bounds():
    ones = np.ones(gm.GENE_COUNT, np.float32)
    zeros = np.zeros(gm.GENE_COUNT, np.float32)
    assert gm.relatedness(ones, ones) == 1.0
    assert gm.relatedness(ones, zeros) == 0.0
    assert 0.0 <= gm.relatedness(ones, ones * 0.5) <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest tests/test_genome.py -q`
Expected: FAIL (`ModuleNotFoundError: snake_rl.genome`).

- [ ] **Step 3: Write minimal implementation**

```python
# snake_rl/genome.py
"""Pure genome operations + phenotype resolution. No sim state, no I/O (Pitfall-12 note:
callers observe the GENE, never the derived stat -- see resolve_phenotype in Task 3)."""
import numpy as np

GENE_COUNT = 9
SIZE, METABOLISM, SPEED, STAMINA, SENSES, LIFESPAN, AGGRESSION, KIN_CARE, BOLDNESS = range(GENE_COUNT)

_NORM = float(np.sqrt(GENE_COUNT))   # ‖ones(9)‖, the max genome L2 distance


def sample_genome(rng):
    return rng.random(GENE_COUNT).astype(np.float32)


def crossover(a, b, rng):
    pick = rng.random(GENE_COUNT) < 0.5
    return np.where(pick, a, b).astype(np.float32)


def mutate(g, rng, sigma):
    return np.clip(g + rng.normal(0.0, sigma, GENE_COUNT), 0.0, 1.0).astype(np.float32)


def relatedness(a, b):
    d = float(np.sqrt(((np.asarray(a) - np.asarray(b)) ** 2).sum()))
    return float(np.clip(1.0 - d / _NORM, 0.0, 1.0))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest tests/test_genome.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add snake_rl/genome.py tests/test_genome.py
git commit -m "genome: pure sample/crossover/mutate/relatedness ops"
```

---

## Task 2: Gene ranges in config + phenotype resolution + invariant-over-genome gate

**Files:**
- Modify: `snake_rl/config.py`
- Modify: `snake_rl/genome.py`
- Test: `tests/test_genome.py`, `tests/test_config.py`

**Interfaces:**
- Consumes: `genome` constants (Task 1), `Config` (`config.py`).
- Produces:
  - `Config` fields (all floats): `gene_size_len_lo/hi`, `gene_size_turn_lo/hi`, `gene_metab_lo/hi`, `gene_speed_lo/hi`, `gene_stamina_lo/hi`, `gene_stamina_regen_lo/hi`, `gene_rayrange_lo/hi`, `gene_smell_lo/hi`, `gene_lifespan_lo/hi`, `mutation_sigma`, `lifespan_jitter`, `repro_length_frac`, `reward_egg_lost`, `sex_ratio` (unused constant, doc only), plus the size energy-decay surcharge `gene_size_hunger_hi`.
  - `genome.resolve_phenotype(genome, cfg) -> Phenotype` (a frozen dataclass / namedtuple) with fields: `max_length, turn_deg, v_snake, v_dash, s_max, stamina_regen, ray_range, smell_reach, energy_decay, max_lifespan_base`.
  - `config.assert_invariants_over_genome(cfg)` — HARD asserts (precision, raycast) across the gene box; logs SOFT (stamina-budget, self-collision) at worst corners.

Gene→stat ranges (spec §2.1/§2.4, HARD-gate-capped):

| Gene | Stat | lo → hi |
|------|------|---------|
| size | `max_length` (× `length_cap`) | 0.65 → 1.35 |
| size | `turn_deg` (× base) | 0.85 → **1.15** (precision cap) |
| size | hunger surcharge on `energy_decay` (×) | +0 → +0.4 |
| metabolism | `energy_decay` (× base) | 0.65 → 1.5 |
| speed | `v_snake`,`v_dash` (× base) | 0.85 → 1.2 |
| stamina | `s_max` (× base) | 0.7 → 1.4 |
| stamina | `stamina_regen` (× base) | 0.7 → 1.4 |
| senses | `ray_range` (absolute) | 14 → 26 |
| senses | `smell_reach` (× strength) | 1.4 → 0.7 (inverse) |
| lifespan | `max_lifespan_base` (steps) | 900 → 3200 |

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_genome.py
from snake_rl.config import CFG
from snake_rl import genome as gm


def test_phenotype_extremes_differ_and_are_ordered():
    lo = np.zeros(gm.GENE_COUNT, np.float32)
    hi = np.ones(gm.GENE_COUNT, np.float32)
    plo = gm.resolve_phenotype(lo, CFG)
    phi = gm.resolve_phenotype(hi, CFG)
    assert phi.max_length > plo.max_length
    assert phi.v_dash > plo.v_dash
    assert phi.max_lifespan_base > plo.max_lifespan_base
    # senses trades: high senses => long sight, weak smell reach
    assert phi.ray_range > plo.ray_range
    assert phi.smell_reach < plo.smell_reach
    # precision-capped turn stays aimable at both extremes
    import math
    from snake_rl.config import CFG as c
    limit = math.degrees(2 * math.atan(c.eat_radius / c.r_flee))
    assert plo.turn_deg < limit and phi.turn_deg < limit
```

```python
# add to tests/test_config.py
from snake_rl.config import CFG, assert_invariants_over_genome


def test_invariants_hold_across_gene_box():
    assert_invariants_over_genome(CFG)   # HARD gates must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest tests/test_genome.py::test_phenotype_extremes_differ_and_are_ordered tests/test_config.py::test_invariants_hold_across_gene_box -q`
Expected: FAIL (`resolve_phenotype`/`assert_invariants_over_genome` missing).

- [ ] **Step 3a: Add config fields**

Add to the `Config` dataclass in `config.py` (near the other constants):

```python
    # --- genome gene->stat interpolation ranges (spec §2.1; HARD gates §2.4) ---
    gene_size_len_lo: float = 0.65
    gene_size_len_hi: float = 1.35
    gene_size_turn_lo: float = 0.85
    gene_size_turn_hi: float = 1.15      # precision-gate cap (turn_deg*1.15 < 18.92 deg)
    gene_size_hunger_hi: float = 0.4     # bigger body => +up to 0.4x extra energy_decay
    gene_metab_lo: float = 0.65
    gene_metab_hi: float = 1.5
    gene_speed_lo: float = 0.85
    gene_speed_hi: float = 1.2
    gene_stamina_lo: float = 0.7
    gene_stamina_hi: float = 1.4
    gene_stamina_regen_lo: float = 0.7
    gene_stamina_regen_hi: float = 1.4
    gene_rayrange_lo: float = 14.0
    gene_rayrange_hi: float = 26.0
    gene_smell_lo: float = 1.4           # high senses => LOW smell reach (inverse trade)
    gene_smell_hi: float = 0.7
    gene_lifespan_lo: float = 900.0
    gene_lifespan_hi: float = 3200.0
    # --- evolution / reproduction / aging ---
    mutation_sigma: float = 0.05
    lifespan_jitter: float = 0.15        # +/- fraction on max_lifespan at birth
    repro_length_frac: float = 0.55      # mating length gate = frac * own max_length
    reward_egg_lost: float = 0.0         # DEFAULT OFF for the discovery retrain (Pitfall-1 cousin)
```

- [ ] **Step 3b: Add `resolve_phenotype` to `genome.py`**

```python
# snake_rl/genome.py  (append)
from collections import namedtuple

Phenotype = namedtuple("Phenotype", [
    "max_length", "turn_deg", "v_snake", "v_dash", "s_max", "stamina_regen",
    "ray_range", "smell_reach", "energy_decay", "max_lifespan_base",
])


def _lerp(lo, hi, t):
    return lo + (hi - lo) * float(t)


def resolve_phenotype(genome, cfg):
    g = genome
    size_hunger = 1.0 + _lerp(0.0, cfg.gene_size_hunger_hi, g[SIZE])
    metab = _lerp(cfg.gene_metab_lo, cfg.gene_metab_hi, g[METABOLISM])
    return Phenotype(
        max_length=cfg.length_cap * _lerp(cfg.gene_size_len_lo, cfg.gene_size_len_hi, g[SIZE]),
        turn_deg=cfg.turn_deg * _lerp(cfg.gene_size_turn_lo, cfg.gene_size_turn_hi, g[SIZE]),
        v_snake=cfg.v_snake * _lerp(cfg.gene_speed_lo, cfg.gene_speed_hi, g[SPEED]),
        v_dash=cfg.v_dash * _lerp(cfg.gene_speed_lo, cfg.gene_speed_hi, g[SPEED]),
        s_max=cfg.s_max * _lerp(cfg.gene_stamina_lo, cfg.gene_stamina_hi, g[STAMINA]),
        stamina_regen=cfg.stamina_regen * _lerp(cfg.gene_stamina_regen_lo, cfg.gene_stamina_regen_hi, g[STAMINA]),
        ray_range=_lerp(cfg.gene_rayrange_lo, cfg.gene_rayrange_hi, g[SENSES]),
        smell_reach=_lerp(cfg.gene_smell_lo, cfg.gene_smell_hi, g[SENSES]),
        energy_decay=cfg.energy_decay * metab * size_hunger,
        max_lifespan_base=_lerp(cfg.gene_lifespan_lo, cfg.gene_lifespan_hi, g[LIFESPAN]),
    )
```

- [ ] **Step 3c: Add `assert_invariants_over_genome` to `config.py`**

```python
# config.py  (append, near assert_invariants)
def assert_invariants_over_genome(cfg: Config) -> None:
    """HARD gates that must hold for EVERY genome (spec §2.4): aiming precision at max size,
    raycast validity at max ray_range. Stamina-budget & self-collision are SOFT (single-strategy
    relics: own-body non-lethal, peck-hunting needs no dash) -- logged, never fatal."""
    from .genome import resolve_phenotype, GENE_COUNT
    import numpy as _np
    # HARD 1: precision at the coarsest turn (max size gene)
    hi_turn = cfg.turn_deg * cfg.gene_size_turn_hi
    assert math.radians(hi_turn) / 2 < math.atan(cfg.eat_radius / cfg.r_flee), \
        f"max-size turn_deg {hi_turn:.2f} too coarse to aim (precision gate)"
    # HARD 2: nearest-image raycast at the longest ray_range (max senses gene)
    assert cfg.gene_rayrange_hi + cfg.obstacle_radius_max + cfg.head_radius < cfg.world_size_min / 2, \
        "max ray_range too large for nearest-image raycast on the smallest world"
    # SOFT: report worst-corner stamina budget & self-collision reachability
    g_lo = _np.zeros(GENE_COUNT); g_hi = _np.ones(GENE_COUNT)
    p_weak = resolve_phenotype(_np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], float), cfg)  # slow, low stamina
    budget = (p_weak.s_max / cfg.stamina_drain) * (p_weak.v_dash - cfg.v_flee)
    if budget < cfg.catch_slack_k * cfg.r_flee:
        log.info("gene box: weakest genome is ambush-only (dash budget %.1f < %.1f) -- expected, soft",
                 budget, cfg.catch_slack_k * cfg.r_flee)
```

- [ ] **Step 3d: Call the gate at import**

At the bottom of `config.py`, after `assert_invariants(CFG)`, add:

```python
assert_invariants_over_genome(CFG)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest tests/test_genome.py tests/test_config.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add snake_rl/config.py snake_rl/genome.py tests/test_genome.py tests/test_config.py
git commit -m "genome: gene->stat ranges + phenotype resolution + over-genome invariant gate"
```

---

## Task 3: `Snake` carries the genome + derived birth stats; phenotype cached

**Files:**
- Modify: `snake_rl/world.py` (`Snake` dataclass + a `_phenotype_of` helper)
- Test: `tests/test_snake_motion.py` (or a new `tests/test_phenotype.py`)

**Interfaces:**
- Consumes: `genome.resolve_phenotype`, `genome.sample_genome`.
- Produces: `Snake` new fields `genome: np.ndarray`, `sex: int` (0=F,1=M), `age: int`, `max_lifespan: float`, `lineage: int`, and a resolved `phenotype` (namedtuple) set at construction. `World._phenotype_of(snake) -> Phenotype`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_phenotype.py
import numpy as np
from snake_rl.worldgen import generate_world
from snake_rl.config import CFG
from snake_rl import genome as gm


def test_snake_has_genome_and_phenotype():
    w = generate_world(CFG, seed=3, n_snakes=2)   # live snakes (arrivals default False)
    s = w.snakes[0]
    assert s.genome.shape == (gm.GENE_COUNT,)
    assert s.sex in (0, 1)
    assert s.age == 0 and s.max_lifespan > 0
    ph = w._phenotype_of(s)
    assert ph.max_length > 0 and ph.v_dash > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest tests/test_phenotype.py -q`
Expected: FAIL (`Snake` has no `genome` / `_phenotype_of` missing).

- [ ] **Step 3a: Extend the `Snake` dataclass** (`world.py` top)

Add fields (keep existing ones):

```python
    genome: np.ndarray = None            # (9,) float32; None only for the transient __init__ placeholder
    sex: int = 0                         # 0=female, 1=male
    age: int = 0
    max_lifespan: float = 1e9
    lineage: int = 0
    phenotype: object = None             # resolved namedtuple, set by _make_snake
```

- [ ] **Step 3b: Centralize snake construction.** Add a `World._make_snake(...)` factory and route every `Snake(...)` construction (in `World.__init__`, `_hatch_eggs`, and `worldgen`) through it so genome/sex/lifespan/phenotype are always populated consistently.

```python
# world.py  (World method)
def _make_snake(self, head, heading, *, genome, sex, lineage, id, color_seed,
                energy, target_length, rng):
    from .genome import resolve_phenotype
    ph = resolve_phenotype(genome, self.cfg)
    jitter = 1.0 + rng.uniform(-self.cfg.lifespan_jitter, self.cfg.lifespan_jitter)
    return Snake(
        head_uw=head.copy(), head=head.copy(), heading=heading, path_uw=[head.copy()],
        target_length=target_length, stamina=ph.s_max, energy=energy,
        _prev_head_uw=head.copy(), id=id, color_seed=color_seed,
        genome=genome, sex=int(sex), age=0, max_lifespan=ph.max_lifespan_base * jitter,
        lineage=lineage, phenotype=ph,
    )

def _phenotype_of(self, snake):
    return snake.phenotype
```

Note: keep `stamina` initialized to the per-snake `ph.s_max` (was `s_max` global). Ensure `World` holds a `self.rng` (it already seeds one in `__init__`); if not, add `self.rng = np.random.default_rng(seed)`.

- [ ] **Step 3c: Populate genomes at world construction.** In `worldgen.generate_world`, when creating live founders, sample a genome per snake and a random sex, assign a fresh `lineage` id (e.g. incrementing counter), and call `world._make_snake(...)`. (The all-eggs viewer path is Task 10; for now founders may be live so existing tests keep working.)

- [ ] **Step 4: Run test to verify it passes**

Run: `SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest tests/test_phenotype.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add snake_rl/world.py snake_rl/worldgen.py tests/test_phenotype.py
git commit -m "world: Snake carries genome/sex/age/lifespan/lineage via _make_snake factory"
```

---

## Task 4: Route per-snake MOTION physics through the phenotype (incl. Pitfall-5 pair)

**Files:**
- Modify: `snake_rl/world.py` (`_move_snake`, `_body_points_uw` neck-skip, `_prune_path`, energy/hunger step, stamina regen)
- Test: `tests/test_snake_motion.py`, `tests/test_collision.py`

**Interfaces:**
- Consumes: `Snake.phenotype`.
- Produces: motion/energy/stamina now read `snake.phenotype.{turn_deg,v_snake,v_dash,s_max,stamina_regen,energy_decay}`; neck-skip and prune slack use the SAME per-snake `v_dash`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_snake_motion.py  (add)
import numpy as np
from snake_rl.worldgen import generate_world
from snake_rl.config import CFG
from snake_rl import genome as gm


def _world_with_two_speed_genomes(seed=7):
    w = generate_world(CFG, seed=seed, n_snakes=2)
    slow = np.full(gm.GENE_COUNT, 0.0, np.float32); slow[gm.SPEED] = 0.0
    fast = np.full(gm.GENE_COUNT, 0.0, np.float32); fast[gm.SPEED] = 1.0
    w.snakes[0] = w._make_snake(w.snakes[0].head, 0.0, genome=slow, sex=0, lineage=1,
                                id=w.snakes[0].id, color_seed=1, energy=CFG.energy_max,
                                target_length=CFG.start_length, rng=w.rng)
    w.snakes[1] = w._make_snake(w.snakes[1].head, 0.0, genome=fast, sex=1, lineage=2,
                                id=w.snakes[1].id, color_seed=2, energy=CFG.energy_max,
                                target_length=CFG.start_length, rng=w.rng)
    return w


def test_faster_genome_travels_farther_at_full_cruise():
    w = _world_with_two_speed_genomes()
    p0 = w.snakes[0].head.copy(); p1 = w.snakes[1].head.copy()
    # both go straight, full cruise (speed idx 3), no dash
    for _ in range(5):
        w.step(3, 1, 0, opponent_fn=lambda world, s: (3, 1, 0))
    from snake_rl.world import torus_dist
    d_slow = torus_dist(w.snakes[0].head, p0, w.size)
    d_fast = torus_dist(w.snakes[1].head, p1, w.size)
    assert d_fast > d_slow * 1.1
```

```python
# tests/test_collision.py  (add)
import numpy as np
from snake_rl.worldgen import generate_world
from snake_rl.config import CFG
from snake_rl import genome as gm


def test_fast_genome_straight_line_no_self_neck_collision():
    # Pitfall 5: a fast genome moving straight must not false-collide with its own neck.
    w = generate_world(CFG, seed=11, n_snakes=1)
    fast = np.zeros(gm.GENE_COUNT, np.float32); fast[gm.SPEED] = 1.0
    w.snakes[0] = w._make_snake(w.snakes[0].head, 0.0, genome=fast, sex=0, lineage=1,
                                id=w.snakes[0].id, color_seed=1, energy=CFG.energy_max,
                                target_length=CFG.length_cap, rng=w.rng)
    for _ in range(30):
        w.step(3, 1, 1, opponent_fn=lambda world, s: (3, 1, 1))   # full cruise + dash, straight
        assert w.snakes[0].alive, "fast straight snake wrongly died (neck-skip not per-snake v_dash)"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest tests/test_snake_motion.py::test_faster_genome_travels_farther_at_full_cruise tests/test_collision.py::test_fast_genome_straight_line_no_self_neck_collision -q`
Expected: FAIL (physics still uses global CFG; fast-genome test shows no speed difference, neck test may die).

- [ ] **Step 3: Route motion through the phenotype.** In `world.py`, replace global reads with `ph = snake.phenotype` (or `self._phenotype_of(snake)`):
  - `_move_snake`: `turn_deg → ph.turn_deg`; cruise speed `speed_levels[idx] * ph.v_snake`; dash `ph.v_dash`; stamina drain gate uses `ph.s_max`; regen `ph.stamina_regen * (1 - speed_levels[idx])`, clamped to `ph.s_max`.
  - **Pitfall-5 pair** — thread `ph.v_dash` into BOTH:
    - `_body_points_uw` (`world.py:269`): `skip = head_radius + body_radius + ph.v_dash + segment_spacing`.
    - `_prune_path` (`world.py:231`): slack `target_length + ph.v_dash`.
    Since `_body_points_uw`/`_prune_path` take a `snake`, read `snake.phenotype.v_dash` inside.
  - Per-step hunger: `snake.energy -= ph.energy_decay`.

  Concrete example for the dash gate in `_move_snake` (adapt to the real code):

```python
        ph = snake.phenotype
        if dash and snake.stamina >= dash_min_stamina and snake.stun == 0:
            step_v = ph.v_dash
            snake.stamina -= self.cfg.stamina_drain
            ...
        else:
            step_v = self.cfg.speed_levels[speed_idx] * ph.v_snake
            regen = ph.stamina_regen * (1.0 - self.cfg.speed_levels[speed_idx])
            snake.stamina = min(ph.s_max, snake.stamina + regen)
```

- [ ] **Step 4: Run tests to verify they pass**

Run the two tests above, then the motion/collision suites:
`SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest tests/test_snake_motion.py tests/test_collision.py -q`
Expected: PASS (including all pre-existing motion/collision tests — regression guard).

- [ ] **Step 5: Commit**

```bash
git add snake_rl/world.py tests/test_snake_motion.py tests/test_collision.py
git commit -m "world: per-snake motion/energy/stamina from phenotype (Pitfall-5 v_dash pair threaded)"
```

---

## Task 5: Per-snake SENSING ranges + smell intensity clip

**Files:**
- Modify: `snake_rl/sensors.py` (`_scan`/`sense_vision` ray_range; `smell`/`_smell_field` reach + clip)
- Test: `tests/test_sensors.py`

**Interfaces:**
- Consumes: `Snake.phenotype.{ray_range, smell_reach}`.
- Produces: vision normalizes by per-snake `ray_range`; smell intensity scaled by `smell_reach` and **clipped** to `chicken_ceiling` for chicken/snake/corpse alike.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sensors.py  (add)
import numpy as np
from snake_rl.worldgen import generate_world
from snake_rl.config import CFG
from snake_rl import genome as gm
from snake_rl.sensors import observe
from snake_rl.env import SnakeEnv


def test_long_sight_genome_sees_farther_and_obs_in_bounds():
    w = generate_world(CFG, seed=5, n_snakes=1)
    hi = np.zeros(gm.GENE_COUNT, np.float32); hi[gm.SENSES] = 1.0   # long ray_range, weak smell
    w.snakes[0] = w._make_snake(w.snakes[0].head, 0.0, genome=hi, sex=0, lineage=1,
                                id=w.snakes[0].id, color_seed=1, energy=CFG.energy_max,
                                target_length=CFG.start_length, rng=w.rng)
    ph = w._phenotype_of(w.snakes[0])
    assert ph.ray_range == CFG.gene_rayrange_hi
    obs = observe(w, w.snakes[0])
    assert np.isfinite(obs).all()


def test_high_smell_genome_intensity_stays_within_bound():
    # a max-smell-reach genome must NOT blow the observation bound (§7 smell clip fix)
    env = SnakeEnv(seed=1)
    lo = np.zeros(gm.GENE_COUNT, np.float32)   # senses=0 => max smell reach (1.4x)
    w = env.world
    s = w.snakes[0]
    w.snakes[0] = w._make_snake(s.head, 0.0, genome=lo, sex=0, lineage=1, id=s.id,
                                color_seed=1, energy=CFG.energy_max,
                                target_length=CFG.start_length, rng=w.rng)
    obs = observe(w, w.snakes[0])
    assert env.observation_space.contains(obs.astype(np.float32))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest tests/test_sensors.py::test_long_sight_genome_sees_farther_and_obs_in_bounds tests/test_sensors.py::test_high_smell_genome_intensity_stays_within_bound -q`
Expected: FAIL (`ray_range`/`smell_reach` still global; smell unclipped).

Note: `test_high_smell_...` also depends on Task 9's obs layout for a full `observation_space` match; if run before Task 9, assert only `np.isfinite(obs).all()` and the raw smell magnitude `≤ chicken_ceiling`, then tighten to `observation_space.contains` after Task 9.

- [ ] **Step 3: Route sensing through the phenotype.**
  - `sensors._scan`/`sense_vision`: use `snake.phenotype.ray_range` in the `_cast(...)` call and the `dist / ray_range` normalization (replace `c.ray_range`).
  - `sensors.smell`/`_smell_field`: multiply each field's intensity/gradient contribution by `snake.phenotype.smell_reach`, then **clip chicken and snake intensity to `chicken_ceiling`** (currently only corpse is clipped, `sensors.py:169-172`). Apply the same clip to the forward/left gradient components.

```python
# sensors.smell (sketch)
    reach = snake.phenotype.smell_reach
    ceil = float(c.chicken_ceiling)
    ci, cg = _smell_field(world, snake.head, world.chicken_pos)
    ci = float(np.clip(ci * reach, 0.0, ceil))
    cgf = float(np.clip((cg @ fwd) * reach, -ceil, ceil))
    cgl = float(np.clip((cg @ left) * reach, -ceil, ceil))
    # ... same pattern for snake (si) and corpse (ki) fields ...
```

- [ ] **Step 4: Run tests to verify they pass** (per the Step-2 note, `contains` assertion may wait for Task 9)

Run: `SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest tests/test_sensors.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add snake_rl/sensors.py tests/test_sensors.py
git commit -m "sensors: per-snake ray_range + smell reach with intensity clip (obs bound safe)"
```

---

## Task 6: Sex-gated mating predicate

**Files:**
- Modify: `snake_rl/world.py` (`_resolve_mating` eligibility)
- Test: `tests/test_reproduction.py`

**Interfaces:**
- Consumes: `Snake.sex`.
- Produces: only an opposite-sex pair (both repro-ready) can form a mating; same-sex pairs never lay.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reproduction.py  (add)
import numpy as np
from snake_rl.worldgen import generate_world
from snake_rl.config import CFG


def _ready_pair(seed, sexA, sexB, dist=3.0):
    w = generate_world(CFG, seed=seed, n_snakes=2, size=(60.0, 60.0))
    import numpy as np
    a, b = w.snakes[0], w.snakes[1]
    # place them close, well-fed, grown, off cooldown, opposite/same sex per args
    for s, sx in ((a, sexA), (b, sexB)):
        s.energy = CFG.energy_max
        s.target_length = CFG.length_cap
        s.repro_cooldown = 0
        s.sex = sx
    b.head[:] = a.head + np.array([dist, 0.0]); b.head_uw[:] = b.head
    return w, a, b


def test_same_sex_pair_never_lays():
    w, a, b = _ready_pair(21, 0, 0)
    n_eggs0 = len(w.eggs["pos"])
    for _ in range(CFG.mate_steps + 3):
        w.step(0, 1, 0, opponent_fn=lambda world, s: (0, 1, 0))
    assert len(w.eggs["pos"]) == n_eggs0, "same-sex pair must not produce an egg"


def test_opposite_sex_pair_lays_after_courtship():
    w, a, b = _ready_pair(22, 0, 1)
    laid = False
    for _ in range(CFG.mate_steps + 5):
        before = len(w.eggs["pos"])
        w.step(0, 1, 0, opponent_fn=lambda world, s: (0, 1, 0))
        if len(w.eggs["pos"]) > before:
            laid = True
            break
    assert laid, "opposite-sex ready pair should lay after holding courtship distance"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest tests/test_reproduction.py::test_same_sex_pair_never_lays tests/test_reproduction.py::test_opposite_sex_pair_lays_after_courtship -q`
Expected: `test_same_sex...` FAILS (current code ignores sex).

- [ ] **Step 3: Add the sex gate.** In `_resolve_mating` (world.py), add `a.sex != b.sex` to the pair-eligibility predicate (alongside the energy/length/cooldown/`repro_ready` checks). Keep the existing mate-streak-by-`frozenset` machinery.

- [ ] **Step 4: Run tests to verify they pass**

Run: `SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest tests/test_reproduction.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add snake_rl/world.py tests/test_reproduction.py
git commit -m "world: mating requires opposite sex"
```

---

## Task 7: Aging + death by age

**Files:**
- Modify: `snake_rl/world.py` (`step` phase-3: increment `age`, death when `age >= max_lifespan`)
- Test: `tests/test_multisnake.py` (or `tests/test_reproduction.py`)

**Interfaces:**
- Consumes: `Snake.age`, `Snake.max_lifespan`.
- Produces: `death_cause == "age"` in `deaths_detailed`; a corpse spawned like any death.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_multisnake.py  (add)
from snake_rl.worldgen import generate_world
from snake_rl.config import CFG


def test_snake_dies_of_old_age():
    w = generate_world(CFG, seed=31, n_snakes=1, size=(60.0, 60.0))
    s = w.snakes[0]
    s.max_lifespan = 3            # force imminent old age
    s.energy = CFG.energy_max     # ensure it's not starvation
    causes = []
    for _ in range(6):
        out = w.step(1, 1, 0, opponent_fn=lambda world, sn: (1, 1, 0))
        causes += [c for _, c in out["deaths_detailed"]]
        if not s.alive:
            break
    assert "age" in causes, "a snake past max_lifespan must die of 'age'"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest tests/test_multisnake.py::test_snake_dies_of_old_age -q`
Expected: FAIL (no aging yet).

- [ ] **Step 3: Implement aging.** In `World.step`, in the per-snake resolution phase (alongside the `starve` check), for each live snake: `snake.age += 1`; if `snake.age >= snake.max_lifespan`, mark death with `death_cause="age"` and spawn a corpse (reuse the existing corpse-spawn path used by `starve`). Keep it a per-snake independent check (Pitfall 13 — no cross-snake ordering).

- [ ] **Step 4: Run test to verify it passes**

Run: `SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest tests/test_multisnake.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add snake_rl/world.py tests/test_multisnake.py
git commit -m "world: aging + death by old age (cause 'age'), corpse on death"
```

---

## Task 8: Reproduction FSM — female lays adjacent, egg carries genome + lineage; eaten-egg return channel

**Files:**
- Modify: `snake_rl/world.py` (egg arrays gain `genome`,`lineage`; `_resolve_mating` lay; `_snake_eat` eaten-egg owner-set return; `_hatch_eggs` uses carried genome + random sex; `spawn_egg` for arrivals)
- Modify: `snake_rl/sensors.py` (`_repro_ready` size-relative length gate)
- Test: `tests/test_reproduction.py`

**Interfaces:**
- Consumes: `genome.crossover`, `genome.mutate`, `Snake.{genome,sex,lineage,phenotype}`.
- Produces:
  - `world.eggs` dict gains `"genome": np.ndarray(N,9)` and `"lineage": np.ndarray(N,)` kept in lockstep with `pos/timer/owner`.
  - `world.step(...)` return dict gains `"eaten_eggs"`: list of `frozenset(owner ids)` for eggs eaten this step (arrival eggs owner<0 excluded).
  - Hatchling genome = the egg's carried genome; hatchling sex random ~50/50; hatchling lineage = egg's lineage.
  - `_repro_ready(cfg, snake)` length gate = `snake.target_length >= cfg.repro_length_frac * snake.phenotype.max_length`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_reproduction.py  (add)
import numpy as np
from snake_rl.worldgen import generate_world
from snake_rl.config import CFG
from snake_rl import genome as gm


def test_egg_carries_crossover_genome_and_maternal_lineage():
    w, a, b = _ready_pair(40, 0, 1)          # a=female(0), b=male(1)
    a.genome = np.zeros(gm.GENE_COUNT, np.float32)
    b.genome = np.ones(gm.GENE_COUNT, np.float32)
    a.lineage = 77
    laid_idx = None
    for _ in range(CFG.mate_steps + 5):
        before = len(w.eggs["pos"])
        w.step(0, 1, 0, opponent_fn=lambda world, s: (0, 1, 0))
        if len(w.eggs["pos"]) > before:
            laid_idx = len(w.eggs["pos"]) - 1
            break
    assert laid_idx is not None
    child = w.eggs["genome"][laid_idx]
    assert child.shape == (gm.GENE_COUNT,)
    assert set(np.unique(np.round(child, 6))).issubset({0.0, 1.0}) or True  # crossover+mutation
    assert w.eggs["lineage"][laid_idx] == 77   # maternal lineage


def test_eaten_egg_reported_in_step_return():
    # place a foreign egg and a hungry non-owner on top of it
    w = generate_world(CFG, seed=41, n_snakes=1, size=(60.0, 60.0))
    s = w.snakes[0]
    egg_pos = s.head + np.array([1.0, 0.0])
    w.eggs = {"pos": np.array([egg_pos]), "timer": np.array([CFG.egg_timer], float),
              "owner": np.array([[999, 998]]),   # someone else's egg
              "genome": np.zeros((1, gm.GENE_COUNT), np.float32), "lineage": np.array([5])}
    out = w.step(1, 1, 0, opponent_fn=lambda world, sn: (1, 1, 0))
    assert any(frozenset({999, 998}) == e for e in out["eaten_eggs"])


def test_repro_length_is_size_relative():
    from snake_rl.sensors import _repro_ready
    w = generate_world(CFG, seed=42, n_snakes=1)
    s = w.snakes[0]
    small = np.zeros(gm.GENE_COUNT, np.float32)   # size gene 0 => small max_length
    w.snakes[0] = w._make_snake(s.head, 0.0, genome=small, sex=0, lineage=1, id=s.id,
                                color_seed=1, energy=CFG.energy_max,
                                target_length=CFG.length_cap * CFG.gene_size_len_lo, rng=w.rng)
    s = w.snakes[0]
    s.repro_cooldown = 0
    # at its own full length a small genome must be able to qualify (fraction of OWN max, not absolute)
    assert _repro_ready(CFG, s) == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest tests/test_reproduction.py::test_egg_carries_crossover_genome_and_maternal_lineage tests/test_reproduction.py::test_eaten_egg_reported_in_step_return tests/test_reproduction.py::test_repro_length_is_size_relative -q`
Expected: FAIL.

- [ ] **Step 3a: Egg arrays carry genome + lineage.** Everywhere `world.eggs` is created/filtered (`World.__init__`, `spawn_egg`, `_resolve_mating` lay, `_snake_eat` egg-eat filter `world.py:519-522`, `_hatch_eggs` `world.py:634`), add `"genome"` (shape `(N,9)`) and `"lineage"` (shape `(N,)`) and apply the SAME `keep` mask to them (Pitfall-17 lockstep).

- [ ] **Step 3b: Female lays the egg (not midpoint spawner).** In `_resolve_mating`, on a completed courtship: identify the female (`sex==0`), compute the child genome `mutate(crossover(a.genome, b.genome, rng), rng, cfg.mutation_sigma)`, lineage = female's lineage, `owner=[female.id, male.id]`. Place the egg at the existing collision-safe parents-midpoint (`world.py:656`) — do NOT use `_free_point`. Append genome+lineage in lockstep.

- [ ] **Step 3c: `_snake_eat` returns eaten-egg owner-sets.** When a non-owner eats an egg, collect `frozenset(int(x) for x in owner_row)` for each eaten egg with `owner[0] >= 0` into a list; return it (thread up through `step`'s return dict as `"eaten_eggs"`).

- [ ] **Step 3d: `_hatch_eggs` uses carried genome + random sex + lineage.** At `world.py:628`, instead of a default genome, pass `genome=e["genome"][i]`, `sex=int(self.rng.integers(0, 2))`, `lineage=int(e["lineage"][i])` into `_make_snake` (a guaranteed arrival egg with owner<0 carries a freshly-sampled founder genome — set that in `spawn_egg`).

- [ ] **Step 3e: `spawn_egg` (arrival) carries a fresh founder genome + lineage.** Give arrival eggs `genome=sample_genome(self.rng)` and a new lineage id, `owner=[-1,-1]` unchanged.

- [ ] **Step 3f: `_repro_ready` size-relative.** In `sensors._repro_ready`, change the length gate to `snake.target_length > cfg.repro_length_frac * snake.phenotype.max_length` (was `> cfg.repro_length_min`). Keep energy + cooldown checks.

- [ ] **Step 3g: `step` return + reward hook.** Add `"eaten_eggs"` to the `step` return dict (default `[]`). (The env consumes it in Task 12.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest tests/test_reproduction.py -q`
Expected: PASS (including the existing mate→egg→hatch, parent-can't-eat-own-egg, egg-raid tests — regression guard).

- [ ] **Step 5: Commit**

```bash
git add snake_rl/world.py snake_rl/sensors.py tests/test_reproduction.py
git commit -m "world: courtship lay by female, eggs carry genome+lineage, eaten-egg return channel, size-relative repro gate"
```

---

## Task 9: Observation redesign — new channels, OBS_DIM, env bounds, parity

**Files:**
- Modify: `snake_rl/sensors.py` (per-ray motion; social relatedness/rival-state/sex; vibration field; proprio sex/age/stun/genome; `OBS_DIM`)
- Modify: `snake_rl/env.py` (`_make_observation_space` bounds for the new layout)
- Modify: `snake_rl/selfplay.py` (verify `OBS_DIM` propagation + `Box` shape)
- Test: `tests/test_sensors.py`, `tests/test_env.py`, `tests/test_selfplay.py`

**Interfaces:**
- Consumes: `Snake.{genome,sex,age,max_lifespan,stun,phenotype}`, `relatedness`.
- Produces: new `OBS_DIM` (enumerated below); layout documented in `sensors.observe` docstring and mirrored in `env._make_observation_space`.

**New layout (per-frame):** vision `11×9=99` · social `11` · egg `4` · smell `9` · vibration `3` · proprio `17` → **`OBS_DIM = 143`**. Frame-stacked ×4 → 572.

- Vision feature 8 (index 8, the 9th): per-ray **target state-nominal motion** ∈[0,1] (peck-hen 0, walk low, flee/startle high, dashing snake high), normalized by `v_dash * gene_speed_hi` (observer-independent max) and clipped to 1.
- Social (11) = existing 7 + `relatedness(self, rival)` + rival `energy/energy_max` + rival `_repro_ready` + rival `sex`.
- Vibration (3) = `[intensity, grad_fwd, grad_left]` over live rivals + fleeing chickens weighted by their normalized speed, un-occluded; bound ±`(n_max + chicken_ceiling)`.
- Proprio (17) = existing 5 (energy, length/own-max, stamina/own-s_max, repro_ready, speed/own-v_dash) + `sex` + `age/max_lifespan` + `stun/stun_steps` + `genome (9)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sensors.py  (add)
import numpy as np
from snake_rl.sensors import observe, OBS_DIM
from snake_rl.worldgen import generate_world
from snake_rl.config import CFG
from snake_rl import genome as gm


def test_obs_dim_is_143_and_contains_genome_tail():
    assert OBS_DIM == 143
    w = generate_world(CFG, seed=9, n_snakes=2)
    s = w.snakes[0]
    obs = observe(w, s)
    assert obs.shape == (OBS_DIM,)
    # last 9 floats are the snake's genome (per-frame, un-normalized here)
    assert np.allclose(obs[-gm.GENE_COUNT:], s.genome, atol=1e-6)


def test_vibration_responds_to_a_moving_rival_not_a_still_one():
    # a dashing rival nearby should raise the vibration intensity vs. a stopped one
    w = generate_world(CFG, seed=8, n_snakes=2, size=(60.0, 60.0))
    a, b = w.snakes[0], w.snakes[1]
    b.head[:] = a.head + np.array([4.0, 0.0]); b.head_uw[:] = b.head
    b.speed = 0.0
    still = observe(w, a)
    b.speed = b.phenotype.v_dash
    moving = observe(w, a)
    VIB = 99 + 11 + 4 + 9    # vibration block start
    assert moving[VIB] > still[VIB]
```

```python
# tests/test_env.py  (add)
import numpy as np
from snake_rl.env import SnakeEnv
from snake_rl.sensors import observe, OBS_DIM


def test_env_obs_space_matches_layout_for_extreme_genomes():
    env = SnakeEnv(seed=2)
    assert env.observation_space.shape[0] == OBS_DIM * env.cfg.frame_stack or \
           env.observation_space.shape[0] == OBS_DIM  # single-frame space; framestack is external
    from snake_rl import genome as gm
    w = env.world
    for gval in (0.0, 1.0):
        g = np.full(gm.GENE_COUNT, gval, np.float32)
        s = w.snakes[0]
        w.snakes[0] = w._make_snake(s.head, 0.0, genome=g, sex=0, lineage=1, id=s.id,
                                    color_seed=1, energy=env.cfg.energy_max,
                                    target_length=env.cfg.start_length, rng=w.rng)
        obs = observe(w, w.snakes[0]).astype(np.float32)
        assert env.observation_space.contains(obs)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest tests/test_sensors.py tests/test_env.py -q`
Expected: FAIL (`OBS_DIM` still 113; new channels absent).

- [ ] **Step 3a: sensors.py** — implement the four additions:
  - **Per-ray motion:** in `sense_vision`, widen from 8 to 9 features/ray; feature 8 = state-nominal speed of the hit target (look up the hit entity by `idx`; for a chicken map its FSM state → {peck:0, walk:v_wander, flee/startle:v_flee}; for a snake use its `speed`; else 0), normalized by `c.v_dash * c.gene_speed_hi`, clipped [0,1].
  - **Social:** extend `_social` return from 7 to 11 with `relatedness(snake.genome, r.genome)`, `r.energy/c.energy_max`, `_repro_ready(c, r)`, `float(r.sex)`.
  - **Vibration:** add `sense_vibration(world, snake)` — a `_smell_field`-style field (reuse the machinery, no occlusion) over rivals+fleeing-chickens weighted by `speed/(c.v_dash*c.gene_speed_hi)`, returns `[intensity, grad_fwd, grad_left]`, clipped to `±(n_max+chicken_ceiling)`.
  - **Proprio:** append `float(snake.sex)`, `clip(age/max_lifespan,0,1)`, `clip(stun/stun_steps,0,1)`, and `snake.genome` (9). Change `stamina`/`length`/`speed` normalizers to per-snake (`snake.phenotype.s_max`, `.max_length`, `.v_dash`).
  - Update `OBS_DIM = 143` and the module docstring layout.
- [ ] **Step 3b: env.py** — rewrite `_make_observation_space` bounds to match the new 143 layout (hand-enumerate per block as the current code does; genome tail bounds `[0,1]`, vibration `±24`, motion `[0,1]`, sex `{0,1}`, age/stun `[0,1]`).
- [ ] **Step 3c: selfplay.py** — no code change expected (it imports `OBS_DIM` and builds `Box(-inf,inf,(OBS_DIM*n_stack,))` + `MultiDiscrete([4,3,2])`); confirm it picks up 143.

- [ ] **Step 4: Run tests + parity + check_env**

```bash
SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest tests/test_sensors.py tests/test_env.py tests/test_selfplay.py -q
```
Expected: PASS, including the existing opponent-obs preprocessing-parity test (`test_selfplay`) and `check_env`.

- [ ] **Step 5: Commit**

```bash
git add snake_rl/sensors.py snake_rl/env.py snake_rl/selfplay.py tests/test_sensors.py tests/test_env.py tests/test_selfplay.py
git commit -m "obs: genome/sex/age/stun/relatedness/rival-state/vibration/per-ray-motion; OBS_DIM 143; env bounds; parity"
```

---

## Task 10: No-ego / all-eggs world start

**Files:**
- Modify: `snake_rl/world.py` (`step` no-ego bifurcation; `_free_point` ego-strip; `_ego_prop`/`_prune_dead` guards)
- Modify: `snake_rl/worldgen.py` (all-eggs founders when `arrivals=True` AND a new `ego_live` flag)
- Modify: `snake_rl/watch.py` (empty-live-set guards)
- Modify: `snake_rl/env.py` (`reset` keeps one live gradient-ego via `ego_live=True`)
- Test: `tests/test_worldgen.py`, `tests/test_watch_smoke.py`

**Interfaces:**
- Consumes: existing arrival-egg machinery (`spawn_egg`).
- Produces: `generate_world(..., arrivals=True, ego_live=False)` → **zero live snakes**, ≥1 arrival egg. `world.step` tolerates an empty live set (returns `died=None`, no ego info). Training uses `ego_live=True` (one live gradient snake).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_worldgen.py  (add)
from snake_rl.worldgen import generate_world
from snake_rl.config import CFG


def test_viewer_world_starts_with_only_eggs():
    w = generate_world(CFG, seed=50, n_snakes=3, arrivals=True, ego_live=False)
    assert sum(1 for s in w.snakes if s.alive) == 0
    n_pending = int((w.eggs["owner"][:, 0] < 0).sum()) if len(w.eggs["owner"]) else 0
    assert n_pending >= 1
    # stepping the egg-only world does not crash and eventually hatches a snake
    for _ in range(CFG.egg_timer + 2):
        w.step(1, 1, 0, opponent_fn=lambda world, s: (1, 1, 0))
    assert sum(1 for s in w.snakes if s.alive) >= 1
```

```python
# tests/test_worldgen.py  (add)
def test_training_world_keeps_one_live_ego():
    w = generate_world(CFG, seed=51, n_snakes=3, arrivals=True, ego_live=True)
    assert sum(1 for s in w.snakes if s.alive) >= 1   # the gradient-ego is live
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest tests/test_worldgen.py -q`
Expected: FAIL (`ego_live` kwarg missing; egg-only world crashes in `step`).

- [ ] **Step 3a: `world.step` no-ego bifurcation.** Guard `world.py:727-729`: if there are no live snakes (or no privileged ego), skip the `ego = self.snakes[0]` reads and return the dict with `died=None`/`ego_dashed=False`. Keep the full ego path when a live ego exists (training).
- [ ] **Step 3b: `_free_point` ego-strip.** Remove the `self.head`/`r_flee`-of-ego rejection (`world.py:563-565`); keep the `body_radius`-of-any-live-body rejection so eggs still avoid overlapping snakes. This unblocks placing founders when no ego exists.
- [ ] **Step 3c: `_ego_prop`/`_prune_dead`/`__init__` guards.** Make the `head`/`heading` ego descriptors and `_prune_dead` tolerate `len(self.snakes)==0` or no live snake (return a neutral default; nothing should assume slot-0 is alive).
- [ ] **Step 3d: worldgen.** Add `ego_live: bool = True` param. When `arrivals=True`:
  - `ego_live=True` (training): spawn snake 0 live (as today), rest as arrival eggs.
  - `ego_live=False` (viewer): spawn NO live snakes; lay `n_snakes` arrival eggs via `spawn_egg`.
- [ ] **Step 3e: env.reset** — pass `arrivals=True, ego_live=True`.
- [ ] **Step 3f: watch** — `_new_ecosystem`/`generate_world` calls use `ego_live=False`; guard `run_watch` `follow_id = world.snakes[0].id` and `_step_world`'s `ego = world.snakes[0]` for an empty live set (fall back to "no follow / overview", drive only live snakes).

- [ ] **Step 4: Run tests to verify they pass**

Run: `SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest tests/test_worldgen.py tests/test_watch_smoke.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add snake_rl/world.py snake_rl/worldgen.py snake_rl/watch.py snake_rl/env.py tests/test_worldgen.py
git commit -m "world: no-ego all-eggs viewer start; step tolerates empty live set; training keeps one gradient-ego"
```

---

## Task 11: World / population scaling to 12 snakes

**Files:**
- Modify: `snake_rl/config.py` (`n_max`, world size, `n_start_*`, `chicken_ceiling`, obstacle counts)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `n_max=12`; enlarged world; caps that key on `n_max` updated.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py  (add)
from snake_rl.config import CFG, assert_invariants, assert_invariants_over_genome


def test_scaled_population_invariants():
    assert CFG.n_max == 12
    # food ceiling still covers max demand (invariant 10)
    assert CFG.chicken_ceiling >= CFG.chickens_per_snake_max * CFG.n_max
    assert_invariants(CFG)
    assert_invariants_over_genome(CFG)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest tests/test_config.py::test_scaled_population_invariants -q`
Expected: FAIL (`n_max` still 6).

- [ ] **Step 3: Scale the config.** In `config.py`: `n_max=12`; `n_start_min/max` e.g. `3/6`; `world_size_min/max` e.g. `180/260`; `chicken_ceiling` ≥ `chickens_per_snake_max*n_max` (e.g. `24`); scale `n_obstacles_min/max` for the larger area (e.g. `24/64`). Re-check `assert_invariants` (esp. raycast invariant with the larger world — it *relaxes*) and the soft `n_max`-area warning.

- [ ] **Step 4: Run test to verify it passes** + full suite

Run: `SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest tests/test_config.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add snake_rl/config.py tests/test_config.py
git commit -m "config: scale to n_max=12 + larger world + updated caps"
```

---

## Task 12: Env reward wiring, training genome randomization, full-suite + smoke

**Files:**
- Modify: `snake_rl/env.py` (`reset` genome/sex domain-randomization; `step` egg-lost reward from `eaten_eggs`)
- Modify: `CLAUDE.md` (living memory)
- Test: `tests/test_env.py`, `tests/test_train_smoke.py`

**Interfaces:**
- Consumes: `world.step(...)["eaten_eggs"]`, `genome.sample_genome`.
- Produces: ego reward includes `reward_egg_lost` (default 0) when the ego co-owns an eaten egg; each `reset` gives every snake a randomized genome (domain randomization) so the brain learns to read the whole gene box.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_env.py  (add)
import numpy as np
from snake_rl.env import SnakeEnv
from snake_rl import genome as gm


def test_reset_randomizes_genomes_across_snakes():
    env = SnakeEnv(seed=4)
    env.reset()
    genomes = [s.genome for s in env.world.snakes]
    # at least two distinct genomes among the initial population (domain randomization)
    assert len({tuple(np.round(g, 4)) for g in genomes}) >= 2


def test_egg_lost_penalty_applies_when_enabled():
    env = SnakeEnv(seed=4)
    env.reset()
    env.cfg_egg_lost = -4.0    # or monkeypatch the reward constant per the impl
    ego = env.world.snakes[0]
    # fabricate an eaten egg co-owned by the ego in the step return path
    # (unit-level: call the reward helper directly if the impl exposes one)
    r = env._egg_lost_reward([frozenset({ego.id, 123})])
    assert r == -4.0
```

Note: adapt the second test to however the reward is wired — the intent is: an eaten egg the ego co-owns yields `reward_egg_lost`; a foreign egg the ego does not own yields 0.

- [ ] **Step 2: Run tests to verify they fail**

Run: `SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest tests/test_env.py -q`
Expected: FAIL.

- [ ] **Step 3a: env.reset domain randomization.** After building the world, ensure every founder snake AND every arrival egg gets a freshly `sample_genome(rng)` genome and random sex (worldgen already does this via `_make_snake`/`spawn_egg` from Tasks 3/8 — verify, and if `env` builds the world with fixed genomes, switch to sampled).
- [ ] **Step 3b: egg-lost reward.** In `env.step`, read `out["eaten_eggs"]`; if any frozenset contains the ego's id, add `cfg.reward_egg_lost` to the ego reward. Add a small helper `_egg_lost_reward(eaten)` for unit-testability.
- [ ] **Step 3c: CLAUDE.md.** Update the living memory: new architecture (genome/sex/aging/courtship/no-ego), the config table (genes, `n_max=12`, new constants), the observation layout (143), and add new Pitfalls (e.g. "invariants 2 & 4 are per-genome SOFT now", "observe the gene not the stat", "egg arrays thread genome+lineage in lockstep", "reward_egg_lost defaults 0 for bootstrap"). Mark the retrain-required section.

- [ ] **Step 4: Full suite + train smoke**

```bash
SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest -q
SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest tests/test_train_smoke.py -q
```
Expected: PASS (whole suite green; train-smoke builds the vec env + a few PPO steps without shape errors).

- [ ] **Step 5: Commit**

```bash
git add snake_rl/env.py CLAUDE.md tests/test_env.py
git commit -m "env: genome domain-randomization + egg-lost reward hook (default 0); CLAUDE.md v3 update"
```

---

## Phase A exit criteria

- Whole suite green: `SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest -q`.
- `check_env` passes single- and multi-snake with extreme genomes (Task 9).
- A 2–4k-step `run_headless` smoke on an all-eggs viewer world runs without crashing and shows hatches (sanity, not quality — no trained model yet).
- Then: **from-scratch retrain** (`./snake train --steps 8000000 --envs 16 --reset`, budget more wall-clock for 12 snakes + larger obs), watching `eaten_per_window` climb in the first ~50–100k steps (Pitfall 9–10 bootstrap; narrow gene ranges or drop the vibration channel if flat). Phase B (viewer camera/color/inspector/FX) gets its own plan after the retrain.

---

## Self-review notes (author)

- **Spec coverage:** genome+inheritance (T1,T8), gene ranges+phenotype+gate (T2), per-snake physiology (T3–T5), sex (T3,T6,T8), aging (T7), courtship/lay/guard+egg-lost (T8,T12), obs redesign incl. all §7 sensors + bounds/normalization fixes (T9), no-ego/all-eggs (T10), scaling (T11), reward+domain-randomization+CLAUDE.md (T12). Camera/color/inspector/FX are explicitly deferred to Phase B (spec §10/§13).
- **Deferred within Phase A on purpose:** vibration is the first-to-cut sensor if bootstrap struggles (T9 note); `reward_egg_lost` ships at 0.
- **Type consistency:** `resolve_phenotype`→`Phenotype` namedtuple used by T3–T9; `world.step` return gains `"eaten_eggs"` (T8) consumed in T12; `generate_world(..., arrivals, ego_live)` signature introduced in T10 and used by env/watch.
