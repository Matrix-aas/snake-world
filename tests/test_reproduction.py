import numpy as np
from snake_rl.config import CFG
from snake_rl.world import World, Snake, wrap
from snake_rl.worldgen import generate_world
from snake_rl import genome as gm


def _two_fed_snakes(w, d=2.0):
    # length_cap clears the size-RELATIVE gate (frac * own max_length) for any genome (Task 8)
    a = w.snakes[0]
    a.head_uw = np.array([40.0, 40.0]); a.head = wrap(a.head_uw, w.size)
    a.energy = CFG.energy_max; a.target_length = CFG.length_cap; a.repro_cooldown = 0
    b = Snake(head_uw=np.array([40.0 + d, 40.0]), head=wrap(np.array([40.0+d,40.0]), w.size),
              heading=np.pi, path_uw=[np.array([40.0+d,40.0])], target_length=CFG.length_cap,
              stamina=CFG.s_max, energy=CFG.energy_max, _prev_head_uw=np.array([40.0+d,40.0]), id=1)
    w.snakes.append(b)
    return a, b


def test_mating_lays_egg_after_streak_and_costs_energy():
    w = World(CFG, seed=6, size=(80.0, 80.0))
    a, b = _two_fed_snakes(w, d=2.0)                        # within r_mate (CFG.r_mate)
    a.sex = 0; b.sex = 1                                    # opposite sexes -- sex gate requires it
    for _ in range(CFG.mate_steps):
        w._resolve_mating()
    assert w.eggs["pos"].shape[0] == 1
    assert set(w.eggs["owner"][0].tolist()) == {0, 1}
    assert a.energy == CFG.energy_max - CFG.repro_cost
    assert b.energy == CFG.energy_max - CFG.repro_cost
    assert a.repro_cooldown > 0 and b.repro_cooldown > 0


def test_no_egg_if_separated_before_streak_completes():
    w = World(CFG, seed=6, size=(80.0, 80.0))
    a, b = _two_fed_snakes(w, d=2.0)
    w._resolve_mating()                                    # 1 qualifying step
    b.head_uw = np.array([70.0, 40.0]); b.head = wrap(b.head_uw, w.size)  # bolt away
    for _ in range(CFG.mate_steps):
        w._resolve_mating()
    assert w.eggs["pos"].shape[0] == 0


def test_egg_hatches_into_new_snake():
    w = World(CFG, seed=8, size=(80.0, 80.0))
    w.eggs = {"pos": np.array([[40.0, 40.0]]), "timer": np.array([1.0]),
              "owner": np.array([[0, 1]]),
              "genome": np.full((1, gm.GENE_COUNT), 0.5, np.float32), "lineage": np.array([3])}
    n0 = len(w.snakes)
    w._hatch_eggs()                                        # timer 1 -> 0 -> hatch
    assert len(w.snakes) == n0 + 1
    baby = w.snakes[-1]
    assert baby.target_length == CFG.start_length
    assert np.allclose(baby.head_uw, np.array([40.0, 40.0]))
    assert baby.lineage == 3                                # hatchling inherits the egg's carried lineage
    assert np.allclose(baby.genome, 0.5)                    # ...and its carried genome
    assert w.eggs["pos"].shape[0] == 0


def test_spawn_egg_is_uneatable_and_hatches_even_at_population_cap():
    # Goal 1: a spawn_egg (owner -1) is a GUARANTEED arrival -- no snake can eat it, and it hatches
    # even when the population is already at n_max (a repro egg would be cap-dropped there).
    from snake_rl.worldgen import generate_world
    w = generate_world(CFG, seed=15, size=(150.0, 150.0), n_snakes=CFG.n_max)   # n_max LIVE snakes
    assert len(w.snakes) == CFG.n_max
    ego = w.snakes[0]; ego.energy = 10.0
    w.spawn_egg(ego.head.copy())                       # sits right on the ego's head
    assert w.try_eat()[0] == 0                          # nobody eats a spawn egg (owner -1)...
    assert w.eggs["pos"].shape[0] == 1 and ego.energy == 10.0
    w.eggs["timer"][:] = 1                             # about to hatch
    n0 = len(w.snakes)
    owners = w._hatch_eggs()
    assert len(w.snakes) == n0 + 1                     # ...and it hatches despite pop == n_max (cap-exempt)
    assert owners == []                                # a spawn hatch is NOT reproduction -> pays nothing


def test_parent_cannot_eat_own_egg_but_rival_can():
    w = World(CFG, seed=8, size=(80.0, 80.0))
    ego = w.snakes[0]; ego.id = 0; ego.head = np.array([40.0, 40.0]); ego.energy = 10.0
    w.eggs = {"pos": np.array([[40.0, 40.0]]), "timer": np.array([30.0]), "owner": np.array([[0, 1]]),
              "genome": np.zeros((1, gm.GENE_COUNT), np.float32), "lineage": np.array([0])}
    assert w.try_eat()[0] == 0 and w.eggs["pos"].shape[0] == 1     # own egg: not eaten
    ego.id = 5                                                     # now a non-owner
    assert w.try_eat()[0] == 1 and w.eggs["pos"].shape[0] == 0     # foreign egg: eaten
    assert ego.energy == 10.0 + CFG.egg_food


def _ready_pair(seed, sexA, sexB, dist=3.0, cfg=CFG):
    w = generate_world(cfg, seed=seed, n_snakes=2, size=(100.0, 100.0))   # >2*(ray_range_max+obst+head) (M4)
    a, b = w.snakes[0], w.snakes[1]
    # place them close, well-fed, grown, off cooldown, opposite/same sex per args
    for s, sx in ((a, sexA), (b, sexB)):
        s.energy = CFG.energy_max
        s.target_length = CFG.length_cap
        s.repro_cooldown = 0
        s.sex = sx
    b.head[:] = a.head + np.array([dist, 0.0]); b.head_uw[:] = b.head
    b.path_uw = [b.head_uw.copy()]; b._prev_head_uw = b.head_uw.copy()   # freshly placed: no phantom
    return w, a, b                                                        # body linking the old spawn to here


def test_same_sex_pair_never_lays():
    w, a, b = _ready_pair(21, 0, 0)
    n_eggs0 = len(w.eggs["pos"])
    for _ in range(CFG.mate_steps + 3):
        w.step(0, 1, 0, opponent_fn=lambda world, s: (0, 1, 0))
    assert len(w.eggs["pos"]) == n_eggs0, "same-sex pair must not produce an egg"


def test_same_sex_pair_lays_when_sex_gate_relaxed_by_curriculum():
    # repro-discovery curriculum easy end: mate_require_sex=False lets a same-sex pair mate too
    from dataclasses import replace
    easy = replace(CFG, mate_require_sex=False)
    w, a, b = _ready_pair(21, 0, 0, cfg=easy)
    n_eggs0 = len(w.eggs["pos"])
    for _ in range(CFG.mate_steps + 3):
        w.step(0, 1, 0, opponent_fn=lambda world, s: (0, 1, 0))
    assert len(w.eggs["pos"]) > n_eggs0, "same-sex pair should lay an egg once the sex gate is relaxed"


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


# --- Task 8: genome-carrying eggs, maternal lineage, eaten-egg channel, size-relative gate ---

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
    assert (child >= 0).all() and (child <= 1).all()
    # parents are all-0 / all-1, mutation sigma small => every gene stays NEAR a parent value
    assert ((child < 0.2) | (child > 0.8)).all(), "child genes should be a mutated per-gene parent pick"
    assert w.eggs["lineage"][laid_idx] == 77   # maternal lineage (female's)


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
