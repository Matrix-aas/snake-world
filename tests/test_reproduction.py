import numpy as np
from snake_rl.config import CFG
from snake_rl.world import World, Snake, wrap


def _two_fed_snakes(w, d=2.0):
    a = w.snakes[0]
    a.head_uw = np.array([40.0, 40.0]); a.head = wrap(a.head_uw, w.size)
    a.energy = CFG.energy_max; a.target_length = CFG.repro_length_min + 2; a.repro_cooldown = 0
    b = Snake(head_uw=np.array([40.0 + d, 40.0]), head=wrap(np.array([40.0+d,40.0]), w.size),
              heading=np.pi, path_uw=[np.array([40.0+d,40.0])], target_length=CFG.repro_length_min+2,
              stamina=CFG.s_max, energy=CFG.energy_max, _prev_head_uw=np.array([40.0+d,40.0]), id=1)
    w.snakes.append(b)
    return a, b


def test_mating_lays_egg_after_streak_and_costs_energy():
    w = World(CFG, seed=6, size=(80.0, 80.0))
    a, b = _two_fed_snakes(w, d=2.0)                        # within r_mate (CFG.r_mate)
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
              "owner": np.array([[0, 1]])}
    n0 = len(w.snakes)
    w._hatch_eggs()                                        # timer 1 -> 0 -> hatch
    assert len(w.snakes) == n0 + 1
    baby = w.snakes[-1]
    assert baby.target_length == CFG.start_length
    assert np.allclose(baby.head_uw, np.array([40.0, 40.0]))
    assert w.eggs["pos"].shape[0] == 0


def test_parent_cannot_eat_own_egg_but_rival_can():
    w = World(CFG, seed=8, size=(80.0, 80.0))
    ego = w.snakes[0]; ego.id = 0; ego.head = np.array([40.0, 40.0]); ego.energy = 10.0
    w.eggs = {"pos": np.array([[40.0, 40.0]]), "timer": np.array([30.0]), "owner": np.array([[0, 1]])}
    assert w.try_eat() == 0 and w.eggs["pos"].shape[0] == 1     # own egg: not eaten
    ego.id = 5                                                  # now a non-owner
    assert w.try_eat() == 1 and w.eggs["pos"].shape[0] == 0     # foreign egg: eaten
    assert ego.energy == 10.0 + CFG.egg_food
