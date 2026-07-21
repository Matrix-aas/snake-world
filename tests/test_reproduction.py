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
    a, b = _two_fed_snakes(w, d=2.0)                        # within r_mate (4.0)
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
