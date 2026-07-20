import numpy as np
from snake_rl.config import CFG
from snake_rl.world import World


def test_eat_removes_chicken_and_grows():
    w = World(CFG, seed=1, size=(60, 60)); w.energy = 50.0
    w.set_chickens([w.head + [0.5, 0.0]])             # within eat_radius (2.0)
    L0 = w.target_length
    n = w.try_eat()
    assert n == 1
    assert len(w.chicken_pos) == 0 and len(w.chicken_id) == 0
    assert w.target_length == min(CFG.length_cap, L0 + CFG.grow_per_chicken)
    assert w.energy == 50.0 + CFG.energy_refill       # 50 -> 90, exact


def test_growth_capped():
    w = World(CFG, seed=1, size=(60, 60)); w.target_length = CFG.length_cap - 0.1
    w.set_chickens([w.head.copy()])
    w.try_eat()
    assert w.target_length <= CFG.length_cap


def test_chicken_flees_away_when_snake_close():
    w = World(CFG, seed=1, size=(60, 60))
    w.head = np.array([30.0, 30.0]); w.head_uw = w.head.copy()
    w.set_chickens([[30.0 + CFG.r_flee * 0.5, 30.0]])
    before = w.chicken_pos[0].copy()
    w.update_chickens()
    assert w.chicken_pos[0][0] > before[0]            # moved away (+x)
    assert abs(np.linalg.norm(w.chicken_pos[0] - before) - CFG.v_flee) < 1e-6


def test_spawn_respects_max():
    w = World(CFG, seed=2, size=(60, 60))
    w.set_chickens(np.zeros((CFG.max_chickens, 2)))
    for _ in range(1000):
        w.maybe_spawn()
    assert len(w.chicken_pos) == CFG.max_chickens


def test_ids_stable_across_eat():
    w = World(CFG, seed=1, size=(60, 60))
    w.head = np.array([1.0, 1.0])
    w.set_chickens([[1.0, 1.0], [5.0, 1.0]])          # id 0 at head (eaten), id 1 survives
    survivor_id = int(w.chicken_id[1])
    w.try_eat()
    assert list(w.chicken_id) == [survivor_id]        # survivor keeps its id after reindex


def test_nearest_chicken():
    w = World(CFG, seed=1, size=(60, 60)); w.head = np.array([1.0, 1.0])
    w.set_chickens([[59.0, 1.0], [5.0, 1.0]])
    idx, dist = w.nearest_chicken()
    assert idx == 0 and abs(dist - 2.0) < 1e-6        # nearest image across seam
    assert w.nearest_chicken_id() == int(w.chicken_id[0])
