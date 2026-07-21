import numpy as np
from snake_rl.config import CFG
from snake_rl.world import World, Snake
from snake_rl.worldgen import generate_world


def test_world_has_snake_list_and_ego_proxies():
    w = World(CFG, seed=1, size=(80.0, 80.0))
    assert isinstance(w.snakes, list) and len(w.snakes) == 1
    s = w.snakes[0]
    assert isinstance(s, Snake)
    # read proxies mirror the ego
    assert np.allclose(w.head, s.head)
    assert w.energy == s.energy and w.stamina == s.stamina
    assert w.heading == s.heading
    # proxies are read/WRITE — assignment must route to snakes[0] (existing tests assign w.head, w.stamina, ...)
    w.stamina = 5.0; assert w.snakes[0].stamina == 5.0
    w.energy = 7.0;  assert w.snakes[0].energy == 7.0


def test_ego_move_proxy_mutates_snakes0():
    w = World(CFG, seed=1, size=(80.0, 80.0))
    before = w.snakes[0].head_uw.copy()
    w.move(1, 0)                      # straight, no dash — via proxy
    assert not np.allclose(w.snakes[0].head_uw, before)
    assert w.snakes[0].steps == 1


def test_n1_trajectory_regression():
    # Determinism/smoke: a fixed seed+size world stepped straight advances finitely each step.
    # NOTE: exact byte-for-byte physics preservation is guarded by the EXISTING pinned tests
    # (test_snake_motion: |speed - v_snake| < 1e-6, exact segment spacing; test_collision:
    # self-collision reachability). This test only checks the refactor didn't NaN/stall the ego.
    w = generate_world(CFG, seed=0, size=(80.0, 80.0))
    heads = []
    for _ in range(30):
        w.step(1, 0)                  # steer straight, no dash
        assert w.snakes[0].alive      # guard: a mid-run death would stall head motion (M3)
        heads.append(w.head.copy())
    heads = np.array(heads)
    assert np.isfinite(heads).all()
    seg = np.linalg.norm(np.diff(heads, axis=0), axis=1)
    assert (seg > 0).all()


def test_two_phase_step_is_order_independent_head_to_head():
    # Two snakes driven head-on into each other die together regardless of list order.
    w = World(CFG, seed=2, size=(80.0, 80.0))
    from snake_rl.world import Snake
    from snake_rl.world import wrap
    import numpy as np
    a = w.snakes[0]
    a.head_uw = np.array([40.0, 40.0]); a.head = wrap(a.head_uw, w.size)
    a.heading = 0.0; a.path_uw = [a.head_uw.copy()]; a._prev_head_uw = a.head_uw.copy()
    b = Snake(head_uw=np.array([43.0, 40.0]), head=wrap(np.array([43.0,40.0]), w.size),
              heading=np.pi, path_uw=[np.array([43.0,40.0])], target_length=CFG.start_length,
              stamina=CFG.s_max, energy=CFG.energy_max, _prev_head_uw=np.array([43.0,40.0]), id=1)
    w.snakes.append(b)
    # will be lethal once Task 4 wires _other_hazard; here assert the step runs both moves first
    out = w.step(1, 0, opponent_fn=lambda world, s: (1, 0))
    assert w.snakes[0].steps == 1 and w.snakes[1].steps == 1   # BOTH moved (phase 1) before any resolve


def test_dead_snake_becomes_corpse_and_is_edible():
    import numpy as np
    from snake_rl.config import CFG
    from snake_rl.world import World, wrap
    w = World(CFG, seed=5, size=(80.0, 80.0))
    w.chicken_pos = np.zeros((0, 2)); w.chicken_dir = np.zeros((0,)); w.chicken_id = np.zeros((0,), int)
    w._spawn_corpse(w.snakes[0])
    assert w.corpses["pos"].shape == (1, 2)
    assert w.corpses["food"][0] == CFG.corpse_food_per_length * w.snakes[0].target_length
    # move ego onto the corpse -> eats it (one item), energy up, corpse gone
    w.snakes[0].energy = 10.0
    w.snakes[0].head = w.corpses["pos"][0].copy()
    n = w.try_eat()
    assert n == 1 and w.corpses["pos"].shape == (0, 2) and w.snakes[0].energy > 10.0
