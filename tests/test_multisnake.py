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
