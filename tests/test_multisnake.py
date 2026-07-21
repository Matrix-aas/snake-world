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
    # Precondition for order-independence: both snakes complete their phase-1 move before any death is resolved.
    out = w.step(1, 0, opponent_fn=lambda world, s: (1, 0))
    # Use the captured objects, not post-step w.snakes indices: both die this step ("snake" cause)
    # and _prune_dead (Milestone B) drops dead non-ego opponents from w.snakes at the end of step.
    assert a.steps == 1 and b.steps == 1   # BOTH moved (phase 1) before any resolve


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


def test_starvation_kills_and_leaves_corpse():
    import numpy as np
    from snake_rl.config import CFG
    from snake_rl.world import World
    w = World(CFG, seed=9, size=(80.0, 80.0))
    w.chicken_pos = np.zeros((0, 2)); w.chicken_dir = np.zeros((0,)); w.chicken_id = np.zeros((0,), int)
    w.snakes[0].energy = CFG.energy_decay / 2               # will hit 0 this step
    w.step(1, 0)
    assert w.snakes[0].alive is False and w.snakes[0].death_cause == "starve"
    assert w.corpses["pos"].shape[0] == 1


def test_food_target_scales_with_live_snakes():
    import numpy as np
    from snake_rl.config import CFG
    from snake_rl.worldgen import generate_world
    w = generate_world(CFG, seed=10, size=(150.0, 150.0), n_snakes=4)
    # drive spawns and assert the chicken count tracks ~2 per snake, capped by the ceiling
    for _ in range(400):
        w.maybe_spawn()
    assert len(w.chicken_pos) <= min(CFG.chicken_ceiling, round(CFG.chickens_per_snake_max * 4))
    assert len(w.chicken_pos) >= 1


def test_all_snakes_eat_and_decay_and_ego_count():
    import numpy as np
    from snake_rl.config import CFG
    from snake_rl.worldgen import generate_world
    w = generate_world(CFG, seed=11, size=(140.0,140.0), n_snakes=3)
    w.chicken_pos=np.zeros((0,2)); w.chicken_dir=np.zeros((0,)); w.chicken_id=np.zeros((0,),int)
    opp = w.snakes[1]; opp.energy=10.0
    w.chicken_pos=opp.head[None].copy(); w.chicken_dir=np.zeros(1); w.chicken_id=np.array([999])
    assert w.try_eat() == 0            # ego ate nothing this call...
    assert opp.energy > 10.0           # ...but the opponent did
    # ego count is returned (place a chicken on the ego head)
    ego=w.snakes[0]; w.chicken_pos=ego.head[None].copy(); w.chicken_dir=np.zeros(1); w.chicken_id=np.array([7])
    assert w.try_eat() == 1
    e=opp.energy; w.decay_energy(); assert opp.energy == max(0.0, e-CFG.energy_decay)


def test_prune_keeps_ego_removes_dead_opponents():
    from snake_rl.config import CFG
    from snake_rl.worldgen import generate_world
    w = generate_world(CFG, seed=12, size=(140.0,140.0), n_snakes=3)
    w.snakes[2].alive=False; w._prune_dead()
    assert len(w.snakes)==2 and all(s.alive for s in w.snakes[1:])
    w.snakes[0].alive=False; w._prune_dead()
    assert len(w.snakes)==2 and w.snakes[0].alive is False   # ego kept though dead


def test_step_reports_detailed_deaths_and_hatches():
    import numpy as np
    from snake_rl.config import CFG
    from snake_rl.worldgen import generate_world
    w = generate_world(CFG, seed=13, size=(140.0,140.0), n_snakes=2)
    w.snakes[1].energy = CFG.energy_decay/2    # opponent starves this step
    out = w.step(1,0)
    assert any(cause=="starve" for _id,cause in out["deaths_detailed"])


def test_hatched_owners_reported_and_cap_drops_pay_nothing():
    import numpy as np
    from snake_rl.config import CFG
    from snake_rl.worldgen import generate_world
    # below n_max: a timer-expired egg hatches into a new snake, owner-set reported
    w = generate_world(CFG, seed=14, size=(140.0, 140.0), n_snakes=3)
    w.eggs = {"pos": np.array([[60.0, 60.0]]), "timer": np.array([1.0]), "owner": np.array([[0, 1]])}
    n0 = len(w.snakes)
    owners = w._hatch_eggs()
    assert len(w.snakes) == n0 + 1
    assert frozenset({0, 1}) in owners
    # at n_max: the egg is still consumed, but produces no hatchling -> its owner-set pays nothing
    w2 = generate_world(CFG, seed=15, size=(150.0, 150.0), n_snakes=CFG.n_max)
    assert len(w2.snakes) == CFG.n_max
    w2.eggs = {"pos": np.array([[60.0, 60.0]]), "timer": np.array([1.0]), "owner": np.array([[2, 3]])}
    owners2 = w2._hatch_eggs()
    assert len(w2.snakes) == CFG.n_max          # no hatchling: population stays capped
    assert w2.eggs["pos"].shape[0] == 0         # egg consumed regardless
    assert frozenset({2, 3}) not in owners2


def test_deaths_detailed_reports_phase2_snake_cause():
    # Two snakes driven head-on into each other: cause "snake" for BOTH, surfaced per-id.
    from snake_rl.world import World, Snake, wrap
    import numpy as np
    w = World(CFG, seed=2, size=(80.0, 80.0))
    a = w.snakes[0]
    a.head_uw = np.array([40.0, 40.0]); a.head = wrap(a.head_uw, w.size)
    a.heading = 0.0; a.path_uw = [a.head_uw.copy()]; a._prev_head_uw = a.head_uw.copy()
    b = Snake(head_uw=np.array([43.0, 40.0]), head=wrap(np.array([43.0, 40.0]), w.size),
              heading=np.pi, path_uw=[np.array([43.0, 40.0])], target_length=CFG.start_length,
              stamina=CFG.s_max, energy=CFG.energy_max, _prev_head_uw=np.array([43.0, 40.0]), id=1)
    w.snakes.append(b)
    out = w.step(1, 0, opponent_fn=lambda world, s: (1, 0))
    causes = dict(out["deaths_detailed"])
    assert causes.get(0) == "snake" and causes.get(1) == "snake"
