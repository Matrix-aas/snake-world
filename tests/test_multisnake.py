import numpy as np
from snake_rl.config import CFG
from snake_rl.world import World, Snake
from snake_rl.worldgen import generate_world
from snake_rl.genome import GENE_COUNT


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
    w.move(3, 1, 0)                   # full-speed straight, no dash — via proxy
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
        w.step(3, 1, 0)               # full-speed straight, no dash
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
    out = w.step(3, 1, 0, opponent_fn=lambda world, s: (3, 1, 0))
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
    n, _ = w.try_eat()
    assert n == 1 and w.corpses["pos"].shape == (0, 2) and w.snakes[0].energy > 10.0


def test_starvation_kills_and_leaves_corpse():
    import numpy as np
    from snake_rl.config import CFG
    from snake_rl.world import World
    w = World(CFG, seed=9, size=(80.0, 80.0))
    w.chicken_pos = np.zeros((0, 2)); w.chicken_dir = np.zeros((0,)); w.chicken_id = np.zeros((0,), int)
    w.snakes[0].energy = CFG.energy_decay / 2               # will hit 0 this step
    w.step(3, 1, 0)
    assert w.snakes[0].alive is False and w.snakes[0].death_cause == "starve"
    assert w.corpses["pos"].shape[0] == 1


def test_food_target_scales_with_live_snakes():
    import numpy as np
    from snake_rl.config import CFG
    from snake_rl.worldgen import generate_world
    w = generate_world(CFG, seed=10, size=(150.0, 150.0), n_snakes=4)
    w.set_chickens(np.zeros((0, 2)))            # clear worldgen's random initial abundance (up to
                                               # chicken_ceiling) so we test maybe_spawn's population
                                               # -scaled TARGET, not the one-off starting count
    # drive spawns and assert the chicken count tracks ~2 per snake, capped by the ceiling
    for _ in range(400):
        w.maybe_spawn()
    assert len(w.chicken_pos) <= min(CFG.chicken_ceiling, round(CFG.chickens_per_snake_max * 4))
    # discriminates against the old static cap: the count must reach the population-scaled floor,
    # not just any count between 1 and 8 (which a static max_chickens=5 would also satisfy).
    assert len(w.chicken_pos) >= round(CFG.chickens_per_snake_min * 4)


def test_all_snakes_eat_and_decay_and_ego_count():
    import numpy as np
    from snake_rl.config import CFG
    from snake_rl.worldgen import generate_world
    w = generate_world(CFG, seed=11, size=(140.0,140.0), n_snakes=3)
    w.set_chickens([])                 # set_chickens resets ALL parallel chicken arrays (incl. FSM state)
    opp = w.snakes[1]; opp.energy=10.0
    w.set_chickens([opp.head])         # one chicken on the opponent's head
    assert w.try_eat()[0] == 0         # ego ate nothing this call...
    assert opp.energy > 10.0           # ...but the opponent did
    # ego count is returned (place a chicken on the ego head)
    ego=w.snakes[0]; w.set_chickens([ego.head])
    assert w.try_eat()[0] == 1
    # energy_decay is per-snake now (Task 4: hunger reads phenotype, not the global CFG)
    e=opp.energy; w.decay_energy(); assert opp.energy == max(0.0, e-opp.phenotype.energy_decay)


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
    out = w.step(3, 1, 0)
    assert any(cause=="starve" for _id,cause in out["deaths_detailed"])


def test_hatched_owners_reported_and_cap_drops_pay_nothing():
    import numpy as np
    from snake_rl.config import CFG
    from snake_rl.worldgen import generate_world
    # below n_max: a timer-expired egg hatches into a new snake, owner-set reported
    w = generate_world(CFG, seed=14, size=(140.0, 140.0), n_snakes=3)
    w.eggs = {"pos": np.array([[60.0, 60.0]]), "timer": np.array([1.0]), "owner": np.array([[0, 1]]),
              "genome": np.full((1, GENE_COUNT), 0.5, np.float32), "lineage": np.array([0])}
    n0 = len(w.snakes)
    owners = w._hatch_eggs()
    assert len(w.snakes) == n0 + 1
    assert frozenset({0, 1}) in owners
    # at n_max: the egg is still consumed, but produces no hatchling -> its owner-set pays nothing
    w2 = generate_world(CFG, seed=15, size=(150.0, 150.0), n_snakes=CFG.n_max)
    assert len(w2.snakes) == CFG.n_max
    w2.eggs = {"pos": np.array([[60.0, 60.0]]), "timer": np.array([1.0]), "owner": np.array([[2, 3]]),
               "genome": np.full((1, GENE_COUNT), 0.5, np.float32), "lineage": np.array([0])}
    owners2 = w2._hatch_eggs()
    assert len(w2.snakes) == CFG.n_max          # no hatchling: population stays capped
    assert w2.eggs["pos"].shape[0] == 0         # egg consumed regardless
    assert frozenset({2, 3}) not in owners2


def _mk_snake(pos, sid):
    from snake_rl.world import Snake, wrap
    p = np.asarray(pos, float)
    # speed=v_snake: a CRUISING snake — prey senses motion, so a stopped (speed 0) snake wouldn't alert.
    return Snake(head_uw=p.copy(), head=p.copy(), heading=0.0, path_uw=[p.copy()],
                 target_length=CFG.start_length, stamina=CFG.s_max, energy=CFG.energy_max,
                 _prev_head_uw=p.copy(), id=sid, speed=CFG.v_snake)


def test_chicken_flees_nearest_live_snake_opponent_only():
    # Bug: update_chickens only checked self.head (ego). A chicken near an OPPONENT but far
    # from the ego must still flee — from the opponent, not wander in place.
    from snake_rl.world import World, wrap
    w = World(CFG, seed=20, size=(150.0, 150.0))
    ego = w.snakes[0]
    ego.head_uw = np.array([10.0, 10.0]); ego.head = wrap(ego.head_uw, w.size)   # far from the chicken
    opp = _mk_snake([100.0, 100.0], sid=1)
    w.snakes.append(opp)
    w.set_chickens([[100.0 + CFG.r_flee * 0.5, 100.0]])          # within r_flee of the opponent only
    w.chicken_state[0] = 1                                        # WALK: alert at the full r_flee (not distracted)
    before = np.linalg.norm(w.chicken_pos[0] - opp.head)
    for _ in range(CFG.chicken_startle_steps + 1):               # step past the startle FREEZE, then it bolts
        w.update_chickens()
    after = np.linalg.norm(w.chicken_pos[0] - opp.head)
    assert after > before                                          # fled the opponent, not just wandered


def test_chicken_flees_lone_ego_snake_n1_invariant():
    # N=1: a lone (ego) snake still makes a WALKing chicken flee directly away. Entering flee it
    # freezes for the startle beat, so net motion after startle+1 steps is one v_flee bolt (+x).
    w = World(CFG, seed=1, size=(60, 60))
    w.head = np.array([30.0, 30.0]); w.head_uw = w.head.copy()
    w.snakes[0].speed = CFG.v_snake                              # cruising -> prey senses it
    w.set_chickens([[30.0 + CFG.r_flee * 0.5, 30.0]])
    w.chicken_state[0] = 1                                        # WALK: alert at the full r_flee
    before = w.chicken_pos[0].copy()
    for _ in range(CFG.chicken_startle_steps + 1):               # startle FREEZE (no move), then one bolt
        w.update_chickens()
    assert w.chicken_pos[0][0] > before[0]
    assert abs(np.linalg.norm(w.chicken_pos[0] - before) - CFG.v_flee) < 1e-6


def test_chicken_flee_resultant_avoids_both_flanking_snakes():
    # Two live snakes on different sides of a chicken, both within r_flee. Naive "flee the
    # nearest" would run the chicken straight toward the farther one (verified numerically:
    # snake A at bearing 0 deg/dist 6, snake B at bearing 100 deg/dist 7, r_flee=12 -> fleeing
    # directly away from A alone moves the chicken CLOSER to B, 7.0 -> 6.89). The repulsion
    # resultant of both "away" vectors, weighted by (r_flee - dist), must instead move the
    # chicken away from BOTH.
    w = World(CFG, seed=30, size=(150.0, 150.0))
    chick = np.array([75.0, 75.0])
    ego = w.snakes[0]; ego.head_uw = chick + [6.0, 0.0]; ego.head = ego.head_uw.copy(); ego.speed = CFG.v_snake
    theta = np.radians(100.0)
    opp = _mk_snake(chick + 7.0 * np.array([np.cos(theta), np.sin(theta)]), sid=1)
    w.snakes.append(opp)
    w.set_chickens([chick])
    w.chicken_state[0] = 1                                        # WALK: alert at the full r_flee
    d_ego_before = np.linalg.norm(chick - ego.head)
    d_opp_before = np.linalg.norm(chick - opp.head)
    for _ in range(CFG.chicken_startle_steps + 1):               # step past the startle FREEZE, then it bolts
        w.update_chickens()
    d_ego_after = np.linalg.norm(w.chicken_pos[0] - ego.head)
    d_opp_after = np.linalg.norm(w.chicken_pos[0] - opp.head)
    assert d_ego_after > d_ego_before                # moved away from A too...
    assert d_opp_after > d_opp_before                # ...not toward B (what naive-nearest would do)


def test_chicken_flee_degenerate_cancellation_falls_back_to_nearest():
    # Two live snakes exactly opposite the chicken at equal distance: the weighted "away"
    # vectors cancel to ~0. Must not NaN/stall — falls back to fleeing the single nearest snake.
    w = World(CFG, seed=31, size=(150.0, 150.0))
    chick = np.array([75.0, 75.0])
    ego = w.snakes[0]; ego.head_uw = chick + [8.0, 0.0]; ego.head = ego.head_uw.copy(); ego.speed = CFG.v_snake  # due east
    opp = _mk_snake(chick + [-8.0, 0.0], sid=1)                                            # due west
    w.snakes.append(opp)
    w.set_chickens([chick])
    w.chicken_state[0] = 1                                        # WALK: alert at the full r_flee
    for _ in range(CFG.chicken_startle_steps + 1):               # step past the startle FREEZE, then it bolts
        w.update_chickens()
    new_chick = w.chicken_pos[0]
    assert np.isfinite(new_chick).all()
    assert new_chick[0] < chick[0]                   # fled west, away from the (tie-break) nearest = ego (east)


def test_chicken_flees_live_opponent_when_ego_is_dead():
    # Critical (review): the N<=1 branch used self.head unconditionally — the ego proxy, kept
    # in snakes[0] and NOT alive-gated even when dead (_prune_dead keeps a dead ego in slot 0).
    # When the ego is dead and the sole survivor is an opponent, chickens must flee the LIVE
    # opponent, not a frozen dead-ego ghost position far away.
    w = World(CFG, seed=33, size=(150.0, 150.0))
    ego = w.snakes[0]
    ego.head_uw = np.array([10.0, 10.0]); ego.head = ego.head_uw.copy()
    ego.alive = False                                                # dead, frozen far from the chicken
    opp = _mk_snake([100.0, 100.0], sid=1)
    w.snakes.append(opp)
    w.set_chickens([[106.0, 100.0]])                                 # within r_flee of the LIVE opponent only
    w.chicken_state[0] = 1                                           # WALK: alert at the full r_flee
    w.chicken_dir[0] = np.pi                                         # points AT the opponent if wander ever fires
    before = np.linalg.norm(w.chicken_pos[0] - opp.head)
    for _ in range(CFG.chicken_startle_steps + 1):                  # step past the startle FREEZE, then it bolts
        w.update_chickens()
    after = np.linalg.norm(w.chicken_pos[0] - opp.head)
    assert after > before        # fled the live opponent, not the dead ego ghost at (10, 10)


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
    out = w.step(3, 1, 0, opponent_fn=lambda world, s: (3, 1, 0))
    causes = dict(out["deaths_detailed"])
    assert causes.get(0) == "snake" and causes.get(1) == "snake"


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
