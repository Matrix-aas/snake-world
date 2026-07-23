import numpy as np
from snake_rl.config import CFG
from snake_rl.world import World, wrap, torus_dist
from snake_rl.worldgen import generate_world
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


def test_dash_into_obstacle_stuns_not_dead():
    # obstacles are SOLID now: a dash into one is non-lethal but leaves the snake dizzy (stunned).
    w = World(CFG, seed=0, size=(60, 60))
    s = w.snakes[0]
    s.head_uw = np.array([30.0, 30.0]); s.head = s.head_uw.copy(); s.heading = 0.0
    s.path_uw = [s.head_uw.copy()]; s.stamina = CFG.s_max
    w.obstacle_pos = np.array([[33.0, 30.0]]); w.obstacle_r = np.array([1.5]); w.obstacle_kind = np.array([0])
    dashed = w._move_snake(s, 3, 1, 1)          # dash +x into the rock
    assert dashed
    assert w._death_cause(s) is None            # solid, not lethal
    assert s.stun == CFG.stun_steps             # head-spinning stun


def test_walk_into_obstacle_slides_not_dead():
    # walking into a solid rock slides along it: never dead, never stunned, rests on the surface.
    w = World(CFG, seed=0, size=(60, 60))
    s = w.snakes[0]
    s.head_uw = np.array([30.0, 30.0]); s.head = s.head_uw.copy(); s.heading = 0.0
    s.path_uw = [s.head_uw.copy()]
    w.obstacle_pos = np.array([[35.0, 30.0]]); w.obstacle_r = np.array([1.5]); w.obstacle_kind = np.array([0])
    for _ in range(10):
        w._move_snake(s, 3, 1, 0)               # walk +x straight into the rock
        assert w._death_cause(s) is None
        assert s.stun == 0                       # a walk into a solid never stuns (M2)
    inflated = CFG.head_radius + 1.5             # obstacle_r + head_radius: the solid surface
    d = float(torus_dist(s.head_uw[None], w.obstacle_pos, w.size)[0])
    assert d >= inflated - 1e-3                  # slid to rest adjacent, never inside the solid


def test_stunned_snake_is_frozen_then_recovers():
    # after a dash-stun the snake does not translate for stun_steps _move_snake calls, then recovers.
    w = World(CFG, seed=0, size=(60, 60))
    s = w.snakes[0]
    s.head_uw = np.array([30.0, 30.0]); s.head = s.head_uw.copy(); s.heading = 0.0
    s.path_uw = [s.head_uw.copy()]; s.stamina = CFG.s_max
    w.obstacle_pos = np.array([[33.0, 30.0]]); w.obstacle_r = np.array([1.5]); w.obstacle_kind = np.array([0])
    w._move_snake(s, 3, 1, 1)                    # dash into rock -> stun
    assert s.stun == CFG.stun_steps
    frozen_at = s.head_uw.copy()
    for k in range(CFG.stun_steps):
        w._move_snake(s, 3, 1, 0)                # dizzy: frozen (steering frozen too)
        assert np.allclose(s.head_uw, frozen_at)
        assert s.stun == CFG.stun_steps - 1 - k
    assert s.stun == 0
    s.heading = np.pi                            # point away from the wall
    w._move_snake(s, 3, 1, 0)
    assert not np.allclose(s.head_uw, frozen_at)  # recovered -> free to move again


def test_no_death_on_empty_world():
    w = World(CFG, seed=0, size=(60, 60))
    out = w.step(3, 1, 0)
    assert not out["died"] and w.alive


def test_step_reports_eat():
    w = World(CFG, seed=0, size=(60, 60)); w.heading = 0.0
    w.head = np.array([30.0, 30.0]); w.head_uw = w.head.copy()
    w.obstacle_pos = np.zeros((0, 2)); w.obstacle_r = np.zeros((0,)); w.obstacle_kind = np.zeros((0,), int)
    w.set_chickens([[31.0, 30.0]])       # sits on the post-move head cell -> dist 0 -> wanders (no flee) -> eaten
    out = w.step(3, 1, 0)
    assert out["ate"] == 1 and not out["died"]


def test_straight_motion_never_self_collides():
    # regression: the neck-skip must clear the whole swept step, else a straight snake "hits its neck"
    for dash in (0, 1):
        w = World(CFG, seed=0, size=(60, 60))
        w.obstacle_pos = np.zeros((0, 2)); w.obstacle_r = np.zeros((0,)); w.obstacle_kind = np.zeros((0,), int)
        w.target_length = CFG.length_cap        # worst case: full body trailing the head
        w.stamina = 1e9                          # keep dashing every step
        for _ in range(60):
            assert not w.step(3, 1, dash)["died"], f"straight motion self-collided (dash={dash})"


def test_straight_full_length_advances_without_neck_deflection():
    # Pitfall-5 regression (M3): a full-length snake going straight (speed_idx=3) advances its head
    # by exactly v_snake*heading with no tangential deflection off its own (now-solid) neck.
    w = World(CFG, seed=0, size=(60, 60)); w.heading = 0.0
    w.head = np.array([30.0, 30.0]); w.head_uw = w.head.copy(); w.path_uw = [w.head_uw.copy()]
    w.obstacle_pos = np.zeros((0, 2)); w.obstacle_r = np.zeros((0,)); w.obstacle_kind = np.zeros((0,), int)
    w.target_length = CFG.length_cap
    for _ in range(60):                          # grow a full body trailing straight behind the head
        w.move(3, 1, 0)
    prev = w.head_uw.copy()
    w.move(3, 1, 0)
    d = w.head_uw - prev
    # v_snake is per-snake now (Task 4); this founder's genome is random (World.__init__), so
    # compare against its OWN resolved phenotype, not the global CFG.
    assert abs(d[0] - w.snakes[0].phenotype.v_snake) < 1e-6 and abs(d[1]) < 1e-9


def test_curled_onto_own_body_not_dead():
    # own body is a SOLID (slid along), never lethal — a full curl no longer kills.
    w = World(CFG, seed=0, size=(80, 80))
    w.target_length = CFG.length_cap
    R = CFG.v_snake / np.radians(CFG.turn_deg)
    ang = np.linspace(0.0, 2 * np.pi * 1.1, 500)      # slightly past a full circle
    center = np.array([40.0, 40.0])
    s = w.snakes[0]
    s.path_uw = [center + R * np.array([np.cos(a), np.sin(a)]) for a in ang]
    s.head_uw = s.path_uw[-1].copy(); s.head = wrap(s.head_uw, w.size)
    s._prev_head_uw = s.path_uw[-2].copy()
    assert w._death_cause(s) is None             # self-collision no longer kills


def test_head_into_other_body_kills_mover():
    import numpy as np
    from snake_rl.config import CFG
    from snake_rl.world import World, Snake, wrap
    w = World(CFG, seed=3, size=(80.0, 80.0))
    victim = w.snakes[0]                                   # long, lying along +x at y=40
    victim.head_uw = np.array([50.0, 40.0]); victim.heading = 0.0
    victim.path_uw = [np.array([40.0, 40.0]), np.array([50.0, 40.0])]
    victim.target_length = 12.0; victim._prev_head_uw = np.array([49.0, 40.0])
    victim.head = wrap(victim.head_uw, w.size)
    attacker = Snake(head_uw=np.array([45.0, 43.0]), head=wrap(np.array([45.0,43.0]), w.size),
                     heading=-np.pi/2, path_uw=[np.array([45.0,46.0]), np.array([45.0,43.0])],
                     target_length=CFG.start_length, stamina=CFG.s_max, energy=CFG.energy_max,
                     _prev_head_uw=np.array([45.0, 46.0]), id=1)
    w.snakes.append(attacker)
    # attacker's swept head crosses the victim's body line (y=40): set it explicitly, then check.
    attacker._prev_head_uw = np.array([45.0, 41.0]); attacker.head_uw = np.array([45.0, 39.0])
    attacker.head = wrap(attacker.head_uw, w.size)
    assert w._check_death(attacker) is True and attacker.death_cause == "snake"
    assert victim.alive is True


def test_mutual_head_to_head_both_die():
    import numpy as np
    from snake_rl.config import CFG
    from snake_rl.world import World, Snake, wrap
    w = World(CFG, seed=4, size=(80.0, 80.0))
    a = w.snakes[0]
    a._prev_head_uw = np.array([39.0, 40.0]); a.head_uw = np.array([41.0, 40.0]); a.heading = 0.0
    a.head = wrap(a.head_uw, w.size); a.path_uw = [a._prev_head_uw.copy(), a.head_uw.copy()]
    b = Snake(head_uw=np.array([41.0, 40.0]), head=wrap(np.array([41.0,40.0]), w.size), heading=np.pi,
              path_uw=[np.array([43.0,40.0]), np.array([41.0,40.0])], target_length=CFG.start_length,
              stamina=CFG.s_max, energy=CFG.energy_max, _prev_head_uw=np.array([43.0,40.0]), id=1)
    w.snakes.append(b)
    dead = [s.id for s in w.snakes if w._death_cause(s)]    # pure decider — matches step's phase-2 (C2)
    assert set(dead) == {0, 1}                              # both heads overlap post-move; neither hides the other
