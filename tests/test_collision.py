import numpy as np
from snake_rl.config import CFG
from snake_rl.world import World, wrap


def test_head_into_obstacle_kills():
    w = World(CFG, seed=0, size=(60, 60))
    w.head = np.array([30.0, 30.0]); w.head_uw = w.head.copy(); w.heading = 0.0
    w.obstacle_pos = np.array([[31.0, 30.0]]); w.obstacle_r = np.array([1.0]); w.obstacle_kind = np.array([0])
    out = w.step(steering=1, dash=0)   # moves +x into obstacle
    assert out["died"] and not w.alive


def test_dash_tunneling_still_kills():
    w = World(CFG, seed=0, size=(60, 60))
    w.head = np.array([30.0, 30.0]); w.head_uw = w.head.copy(); w.heading = 0.0
    w.stamina = CFG.s_max
    w.obstacle_pos = np.array([[31.2, 30.0]]); w.obstacle_r = np.array([0.3]); w.obstacle_kind = np.array([0])
    out = w.step(steering=1, dash=1)   # v_dash step would leap past a thin obstacle
    assert out["died"]


def test_no_death_on_empty_world():
    w = World(CFG, seed=0, size=(60, 60))
    out = w.step(steering=1, dash=0)
    assert not out["died"] and w.alive


def test_step_reports_eat():
    w = World(CFG, seed=0, size=(60, 60)); w.heading = 0.0
    w.head = np.array([30.0, 30.0]); w.head_uw = w.head.copy()
    w.obstacle_pos = np.zeros((0, 2)); w.obstacle_r = np.zeros((0,)); w.obstacle_kind = np.zeros((0,), int)
    w.set_chickens([[31.0, 30.0]])       # sits on the post-move head cell -> dist 0 -> wanders (no flee) -> eaten
    out = w.step(steering=1, dash=0)
    assert out["ate"] == 1 and not out["died"]


def test_straight_motion_never_self_collides():
    # regression: the neck-skip must clear the whole swept step, else a straight snake "hits its neck"
    for dash in (0, 1):
        w = World(CFG, seed=0, size=(60, 60))
        w.obstacle_pos = np.zeros((0, 2)); w.obstacle_r = np.zeros((0,)); w.obstacle_kind = np.zeros((0,), int)
        w.target_length = CFG.length_cap        # worst case: full body trailing the head
        w.stamina = 1e9                          # keep dashing every step
        for _ in range(60):
            assert not w.step(steering=1, dash=dash)["died"], f"straight motion self-collided (dash={dash})"


def test_constant_turning_eventually_self_collides():
    w = World(CFG, seed=0, size=(80, 80))
    w.obstacle_pos = np.zeros((0, 2)); w.obstacle_r = np.zeros((0,)); w.obstacle_kind = np.zeros((0,), int)
    w.target_length = CFG.length_cap
    died = any(w.step(steering=2, dash=0)["died"] for _ in range(40))   # ~1.8 full loops
    assert died                                 # curling a full circle brings the head onto its own body


def test_self_collision_when_curled():
    # craft a full circular curl (radius = min turn radius); head returns onto the tail
    w = World(CFG, seed=0, size=(80, 80))
    w.target_length = CFG.length_cap
    R = CFG.v_snake / np.radians(CFG.turn_deg)
    ang = np.linspace(0.0, 2 * np.pi * 1.1, 500)      # slightly past a full circle
    center = np.array([40.0, 40.0])
    w.path_uw = [center + R * np.array([np.cos(a), np.sin(a)]) for a in ang]
    w.head_uw = w.path_uw[-1].copy(); w.head = wrap(w.head_uw, w.size)
    w._prev_head_uw = w.path_uw[-2].copy()
    assert w.check_death()               # head overlaps a body point ~one circumference back


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
