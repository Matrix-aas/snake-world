import numpy as np
from snake_rl.config import CFG
from snake_rl.world import World


def test_eat_removes_chicken_and_grows():
    w = World(CFG, seed=1, size=(60, 60)); w.energy = 50.0
    w.set_chickens([w.head + [0.5, 0.0]])             # within eat_radius (2.0)
    L0 = w.target_length
    n, _ = w.try_eat()
    assert n == 1
    assert len(w.chicken_pos) == 0 and len(w.chicken_id) == 0
    assert w.target_length == min(CFG.length_cap, L0 + CFG.grow_per_chicken)
    assert w.energy == 50.0 + CFG.energy_refill       # 50 -> 90, exact


def test_growth_capped():
    w = World(CFG, seed=1, size=(60, 60)); w.target_length = CFG.length_cap - 0.1
    w.set_chickens([w.head.copy()])
    w.try_eat()
    assert w.target_length <= CFG.length_cap


# --- behavior FSM (peck / walk / flee + startle) ---

def _far_snake(w):
    """Park the ego snake far from the world centre so a centred chicken is safe (no flee)."""
    w.head = np.array([5.0, 5.0]); w.head_uw = w.head.copy()


def test_chicken_cycles_peck_walk_peck_when_safe():
    # No snake near: the FSM toggles peck -> walk -> peck as the per-state timer expires.
    w = World(CFG, seed=1, size=(60, 60)); _far_snake(w)
    w.set_chickens([[40.0, 40.0]])                    # > r_flee from the snake at (5,5)
    w.chicken_state[0] = 0; w.chicken_timer[0] = 1    # pecking, timer about to expire
    w.update_chickens()
    assert w.chicken_state[0] == 1                    # peck -> walk
    w.chicken_timer[0] = 1
    w.update_chickens()
    assert w.chicken_state[0] == 0                    # walk -> peck


def test_pecking_chicken_stands_still_is_catchable():
    w = World(CFG, seed=2, size=(60, 60)); _far_snake(w)
    w.set_chickens([[40.0, 40.0]])
    w.chicken_state[0] = 0; w.chicken_timer[0] = 50   # stay pecking this step
    before = w.chicken_pos[0].copy()
    w.update_chickens()
    assert np.linalg.norm(w.chicken_pos[0] - before) < 1e-9   # speed 0 -> a snake can walk up and eat it


def test_walking_chicken_ambles_at_v_wander():
    w = World(CFG, seed=2, size=(60, 60)); _far_snake(w)
    w.set_chickens([[40.0, 40.0]])
    w.chicken_state[0] = 1; w.chicken_timer[0] = 50   # stay walking this step
    before = w.chicken_pos[0].copy()
    w.update_chickens()
    assert abs(np.linalg.norm(w.chicken_pos[0] - before) - CFG.v_wander) < 1e-6


def test_pecking_chicken_ignores_snake_beyond_r_flee_peck():
    # Peck-distraction: a head-down pecking chicken does NOT flee a snake inside the WALK alert
    # range (r_flee) but outside the tight peck range (r_flee_peck) — the stalk-and-pounce window.
    w = World(CFG, seed=3, size=(60, 60))
    w.head = np.array([30.0, 30.0]); w.head_uw = w.head.copy()
    w.snakes[0].speed = CFG.v_snake                     # cruising -> prey senses it (still distracted by range)
    d = 0.5 * (CFG.r_flee_peck + CFG.r_flee)            # between the two ranges
    w.set_chickens([[30.0 + d, 30.0]])
    w.chicken_state[0] = 0; w.chicken_timer[0] = 50     # pecking
    before = w.chicken_pos[0].copy()
    w.update_chickens()
    assert w.chicken_state[0] == 0                       # still pecking (distracted)
    assert np.linalg.norm(w.chicken_pos[0] - before) < 1e-9   # didn't move -> catchable


def test_pecking_chicken_startles_when_snake_within_r_flee_peck():
    # Snake stalks to within r_flee_peck -> the pecking chicken startles: FLEE + a freeze beat.
    w = World(CFG, seed=3, size=(60, 60))
    w.head = np.array([30.0, 30.0]); w.head_uw = w.head.copy()
    w.snakes[0].speed = CFG.v_snake                          # cruising -> prey senses it
    w.set_chickens([[30.0 + CFG.r_flee_peck * 0.5, 30.0]])   # inside the tight peck range
    w.chicken_state[0] = 0; w.chicken_timer[0] = 50
    before = w.chicken_pos[0].copy()
    w.update_chickens()
    assert w.chicken_state[0] == 2                       # startled -> FLEE
    assert np.linalg.norm(w.chicken_pos[0] - before) < 1e-9   # frozen in surprise (speed 0) this step


def test_walking_chicken_flees_at_full_r_flee():
    # A WALKing chicken is alert: it flees at the full r_flee (a stalk only works on a pecking bird).
    w = World(CFG, seed=3, size=(60, 60))
    w.head = np.array([30.0, 30.0]); w.head_uw = w.head.copy()
    w.snakes[0].speed = CFG.v_snake                     # cruising -> prey senses it at the full r_flee
    w.set_chickens([[30.0 + CFG.r_flee * 0.7, 30.0]])   # inside r_flee, far outside r_flee_peck
    w.chicken_state[0] = 1; w.chicken_timer[0] = 50     # walking
    w.update_chickens()
    assert w.chicken_state[0] == 2                       # WALK chicken flees at the full r_flee


def test_prey_senses_snake_speed():
    # Prey senses MOTION: a stopped (speed 0) snake right next to a chicken does NOT alert it; the
    # same snake at full cruise within r_flee does. Alert scales with speed, capped at 1x base.
    w = World(CFG, seed=7, size=(60, 60))
    w.head = np.array([30.0, 30.0]); w.head_uw = w.head.copy()
    w.set_chickens([[31.0, 30.0]])                       # 1 unit away, well inside r_flee
    w.chicken_state[0] = 1; w.chicken_timer[0] = 50      # WALK: full r_flee alert
    w.snakes[0].speed = 0.0                              # stopped => invisible to prey
    w.update_chickens()
    assert w.chicken_state[0] != 2                        # did not flee
    w.snakes[0].speed = CFG.v_snake                      # full cruise => alerts
    w.update_chickens()
    assert w.chicken_state[0] == 2                        # startled -> flee


def test_scared_chicken_keeps_fleeing_after_snake_stops():
    # FEAR PERSISTENCE: once a hen is fleeing, a snake that stops dead (speed 0) must NOT instantly
    # calm it -- it keeps bolting for chicken_flee_persist steps. This kills the "spook the chicken,
    # then freeze so it re-settles, then grab it" exploit.
    w = World(CFG, seed=5, size=(120, 120))
    w.head = np.array([40.0, 40.0]); w.head_uw = w.head.copy()
    w.snakes[0].speed = CFG.v_snake                      # cruising -> scares the chicken
    w.set_chickens([[46.0, 40.0]])                       # inside r_flee
    w.chicken_state[0] = 1; w.chicken_timer[0] = 50      # walking -> flees
    for _ in range(CFG.chicken_startle_steps + 1):       # get past the startle freeze -> actually bolting
        w.update_chickens()
    assert w.chicken_state[0] == 2
    w.snakes[0].speed = 0.0                              # snake stops dead: a buggy FSM would re-calm it now
    moved = 0
    for _ in range(CFG.chicken_flee_persist - 1):        # still inside the panic window
        before = w.chicken_pos[0].copy()
        w.update_chickens()
        if np.linalg.norm(w.chicken_pos[0] - before) > 1e-6:
            moved += 1
        assert w.chicken_state[0] == 2                    # STILL fleeing despite the stopped snake
    assert moved >= 1                                     # and it actually kept bolting away
    for _ in range(3):                                    # panic expires (snake still stopped) -> calms
        w.update_chickens()
    assert w.chicken_state[0] == 1                        # settled back to WALK


def test_startle_freeze_then_bolts_at_v_flee():
    # Entering flee: FREEZE (speed 0) for chicken_startle_steps, THEN bolt away at v_flee.
    w = World(CFG, seed=4, size=(120, 120))
    w.head = np.array([40.0, 40.0]); w.head_uw = w.head.copy()
    w.snakes[0].speed = CFG.v_snake                     # cruising -> prey senses it
    w.set_chickens([[42.0, 40.0]])                      # inside r_flee_peck AND r_flee
    w.chicken_state[0] = 0; w.chicken_timer[0] = 50     # pecking -> startles
    for step in range(CFG.chicken_startle_steps + 1):
        before = w.chicken_pos[0].copy()
        w.update_chickens()
        disp = np.linalg.norm(w.chicken_pos[0] - before)
        expected = 0.0 if step < CFG.chicken_startle_steps else CFG.v_flee   # freeze, then bolt
        assert abs(disp - expected) < 1e-6


def test_flee_settles_to_walk_when_snake_leaves():
    # Threat gone -> after the fear-persistence window runs out, resume WALK (not straight back to
    # pecking under the snake's nose).
    w = World(CFG, seed=5, size=(120, 120))
    w.head = np.array([40.0, 40.0]); w.head_uw = w.head.copy()
    w.snakes[0].speed = CFG.v_snake                     # cruising -> prey senses it
    w.set_chickens([[46.0, 40.0]])                      # inside r_flee of a WALKing chicken -> flee
    w.chicken_state[0] = 1; w.chicken_timer[0] = 50     # walking (alert at the full r_flee)
    w.update_chickens()
    assert w.chicken_state[0] == 2                       # fleeing
    w.head = np.array([100.0, 100.0]); w.head_uw = w.head.copy()   # snake leaves (well beyond r_flee)
    for _ in range(CFG.chicken_flee_persist + 2):        # keeps bolting through the panic window, then calms
        w.update_chickens()
    assert w.chicken_state[0] == 1                       # settled to WALK (not peck)


def test_fsm_arrays_stay_consistent_after_eat():
    # Array consistency: eating chickens must filter chicken_state/timer/startle in lock-step.
    w = World(CFG, seed=6, size=(60, 60))
    w.head = np.array([30.0, 30.0])
    w.set_chickens([[30.0, 30.0], [30.5, 30.0], [50.0, 30.0]])   # first two within eat_radius, third survives
    w.try_eat()
    assert len(w.chicken_pos) == 1
    assert len(w.chicken_state) == len(w.chicken_pos)
    assert len(w.chicken_timer) == len(w.chicken_pos)
    assert len(w.chicken_startle) == len(w.chicken_pos)


def test_spawn_respects_max():
    # Task 9: target is population-scaled, not the static CFG.max_chickens.
    w = World(CFG, seed=2, size=(60, 60))
    n_alive = max(1, sum(1 for s in w.snakes if s.alive))
    # mirrors world.maybe_spawn's round-then-clip order (world.py: int(np.clip(round(rate*n), 1, ceiling)))
    max_target = int(np.clip(round(CFG.chickens_per_snake_max * n_alive), 1, CFG.chicken_ceiling))
    w.set_chickens(np.zeros((max_target, 2)))
    for _ in range(1000):
        w.maybe_spawn()
    assert len(w.chicken_pos) == max_target


def test_spawn_refills_to_minimum():
    # Task 9: targets are population-scaled (chickens_per_snake_max/min * live snakes,
    # clamped to chicken_ceiling), not the static CFG.min_chickens/max_chickens.
    w = World(CFG, seed=3, size=(80, 80))
    n_alive = max(1, sum(1 for s in w.snakes if s.alive))
    # mirrors world.maybe_spawn's round-then-clip order (world.py: int(np.clip(round(rate*n), 1, ceiling)))
    max_target = int(np.clip(round(CFG.chickens_per_snake_max * n_alive), 1, CFG.chicken_ceiling))
    min_target = int(np.clip(round(CFG.chickens_per_snake_min * n_alive), 1, max_target))
    w.set_chickens(np.zeros((0, 2)))                  # empty world
    for _ in range(300):
        w.maybe_spawn()
    assert min_target <= len(w.chicken_pos) <= max_target


def test_ids_stable_across_eat():
    w = World(CFG, seed=1, size=(60, 60))
    w.head = np.array([1.0, 1.0])
    w.set_chickens([[1.0, 1.0], [5.0, 1.0]])          # id 0 at head (eaten), id 1 survives
    survivor_id = int(w.chicken_id[1])
    w.try_eat()
    assert list(w.chicken_id) == [survivor_id]        # survivor keeps its id after reindex


def test_chicken_never_enters_obstacle():
    w = World(CFG, seed=1, size=(60, 60))
    w.head = np.array([30.0, 30.0]); w.head_uw = w.head.copy()
    w.snakes[0].speed = CFG.v_snake                   # cruising -> prey senses it
    w.obstacle_pos = np.array([[40.0, 30.0]]); w.obstacle_r = np.array([3.0]); w.obstacle_kind = np.array([0])
    w.set_chickens([[34.0, 30.0]])                    # close to snake -> flees +x straight at the rock
    for _ in range(30):
        w.update_chickens()
        d = np.linalg.norm(w.chicken_pos[0] - np.array([40.0, 30.0]))
        assert d >= w.obstacle_r[0] + CFG.chicken_radius - 1e-6   # blocked at the rock surface, never inside


def test_chicken_arrives_from_sky_before_it_is_huntable_and_sensed():
    # Goal 2: a spawned chicken first FALLS from the sky (world.arriving) -- invisible to eat + vision
    # -- then LANDS into the real chicken arrays after chicken_arrive_steps, becoming a normal chicken.
    from snake_rl.sensors import observe
    w = World(CFG, seed=1, size=(60, 60))
    w.head = np.array([30.0, 30.0]); w.head_uw = w.head.copy()
    w._add_chicken([30.5, 30.0], arriving=True)               # drops right by the snake's head
    assert len(w.chicken_pos) == 0 and len(w.arriving["pos"]) == 1   # in flight, not a real chicken yet
    assert len(w.arriving["head"]) == 1                       # a stable landing heading rides along (Pitfall 17)
    landing_head = float(w.arriving["head"][0])
    assert w.try_eat()[0] == 0                                # can't eat a chicken still in the air
    is_chicken = observe(w)[:88].reshape(11, 8)[:, 2]        # per-ray is_chicken one-hot
    assert not is_chicken.any()                               # ...and no vision ray reports it
    for _ in range(CFG.chicken_arrive_steps):                 # let it fall all the way down
        w._land_arrivals()
    assert len(w.arriving["pos"]) == 0 and len(w.chicken_pos) == 1   # landed -> a real chicken
    assert float(w.chicken_dir[0]) == landing_head            # landed hen keeps its in-flight facing
    assert w.try_eat()[0] == 1                                # now catchable on the snake's doorstep


def test_nearest_chicken():
    w = World(CFG, seed=1, size=(60, 60)); w.head = np.array([1.0, 1.0])
    w.set_chickens([[59.0, 1.0], [5.0, 1.0]])
    idx, dist = w.nearest_chicken()
    assert idx == 0 and abs(dist - 2.0) < 1e-6        # nearest image across seam
    assert w.nearest_chicken_id() == int(w.chicken_id[0])
