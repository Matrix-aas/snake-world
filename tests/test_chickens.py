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


def test_snake_in_r_flee_triggers_flee_and_startle_burst():
    w = World(CFG, seed=3, size=(60, 60))
    w.head = np.array([30.0, 30.0]); w.head_uw = w.head.copy()
    w.set_chickens([[30.0 + CFG.r_flee * 0.5, 30.0]])  # inside r_flee, east of the snake
    w.chicken_state[0] = 0; w.chicken_timer[0] = 50    # was pecking
    before = w.chicken_pos[0].copy()
    w.update_chickens()
    assert w.chicken_state[0] == 2                      # flipped to FLEE
    assert w.chicken_pos[0][0] > before[0]              # ran away (+x, away from the snake)
    assert abs(np.linalg.norm(w.chicken_pos[0] - before) - CFG.v_startle) < 1e-6   # startle flutter


def test_startle_burst_then_settles_to_flee_speed():
    # First chicken_startle_steps flee steps flutter at v_startle, then it settles to v_flee
    # while the (fixed) snake stays inside r_flee.
    w = World(CFG, seed=4, size=(120, 120))
    w.head = np.array([40.0, 40.0]); w.head_uw = w.head.copy()
    w.set_chickens([[42.0, 40.0]])                     # 2 units east -> flees east, stays in r_flee for a while
    w.chicken_state[0] = 1; w.chicken_timer[0] = 50    # was walking
    for step in range(CFG.chicken_startle_steps + 1):
        before = w.chicken_pos[0].copy()
        w.update_chickens()
        disp = np.linalg.norm(w.chicken_pos[0] - before)
        expected = CFG.v_startle if step < CFG.chicken_startle_steps else CFG.v_flee
        assert abs(disp - expected) < 1e-6


def test_flee_settles_to_walk_when_snake_leaves():
    # Threat gone -> resume WALK (not straight back to pecking under the snake's nose).
    w = World(CFG, seed=5, size=(120, 120))
    w.head = np.array([40.0, 40.0]); w.head_uw = w.head.copy()
    w.set_chickens([[46.0, 40.0]])                     # inside r_flee -> flee
    w.chicken_state[0] = 0; w.chicken_timer[0] = 50
    w.update_chickens()
    assert w.chicken_state[0] == 2                      # fleeing
    w.head = np.array([100.0, 100.0]); w.head_uw = w.head.copy()   # snake leaves (well beyond r_flee)
    w.update_chickens()
    assert w.chicken_state[0] == 1                      # settled to WALK


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
    max_target = round(np.clip(CFG.chickens_per_snake_max * n_alive, 1, CFG.chicken_ceiling))
    w.set_chickens(np.zeros((max_target, 2)))
    for _ in range(1000):
        w.maybe_spawn()
    assert len(w.chicken_pos) == max_target


def test_spawn_refills_to_minimum():
    # Task 9: targets are population-scaled (chickens_per_snake_max/min * live snakes,
    # clamped to chicken_ceiling), not the static CFG.min_chickens/max_chickens.
    w = World(CFG, seed=3, size=(80, 80))
    n_alive = max(1, sum(1 for s in w.snakes if s.alive))
    max_target = round(np.clip(CFG.chickens_per_snake_max * n_alive, 1, CFG.chicken_ceiling))
    min_target = round(np.clip(CFG.chickens_per_snake_min * n_alive, 1, max_target))
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
    w.obstacle_pos = np.array([[40.0, 30.0]]); w.obstacle_r = np.array([3.0]); w.obstacle_kind = np.array([0])
    w.set_chickens([[34.0, 30.0]])                    # close to snake -> flees +x straight at the rock
    for _ in range(30):
        w.update_chickens()
        d = np.linalg.norm(w.chicken_pos[0] - np.array([40.0, 30.0]))
        assert d >= w.obstacle_r[0] + CFG.chicken_radius - 1e-6   # blocked at the rock surface, never inside


def test_nearest_chicken():
    w = World(CFG, seed=1, size=(60, 60)); w.head = np.array([1.0, 1.0])
    w.set_chickens([[59.0, 1.0], [5.0, 1.0]])
    idx, dist = w.nearest_chicken()
    assert idx == 0 and abs(dist - 2.0) < 1e-6        # nearest image across seam
    assert w.nearest_chicken_id() == int(w.chicken_id[0])
