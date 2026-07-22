import numpy as np
from gymnasium.utils.env_checker import check_env
from snake_rl.config import CFG
from snake_rl.env import SnakeEnv


def test_gymnasium_check_env():
    check_env(SnakeEnv(seed=0), skip_render_check=True)


def test_reset_seed_deterministic():
    o1, _ = SnakeEnv(seed=7).reset(seed=7)
    o2, _ = SnakeEnv(seed=7).reset(seed=7)
    np.testing.assert_allclose(o1, o2)


def test_truncates_at_horizon():
    env = SnakeEnv(seed=0); env.reset()
    term = trunc = False
    for _ in range(CFG.episode_horizon + 5):
        _, _, term, trunc, _ = env.step([1, 0])
        if term or trunc:
            break
    assert trunc or term


def test_eat_gives_positive_reward():
    env = SnakeEnv(seed=0); env.reset()
    w = env.world
    w.head = np.array([30.0, 30.0]); w.head_uw = w.head.copy(); w.heading = 0.0
    w.obstacle_pos = np.zeros((0, 2)); w.obstacle_r = np.zeros((0,)); w.obstacle_kind = np.zeros((0,), int)
    w.set_chickens([[31.0, 30.0]])          # lands on the post-move head cell (dist 0 -> wanders, not fled) -> eaten
    _, r, _, _, info = env.step([1, 0])
    assert info["ate"] == 1 and r > 5


def test_pbrs_closed_loop_nets_zero_when_gamma_one():
    # PBRS is policy-invariant on the DISCOUNTED return; a closed loop nets exactly 0 only at gamma=1.
    from dataclasses import replace
    env = SnakeEnv(cfg=replace(CFG, gamma=1.0), seed=0); env.reset()
    w = env.world
    w.set_chickens([[45.0, 30.0]])
    w.head = np.array([30.0, 30.0]); w.head_uw = w.head.copy()
    env._last_phi = env._phi(); env._last_ids = frozenset(int(i) for i in w.chicken_id)
    total = 0.0
    for p in [np.array([31.0, 30.0]), np.array([30.0, 30.0])]:   # out and back to same phi
        w.head = p; w.head_uw = p.copy()
        total += env._shaping()
    assert abs(total) < 1e-9                 # telescopes exactly at gamma=1


def test_pbrs_zeroes_on_set_change_not_on_nearest_switch():
    # nearest-distance is continuous as the nearest identity switches among a fixed set -> shaping is PAID;
    # only a change to the chicken SET (eat/spawn) zeroes it.
    env = SnakeEnv(seed=0); env.reset()
    w = env.world
    w.obstacle_pos = np.zeros((0, 2)); w.obstacle_r = np.zeros((0,))   # isolate the chicken term
    w.set_chickens([[45.0, 30.0], [15.0, 30.0]])
    w.head = np.array([31.0, 30.0]); w.head_uw = w.head.copy()     # nearest = chicken@45 (d14 < d16)
    env._last_phi = env._phi(); env._last_ids = frozenset(int(i) for i in w.chicken_id)
    w.head = np.array([25.0, 30.0]); w.head_uw = w.head.copy()     # nearest switches to @15, SAME set
    assert env._shaping() != 0.0                                   # continuous shaping paid, not zeroed
    w.chicken_pos = w.chicken_pos[:1]; w.chicken_id = w.chicken_id[:1]   # a chicken removed (eaten)
    assert env._shaping() == 0.0                                   # set changed -> zeroed


def test_obstacle_pbrs_well_and_steer_away():
    # PBRS potential well around obstacles: flat 0 beyond obs_avoid_range, deepest (-weight) at the
    # lethal surface (obstacle_r + head_radius); moving toward costs shaping, moving away pays it.
    env = SnakeEnv(seed=0); env.reset()
    w = env.world
    w.chicken_pos = np.zeros((0, 2)); w.chicken_id = np.zeros((0,), int)   # isolate the obstacle term
    w.obstacle_pos = np.array([[30.0, 30.0]]); w.obstacle_r = np.array([3.0])
    surface = 3.0 + CFG.head_radius                                        # center-dist of lethal contact
    at = lambda x: (w.__setattr__("head", np.array([x, 30.0])), w.__setattr__("head_uw", w.head.copy()))
    at(30.0 + surface + CFG.obs_avoid_range + 5.0)
    assert env._phi_obstacle() == 0.0                                      # far: flat well floor
    at(30.0 + surface)
    assert abs(env._phi_obstacle() + CFG.obs_avoid_weight) < 1e-9          # surface: deepest
    # step-by-step shaping sign (chicken term is 0 with no chickens)
    env._last_ids = frozenset(); env._last_phi = env._phi()
    at(30.0 + surface + 6.0); env._last_phi_obs = env._phi_obstacle()
    at(30.0 + surface + 3.0); assert env._shaping() < 0.0                   # moved TOWARD -> penalty
    at(30.0 + surface + 7.0); assert env._shaping() > 0.0                   # moved AWAY  -> reward


def test_obstacle_death_costs_more_than_other_deaths():
    # a crash into a rock pays reward_death_obstacle (heavier), not the flat reward_death -- this is
    # what flips the "chase a chicken into a rock-gap" gamble negative-EV (Pitfall 16).
    env = SnakeEnv(seed=0); env.reset()
    w = env.world
    w.chicken_pos = np.zeros((0, 2)); w.chicken_id = np.zeros((0,), int)   # no eat/shaping noise
    w.head = np.array([30.0, 30.0]); w.head_uw = np.array([30.0, 30.0]); w.heading = 0.0
    w.path_uw = [np.array([30.0, 30.0])]
    w.obstacle_pos = np.array([[31.5, 30.0]]); w.obstacle_r = np.array([1.5])
    w.obstacle_kind = np.zeros(1, int)
    _, r, term, _, info = env.step([1, 0])                                 # step straight into the rock
    assert term and info["death_cause"] == "obstacle"
    assert r < CFG.reward_death                                            # heavier than the flat -10
    assert abs(r - CFG.reward_death_obstacle) < 0.5                        # ~= the obstacle-specific cost
