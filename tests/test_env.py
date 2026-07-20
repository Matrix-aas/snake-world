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
    env._last_phi = env._phi(); env._last_nearest_id = w.nearest_chicken_id()
    total = 0.0
    for p in [np.array([31.0, 30.0]), np.array([30.0, 30.0])]:   # out and back to same phi
        w.head = p; w.head_uw = p.copy()
        total += env._shaping()
    assert abs(total) < 1e-9                 # telescopes exactly at gamma=1


def test_pbrs_zeroes_on_target_switch():
    # when the nearest chicken (by id) changes, shaping must pay 0 (no spurious jump between targets)
    env = SnakeEnv(seed=0); env.reset()
    w = env.world
    w.set_chickens([[45.0, 30.0], [15.0, 30.0]])
    w.head = np.array([31.0, 30.0]); w.head_uw = w.head.copy()     # nearest = chicken@45 (d14 < d16)
    env._last_phi = env._phi(); env._last_nearest_id = w.nearest_chicken_id()
    id_before = w.nearest_chicken_id()
    w.head = np.array([25.0, 30.0]); w.head_uw = w.head.copy()     # nearest switches to chicken@15 (d10)
    f = env._shaping()
    assert w.nearest_chicken_id() != id_before   # sanity: target identity switched
    assert f == 0.0                              # shaping zeroed on the switch step
