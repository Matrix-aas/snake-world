import numpy as np
from types import SimpleNamespace
from snake_rl.config import CFG
from snake_rl.sensors import OBS_DIM
from snake_rl.selfplay import OpponentController
from snake_rl.worldgen import generate_world
from snake_rl.world import torus_dist


def _numpy_state_dict(ctrl):
    return {k: v.detach().cpu().numpy() for k, v in ctrl.policy.state_dict().items()}


def test_preprocess_matches_0a_recipe():
    ctrl = OpponentController()
    rng = np.random.default_rng(0)
    D = OBS_DIM * CFG.frame_stack
    mean = rng.normal(size=D).astype(np.float32)
    var = rng.uniform(0.1, 2.0, size=D).astype(np.float32)
    ctrl.sync(_numpy_state_dict(ctrl), SimpleNamespace(mean=mean, var=var),
              clip_obs=10.0, epsilon=1e-8)
    ring = rng.normal(size=(CFG.frame_stack, OBS_DIM)).astype(np.float32)
    stack = ring.reshape(-1)
    expected = np.clip((stack - mean) / np.sqrt(var + 1e-8), -10.0, 10.0)
    np.testing.assert_allclose(ctrl._preprocess(ring), expected, rtol=1e-6, atol=1e-6)
    # newest frame lives in the LAST OBS_DIM slots of the stacked vector
    assert np.array_equal(stack[-OBS_DIM:], ring[-1])


def test_reset_all_and_reset_snake_zero_rings():
    w = generate_world(CFG, seed=3, size=(140.0, 140.0), n_snakes=2)
    s = w.snakes[1]
    ctrl = OpponentController()
    D = OBS_DIM * CFG.frame_stack
    ctrl.sync(_numpy_state_dict(ctrl),
              SimpleNamespace(mean=np.zeros(D, np.float32), var=np.ones(D, np.float32)), 10.0, 1e-8)
    for _ in range(CFG.frame_stack):
        ctrl.act(w, s)
    assert np.any(ctrl.rings[s.id] != 0)             # ring filled after enough acts
    ctrl.reset_snake(s.id)
    assert s.id not in ctrl.rings                     # dropped -> re-created zeroed on next act
    ctrl.act(w, s)
    ring = ctrl.rings[s.id]
    assert np.all(ring[:-1] == 0) and np.any(ring[-1] != 0)   # ONLY the newest slot written
    ctrl.reset_all()
    assert ctrl.rings == {}


def test_bootstrap_acts_straight_before_sync():
    w = generate_world(CFG, seed=4, size=(140.0, 140.0), n_snakes=2)
    ctrl = OpponentController()
    assert ctrl.act(w, w.snakes[1]) == (1, 0)
    assert ctrl.rings == {}                           # pre-sync must not touch rings


def test_env_spawns_multiple_snakes():
    from snake_rl.env import SnakeEnv
    env = SnakeEnv(seed=0)
    for _ in range(6):
        env.reset()
        n = len(env.world.snakes)
        assert CFG.n_start_min <= n <= CFG.n_start_max


def test_opponents_move_and_reward_finite():
    from snake_rl.env import SnakeEnv
    env = SnakeEnv(seed=1)
    env.reset()
    before = np.array([s.head_uw.copy() for s in env.world.snakes[1:]])
    moved = False
    for _ in range(40):
        _, r, term, trunc, _ = env.step([1, 0])
        assert np.isfinite(r)
        if term or trunc:
            break
    after = np.array([s.head_uw.copy() for s in env.world.snakes[1:len(before) + 1]])
    if len(before) and len(after):
        moved = bool(np.any(np.linalg.norm(after - before, axis=1) > 1e-6))
    assert moved


def test_repro_reward_only_on_ego_hatch():
    from snake_rl.env import SnakeEnv

    def base_env():
        env = SnakeEnv(seed=5); env.reset(); w = env.world
        w.set_chickens([])                    # resets ALL parallel chicken arrays (incl. FSM state) consistently
        w.obstacle_pos = np.zeros((0, 2)); w.obstacle_r = np.zeros(0); w.obstacle_kind = np.zeros(0, int)
        env._last_phi = env._phi(); env._last_ids = frozenset()
        return env, w

    def far_point(w):                                 # a spot no head can reach/eat this step
        heads = np.array([s.head for s in w.snakes])
        best, bd = None, -1.0
        for gx in np.linspace(0, w.size[0], 15):
            for gy in np.linspace(0, w.size[1], 15):
                p = np.array([gx, gy]); d = float(torus_dist(heads, p, w.size).min())
                if d > bd:
                    bd, best = d, p
        return best

    def run_hatch(owner):
        env, w = base_env()
        pos = far_point(w)
        w.eggs = {"pos": pos[None].copy(), "timer": np.array([1.0]), "owner": np.array([owner])}
        return env.step([1, 0])[1]

    r_ego = run_hatch([0, 1])                          # ego (id 0) owned egg hatches -> pays reward_repro
    r_non = run_hatch([1, 2])                          # non-ego egg hatches -> pays nothing
    assert abs((r_ego - r_non) - CFG.reward_repro) < 1e-6

    # an ego-owned egg RAIDED (eaten by an opponent, never hatches) pays nothing == no egg at all
    env, w = base_env()
    w.eggs = {"pos": w.snakes[1].head.copy()[None],   # sits on opponent 1 -> foreign -> eaten next step
              "timer": np.array([45.0]), "owner": np.array([[0, 99]])}
    r_raided = env.step([1, 0])[1]
    env, _ = base_env()
    r_noegg = env.step([1, 0])[1]
    assert abs(r_raided - r_noegg) < 1e-6


def test_set_opponent_policy_syncs_and_runs():
    from snake_rl.env import SnakeEnv
    env = SnakeEnv(seed=2); env.reset()
    sd = _numpy_state_dict(env._opp)
    D = OBS_DIM * CFG.frame_stack
    env.set_opponent_policy(sd, SimpleNamespace(mean=np.zeros(D, np.float32),
                                                var=np.ones(D, np.float32)), 10.0, 1e-8)
    assert env._opp._synced
    for _ in range(20):
        _, r, term, trunc, _ = env.step([1, 0])
        assert np.isfinite(r)
        if term or trunc:
            break


def test_check_env_multisnake():
    from gymnasium.utils.env_checker import check_env
    from snake_rl.env import SnakeEnv
    check_env(SnakeEnv(seed=0), skip_render_check=True)
