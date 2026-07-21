import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
from snake_rl.config import CFG
from snake_rl.train import train
from snake_rl.watch import rollout_once, run_headless, _step_world, _load_model
from snake_rl.selfplay import OpponentController
from snake_rl.worldgen import generate_world
from stable_baselines3 import PPO


def test_rollout_runs(tmp_path):
    model_path = tmp_path / "m.zip"
    train(total_steps=256, n_envs=1, model_path=str(model_path), reset=True, seed=0)
    model = PPO.load(str(model_path), device="cpu")
    out = rollout_once(model, str(tmp_path / "vecnormalize.pkl"), seed=0, max_steps=50)
    assert out["steps"] >= 1


def test_persistent_world_does_not_reset_on_non_ego_death():
    # Drive the SAME world object through several steps via _step_world (the exact helper
    # run_headless/run_watch use), forcing a non-ego death, and keep stepping the same world
    # afterward -- proves the persistent world is never regenerated on a death.
    w = generate_world(CFG, seed=7, size=(140.0, 140.0), n_snakes=3)
    ctrl = OpponentController(CFG)     # unsynced -> every snake acts straight (1, 0), deterministic
    victim = w.snakes[1]                            # capture the object -- a death prunes it out of
    start_pop = sum(1 for s in w.snakes if s.alive)  # w.snakes, so re-indexing [1] after would be wrong
    victim.energy = CFG.energy_decay / 2            # starves on the very next step (non-ego)
    _step_world(w, ctrl)
    assert victim.alive is False and victim.death_cause == "starve"
    alive_after = sum(1 for s in w.snakes if s.alive)
    assert alive_after < start_pop                  # population dropped below the start count...
    for _ in range(5):
        _step_world(w, ctrl)                        # ...and the SAME world keeps running (no reset)
    assert victim.alive is False                    # still dead, never respawned by a reset
    assert victim not in w.snakes                   # a dead non-ego opponent is pruned, not reset


def test_run_headless_returns_ecosystem_metrics_dict(tmp_path):
    model_path = tmp_path / "m.zip"
    train(total_steps=256, n_envs=1, model_path=str(model_path), reset=True, seed=0)
    metrics = run_headless(str(model_path), seed=0, episodes=1, max_steps=120)
    assert set(metrics["deaths"]) == {"obstacle", "self", "snake", "starve"}
    assert len(metrics["population"]) == 120
    for key in ("births", "kills", "starvations", "catch_rate", "dash_usage", "steps"):
        assert key in metrics


def test_load_model_rejects_dim_mismatched_model(tmp_path):
    import gymnasium as gym
    from gymnasium import spaces

    class TinyEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Box(-1.0, 1.0, (10,), dtype=np.float32)
            self.action_space = spaces.MultiDiscrete([3, 2])

        def reset(self, *, seed=None, options=None):
            return self.observation_space.sample(), {}

        def step(self, action):
            return self.observation_space.sample(), 0.0, False, False, {}

    model = PPO("MlpPolicy", TinyEnv(), device="cpu", n_steps=8, batch_size=8, n_epochs=1)
    model.learn(8)
    path = tmp_path / "bad_dim.zip"
    model.save(str(path))
    try:
        _load_model(str(path))
        assert False, "expected a ValueError for a dim-mismatched model"
    except ValueError:
        pass
