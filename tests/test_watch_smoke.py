import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from snake_rl.train import train
from snake_rl.watch import rollout_once
from stable_baselines3 import PPO


def test_rollout_runs(tmp_path):
    model_path = tmp_path / "m.zip"
    train(total_steps=256, n_envs=1, model_path=str(model_path), reset=True, seed=0)
    model = PPO.load(str(model_path), device="cpu")
    out = rollout_once(model, str(tmp_path / "vecnormalize.pkl"), seed=0, max_steps=50)
    assert out["steps"] >= 1
