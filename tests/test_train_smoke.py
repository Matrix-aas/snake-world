import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from snake_rl.train import build_vec, train


def test_build_vec_runs_one_step():
    vec = build_vec(n_envs=1, seed=0)
    obs = vec.reset()
    assert obs.shape[1] == 42 * 4          # frame_stack=4
    vec.step(vec.action_space.sample()[None])
    vec.close()


def test_train_smoke(tmp_path):
    model = tmp_path / "m.zip"
    train(total_steps=256, n_envs=1, model_path=str(model), reset=True, seed=0)
    assert model.exists()
    assert (tmp_path / "vecnormalize.pkl").exists()
