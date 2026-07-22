import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from snake_rl.train import build_vec, train
from snake_rl.sensors import OBS_DIM


def test_build_vec_runs_one_step():
    vec = build_vec(n_envs=1, seed=0)
    obs = vec.reset()
    assert obs.shape[1] == OBS_DIM * 4     # frame_stack=4
    vec.step(vec.action_space.sample()[None])
    vec.close()


def test_train_smoke(tmp_path):
    model = tmp_path / "m.zip"
    train(total_steps=256, n_envs=1, model_path=str(model), reset=True, seed=0)
    assert model.exists()
    assert (tmp_path / "vecnormalize.pkl").exists()


def test_anneal_hardness_curriculum_continuation():
    # A curriculum-CONTINUING resume feeds AnnealHardness the ORIGINAL schedule denominator, so
    # f(preserved_num_timesteps / curriculum_total) picks the ramp up exactly where it left off --
    # NOT a jump to full hardness. Verify the ramp math at the endpoints + a resumed mid-point.
    from snake_rl.train import AnnealHardness
    from snake_rl.config import CFG
    a = AnnealHardness(8_000_000, CFG.hardness_warmup, CFG.hardness_full)
    assert a._hardness(0.30) == 0.0                       # still in warmup (< hardness_warmup) -> easy
    assert a._hardness(0.90) == 1.0                       # past hardness_full -> fully hard
    p = 6_000_000 / 8_000_000                             # a resume at 6M of the 8M schedule
    expected = (p - CFG.hardness_warmup) / (CFG.hardness_full - CFG.hardness_warmup)
    assert abs(a._hardness(p) - expected) < 1e-12
    assert 0.0 < a._hardness(p) < 1.0                     # genuinely mid-ramp, continued (not a jump to 1.0)
