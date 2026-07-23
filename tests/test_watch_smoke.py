import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
from snake_rl.config import CFG
from snake_rl.train import train
from snake_rl.watch import (rollout_once, run_headless, _step_world, _reseed_floor, _load_model,
                            _new_ecosystem, _new_camera, _cycle_follow, _camera_view, DEATH_LINGER_S,
                            _update_life_stats)
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


def test_reseed_floor_lays_arrival_eggs_that_hatch_to_the_floor():
    # Screensaver guarantee (Goal 1): wipe out the whole population; _step_world (via _reseed_floor)
    # now tops the world back up with GUARANTEED ARRIVAL EGGS (owner -1) rather than popping snakes in.
    # Invariant every step: live snakes PLUS pending arrival eggs covers the floor; and those eggs
    # really do hatch into live snakes (so it's not just eggs forever).
    w = generate_world(CFG, seed=9, size=(140.0, 140.0), n_snakes=CFG.n_start_min)
    ctrl = OpponentController(CFG)          # unsynced -> straight-line actor, deterministic
    for s in w.snakes:
        s.alive = False
    assert sum(1 for s in w.snakes if s.alive) == 0
    ever_live_at_floor = False
    for _ in range(CFG.egg_timer * 2):
        _step_world(w, ctrl)
        live = sum(1 for s in w.snakes if s.alive)
        pending = int((w.eggs["owner"][:, 0] < 0).sum()) if len(w.eggs["owner"]) else 0
        assert live + pending >= CFG.n_start_min          # arrivals guaranteed (live or still in-egg)
        ever_live_at_floor = ever_live_at_floor or live >= CFG.n_start_min
    assert ever_live_at_floor                              # the arrival eggs really hatch to the floor


def test_reseed_floor_is_a_noop_above_the_floor():
    # Above the floor, _reseed_floor must not add anyone -- natural population dynamics dominate.
    w = generate_world(CFG, seed=9, size=(140.0, 140.0), n_snakes=CFG.n_start_min)
    ctrl = OpponentController(CFG)
    n_before = len(w.snakes)
    _reseed_floor(w, ctrl)
    assert len(w.snakes) == n_before


def test_run_headless_returns_ecosystem_metrics_dict(tmp_path):
    model_path = tmp_path / "m.zip"
    train(total_steps=256, n_envs=1, model_path=str(model_path), reset=True, seed=0)
    metrics = run_headless(str(model_path), seed=0, episodes=1, max_steps=120)
    assert set(metrics["deaths"]) == {"snake", "starve"}
    assert len(metrics["population"]) == 120
    for key in ("births", "kills", "starvations", "catch_rate", "dash_usage", "steps"):
        assert key in metrics


def test_step_world_advances_each_ring_once_in_no_ego_world(tmp_path):
    # Regression: in a no_ego viewer world, world.step drives EVERY live snake via opponent_fn, so
    # _step_world must NOT also call controller.act for a positional slot -- that would roll the first
    # live snake's frame ring TWICE (two identical newest frames -> corrupt velocity signal + skewed
    # run_headless metrics). After a single step a fresh ring must hold exactly ONE populated frame.
    model_path = tmp_path / "m.zip"
    train(total_steps=256, n_envs=1, model_path=str(model_path), reset=True, seed=0)
    _model, ctrl, _ = _new_ecosystem(str(model_path), seed=0)   # a SYNCED controller (rings actually roll)
    w = generate_world(CFG, seed=11, size=(140.0, 140.0), n_snakes=3)   # live snakes to drive
    w.no_ego = True
    first = [s for s in w.snakes if s.alive][0]
    _step_world(w, ctrl)
    ring = ctrl.rings[first.id]
    populated = sum(bool(np.any(row)) for row in ring)
    assert populated == 1                                       # a double-roll would populate 2 frames


def _bodies(w):
    return {s.id: w._body_render_path_uw(s) for s in w.snakes if s.alive}


def test_camera_follow_cycle_and_death_linger():
    # Phase B increment 1: follow tracks a snake; [/] cycle stable-by-id; a followed snake's death
    # holds the camera at its death spot for DEATH_LINGER_S, then advances to the next live snake.
    w = generate_world(CFG, seed=7, size=(140.0, 140.0), n_snakes=3)      # 3 live snakes
    ids = sorted(s.id for s in w.snakes if s.alive)
    cam = _new_camera(w)
    assert cam["follow_id"] == ids[0] and cam["mode"] == "follow"
    _, z = _camera_view(cam, w, _bodies(w), now=1.0)
    assert z == cam["zoom"]
    _cycle_follow(cam, w, +1); assert cam["follow_id"] == ids[1]          # next
    _cycle_follow(cam, w, -1); assert cam["follow_id"] == ids[0]          # prev (wraps)
    _camera_view(cam, w, _bodies(w), now=10.0)                            # record last_head while alive
    last = cam["last_head"].copy()
    next(s for s in w.snakes if s.id == cam["follow_id"]).alive = False   # kill the followed snake
    c1, _ = _camera_view(cam, w, _bodies(w), now=10.2)                    # within linger -> hold
    assert np.allclose(c1, last) and cam["follow_id"] == ids[0]
    _camera_view(cam, w, _bodies(w), now=100.0)                           # linger elapsed -> advance
    assert cam["follow_id"] in (ids[1], ids[2])


def test_camera_overview_when_no_live_snakes():
    # All-eggs viewer world (0 live): follow has no target -> whole-world overview at zoom 1, centered.
    w = generate_world(CFG, seed=2, size=(120.0, 120.0), n_snakes=3, arrivals=True, ego_live=False)
    cam = _new_camera(w)
    assert cam["follow_id"] is None
    center, z = _camera_view(cam, w, {}, now=0.0)
    assert z == 1.0 and np.allclose(center, np.asarray(w.size, float) / 2)


def test_life_stats_offspring_exact_and_kills_nearest_rival():
    # Phase B increment 3: offspring credits BOTH co-owners of every real hatch (exact); a cut-off
    # ('snake') death credits the nearest pre-step rival (approx, since deaths_detailed lacks a killer).
    size = np.array([100.0, 100.0])
    stats = {}
    out = {"hatched_owners": [frozenset((4, 7)), frozenset((7, 9))],
           "deaths_detailed": [(2, "snake"), (5, "starve")]}
    pre = {2: np.array([10.0, 10.0]),          # victim
           3: np.array([11.0, 10.0]),          # nearest rival -> credited the kill
           8: np.array([90.0, 90.0])}          # far rival
    _update_life_stats(stats, out, pre, size)
    assert stats[7]["offspring"] == 2 and stats[4]["offspring"] == 1 and stats[9]["offspring"] == 1
    assert stats[3]["kills"] == 1              # nearest to the victim
    assert 8 not in stats                      # far rival not credited
    assert 5 not in stats                      # a 'starve' death is not a kill


def test_load_model_rejects_dim_mismatched_model(tmp_path):
    import gymnasium as gym
    from gymnasium import spaces

    class TinyEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Box(-1.0, 1.0, (10,), dtype=np.float32)
            self.action_space = spaces.MultiDiscrete([4, 3, 2])

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
