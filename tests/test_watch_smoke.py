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
from snake_rl.world import wrap, torus_dist
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


def _settle(cam, w, r, frames=40, now=0.0, dt=0.1):
    out = (None, None, None)
    for _ in range(frames):
        out = _camera_view(cam, w, _bodies(w), r, now=now, dt=dt)
    return out


def test_camera_follow_cycle_adaptive_zoom_ease_and_linger():
    # Polish #1/#4: with no eggs the camera adopts a live snake, EASES toward its head (framerate-
    # independent), auto-frames at a comfortable zoom (< the old 3.0 default), and on death holds at
    # the death spot for DEATH_LINGER_S (keeping the fallen snake for its panel) then advances.
    from snake_rl.render import Renderer
    w = generate_world(CFG, seed=7, size=(150.0, 150.0), n_snakes=3)      # live, NO eggs
    ids = sorted(s.id for s in w.snakes if s.alive)
    r = Renderer(scale=4); r.draw(w)                                       # sets cw/base for zoom_for_span
    cam = _new_camera(w)
    assert cam["follow_id"] is None and cam["watch_egg"] is None           # no eggs -> will adopt a live snake
    _camera_view(cam, w, _bodies(w), r, now=0.0, dt=0.1)
    assert cam["follow_id"] == ids[0]                                      # adopted the lowest-id live snake
    head = np.asarray(wrap(next(s for s in w.snakes if s.id == ids[0]).head_uw, w.size), float)
    c, z, fallen = _settle(cam, w, r)
    assert float(torus_dist(c[None], head, w.size)[0]) < 1.0               # eased in to the head
    assert 1.0 < z < 3.0 and fallen is None                               # zoomed OUT vs the old 3.0
    _cycle_follow(cam, w, +1); assert cam["follow_id"] == ids[1]
    _camera_view(cam, w, _bodies(w), r, now=1.0, dt=0.1)                  # record follow_snake=ids[1]
    victim = next(s for s in w.snakes if s.id == ids[1]); victim.alive = False
    _, _, f1 = _camera_view(cam, w, _bodies(w), r, now=10.0, dt=0.1)      # death -> linger starts
    assert f1 is victim and cam["follow_id"] == ids[1]                    # panel still has the fallen snake
    _, _, f2 = _camera_view(cam, w, _bodies(w), r, now=10.0 + DEATH_LINGER_S - 0.5, dt=0.1)
    assert f2 is victim                                                    # still lingering
    _camera_view(cam, w, _bodies(w), r, now=10.0 + DEATH_LINGER_S + 0.5, dt=0.1)
    assert cam["follow_id"] in (ids[0], ids[2])                           # linger over -> advanced
    r.close()


def test_run_watch_inner_loop_integrates_camera_ground_and_fallen():
    # Integration smoke: drive run_watch's inner loop by hand (no display event loop) -- step the
    # world, ease the camera, and draw with inspector+rings across an egg-opening -> hatch -> follow,
    # so _draw_ground, _camera_view and draw(fallen=) all compose over an evolving camera without error.
    from snake_rl.render import Renderer
    from snake_rl.watch import (_snake_snap, _chicken_snap, _interp_bodies, _interp_chickens)
    w = generate_world(CFG, seed=5, size=(140.0, 120.0), n_snakes=3, arrivals=True, ego_live=False)
    ctrl = OpponentController(CFG)                              # unsynced straight-line actor
    r = Renderer(scale=3, show_inspector=True, show_rings=True)
    cam = _new_camera(w)
    prev = cur = (_snake_snap(w), _chicken_snap(w))
    for i in range(40):
        _step_world(w, ctrl)
        prev, cur = cur, (_snake_snap(w), _chicken_snap(w))
        bodies = _interp_bodies(prev[0], cur[0], 0.5)
        cpos, cdir = _interp_chickens(prev[1], cur[1], 0.5, w.size)
        c, z, fallen = _camera_view(cam, w, bodies, r, now=i * 0.1, dt=0.1)
        r.draw(w, bodies, cpos, cdir, follow_id=cam["follow_id"], cam_center=c, zoom=z, fallen=fallen)
    r.close()


def test_camera_egg_opening_and_hatch_handoff():
    # Polish #2: opening frames the soonest egg (not overview), then hands off follow to the hatchling
    # that appears at the egg once it's gone.
    from snake_rl.render import Renderer
    from snake_rl.world import Snake
    w = generate_world(CFG, seed=2, size=(120.0, 120.0), n_snakes=3, arrivals=True, ego_live=False)
    assert len(w.eggs["pos"]) >= 1 and sum(1 for s in w.snakes if s.alive) == 0   # all-eggs start
    r = Renderer(scale=4); r.draw(w)
    cam = _new_camera(w)
    assert cam["watch_egg"] is not None and cam["follow_id"] is None      # opening watches an egg
    egg = cam["watch_egg"].copy()
    _camera_view(cam, w, {}, r, now=0.0, dt=0.1)
    assert cam["follow_id"] is None                                       # still framing the egg
    pos = egg.copy()                                                       # hatch: egg gone, snake at egg pos
    w.snakes.append(Snake(head_uw=pos.copy(), head=wrap(pos, w.size), heading=0.0, path_uw=[pos.copy()],
                          target_length=CFG.start_length, stamina=CFG.s_max, energy=CFG.energy_max,
                          _prev_head_uw=pos.copy(), id=999, color_seed=999, lineage=42))
    w.eggs["pos"] = np.zeros((0, 2))                                       # the watched egg is gone
    _camera_view(cam, w, {}, r, now=1.0, dt=0.1)
    assert cam["follow_id"] == 999 and cam["watch_egg"] is None           # handed off to the hatchling
    r.close()


def test_camera_overview_when_no_eggs_and_no_snakes():
    # Only when there are neither eggs nor live snakes: overview eases toward world center at zoom 1.
    from snake_rl.render import Renderer
    w = generate_world(CFG, seed=7, size=(120.0, 120.0), n_snakes=3)      # live, no eggs
    for s in w.snakes:
        s.alive = False
    w.eggs["pos"] = np.zeros((0, 2))
    r = Renderer(scale=4); r.draw(w)
    cam = _new_camera(w)
    assert cam["watch_egg"] is None
    c, z, fallen = _settle(cam, w, r, frames=80)
    assert np.allclose(c, np.asarray(w.size, float) / 2, atol=1.0) and abs(z - 1.0) < 0.1
    assert cam["follow_id"] is None and fallen is None
    r.close()


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
