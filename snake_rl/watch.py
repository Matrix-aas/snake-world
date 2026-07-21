"""Watch mode: load a checkpoint and run one env with the pygame renderer, smoothly interpolated."""
import os
import sys
import numpy as np
import pygame
from stable_baselines3 import PPO
from .config import CFG
from .train import build_vec
from .render import Renderer
from .world import wrap, torus_delta


def _norm_path_for(model_path):
    return os.path.join(os.path.dirname(model_path) or ".", "vecnormalize.pkl")


def _require_files(model_path, norm_path):
    if not os.path.exists(model_path):
        sys.exit(f"No model at {model_path} — run `python -m snake_rl train` first.")
    if not os.path.exists(norm_path):
        sys.exit(f"Missing {norm_path} (normalization stats) — retrain or restore it.")


def _world_of(vec):
    """Reach the underlying World through VecNormalize -> VecFrameStack -> DummyVecEnv -> Monitor."""
    return vec.venv.venv.envs[0].unwrapped.world


def _interp_body(prev, cur, f):
    # Blend the head-side prefix (stable between steps); when the snake grows, the extra tail
    # points just snap in far from the head — no visible stutter on the eat step.
    n = min(len(prev), len(cur))
    out = cur.copy()
    out[:n] = prev[:n] + (cur[:n] - prev[:n]) * f
    return out


def _chicken_snap(world):
    return {int(i): (world.chicken_pos[k].copy(), float(world.chicken_dir[k]))
            for k, i in enumerate(world.chicken_id)}


def _interp_chickens(prev, cur, f, size):
    """Blend chicken positions by stable id, taking the nearest image across the torus seam."""
    pos, dirs = [], []
    for cid, (cp, cd) in cur.items():
        if cid in prev:
            pp, pd = prev[cid]
            pos.append(wrap(pp + torus_delta(cp, pp, size) * f, size))
            da = (cd - pd + np.pi) % (2 * np.pi) - np.pi
            dirs.append(pd + da * f)
        else:
            pos.append(cp); dirs.append(cd)
    return (np.array(pos) if pos else np.zeros((0, 2))), np.array(dirs)


def rollout_once(model, norm_path, seed=0, max_steps=CFG.episode_horizon):
    """Headless single-episode rollout with frame-stacking + obs normalization."""
    vec = build_vec(1, seed, training=False, norm_path=norm_path)
    try:
        obs = vec.reset()
        eaten = died = 0
        steps = 0
        for steps in range(1, max_steps + 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, infos = vec.step(action)
            eaten += infos[0].get("ate", 0)
            if done[0]:
                died = 1 if infos[0].get("alive") is False else 0
                break
        return {"steps": steps, "eaten": eaten, "died": died}
    finally:
        vec.close()


def run_headless(model_path="models/snake.zip", seed=None, episodes=5):
    """Behavioral evaluation: hunting rate, deliberate-dash usage, stamina cycling, deaths by cause."""
    norm_path = _norm_path_for(model_path)
    _require_files(model_path, norm_path)
    model = PPO.load(model_path, device="cpu")
    vec = build_vec(1, seed or 0, training=False, norm_path=norm_path)
    try:
        obs = vec.reset()
        eaten = act_dash = steps = 0
        deaths = {"obstacle": 0, "self": 0}
        stam, eplens, cur = [], [], 0
        while len(eplens) < max(1, episodes):
            a, _ = model.predict(obs, deterministic=False)   # stochastic, matching watch's default look
            w = _world_of(vec); prev = w.stamina; stam.append(prev)
            obs, _, done, infos = vec.step(a)                # on done, SB3 autoresets and returns the new obs
            if not done[0]:
                act_dash += int(_world_of(vec).stamina < prev)   # stamina dropped => actually dashed
            eaten += infos[0].get("ate", 0); steps += 1; cur += 1
            if done[0]:
                dc = infos[0].get("death_cause")
                if dc in deaths:
                    deaths[dc] += 1
                eplens.append(cur); cur = 0                  # keep the autoreset obs (no second reset)
        steps = max(1, steps)
        s = np.array(stam)
        print(f"over {steps} steps, {len(eplens)} episodes:")
        print(f"  catch rate:  {eaten / steps * 1000:5.1f} chickens / 1000 steps")
        print(f"  dash usage:  {act_dash / steps * 100:5.0f}% of steps (deliberate bursts, not constant)")
        print(f"  stamina:     mean {s.mean():4.1f}/{CFG.s_max:.0f}   reserve builds: {(s > 10).mean() * 100:.0f}% of time above 10")
        print(f"  episode len: {np.mean(eplens):5.0f} steps mean")
        print(f"  deaths:      obstacle {deaths['obstacle']}, self {deaths['self']}")
    finally:
        vec.close()


def _screen_fit_world_size(short=72.0):
    """World size (in sim units) whose aspect matches the desktop, short side fixed to `short`.
    The net is size-agnostic (egocentric senses), so any size plays fine; `short` keeps density sane."""
    pygame.init()
    info = pygame.display.Info()
    sw, sh = info.current_w, info.current_h
    if sw >= sh:
        return (short * sw / sh, short), (sw, sh)
    return (short, short * sh / sw), (sw, sh)


def run_watch(model_path="models/snake.zip", seed=None, fps=60, sim_hz=10, deterministic=False, fullscreen=True):
    # The sim advances at sim_hz steps/sec; rendering runs at `fps` and interpolates the whole
    # scene (snake body + chickens, seam-aware) between steps for smooth motion. Stochastic by default.
    norm_path = _norm_path_for(model_path)
    _require_files(model_path, norm_path)
    model = PPO.load(model_path, device="cpu")
    world_size = screen_size = None
    if fullscreen:
        world_size, screen_size = _screen_fit_world_size()   # map fills the screen at its aspect
    vec = build_vec(1, seed or 0, training=False, norm_path=norm_path, world_size=world_size)
    renderer = Renderer(fullscreen=fullscreen, screen_size=screen_size)
    clock = pygame.time.Clock()
    paused = False
    running = True

    def snapshot(world):
        return world.body_render_path_uw(), _chicken_snap(world)

    try:
        obs = vec.reset()
        world = _world_of(vec)
        prev_body, prev_ch = cur_body, cur_ch = snapshot(world)
        since = 0.0
        while running:
            frame_dt = clock.tick(fps) / 1000.0
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        running = False
                    elif e.key == pygame.K_SPACE:
                        paused = not paused
                    elif e.key == pygame.K_s:
                        renderer.toggle_sensors()
                    elif e.key == pygame.K_d:
                        deterministic = not deterministic
                    elif e.key in (pygame.K_UP, pygame.K_EQUALS, pygame.K_PLUS):
                        sim_hz = min(60, sim_hz + 2)
                    elif e.key in (pygame.K_DOWN, pygame.K_MINUS):
                        sim_hz = max(2, sim_hz - 2)
                    elif e.key == pygame.K_n:
                        obs = vec.reset(); world = _world_of(vec)
                        prev_body, prev_ch = cur_body, cur_ch = snapshot(world); since = 0.0
            interval = 1.0 / sim_hz
            if not paused:
                since += frame_dt
                while since >= interval:
                    since -= interval
                    action, _ = model.predict(obs, deterministic=deterministic)
                    obs, _, done, infos = vec.step(action)
                    world = _world_of(vec)
                    if not done[0] and infos[0].get("ate", 0):
                        renderer.add_flash(world.head.copy())   # catch effect at the strike
                    prev_body, prev_ch = cur_body, cur_ch
                    cur_body, cur_ch = snapshot(world)
                    if done[0]:                              # autoreset already gave a fresh world
                        prev_body, prev_ch = cur_body, cur_ch; since = 0.0
                        break
            f = 0.0 if paused else min(1.0, since / interval)
            body = _interp_body(prev_body, cur_body, f)
            cpos, cdir = _interp_chickens(prev_ch, cur_ch, f, world.size)
            renderer.draw(world, body, cpos, cdir)
    finally:
        renderer.close()
        vec.close()
