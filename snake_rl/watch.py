"""Watch mode: load a checkpoint and run one env with the pygame renderer (deterministic)."""
import os
import sys
import pygame
from stable_baselines3 import PPO
from .config import CFG
from .train import build_vec
from .render import Renderer


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
    norm_path = _norm_path_for(model_path)
    _require_files(model_path, norm_path)
    model = PPO.load(model_path, device="cpu")
    for ep in range(episodes):
        out = rollout_once(model, norm_path, seed=(seed or 0) + ep)
        print(f"episode {ep}: steps={out['steps']} eaten={out['eaten']} died={out['died']}")


def run_watch(model_path="models/snake.zip", seed=None, fps=30):
    norm_path = _norm_path_for(model_path)
    _require_files(model_path, norm_path)
    model = PPO.load(model_path, device="cpu")
    vec = build_vec(1, seed or 0, training=False, norm_path=norm_path)
    renderer = Renderer()
    clock = pygame.time.Clock()
    paused = False
    running = True
    try:
        obs = vec.reset()
        while running:
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
                    elif e.key == pygame.K_n:
                        obs = vec.reset()
            renderer.draw(_world_of(vec))            # draw current state BEFORE stepping
            if not paused:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _ = vec.step(action)   # VecEnv autoresets on done (no manual reset)
                if done[0]:
                    pygame.time.wait(400)            # brief pause so a death reads as a fresh world
            clock.tick(fps)
    finally:
        renderer.close()
        vec.close()
