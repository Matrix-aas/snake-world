"""Watch mode: load a checkpoint and run one env with the pygame renderer (deterministic)."""
import os
import pygame
from stable_baselines3 import PPO
from .config import CFG
from .train import build_vec
from .render import Renderer


def rollout_once(model, norm_path, seed=0, max_steps=CFG.episode_horizon):
    """Headless single-episode rollout with frame-stacking + obs normalization."""
    vec = build_vec(1, seed, training=False, norm_path=norm_path)
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
    vec.close()
    return {"steps": steps, "eaten": eaten, "died": died}


def _world_of(vec):
    """Reach the underlying World through VecNormalize -> VecFrameStack -> DummyVecEnv."""
    return vec.venv.venv.envs[0].unwrapped.world


def run_watch(model_path="models/snake.zip", seed=None, fps=30):
    norm_path = os.path.join(os.path.dirname(model_path) or ".", "vecnormalize.pkl")
    model = PPO.load(model_path, device="cpu")
    vec = build_vec(1, seed or 0, training=False, norm_path=norm_path)
    renderer = Renderer()
    obs = vec.reset()
    clock = pygame.time.Clock()
    paused = False
    running = True
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
        if not paused:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = vec.step(action)
            if done[0]:
                obs = vec.reset()
        renderer.draw(_world_of(vec))
        clock.tick(fps)
    renderer.close()
    vec.close()
