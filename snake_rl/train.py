"""PPO training: SubprocVecEnv of random worlds + FrameStack + VecNormalize, on CPU."""
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from .config import CFG
from .env import SnakeEnv


def make_env(rank, seed):
    def _thunk():
        return SnakeEnv(seed=seed + rank)
    return _thunk


def build_vec(n_envs, seed, training=True, norm_path=None):
    cls = DummyVecEnv if n_envs == 1 else SubprocVecEnv
    vec = cls([make_env(i, seed) for i in range(n_envs)])
    vec = VecFrameStack(vec, CFG.frame_stack)
    if norm_path and os.path.exists(norm_path):
        vec = VecNormalize.load(norm_path, vec)
        vec.training = training
        vec.norm_reward = training
    else:
        vec = VecNormalize(vec, norm_obs=True, norm_reward=training, clip_obs=10.0)
    return vec


class EpisodeStatsCallback(BaseCallback):
    def __init__(self, every=10000):
        super().__init__(); self.every = every; self.eaten = 0; self.deaths = 0

    def _on_step(self):
        for info in self.locals["infos"]:
            self.eaten += info.get("ate", 0)
            if info.get("alive") is False:
                self.deaths += 1
        if self.num_timesteps % self.every < self.training_env.num_envs:
            print(f"[{self.num_timesteps}] eaten={self.eaten} deaths={self.deaths}")
        return True


def train(total_steps, n_envs=8, model_path="models/snake.zip", reset=False, seed=0):
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    norm_path = os.path.join(os.path.dirname(model_path) or ".", "vecnormalize.pkl")
    vec = build_vec(n_envs, seed, training=True,
                    norm_path=None if reset else norm_path)
    if (not reset) and os.path.exists(model_path):
        model = PPO.load(model_path, env=vec, device="cpu")
    else:
        model = PPO("MlpPolicy", vec, device="cpu", verbose=1,
                    n_steps=1024, batch_size=256, gamma=CFG.gamma, ent_coef=0.01)
    model.learn(total_timesteps=total_steps, callback=EpisodeStatsCallback())
    model.save(model_path)
    vec.save(norm_path)
    vec.close()
