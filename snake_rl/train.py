"""PPO training: SubprocVecEnv of random worlds + FrameStack + VecNormalize, on CPU."""
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from .config import CFG
from .env import SnakeEnv


def _linear_lr(initial):
    """SB3 schedule: learning rate decays linearly from `initial` to 0 over the run."""
    def f(progress_remaining):
        return progress_remaining * initial
    return f


def make_env(rank, seed, world_size=None, dash_penalty=None, easy_stamina=False):
    def _thunk():
        return Monitor(SnakeEnv(seed=seed + rank, world_size=world_size,
                                dash_penalty=dash_penalty, easy_stamina=easy_stamina))
    return _thunk


def build_vec(n_envs, seed, training=True, norm_path=None, world_size=None, dash_penalty=None, easy_stamina=False):
    cls = DummyVecEnv if n_envs == 1 else SubprocVecEnv
    vec = cls([make_env(i, seed, world_size, dash_penalty, easy_stamina) for i in range(n_envs)])
    vec = VecFrameStack(vec, CFG.frame_stack)
    if norm_path and os.path.exists(norm_path):
        vec = VecNormalize.load(norm_path, vec)
    else:
        vec = VecNormalize(vec, norm_obs=True, norm_reward=training, clip_obs=10.0)
    vec.training = training
    vec.norm_reward = training
    return vec


class EpisodeStatsCallback(BaseCallback):
    """Log chickens eaten and deaths PER WINDOW (delta, not lifetime total)."""
    def __init__(self, every=10000):
        super().__init__(); self.every = every; self.eaten = 0; self.deaths = 0; self._last = 0

    def _on_step(self):
        for info in self.locals["infos"]:
            self.eaten += info.get("ate", 0)
            if info.get("alive") is False:
                self.deaths += 1
        if self.num_timesteps - self._last >= self.every:
            self.logger.record("snake/eaten_per_window", self.eaten)
            self.logger.record("snake/deaths_per_window", self.deaths)
            print(f"[{self.num_timesteps}] eaten+={self.eaten} deaths+={self.deaths}")
            self.eaten = self.deaths = 0
            self._last = self.num_timesteps
        return True


class SaveEvery(BaseCallback):
    """Overwrite the fixed model + VecNormalize paths periodically so watch always has a fresh checkpoint."""
    def __init__(self, save_freq_steps, model_path, norm_path, n_envs):
        super().__init__()
        self.every = max(save_freq_steps // n_envs, 1)   # _on_step fires once per env-batch
        self.model_path = model_path; self.norm_path = norm_path

    def _on_step(self):
        if self.n_calls % self.every == 0:
            self.model.save(self.model_path)
            self.training_env.save(self.norm_path)
        return True


def train(total_steps, n_envs=8, model_path="models/snake.zip", reset=False, seed=0,
          save_every=50000, log_every=10000, dash_penalty=None, easy_stamina=False):
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    norm_path = os.path.join(os.path.dirname(model_path) or ".", "vecnormalize.pkl")
    resuming = (not reset) and os.path.exists(model_path)
    vec = build_vec(n_envs, seed, training=True, norm_path=None if reset else norm_path,
                    dash_penalty=dash_penalty, easy_stamina=easy_stamina)
    if resuming:
        model = PPO.load(model_path, env=vec, device="cpu")
    else:
        model = PPO("MlpPolicy", vec, device="cpu", verbose=1, seed=seed,
                    n_steps=1024, batch_size=512, n_epochs=10,
                    learning_rate=_linear_lr(3e-4),   # decay to 0 over the run
                    gamma=CFG.gamma, gae_lambda=0.95, clip_range=0.2,
                    ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, target_kl=0.03,
                    policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])))
    callbacks = [EpisodeStatsCallback(log_every),
                 SaveEvery(save_every, model_path, norm_path, n_envs)]
    try:
        model.learn(total_timesteps=total_steps, callback=callbacks,
                    reset_num_timesteps=not resuming)
    finally:
        model.save(model_path)          # always persist progress, even on Ctrl-C / crash
        vec.save(norm_path)
        vec.close()
