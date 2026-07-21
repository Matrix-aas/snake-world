"""PPO training: SubprocVecEnv of random worlds + FrameStack + VecNormalize, on CPU."""
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from .config import CFG
from .env import SnakeEnv


def make_env(rank, seed, world_size=None, dash_penalty=None, hardness=1.0):
    def _thunk():
        return Monitor(SnakeEnv(seed=seed + rank, world_size=world_size,
                                dash_penalty=dash_penalty, hardness=hardness))
    return _thunk


def build_vec(n_envs, seed, training=True, norm_path=None, world_size=None, dash_penalty=None, hardness=1.0):
    cls = DummyVecEnv if n_envs == 1 else SubprocVecEnv
    vec = cls([make_env(i, seed, world_size, dash_penalty, hardness) for i in range(n_envs)])
    vec = VecFrameStack(vec, CFG.frame_stack)
    if norm_path and os.path.exists(norm_path):
        vec = VecNormalize.load(norm_path, vec)
    else:
        vec = VecNormalize(vec, norm_obs=True, norm_reward=training, clip_obs=10.0)
    vec.training = training
    vec.norm_reward = training
    return vec


class EpisodeStatsCallback(BaseCallback):
    """Log chickens eaten, deaths, and reproduction events PER WINDOW (delta, not lifetime total)."""
    def __init__(self, every=10000):
        super().__init__(); self.every = every
        self.eaten = 0; self.deaths = 0; self.repro = 0; self.hatched = 0; self._last = 0

    def _on_step(self):
        for info in self.locals["infos"]:
            self.eaten += info.get("ate", 0)
            self.repro += info.get("repro_ego", 0)      # ego-owned eggs that hatched -> +reward_repro fired
            self.hatched += info.get("hatched", 0)      # all hatchlings born (any owner)
            if info.get("alive") is False:
                self.deaths += 1
        if self.num_timesteps - self._last >= self.every:
            self.logger.record("snake/eaten_per_window", self.eaten)
            self.logger.record("snake/deaths_per_window", self.deaths)
            self.logger.record("snake/repro_per_window", self.repro)
            self.logger.record("snake/hatched_per_window", self.hatched)
            print(f"[{self.num_timesteps}] eaten+={self.eaten} deaths+={self.deaths} "
                  f"repro+={self.repro} hatched+={self.hatched}")
            self.eaten = self.deaths = self.repro = self.hatched = 0
            self._last = self.num_timesteps
        return True


class AnnealHardness(BaseCallback):
    """Curriculum: keep stamina easy for `warmup` of training (learn to hunt), then ramp the real
    reserve mechanic in linearly by `full`. Smooth annealing avoids the survive-only collapse an
    abrupt easy->hard switch causes."""
    def __init__(self, total_steps, warmup, full, every=16384):
        super().__init__()
        self.total = total_steps; self.warmup = warmup; self.full = full; self.every = every; self._last = -1

    def _hardness(self, p):
        if p <= self.warmup:
            return 0.0
        if p >= self.full:
            return 1.0
        return (p - self.warmup) / (self.full - self.warmup)

    def _on_step(self):
        if self._last < 0 or self.num_timesteps - self._last >= self.every:
            self._last = self.num_timesteps
            h = self._hardness(self.num_timesteps / max(1, self.total))
            self.training_env.env_method("set_hardness", h)
            self.logger.record("snake/hardness", h)
        return True


class SyncOpponentPolicy(BaseCallback):
    """Self-play: each rollout, snapshot the learner's policy + the VecNormalize stats and push them
    into every env's OpponentController, so in-env opponents mirror the current ego brain."""
    def _on_step(self):
        return True

    def _on_rollout_end(self):
        sd = {k: v.detach().cpu().numpy() for k, v in self.model.policy.state_dict().items()}
        vn = self.model.get_vec_normalize_env()
        self.training_env.env_method("set_opponent_policy", sd, vn.obs_rms, vn.clip_obs, vn.epsilon)


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


def train(total_steps, n_envs=16, model_path="models/snake.zip", reset=False, seed=0,
          save_every=50000, log_every=10000, dash_penalty=None):
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    norm_path = os.path.join(os.path.dirname(model_path) or ".", "vecnormalize.pkl")
    resuming = (not reset) and os.path.exists(model_path)
    # fresh run starts easy (hardness 0) and anneals the reserve mechanic in; a resume is assumed post-curriculum
    hardness0 = 1.0 if resuming else 0.0
    vec = build_vec(n_envs, seed, training=True, norm_path=None if reset else norm_path,
                    dash_penalty=dash_penalty, hardness=hardness0)
    if resuming:
        model = PPO.load(model_path, env=vec, device="cpu")
    else:
        model = PPO("MlpPolicy", vec, device="cpu", verbose=1, seed=seed,
                    n_steps=1024, batch_size=256, n_epochs=10,   # 64 minibatches over the 16384 buffer
                    learning_rate=3e-4,   # CONSTANT: task is non-stationary (stamina hardens); a decaying lr
                    gamma=CFG.gamma, gae_lambda=0.95, clip_range=0.2,   # can't adapt to late hardening.
                    ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, target_kl=0.03,  # target_kl = stability guard
                    policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])))
    callbacks = [EpisodeStatsCallback(log_every),
                 SyncOpponentPolicy(),          # self-play: opponents mirror the learner each rollout
                 SaveEvery(save_every, model_path, norm_path, n_envs)]
    if not resuming:
        callbacks.append(AnnealHardness(total_steps, CFG.hardness_warmup, CFG.hardness_full))
    try:
        model.learn(total_timesteps=total_steps, callback=callbacks,
                    reset_num_timesteps=not resuming)
    finally:
        model.save(model_path)          # always persist progress, even on Ctrl-C / crash
        vec.save(norm_path)
        vec.close()
