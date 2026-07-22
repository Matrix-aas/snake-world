"""In-env opponents driven by a policy snapshot, using the EXACT VecFrameStack+VecNormalize
preprocessing the SB3 learner sees at train time (spike 0a, PROVEN). The ego is the learner;
every opponent runs its own frame ring through this controller.

Preprocessing parity (do NOT improvise): stack 4 frames (newest LAST) THEN normalize over the
stacked vector -- never per-frame -- then clip. On a snake (re)spawn or full env reset the ring is
ZEROED, not rolled (no stale frame leaks). mean/var/clip_obs/epsilon come from the loaded
VecNormalize and are pushed in by `sync` once per rollout.
"""
import numpy as np
import torch
from gymnasium.spaces import Box, MultiDiscrete
from stable_baselines3.common.policies import ActorCriticPolicy
from .config import CFG
from .sensors import observe, OBS_DIM


class OpponentController:
    def __init__(self, cfg=CFG):
        self.cfg = cfg
        self.n_stack = cfg.frame_stack
        # policy skeleton [I-6]: consumes the ALREADY-normalized (OBS_DIM*n_stack,) vector, exactly
        # what model.policy sees after VecFrameStack+VecNormalize at train time (no internal norm).
        self.policy = ActorCriticPolicy(
            observation_space=Box(-np.inf, np.inf, (OBS_DIM * self.n_stack,), np.float32),
            action_space=MultiDiscrete([4, 3, 2]),   # speed x steer x dash (must match SnakeEnv's action net)
            lr_schedule=lambda _: 0.0,
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
        )
        self.mean = self.var = None
        self.clip_obs = 10.0
        self.epsilon = 1e-8
        self._synced = False
        self.rings = {}                          # snake_id -> np.ndarray(n_stack, OBS_DIM), newest at [-1]

    def sync(self, state_dict, obs_rms, clip_obs, epsilon):
        """Load a policy snapshot + the VecNormalize stats. Idempotent, pure."""
        self.policy.load_state_dict({k: torch.as_tensor(v) for k, v in state_dict.items()})
        self.mean = np.asarray(obs_rms.mean, np.float32)
        self.var = np.asarray(obs_rms.var, np.float32)
        self.clip_obs = float(clip_obs)
        self.epsilon = float(epsilon)
        self._synced = True

    def _preprocess(self, ring):
        stack = ring.reshape(-1)                 # (OBS_DIM*n_stack,), newest frame LAST
        return np.clip((stack - self.mean) / np.sqrt(self.var + self.epsilon),
                       -self.clip_obs, self.clip_obs).astype(np.float32)

    def act(self, world, snake):
        """(speed, steer, dash) for one opponent from the PRE-STEP world. ⅓-cruise straight until sync."""
        if not self._synced:
            return (1, 1, 0)                     # [I-1 bootstrap]: a fresh env is valid pre-sync
        ring = self.rings.get(snake.id)
        if ring is None:
            ring = np.zeros((self.n_stack, OBS_DIM), np.float32)
        ring = np.roll(ring, -1, axis=0)         # drop oldest; open the newest slot at [-1]
        ring[-1] = observe(world, snake)
        self.rings[snake.id] = ring
        a, _ = self.policy.predict(self._preprocess(ring)[None], deterministic=False)
        return (int(a[0][0]), int(a[0][1]), int(a[0][2]))

    def reset_snake(self, snake_id):
        """Drop a snake's ring (call on hatch + death) -> re-created zeroed on its next act."""
        self.rings.pop(snake_id, None)

    def reset_all(self):
        """Drop every ring (call on env reset) [I-1]."""
        self.rings.clear()
