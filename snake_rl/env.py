"""Gymnasium environment: sensors -> obs, MultiDiscrete action, PBRS reward, truncation."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .config import CFG, assert_invariants
from .worldgen import generate_world
from .sensors import observe, OBS_DIM


class SnakeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg=CFG, seed=None, world_size=None, dash_penalty=None):
        super().__init__()
        assert_invariants(cfg)
        self.cfg = cfg
        self._seed = seed
        self._seeded = False
        self._world_size = world_size          # None -> random size per episode; fixed -> e.g. screen-fit
        self._dash_penalty = cfg.dash_penalty if dash_penalty is None else dash_penalty  # curriculum override
        self.action_space = spaces.MultiDiscrete([3, 2])
        # bounded where known: ray dist & one-hots (0..35) and proprio (39..41) in [0,1];
        # smell (36..38): intensity in [0, max_chickens], gradient components in [-max_chickens, max_chickens]
        # (each of <=max_chickens chickens contributes <=1 to intensity and <=1 in magnitude to the gradient)
        m = float(cfg.max_chickens)
        low = np.zeros(OBS_DIM, np.float32)
        high = np.ones(OBS_DIM, np.float32)
        low[36:39] = [0.0, -m, -m]
        high[36:39] = [m, m, m]
        self.observation_space = spaces.Box(low, high, (OBS_DIM,), np.float32)
        self.world = None
        self._last_phi = 0.0
        self._last_ids = frozenset()

    def _phi(self):
        _, d = self.world.nearest_chicken()
        d = min(d, self.cfg.ray_range)
        return -d / self.cfg.ray_range

    def reset(self, *, seed=None, options=None):
        # First seedless reset seeds the stream from the constructor seed; every reset then
        # draws a FRESH world seed -> real domain randomization across episodes (not one fixed world).
        if seed is None and not self._seeded:
            seed = self._seed
        super().reset(seed=seed)
        self._seeded = True
        world_seed = int(self.np_random.integers(0, 2 ** 31 - 1))
        self.world = generate_world(self.cfg, seed=world_seed, size=self._world_size)
        self._last_phi = self._phi()
        self._last_ids = frozenset(int(i) for i in self.world.chicken_id)
        return observe(self.world), {}

    def _shaping(self):
        """PBRS: gamma*phi' - phi. Phi = -dist_to_nearest is CONTINUOUS as the nearest identity
        switches among a fixed set (min of continuous distances), so we pay shaping normally then;
        we only zero it when the chicken SET changes (eat/spawn), where the distance can jump."""
        ids = frozenset(int(i) for i in self.world.chicken_id)
        phi = self._phi()
        if not ids or ids != self._last_ids:
            f = 0.0                                  # set changed (eat/spawn) or none: don't pay the jump
        else:
            f = self.cfg.gamma * phi - self._last_phi
        self._last_phi = phi
        self._last_ids = ids
        return f

    def step(self, action):
        c = self.cfg
        steering, dash = int(action[0]), int(action[1])
        out = self.world.step(steering, dash)
        terminated = out["died"]
        truncated = self.world.steps >= c.episode_horizon
        reward = c.reward_eat * out["ate"]
        if terminated:
            reward += c.reward_death
        else:
            reward += self._shaping()
        # PBRS shaping (>=0 when a chicken is far) partly offsets this while well-fed; that's the
        # standard potential-shaping artifact — hunger ramps the penalty and restores "get moving" pressure.
        hunger = 1.0 - self.world.energy / c.energy_max
        reward -= c.step_penalty * (1.0 + hunger)
        if out["dashed"]:                         # dashing burns energy -> use it only to chase, not constantly
            reward -= self._dash_penalty
        info = {"ate": out["ate"], "alive": self.world.alive, "steps": self.world.steps}
        return observe(self.world), float(reward), terminated, truncated, info
