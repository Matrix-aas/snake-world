"""Gymnasium environment: sensors -> obs, MultiDiscrete action, PBRS reward, truncation."""
import numpy as np
from dataclasses import replace
import gymnasium as gym
from gymnasium import spaces
from .config import CFG, assert_invariants
from .worldgen import generate_world
from .sensors import observe, OBS_DIM


def _make_observation_space(cfg):
    """Bounds for the 75-float layout (sensors.observe), every signed/scaled channel enumerated
    by index so a bound mistake fails `observation_space.contains` instead of silently clipping:
      rays        0:54  9 x [dist, is_obstacle, is_chicken, is_self, is_other_body, is_egg] -> [0,1]
      social     54:61  [has_rival, rel_pos_fwd, rel_pos_left, rival_heading_fwd,
                          rival_heading_left, size_ratio, rival_is_dashing]
      egg        61:65  [has_egg, rel_pos_fwd, rel_pos_left, is_mine]
      smell      65:71  [chicken_intensity, chicken_grad_fwd, chicken_grad_left,
                          snake_intensity, snake_grad_fwd, snake_grad_left]
      proprio    71:75  [energy, length, stamina, repro_ready] -> [0,1]
    """
    assert OBS_DIM == 75, "observation_space layout is hand-enumerated for OBS_DIM=75"
    low = np.zeros(OBS_DIM, np.float32)
    high = np.ones(OBS_DIM, np.float32)                          # default [0,1]: rays, one-hots, proprio,
                                                                   # and the social/egg presence+ratio bits
    SOC = 54
    low[SOC + 1:SOC + 5] = -1.0; high[SOC + 1:SOC + 5] = 1.0      # rel_pos_fwd/left, rival_heading_fwd/left
    EGG = 61
    low[EGG + 1:EGG + 3] = -1.0; high[EGG + 1:EGG + 3] = 1.0      # rel_pos_fwd/left
    SM = 65
    ceil, nmax = float(cfg.chicken_ceiling), float(cfg.n_max)
    low[SM] = 0.0;     high[SM] = ceil                            # chicken_intensity
    low[SM + 1] = -ceil; high[SM + 1] = ceil                      # chicken_grad_fwd
    low[SM + 2] = -ceil; high[SM + 2] = ceil                      # chicken_grad_left
    low[SM + 3] = 0.0;   high[SM + 3] = nmax                      # snake_intensity
    low[SM + 4] = -nmax; high[SM + 4] = nmax                      # snake_grad_fwd
    low[SM + 5] = -nmax; high[SM + 5] = nmax                      # snake_grad_left
    assert SM + 6 == 71 and 71 + 4 == OBS_DIM                     # proprio 71:75 stays at the [0,1] default
    return spaces.Box(low, high, (OBS_DIM,), np.float32)


class SnakeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg=CFG, seed=None, world_size=None, dash_penalty=None, hardness=1.0):
        super().__init__()
        assert_invariants(cfg)
        self._base_cfg = cfg
        self._seed = seed
        self._seeded = False
        self._world_size = world_size          # None -> random size per episode; fixed -> e.g. screen-fit
        self._dash_penalty = cfg.dash_penalty if dash_penalty is None else dash_penalty
        self.set_hardness(hardness)            # sets self.cfg (stamina gate/regen interpolated easy<->hard)
        self.action_space = spaces.MultiDiscrete([3, 2])
        self.observation_space = _make_observation_space(cfg)
        self.world = None
        self._last_phi = 0.0
        self._last_ids = frozenset()

    def set_hardness(self, h):
        """Curriculum knob: h=0 -> easy always-available dash (learn hunting); h=1 -> the real reserve mechanic.
        Interpolates the stamina gate + regen. Applied to new episodes (world reads cfg at reset)."""
        h = float(min(1.0, max(0.0, h)))
        self._hardness = h
        b = self._base_cfg
        self.cfg = replace(
            b,
            dash_min_stamina=b.dash_min_stamina_easy + h * (b.dash_min_stamina - b.dash_min_stamina_easy),
            stamina_regen=b.stamina_regen_easy + h * (b.stamina_regen - b.stamina_regen_easy),
        )

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
        info = {"ate": out["ate"], "alive": self.world.alive, "steps": self.world.steps,
                "death_cause": self.world.death_cause}
        return observe(self.world), float(reward), terminated, truncated, info
