"""Gymnasium environment: sensors -> obs, MultiDiscrete action, PBRS reward, truncation."""
import numpy as np
from dataclasses import replace
import gymnasium as gym
from gymnasium import spaces
from .config import CFG, assert_invariants
from .worldgen import generate_world
from .sensors import observe, OBS_DIM
from .selfplay import OpponentController


def _make_observation_space(cfg):
    """Bounds for the 143-float layout (sensors.observe), every signed/scaled channel enumerated
    by index so a bound mistake fails `observation_space.contains` instead of silently clipping:
      rays        0:99  11 x [dist, is_obstacle, is_chicken, is_self, is_other_body, is_egg,
                              is_corpse, obstacle_clearance, target_motion] -> [0,1]
      social     99:110 [has_rival, rel_pos_fwd, rel_pos_left, rival_heading_fwd, rival_heading_left,
                          size_ratio, rival_is_dashing, relatedness, rival_energy, rival_repro_ready,
                          rival_sex]
      egg       110:114 [has_egg, rel_pos_fwd, rel_pos_left, is_mine]
      smell     114:123 [chicken_intensity, chicken_grad_fwd, chicken_grad_left,
                          snake_intensity, snake_grad_fwd, snake_grad_left,
                          corpse_intensity, corpse_grad_fwd, corpse_grad_left]
      vibration 123:126 [intensity, grad_fwd, grad_left]
      proprio   126:143 [energy, length, stamina, repro_ready, speed, sex, age_frac, stun_frac, genome x9] -> [0,1]

    chicken counts are capped by `world.maybe_spawn` (<= chicken_ceiling) and rivals by hatching's
    n_max refusal, but `sensors.smell` scales every field by a per-genome smell_reach (up to 1.4x) and
    then CLIPS all three fields to `chicken_ceiling` -- so each smell bound is `chicken_ceiling` to
    exactly match the clip (the snake channels are widened from n_max, which a dense high-reach cluster
    could exceed). Corpses have no population cap at all (persist until eaten) -- same clip covers them.
    Vibration is a speed-weighted field over rivals + fleeing chickens, so its natural bound is the two
    populations combined: `n_max + chicken_ceiling`, derived from cfg (never a literal).
    """
    assert OBS_DIM == 143, "observation_space layout is hand-enumerated for OBS_DIM=143"
    low = np.zeros(OBS_DIM, np.float32)
    high = np.ones(OBS_DIM, np.float32)                          # default [0,1]: rays (incl. clearance +
                                                                   # motion), one-hots, proprio, presence bits
    SOC = 99
    low[SOC + 1:SOC + 5] = -1.0; high[SOC + 1:SOC + 5] = 1.0      # rel_pos_fwd/left, rival_heading_fwd/left
                                                                   # SOC+5..+10 (size,dash,relatedness,energy,
                                                                   # repro_ready,sex) stay at the [0,1] default
    EGG = 110
    low[EGG + 1:EGG + 3] = -1.0; high[EGG + 1:EGG + 3] = 1.0      # rel_pos_fwd/left
    SM = 114
    ceil = float(cfg.chicken_ceiling)                            # every smell channel: clipped to this in sensors
    low[SM] = 0.0;       high[SM] = ceil                          # chicken_intensity
    low[SM + 1] = -ceil; high[SM + 1] = ceil                      # chicken_grad_fwd
    low[SM + 2] = -ceil; high[SM + 2] = ceil                      # chicken_grad_left
    low[SM + 3] = 0.0;   high[SM + 3] = ceil                      # snake_intensity (WIDENED n_max->ceil = clip)
    low[SM + 4] = -ceil; high[SM + 4] = ceil                      # snake_grad_fwd  (WIDENED)
    low[SM + 5] = -ceil; high[SM + 5] = ceil                      # snake_grad_left (WIDENED)
    low[SM + 6] = 0.0;   high[SM + 6] = ceil                      # corpse_intensity
    low[SM + 7] = -ceil; high[SM + 7] = ceil                      # corpse_grad_fwd
    low[SM + 8] = -ceil; high[SM + 8] = ceil                      # corpse_grad_left
    VIB = 123
    vceil = float(cfg.n_max + cfg.chicken_ceiling)               # derived from cfg (review I2), NOT a literal
    low[VIB] = 0.0;        high[VIB] = vceil                      # vibration_intensity
    low[VIB + 1] = -vceil; high[VIB + 1] = vceil                  # vibration_grad_fwd
    low[VIB + 2] = -vceil; high[VIB + 2] = vceil                  # vibration_grad_left
    PROP = 126
    assert VIB + 3 == PROP and PROP + 17 == OBS_DIM              # proprio 126:143 stays [0,1] (incl. genome tail)
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
        self._opp = OpponentController(cfg)    # drives in-env opponents from a policy snapshot (self-play)
        self.set_hardness(hardness)            # sets self.cfg (stamina gate/regen interpolated easy<->hard)
        self.action_space = spaces.MultiDiscrete([4, 3, 2])   # speed x steer x dash
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
        lerp = lambda easy, hard: easy + h * (hard - easy)
        self.cfg = replace(
            b,
            dash_min_stamina=lerp(b.dash_min_stamina_easy, b.dash_min_stamina),
            stamina_regen=lerp(b.stamina_regen_easy, b.stamina_regen),
            # mating curriculum: easy to discover early (close/instant/short), tightens as h -> 1.
            # The length gate is now a size-RELATIVE fraction (of each snake's own max_length); the
            # swept frac is read IDENTICALLY by world._resolve_mating and sensors._repro_ready, so the
            # observed repro_ready bit and the real eligibility ramp together (review I1).
            r_mate=lerp(b.r_mate_easy, b.r_mate),
            mate_steps=int(round(lerp(b.mate_steps_easy, b.mate_steps))),
            repro_length_frac=lerp(b.repro_length_frac_easy, b.repro_length_frac),
        )

    def _egg_lost_reward(self, eaten):
        """cfg.reward_egg_lost (negative, DEFAULT 0.0) if the ego co-owns >=1 eaten egg, else 0.0.
        `eaten` = world.step()["eaten_eggs"]: a list of frozenset(owner ids) for REAL eggs eaten this
        step. Flat (not per-egg) -- guarding is what the penalty teaches; the count is incidental."""
        ego_id = self.world.snakes[0].id
        return self.cfg.reward_egg_lost if any(ego_id in owners for owners in eaten) else 0.0

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
        n = int(self.np_random.integers(self.cfg.n_start_min, self.cfg.n_start_max + 1))   # [C-1] spawn N
        # arrivals=True: opponents ARRIVE via hatching eggs + runtime chickens drop from the sky, so the
        # policy trains on the same world dynamics the viewer shows. ego_live=True keeps ONE live
        # gradient-ego in slot 0 (SB3 drives it and can't steer an inert egg) -- unlike the viewer.
        self.world = generate_world(self.cfg, seed=world_seed, size=self._world_size, n_snakes=n,
                                    arrivals=True, ego_live=True)
        self._opp.reset_all()                  # [I-1] drop stale opponent frame rings for the new world
        self._last_phi = self._phi()
        self._last_ids = frozenset(int(i) for i in self.world.chicken_id)
        return observe(self.world), {}

    def set_opponent_policy(self, state_dict, obs_rms, clip_obs, epsilon):
        """[I-5] Push a fresh policy snapshot + VecNormalize stats into the opponent controller."""
        self._opp.sync(state_dict, obs_rms, clip_obs, epsilon)

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
        speed, steer, dash = int(action[0]), int(action[1]), int(action[2])
        ego_id = self.world.snakes[0].id
        out = self.world.step(speed, steer, dash, opponent_fn=lambda world, s: self._opp.act(world, s))
        terminated = out["died"]
        truncated = self.world.steps >= c.episode_horizon
        reward = c.reward_eat * out["ate"]
        # [I-2] pay reproduction ONLY for eggs that actually hatched AND that the ego co-owns; a raided
        # or population-cap-dropped ego egg is absent from hatched_owners, so it pays nothing.
        n_repro_ego = sum(1 for owners in out["hatched_owners"] if ego_id in owners)
        reward += c.reward_repro * n_repro_ego
        # egg-lost hook (Task 12, DEFAULT 0.0): an eaten ego-co-owned egg pays reward_egg_lost -- a
        # post-competence guarding-sharpening knob, OFF for the discovery retrain (Pitfall-1 cousin).
        reward += self._egg_lost_reward(out["eaten_eggs"])
        for sid, _cause in out["deaths_detailed"]:
            self._opp.reset_snake(sid)         # clear a dead snake's frame ring (no stale frames)
        if terminated:
            reward += c.reward_death           # flat cost on any death (starve / rival cut-off)
        else:
            reward += self._shaping()
        # PBRS shaping (>=0 when a chicken is far) partly offsets this while well-fed; that's the
        # standard potential-shaping artifact — hunger ramps the penalty and restores "get moving" pressure.
        hunger = 1.0 - self.world.energy / c.energy_max
        reward -= c.step_penalty * (1.0 + hunger)
        if out["dashed"]:                         # dashing burns energy -> use it only to chase, not constantly
            reward -= self._dash_penalty
        info = {"ate": out["ate"], "alive": self.world.alive, "steps": self.world.steps,
                "death_cause": self.world.death_cause,
                "repro_ego": n_repro_ego, "hatched": len(out["hatched_owners"])}
        return observe(self.world), float(reward), terminated, truncated, info
