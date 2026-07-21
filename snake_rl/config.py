"""All balance constants and feasibility invariants — the single source of numbers."""
from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # world
    world_size_min: float = 60.0
    world_size_max: float = 100.0
    # snake motion
    v_snake: float = 1.0
    v_dash: float = 2.0
    turn_deg: float = 16.0            # sharp enough that a full curl (circumference ~22.5) fits under length_cap
    # chickens
    v_wander: float = 0.25
    v_flee: float = 1.15
    r_flee: float = 12.0             # chickens bolt at this distance; catching a runner needs a dash burst
    chicken_radius: float = 1.0
    max_chickens: int = 5
    min_chickens: int = 3            # keep the world populated (fast refill below this)
    spawn_period: int = 90           # avg steps between random spawns between min and max
    # stamina
    s_max: float = 30.0
    stamina_drain: float = 1.0
    # HARD (final) stamina: dash needs a full-unit reserve, refills slowly -> deliberate bursts
    stamina_regen: float = 0.3       # refills the reserve in ~100 steps: fast enough to hunt, slow enough to matter
    dash_min_stamina: float = 1.0    # need a full unit to enter a dash -> stamina is a real reserve to spend
    # EASY (warmup) stamina: cheap, always-available dash so the agent can first LEARN to hunt
    stamina_regen_easy: float = 0.6
    dash_min_stamina_easy: float = 0.05
    # curriculum: keep easy for the first `hardness_warmup` of training, ramp to full-hard by `hardness_full`
    hardness_warmup: float = 0.42    # longer warmup -> a stronger hunter before the reserve tightens
    hardness_full: float = 0.85
    # geometry
    head_radius: float = 1.0
    body_radius: float = 0.5
    segment_spacing: float = 0.6
    obstacle_radius_min: float = 1.5
    obstacle_radius_max: float = 4.0
    n_obstacles_min: int = 6
    n_obstacles_max: int = 16
    # energy (hunger — not lethal)
    energy_max: float = 100.0
    energy_decay: float = 0.05
    energy_refill: float = 40.0
    # snake growth / cap
    start_length: float = 6.0        # target body length (> neck-skip); the body fills in over the first few steps
    grow_per_chicken: float = 2.0
    length_cap: float = 24.0         # > tightest-curl circumference (~22.5) so self-collision is reachable, < world/2
    # sensing
    n_rays: int = 9
    fov_deg: float = 270.0           # total arc, centered forward (±135°)
    ray_range: float = 20.0
    frame_stack: int = 4
    # rl / episode
    episode_horizon: int = 2000
    gamma: float = 0.99              # reliable for discovering hunting (0.995 slowed early value learning)
    # reward
    reward_eat: float = 10.0
    reward_death: float = -10.0
    step_penalty: float = 0.01
    dash_penalty: float = 0.0        # dashing is rationed by the stamina reserve itself (gate + slow regen),
                                     # so no extra reward penalty is needed (one over-suppresses hunting)
    catch_slack_k: float = 1.5

    @property
    def eat_radius(self) -> float:
        return self.head_radius + self.chicken_radius


def assert_invariants(cfg: Config) -> None:
    # (1) dash beats flee
    assert cfg.v_dash > cfg.v_flee, "v_dash must exceed v_flee"
    # (2) stamina budget closes the flee radius with slack
    budget = (cfg.s_max / cfg.stamina_drain) * (cfg.v_dash - cfg.v_flee)
    assert budget >= cfg.catch_slack_k * cfg.r_flee, "stamina budget too small for guaranteed catch"
    # (3) aiming precision: can point into the eat corridor
    assert math.radians(cfg.turn_deg) / 2 < math.atan(cfg.eat_radius / cfg.r_flee), \
        "turn_deg too coarse to aim at chickens"
    # body never wraps to meet its own head across the seam
    assert cfg.length_cap < cfg.world_size_min / 2, "length_cap must stay below half the smallest world"
    # (4) self-collision must be physically reachable: tightest full curl fits within the body length
    turn_circumference = 2 * math.pi * cfg.v_snake / math.radians(cfg.turn_deg)
    assert turn_circumference < cfg.length_cap, "turn radius too wide for the body to curl onto itself"
    # (5) nearest-image raycast is only valid if no second image is reachable within ray range
    assert cfg.ray_range + cfg.obstacle_radius_max < cfg.world_size_min / 2, \
        "ray_range too large for nearest-image raycasting on the smallest world"


CFG = Config()
assert_invariants(CFG)
