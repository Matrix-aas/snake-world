"""All balance constants and feasibility invariants — the single source of numbers."""
from __future__ import annotations
import logging
import math
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    # world
    world_size_min: float = 110.0
    world_size_max: float = 160.0
    # population
    n_start_min: int = 2
    n_start_max: int = 4
    n_max: int = 6
    # snake motion
    v_snake: float = 1.0
    v_dash: float = 2.0
    turn_deg: float = 16.0            # sharp enough that a full curl (circumference ~22.5) fits under length_cap
    speed_levels: tuple = (0.0, 1 / 3, 2 / 3, 1.0)  # cruise fractions of v_snake; dash overrides
    stun_steps: int = 10             # dash into a solid -> frozen this many steps ("head spinning")
    # chickens
    v_wander: float = 0.25
    v_flee: float = 1.15
    r_flee: float = 12.0             # WALK alert distance; catching a runner needs a dash burst
    r_flee_peck: float = 2.5         # a head-down PECKING chicken is distracted: only startles when a
                                     # snake is THIS close (< r_flee_peck) -> the stalk-and-pounce catch window
    # behavior FSM (peck/walk/flee): a chicken pecks in place (prime catch window), ambles, then on
    # threat freezes in surprise for a beat and bolts. Durations in steps (staggered by rng at spawn).
    chicken_peck_min: int = 6        # halved (was 12/35): the head-down stationary catch window is now
    chicken_peck_max: int = 18       # shorter, so snakes can't lean on it (+ the speed/stop exploits)
    chicken_walk_min: int = 18
    chicken_walk_max: int = 45
    chicken_startle_steps: int = 4   # entering flee: FREEZE (speed 0) for this many steps, then bolt at v_flee
    chicken_flee_persist: int = 15   # FEAR PERSISTENCE: a scared hen keeps bolting this many steps after the
                                     # last time a snake was within reach, so a snake can't "un-spook" it by
                                     # freezing (speed 0 -> alert 0). Re-armed every step a snake IS near.
    chicken_arrive_steps: int = 12   # a spawned chicken DROPS FROM THE SKY over this many steps (falling +
                                     # growing shadow) before it lands and becomes a real, huntable/sensed
                                     # chicken. Purely a spawn presentation: in-flight chickens live in a
                                     # separate world.arriving array, so sensors/eat never see them (Goal 2).
    chicken_radius: float = 1.0
    spawn_period: int = 90           # avg steps between random spawns between min and max
    # food, population-scaled (rates; Task 9 derives the live target from snake count)
    chickens_per_snake_max: float = 2.0
    chickens_per_snake_min: float = 1.0
    chicken_ceiling: int = 12        # hard cap regardless of population
    # stamina
    s_max: float = 30.0
    stamina_drain: float = 1.0
    # HARD (final) stamina: dash needs a full-unit reserve, refills slowly -> deliberate bursts.
    # 0.42 (ecosystem-sustain tuning; v1 trained at 0.3 then eased at runtime, v2 trains on 0.42
    # from scratch -- see CLAUDE.md history): more energy surplus for the persistent world's mating
    # economy; still a real reserve, just less starved for it.
    stamina_regen: float = 0.42      # refills the reserve in ~70 steps: fast enough to hunt, slow enough to matter
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
    # reproduction / eggs / corpses
    repro_energy_frac: float = 0.7   # min energy fraction to qualify for mating
    # v2 from-scratch training endpoints (ecosystem-sustain tuning; v1 trained tighter --
    # repro_length_min 10, r_mate 4, mate_steps 4, repro_cost 30, repro_cooldown 120 -- then eased
    # these at runtime; v2 trains on the eased values directly, see CLAUDE.md history). NOTE:
    # repro_length_min IS observed (sensors._repro_ready, Pitfall 12) -- do NOT ease it past the
    # curriculum's swept [repro_length_min_easy, repro_length_min] range without a retrain.
    repro_length_min: float = 8.0    # min body length to qualify for mating
    r_mate: float = 7.0              # mating distance
    mate_steps: int = 2              # steps two qualified snakes must hold mating distance
    repro_cost: float = 18.0         # energy spent by each parent on a successful mating
    repro_cooldown: int = 80         # steps before a snake can mate again
    egg_timer: int = 45              # steps until an egg hatches
    hatch_energy_frac: float = 0.5   # hatchling starting energy fraction
    egg_food: float = 25.0           # food value if an egg is eaten instead of hatching
    corpse_food_per_length: float = 4.0  # food value of a dead snake's corpse, per unit length
    egg_radius: float = 1.0          # egg footprint for ray hit-testing (Minkowski +head_radius, Pitfall 8)
    # mating curriculum (easy warmup values; set_hardness interpolates each toward the hard value above
    # so reproduction is easy to DISCOVER early, then tightens as hardness ramps to 1.0)
    r_mate_easy: float = 12.0
    mate_steps_easy: int = 1
    repro_length_min_easy: float = 6.0
    # sensing
    n_rays: int = 9
    n_fwd_rays: int = 2              # extra forward rays; RAY_COUNT = n_rays + n_fwd_rays = 11
    fov_deg: float = 270.0           # total arc, centered forward (±135°)
    ray_range: float = 20.0
    frame_stack: int = 4
    # rl / episode
    episode_horizon: int = 2000
    gamma: float = 0.99              # reliable for discovering hunting (0.995 slowed early value learning)
    # reward
    reward_eat: float = 10.0
    reward_repro: float = 12.0
    reward_death: float = -10.0      # flat cost on ANY death (starve / rival cut-off) -- obstacles and
                                     # own body are now solid-slide, non-lethal, so no cause-specific cost
    step_penalty: float = 0.01
    dash_penalty: float = 0.0        # dashing is rationed by the stamina reserve itself (gate + slow regen),
                                     # so no extra reward penalty is needed (one over-suppresses hunting)
    catch_slack_k: float = 1.5

    @property
    def eat_radius(self) -> float:
        return self.head_radius + self.chicken_radius


def assert_invariants(cfg: Config) -> None:
    # (1) dash beats flee — a dash always closes on a bolting chicken
    assert cfg.v_dash > cfg.v_flee, "v_dash must exceed v_flee"
    # a pecking chicken is distracted: its startle range must be tighter than the walking alert range
    assert cfg.r_flee_peck < cfg.r_flee, "r_flee_peck must be smaller than r_flee (peck = distracted)"
    # a scared hen's fear-persistence must outlast its startle freeze, or it could expire the panic
    # window mid-freeze and settle to WALK without ever bolting (the fix would be a no-op)
    assert cfg.chicken_flee_persist > cfg.chicken_startle_steps, \
        "chicken_flee_persist must exceed chicken_startle_steps so a scared hen actually bolts"
    # motion/collision: a stun lasts at least one step, cruise levels span 0 -> full v_snake, forward rays >= 0
    assert cfg.stun_steps >= 1, "stun_steps must be at least 1"
    assert cfg.speed_levels[0] == 0.0 and cfg.speed_levels[-1] == 1.0, \
        "speed_levels must run from a full stop (0.0) to full cruise (1.0)"
    assert cfg.n_fwd_rays >= 0, "n_fwd_rays must be non-negative"
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
    #     (vision inflates targets by head_radius, so include it in the reach)
    assert cfg.ray_range + cfg.obstacle_radius_max + cfg.head_radius < cfg.world_size_min / 2, \
        "ray_range too large for nearest-image raycasting on the smallest world"
    # (7) two snakes can sit at mating distance without a forced cut-off
    assert cfg.r_mate >= 2 * cfg.head_radius, "r_mate too small: mating forces a collision"
    # (8) a snake that just crossed the energy threshold can pay the repro cost and live
    assert cfg.repro_cost < cfg.repro_energy_frac * cfg.energy_max, "repro_cost exceeds the mating gate"
    # (9) hatchling viable
    assert cfg.hatch_energy_frac * cfg.energy_max > 0 and cfg.start_length >= (
        cfg.head_radius + cfg.body_radius + cfg.v_dash + cfg.segment_spacing), "hatchling not viable"
    # (10) food ceiling covers the population-scaled demand (soft feasibility)
    assert cfg.chicken_ceiling >= cfg.chickens_per_snake_max * cfg.n_max, "chicken_ceiling too low"
    # (spec §10.6, soft) n_max bodies should occupy much less than the smallest world's area —
    # not fatal if violated (worlds still function), but a full n_max of full-length snakes
    # packed into a tiny world would make _free_point/mating geometry miserable. Log, don't fail.
    body_area = cfg.n_max * (cfg.length_cap * 2 * cfg.body_radius + math.pi * cfg.head_radius ** 2)
    world_area = cfg.world_size_min ** 2
    if body_area > 0.25 * world_area:
        log.warning("n_max snake bodies occupy %.0f%% of the smallest world's area (%.1f / %.1f) — "
                    "consider a bigger world_size_min or a lower n_max", 100 * body_area / world_area,
                    body_area, world_area)


CFG = Config()
assert_invariants(CFG)
