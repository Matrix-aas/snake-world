import math
import pytest
from snake_rl.config import Config, CFG, assert_invariants, assert_invariants_over_genome


def test_default_config_satisfies_invariants():
    assert_invariants(CFG)  # must not raise


def test_invariants_hold_across_gene_box():
    assert_invariants_over_genome(CFG)   # HARD gates must not raise


def test_catch_invariant_math():
    c = CFG
    assert c.v_dash > c.v_flee
    budget = (c.s_max / c.stamina_drain) * (c.v_dash - c.v_flee)
    assert budget >= c.catch_slack_k * c.r_flee


def test_aiming_precision_invariant():
    c = CFG
    assert math.radians(c.turn_deg) / 2 < math.atan(c.eat_radius / c.r_flee)


def test_miscalibrated_config_raises():
    bad = Config(v_dash=1.0, v_flee=1.5)  # dash slower than flee
    with pytest.raises(AssertionError):
        assert_invariants(bad)


def test_body_cap_below_half_world():
    c = CFG
    assert c.length_cap < c.world_size_min / 2


def test_self_collision_reachable():
    c = CFG
    turn_circumference = 2 * math.pi * c.v_snake / math.radians(c.turn_deg)
    assert turn_circumference < c.length_cap   # a full curl fits inside the body -> head can hit itself


def test_motion_collision_constants_present_and_sane():
    c = CFG
    assert c.speed_levels[0] == 0.0 and c.speed_levels[-1] == 1.0 and len(c.speed_levels) == 4
    assert isinstance(c.stun_steps, int) and c.stun_steps >= 1
    assert isinstance(c.n_fwd_rays, int) and c.n_fwd_rays >= 0


def test_pitfall16_reward_constants_removed():
    # the whole obstacle-reward machinery is gone (obstacles are solid-slide now, non-lethal)
    assert not hasattr(CFG, "reward_death_obstacle")
    assert not hasattr(CFG, "obs_avoid_weight")
    assert not hasattr(CFG, "obs_avoid_range")


def test_multisnake_invariants_hold():
    from snake_rl.config import CFG, assert_invariants
    assert_invariants(CFG)                                  # must not raise
    assert CFG.world_size_min == 180.0 and CFG.n_max == 12
    # mating distance lets two snakes coexist without a forced cut-off
    assert CFG.r_mate >= 2 * CFG.head_radius
    # a just-qualified snake can pay the repro cost and survive
    assert CFG.repro_cost < CFG.repro_energy_frac * CFG.energy_max
    # food ceiling covers the population-scaled max
    assert CFG.chicken_ceiling >= CFG.chickens_per_snake_max * CFG.n_max


def test_scaled_population_invariants():
    assert CFG.n_max == 12
    # food ceiling still covers max demand (invariant 10)
    assert CFG.chicken_ceiling >= CFG.chickens_per_snake_max * CFG.n_max
    assert_invariants(CFG)
    assert_invariants_over_genome(CFG)


def test_repro_curriculum_defaults_ship_hard():
    # base CFG is the viewer/headless mechanic: full energy gate + opposite-sex requirement
    assert CFG.mate_require_sex is True
    assert CFG.repro_energy_frac == 0.7


def test_easy_repro_energy_gate_still_affords_repro_cost():
    c = CFG
    assert c.repro_cost < c.repro_energy_frac_easy * c.energy_max
    assert c.repro_cost < c.repro_energy_frac * c.energy_max
