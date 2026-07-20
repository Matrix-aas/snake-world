import math
import pytest
from snake_rl.config import Config, CFG, assert_invariants


def test_default_config_satisfies_invariants():
    assert_invariants(CFG)  # must not raise


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
