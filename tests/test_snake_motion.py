import numpy as np
from snake_rl.config import CFG
from snake_rl.world import World


def fresh():
    return World(CFG, seed=0)


def test_straight_move_advances_by_v_snake():
    w = fresh(); h0 = w.head_uw.copy(); head0_angle = w.heading
    w.move(steering=1, dash=0)                 # 1 = straight
    assert abs(w.heading - head0_angle) < 1e-9
    assert abs(np.linalg.norm(w.head_uw - h0) - CFG.v_snake) < 1e-6


def test_turn_changes_heading_by_delta():
    w = fresh(); a0 = w.heading
    w.move(steering=2, dash=0)                  # 2 = +turn
    diff = (w.heading - a0 + np.pi) % (2 * np.pi) - np.pi   # wrapped angular diff (seed-robust)
    assert abs(diff - np.radians(CFG.turn_deg)) < 1e-6


def test_dash_uses_v_dash_and_drains_stamina():
    w = fresh(); h0 = w.head_uw.copy(); s0 = w.stamina
    dashed = w.move(steering=1, dash=1)
    assert dashed
    assert abs(np.linalg.norm(w.head_uw - h0) - CFG.v_dash) < 1e-6
    assert abs(w.stamina - (s0 - CFG.stamina_drain)) < 1e-6


def test_dash_ignored_when_stamina_empty():
    w = fresh(); w.stamina = 0.0; h0 = w.head_uw.copy()
    dashed = w.move(steering=1, dash=1)
    assert not dashed
    assert abs(np.linalg.norm(w.head_uw - h0) - CFG.v_snake) < 1e-6


def test_stamina_regens_when_not_dashing():
    w = fresh(); w.stamina = 10.0
    w.move(steering=1, dash=0)
    assert abs(w.stamina - min(CFG.s_max, 10.0 + CFG.stamina_regen)) < 1e-6


def test_head_stays_wrapped():
    w = fresh()
    for _ in range(500):
        w.move(steering=1, dash=1)
    assert (0 <= w.head).all() and (w.head < w.size).all()


def test_body_points_interpolated_at_segment_spacing():
    # drive a straight line so path is dense, then check body points are evenly spaced
    w = fresh(); w.heading = 0.0
    w.head = np.array([30.0, 30.0]); w.head_uw = w.head.copy()
    w.path_uw = [w.head_uw.copy()]
    w.target_length = 10.0
    for _ in range(40):
        w.move(steering=1, dash=0)
    pts = w.body_points_uw()
    assert len(pts) >= 3
    gaps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    np.testing.assert_allclose(gaps, CFG.segment_spacing, atol=1e-6)   # exact spacing, no duplicates
    # first body point sits past the head-adjacent skip
    skip = CFG.head_radius + CFG.segment_spacing + CFG.body_radius
    assert abs(np.linalg.norm(pts[0] - w.head_uw) - skip) < 1e-6
