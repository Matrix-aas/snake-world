import numpy as np
from snake_rl.config import CFG
from snake_rl.world import World


def fresh():
    return World(CFG, seed=0)


def test_straight_move_advances_by_v_snake():
    w = fresh(); h0 = w.head_uw.copy(); head0_angle = w.heading
    w.move(3, 1, 0)                 # 1 = straight
    assert abs(w.heading - head0_angle) < 1e-9
    assert abs(np.linalg.norm(w.head_uw - h0) - CFG.v_snake) < 1e-6


def test_turn_changes_heading_by_delta():
    w = fresh(); a0 = w.heading
    w.move(3, 2, 0)                  # 2 = +turn
    diff = (w.heading - a0 + np.pi) % (2 * np.pi) - np.pi   # wrapped angular diff (seed-robust)
    assert abs(diff - np.radians(CFG.turn_deg)) < 1e-6


def test_dash_uses_v_dash_and_drains_stamina():
    w = fresh(); h0 = w.head_uw.copy(); s0 = w.stamina
    dashed = w.move(3, 1, 1)
    assert dashed
    assert abs(np.linalg.norm(w.head_uw - h0) - CFG.v_dash) < 1e-6
    assert abs(w.stamina - (s0 - CFG.stamina_drain)) < 1e-6


def test_dash_ignored_when_stamina_empty():
    w = fresh(); w.stamina = 0.0; h0 = w.head_uw.copy()
    dashed = w.move(3, 1, 1)
    assert not dashed
    assert abs(np.linalg.norm(w.head_uw - h0) - CFG.v_snake) < 1e-6


def test_stamina_regen_scales_with_cruise_speed():
    # regen is speed-scaled: full at a dead stop (speed_idx 0), ZERO at full cruise (speed_idx 3),
    # linear in between -> standing still recharges the dash reserve fastest (v2.1 ambush economy).
    w = fresh(); w.stamina = 10.0
    w.move(0, 1, 0)                                                   # stopped -> full regen
    assert abs(w.stamina - (10.0 + CFG.stamina_regen)) < 1e-6
    w = fresh(); w.stamina = 10.0
    w.move(3, 1, 0)                                                   # full cruise -> zero regen
    assert abs(w.stamina - 10.0) < 1e-6
    w = fresh(); w.stamina = 10.0
    w.move(1, 1, 0)                                                   # 1/3 cruise -> partial regen
    assert abs(w.stamina - (10.0 + CFG.stamina_regen * (1 - CFG.speed_levels[1]))) < 1e-6


def test_head_stays_wrapped():
    w = fresh()
    for _ in range(500):
        w.move(3, 1, 1)
    assert (0 <= w.head).all() and (w.head < w.size).all()


def test_body_points_interpolated_at_segment_spacing():
    # drive a straight line so path is dense, then check body points are evenly spaced
    w = fresh(); w.heading = 0.0
    w.head = np.array([30.0, 30.0]); w.head_uw = w.head.copy()
    w.path_uw = [w.head_uw.copy()]
    w.target_length = 10.0
    for _ in range(40):
        w.move(3, 1, 0)
    pts = w.body_points_uw()
    assert len(pts) >= 3
    gaps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    np.testing.assert_allclose(gaps, CFG.segment_spacing, atol=1e-6)   # exact spacing, no duplicates
    # first body point sits past the head-adjacent skip
    skip = CFG.head_radius + CFG.body_radius + CFG.v_dash + CFG.segment_spacing
    assert abs(np.linalg.norm(pts[0] - w.head_uw) - skip) < 1e-6


def test_body_render_path_starts_at_head():
    # rendering body must connect to the head (no neck gap -> no "detached red dot")
    w = fresh(); w.heading = 0.0
    w.head = np.array([20.0, 30.0]); w.head_uw = w.head.copy(); w.path_uw = [w.head_uw.copy()]
    for _ in range(30):
        w.move(3, 1, 0)
    rp = w.body_render_path_uw()
    assert len(rp) >= 5
    np.testing.assert_allclose(rp[0], w.head_uw)     # index 0 IS the head
    assert np.linalg.norm(np.diff(rp, axis=0), axis=1).max() < 1.0   # dense & continuous


def test_body_points_no_tail_drop_at_full_length():
    # after growing to length_cap, every body target must be emitted (prune slack must not truncate the tail)
    w = fresh(); w.heading = 0.0
    w.head = np.array([10.0, 30.0]); w.head_uw = w.head.copy()
    w.path_uw = [w.head_uw.copy()]
    w.target_length = CFG.length_cap
    for _ in range(80):
        w.move(3, 1, 0)
    skip = CFG.head_radius + CFG.body_radius + CFG.v_dash + CFG.segment_spacing
    n_expected = int((CFG.length_cap - skip) // CFG.segment_spacing) + 1
    assert len(w.body_points_uw()) == n_expected     # no silently dropped tail points

