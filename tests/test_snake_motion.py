import numpy as np
from snake_rl.config import CFG
from snake_rl.world import World
from snake_rl.worldgen import generate_world
from snake_rl import genome as gm


def fresh():
    return World(CFG, seed=0)


def _world_with_two_speed_genomes(seed=7):
    w = generate_world(CFG, seed=seed, n_snakes=2)
    slow = np.full(gm.GENE_COUNT, 0.0, np.float32); slow[gm.SPEED] = 0.0
    fast = np.full(gm.GENE_COUNT, 0.0, np.float32); fast[gm.SPEED] = 1.0
    w.snakes[0] = w._make_snake(w.snakes[0].head, 0.0, genome=slow, sex=0, lineage=1,
                                id=w.snakes[0].id, color_seed=1, energy=CFG.energy_max,
                                target_length=CFG.start_length, rng=w.rng)
    w.snakes[1] = w._make_snake(w.snakes[1].head, 0.0, genome=fast, sex=1, lineage=2,
                                id=w.snakes[1].id, color_seed=2, energy=CFG.energy_max,
                                target_length=CFG.start_length, rng=w.rng)
    return w


def test_faster_genome_travels_farther_at_full_cruise():
    w = _world_with_two_speed_genomes()
    p0 = w.snakes[0].head.copy(); p1 = w.snakes[1].head.copy()
    # both go straight, full cruise (speed idx 3), no dash
    for _ in range(5):
        w.step(3, 1, 0, opponent_fn=lambda world, s: (3, 1, 0))
    from snake_rl.world import torus_dist
    d_slow = torus_dist(w.snakes[0].head, p0, w.size)
    d_fast = torus_dist(w.snakes[1].head, p1, w.size)
    assert d_fast > d_slow * 1.1


def test_straight_move_advances_by_v_snake():
    # NOTE: the founder snake carries a RANDOM genome (World.__init__ -> sample_genome), so its
    # cruise speed is its OWN phenotype.v_snake, not the global CFG.v_snake (Task 4: motion now
    # reads per-snake phenotype) -- assert against the snake's own resolved stat.
    w = fresh(); h0 = w.head_uw.copy(); head0_angle = w.heading; ph = w.snakes[0].phenotype
    w.move(3, 1, 0)                 # 1 = straight
    assert abs(w.heading - head0_angle) < 1e-9
    assert abs(np.linalg.norm(w.head_uw - h0) - ph.v_snake) < 1e-6


def test_size_gene_caps_body_growth_at_per_snake_max_length():
    # The size gene must have a BENEFIT: a body grows to its OWN phenotype.max_length, not the global
    # length_cap. Small size (gene 0) -> max_length 24*0.65=15.6 (below length_cap); large size
    # (gene 1) -> 24*1.35=32.4 (above it). Feed both well past 24 and check each caps at its own max.
    w = generate_world(CFG, seed=3, n_snakes=2)
    small_g = np.full(gm.GENE_COUNT, 0.5, np.float32); small_g[gm.SIZE] = 0.0
    large_g = np.full(gm.GENE_COUNT, 0.5, np.float32); large_g[gm.SIZE] = 1.0
    small = w._make_snake(w.snakes[0].head, 0.0, genome=small_g, sex=0, lineage=1,
                          id=w.snakes[0].id, color_seed=1, energy=CFG.energy_max,
                          target_length=CFG.start_length, rng=w.rng)
    large = w._make_snake(w.snakes[1].head, 0.0, genome=large_g, sex=1, lineage=2,
                          id=w.snakes[1].id, color_seed=2, energy=CFG.energy_max,
                          target_length=CFG.start_length, rng=w.rng)
    w.snakes[0] = small; w.snakes[1] = large
    assert small.phenotype.max_length < CFG.length_cap < large.phenotype.max_length  # 15.6 < 24 < 32.4

    for s in (small, large):                       # sequential so no cross-eating (shared chicken array)
        for _ in range(30):
            w._add_chicken(s.head)                 # 30 chickens on the head => growth attempt 6+60=66
        w._snake_eat(s)                            #   which the cap must clip to the snake's own max

    eps = 1e-6
    assert large.target_length > small.target_length              # (a) size gene's benefit is real
    assert small.target_length <= small.phenotype.max_length + eps  # (b) neither exceeds its own max
    assert large.target_length <= large.phenotype.max_length + eps
    assert small.target_length < CFG.length_cap                    # (c) small caps BELOW the global 24


def test_turn_changes_heading_by_delta():
    w = fresh(); a0 = w.heading; ph = w.snakes[0].phenotype
    w.move(3, 2, 0)                  # 2 = +turn
    diff = (w.heading - a0 + np.pi) % (2 * np.pi) - np.pi   # wrapped angular diff (seed-robust)
    assert abs(diff - np.radians(ph.turn_deg)) < 1e-6


def test_dash_uses_v_dash_and_drains_stamina():
    w = fresh(); h0 = w.head_uw.copy(); s0 = w.stamina; ph = w.snakes[0].phenotype
    dashed = w.move(3, 1, 1)
    assert dashed
    assert abs(np.linalg.norm(w.head_uw - h0) - ph.v_dash) < 1e-6
    assert abs(w.stamina - (s0 - CFG.stamina_drain)) < 1e-6   # drain rate stays global (not per-genome)


def test_dash_ignored_when_stamina_empty():
    w = fresh(); w.stamina = 0.0; h0 = w.head_uw.copy(); ph = w.snakes[0].phenotype
    dashed = w.move(3, 1, 1)
    assert not dashed
    assert abs(np.linalg.norm(w.head_uw - h0) - ph.v_snake) < 1e-6


def test_stamina_regen_scales_with_cruise_speed():
    # regen is speed-scaled: full at a dead stop (speed_idx 0), ZERO at full cruise (speed_idx 3),
    # linear in between -> standing still recharges the dash reserve fastest (v2.1 ambush economy).
    w = fresh(); w.stamina = 10.0; ph = w.snakes[0].phenotype
    w.move(0, 1, 0)                                                   # stopped -> full regen
    assert abs(w.stamina - (10.0 + ph.stamina_regen)) < 1e-6
    w = fresh(); w.stamina = 10.0
    w.move(3, 1, 0)                                                   # full cruise -> zero regen (genome-independent: ×0)
    assert abs(w.stamina - 10.0) < 1e-6
    w = fresh(); w.stamina = 10.0; ph = w.snakes[0].phenotype
    w.move(1, 1, 0)                                                   # 1/3 cruise -> partial regen
    assert abs(w.stamina - (10.0 + ph.stamina_regen * (1 - CFG.speed_levels[1]))) < 1e-6


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
    # first body point sits past the head-adjacent skip (v_dash is per-snake, Task 4/Pitfall 5)
    skip = CFG.head_radius + CFG.body_radius + w.snakes[0].phenotype.v_dash + CFG.segment_spacing
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
    skip = CFG.head_radius + CFG.body_radius + w.snakes[0].phenotype.v_dash + CFG.segment_spacing
    n_expected = int((CFG.length_cap - skip) // CFG.segment_spacing) + 1
    assert len(w.body_points_uw()) == n_expected     # no silently dropped tail points

