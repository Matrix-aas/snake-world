import numpy as np
from snake_rl.config import CFG
from snake_rl.world import World, Snake, wrap, torus_delta
from snake_rl.worldgen import generate_world
from snake_rl.sensors import observe, sense_vision, smell, ray_dirs, OBS_DIM
from snake_rl import genome as gm


def straight_world():
    w = World(CFG, seed=0, size=(60, 60))
    w.head = np.array([30.0, 30.0]); w.head_uw = w.head.copy(); w.heading = 0.0  # facing +x
    return w


def _add_rival(w, pos, id=1, heading=np.pi, target_length=None):
    p = np.asarray(pos, float)
    r = Snake(head_uw=p.copy(), head=wrap(p, w.size), heading=heading, path_uw=[p.copy()],
              target_length=target_length if target_length is not None else CFG.start_length,
              stamina=CFG.s_max, energy=CFG.energy_max, _prev_head_uw=p.copy(), id=id)
    w.snakes.append(r)
    return r


def test_obs_dim_is_143_and_contains_genome_tail():
    assert OBS_DIM == 143
    w = generate_world(CFG, seed=9, n_snakes=2)
    s = w.snakes[0]
    obs = observe(w, s)
    assert obs.shape == (OBS_DIM,)
    # last 9 floats are the snake's genome (per-frame, un-normalized here)
    assert np.allclose(obs[-gm.GENE_COUNT:], s.genome, atol=1e-6)


def test_vibration_responds_to_a_moving_rival_not_a_still_one():
    # a dashing rival nearby should raise the vibration intensity vs. a stopped one
    w = generate_world(CFG, seed=8, n_snakes=2, size=(60.0, 60.0))
    a, b = w.snakes[0], w.snakes[1]
    b.head[:] = a.head + np.array([4.0, 0.0]); b.head_uw[:] = b.head
    b.speed = 0.0
    still = observe(w, a)
    b.speed = b.phenotype.v_dash
    moving = observe(w, a)
    VIB = 99 + 11 + 4 + 9    # vibration block start
    assert moving[VIB] > still[VIB]


def test_obs_shape_and_bounds():
    w = straight_world(); w.maybe_spawn_forced()
    o = observe(w)
    assert OBS_DIM == 143
    assert o.shape == (OBS_DIM,) and o.dtype == np.float32
    assert np.isfinite(o).all()
    assert (o[:99].reshape(11, 9)[:, 0] >= 0).all() and (o[:99].reshape(11, 9)[:, 0] <= 1).all()


def test_ray_dirs_count_and_forward_angles():
    dirs = ray_dirs(CFG, 0.0)
    assert len(dirs) == 11                                # 9 uniform + 2 forward
    fwd = np.degrees(np.arctan2(dirs[-2:, 1], dirs[-2:, 0]))   # the two appended forward rays
    assert abs(abs(fwd[0]) - 16.875) < 1e-6 and abs(abs(fwd[1]) - 16.875) < 1e-6
    assert fwd[0] < 0 < fwd[1]                             # symmetric ±16.875° about heading


def test_unmask_reports_obstacle_behind_a_nearer_chicken():
    # chicken directly ahead, a rock just beyond it on the same forward ray: the ray's NEAREST hit is
    # the chicken (is_chicken=1) but the un-mask channel still reports the rock < ray_range.
    w = straight_world()
    w.set_chickens([[35.0, 30.0]])                        # nearer, on the +x forward ray
    w.obstacle_pos = np.array([[40.0, 30.0]]); w.obstacle_r = np.array([1.0]); w.obstacle_kind = np.array([0])
    center = sense_vision(w)[CFG.n_rays // 2]             # exactly-forward ray
    assert center[2] == 1.0                               # is_chicken (nearest target)
    assert center[1] == 0.0                               # NOT reported as an obstacle by the hit one-hot
    assert 0.0 < center[7] < 1.0                          # un-mask: the rock behind it, clearance < ray_range


def test_center_ray_sees_obstacle_ahead():
    w = straight_world()
    w.obstacle_pos = np.array([[40.0, 30.0]]); w.obstacle_r = np.array([1.0]); w.obstacle_kind = np.array([0])
    v = sense_vision(w)
    center = v[CFG.n_rays // 2]
    assert center[1] == 1.0                       # obstacle channel
    # distance until the head EDGE touches: 40 - (obstacle_r + head_radius) - 30, normalized by the
    # snake's OWN per-genome ray_range (not the global CFG.ray_range -- founders carry random genomes)
    ray_range = w.snakes[0].phenotype.ray_range
    assert abs(center[0] - (40 - 1 - CFG.head_radius - 30) / ray_range) < 1e-2


def test_empty_ray_encoding():
    w = straight_world()
    v = sense_vision(w)
    # nothing hit -> dist=1 (ray_range), all one-hots 0, obstacle_clearance=1 (no rock on any bearing),
    # motion 0 (no target -> no motion channel)
    assert (v == np.array([1.0, 0, 0, 0, 0, 0, 0, 1.0, 0.0])).all()


def test_ray_detects_corpse():
    w = straight_world()
    w.corpses = {"pos": np.array([[40.0, 30.0]]), "food": np.array([5.0])}
    v = sense_vision(w)
    center = v[CFG.n_rays // 2]
    assert center[6] == 1.0                           # is_corpse
    assert center[1] == 0.0 and center[2] == 0.0 and center[3] == 0.0 and center[4] == 0.0 and center[5] == 0.0
    # a ray pointed away from the corpse stays clear
    other = v[0]
    assert other[6] == 0.0


def test_smell_stronger_when_closer_and_blocked_by_obstacle():
    w = straight_world()
    w.set_chickens([[35.0, 30.0]])
    near = smell(w)[0]
    w.set_chickens([[45.0, 30.0]])
    far = smell(w)[0]
    assert near > far
    # place an obstacle between head and the chicken -> intensity drops to 0
    w.set_chickens([[45.0, 30.0]])
    w.obstacle_pos = np.array([[38.0, 30.0]]); w.obstacle_r = np.array([2.0]); w.obstacle_kind = np.array([0])
    assert smell(w)[0] == 0.0


def test_smell_gradient_points_forward_to_chicken_ahead():
    w = straight_world()
    w.set_chickens([[40.0, 30.0]])
    g = smell(w)
    assert g[1] > abs(g[2])                       # chicken forward gradient dominates


def test_smell_snake_field_present_and_zero_with_no_rivals():
    w = straight_world()
    g = smell(w)
    assert g.shape == (9,)
    assert g[3] == 0.0 and g[4] == 0.0 and g[5] == 0.0    # no rival -> snake smell is 0
    assert g[6] == 0.0 and g[7] == 0.0 and g[8] == 0.0    # no corpse -> corpse smell is 0


def test_smell_snake_field_points_toward_rival():
    w = straight_world()
    _add_rival(w, [40.0, 30.0])                    # rival ahead (+x)
    g = smell(w)
    assert g[3] > 0.0                               # snake_intensity
    assert g[4] > abs(g[5])                         # snake_grad_fwd dominates


def test_smell_stronger_when_closer_and_blocked_by_obstacle_corpse():
    w = straight_world()
    w.corpses = {"pos": np.array([[35.0, 30.0]]), "food": np.array([5.0])}
    near = smell(w)[6]
    w.corpses = {"pos": np.array([[45.0, 30.0]]), "food": np.array([5.0])}
    far = smell(w)[6]
    assert near > far > 0.0
    # place an obstacle between head and the corpse -> intensity drops to 0
    w.obstacle_pos = np.array([[38.0, 30.0]]); w.obstacle_r = np.array([2.0]); w.obstacle_kind = np.array([0])
    assert smell(w)[6] == 0.0


def test_smell_corpse_gradient_points_forward_to_corpse_ahead():
    w = straight_world()
    w.corpses = {"pos": np.array([[40.0, 30.0]]), "food": np.array([5.0])}
    g = smell(w)
    assert g[7] > abs(g[8])                        # corpse forward gradient dominates


def test_rival_dead_ahead_social_channel():
    w = straight_world()
    _add_rival(w, [45.0, 30.0])
    o = observe(w)
    social = o[99:110]                               # social block widened 7 -> 11
    assert social[0] == 1.0                          # has_rival
    assert social[1] > 0                              # rel_pos_fwd (rival ahead)
    assert (np.abs(social) <= 1.0).all()


def test_no_rival_social_channel_is_zero():
    w = straight_world()
    o = observe(w)
    social = o[99:110]
    assert social[0] == 0.0
    assert (social == 0.0).all()


def test_self_owned_egg_ahead_egg_channel():
    w = straight_world()
    w.snakes[0].id = 0
    w.eggs = {"pos": np.array([[35.0, 30.0]]), "timer": np.array([10.0]), "owner": np.array([[0, 1]])}
    o = observe(w)
    egg = o[110:114]                                  # egg block after the 11-wide social
    assert egg[0] == 1.0 and egg[3] == 1.0            # has_egg, is_mine
    assert egg[1] > 0                                 # egg ahead


def test_foreign_egg_is_mine_zero():
    w = straight_world()
    w.snakes[0].id = 5
    w.eggs = {"pos": np.array([[35.0, 30.0]]), "timer": np.array([10.0]), "owner": np.array([[0, 1]])}
    o = observe(w)
    egg = o[110:114]
    assert egg[0] == 1.0 and egg[3] == 0.0


def test_no_egg_channel_is_zero():
    w = straight_world()
    o = observe(w)
    assert (o[110:114] == 0.0).all()


def test_arrival_egg_is_uneatable_but_foreign_egg_shows():
    # an arrival egg (owner [-1,-1]) is not yet a real egg -> egg channel stays empty (has_egg=0);
    # a real foreign egg -> has_egg=1.
    w = straight_world(); w.snakes[0].id = 5
    w.eggs = {"pos": np.array([[35.0, 30.0]]), "timer": np.array([10.0]), "owner": np.array([[-1, -1]])}
    assert observe(w)[110] == 0.0                         # arrival egg: uneatable, unsensed by egg channel
    w.eggs = {"pos": np.array([[35.0, 30.0]]), "timer": np.array([10.0]), "owner": np.array([[0, 1]])}
    assert observe(w)[110] == 1.0                         # real foreign egg: sensed


def test_ray_detects_rival_head_as_other_body_distinct_from_self_and_chicken():
    w = straight_world()
    _add_rival(w, [45.0, 30.0])
    v = sense_vision(w)
    center = v[CFG.n_rays // 2]
    assert center[4] == 1.0                           # is_other_body
    assert center[2] == 0.0 and center[3] == 0.0       # not chicken, not self


def test_ray_detects_egg():
    w = straight_world()
    w.eggs = {"pos": np.array([[40.0, 30.0]]), "timer": np.array([10.0]), "owner": np.array([[9, 8]]),
              "genome": np.full((1, gm.GENE_COUNT), 0.5, np.float32), "lineage": np.array([0])}
    v = sense_vision(w)
    center = v[CFG.n_rays // 2]
    assert center[5] == 1.0                           # is_egg


def test_repro_ready_toggles_on_three_way_gate():
    w = straight_world()
    s = w.snakes[0]
    gate = CFG.repro_length_frac * s.phenotype.max_length   # size-RELATIVE length gate (Task 8)
    s.energy = CFG.repro_energy_frac * CFG.energy_max + 1
    s.target_length = gate + 1.0
    s.repro_cooldown = 0
    assert observe(w)[129] == 1.0                       # all three gates pass (proprio repro_ready @ 126+3)
    s.repro_cooldown = 5
    assert observe(w)[129] == 0.0                        # cooldown blocks it
    s.repro_cooldown = 0
    s.energy = 0.0
    assert observe(w)[129] == 0.0                        # energy gate blocks it
    s.energy = CFG.repro_energy_frac * CFG.energy_max + 1
    s.target_length = gate - 1.0
    assert observe(w)[129] == 0.0                        # length gate blocks it


def test_size_ratio_capped_at_one_for_max_length_rival():
    w = straight_world()
    _add_rival(w, [45.0, 30.0], target_length=CFG.length_cap)
    o = observe(w)
    assert o[104] == 1.0                                # size_ratio (social[5] @ 99+5)


def test_observation_space_contains_generated_multisnake_world():
    from snake_rl.env import SnakeEnv
    env = SnakeEnv(seed=0)
    w = generate_world(CFG, seed=21, size=(140.0, 140.0), n_snakes=CFG.n_max)
    w.corpses = {"pos": np.array([wrap(w.snakes[0].head + np.array([3.0, 0.0]), w.size)]), "food": np.array([5.0])}
    for s in w.snakes:
        o = observe(w, s)
        assert env.observation_space.contains(o), f"snake {s.id} obs out of bounds"


def test_observation_space_contains_crowded_stress_world():
    # crowd chickens + eggs near the ego to push smell/social channels toward their bounds
    from snake_rl.env import SnakeEnv
    env = SnakeEnv(seed=0)
    w = generate_world(CFG, seed=22, size=(140.0, 140.0), n_snakes=CFG.n_max)
    ego = w.snakes[0]
    positions = [wrap(ego.head + np.array([1.0 + 0.3 * i, 0.0]), w.size) for i in range(CFG.chicken_ceiling)]
    w.set_chickens(positions)
    w.eggs = {"pos": np.array([wrap(ego.head + d, w.size) for d in
                               ([2.0, 0.0], [0.0, 2.0], [-2.0, 0.0])]),
              "timer": np.array([10.0, 10.0, 10.0]), "owner": np.array([[0, 1], [1, 2], [2, 3]])}
    for s in w.snakes:
        o = observe(w, s)
        assert env.observation_space.contains(o), f"snake {s.id} obs out of bounds"


def test_per_snake_ray_range_gates_obstacle_detection():
    # obstacle sits at a hit-distance of 20: INSIDE gene_rayrange_hi (26) but OUTSIDE
    # gene_rayrange_lo (14). Only a per-snake (genome-resolved) ray_range can tell these two
    # genomes apart here -- a reversion to the old global CFG.ray_range (20.0, itself >= the
    # 20 hit-distance) would see the obstacle in BOTH cases, so this fails if that line reverts.
    w = straight_world()
    w.obstacle_pos = np.array([[52.0, 30.0]]); w.obstacle_r = np.array([1.0]); w.obstacle_kind = np.array([0])
    center_idx = CFG.n_rays // 2

    hi = np.zeros(gm.GENE_COUNT, np.float32); hi[gm.SENSES] = 1.0   # long ray_range, weak smell
    w.snakes[0] = w._make_snake(w.snakes[0].head, 0.0, genome=hi, sex=0, lineage=1,
                                id=w.snakes[0].id, color_seed=1, energy=CFG.energy_max,
                                target_length=CFG.start_length, rng=w.rng)
    assert w.snakes[0].phenotype.ray_range == CFG.gene_rayrange_hi
    v_hi = sense_vision(w)[center_idx]
    assert v_hi[1] == 1.0                                 # long sight: obstacle IS seen

    lo = np.zeros(gm.GENE_COUNT, np.float32)               # SENSES=0 -> gene_rayrange_lo
    w.snakes[0] = w._make_snake(w.snakes[0].head, 0.0, genome=lo, sex=0, lineage=1,
                                id=w.snakes[0].id, color_seed=1, energy=CFG.energy_max,
                                target_length=CFG.start_length, rng=w.rng)
    assert w.snakes[0].phenotype.ray_range == CFG.gene_rayrange_lo
    v_lo = sense_vision(w)[center_idx]
    assert v_lo[1] == 0.0                                 # short sight: SAME obstacle now out of range
    assert v_lo[0] == 1.0                                 # ray reports max range (no hit)


def test_smell_clip_and_reach_bound_live_rival_intensity():
    # generate_world(arrivals=False) -> founders are LIVE snakes (not incubating eggs, unlike the
    # trivial predecessor test which used arrivals=True and left the snake-smell channel at 0
    # always). Cluster the other n_max-1 rivals right on the observer's head, and give the
    # observer a low-SENSES genome (smell_reach = gene_smell_lo = 1.4x, the max).
    w = generate_world(CFG, seed=7, size=(140.0, 140.0), n_snakes=CFG.n_max, arrivals=False)
    w.obstacle_pos = np.zeros((0, 2)); w.obstacle_r = np.zeros((0,))   # no LOS occlusion -> isolate reach/clip
    obs = w.snakes[0]
    lo = np.zeros(gm.GENE_COUNT, np.float32)               # SENSES=0 -> max smell reach
    w.snakes[0] = w._make_snake(obs.head, obs.heading, genome=lo, sex=0, lineage=1, id=obs.id,
                                color_seed=1, energy=CFG.energy_max, target_length=CFG.start_length,
                                rng=w.rng)
    obs = w.snakes[0]
    assert obs.phenotype.smell_reach == CFG.gene_smell_lo

    angles = np.linspace(0, 2 * np.pi, len(w.snakes) - 1, endpoint=False)
    r = 0.05
    for rival, a in zip(w.snakes[1:], angles):             # park every rival tight on the observer's head
        rival.head = wrap(obs.head + r * np.array([np.cos(a), np.sin(a)]), w.size)
        rival.head_uw = rival.head.copy()

    si = smell(w, obs)[3]                                  # snake_intensity channel
    assert np.isfinite(si) and 0.0 <= si <= CFG.chicken_ceiling

    # raw, un-clipped snake intensity computed independently (mirrors _smell_field's formula with
    # no obstacles to occlude): sum of smell_reach/(1+dist) over the clustered rivals.
    dists = np.array([np.linalg.norm(torus_delta(rv.head, obs.head, w.size)) for rv in w.snakes[1:]])
    raw = float((obs.phenotype.smell_reach / (1.0 + dists)).sum())
    unscaled_raw = float((1.0 / (1.0 + dists)).sum())
    # NOTE: at today's n_max=6 (5 rivals), even at max reach (1.4x) and r->0 each term is bounded by
    # 1, so raw tops out near 5*1.4=6.7 -- structurally short of chicken_ceiling=12 (unlike the corpse
    # field, which has no population cap -- see test_observation_space_contains_corpse_pileup_beyond_
    # ceiling). So the clip can't be forced to bite via rival smell alone until Task 11 raises n_max;
    # what we CAN prove here is that the per-snake reach scaling is genuinely wired in (si tracks the
    # reach-scaled raw, not the unscaled raw) and the clip is a safe finite no-op in the meantime.
    assert abs(si - raw) < 1e-4                            # si == reach-scaled raw (clip didn't need to fire)
    assert raw > unscaled_raw * 1.3                        # reach (1.4x) demonstrably inflates the raw sum


def test_observation_space_contains_corpse_pileup_beyond_ceiling():
    # corpses have no structural population cap (unlike chickens/rivals) -- pack MORE than
    # chicken_ceiling of them right next to the ego's head so the RAW smell sum genuinely
    # crosses chicken_ceiling (not just "stays under it"), proving sensors.smell's defensive
    # clip actually engages rather than being a no-op safety margin.
    from snake_rl.env import SnakeEnv
    env = SnakeEnv(seed=0)
    w = generate_world(CFG, seed=23, size=(140.0, 140.0), n_snakes=CFG.n_max)
    w.obstacle_pos = np.zeros((0, 2)); w.obstacle_r = np.zeros((0,))   # no LOS occlusion -> the raw
                                                                       # corpse smell sum genuinely
                                                                       # exceeds the ceiling (isolate the clip)
    ego = w.snakes[0]
    ego.heading = 0.0                                                  # face +x so the +x corpse line is
                                                                       # straight ahead -> grad_fwd is maximal
                                                                       # (deterministic, not RNG-heading dependent)
    n = CFG.chicken_ceiling * 3
    positions = np.array([wrap(ego.head + np.array([0.05 + 0.03 * i, 0.0]), w.size) for i in range(n)])
    w.corpses = {"pos": positions, "food": np.full(n, 5.0)}
    for s in w.snakes:
        o = observe(w, s)
        assert env.observation_space.contains(o), f"snake {s.id} obs out of bounds"
    ego_obs = observe(w, ego)
    # raw (unclipped) intensity/fwd-gradient here are ~23.8/~12.5 -- both above chicken_ceiling=12
    # -- so landing exactly on the ceiling proves the clip fired, not that we stayed under it.
    assert ego_obs[120] == CFG.chicken_ceiling            # corpse_intensity clipped (smell@114, corpse_int=+6)
    assert ego_obs[121] == CFG.chicken_ceiling            # corpse_grad_fwd clipped to the ceiling
