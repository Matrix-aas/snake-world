import numpy as np
from snake_rl.config import CFG
from snake_rl.world import World, Snake, wrap
from snake_rl.worldgen import generate_world
from snake_rl.sensors import observe, sense_vision, smell, OBS_DIM


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


def test_obs_shape_and_bounds():
    w = straight_world(); w.maybe_spawn_forced()
    o = observe(w)
    assert OBS_DIM == 87
    assert o.shape == (OBS_DIM,) and o.dtype == np.float32
    assert np.isfinite(o).all()
    assert (o[:63].reshape(9, 7)[:, 0] >= 0).all() and (o[:63].reshape(9, 7)[:, 0] <= 1).all()


def test_center_ray_sees_obstacle_ahead():
    w = straight_world()
    w.obstacle_pos = np.array([[40.0, 30.0]]); w.obstacle_r = np.array([1.0]); w.obstacle_kind = np.array([0])
    v = sense_vision(w)
    center = v[CFG.n_rays // 2]
    assert center[1] == 1.0                       # obstacle channel
    # distance until the head EDGE touches: 40 - (obstacle_r + head_radius) - 30, normalized
    assert abs(center[0] - (40 - 1 - CFG.head_radius - 30) / CFG.ray_range) < 1e-2


def test_empty_ray_encoding():
    w = straight_world()
    v = sense_vision(w)
    assert (v == np.array([1.0, 0, 0, 0, 0, 0, 0])).all()


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
    social = o[63:70]
    assert social[0] == 1.0                          # has_rival
    assert social[1] > 0                              # rel_pos_fwd (rival ahead)
    assert (np.abs(social) <= 1.0).all()


def test_no_rival_social_channel_is_zero():
    w = straight_world()
    o = observe(w)
    social = o[63:70]
    assert social[0] == 0.0
    assert (social == 0.0).all()


def test_self_owned_egg_ahead_egg_channel():
    w = straight_world()
    w.snakes[0].id = 0
    w.eggs = {"pos": np.array([[35.0, 30.0]]), "timer": np.array([10.0]), "owner": np.array([[0, 1]])}
    o = observe(w)
    egg = o[70:74]
    assert egg[0] == 1.0 and egg[3] == 1.0            # has_egg, is_mine
    assert egg[1] > 0                                 # egg ahead


def test_foreign_egg_is_mine_zero():
    w = straight_world()
    w.snakes[0].id = 5
    w.eggs = {"pos": np.array([[35.0, 30.0]]), "timer": np.array([10.0]), "owner": np.array([[0, 1]])}
    o = observe(w)
    egg = o[70:74]
    assert egg[0] == 1.0 and egg[3] == 0.0


def test_no_egg_channel_is_zero():
    w = straight_world()
    o = observe(w)
    assert (o[70:74] == 0.0).all()


def test_ray_detects_rival_head_as_other_body_distinct_from_self_and_chicken():
    w = straight_world()
    _add_rival(w, [45.0, 30.0])
    v = sense_vision(w)
    center = v[CFG.n_rays // 2]
    assert center[4] == 1.0                           # is_other_body
    assert center[2] == 0.0 and center[3] == 0.0       # not chicken, not self


def test_ray_detects_egg():
    w = straight_world()
    w.eggs = {"pos": np.array([[40.0, 30.0]]), "timer": np.array([10.0]), "owner": np.array([[9, 8]])}
    v = sense_vision(w)
    center = v[CFG.n_rays // 2]
    assert center[5] == 1.0                           # is_egg


def test_repro_ready_toggles_on_three_way_gate():
    w = straight_world()
    s = w.snakes[0]
    s.energy = CFG.repro_energy_frac * CFG.energy_max + 1
    s.target_length = CFG.repro_length_min + 1
    s.repro_cooldown = 0
    assert observe(w)[86] == 1.0                       # all three gates pass
    s.repro_cooldown = 5
    assert observe(w)[86] == 0.0                        # cooldown blocks it
    s.repro_cooldown = 0
    s.energy = 0.0
    assert observe(w)[86] == 0.0                        # energy gate blocks it
    s.energy = CFG.repro_energy_frac * CFG.energy_max + 1
    s.target_length = CFG.repro_length_min - 1
    assert observe(w)[86] == 0.0                        # length gate blocks it


def test_size_ratio_capped_at_one_for_max_length_rival():
    w = straight_world()
    _add_rival(w, [45.0, 30.0], target_length=CFG.length_cap)
    o = observe(w)
    assert o[68] == 1.0                                 # size_ratio


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


def test_observation_space_contains_corpse_pileup_beyond_ceiling():
    # corpses have no structural population cap (unlike chickens/rivals) -- pile up MORE than
    # chicken_ceiling of them near the ego to prove the corpse-smell bound holds even then.
    from snake_rl.env import SnakeEnv
    env = SnakeEnv(seed=0)
    w = generate_world(CFG, seed=23, size=(140.0, 140.0), n_snakes=CFG.n_max)
    ego = w.snakes[0]
    n = CFG.chicken_ceiling * 3
    positions = np.array([wrap(ego.head + np.array([0.5 + 0.2 * i, 0.0]), w.size) for i in range(n)])
    w.corpses = {"pos": positions, "food": np.full(n, 5.0)}
    for s in w.snakes:
        o = observe(w, s)
        assert env.observation_space.contains(o), f"snake {s.id} obs out of bounds"
