import numpy as np
from gymnasium.utils.env_checker import check_env
from snake_rl.config import CFG
from snake_rl.env import SnakeEnv
from snake_rl.sensors import observe, OBS_DIM


def test_gymnasium_check_env():
    check_env(SnakeEnv(seed=0), skip_render_check=True)


def test_env_obs_space_matches_layout_for_extreme_genomes():
    env = SnakeEnv(seed=2)
    assert env.observation_space.shape[0] == OBS_DIM * env.cfg.frame_stack or \
           env.observation_space.shape[0] == OBS_DIM  # single-frame space; framestack is external
    from snake_rl import genome as gm
    env.reset()                      # world is None until reset() (review C3)
    w = env.world
    for gval in (0.0, 1.0):
        g = np.full(gm.GENE_COUNT, gval, np.float32)
        s = w.snakes[0]
        w.snakes[0] = w._make_snake(s.head, 0.0, genome=g, sex=0, lineage=1, id=s.id,
                                    color_seed=1, energy=env.cfg.energy_max,
                                    target_length=env.cfg.start_length, rng=w.rng)
        obs = observe(w, w.snakes[0]).astype(np.float32)
        assert env.observation_space.contains(obs)


def test_reset_randomizes_genomes_across_the_founder_population():
    # Domain randomization (spec §8): every founder snake AND every arrival egg gets a freshly
    # sampled genome, so the brain learns to read the whole gene box. arrivals=True => snakes holds
    # the one live gradient-ego; the other founders are PENDING EGGS -- vary across both.
    env = SnakeEnv(seed=4)
    env.reset()
    genomes = [s.genome for s in env.world.snakes]
    if len(env.world.eggs.get("genome", [])):
        genomes += list(env.world.eggs["genome"])
    assert len({tuple(np.round(g, 4)) for g in genomes}) >= 2


def test_egg_lost_reward_only_for_ego_owned_eggs():
    import dataclasses
    env = SnakeEnv(seed=4)
    env.reset()
    env.cfg = dataclasses.replace(env.cfg, reward_egg_lost=-4.0)   # Config is frozen
    ego = env.world.snakes[0]
    assert env._egg_lost_reward([frozenset({ego.id, 123})]) == -4.0   # ego co-owns -> penalized
    assert env._egg_lost_reward([frozenset({999, 123})]) == 0.0       # foreign egg -> nothing
    assert env._egg_lost_reward([]) == 0.0                            # nothing eaten -> nothing


def test_reset_seed_deterministic():
    o1, _ = SnakeEnv(seed=7).reset(seed=7)
    o2, _ = SnakeEnv(seed=7).reset(seed=7)
    np.testing.assert_allclose(o1, o2)


def test_truncates_at_horizon():
    env = SnakeEnv(seed=0); env.reset()
    term = trunc = False
    for _ in range(CFG.episode_horizon + 5):
        _, _, term, trunc, _ = env.step([1, 1, 0])
        if term or trunc:
            break
    assert trunc or term


def test_eat_gives_positive_reward():
    env = SnakeEnv(seed=0); env.reset()
    w = env.world
    w.head = np.array([30.0, 30.0]); w.head_uw = w.head.copy(); w.heading = 0.0
    w.obstacle_pos = np.zeros((0, 2)); w.obstacle_r = np.zeros((0,)); w.obstacle_kind = np.zeros((0,), int)
    w.set_chickens([[31.0, 30.0]])          # lands on the post-move head cell (dist 0 -> wanders, not fled) -> eaten
    _, r, _, _, info = env.step([3, 1, 0])
    assert info["ate"] == 1 and r > 5


def test_pbrs_closed_loop_nets_zero_when_gamma_one():
    # PBRS is policy-invariant on the DISCOUNTED return; a closed loop nets exactly 0 only at gamma=1.
    from dataclasses import replace
    env = SnakeEnv(cfg=replace(CFG, gamma=1.0), seed=0); env.reset()
    w = env.world
    w.set_chickens([[45.0, 30.0]])
    w.head = np.array([30.0, 30.0]); w.head_uw = w.head.copy()
    env._last_phi = env._phi(); env._last_ids = frozenset(int(i) for i in w.chicken_id)
    total = 0.0
    for p in [np.array([31.0, 30.0]), np.array([30.0, 30.0])]:   # out and back to same phi
        w.head = p; w.head_uw = p.copy()
        total += env._shaping()
    assert abs(total) < 1e-9                 # telescopes exactly at gamma=1


def test_pbrs_zeroes_on_set_change_not_on_nearest_switch():
    # nearest-distance is continuous as the nearest identity switches among a fixed set -> shaping is PAID;
    # only a change to the chicken SET (eat/spawn) zeroes it.
    env = SnakeEnv(seed=0); env.reset()
    w = env.world
    w.obstacle_pos = np.zeros((0, 2)); w.obstacle_r = np.zeros((0,))   # isolate the chicken term
    w.set_chickens([[45.0, 30.0], [15.0, 30.0]])
    w.head = np.array([31.0, 30.0]); w.head_uw = w.head.copy()     # nearest = chicken@45 (d14 < d16)
    env._last_phi = env._phi(); env._last_ids = frozenset(int(i) for i in w.chicken_id)
    w.head = np.array([25.0, 30.0]); w.head_uw = w.head.copy()     # nearest switches to @15, SAME set
    assert env._shaping() != 0.0                                   # continuous shaping paid, not zeroed
    w.chicken_pos = w.chicken_pos[:1]; w.chicken_id = w.chicken_id[:1]   # a chicken removed (eaten)
    assert env._shaping() == 0.0                                   # set changed -> zeroed
