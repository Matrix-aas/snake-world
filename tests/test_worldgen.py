import numpy as np
from snake_rl.config import CFG
from snake_rl.worldgen import generate_world
from snake_rl.world import torus_dist


def test_generate_is_deterministic_by_seed():
    a = generate_world(CFG, seed=42); b = generate_world(CFG, seed=42)
    np.testing.assert_allclose(a.size, b.size)
    np.testing.assert_allclose(a.obstacle_pos, b.obstacle_pos)


def test_has_chicken_and_obstacles_in_range():
    w = generate_world(CFG, seed=1)
    assert len(w.chicken_pos) >= 1
    assert CFG.n_obstacles_min <= len(w.obstacle_pos) <= CFG.n_obstacles_max
    assert (CFG.world_size_min <= w.size).all() and (w.size <= CFG.world_size_max).all()


def test_no_obstacle_on_snake_start():
    w = generate_world(CFG, seed=3)
    d = torus_dist(w.obstacle_pos, w.head, w.size)
    assert (d > w.obstacle_r + CFG.head_radius).all()


def test_generate_world_multi_snake_spread():
    from snake_rl.worldgen import generate_world
    from snake_rl.world import torus_dist
    import numpy as np
    w = generate_world(__import__("snake_rl.config", fromlist=["CFG"]).CFG,
                       seed=7, size=(140.0, 140.0), n_snakes=4)
    assert len(w.snakes) == 4
    assert [s.id for s in w.snakes] == [0, 1, 2, 3]
    heads = np.array([s.head for s in w.snakes])
    # no two snakes spawn on top of each other
    for i in range(4):
        for j in range(i + 1, 4):
            assert torus_dist(heads[i][None], heads[j], w.size)[0] > 2.0


def test_generate_world_arrivals_ego_live_others_as_eggs():
    # Goal 1: with arrivals=True only the ego (id 0) is a LIVE snake at step 0; every OTHER snake
    # ARRIVES via a guaranteed egg (owner -1), placed spread-out where a snake would have spawned.
    w = generate_world(CFG, seed=7, size=(140.0, 140.0), n_snakes=4, arrivals=True)
    assert len(w.snakes) == 1 and w.snakes[0].alive and w.snakes[0].id == 0
    owner = w.eggs["owner"]
    assert int((owner[:, 0] < 0).sum()) == 3                    # 3 guaranteed arrival eggs (n_snakes - 1)
    assert w.chicken_sky is True                                # runtime chickens will drop from the sky
    pts = np.vstack([w.eggs["pos"], w.snakes[0].head_uw[None]])  # eggs + ego spread apart
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            assert torus_dist(pts[i][None], pts[j], w.size)[0] > 2.0


def test_generate_world_default_is_single_and_centered():
    from snake_rl.worldgen import generate_world
    from snake_rl.config import CFG
    import numpy as np
    w = generate_world(CFG, seed=7, size=(80.0, 80.0))       # n_snakes defaults to 1
    assert len(w.snakes) == 1
    assert np.allclose(w.snakes[0].head_uw, np.array(w.size) / 2.0)
