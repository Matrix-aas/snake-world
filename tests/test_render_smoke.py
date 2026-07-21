import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
from snake_rl.config import CFG
from snake_rl.worldgen import generate_world
from snake_rl.render import Renderer, RAY_KIND, color_for
from snake_rl.world import wrap


def test_render_draws_without_error():
    w = generate_world(CFG, seed=0)
    r = Renderer(scale=6)
    r.draw(w)             # must not raise
    r.toggle_sensors()
    r.draw(w)
    r.close()


def test_color_for_is_deterministic_and_distinct_for_0_to_5():
    colors_a = [color_for(i) for i in range(6)]
    colors_b = [color_for(i) for i in range(6)]
    assert colors_a == colors_b                # deterministic
    assert len(set(colors_a)) == 6              # distinct
    for c in colors_a:
        assert all(0 <= ch <= 255 for ch in c)


def test_ray_kind_covers_other_body_and_egg():
    # B2 added ray kinds 3 (other_body) and 4 (egg); the sensor overlay must not KeyError on them.
    assert set(RAY_KIND) == {-1, 0, 1, 2, 3, 4}


def test_render_draws_multisnake_world_with_eggs_and_corpses():
    w = generate_world(CFG, seed=1, size=(120.0, 120.0), n_snakes=3)
    w.eggs = {"pos": np.array([[60.0, 60.0]]), "timer": np.array([20.0]), "owner": np.array([[0, 1]])}
    w._spawn_corpse(w.snakes[0])
    r = Renderer(scale=4)
    r.draw(w, follow_id=w.snakes[0].id)
    r.draw(w, follow_id=None)          # overview (no explicit follow) must also work
    r.close()


def test_render_sensors_handle_other_body_and_egg_ray_kinds():
    # Force a ray to hit another snake's body (kind 3) and an egg (kind 4) sits nearby (kind 4) --
    # this is the exact scenario that KeyErrors if RAY_KIND isn't extended for B2's new kinds.
    w = generate_world(CFG, seed=1, size=(120.0, 120.0), n_snakes=2)
    ego = w.snakes[0]
    fwd = ego.heading_vec()
    w.snakes[1].head_uw = ego.head_uw + fwd * 5.0
    w.snakes[1].head = wrap(w.snakes[1].head_uw, w.size)
    w.snakes[1].path_uw = [w.snakes[1].head_uw.copy()]
    w.eggs = {"pos": ((ego.head_uw + fwd * 3.0) % w.size)[None].copy(),
              "timer": np.array([10.0]), "owner": np.array([[5, 6]])}
    r = Renderer(scale=4, show_sensors=True)
    r.draw(w)
    r.close()
