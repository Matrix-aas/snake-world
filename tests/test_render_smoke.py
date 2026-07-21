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


def test_ray_kind_covers_other_body_egg_and_corpse():
    # B2 added ray kinds 3 (other_body) and 4 (egg); corpse-sensing adds kind 5. The sensor
    # overlay must not KeyError on any of them.
    assert set(RAY_KIND) == {-1, 0, 1, 2, 3, 4, 5}


def test_render_draws_multisnake_world_with_eggs_and_corpses():
    w = generate_world(CFG, seed=1, size=(120.0, 120.0), n_snakes=3)
    w.eggs = {"pos": np.array([[60.0, 60.0]]), "timer": np.array([20.0]), "owner": np.array([[0, 1]])}
    w._spawn_corpse(w.snakes[0])
    r = Renderer(scale=4)
    r.draw(w, follow_id=w.snakes[0].id)
    r.draw(w, follow_id=None)          # overview (no explicit follow) must also work
    r.close()


def test_render_sensors_handle_other_body_and_egg_ray_kinds():
    # Force a ray to hit another snake's body (kind 3) and a DIFFERENT ray to hit an egg (kind 4)
    # -- this is the exact scenario that KeyErrors if RAY_KIND isn't extended for B2's new kinds.
    # The egg sits off to the side (perp), not in front (fwd), so it doesn't occlude the other
    # snake's head on the shared forward ray -- both kinds must actually appear, not just not-crash.
    from snake_rl.sensors import vision_distances
    w = generate_world(CFG, seed=1, size=(120.0, 120.0), n_snakes=2)
    ego = w.snakes[0]
    fwd = ego.heading_vec()
    perp = np.array([-fwd[1], fwd[0]])
    w.snakes[1].head_uw = ego.head_uw + fwd * 5.0
    w.snakes[1].head = wrap(w.snakes[1].head_uw, w.size)
    w.snakes[1].path_uw = [w.snakes[1].head_uw.copy()]
    w.eggs = {"pos": ((ego.head_uw + perp * 3.0) % w.size)[None].copy(),
              "timer": np.array([10.0]), "owner": np.array([[5, 6]])}
    _, _, kinds = vision_distances(w, ego.head, ego.heading)
    assert 3 in kinds and 4 in kinds        # the scene actually produces both kinds, as claimed
    r = Renderer(scale=4, show_sensors=True)
    r.draw(w)      # must not KeyError
    r.close()


def test_render_sensors_handle_corpse_ray_kind():
    # Force a ray to hit a corpse (kind 5) -- KeyErrors if RAY_KIND isn't extended for it.
    from snake_rl.sensors import vision_distances
    w = generate_world(CFG, seed=1, size=(120.0, 120.0), n_snakes=2)
    ego = w.snakes[0]
    fwd = ego.heading_vec()
    w.corpses = {"pos": ((ego.head_uw + fwd * 5.0) % w.size)[None].copy(), "food": np.array([5.0])}
    _, _, kinds = vision_distances(w, ego.head, ego.heading)
    assert 5 in kinds                       # the scene actually produces the corpse kind, as claimed
    r = Renderer(scale=4, show_sensors=True)
    r.draw(w)      # must not KeyError
    r.close()
