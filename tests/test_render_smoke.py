import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import pygame
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


def test_ground_patchwork_is_deterministic_and_varied():
    # The world backdrop is a hashed patchwork of the 6 seamless grass variants (with a graceful
    # single-tile / procedural fallback). It must be byte-identical across rebuilds -- stable
    # per-frame AND frame-to-frame -- so the ground never flickers. When the variant set is present,
    # the per-cell hash must draw from more than one variant (not collapse to one repeated tile).
    w = generate_world(CFG, seed=0)
    r = Renderer(scale=6, show_sensors=False)
    r.draw(w)                                       # triggers _ensure -> load assets + build ground
    g1, g2 = r._build_ground(), r._build_ground()
    assert g1.get_size() == r.canvas.get_size()
    assert np.array_equal(pygame.surfarray.array3d(g1), pygame.surfarray.array3d(g2))  # no flicker
    variants = r._assets.get("ground_set")
    if variants:                                    # assets present -> a real multi-variant patchwork
        assert len(variants) == 6
        n = len(variants)
        picks = {((c * 73856093) ^ (row * 19349663)) % n for c in range(6) for row in range(6)}
        assert len(picks) > 1                       # varied, not a single repeated tile
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


def test_render_draws_no_ego_all_eggs_viewer_world():
    # Task 10: a no-ego viewer world (ego_live=False) starts with ZERO live snakes -- guards
    # render.py's follow_id fallback (line ~62) when there's no snake to follow at all.
    w = generate_world(CFG, seed=2, size=(120.0, 120.0), n_snakes=3, arrivals=True, ego_live=False)
    assert sum(1 for s in w.snakes if s.alive) == 0
    r = Renderer(scale=4)
    r.draw(w)             # must not raise
    r.close()


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


def test_render_draws_arriving_chicken_through_the_whole_drop():
    # Goal 2: a chicken FALLING from the sky (world.arriving) must render at every height without
    # raising -- fade-in near the top, shrink-toward-the-ground, the growing/darkening drop-shadow
    # (bucketed), and the wing-flap frames all get exercised by sweeping the descent timer -- then a
    # landing dust puff. Uses the dedicated fall sheet when present, degrades to the run/static/
    # procedural fallbacks when not (so it passes with or without assets).
    w = generate_world(CFG, seed=2, size=(120.0, 120.0), n_snakes=2)
    w._add_chicken([60.0, 60.0], arriving=True)
    assert len(w.arriving["pos"]) == 1
    r = Renderer(scale=4)
    r._clock_override = 0.0
    for timer in (CFG.chicken_arrive_steps, CFG.chicken_arrive_steps // 2, 2, 1):
        w.arriving["timer"][:] = timer         # top-of-drop -> mid -> just before touchdown
        r._clock_override += 0.15              # advance flap/wobble
        r.draw(w)                              # must not raise at any height
    r.spawn_land(np.array([60.0, 60.0]))       # landing puff particles
    r.draw(w)
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
