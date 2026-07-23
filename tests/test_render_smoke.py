import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import pygame
from snake_rl.config import CFG
from snake_rl.worldgen import generate_world
from snake_rl.render import Renderer, RAY_KIND, color_for, ZOOM_MAX
from snake_rl.world import wrap


def test_render_draws_without_error():
    w = generate_world(CFG, seed=0)
    r = Renderer(scale=6)
    r.draw(w)             # must not raise
    r.toggle_sensors()
    r.draw(w)
    r.close()


def test_ground_tile_is_sprite_based_deterministic_and_grassy():
    # The ground tile is built from the codex-vision grass sprites (ground_0..5): a 2x2 tone-matched
    # patchwork. It must load the sprites, be deterministic across rebuilds (no flicker), be a REAL
    # multi-tone texture (not a flat/binary fill), and read as green turf. (Seamless camera-locked
    # coverage is covered separately by test_ground_fully_covers_canvas_at_all_camera_offsets_and_zooms.)
    w = generate_world(CFG, seed=0)
    r = Renderer(scale=6)
    r.draw(w)                                       # triggers _ensure -> builds _ground_tile0 from sprites
    assert r._assets.get("ground_set"), "grass sprites (ground_0..5) must be loaded"
    a1 = pygame.surfarray.array3d(r._build_ground_tile())
    a2 = pygame.surfarray.array3d(r._build_ground_tile())
    assert np.array_equal(a1, a2)                                   # deterministic -> no flicker
    assert len(np.unique(a1.reshape(-1, 3), axis=0)) > 200          # a real texture, not a flat/binary fill
    assert a1[..., 1].std() > 3                                     # green channel actually varies (texture)
    g, rd, bl = a1[..., 1].mean(), a1[..., 0].mean(), a1[..., 2].mean()
    assert g > rd and g > bl                                        # reads as green turf
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


def test_anchored_line_does_not_split_across_the_camera_antipode():
    # Root-cause guard for the stray red "ray that lags": a short line (snake tongue / vision ray) drawn
    # for an entity near the camera's torus ANTIPODE must NOT split its endpoints a full period apart.
    from snake_rl.render import TONGUE
    w = generate_world(CFG, seed=1, size=(120.0, 120.0), n_snakes=1)
    r = Renderer(scale=4)
    r.draw(w, cam_center=(60.0, 60.0), zoom=2.0)                 # camera at world center
    head = np.array([0.5, 60.0]); tip = head + np.array([-1.35, 0.0])   # head at the camera's antipode
    assert abs(r._p(head)[0] - r._p(tip)[0]) > 0.4 * r._perx      # naive per-point transform SPLITS (the bug)
    a, b = r._line_anchored(TONGUE, head, head, tip, 2)          # anchored keeps the endpoints adjacent (fix)
    assert abs(a[0] - b[0]) < 6 * r._scale and abs(a[1] - b[1]) < 6 * r._scale
    r.close()


def test_gore_burst_lands_at_camera_correct_screen_position():
    # Guard: a red eat-burst spawned at a known world position lands at the CAMERA-correct screen spot
    # (routed through _world_to_canvas), not the old un-offset fit-to-screen spot.
    w = generate_world(CFG, seed=1, size=(120.0, 120.0), n_snakes=1)
    r = Renderer(scale=4)
    P = np.array([30.0, 40.0])
    r.draw(w, cam_center=(40.0, 60.0), zoom=1.5)                 # offset+zoom, burst stays on-canvas
    r.canvas.fill((0, 0, 0)); r._particles = []
    r.spawn_eat(P); r._draw_particles()                          # red blood burst at P
    arr = pygame.surfarray.array3d(r.canvas)                     # [x, y, rgb]
    red = (arr[:, :, 0] > 120) & (arr[:, :, 1] < 90) & (arr[:, :, 2] < 90)
    assert red.any()
    xs, ys = np.where(red); cx, cy = xs.mean(), ys.mean()
    want = r._world_to_canvas(P)
    naive = (P[0] * r._base_scale, P[1] * r._base_scale)         # un-offset (what the camera bug would show)
    dw = float(np.hypot(cx - want[0], cy - want[1]))
    dn = float(np.hypot(cx - naive[0], cy - naive[1]))
    assert dw < 70 and dw < dn                                   # near camera-correct, far from un-offset
    r.close()


def test_vignette_alpha_increases_toward_edges():
    # Polish #4: the cinematic vignette is transparent at center and darkens toward the corners.
    w = generate_world(CFG, seed=0, size=(120.0, 120.0), n_snakes=1)
    r = Renderer(scale=5); r.draw(w)
    a = pygame.surfarray.array_alpha(r._vignette)
    cw, ch = r._vignette.get_size()
    assert a[cw // 2, ch // 2] == 0                         # center fully transparent
    assert a[2, 2] > a[cw // 2, ch // 2] and a[2, 2] > 20   # corner darkened
    r.close()


def test_ground_fully_covers_canvas_at_all_camera_offsets_and_zooms():
    # Phase B polish #3: the camera-locked ground must FULLY cover the canvas at every offset/zoom --
    # no gaps (the red-seam bug was uncovered clear-color bleeding through misaligned tiles). Fill the
    # canvas with a magenta sentinel, redraw JUST the ground, and assert not one sentinel pixel remains.
    w = generate_world(CFG, seed=3, size=(190.0, 150.0), n_snakes=2)   # non-square world (aspect stress)
    r = Renderer(scale=4)
    SENT = (255, 0, 255)
    for cam, zoom in [((0.0, 0.0), 1.0), ((95.0, 75.0), 1.0), ((10.0, 140.0), 2.3),
                      ((180.0, 5.0), 4.0), ((47.3, 88.9), 1.7), ((190.0, 150.0), 6.0)]:
        r.draw(w, cam_center=cam, zoom=zoom)      # sets the camera transform state
        r.canvas.fill(SENT)
        r._draw_ground()
        arr = pygame.surfarray.array3d(r.canvas)
        gap = (arr[:, :, 0] == 255) & (arr[:, :, 1] == 0) & (arr[:, :, 2] == 255)
        assert not np.any(gap), f"ground left {int(gap.sum())} uncovered px at cam={cam} zoom={zoom}"
    r.close()


def test_render_camera_offset_zoom_and_zero_live_snakes():
    # Phase B increment 1: draw through the camera transform (offset + zoom), including a 0-live-snakes
    # all-eggs world, must not crash; cam_center maps to the canvas center; zoom clamps to [1, ZOOM_MAX].
    w = generate_world(CFG, seed=2, size=(120.0, 90.0), n_snakes=3, arrivals=True, ego_live=False)
    assert sum(1 for s in w.snakes if s.alive) == 0            # all-eggs start: no live snake to follow
    r = Renderer(scale=5)
    r.draw(w, cam_center=(30.0, 40.0), zoom=3.0)              # free/follow: offset + zoom, 0 live
    cx, cy = r._world_to_canvas((30.0, 40.0))                 # camera center lands at canvas center
    assert abs(cx - r.cw / 2) < 1e-6 and abs(cy - r.ch / 2) < 1e-6
    r.draw(w)                                                  # default (overview fit-to-screen)
    r.draw(w, zoom=999.0)
    assert r.zoom == ZOOM_MAX                                  # clamped
    # a live multi-snake world zoomed in near a snake's head must also render cleanly
    w2 = generate_world(CFG, seed=1, size=(120.0, 120.0), n_snakes=3)
    r.draw(w2, cam_center=tuple(w2.snakes[0].head_uw), zoom=4.0, follow_id=w2.snakes[0].id)
    r.close()


def test_snake_palette_differs_by_lineage_and_genome():
    # Phase B increment 2: genome -> visible phenotype. Different lineage (family hue) OR different
    # aggression/metabolism genes must yield a different color; same genome+lineage is stable.
    from snake_rl.render import _snake_palette
    from snake_rl.genome import AGGRESSION, METABOLISM
    w = generate_world(CFG, seed=1, size=(120.0, 120.0), n_snakes=2)
    a, b = w.snakes[0], w.snakes[1]
    a.lineage, b.lineage = 3, 7                                  # distinct families
    a.genome = np.full(9, 0.5, np.float32); b.genome = a.genome.copy()
    assert _snake_palette(a) == _snake_palette(a)               # deterministic
    assert _snake_palette(a)[0] != _snake_palette(b)[0]         # lineage -> different family hue
    b.lineage = 3                                                # same family now...
    b.genome[AGGRESSION] = 1.0; b.genome[METABOLISM] = 1.0      # ...but louder genes
    assert _snake_palette(a) != _snake_palette(b)               # genes still shift color
    r = Renderer(scale=4)
    r.draw(w, follow_id=a.id)                                    # renders (size-gene scaled body) w/o crash
    r.close()


def test_inspector_overlay_renders_for_followed_snake():
    # Phase B increment 3: the genome inspector overlay (toggle I) must render for the followed snake
    # -- 9 gene bars + sex + age + lineage swatch + life stats -- without crashing, and be a no-op
    # when off or when there's no live snake to follow.
    w = generate_world(CFG, seed=1, size=(120.0, 120.0), n_snakes=2)
    r = Renderer(scale=5, show_inspector=True)
    fid = w.snakes[0].id
    r.draw(w, follow_id=fid, cam_center=tuple(w.snakes[0].head_uw), zoom=3.0,
           inspector_stats={fid: {"kills": 2, "offspring": 3}})    # with stats
    r.draw(w, follow_id=fid)                                        # stats None -> shows zeros, no crash
    r.toggle_inspector()                                           # off
    r.draw(w, follow_id=fid)
    # all-eggs world: no followed snake -> overlay is a silent no-op
    r.show_inspector = True
    we = generate_world(CFG, seed=2, size=(120.0, 120.0), n_snakes=2, arrivals=True, ego_live=False)
    r.draw(we)
    r.close()


def test_inspector_fallen_panel_and_ring_hud_size_gene():
    # Phase B polish #5: during the death-linger the inspector keeps showing the fallen snake (passed
    # via `fallen=`) instead of blanking; and _ring_hud scales its head radius by the size gene like
    # _draw_snake, so a big/small snake's badge tracks its actual drawn head. Both must render clean.
    from snake_rl.genome import SIZE
    w = generate_world(CFG, seed=1, size=(120.0, 120.0), n_snakes=2)
    dead = w.snakes[0]                                          # a stand-in "fallen" snake object
    r = Renderer(scale=5, show_inspector=True, show_rings=True)
    # no live follow target, but a fallen snake is supplied -> panel stays up (tagged fallen)
    r.draw(w, follow_id=999, inspector_stats={dead.id: {"kills": 1, "offspring": 2}}, fallen=dead)
    # ring HUD radius follows the size gene (bigger gene -> larger badge)
    w.snakes[1].genome = w.snakes[1].genome.copy(); w.snakes[1].genome[SIZE] = 1.0
    r.draw(w, follow_id=w.snakes[1].id)
    r.close()


def test_fx_stun_courtship_and_guarded_egg_render():
    # Phase B increment 4: a stunned snake (dizzy stars), a courting pair (world._mate_streak read
    # read-only -> hearts), a guarded repro egg (owner>=0 glow) and an arrival egg (owner -1, no glow)
    # must all render without crashing. The courtship read is a viewer-only read; mating logic untouched.
    w = generate_world(CFG, seed=1, size=(120.0, 120.0), n_snakes=2)
    a, b = w.snakes[0], w.snakes[1]
    a.stun = CFG.stun_steps                                       # dizzy
    b.head_uw = a.head_uw + np.array([CFG.r_mate * 0.5, 0.0]); b.head = wrap(b.head_uw, w.size)
    w._mate_streak = {frozenset((a.id, b.id)): 2}                 # a courting pair (read-only in render)
    w.eggs = {"pos": np.array([[60.0, 60.0], [40.0, 40.0]]), "timer": np.array([20.0, 20.0]),
              "owner": np.array([[a.id, b.id], [-1, -1]])}        # guarded repro egg + arrival egg
    r = Renderer(scale=5)
    r._clock_override = 0.3
    r.draw(w, follow_id=a.id, cam_center=tuple(a.head_uw), zoom=3.0)   # must not raise
    del w._mate_streak                                            # no courting state -> no-op path
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
