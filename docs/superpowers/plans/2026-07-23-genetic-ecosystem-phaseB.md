# Genetic Ecosystem — Phase B (Viewer) Implementation Plan

> Executor: build INCREMENTALLY, one increment = one commit, headless-smoke each before moving on.
> Visual work — no TDD; acceptance is "headless smoke passes (no crash) + the described behavior is
> wired". FINAL visual verification (screenshots) happens after the v3 retrain model lands — the
> controller does that pass. NO retrain needed (render/watch only).

**Goal:** Make the evolving ecosystem watchable and legible — a free/follow camera over a
bigger-than-screen world, genome expressed as visible phenotype (family color + build + markings), a
genome inspector for the followed snake, and visual FX (stun, courtship, egg). Design source: the
spec `docs/superpowers/specs/2026-07-23-genetic-ecosystem-design.md` §10.

**Tech:** Python 3.13 (`.venv/`), pygame, numpy. Run headless smokes with
`SDL_VIDEODRIVER=dummy PYTHONPATH="$PWD" .venv/bin/python -m pytest tests/test_render_smoke.py tests/test_watch_smoke.py -q`.
Read the REAL `snake_rl/render.py` and `snake_rl/watch.py` before each increment.

## Global constraints
- **No retrain / no obs / no physics change.** Touch only `render.py`, `watch.py`, and (minimally)
  read-only accessors. If you need a per-snake stat the viewer doesn't have (e.g. courting state,
  kill/offspring counts), add a LIGHTWEIGHT tracker in `watch.py` (accumulate from `world.step`'s
  `deaths_detailed`/`hatched_owners`) or expose a read-only flag — do NOT change training/env/world
  physics or the reward.
- Keep existing viewer behavior working: `SPACE` pause, `N` new world, `S` rays, `H` ring HUD, `ESC`
  quit, fullscreen `FULLSCREEN|SCALED`, interpolation, gore/hatch FX, egg-based reseed floor.
- macOS/pygame pitfalls (7): `.convert_alpha()` on SRCALPHA; smoothscale to temp then blit; clear
  sprite cache on `set_mode`; Retina `display.get_size()`.
- The viewer world is now **all-eggs at start** (`ego_live=False`, `no_ego=True`) and can have **0
  live snakes** transiently — every increment must tolerate an empty live set (Phase A already
  guards `render.draw`/`watch` for this; don't regress it).
- No commit attribution lines. Each increment: implement → headless smoke → commit.

---

## Increment 1 — Camera: free-pan + zoom + follow/cycle + death-linger

**Files:** `snake_rl/render.py` (transform), `snake_rl/watch.py` (input + camera state).

The world is now bigger than the screen, so the viewer needs a real camera. Today `render` fits the
whole world to the canvas (`_scale = cw/world.size[0]`, `_p(xy)=xy*scale`, torus wrap in
`_circle`/`_blit_world`/`_draw_*`). Add a **camera offset (world units) + zoom (scale multiplier)**.

- Add camera state to `render` (or pass it into `draw`): `cam_center` (world-unit point the view is
  centered on) and `zoom` (≥1 zooms in). Effective transform: a world point `p` maps to canvas
  `((p - cam_center) mod world.size, nearest-image) * base_scale * zoom + canvas_center`. Route
  **every** draw through it: refactor `_p`, `_circle`, `_blit_world`, and the ray/HUD/arrival draws to
  a single `_world_to_canvas(p)` + keep the torus triple-wrap (`ox in {0,-cw,cw}`) but relative to the
  camera so entities near the seam still draw. Verify body/chicken interpolation still works.
- **Two camera modes (state in `watch.run_watch`):**
  - **FREE:** arrow keys pan `cam_center` (speed scaled by zoom); mouse wheel or `+`/`-` zoom
    (clamp zoom to a sane range, e.g. 1.0–6.0). Panning wraps on the torus.
  - **FOLLOW:** `cam_center` tracks a chosen live snake's (interpolated) head each frame; `[` / `]`
    cycle to prev/next live snake (stable order by id); the followed snake still gets the "big" ring
    HUD (existing `follow_id`).
  - `Tab` toggles FREE ↔ FOLLOW.
  - **Death-linger:** in FOLLOW, when the followed snake dies, hold `cam_center` at its last position
    for ~3 seconds (wall-clock), then advance to the next live snake (or stay put if none).
- **Rebind sim-speed** off the arrows (they're now pan) to `,` / `.` (keep `↑/↓`? no — arrows are
  pan; move sim-speed fully to `,`/`.`). Update the on-screen/help hints if any.
- Default mode on launch: FOLLOW a live snake if any, else FREE overview. Tolerate 0 live snakes
  (all-eggs start) → FREE overview until something hatches.

**Headless smoke:** extend `tests/test_watch_smoke.py` — build an `ego_live=False` world, step a few
ticks, call `render.draw(...)` in FREE and FOLLOW with a camera offset+zoom and with 0 live snakes,
assert no crash. Commit: `viewer: free/follow camera with pan+zoom, snake cycling, 3s death-linger`.

---

## Increment 2 — Genome → visible phenotype (family color + build + markings)

**Files:** `snake_rl/render.py` (snake coloring), possibly a tiny helper.

Today snakes color by `color_for(snake.color_seed)` (seed = sid). Make the genome legible:
- **Hue = lineage** (family): derive the base hue from `snake.lineage` (golden-angle like `color_for`,
  but keyed on `lineage` so a family shares a stable hue across generations). Offspring inherit the
  maternal `lineage` (Phase A), so lines are visually trackable.
- **Saturation / value / markings = a couple of genes** (e.g. `aggression`→warmer/red shift,
  `metabolism`→brightness, or a stripe/spot count from a gene) — pick 1–2 that read clearly; keep it
  tasteful, not garish.
- **Body size** already follows `target_length` (which now caps at the per-snake `phenotype.max_length`
  from the `size` gene), so big-`size` snakes are visibly longer — good. Optionally scale head/segment
  radius slightly with the `size` gene for extra readability.
- Keep the head/tail gradient idea (`_snake_colors`), just re-source the base hue from lineage + genes.

**Headless smoke:** assert two snakes with different `lineage`/genome produce different colors; render
doesn't crash. Commit: `viewer: genome as phenotype — lineage family color + trait markings + size`.

---

## Increment 3 — Genome inspector overlay (followed snake)

**Files:** `snake_rl/render.py` (overlay draw), `snake_rl/watch.py` (per-snake life-stat tracker).

For the FOLLOWED snake, draw a compact overlay panel (corner) showing:
- 9 **gene bars** (labeled: size/metabolism/speed/stamina/senses/lifespan/aggression/kin_care/boldness),
  each 0–1.
- **sex** (♀/♂), **age / max_lifespan** (a life-fraction bar), **lineage** id + its family color swatch.
- **life stats:** kills (cut-off), offspring (hatched eggs it co-owned) — track these in `watch` by
  accumulating `world.step`'s `deaths_detailed` (attribute a `snake`-cause death to the killer if
  cheaply available; if attribution isn't exposed, show total `snake`-deaths near it or omit kills and
  keep offspring) and `hatched_owners` (increment offspring for surviving co-owners). Keep the tracker
  a small dict keyed by snake id, reset on `N`/new world.
- Toggle with a key (e.g. `I`); off by default is fine, or on in FOLLOW mode. Legible font, semi-
  transparent panel, `.convert_alpha()`.

**Headless smoke:** render the overlay for a followed snake without crash; tracker increments
offspring on a hatch. Commit: `viewer: genome inspector overlay (gene bars, sex, age, lineage, stats)`.

---

## Increment 4 — Visual FX: stun, courtship, egg

**Files:** `snake_rl/render.py`, minimal `world`/`watch` read-only exposure if needed.

- **Stun "dizzy":** while `snake.stun > 0`, draw spinning stars/birds circling the head (the field
  already exists; `Snake.stun` is readable). Fade as stun counts down.
- **Courtship:** when an eligible pair is holding mating distance (the "courting" state), draw hearts /
  a soft pulse between them. `_resolve_mating` tracks the mate-streak internally — expose a **read-only**
  snapshot the viewer can read (e.g. `world.courting_pairs` = list of (id,id) currently mid-streak),
  set each step in `_resolve_mating` WITHOUT changing mating logic. If exposing it cleanly is hard,
  approximate in the viewer (two repro-ready opposite-sex snakes within `r_mate`) and note the
  approximation.
- **Egg wobble/pulse:** eggs already animate per CLAUDE.md (`egg wobble`); if a repro egg (owner ≥ 0)
  should read differently from an arrival egg, add a subtle guarded-egg glow. Keep the existing hatch
  crack.

**Headless smoke:** render a world with a stunned snake + an egg without crash. Commit:
`viewer: FX — stun dizzy, courtship hearts, guarded-egg glow`.

---

## Phase B exit
- All four increments committed, `tests/test_render_smoke.py tests/test_watch_smoke.py` green, full
  suite still green.
- Controller does the visual-verification pass with the v3 model once the retrain lands (launch the
  real viewer / capture screenshots, iterate on the look). Update CLAUDE.md's viewer/keys section.

## Notes for the executor
- Keep heavy test runs minimal while a retrain is using the CPU (a from-scratch retrain may be
  running in parallel) — the render/watch smokes are cheap; don't run the full 175-test suite in a
  tight loop, once per increment is enough.
- If any increment needs a change outside render/watch (world/env), STOP and report it as a concern
  rather than editing training/physics code — the running retrain must not be invalidated.
