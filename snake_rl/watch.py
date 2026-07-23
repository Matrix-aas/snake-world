"""Watch mode: load a checkpoint and run one env with the pygame renderer, smoothly interpolated."""
import os
import sys
import numpy as np
import pygame
from stable_baselines3 import PPO
from .config import CFG
from .train import build_vec
from .render import Renderer, ZOOM_MIN, ZOOM_MAX
from .world import wrap, torus_delta, torus_dist
from .worldgen import generate_world
from .selfplay import OpponentController
from .sensors import OBS_DIM


def _norm_path_for(model_path):
    return os.path.join(os.path.dirname(model_path) or ".", "vecnormalize.pkl")


def _require_files(model_path, norm_path):
    if not os.path.exists(model_path):
        sys.exit(f"No model at {model_path} — run `python -m snake_rl train` first.")
    if not os.path.exists(norm_path):
        sys.exit(f"Missing {norm_path} (normalization stats) — retrain or restore it.")


def _load_model(model_path):
    """[M-3] Load a checkpoint and reject one trained against a different observation layout
    (e.g. a pre-multi-snake OBS_DIM) instead of quietly feeding it obs it was never trained on."""
    model = PPO.load(model_path, device="cpu")
    expected = (OBS_DIM * CFG.frame_stack,)
    got = tuple(model.observation_space.shape)
    if got != expected:
        raise ValueError(
            f"Model at {model_path} has observation_space shape {got}, expected {expected} "
            f"(OBS_DIM={OBS_DIM} x frame_stack={CFG.frame_stack}). This looks like a checkpoint "
            "from a different observation layout — retrain before watching."
        )
    return model


def _load_norm_stats(norm_path, seed):
    """Read obs_rms/clip_obs/epsilon out of a saved VecNormalize without keeping its env stack
    around -- just enough to sync an OpponentController for the persistent-world viewer/eval."""
    vec = build_vec(1, seed, training=False, norm_path=norm_path)
    try:
        return vec.obs_rms, vec.clip_obs, vec.epsilon
    finally:
        vec.close()


def _world_of(vec):
    """Reach the underlying World through VecNormalize -> VecFrameStack -> DummyVecEnv -> Monitor."""
    return vec.venv.venv.envs[0].unwrapped.world


def _new_ecosystem(model_path, seed, world_size=None):
    """Load the model + normalization stats and build a fresh persistent multi-snake World,
    with ONE OpponentController synced to drive every snake in it (self-play brain vs itself)."""
    norm_path = _norm_path_for(model_path)
    _require_files(model_path, norm_path)
    model = _load_model(model_path)
    obs_rms, clip_obs, epsilon = _load_norm_stats(norm_path, seed)
    controller = OpponentController(CFG)
    sd = {k: v.detach().cpu().numpy() for k, v in model.policy.state_dict().items()}
    controller.sync(sd, obs_rms, clip_obs, epsilon)
    rng = np.random.default_rng(seed)
    n0 = int(rng.integers(CFG.n_start_min, CFG.n_start_max + 1))
    # ego_live=False: the WATCHED world has no privileged ego -- it starts as all eggs, snakes appear
    # only by hatching (Task 10). Every snake is driven by the same synced brain via `controller`.
    world = generate_world(CFG, seed=seed, size=world_size, n_snakes=n0, arrivals=True, ego_live=False)
    return model, controller, world


def _reseed_floor(world, controller):
    """Screensaver guarantee: a persistent world never resets, so a bad run of deaths can empty it
    for good. If live snakes PLUS pending arrival eggs fall below the sustain floor (cfg.n_start_min),
    lay fresh guaranteed arrival egg(s) -- placed like worldgen's spread spawn via `_free_point` --
    that hatch (via the SAME synced brain) into full snakes a few steps later, so even a reseed
    ARRIVES via an egg rather than popping in (Goal 1). Pending eggs count toward the floor so we lay
    exactly enough, not one per step while they incubate. A no-op above the floor, so natural
    birth/death dynamics dominate whenever the population is healthy. (`controller` unused now: a
    hatchling's ring is created zeroed on its first `act`, and _step_world resets rings on death.)"""
    c = world.cfg

    def deficit():
        n_alive = sum(1 for s in world.snakes if s.alive)
        owner = world.eggs["owner"]
        n_pending = int((owner[:, 0] < 0).sum()) if len(owner) else 0     # unhatched arrival eggs
        return c.n_start_min - (n_alive + n_pending)

    while deficit() > 0:
        world.spawn_egg(world._free_point(c.head_radius))


def _step_world(world, controller):
    """Advance the persistent world one tick. The viewer world is no-ego (starts as all founder
    eggs, zero live snakes) -- EVERY snake that hatches is driven by the SAME synced policy through
    the controller; no SB3 stepping, no autoreset. Drops a dead snake's frame ring so a later
    hatchling reusing that id starts cold (mirrors env.py's per-death ring reset). Then tops the
    population back up to the sustain floor if this step dropped it below (see _reseed_floor) --
    run_watch's gore diff snapshots BEFORE this call, so a reseed reads as a hatch effect for free."""
    # The viewer world is no-ego (all snakes equal): the positional action is IGNORED by world.step,
    # which drives EVERY live snake via opponent_fn. So DON'T call controller.act for the positional
    # slot here -- that snake is also driven by opponent_fn, and a second act() would roll its frame
    # ring twice (two identical newest frames -> corrupt velocity signal). Pass a constant instead.
    # In an ego world the positional action drives snakes[0] (not covered by opponent_fn), so act() it.
    live = [s for s in world.snakes if s.alive]
    a = (1, 1, 0) if world.no_ego else (controller.act(world, live[0]) if live else (1, 1, 0))
    out = world.step(*a, opponent_fn=lambda w, s: controller.act(w, s))
    for sid, _cause in out["deaths_detailed"]:
        controller.reset_snake(sid)
    _reseed_floor(world, controller)
    return out


def _interp_body(prev, cur, f):
    # Blend the head-side prefix (stable between steps); when the snake grows, the extra tail
    # points just snap in far from the head — no visible stutter on the eat step.
    n = min(len(prev), len(cur))
    out = cur.copy()
    out[:n] = prev[:n] + (cur[:n] - prev[:n]) * f
    return out


def _snake_snap(world):
    return {s.id: world._body_render_path_uw(s) for s in world.snakes if s.alive}


def _gore_state(world):
    """Snapshot the state we diff across a step to fire gore effects on the REAL events:
    {chicken_id: pos}, {corpse position}, {live snake id}."""
    return ({int(i): world.chicken_pos[k].copy() for k, i in enumerate(world.chicken_id)},
            {(round(float(p[0]), 2), round(float(p[1]), 2)) for p in world.corpses["pos"]},
            {s.id for s in world.snakes if s.alive})


def _emit_gore(renderer, before, world):
    """Compare pre/post-step state and trigger blood/gore at the real strike points (covers EVERY
    snake, not just the ego): a chicken that vanished -> eat burst + decal; a new corpse -> death
    burst + decal; a newly-alive snake id -> egg-hatch shell crack."""
    ch_before, corpses_before, ids_before = before
    ch_now = {int(i): world.chicken_pos[k].copy() for k, i in enumerate(world.chicken_id)}
    for cid, pos in ch_before.items():
        if cid not in ch_now:
            renderer.spawn_eat(pos)
    for cid, pos in ch_now.items():
        if cid not in ch_before:
            renderer.spawn_land(pos)                              # a sky-dropped chicken touched down
    for p in world.corpses["pos"]:
        if (round(float(p[0]), 2), round(float(p[1]), 2)) not in corpses_before:
            renderer.spawn_death(p.copy())
    for s in world.snakes:
        if s.alive and s.id not in ids_before:
            renderer.spawn_hatch(s.head.copy())


def _interp_bodies(prev, cur, f):
    """Blend each live snake's body by stable id (same idea as _interp_chickens); a snake missing
    from `prev` (just hatched) or grown/shrunk is handled by _interp_body's own length-mismatch."""
    return {sid: (_interp_body(prev[sid], b, f) if sid in prev else b) for sid, b in cur.items()}


def _chicken_snap(world):
    return {int(i): (world.chicken_pos[k].copy(), float(world.chicken_dir[k]))
            for k, i in enumerate(world.chicken_id)}


def _interp_chickens(prev, cur, f, size):
    """Blend chicken positions by stable id, taking the nearest image across the torus seam."""
    pos, dirs = [], []
    for cid, (cp, cd) in cur.items():
        if cid in prev:
            pp, pd = prev[cid]
            pos.append(wrap(pp + torus_delta(cp, pp, size) * f, size))
            da = (cd - pd + np.pi) % (2 * np.pi) - np.pi
            dirs.append(pd + da * f)
        else:
            pos.append(cp); dirs.append(cd)
    return (np.array(pos) if pos else np.zeros((0, 2))), np.array(dirs)


def rollout_once(model, norm_path, seed=0, max_steps=CFG.episode_horizon):
    """Headless single-episode rollout with frame-stacking + obs normalization."""
    vec = build_vec(1, seed, training=False, norm_path=norm_path)
    try:
        obs = vec.reset()
        eaten = died = 0
        steps = 0
        for steps in range(1, max_steps + 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, infos = vec.step(action)
            eaten += infos[0].get("ate", 0)
            if done[0]:
                died = 1 if infos[0].get("alive") is False else 0
                break
        return {"steps": steps, "eaten": eaten, "died": died}
    finally:
        vec.close()


def run_headless(model_path="models/snake.zip", seed=None, episodes=5, max_steps=None):
    """Ecosystem evaluation over a PERSISTENT multi-snake world (never reset on a death, same as
    `run_watch`): catch rate + dash usage (aggregated across every snake, inferred without touching
    world.py -- see below), and ecosystem series from `step`'s deaths_detailed/hatched_owners:
    births, kills (cause=="snake"), starvations (cause=="starve"), and
    population over time. `episodes` scales the run length (episodes * episode_horizon steps);
    `max_steps` overrides that directly (handy for fast tests). Returns the metrics dict."""
    seed = seed if seed is not None else 0
    model, controller, world = _new_ecosystem(model_path, seed)
    total_steps = max_steps if max_steps is not None else max(1, episodes) * CFG.episode_horizon

    deaths = {"snake": 0, "starve": 0}     # obstacles/own body are solid-slide non-lethal now: only these two
    births = 0
    population = []
    catches = 0
    dash_steps = 0
    snake_steps = 0
    for _ in range(total_steps):
        prev = {s.id: (s.stamina, s.target_length) for s in world.snakes if s.alive}
        out = _step_world(world, controller)
        for sid, cause in out["deaths_detailed"]:
            if cause in deaths:
                deaths[cause] += 1
        births += len(out["hatched_owners"])
        for s in world.snakes:
            if not s.alive or s.id not in prev:
                continue
            prev_stam, prev_len = prev[s.id]
            snake_steps += 1
            if s.stamina < prev_stam:                          # stamina dropped => actually dashed
                dash_steps += 1
            if s.target_length > prev_len:                     # grew => ate something this step
                # approx: infer item count from growth (grow_per_chicken/item); undercounts once
                # a snake sits at length_cap, where further eating no longer grows it.
                catches += round((s.target_length - prev_len) / CFG.grow_per_chicken)
        population.append(sum(1 for s in world.snakes if s.alive))
    steps = max(1, total_steps)
    snake_steps = max(1, snake_steps)
    metrics = {
        "steps": steps,
        "population": population,
        "births": births,
        "kills": deaths["snake"],
        "starvations": deaths["starve"],
        "deaths": deaths,
        # per-snake, not population-summed -- matches CLAUDE.md's judging band (10-14/1000) and
        # dash_usage's own normalization; a population-summed rate would scale with snake count.
        "catch_rate": catches / snake_steps * 1000,
        "dash_usage": dash_steps / snake_steps * 100,
    }
    print(f"over {steps} steps, persistent {episodes}-episode-equivalent ecosystem run:")
    print(f"  catch rate:  {metrics['catch_rate']:5.1f} items / 1000 snake-steps (per snake)")
    print(f"  dash usage:  {metrics['dash_usage']:5.0f}% of live snake-steps")
    print(f"  population:  mean {np.mean(population):4.1f}   min {min(population)}   max {max(population)}")
    print(f"  births:      {births}    kills: {metrics['kills']}    starvations: {metrics['starvations']}")
    print(f"  deaths:      snake {deaths['snake']}, starve {deaths['starve']}")
    return metrics


def _screen_fit_world_size(short=86.4):
    """World size (in sim units) whose aspect matches the desktop, short side fixed to `short`.
    The net is size-agnostic (egocentric senses), so any size plays fine; `short` keeps density sane.
    86.4 = 72 * 1.2 -> ~20% bigger/roomier watch map than the original."""
    pygame.init()
    info = pygame.display.Info()
    sw, sh = info.current_w, info.current_h
    if sw >= sh:
        return (short * sw / sh, short), (sw, sh)
    return (short, short * sh / sw), (sw, sh)


def _update_life_stats(stats, out, pre_heads, size):
    """Lightweight per-snake life stats for the genome inspector (Phase B increment 3). Offspring:
    each real hatch credits BOTH co-owners (out['hatched_owners'] = frozensets of owner ids).
    Kills: deaths_detailed carries only the victim + cause, not the killer, so credit the nearest
    pre-step rival (a cut-off death means the victim crossed in front of them). Snake ids are
    monotonic (never reused), so a dict keyed by id can't leak across occupants.
    ponytail: nearest-rival is an approximation for the killer; exact attribution would re-run the
    cut-off geometry -- not worth it for a viewer label."""
    for owners in out["hatched_owners"]:
        for oid in owners:
            stats.setdefault(int(oid), {"kills": 0, "offspring": 0})["offspring"] += 1
    for sid, cause in out["deaths_detailed"]:
        if cause != "snake" or sid not in pre_heads:
            continue
        others = [(oid, h) for oid, h in pre_heads.items() if oid != sid]
        if not others:
            continue
        vh = pre_heads[sid]
        killer = min(others, key=lambda oh: float(torus_dist(vh[None], oh[1], size)[0]))[0]
        stats.setdefault(int(killer), {"kills": 0, "offspring": 0})["kills"] += 1


# --- viewer camera (free-pan + follow, Phase B increment 1) ---
CAM_FOLLOW_ZOOM = 2.0       # fallback manual follow zoom (auto-framing overrides it while auto_zoom on)
CAM_EASE_K = 6.0            # camera-center ease rate per second (1-exp(-k*dt)): smooth catch-up
CAM_ZOOM_EASE_K = 4.0       # zoom ease rate per second
EGG_TRACK_R = 7.0           # world-units: an egg / its hatchling counts as "the watched egg" within this
EGG_SPAN = 55.0            # world-units framed for the opening egg close-up
PAN_UNITS_PER_SEC = 45.0    # free-pan speed at zoom 1 (scaled by 1/zoom so panning feels constant)
DEATH_LINGER_S = 3.0        # hold the camera at a followed snake's death spot this long, then advance


def _live_ids(world):
    return sorted(s.id for s in world.snakes if s.alive)


def _live_map(world):
    return {s.id: s for s in world.snakes if s.alive}


def _follow_span(snake):
    """World-units to frame around a followed snake: its body length + generous headroom, so a good
    action-cam shows the snake AND its surroundings (bigger snakes pull the camera out further)."""
    return float(np.clip(3.0 * snake.target_length + 80.0, 100.0, 175.0))


def _soonest_egg(world):
    """The egg closest to hatching (the opening close-up's payoff comes soon), or None."""
    e = world.eggs
    if not len(e["pos"]):
        return None
    return np.asarray(wrap(e["pos"][int(np.argmin(e["timer"]))], world.size), float)


def _egg_near(world, pos, r):
    e = world.eggs
    if pos is None or not len(e["pos"]):
        return False
    return bool(np.any(torus_dist(np.asarray(e["pos"], float), pos, world.size) <= r))


def _nearest_live_near(world, pos, r):
    best, bd = None, r
    for s in world.snakes:
        if not s.alive:
            continue
        d = float(torus_dist(pos[None], s.head_uw, world.size)[0])
        if d <= bd:
            best, bd = s, d
    return best


def _new_camera(world):
    """Opening shot: if any eggs exist, FRAME the soonest-to-hatch one (close-up), else fall to the
    first live snake / overview. The eased zoom starts at 1 (whole-world) so the opening PUSHES IN."""
    c = np.asarray(world.size, float) / 2.0
    return {"mode": "follow", "zoom": CAM_FOLLOW_ZOOM, "auto_zoom": True,
            "pan": c.copy(), "center": c.copy(), "zoom_eased": 1.0,
            "follow_id": None, "watch_egg": _soonest_egg(world),
            "follow_snake": None, "last_head": None, "death_t": None}


def _cycle_follow(cam, world, delta):
    """`[` / `]`: switch the followed snake to the prev/next live one (stable order by id)."""
    ids = _live_ids(world)
    if not ids:
        return
    cam["mode"] = "follow"
    i = (ids.index(cam["follow_id"]) + delta) % len(ids) if cam["follow_id"] in ids else \
        (0 if delta >= 0 else len(ids) - 1)
    cam["follow_id"] = ids[i]
    cam["watch_egg"] = None; cam["death_t"] = None; cam["auto_zoom"] = True


def _resolve_target(cam, world, bodies, renderer, now):
    """What the camera should FRAME this instant: (target_center_world, target_zoom, fallen_snake).
    Precedence: live followed snake -> death-linger hold (keeps the fallen snake for its panel) ->
    egg-watch (opening close-up, then hand off to the hatchling that appears at the egg) -> adopt any
    live snake -> whole-world overview."""
    if cam["mode"] == "free":
        return cam["pan"], cam["zoom"], None
    live = _live_map(world)

    def frame_snake(s):
        cam["follow_snake"] = s; cam["death_t"] = None; cam["watch_egg"] = None
        h = bodies.get(s.id, [s.head_uw])[0]
        cam["last_head"] = np.asarray(wrap(h, world.size), float)
        z = renderer.zoom_for_span(_follow_span(s), cam["zoom"]) if cam["auto_zoom"] else cam["zoom"]
        return cam["last_head"], z, None

    fid = cam["follow_id"]
    if fid in live:
        return frame_snake(live[fid])
    if fid is not None and cam["death_t"] is None:
        cam["death_t"] = now                                  # followed snake just died -> start linger
    if cam["death_t"] is not None and now - cam["death_t"] < DEATH_LINGER_S and cam["last_head"] is not None:
        fs = cam["follow_snake"]
        zt = renderer.zoom_for_span(_follow_span(fs), cam["zoom"]) if fs is not None else cam["zoom"]
        return cam["last_head"], zt, fs                       # hold at the death spot, keep the panel
    egg = cam["watch_egg"]                                     # need a fresh target
    if egg is not None:
        if _egg_near(world, egg, EGG_TRACK_R):
            return egg, renderer.zoom_for_span(EGG_SPAN, cam["zoom"]), None       # frame the egg
        hatch = _nearest_live_near(world, egg, EGG_TRACK_R)   # egg gone -> hand off to its hatchling
        if hatch is not None:
            cam["follow_id"] = hatch.id; cam["auto_zoom"] = True
            return frame_snake(hatch)
        cam["watch_egg"] = None                               # egg vanished with no nearby hatchling
    if live:
        s = live[min(live)]; cam["follow_id"] = s.id; cam["auto_zoom"] = True
        return frame_snake(s)
    cam["follow_id"] = None
    return np.asarray(world.size, float) / 2.0, 1.0, None     # nobody alive, no eggs -> overview


def _camera_view(cam, world, bodies, renderer, now, dt):
    """Resolve the frame's target then EASE toward it, framerate-independently (1-exp(-k*dt)), so the
    camera smoothly catches up instead of hard-snapping. The center eases toward the NEAREST-IMAGE
    target so it never whips across a torus seam. FREE pan is user-driven, so it snaps (no lag).
    Returns (cam_center_world, draw_zoom, fallen_snake)."""
    dt = float(min(max(dt, 0.0), 0.1))                        # clamp (a long stall shouldn't teleport)
    tgt, tz, fallen = _resolve_target(cam, world, bodies, renderer, now)
    tgt = np.asarray(tgt, float)
    if cam["mode"] == "free":
        cam["center"] = tgt.copy()
    else:
        a = 1.0 - np.exp(-CAM_EASE_K * dt)
        d = torus_delta(tgt, cam["center"], world.size)
        cam["center"] = wrap(cam["center"] + d * a, world.size)
    cam["zoom_eased"] += (float(tz) - cam["zoom_eased"]) * (1.0 - np.exp(-CAM_ZOOM_EASE_K * dt))
    return cam["center"], float(np.clip(cam["zoom_eased"], ZOOM_MIN, ZOOM_MAX)), fallen


def run_watch(model_path="models/snake.zip", seed=None, fps=60, sim_hz=10, fullscreen=True):
    """Persistent-world viewer [I-7]: a plain World is stepped directly (no SB3 VecEnv, no
    autoreset) with EVERY snake -- including the nominal ego slot -- driven by one
    OpponentController synced from the loaded checkpoint. The world never resets on any single
    snake's death; the camera follows a chosen live snake and re-targets when it dies (falls
    back to an overview -- slot-0 -- once nobody is left). The sim advances at sim_hz steps/sec;
    rendering runs at `fps` and interpolates every live snake's body + the chickens (seam-aware)
    between steps for smooth motion.
    """
    # random map on every launch (so it's fresh each time, not the same fixed world); pass --seed
    # for a reproducible one. The N key still steps forward from whatever seed we start on.
    seed = seed if seed is not None else int.from_bytes(os.urandom(4), "little")
    # The viewer world is now BIGGER THAN THE SCREEN: sample its size from cfg.world_size_min/max
    # (like training/worldgen), and use the desktop resolution ONLY for the display surface. The
    # camera + zoom does the fitting (overview = whole world letterboxed; follow zooms in).
    world_size = None
    screen_size = _screen_fit_world_size()[1] if fullscreen else None
    model, controller, world = _new_ecosystem(model_path, seed, world_size=world_size)
    renderer = Renderer(fullscreen=fullscreen, screen_size=screen_size)
    clock = pygame.time.Clock()
    paused = False
    running = True
    cam = _new_camera(world)   # free-pan/follow camera (arrows pan, [/] cycle, Tab toggle, wheel/+/- zoom)
    life_stats = {}            # per-snake {kills, offspring} for the genome inspector (I); reset on N

    def snapshot(w):
        return _snake_snap(w), _chicken_snap(w)

    try:
        prev_bodies, prev_ch = cur_bodies, cur_ch = snapshot(world)
        since = 0.0
        while running:
            frame_dt = clock.tick(fps) / 1000.0
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.MOUSEWHEEL:
                    cam["zoom"] = float(np.clip(cam["zoom"] * (1.12 ** e.y), ZOOM_MIN, ZOOM_MAX))
                    cam["auto_zoom"] = False                  # manual zoom overrides action-cam framing
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        running = False
                    elif e.key == pygame.K_SPACE:
                        paused = not paused
                    elif e.key == pygame.K_s:
                        renderer.toggle_sensors()
                    elif e.key == pygame.K_h:
                        renderer.toggle_rings()
                    elif e.key == pygame.K_i:                 # genome inspector (followed snake)
                        renderer.toggle_inspector()
                    elif e.key == pygame.K_TAB:               # free <-> follow
                        if cam["mode"] == "follow":
                            cam["pan"] = np.asarray(cam["center"], float).copy(); cam["mode"] = "free"
                        else:
                            cam["mode"] = "follow"; cam["death_t"] = None
                    elif e.key == pygame.K_LEFTBRACKET:
                        _cycle_follow(cam, world, -1)
                    elif e.key == pygame.K_RIGHTBRACKET:
                        _cycle_follow(cam, world, +1)
                    elif e.key in (pygame.K_EQUALS, pygame.K_PLUS):
                        cam["zoom"] = float(np.clip(cam["zoom"] * 1.25, ZOOM_MIN, ZOOM_MAX))
                        cam["auto_zoom"] = False
                    elif e.key == pygame.K_MINUS:
                        cam["zoom"] = float(np.clip(cam["zoom"] / 1.25, ZOOM_MIN, ZOOM_MAX))
                        cam["auto_zoom"] = False
                    elif e.key == pygame.K_PERIOD:            # sim speed (rebound off the arrows)
                        sim_hz = min(60, sim_hz + 2)
                    elif e.key == pygame.K_COMMA:
                        sim_hz = max(2, sim_hz - 2)
                    elif e.key == pygame.K_n:                 # fresh persistent world (not an autoreset)
                        seed += 1
                        world = generate_world(CFG, seed=seed, size=world_size,
                                               n_snakes=int(np.random.default_rng(seed).integers(
                                                   CFG.n_start_min, CFG.n_start_max + 1)),
                                               arrivals=True, ego_live=False)
                        controller.reset_all()
                        cam = _new_camera(world)
                        life_stats = {}
                        prev_bodies, prev_ch = cur_bodies, cur_ch = snapshot(world); since = 0.0
            if cam["mode"] == "free":                          # arrow-key pan (held keys, smooth)
                keys = pygame.key.get_pressed()
                dx = keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]
                dy = keys[pygame.K_DOWN] - keys[pygame.K_UP]
                if dx or dy:
                    step = PAN_UNITS_PER_SEC * frame_dt / max(1e-6, cam["zoom_eased"])
                    cam["pan"] = wrap(cam["pan"] + np.array([dx, dy], float) * step, world.size)
            interval = 1.0 / sim_hz
            if not paused:
                since += frame_dt
                while since >= interval:
                    since -= interval
                    before = _gore_state(world)
                    pre_heads = {s.id: s.head_uw.copy() for s in world.snakes if s.alive}
                    out = _step_world(world, controller)
                    _emit_gore(renderer, before, world)                   # blood/gore on eat/death/hatch
                    _update_life_stats(life_stats, out, pre_heads, world.size)   # inspector kills/offspring
                    prev_bodies, prev_ch = cur_bodies, cur_ch
                    cur_bodies, cur_ch = snapshot(world)
            f = 0.0 if paused else min(1.0, since / interval)
            bodies = _interp_bodies(prev_bodies, cur_bodies, f)
            cpos, cdir = _interp_chickens(prev_ch, cur_ch, f, world.size)
            cam_center, draw_zoom, _fallen = _camera_view(
                cam, world, bodies, renderer, pygame.time.get_ticks() / 1000.0, frame_dt)
            renderer.draw(world, bodies, cpos, cdir, follow_id=cam["follow_id"],
                          cam_center=cam_center, zoom=draw_zoom, inspector_stats=life_stats)
    finally:
        renderer.close()
