"""Watch mode: load a checkpoint and run one env with the pygame renderer, smoothly interpolated."""
import os
import sys
import numpy as np
import pygame
from stable_baselines3 import PPO
from .config import CFG
from .train import build_vec
from .render import Renderer
from .world import wrap, torus_delta, Snake
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
    world = generate_world(CFG, seed=seed, size=world_size, n_snakes=n0)
    return model, controller, world


def _reseed_floor(world, controller):
    """Screensaver guarantee: a persistent world never resets, so a bad run of deaths can empty
    it for good. If the live population falls below the sustain floor (cfg.n_start_min), spawn
    fresh snake(s) -- placed like worldgen's initial multi-snake spawn via `_free_point`, driven
    by the SAME synced brain -- to bring it back up to the floor. A no-op above the floor, so
    natural birth/death dynamics dominate whenever the population is healthy."""
    c = world.cfg
    n_alive = sum(1 for s in world.snakes if s.alive)
    while n_alive < c.n_start_min:
        p = world._free_point(c.head_radius)
        sid = world._next_snake_id
        world._next_snake_id += 1
        world.snakes.append(Snake(
            head_uw=p.copy(), head=wrap(p, world.size),
            heading=float(world.rng.uniform(0, 2 * np.pi)), path_uw=[p.copy()],
            target_length=c.start_length, stamina=c.s_max, energy=c.energy_max,
            _prev_head_uw=p.copy(), id=sid, color_seed=sid,
        ))
        controller.reset_snake(sid)     # fresh ring, not a stale/reused one
        n_alive += 1


def _step_world(world, controller):
    """Advance the persistent world one tick. EVERY snake -- including the nominal slot-0 'ego'
    left over from the single-snake days -- is driven by the SAME synced policy through the
    controller; no SB3 stepping, no autoreset. Drops a dead snake's frame ring so a later
    hatchling reusing that id starts cold (mirrors env.py's per-death ring reset). Then tops the
    population back up to the sustain floor if this step dropped it below (see _reseed_floor) --
    run_watch's gore diff snapshots BEFORE this call, so a reseed reads as a hatch effect for free."""
    ego = world.snakes[0]
    steer, dash = controller.act(world, ego) if ego.alive else (1, 0)
    out = world.step(steer, dash, opponent_fn=lambda w, s: controller.act(w, s))
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
    ch_now = {int(i) for i in world.chicken_id}
    for cid, pos in ch_before.items():
        if cid not in ch_now:
            renderer.spawn_eat(pos)
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
    births, kills (cause=="snake"), starvations (cause=="starve"), obstacle/self deaths, and
    population over time. `episodes` scales the run length (episodes * episode_horizon steps);
    `max_steps` overrides that directly (handy for fast tests). Returns the metrics dict."""
    seed = seed if seed is not None else 0
    model, controller, world = _new_ecosystem(model_path, seed)
    total_steps = max_steps if max_steps is not None else max(1, episodes) * CFG.episode_horizon

    deaths = {"obstacle": 0, "self": 0, "snake": 0, "starve": 0}
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
    print(f"  deaths:      obstacle {deaths['obstacle']}, self {deaths['self']}, "
          f"snake {deaths['snake']}, starve {deaths['starve']}")
    return metrics


def _screen_fit_world_size(short=72.0):
    """World size (in sim units) whose aspect matches the desktop, short side fixed to `short`.
    The net is size-agnostic (egocentric senses), so any size plays fine; `short` keeps density sane."""
    pygame.init()
    info = pygame.display.Info()
    sw, sh = info.current_w, info.current_h
    if sw >= sh:
        return (short * sw / sh, short), (sw, sh)
    return (short, short * sh / sw), (sw, sh)


def run_watch(model_path="models/snake.zip", seed=None, fps=60, sim_hz=10, fullscreen=True):
    """Persistent-world viewer [I-7]: a plain World is stepped directly (no SB3 VecEnv, no
    autoreset) with EVERY snake -- including the nominal ego slot -- driven by one
    OpponentController synced from the loaded checkpoint. The world never resets on any single
    snake's death; the camera follows a chosen live snake and re-targets when it dies (falls
    back to an overview -- slot-0 -- once nobody is left). The sim advances at sim_hz steps/sec;
    rendering runs at `fps` and interpolates every live snake's body + the chickens (seam-aware)
    between steps for smooth motion.
    """
    seed = seed if seed is not None else 0
    world_size = screen_size = None
    if fullscreen:
        world_size, screen_size = _screen_fit_world_size()   # map fills the screen at its aspect
    model, controller, world = _new_ecosystem(model_path, seed, world_size=world_size)
    renderer = Renderer(fullscreen=fullscreen, screen_size=screen_size)
    clock = pygame.time.Clock()
    paused = False
    running = True
    follow_id = world.snakes[0].id

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
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        running = False
                    elif e.key == pygame.K_SPACE:
                        paused = not paused
                    elif e.key == pygame.K_s:
                        renderer.toggle_sensors()
                    elif e.key in (pygame.K_UP, pygame.K_EQUALS, pygame.K_PLUS):
                        sim_hz = min(60, sim_hz + 2)
                    elif e.key in (pygame.K_DOWN, pygame.K_MINUS):
                        sim_hz = max(2, sim_hz - 2)
                    elif e.key == pygame.K_n:                 # fresh persistent world (not an autoreset)
                        seed += 1
                        world = generate_world(CFG, seed=seed, size=world_size,
                                               n_snakes=int(np.random.default_rng(seed).integers(
                                                   CFG.n_start_min, CFG.n_start_max + 1)))
                        controller.reset_all()
                        follow_id = world.snakes[0].id
                        prev_bodies, prev_ch = cur_bodies, cur_ch = snapshot(world); since = 0.0
            interval = 1.0 / sim_hz
            if not paused:
                since += frame_dt
                while since >= interval:
                    since -= interval
                    before = _gore_state(world)
                    _step_world(world, controller)
                    _emit_gore(renderer, before, world)                   # blood/gore on eat/death/hatch
                    alive_ids = {s.id for s in world.snakes if s.alive}
                    if follow_id not in alive_ids:            # camera re-targets on death, else overview
                        follow_id = next(iter(alive_ids), world.snakes[0].id)
                    prev_bodies, prev_ch = cur_bodies, cur_ch
                    cur_bodies, cur_ch = snapshot(world)
            f = 0.0 if paused else min(1.0, since / interval)
            bodies = _interp_bodies(prev_bodies, cur_bodies, f)
            cpos, cdir = _interp_chickens(prev_ch, cur_ch, f, world.size)
            renderer.draw(world, bodies, cpos, cdir, follow_id=follow_id)
    finally:
        renderer.close()
