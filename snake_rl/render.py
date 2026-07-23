"""pygame renderer: sprite-composited atmospheric world (tiled ground, obstacles, chickens,
snakes) + procedural BLOOD/GORE effects + sprite-sheet ANIMATION. Supersampled AA, torus-aware.

Animation: a wall-clock frame clock (`self._clock`, real seconds — decoupled from sim steps and
render fps so cycles play smoothly at any speed) drives sprite-sheet frame selection; a per-entity
PHASE offset (from stable id / position) keeps chickens and trees out of lockstep. The chicken
plays peck / walk / run sheets by its FSM `chicken_state`; trees sway procedurally.

Assets live in snake_rl/assets/ (generated). Every sprite path has a procedural fallback, so the
game still runs (and the smoke tests still pass) if an asset is missing. Pitfall 11 is honored:
SRCALPHA sprites are .convert_alpha()'d, we smoothscale into a temp then blit into the window,
fullscreen uses FULLSCREEN|SCALED and reads back the granted size, and the sprite cache is cleared
on every set_mode."""
import colorsys
import os
import numpy as np
import pygame
from .sensors import vision_distances
from .world import wrap

SS = 2                   # supersample factor (draw big, smoothscale down for anti-aliasing)
TARGET_PX = 860          # display fits ~this many pixels on its long side
ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")
ASSET_FILES = {
    "ground": "ground.png",                                     # single-tile fallback (see _build_ground)
    "ground_set": ["ground_0.png", "ground_1.png", "ground_2.png",   # 6 mutually-tileable grass variants
                   "ground_3.png", "ground_4.png", "ground_5.png"],  # patchworked into the world backdrop
    "rock": ["rock1.png", "rock2.png"],
    "tree": ["tree1.png", "tree2.png"],
    "chicken": "chicken.png",                                   # static fallback for the sheets below
    "chicken_peck": ["chicken_peck_0.png", "chicken_peck_1.png", "chicken_peck_2.png", "chicken_peck_3.png"],
    "chicken_walk": ["chicken_walk_0.png", "chicken_walk_1.png", "chicken_walk_2.png", "chicken_walk_3.png"],
    "chicken_run": ["chicken_run_0.png", "chicken_run_1.png", "chicken_run_2.png", "chicken_run_3.png"],
    "chicken_fall": ["chicken_fall_0.png", "chicken_fall_1.png", "chicken_fall_2.png", "chicken_fall_3.png"],
    "corpse": "corpse.png",
    "blood": ["blood1.png", "blood2.png"],
    "egg": "egg.png",
    "egg_cracked": "egg_cracked.png",
    "head": "snake_head.png",
}

# procedural fallbacks (used only when the matching sprite is missing)
BG = (34, 46, 30); GRID = (30, 42, 28)
ROCK = (110, 112, 122); ROCK_HI = (156, 158, 168); ROCK_SH = (60, 62, 72)
TRUNK = (98, 68, 44); LEAF = (54, 132, 72); LEAF2 = (84, 174, 104)
CHICK = (247, 243, 231); CHICK_SH = (206, 200, 184); BEAK = (242, 168, 62); EYE = (26, 26, 30); COMB = (226, 74, 74)
EGG_SHELL = (236, 224, 196); EGG_SHADE = (206, 188, 150)
CORPSE = (92, 40, 34)
SNAKE_RIM = (16, 40, 34)
SNAKE_GLOSS = (196, 252, 214); SNAKE_EYE = (250, 250, 250); PUPIL = (16, 20, 18)
TONGUE = (236, 74, 96)
RING_TRACK = (44, 48, 58)
# ring HUD: each ring's color encodes WHICH stat (not the snake's identity color, or all 3 rings
# look the same) -- outer energy=green, middle stamina=cyan, inner length=amber.
RING_COLORS = ((96, 214, 120), (86, 200, 232), (240, 186, 72))
RAY_NONE = (70, 92, 120); RAY_OBST = (226, 96, 84); RAY_CHICK = (240, 208, 96); RAY_SELF = (168, 128, 224)
RAY_OTHER = (110, 200, 240); RAY_EGG = (240, 176, 224)          # B2 ray kinds 3=other_body, 4=egg
RAY_CORPSE = (176, 132, 84)                                     # ray kind 5=corpse
RAY_KIND = {-1: RAY_NONE, 0: RAY_OBST, 1: RAY_CHICK, 2: RAY_SELF, 3: RAY_OTHER, 4: RAY_EGG, 5: RAY_CORPSE}

# chicken FSM state (world.chicken_state) -> (sheet asset key, playback fps, frame count).
# 0=PECK (slow head-bob), 1=WALK (waddle), 2=FLEE (fast flustered flap).
CHICK_ANIM = {0: ("chicken_peck", 6.0, 4), 1: ("chicken_walk", 7.0, 4), 2: ("chicken_run", 13.0, 4)}

# gore palettes (procedural particles)
BLOOD = [(150, 12, 14), (176, 22, 20), (120, 8, 12), (198, 44, 36)]
GORE = [(128, 34, 28), (96, 22, 20), (150, 60, 48), (74, 16, 16)]   # flesh / gut bits
DUST = [(196, 186, 156), (176, 166, 138), (210, 200, 172)]          # dash kick / landing puff
# sky-drop chicken arrival (Goal 2) -- all render-only tuning. Depth is carried mostly by the SHRINK
# + the growing shadow (top-down); the vertical offset is a small parallax that closes to 0 on land
# (a large one just reads as a hen hovering high above its shadow).
CHICK_FALL_TINT = (255, 233, 186)   # multiply-tint mapping the white fall art -> the landed hen's cream
ARRIVE_FALL = 5.0        # world-units the hen is drawn ABOVE its landing shadow when it first appears
FALL_ZOOM = 0.85         # extra scale up high (nearer the overhead camera) -> recedes to 1x on the ground
FALL_FADEIN = 0.12       # brief fade-in as it enters from the sky (no hard pop), then solid for the drop
FALL_FLAP_FPS = 9.0      # wing-flap sheet playback while falling
FALL_WOBBLE_DEG = 14.0   # side-to-side rock amplitude as it flutters down
FALL_WOBBLE_HZ = 2.6     # rock speed (wall-clock driven)
FALL_SMOOTH = 0.2        # per-frame damped-follow of the drop (the sim ticks ~10Hz but we draw ~60fps,
                         # so ease the discrete per-tick prog into a smooth glide instead of stair-steps)
SHELL = [(234, 224, 198), (212, 198, 166), (196, 182, 150)]         # egg shards


def color_for(seed, s=0.55, v=0.92):
    """Golden-angle hue palette: deterministic and visually distinct per integer seed,
    so a snake keeps a stable, distinguishable color across frames (snake.color_seed)."""
    hue = (seed * 137.5) % 360
    r, g, b = colorsys.hsv_to_rgb(hue / 360.0, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def _snake_colors(seed):
    """Head/tail gradient endpoints derived from the snake's hue (tapered body tint)."""
    return color_for(seed, s=0.42, v=0.96), color_for(seed, s=0.80, v=0.42)


def _lerp(a, b, t):
    return (int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t), int(a[2] + (b[2] - a[2]) * t))


class Renderer:
    def __init__(self, scale=None, show_sensors=False, show_rings=False, fullscreen=False, screen_size=None):
        pygame.init()
        self.scale = scale
        self.show_sensors = show_sensors      # vision-ray overlay (toggle key: S) -- OFF by default
        self.show_rings = show_rings          # per-snake ring HUD circles (toggle key: H) -- OFF by default
        self.fullscreen = fullscreen
        self.screen_size = screen_size
        self.canvas = self.display = None
        self.cw = self.ch = self.dw = self.dh = 0
        self._scale = 1
        self._ss = SS
        self._world_key = None
        self._t = 0                # render-frame counter (particle seeds, tongue flick)
        self._clock = 0.0          # wall-clock animation time in seconds (set each draw)
        self._clock_override = None  # test seam: force _clock to a fixed value for deterministic frames
        self._particles = []       # gore particles: [x,y,vx,vy,age,life,rad,color,gravity,grow]
        self._decals = []          # blood splatters on the ground: [x,y,variant,angle,diam,age,life]
        self._transient = []       # short sprite effects: [kind, x, y, age, life]
        self._motes = None         # ambient drifting dust (Nx4: x,y,vx,vy)
        self.font = pygame.font.SysFont("menlo,consolas,monospace", 15)
        self._sprite_cache = {}
        self._assets = {}
        self._chick_scale = {}     # per-chicken-sheet body-size normalization (see _load_assets)
        self._arrivals = {}        # sky-drop render state per bird: {pos_key: {"prog": smoothed 0..1}}
        self._ground_surf = None

    # --- setup / assets ---
    def _load_assets(self):
        self._assets = {}
        if not os.path.isdir(ASSET_DIR):
            return

        def load(name):
            p = os.path.join(ASSET_DIR, name)
            if not os.path.exists(p):
                return None
            try:
                return pygame.image.load(p).convert_alpha()   # Pitfall 11: convert for the display format
            except Exception:
                return None
        for key, val in ASSET_FILES.items():
            if isinstance(val, list):
                surfs = [s for s in (load(n) for n in val) if s is not None]
                if surfs:
                    self._assets[key] = surfs
            else:
                s = load(val)
                if s is not None:
                    self._assets[key] = s
        self._calibrate_chicken_scale()

    @staticmethod
    def _alpha_maxdim(surf):
        """Largest bbox dimension (px) of a sprite's opaque pixels — a size proxy for the subject."""
        a = pygame.surfarray.array_alpha(surf)
        xs, ys = np.where(a > 24)
        if not len(xs):
            return surf.get_width()
        return max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))

    def _calibrate_chicken_scale(self):
        """Equalize on-screen BODY size across the chicken anim sheets. slice_sheet scales each
        sheet to its WIDEST frame, so the wings-spread RUN frames leave the run BODY small in its
        canvas -> it'd render smaller than the peck/walk hen at the same diameter. Rescale each
        sheet's draw diameter by (max body proxy / this sheet's body proxy). Body proxy = the sheet's
        MOST COMPACT frame (wings tucked ~= body only), so run scales up to match (its wings stay
        proportionally bigger, which suits a panicked flee). Self-calibrating: survives regeneration."""
        refs = {}
        for key in ("chicken_peck", "chicken_walk", "chicken_run", "chicken_fall"):
            frames = self._assets.get(key)
            if frames:
                refs[key] = min(self._alpha_maxdim(f) for f in frames)
        target = max(refs.values()) if refs else 1
        self._chick_scale = {k: target / v for k, v in refs.items()}

    def _ensure(self, world):
        key = (round(float(world.size[0]), 3), round(float(world.size[1]), 3))
        if self.display is not None and self._world_key == key:
            return
        self._world_key = key
        fs = bool(self.fullscreen and self.screen_size)
        if fs:
            # FULLSCREEN|SCALED renders at the requested logical size and lets SDL scale to the panel.
            self.display = pygame.display.set_mode(self.screen_size, pygame.FULLSCREEN | pygame.SCALED, vsync=1)
        else:
            base = self.scale or max(4, int(TARGET_PX / max(world.size)))
            self.display = pygame.display.set_mode((int(world.size[0] * base), int(world.size[1] * base)))
        pygame.display.set_caption("Snake-RL")
        pygame.mouse.set_visible(not fs)
        dw, dh = self.display.get_size()              # trust what SDL actually granted (Retina-aware)
        self.dw, self.dh = dw, dh
        self._ss = 2 if max(dw, dh) <= 1100 else 1    # supersample small windows; native res is crisp enough
        self.cw, self.ch = dw * self._ss, dh * self._ss
        self._scale = self.cw / world.size[0]         # world units -> canvas px
        self.canvas = pygame.Surface((self.cw, self.ch)).convert()
        self._sprite_cache = {}                       # fresh sprites for the new surface format (Pitfall 11)
        self._load_assets()
        self._ground_surf = self._build_ground()
        self._motes = self._make_motes(world)

    def _p(self, xy):
        return (int(xy[0] * self._scale), int(xy[1] * self._scale))

    # --- torus-aware primitives ---
    def _circle(self, color, pos, r):
        bx, by = int(pos[0] * self._scale), int(pos[1] * self._scale)
        for ox in (0, -self.cw, self.cw):
            for oy in (0, -self.ch, self.ch):
                if -r <= bx + ox <= self.cw + r and -r <= by + oy <= self.ch + r:
                    pygame.draw.circle(self.canvas, color, (bx + ox, by + oy), r)

    def _blit_world(self, surf, pos, off=(0, 0)):
        """Blit a surface centered at a world position, repeated across the torus seams."""
        w, h = surf.get_size()
        bx = pos[0] * self._scale - w / 2 + off[0]
        by = pos[1] * self._scale - h / 2 + off[1]
        for ox in (0, -self.cw, self.cw):
            for oy in (0, -self.ch, self.ch):
                x, y = bx + ox, by + oy
                if -w < x < self.cw and -h < y < self.ch:
                    self.canvas.blit(surf, (int(x), int(y)))

    def _sprite(self, key, diam, angle=0.0, variant=0, tint=None):
        """Scaled (+optionally rotated/tinted) cached sprite, or None if the asset is missing.
        angle is a WORLD heading in degrees; the source art faces +X (right)."""
        base = self._assets.get(key)
        if base is None:
            return None
        if isinstance(base, list):
            base = base[variant % len(base)]
        diam = max(2, int(diam))
        ab = int(round(angle / 6.0)) * 6 % 360        # 6-degree rotation buckets bound the cache
        ck = (key, id(base), diam, ab, tint)
        s = self._sprite_cache.get(ck)
        if s is None:
            scale = diam / base.get_width()
            s = pygame.transform.rotozoom(base, -ab, scale)   # rotozoom: smoothed rotate+scale; -ab: screen y is down
            if tint is not None:
                s = s.copy()
                s.fill((*tint, 255), special_flags=pygame.BLEND_RGBA_MULT)
            self._sprite_cache[ck] = s
        return s

    def _dot(self, rpx, color, alpha):
        rpx = max(1, int(rpx)); alpha = min(255, max(0, int(alpha)) // 16 * 16)
        key = ("dot", rpx, color, alpha)
        s = self._sprite_cache.get(key)
        if s is None:
            s = pygame.Surface((2 * rpx, 2 * rpx), pygame.SRCALPHA)
            pygame.draw.circle(s, (*color, alpha), (rpx, rpx), rpx)
            s = s.convert_alpha()
            self._sprite_cache[key] = s
        return s

    def _shadow(self, pos, r_px):
        self._blit_world(self._shadow_sprite(int(r_px * 1.3)), pos, off=(0, int(r_px * 0.32)))

    def _shadow_sprite(self, r):
        key = ("shadow", r)
        s = self._sprite_cache.get(key)
        if s is None:
            s = pygame.Surface((2 * r, 2 * r), pygame.SRCALPHA)
            for rr in range(r, 0, -1):
                a = int(120 * (1 - rr / r) ** 2)
                pygame.draw.circle(s, (0, 0, 0, a), (r, r), rr)
            s = s.convert_alpha()
            self._sprite_cache[key] = s
        return s

    def _anim_frame(self, nframes, fps, phase=0.0):
        """Which sheet frame to show now: wall-clock time * fps, +phase (a per-entity offset so
        entities don't animate in lockstep), wrapped to [0, nframes). Smooth at any sim speed."""
        return int(self._clock * fps + phase) % nframes

    def _make_motes(self, world):
        rng = np.random.default_rng(12345)
        n = 60
        pos = rng.uniform([0, 0], world.size, size=(n, 2))
        vel = rng.uniform(-0.05, 0.05, size=(n, 2))
        return np.hstack([pos, vel])

    def _build_ground(self):
        """Pre-tile the world backdrop into one canvas-sized surface (blit once/frame). Uses the 6
        seamless grass VARIANTS as a hashed patchwork: each grid cell picks a variant deterministically
        from (col,row) -- stable per-frame AND frame-to-frame (no flicker) -- so the ground reads as
        natural terrain variation, not one repeated tile. The variants are tone-matched to a shared
        mean (kills rectangular brightness blocks between cells) and cross-faded over a small overlap
        band at the cell edges (soft transitions; interiors stay single-tile crisp), so no hard seam
        shows. Falls back to a single ground.png, then a procedural grid, if assets are missing."""
        surf = pygame.Surface((self.cw, self.ch)).convert()
        variants = self._assets.get("ground_set") or (
            [self._assets["ground"]] if "ground" in self._assets else None)
        if not variants:
            surf.fill(BG)
            step = max(6, int(6 * self._scale))
            for x in range(0, self.cw, step):
                pygame.draw.line(surf, GRID, (x, 0), (x, self.ch))
            for y in range(0, self.ch, step):
                pygame.draw.line(surf, GRID, (0, y), (self.cw, y))
            return surf
        tile_px = int(np.clip(min(self.cw, self.ch) / 3.2, 220, 640))
        f = max(8, tile_px // 6)                                # cross-fade band width
        base, feather = self._ground_tileset(variants, tile_px, f)
        n = len(base)

        def pick(c, r):                                        # deterministic per-cell variant
            return ((c * 73856093) ^ (r * 19349663)) % n
        cols, rows = self.cw // tile_px + 1, self.ch // tile_px + 1
        for r in range(rows):                                  # opaque base pass: crisp interiors
            for c in range(cols):
                surf.blit(base[pick(c, r)], (c * tile_px, r * tile_px))
        for r in range(rows):                                  # feather pass: cross-fade the boundaries
            for c in range(cols):
                surf.blit(feather[pick(c, r)], (c * tile_px - f, r * tile_px - f))
        return surf

    def _ground_tileset(self, variants, T, f):
        """Prepare the patchwork tiles once: each variant tone-matched (85% toward the shared mean)
        and scaled to T (the crisp opaque base), plus an oversized (T+2f) copy with a feathered
        alpha edge for the overlap cross-fade. Returns (base_surfs, feather_surfs)."""
        arrs = [np.transpose(pygame.surfarray.array3d(pygame.transform.smoothscale(v, (T, T))),
                             (1, 0, 2)).astype(np.float32) for v in variants]
        gmean = np.mean([a.reshape(-1, 3).mean(0) for a in arrs], axis=0)
        TT = T + 2 * f
        w = np.clip(np.minimum(np.arange(TT), TT - 1 - np.arange(TT)) / f, 0.0, 1.0)   # edge ramp 0->1
        alpha = (np.minimum(w[:, None], w[None, :]) * 255).astype(np.uint8)            # symmetric, [x,y]
        base, feather = [], []
        for a in arrs:
            eq = np.clip(a + 0.85 * (gmean - a.reshape(-1, 3).mean(0)), 0, 255).astype(np.uint8)
            b = pygame.surfarray.make_surface(np.transpose(eq, (1, 0, 2))).convert()
            base.append(b)
            big = pygame.transform.smoothscale(b, (TT, TT)).convert_alpha()
            av = pygame.surfarray.pixels_alpha(big)
            av[:] = alpha                                     # feather the edges to 0 (overlap blend)
            del av                                            # unlock the surface before blitting
            feather.append(big)
        return base, feather

    # --- scene layers ---
    def _draw_obstacles(self, world):
        for pos, r, kind in zip(world.obstacle_pos, world.obstacle_r, world.obstacle_kind):
            rp = int(r * self._scale)
            self._shadow(pos, rp)
            ang = (int(pos[0] * 7.0 + pos[1] * 13.0) % 360)          # stable per-instance rotation for variety
            var = int(pos[0] * 3 + pos[1] * 5) % 2
            if kind == 1:                                            # tree
                s = self._sprite("tree", 2.7 * r * self._scale, angle=ang, variant=var)
                if s is not None:
                    # gentle wind sway: crown drifts in a small ellipse, phase from position so
                    # trees don't sway in sync (top-down, so the crown translates rather than tilts).
                    ph = pos[0] * 0.7 + pos[1] * 1.3
                    sway = (0.10 * r * self._scale * np.sin(self._clock * 0.9 + ph),
                            0.05 * r * self._scale * np.sin(self._clock * 1.3 + ph * 1.7))
                    self._blit_world(s, pos, off=sway)
                    continue
                self._circle(TRUNK, pos, max(2, rp // 3))
                self._circle(LEAF, pos, rp)
                self._circle(LEAF2, pos + np.array([-r * 0.25, -r * 0.3]), int(rp * 0.66))
            else:                                                    # rock
                s = self._sprite("rock", 2.25 * r * self._scale, angle=ang, variant=var)
                if s is not None:
                    self._blit_world(s, pos)
                    continue
                self._circle(ROCK_SH, pos, rp)
                self._circle(ROCK, pos, int(rp * 0.9))
                self._circle(ROCK_HI, pos + np.array([-r * 0.3, -r * 0.32]), int(rp * 0.38))

    def _draw_decals(self):
        alive = []
        for d in self._decals:
            x, y, variant, angle, diam, age, life = d
            s = self._sprite("blood", diam * self._scale, angle=angle, variant=variant)
            if s is not None:
                a = 255 if age < life * 0.55 else int(255 * (1 - (age - life * 0.55) / (life * 0.45)))
                if a > 4:
                    self._blit_world(_faded(s, a), (x, y))
            if age + 1 < life:
                d[5] = age + 1
                alive.append(d)
        self._decals = alive[-48:]                                  # bound the count (drop oldest)

    def _draw_corpses(self, world):
        corpses = world.corpses
        if not len(corpses["pos"]):
            return
        cfg = world.cfg
        for pos, food in zip(corpses["pos"], corpses["food"]):
            p = wrap(pos, world.size)
            r = float(np.clip(food / (cfg.corpse_food_per_length * 6), 0.7, 2.4))
            rp = int(r * self._scale)
            self._shadow(p, rp)
            s = self._sprite("corpse", 3.0 * r * self._scale, angle=(int(p[0] * 11 + p[1] * 7) % 360))
            if s is not None:
                self._blit_world(s, p)
                continue
            for ox, oy in [(0, 0), (-0.5, 0.35), (0.5, 0.3), (-0.15, -0.4)]:
                self._circle(CORPSE, p + np.array([ox, oy]) * r, max(1, int(rp * 0.6)))

    def _draw_eggs(self, world):
        e = world.eggs
        if not len(e["pos"]):
            return
        c = world.cfg
        pulse = 0.94 + 0.06 * np.sin(self._clock * 2.2)              # gentle clock-driven breathing
        for pos, timer in zip(e["pos"], e["timer"]):
            p = wrap(pos, world.size)
            r = c.egg_radius * pulse
            rp = max(2, int(r * self._scale))
            self._shadow(p, rp)
            about_to_hatch = timer <= 6
            s = self._sprite("egg_cracked" if about_to_hatch else "egg", 2.6 * r * self._scale,
                             angle=(int(p[0] * 5 + p[1] * 9) % 360))
            if s is not None:
                # rattle: still normally, shivering harder the closer it is to hatching (anticipation)
                ph = p[0] * 1.7 + p[1] * 0.9
                rattle = 1.0 + 3.0 * max(0.0, 1.0 - timer / 30.0)
                wob = 0.06 * c.egg_radius * self._scale * rattle
                off = (wob * np.sin(self._clock * 9.0 * rattle + ph),
                       0.4 * wob * np.sin(self._clock * 7.0 + ph))
                self._blit_world(s, p, off=off)
                continue
            self._circle(EGG_SHELL, p, rp)
            self._circle(EGG_SHADE, p + np.array([-r * 0.28, -r * 0.28]), max(1, int(rp * 0.45)))

    def _draw_chickens(self, world, positions, dirs):
        cr = world.cfg.chicken_radius
        rp = max(4, int(cr * self._scale))
        diam = 3.4 * cr * self._scale
        # positions/dirs arrive in world-chicken order (both watch.py's interp path and the default
        # path), so chicken_state[i]/chicken_id[i] line up by index -> pick the anim + a stable phase.
        states, ids = world.chicken_state, world.chicken_id
        for i, (cpos, cdir) in enumerate(zip(positions, dirs)):
            p = wrap(cpos, world.size)
            self._shadow(p, rp)
            st = int(states[i]) if i < len(states) else 1
            phase = (int(ids[i]) * 1.7) if i < len(ids) else 0.0     # per-chicken offset (out of lockstep)
            key, fps, nf = CHICK_ANIM[st]
            s = self._sprite(key, diam * self._chick_scale.get(key, 1.0), angle=float(np.degrees(cdir)),
                             variant=self._anim_frame(nf, fps, phase))
            if s is None:                                            # sheet missing -> static hen sprite
                s = self._sprite("chicken", diam, angle=float(np.degrees(cdir)))
            if s is not None:
                self._blit_world(s, p)
                continue
            # last-resort procedural hen (no sprites at all): a state-aware bob so it still animates.
            bob = 0.12 * cr * np.sin(self._clock * fps + phase) * (0.5 if st == 0 else 1.0)
            p = p + np.array([0.0, bob])
            for dx in (-0.3, 0.0, 0.3):
                pygame.draw.circle(self.canvas, COMB, self._p(p + np.array([dx * cr, -cr * 0.7])),
                                   max(2, int(cr * 0.24 * self._scale)))
            self._circle(CHICK_SH, p + np.array([0, cr * 0.28]), rp)
            self._circle(CHICK, p, int(cr * 0.9 * self._scale))
            d = np.array([np.cos(cdir), np.sin(cdir)]); perp = np.array([-d[1], d[0]])
            pygame.draw.circle(self.canvas, BEAK, self._p(p + d * cr * 0.95), max(2, int(cr * 0.32 * self._scale)))
            pygame.draw.circle(self.canvas, EYE, self._p(p + d * cr * 0.3 + perp * cr * 0.35),
                               max(1, int(cr * 0.2 * self._scale)))

    def _drop_shadow(self, pos, r_px, prog):
        """Ground shadow of a falling object: high up it's small, faint and soft; as `prog`->1 it
        GROWS (r_px is passed in bigger), DARKENS and SHARPENS (harder edge) up to touchdown."""
        r_px = max(2, int(r_px))
        hard = 0.3 + 0.7 * float(np.clip(prog, 0.0, 1.0))               # darkness + edge sharpness
        key = ("dropshadow", r_px, round(hard, 1))
        s = self._sprite_cache.get(key)
        if s is None:
            s = pygame.Surface((2 * r_px, 2 * r_px), pygame.SRCALPHA)
            exp = 2.0 - 1.3 * hard                                      # soft(2.0) up high -> hard(0.7) edge
            for rr in range(r_px, 0, -1):
                a = int(150 * hard * (1 - rr / r_px) ** exp)
                pygame.draw.circle(s, (0, 0, 0, a), (r_px, r_px), rr)
            s = s.convert_alpha()
            self._sprite_cache[key] = s
        self._blit_world(s, pos)

    def _draw_arrivals(self, world):
        """Chickens dropping in from the sky (world.arriving): a top-down hen that FALLS with gravity
        (ease-in), flapping its wings and rocking as it flutters, SHRINKING toward the ground plane
        (it started nearer the overhead camera), pointing its head a stable RANDOM way, over a
        drop-shadow that grows/darkens/sharpens up to touchdown -- then a dust puff (spawn_land, fired
        from watch on the real landing). The discrete per-sim-tick descent is damped-followed into a
        smooth glide (FALL_SMOOTH). In-flight birds are separate from world.chicken_pos, so they're
        not huntable/sensed until they land (Goal 2). Kept subtle -- it's a screensaver."""
        arr = world.arriving
        seen = set()
        if len(arr["pos"]):
            cr = world.cfg.chicken_radius
            steps = max(1, world.cfg.chicken_arrive_steps)
            base_diam = 3.4 * cr * self._scale
            rp0 = max(2.0, cr * self._scale)
            heads = arr.get("head")                                    # world.py may carry the landing heading
            for i, (pos, timer) in enumerate(zip(arr["pos"], arr["timer"])):
                p = wrap(pos, world.size)
                key = (round(float(pos[0]), 2), round(float(pos[1]), 2))   # a falling bird is stationary in x/y
                seen.add(key)
                target = 1.0 - timer / steps                          # 0 high in the sky -> 1 landed
                st = self._arrivals.get(key)
                if st is None:
                    st = self._arrivals[key] = {"prog": target}       # first sighting: start exact, then glide
                st["prog"] += (target - st["prog"]) * FALL_SMOOTH     # smooth the per-tick step into a glide
                prog = float(np.clip(st["prog"], 0.0, 1.0))
                # head direction: use world's landing heading if it ever carries one (then the fall
                # matches the landed hen); else a stable per-position pseudo-random angle (varied, not
                # all head-up, and steady across the whole fall).
                head = (float(heads[i]) if heads is not None and i < len(heads)
                        else float((pos[0] * 0.7 + pos[1] * 1.3) % (2 * np.pi)))
                self._drop_shadow(p, rp0 * (0.5 + 0.9 * prog), prog)  # true ground spot: grows toward land
                up = ARRIVE_FALL * (1.0 - prog * prog) * self._scale  # ease-in (gravity) descent, screen px
                depth = 1.0 + FALL_ZOOM * (1.0 - prog)                # bigger up high -> recedes to 1x
                phase = p[0] * 1.9 + p[1] * 1.1                       # per-bird desync (flap + wobble)
                wob = FALL_WOBBLE_DEG * float(np.sin(self._clock * FALL_WOBBLE_HZ + phase))   # flutter rock
                angle = float(np.degrees(head)) + 90.0 + wob         # +90: the fall art faces "up", not +X
                flap = self._anim_frame(4, FALL_FLAP_FPS, phase)
                s = (self._sprite("chicken_fall", base_diam * depth * self._chick_scale.get("chicken_fall", 1.0),
                                  angle=angle, variant=flap, tint=CHICK_FALL_TINT)   # -> the LANDED hen's cream
                     or self._sprite("chicken_run", base_diam * depth * self._chick_scale.get("chicken_run", 1.0),
                                     angle=angle, variant=flap)       # wings-spread flap fallback
                     or self._sprite("chicken", base_diam * depth, angle=angle))
                if s is not None:
                    if prog < FALL_FADEIN:                            # enters from the sky -> fade in, no pop
                        s = _faded(s, int(255 * prog / FALL_FADEIN))
                    self._blit_world(s, p, off=(0, -up))
                else:
                    self._circle(CHICK, p + np.array([0.0, -up / self._scale]), int(rp0 * depth))
        for k in list(self._arrivals):                               # forget landed / gone birds
            if k not in seen:
                del self._arrivals[k]

    def _draw_snake(self, world, snake, body_uw, big=False):
        n = len(body_uw)
        hr = world.cfg.head_radius
        head_c, tail_c = _snake_colors(snake.color_seed)
        wpts = [wrap(body_uw[k], world.size) for k in range(n)]
        radii = [hr * (1 - k / max(1, n - 1)) + world.cfg.body_radius * 0.7 * (k / max(1, n - 1)) for k in range(n)]
        for k in range(n - 1, -1, -1):                     # rim pass
            self._circle(SNAKE_RIM, wpts[k], max(3, int((radii[k] + 0.28) * self._scale)))
        for k in range(n - 1, -1, -1):                     # body pass
            self._circle(_lerp(head_c, tail_c, k / max(1, n - 1)), wpts[k], max(2, int(radii[k] * self._scale)))
        for k in range(min(n, n * 3 // 4) - 1, -1, -1):    # gloss highlight (front ~3/4)
            self._circle(SNAKE_GLOSS, wpts[k] - np.array([hr * 0.18, hr * 0.22]), max(1, int(radii[k] * 0.34 * self._scale)))
        head = wpts[0]
        d = body_uw[0] - body_uw[1] if n > 1 else snake.heading_vec()
        nrm = np.linalg.norm(d); d = d / nrm if nrm > 1e-6 else snake.heading_vec()
        perp = np.array([-d[1], d[0]])
        if snake.dashed:                                   # dash kick: dust puffs off the tail-side of the head
            self._spawn_dust(head - d * hr * 0.9, -d, hr)
        head_sprite = self._sprite("head", 3.0 * hr * self._scale,
                                   angle=float(np.degrees(np.arctan2(d[1], d[0]))),
                                   tint=color_for(snake.color_seed, s=0.62, v=0.98))
        if head_sprite is not None:
            self._blit_world(head_sprite, head)
        flick = snake.dashed or (self._t % 48 < 6)         # forked tongue tasting the air
        if flick:
            reach = 1.5 if snake.dashed else 1.2
            tip = head + d * hr * reach; base = head + d * hr * 0.9
            pygame.draw.line(self.canvas, TONGUE, self._p(base), self._p(tip), max(2, int(0.14 * self._scale)))
            for s2 in (1, -1):
                pygame.draw.line(self.canvas, TONGUE, self._p(tip),
                                 self._p(tip + (d * 0.45 + perp * 0.45 * s2) * hr), max(2, int(0.12 * self._scale)))
        if head_sprite is None:                            # procedural eyes only when no head sprite
            for s2 in (1, -1):
                e = head + d * hr * 0.32 + perp * hr * 0.46 * s2
                pygame.draw.circle(self.canvas, SNAKE_EYE, self._p(e), max(2, int(hr * 0.3 * self._scale)))
                pygame.draw.circle(self.canvas, PUPIL, self._p(e + d * hr * 0.13), max(1, int(hr * 0.15 * self._scale)))

    def _ring_hud(self, world, snake, head_pos, big=False):
        """Per-snake 3-ring badge above the head: outer=energy(green), middle=stamina(cyan),
        inner=length(amber); a thin outline in the snake's hue groups the badge to its owner.
        `head_pos` is the (interpolated) wrapped head the body was drawn at, so it doesn't jitter."""
        c = world.cfg
        hr = c.head_radius
        r_out = hr * (2.4 if big else 1.6)
        step = r_out * 0.3
        center = head_pos + np.array([0.0, -(r_out + hr * 1.3)])
        p = self._p(center)
        lw = max(2, int((0.22 if big else 0.15) * self._scale))
        outline_r = max(3, int((r_out + step * 0.6) * self._scale))
        pygame.draw.circle(self.canvas, color_for(snake.color_seed), p, outline_r, 1)
        fracs = (np.clip(snake.energy / c.energy_max, 0.0, 1.0),
                 np.clip(snake.stamina / c.s_max, 0.0, 1.0),
                 np.clip(snake.target_length / c.length_cap, 0.0, 1.0))
        for i, frac in enumerate(fracs):
            r_px = max(3, int((r_out - i * step) * self._scale))
            pygame.draw.circle(self.canvas, RING_TRACK, p, r_px, max(1, lw // 2))
            if frac > 0.01:
                start = -np.pi / 2
                end = start + float(frac) * 2 * np.pi
                rect = pygame.Rect(p[0] - r_px, p[1] - r_px, 2 * r_px, 2 * r_px)
                pygame.draw.arc(self.canvas, RING_COLORS[i], rect, start, end, lw)

    # --- gore / effects ---
    def spawn_eat(self, pos):
        """Chicken eaten: a burst of blood droplets + a few flesh/feather bits + a small decal."""
        self._blood_burst(pos, n=15, smin=0.35, smax=1.25, lmin=16, lmax=32)
        self._gore_bits(pos, n=4, big=False)
        self._add_decal(pos, dmin=2.6, dmax=4.2)

    def spawn_death(self, pos):
        """Snake death: a bigger blood burst + gut bits + a large decal (the corpse sprite is
        drawn separately from world.corpses)."""
        self._blood_burst(pos, n=28, smin=0.5, smax=1.9, lmin=22, lmax=46)
        self._gore_bits(pos, n=9, big=True)
        self._add_decal(pos, dmin=5.5, dmax=8.5)

    def spawn_hatch(self, pos):
        """Egg hatch: the shell physically cracks apart -- shell shards fly out + a cracked-egg
        sprite lingers a moment. No sparkles."""
        rng = np.random.default_rng(int(self._t * 131 + pos[0] * 7 + pos[1] * 13))
        for _ in range(9):
            a = rng.uniform(0, 2 * np.pi); sp = rng.uniform(0.3, 1.0)
            self._particles.append([float(pos[0]), float(pos[1]),
                                    np.cos(a) * sp, np.sin(a) * sp, 0, int(rng.integers(14, 26)),
                                    rng.uniform(0.16, 0.30), SHELL[int(rng.integers(0, len(SHELL)))], 0.03, 0.0])
        self._transient.append(["egg_cracked", float(pos[0]), float(pos[1]), 0, 22])

    def spawn_land(self, pos):
        """A sky-dropped chicken touched down: a small, low dust puff kicked outward (Goal 2)."""
        rng = np.random.default_rng(int(self._t * 61 + pos[0] * 17 + pos[1] * 23))
        for _ in range(6):
            a = rng.uniform(0, 2 * np.pi); sp = rng.uniform(0.15, 0.5)
            self._particles.append([float(pos[0]), float(pos[1]),
                                    np.cos(a) * sp, np.sin(a) * sp - 0.05, 0, int(rng.integers(10, 20)),
                                    rng.uniform(0.12, 0.24), DUST[int(rng.integers(0, len(DUST)))], 0.0, 0.015])

    def _blood_burst(self, pos, n, smin, smax, lmin, lmax):
        rng = np.random.default_rng(int(self._t * 97 + pos[0] * 31 + pos[1] * 17))
        for _ in range(n):
            a = rng.uniform(0, 2 * np.pi); sp = rng.uniform(smin, smax)
            self._particles.append([float(pos[0]), float(pos[1]),
                                    np.cos(a) * sp, np.sin(a) * sp - rng.uniform(0.1, 0.6),
                                    0, int(rng.integers(lmin, lmax)),
                                    rng.uniform(0.10, 0.30), BLOOD[int(rng.integers(0, len(BLOOD)))], 0.075, 0.0])

    def _gore_bits(self, pos, n, big):
        rng = np.random.default_rng(int(self._t * 53 + pos[0] * 19 + pos[1] * 29))
        for _ in range(n):
            a = rng.uniform(0, 2 * np.pi); sp = rng.uniform(0.2, 0.9) * (1.4 if big else 1.0)
            self._particles.append([float(pos[0]), float(pos[1]),
                                    np.cos(a) * sp, np.sin(a) * sp - 0.3, 0, int(rng.integers(26, 52)),
                                    rng.uniform(0.24, 0.5) * (1.4 if big else 1.0),
                                    GORE[int(rng.integers(0, len(GORE)))], 0.06, 0.0])

    def _spawn_dust(self, pos, back, hr):
        rng = np.random.default_rng(int(self._t * 41 + pos[0] * 13 + pos[1] * 7))
        for _ in range(2):
            j = rng.uniform(-0.4, 0.4)
            v = back * rng.uniform(0.1, 0.35) + np.array([-back[1], back[0]]) * j
            self._particles.append([float(pos[0]), float(pos[1]), float(v[0]), float(v[1]),
                                    0, int(rng.integers(10, 20)), rng.uniform(0.18, 0.34),
                                    DUST[int(rng.integers(0, len(DUST)))], 0.0, 0.02])

    def _add_decal(self, pos, dmin, dmax):
        rng = np.random.default_rng(int(self._t * 71 + pos[0] * 23 + pos[1] * 11))
        self._decals.append([float(pos[0]), float(pos[1]), int(rng.integers(0, 2)),
                             float(rng.uniform(0, 360)), float(rng.uniform(dmin, dmax)),
                             0, int(rng.integers(700, 1100))])

    def _draw_particles(self):
        alive = []
        for pr in self._particles:
            pr[0] += pr[2]; pr[1] += pr[3]; pr[3] += pr[8]; pr[4] += 1     # move + gravity + age
            age, life = pr[4], pr[5]
            if age >= life:
                continue
            alive.append(pr)
            alpha = int(235 * (1 - age / life))
            if alpha <= 4:
                continue
            rpx = max(1, int((pr[6] + pr[9] * age) * self._scale))
            self._blit_world(self._dot(rpx, pr[7], alpha), (pr[0], pr[1]))
        self._particles = alive

    def _draw_transient(self, world):
        alive = []
        for tr in self._transient:
            kind, x, y, age, life = tr
            frac = age / life
            diam = (2.6 + 1.4 * frac) * world.cfg.egg_radius * self._scale     # crack expands slightly
            s = self._sprite(kind, diam, angle=(int(x * 5 + y * 9) % 360))
            if s is not None:
                a = int(255 * (1 - frac))
                if a > 4:
                    self._blit_world(_faded(s, a), (x, y))
            if age + 1 < life:
                tr[3] = age + 1
                alive.append(tr)
        self._transient = alive

    def _draw_motes(self, world):
        if self._motes is None:
            return
        m = self._motes
        m[:, 0] = (m[:, 0] + m[:, 2]) % world.size[0]
        m[:, 1] = (m[:, 1] + m[:, 3]) % world.size[1]
        rpx = max(1, int(0.18 * self._scale))
        for idx, (x, y, _vx, _vy) in enumerate(m):
            tw = 0.5 + 0.5 * np.sin(self._clock * 1.5 + idx * 2.399)   # per-mote firefly twinkle
            self._blit_world(self._dot(rpx, (236, 244, 206), 22 + int(tw * 48)), (x, y))

    def _draw_sensors(self, world, head_uw, heading, snake=None):
        head = wrap(head_uw, world.size)
        dirs, dist, kinds = vision_distances(world, head, heading, snake)
        for u, dd, kd in zip(dirs, dist, kinds):
            nseg = max(2, int(dd))
            pts = [np.asarray(self._p(wrap(head + u * (dd * k / nseg), world.size))) for k in range(nseg + 1)]
            col = RAY_KIND[int(kd)]
            for a, b in zip(pts, pts[1:]):
                if abs(a[0] - b[0]) < self.cw / 2 and abs(a[1] - b[1]) < self.ch / 2:
                    pygame.draw.line(self.canvas, col, a, b, max(1, SS))
            if kd != -1:
                pygame.draw.circle(self.canvas, col, tuple(pts[-1]), max(2, SS + 1))

    def draw(self, world, bodies=None, chick_pos=None, chick_dir=None, follow_id=None):
        """Composite the scene in depth order: ground -> obstacles -> blood decals -> corpses ->
        eggs -> chickens -> sky-dropping chickens -> snakes -> gore particles -> ambient -> HUD/sensors.
        `bodies`: optional {snake_id: interpolated unwrapped body polyline}. `follow_id`: which
        snake gets the larger ring HUD + sensor overlay (defaults to slot-0)."""
        self._t += 1
        self._clock = self._clock_override if self._clock_override is not None \
            else pygame.time.get_ticks() / 1000.0     # wall-clock anim time (smooth, sim-independent)
        self._ensure(world)
        if chick_pos is None:
            chick_pos, chick_dir = world.chicken_pos, world.chicken_dir
        self.canvas.blit(self._ground_surf, (0, 0))
        self._draw_obstacles(world)
        self._draw_decals()
        self._draw_corpses(world)
        self._draw_eggs(world)
        self._draw_transient(world)
        self._draw_chickens(world, chick_pos, chick_dir)
        self._draw_arrivals(world)                                   # sky-dropping chickens (Goal 2)
        follow = follow_id if follow_id is not None else (world.snakes[0].id if world.snakes else -1)
        sensor_snake = None
        for s in world.snakes:
            if not s.alive:
                continue
            big = (s.id == follow)
            b = (bodies or {}).get(s.id)
            if b is None:
                b = world._body_render_path_uw(s)
            self._draw_snake(world, s, b, big=big)
            if self.show_rings:
                self._ring_hud(world, s, wrap(b[0], world.size), big=big)
            if big:
                sensor_snake = (s, b)
        self._draw_particles()
        self._draw_motes(world)
        if self.show_sensors and sensor_snake is not None:
            s, b = sensor_snake
            d = b[0] - b[1] if len(b) > 1 else s.heading_vec()
            nrm = np.linalg.norm(d); d = d / nrm if nrm > 1e-6 else s.heading_vec()
            self._draw_sensors(world, b[0], float(np.arctan2(d[1], d[0])), s)
        # Blit into the window via a temp surface (writing straight into the window renders black on
        # some backends). At SS=1 the canvas is already display-sized.
        if self._ss == 1:
            self.display.blit(self.canvas, (0, 0))
        else:
            self.display.blit(pygame.transform.smoothscale(self.canvas, (self.dw, self.dh)), (0, 0))
        pygame.display.flip()

    def toggle_sensors(self):
        self.show_sensors = not self.show_sensors

    def toggle_rings(self):
        self.show_rings = not self.show_rings

    def close(self):
        pygame.quit()


def _faded(surf, a):
    """Return `surf` with its alpha channel scaled by a/255 (for fading per-pixel-alpha sprites;
    Surface.set_alpha is ignored on SRCALPHA surfaces, so multiply the alpha channel instead)."""
    if a >= 255:
        return surf
    c = surf.copy()
    c.fill((255, 255, 255, max(0, a)), special_flags=pygame.BLEND_RGBA_MULT)
    return c
