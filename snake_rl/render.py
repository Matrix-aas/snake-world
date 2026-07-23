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
from .world import wrap, torus_delta

SS = 2                   # supersample factor (draw big, smoothscale down for anti-aliasing)
TARGET_PX = 860          # display fits ~this many pixels on its long side
ZOOM_MIN, ZOOM_MAX = 1.0, 6.0     # camera zoom clamp (1 = whole world fits; >1 zooms in)
GROUND_TILE_WORLD = 44.0          # world-units spanned by one procedural ground tile (camera-locked)
ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")
ASSET_FILES = {
    # (ground is now a procedural tile -- see _build_ground_tile; no grass PNGs loaded)
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

# genome inspector (Phase B increment 3): short labels for the 9 genes, in genome.py index order
# (SIZE, METABOLISM, SPEED, STAMINA, SENSES, LIFESPAN, AGGRESSION, KIN_CARE, BOLDNESS).
GENE_LABELS = ("size", "metab", "speed", "stamina", "senses", "lifespan", "aggr", "kin", "bold")

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


def _hsv(hue, s, v):
    r, g, b = colorsys.hsv_to_rgb((hue % 360) / 360.0, max(0.0, min(1.0, s)), max(0.0, min(1.0, v)))
    return (int(r * 255), int(g * 255), int(b * 255))


def _snake_palette(snake):
    """Genome -> visible phenotype colors (Phase B increment 2). Base HUE = `snake.lineage` (a stable
    FAMILY color carried down a maternal line via golden angle, so kin read alike across generations);
    the `aggression` gene warms the hue toward red and raises saturation (vivid = fierce), and
    `metabolism` raises brightness (fast burners read brighter). Returns (head_color, tail_color,
    head_tint) for the tapered body gradient + the head-sprite tint. `size` drives the DRAWN body
    scale in _draw_snake (on top of the longer body its max_length already gives)."""
    from .genome import METABOLISM, AGGRESSION
    g = snake.genome
    aggr = float(g[AGGRESSION]); metab = float(g[METABOLISM])
    hue = (snake.lineage * 137.5 - 30.0 * aggr) % 360.0     # family hue, warmed toward red by aggression
    sat = 0.42 + 0.30 * aggr                                # aggressive lines are more saturated
    val = 0.80 + 0.18 * metab                               # fast metabolism reads brighter
    head_c = _hsv(hue, sat * 0.85, min(1.0, val + 0.04))
    tail_c = _hsv(hue, min(1.0, sat + 0.30), val * 0.5)
    head_tint = _hsv(hue, sat, min(1.0, val + 0.06))
    return head_c, tail_c, head_tint


def _lerp(a, b, t):
    return (int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t), int(a[2] + (b[2] - a[2]) * t))


class Renderer:
    def __init__(self, scale=None, show_sensors=False, show_rings=False, fullscreen=False, screen_size=None,
                 show_inspector=False):
        pygame.init()
        self.scale = scale
        self.show_sensors = show_sensors      # vision-ray overlay (toggle key: S) -- OFF by default
        self.show_rings = show_rings          # per-snake ring HUD circles (toggle key: H) -- OFF by default
        self.show_inspector = show_inspector  # genome inspector for the followed snake (toggle key: I)
        self.fullscreen = fullscreen
        self.screen_size = screen_size
        self.canvas = self.display = None
        self.cw = self.ch = self.dw = self.dh = 0
        self._scale = 1
        self._base_scale = 1       # world->canvas px at zoom 1 (fit-to-screen); _scale = base * zoom
        self.zoom = 1.0
        self._camc = None          # camera center in world units (None -> world center, set each draw)
        self._wsize = None
        self._perx = self._pery = 0  # torus period in canvas px (world.size * _scale)
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
        self._ground_tile0 = None       # seamless procedural ground tile (camera-locked tiling, polish #3)
        self._ground_tile_cache = {}    # {tile_px: scaled tile} -- multi-bucket so zoom-ease doesn't thrash
        self._vignette = None           # cached cinematic edge-darkening overlay (polish #4)

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
        # base scale FITS THE WHOLE WORLD in the frame (min over both axes -> letterboxed overview);
        # the world is now bigger than / a different aspect from the screen, so this must not assume
        # a width-only fit. draw() re-derives self._scale = base * zoom each frame.
        self._base_scale = min(self.cw / world.size[0], self.ch / world.size[1])
        self._scale = self._base_scale
        self.canvas = pygame.Surface((self.cw, self.ch)).convert()
        self._panel_font = pygame.font.SysFont("menlo,consolas,monospace", max(11, int(13 * self._ss)))
        self._sprite_cache = {}                       # fresh sprites for the new surface format (Pitfall 11)
        self._load_assets()
        self._ground_tile0 = self._build_ground_tile()   # seamless tile for camera-locked ground (#3)
        self._ground_tile_cache = {}
        self._vignette = self._make_vignette()           # cinematic edge-darkening (#4)
        self._motes = self._make_motes(world)

    def _world_to_canvas(self, p):
        """World point -> canvas px through the camera: nearest-image of `p` relative to the camera
        center (torus), scaled by _scale (base * zoom), centered on the canvas. With the default
        camera (center of world, zoom 1) this is exactly the old `p * _scale` fit-to-screen map."""
        w = self._wsize
        c = self._camc if self._camc is not None else (w / 2 if w is not None else np.zeros(2))
        d0 = ((float(p[0]) - c[0] + w[0] / 2) % w[0]) - w[0] / 2
        d1 = ((float(p[1]) - c[1] + w[1] / 2) % w[1]) - w[1] / 2
        return (self.cw * 0.5 + d0 * self._scale, self.ch * 0.5 + d1 * self._scale)

    def _p(self, xy):
        x, y = self._world_to_canvas(xy)
        return (int(x), int(y))

    def _line_anchored(self, color, anchor, p0, p1, width):
        """Draw a SHORT line whose two endpoints share `anchor`'s torus wrap-image -- offsets are
        added in screen px from the anchor, not re-wrapped per point. `_world_to_canvas` resolves each
        point to its OWN nearest-camera image, so for an entity near the camera's torus antipode a
        1-unit line (a snake's tongue, a vision ray) would split its endpoints a full period apart and
        streak across the whole screen -- the stray "red ray that lags" bug. Shows on the anchor's
        image only (fine for a tiny mark). Returns the two canvas points."""
        ax, ay = self._world_to_canvas(anchor)
        a = (int(ax + (p0[0] - anchor[0]) * self._scale), int(ay + (p0[1] - anchor[1]) * self._scale))
        b = (int(ax + (p1[0] - anchor[0]) * self._scale), int(ay + (p1[1] - anchor[1]) * self._scale))
        pygame.draw.line(self.canvas, color, a, b, width)
        return a, b

    # --- torus-aware primitives ---
    def _circle(self, color, pos, r):
        bx, by = self._world_to_canvas(pos)
        for ox in (0, -self._perx, self._perx):
            for oy in (0, -self._pery, self._pery):
                x, y = bx + ox, by + oy
                if -r <= x <= self.cw + r and -r <= y <= self.ch + r:
                    pygame.draw.circle(self.canvas, color, (int(x), int(y)), r)

    def _blit_world(self, surf, pos, off=(0, 0)):
        """Blit a surface centered at a world position, repeated across the torus seams."""
        w, h = surf.get_size()
        cx, cy = self._world_to_canvas(pos)
        bx = cx - w / 2 + off[0]
        by = cy - h / 2 + off[1]
        for ox in (0, -self._perx, self._perx):
            for oy in (0, -self._pery, self._pery):
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

    def _star_sprite(self, r, alpha):
        """Small yellow 5-point star for the stun 'dizzy' orbit (Phase B increment 4)."""
        r = max(2, int(r)); alpha = min(255, max(0, int(alpha)) // 16 * 16)
        key = ("star", r, alpha)
        s = self._sprite_cache.get(key)
        if s is None:
            s = pygame.Surface((2 * r, 2 * r), pygame.SRCALPHA)
            pts = [(r + np.cos(-np.pi / 2 + i * np.pi / 5) * (r if i % 2 == 0 else r * 0.42),
                    r + np.sin(-np.pi / 2 + i * np.pi / 5) * (r if i % 2 == 0 else r * 0.42))
                   for i in range(10)]
            pygame.draw.polygon(s, (255, 236, 130, alpha), pts)
            s = s.convert_alpha()
            self._sprite_cache[key] = s
        return s

    def _heart_sprite(self, r, alpha):
        """Pink heart (two lobes + a point) for courtship (Phase B increment 4)."""
        r = max(3, int(r)); alpha = min(255, max(0, int(alpha)) // 16 * 16)
        key = ("heart", r, alpha)
        s = self._sprite_cache.get(key)
        if s is None:
            d = 2 * r
            s = pygame.Surface((d, d), pygame.SRCALPHA)
            col = (240, 96, 128, alpha)
            lobe = max(2, int(r * 0.55)); cy = int(r * 0.72)
            pygame.draw.circle(s, col, (int(r - lobe * 0.62), cy), lobe)
            pygame.draw.circle(s, col, (int(r + lobe * 0.62), cy), lobe)
            pygame.draw.polygon(s, col, [(int(r - lobe * 1.15), int(cy + lobe * 0.3)),
                                         (int(r + lobe * 1.15), int(cy + lobe * 0.3)),
                                         (r, int(d * 0.92))])
            s = s.convert_alpha()
            self._sprite_cache[key] = s
        return s

    def _draw_dizzy(self, pos, hr, stun, stun_max):
        """Spinning stars circling a stunned snake's head (fades as stun counts down)."""
        frac = float(np.clip(stun / max(1, stun_max), 0.0, 1.0))
        alpha = int(110 + 130 * frac)
        star = self._star_sprite(max(3, int(hr * 0.5 * self._scale)), alpha)
        orbit = hr * 1.4 * self._scale
        above = hr * 1.7 * self._scale
        for i in range(3):
            a = self._clock * 6.0 + i * (2 * np.pi / 3)
            self._blit_world(star, pos, off=(np.cos(a) * orbit, np.sin(a) * orbit * 0.5 - above))

    def _draw_courtship(self, world):
        """Pulsing hearts between a courting pair. Reads world._mate_streak READ-ONLY (a pair holding
        mating distance has streak >= 1) -- no change to mating logic. A no-op when nobody's courting."""
        ms = getattr(world, "_mate_streak", None)
        if not ms:
            return
        live = {s.id: s for s in world.snakes if s.alive}
        hr = world.cfg.head_radius
        pulse = 0.6 + 0.4 * np.sin(self._clock * 4.0)
        heart = self._heart_sprite(max(4, int(hr * 1.1 * self._scale)), int(150 + 90 * pulse))
        for key, streak in ms.items():
            if streak < 1:
                continue
            ids = tuple(key)
            a, b = (live.get(ids[0]), live.get(ids[1])) if len(ids) == 2 else (None, None)
            if a is None or b is None:
                continue
            mid = wrap(a.head_uw + torus_delta(b.head_uw, a.head_uw, world.size) / 2, world.size)
            self._blit_world(heart, mid, off=(0, -hr * (0.8 + 0.6 * pulse) * self._scale))

    def _anim_frame(self, nframes, fps, phase=0.0):
        """Which sheet frame to show now: wall-clock time * fps, +phase (a per-entity offset so
        entities don't animate in lockstep), wrapped to [0, nframes). Smooth at any sim speed."""
        return int(self._clock * fps + phase) % nframes

    def _make_vignette(self):
        """Subtle cinematic edge-darkening, built once per canvas size (polish #4): a radial alpha ramp
        (transparent center -> ~110/255 at the corners) that focuses the eye toward the followed snake.
        One cached full-canvas alpha blit per frame -- cheap and self-contained."""
        xx, yy = np.mgrid[0:self.cw, 0:self.ch].astype(np.float32)   # [x, y]
        cx, cy = self.cw * 0.5, self.ch * 0.5
        r = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)   # 0 center -> ~1.41 corner
        a = (np.clip((r - 0.6) / 0.8, 0.0, 1.0) ** 2 * 110).astype(np.uint8)   # ease in past ~60% radius
        v = pygame.Surface((self.cw, self.ch), pygame.SRCALPHA)
        v.fill((0, 0, 0, 0))
        pa = pygame.surfarray.pixels_alpha(v)
        pa[:] = a
        del pa                                                       # unlock before convert/blit
        return v.convert_alpha()

    def _make_motes(self, world):
        rng = np.random.default_rng(12345)
        n = 60
        pos = rng.uniform([0, 0], world.size, size=(n, 2))
        vel = rng.uniform(-0.05, 0.05, size=(n, 2))
        return np.hstack([pos, vel])

    def _draw_ground(self):
        """Draw the ground through the SAME world transform entities use, so it pans + zooms in exact
        lockstep with no jerk. `_ground_tile0` is a seamless procedural tile spanning GROUND_TILE_WORLD
        world-units; each frame it's scaled to that span in canvas px (cached PER integer px size, so a
        smooth zoom-ease reuses recent sizes instead of thrashing a single bucket) and blitted on a flat
        world-locked grid across the visible region with a 1-tile margin and a +1px overlap -> full
        coverage, no seams, no integer-period modulo discontinuity (the old jerk). Grass is uniform, so
        a flat (non-torus) wallpaper reads fine."""
        scale = self._scale
        tw = GROUND_TILE_WORLD
        tps = max(8, int(round(tw * scale)) + 1)               # +1px overlap so adjacent tiles never gap
        t = self._ground_tile_cache.get(tps)
        if t is None:
            t = pygame.transform.smoothscale(self._ground_tile0, (tps, tps))
            if len(self._ground_tile_cache) > 12:              # bound the cache during zoom sweeps
                self._ground_tile_cache = {}
            self._ground_tile_cache[tps] = t
        cx, cy = float(self._camc[0]), float(self._camc[1])
        hw, hh = self.cw / (2 * scale), self.ch / (2 * scale)  # half-viewport in world units
        i0, i1 = int(np.floor((cx - hw) / tw)) - 1, int(np.floor((cx + hw) / tw)) + 1
        j0, j1 = int(np.floor((cy - hh) / tw)) - 1, int(np.floor((cy + hh) / tw)) + 1
        for i in range(i0, i1 + 1):
            bx = int(np.floor(self.cw * 0.5 + (i * tw - cx) * scale))
            for j in range(j0, j1 + 1):
                by = int(np.floor(self.ch * 0.5 + (j * tw - cy) * scale))
                self.canvas.blit(t, (bx, by))

    def _build_ground_tile(self):
        """A RICH, seamless procedural ground tile (Phase B polish #3) -- replaces the crude
        variant-patchwork ('0/1'-looking) art. Layered integer-frequency trig noise (perfectly tileable
        over the tile) mapped through a 3-stop organic green ramp (deep shade -> grass -> sun-touched)
        with a faint high-frequency grain, so the backdrop reads as soft, multi-tone turf that tiles
        without any seam. No assets, no RNG-per-frame: built once, deterministic."""
        T = 256
        yy, xx = (np.mgrid[0:T, 0:T].astype(np.float32)) * (2 * np.pi / T)   # 0..2pi, periodic in T
        rng = np.random.default_rng(20240723)
        field = np.zeros((T, T), np.float32)
        for freq, amp in ((1, 1.0), (2, 0.6), (3, 0.42), (5, 0.26), (8, 0.16), (13, 0.1)):
            a1, a2, a3, a4 = rng.uniform(0, 2 * np.pi, 4)
            field += amp * (np.sin(freq * xx + a1) * np.cos(freq * yy + a2)
                            + 0.6 * np.cos(freq * (xx + yy) + a3) + 0.6 * np.sin(freq * (xx - yy) + a4))
        field = (field - field.min()) / (np.ptp(field) + 1e-6)              # -> 0..1, seamless
        c0 = np.array([36, 56, 34], np.float32)     # deep shade / moss
        c1 = np.array([60, 92, 52], np.float32)     # base grass
        c2 = np.array([106, 144, 84], np.float32)   # sun-touched blades
        t = field[..., None]
        rgb = np.where(t < 0.5, c0 + (c1 - c0) * (t / 0.5), c1 + (c2 - c1) * ((t - 0.5) / 0.5))
        grain = 0.05 * np.sin(37.0 * xx) * np.sin(41.0 * yy)                 # tileable fine grain
        rgb = np.clip(rgb * (1.0 + grain[..., None]), 0, 255).astype(np.uint8)
        return pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2))).convert()

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
        owners = e.get("owner")
        pulse = 0.94 + 0.06 * np.sin(self._clock * 2.2)              # gentle clock-driven breathing
        glow_a = int(60 + 40 * (0.5 + 0.5 * np.sin(self._clock * 3.0)))
        for i, (pos, timer) in enumerate(zip(e["pos"], e["timer"])):
            p = wrap(pos, world.size)
            r = c.egg_radius * pulse
            rp = max(2, int(r * self._scale))
            guarded = owners is not None and i < len(owners) and owners[i][0] >= 0   # repro egg (owner>=0)
            if guarded:                                             # arrival eggs (owner -1) get no glow
                self._blit_world(self._dot(max(3, int(r * 2.1 * self._scale)), (240, 210, 130), glow_a), p)
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
        from .genome import SIZE
        n = len(body_uw)
        sz = 0.85 + 0.30 * float(snake.genome[SIZE])       # size gene -> chunkier drawn body (visual only)
        hr = world.cfg.head_radius * sz
        head_c, tail_c, head_tint = _snake_palette(snake)  # lineage family hue + gene-driven sat/value
        wpts = [wrap(body_uw[k], world.size) for k in range(n)]
        radii = [hr * (1 - k / max(1, n - 1)) + world.cfg.body_radius * sz * 0.7 * (k / max(1, n - 1)) for k in range(n)]
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
                                   tint=head_tint)
        if head_sprite is not None:
            self._blit_world(head_sprite, head)
        flick = snake.dashed or (self._t % 48 < 6)         # forked tongue tasting the air
        if flick:
            reach = 1.5 if snake.dashed else 1.2
            tip = head + d * hr * reach; base = head + d * hr * 0.9
            self._line_anchored(TONGUE, head, base, tip, max(2, int(0.14 * self._scale)))
            for s2 in (1, -1):
                self._line_anchored(TONGUE, head, tip, tip + (d * 0.45 + perp * 0.45 * s2) * hr,
                                    max(2, int(0.12 * self._scale)))
        if head_sprite is None:                            # procedural eyes only when no head sprite
            for s2 in (1, -1):
                e = head + d * hr * 0.32 + perp * hr * 0.46 * s2
                pygame.draw.circle(self.canvas, SNAKE_EYE, self._p(e), max(2, int(hr * 0.3 * self._scale)))
                pygame.draw.circle(self.canvas, PUPIL, self._p(e + d * hr * 0.13), max(1, int(hr * 0.15 * self._scale)))
        if snake.stun > 0:                                 # dizzy: stars spin over a stunned head
            self._draw_dizzy(head, hr, snake.stun, world.cfg.stun_steps)

    def _ring_hud(self, world, snake, head_pos, big=False):
        """Per-snake 3-ring badge above the head: outer=energy(green), middle=stamina(cyan),
        inner=length(amber); a thin outline in the snake's hue groups the badge to its owner.
        `head_pos` is the (interpolated) wrapped head the body was drawn at, so it doesn't jitter."""
        from .genome import SIZE
        c = world.cfg
        hr = c.head_radius * (0.85 + 0.30 * float(snake.genome[SIZE]))   # match _draw_snake's size-gene scale
        r_out = hr * (2.4 if big else 1.6)
        step = r_out * 0.3
        center = head_pos + np.array([0.0, -(r_out + hr * 1.3)])
        p = self._p(center)
        lw = max(2, int((0.22 if big else 0.15) * self._scale))
        outline_r = max(3, int((r_out + step * 0.6) * self._scale))
        pygame.draw.circle(self.canvas, color_for(snake.lineage), p, outline_r, 1)   # family-color badge outline
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

    def _panel_bar(self, surf, x, y, labelw, barw, label, frac, color):
        f = self._panel_font
        surf.blit(f.render(label, True, (200, 206, 214)), (x, y))
        bx = x + labelw
        bh = f.get_height()
        frac = float(np.clip(frac, 0.0, 1.0))
        pygame.draw.rect(surf, (44, 50, 60), (bx, y, barw, bh))
        pygame.draw.rect(surf, color, (bx, y, int(barw * frac), bh))
        surf.blit(f.render(f"{frac:.2f}", True, (230, 234, 240)), (bx + barw + int(6 * self._ss), y))

    def _draw_inspector(self, world, snake, stats, fallen=False):
        """Genome inspector overlay for the FOLLOWED snake (toggle key I): a semi-transparent corner
        panel with the family swatch + lineage id, sex, an age/lifespan bar, the 9 gene bars (0..1),
        and lightweight life stats (kills/offspring, tracked in watch from step's deaths/hatches).
        `fallen`: the followed snake just died -- keep its LAST stats up during the death-linger
        (tagged) instead of blanking the panel."""
        f = self._panel_font
        s = self._ss
        lh = f.get_height() + int(4 * s)
        pad = int(10 * s)
        labelw = int(78 * s); barw = int(120 * s)
        width = pad * 2 + labelw + barw + int(52 * s)
        rows = len(GENE_LABELS) + 3                       # header, age bar, 9 genes, stats line
        height = pad * 2 + lh * rows
        panel = pygame.Surface((width, height), pygame.SRCALPHA)
        panel.fill((14, 18, 24, 214))
        pygame.draw.rect(panel, (60, 70, 86, 235), panel.get_rect(), max(1, s))
        y = pad
        fam = color_for(snake.lineage)                    # family color swatch + lineage id + sex glyph
        sw = int(14 * s)
        pygame.draw.rect(panel, fam, (pad, y + int(1 * s), sw, sw))
        sexg = "♀" if int(snake.sex) == 0 else "♂"   # female / male
        header = f"line {int(snake.lineage)}   {sexg}" + ("   (fallen)" if fallen else "")
        panel.blit(f.render(header, True, (228, 150, 150) if fallen else (236, 240, 246)),
                   (pad + sw + int(8 * s), y))
        y += lh
        agefrac = snake.age / max(1.0, float(snake.max_lifespan))
        self._panel_bar(panel, pad, y, labelw, barw, "age", agefrac, (206, 180, 118)); y += lh
        g = np.asarray(snake.genome, float)
        for i, lab in enumerate(GENE_LABELS):
            self._panel_bar(panel, pad, y, labelw, barw, lab, g[i] if i < len(g) else 0.0,
                            (120, 190, 150)); y += lh
        st = (stats or {}).get(snake.id, {})       # kills~ is the nearest-rival APPROXIMATION (see watch)
        panel.blit(f.render(f"kills~{st.get('kills', 0)}   offspring {st.get('offspring', 0)}",
                            True, (240, 212, 160)), (pad, y))
        self.canvas.blit(panel.convert_alpha(), (int(14 * s), int(14 * s)))

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
        hx, hy = self._world_to_canvas(head)
        for u, dd, kd in zip(dirs, dist, kinds):
            col = RAY_KIND[int(kd)]
            end = head + u * dd                                # a straight ray: one anchored line, no seam-split
            _, b = self._line_anchored(col, head, head, end, max(1, SS))
            if kd != -1:
                pygame.draw.circle(self.canvas, col, b, max(2, SS + 1))

    def draw(self, world, bodies=None, chick_pos=None, chick_dir=None, follow_id=None,
             cam_center=None, zoom=1.0, inspector_stats=None, fallen=None):
        """Composite the scene in depth order: ground -> obstacles -> blood decals -> corpses ->
        eggs -> chickens -> sky-dropping chickens -> snakes -> gore particles -> ambient -> HUD/sensors.
        `bodies`: optional {snake_id: interpolated unwrapped body polyline}. `follow_id`: which
        snake gets the larger ring HUD + sensor overlay (defaults to slot-0). `cam_center` (world
        units, None -> world center) + `zoom` (>=1 zooms in) drive the camera transform; the default
        reproduces the old whole-world fit-to-screen."""
        self._t += 1
        self._clock = self._clock_override if self._clock_override is not None \
            else pygame.time.get_ticks() / 1000.0     # wall-clock anim time (smooth, sim-independent)
        self._ensure(world)
        self._wsize = np.asarray(world.size, float)
        self.zoom = float(np.clip(zoom, ZOOM_MIN, ZOOM_MAX))
        self._scale = self._base_scale * self.zoom
        self._camc = np.asarray(cam_center, float) if cam_center is not None else self._wsize / 2
        self._perx = float(world.size[0]) * self._scale
        self._pery = float(world.size[1]) * self._scale
        if chick_pos is None:
            chick_pos, chick_dir = world.chicken_pos, world.chicken_dir
        self._draw_ground()
        self._draw_obstacles(world)
        self._draw_decals()
        self._draw_corpses(world)
        self._draw_eggs(world)
        self._draw_transient(world)
        self._draw_chickens(world, chick_pos, chick_dir)
        self._draw_arrivals(world)                                   # sky-dropping chickens (Goal 2)
        follow = follow_id if follow_id is not None else (world.snakes[0].id if world.snakes else -1)
        sensor_snake = None
        follow_snake = None
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
                follow_snake = s
        self._draw_courtship(world)                                  # hearts between courting pairs
        self._draw_particles()
        self._draw_motes(world)
        if self._vignette is not None:                               # cinematic edge-darkening (#4)
            self.canvas.blit(self._vignette, (0, 0))
        if self.show_sensors and sensor_snake is not None:
            s, b = sensor_snake
            d = b[0] - b[1] if len(b) > 1 else s.heading_vec()
            nrm = np.linalg.norm(d); d = d / nrm if nrm > 1e-6 else s.heading_vec()
            self._draw_sensors(world, b[0], float(np.arctan2(d[1], d[0])), s)
        if self.show_inspector:
            if follow_snake is not None:
                self._draw_inspector(world, follow_snake, inspector_stats)
            elif fallen is not None:                          # death-linger: keep the fallen snake's panel
                self._draw_inspector(world, fallen, inspector_stats, fallen=True)
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

    def toggle_inspector(self):
        self.show_inspector = not self.show_inspector

    def zoom_for_span(self, span, default=1.0):
        """The zoom that frames `span` world-units across the SHORTER canvas axis (action-cam
        framing). Returns `default` until the display is set up (cw unknown before the first draw)."""
        if not self.cw or not self._base_scale or span <= 0:
            return default
        return float(np.clip(min(self.cw, self.ch) / (self._base_scale * span), ZOOM_MIN, ZOOM_MAX))

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
