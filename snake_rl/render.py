"""pygame renderer: supersampled AA, torus-aware, shadows/vignette, continuous tapering snake."""
import colorsys
import numpy as np
import pygame
from .sensors import vision_distances
from .world import wrap

SS = 2                   # supersample factor (draw big, smoothscale down for anti-aliasing)
TARGET_PX = 860          # display fits ~this many pixels on its long side

BG = (15, 17, 24); GRID = (24, 28, 40)
ROCK = (110, 112, 122); ROCK_HI = (156, 158, 168); ROCK_SH = (60, 62, 72)
TRUNK = (98, 68, 44); LEAF = (54, 132, 72); LEAF2 = (84, 174, 104)
CHICK = (247, 243, 231); CHICK_SH = (206, 200, 184); BEAK = (242, 168, 62); EYE = (26, 26, 30); COMB = (226, 74, 74)
SNAKE_RIM = (16, 40, 34)
SNAKE_GLOSS = (196, 252, 214); SNAKE_EYE = (250, 250, 250); PUPIL = (16, 20, 18)
TONGUE = (236, 74, 96)
EGG_SHELL = (236, 224, 196); EGG_SHADE = (206, 188, 150); EGG_TEXT = (30, 26, 20)
CORPSE = (92, 74, 58)
RING_TRACK = (44, 48, 58)
# ring HUD: each ring's color encodes WHICH stat (not the snake's identity color, or all 3 rings
# look the same) -- outer energy=green, middle stamina=cyan, inner length=amber.
RING_COLORS = ((96, 214, 120), (86, 200, 232), (240, 186, 72))
RAY_NONE = (70, 92, 120); RAY_OBST = (226, 96, 84); RAY_CHICK = (240, 208, 96); RAY_SELF = (168, 128, 224)
RAY_OTHER = (110, 200, 240); RAY_EGG = (240, 176, 224)          # B2 ray kinds 3=other_body, 4=egg
RAY_CORPSE = (176, 132, 84)                                     # ray kind 5=corpse
RAY_KIND = {-1: RAY_NONE, 0: RAY_OBST, 1: RAY_CHICK, 2: RAY_SELF, 3: RAY_OTHER, 4: RAY_EGG, 5: RAY_CORPSE}


def color_for(seed, s=0.55, v=0.92):
    """Golden-angle hue palette: deterministic and visually distinct per integer seed,
    so a snake keeps a stable, distinguishable color across frames (snake.color_seed)."""
    hue = (seed * 137.5) % 360
    r, g, b = colorsys.hsv_to_rgb(hue / 360.0, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def _snake_colors(seed):
    """Head/tail gradient endpoints derived from the snake's hue (mirrors the old fixed-green
    SNAKE_HEAD/SNAKE_TAIL look, just re-tinted per snake)."""
    return color_for(seed, s=0.42, v=0.96), color_for(seed, s=0.80, v=0.42)


def _lerp(a, b, t):
    return (int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t), int(a[2] + (b[2] - a[2]) * t))


class Renderer:
    def __init__(self, scale=None, show_sensors=True, fullscreen=False, screen_size=None):
        pygame.init()
        self.scale = scale
        self.show_sensors = show_sensors
        self.fullscreen = fullscreen
        self.screen_size = screen_size
        self.canvas = self.display = None
        self.cw = self.ch = self.dw = self.dh = 0
        self._scale = 1
        self._ss = SS
        self._world_key = None
        self._t = 0                # frame counter for idle animation (periodic tongue flicks)
        self._flashes = []         # transient catch effects: [wrapped_pos, age]
        self.font = pygame.font.SysFont("menlo,consolas,monospace", 15)
        self._sprite_cache = {}
        self._vignette = None

    def _ensure(self, world):
        key = (round(float(world.size[0]), 3), round(float(world.size[1]), 3))
        if self.display is not None and self._world_key == key:
            return
        self._world_key = key
        fs = bool(self.fullscreen and self.screen_size)
        if fs:
            # FULLSCREEN|SCALED renders at the requested logical size and lets SDL scale to the panel
            # (handles Retina / mode-snap; without SCALED a real mode switch can clip or black-margin).
            self.display = pygame.display.set_mode(self.screen_size, pygame.FULLSCREEN | pygame.SCALED, vsync=1)
        else:
            base = self.scale or max(4, int(TARGET_PX / max(world.size)))
            self.display = pygame.display.set_mode((int(world.size[0] * base), int(world.size[1] * base)))
        pygame.display.set_caption("Snake-RL")
        pygame.mouse.set_visible(not fs)
        dw, dh = self.display.get_size()              # trust what SDL actually granted
        self.dw, self.dh = dw, dh
        self._ss = 2 if max(dw, dh) <= 1100 else 1    # supersample small windows; native res is crisp enough
        self.cw, self.ch = dw * self._ss, dh * self._ss
        self._scale = self.cw / world.size[0]         # world units -> canvas px (aspect matches the display)
        self.canvas = pygame.Surface((self.cw, self.ch)).convert()   # match display format (perf + no artifacts)
        self._sprite_cache = {}                       # fresh sprites for the new surface format
        self._vignette = self._make_vignette(dw, dh)

    def _p(self, xy):
        return (int(xy[0] * self._scale), int(xy[1] * self._scale))

    def _circle(self, color, pos, r):
        bx, by = int(pos[0] * self._scale), int(pos[1] * self._scale)
        for ox in (0, -self.cw, self.cw):
            for oy in (0, -self.ch, self.ch):
                if -r <= bx + ox <= self.cw + r and -r <= by + oy <= self.ch + r:
                    pygame.draw.circle(self.canvas, color, (bx + ox, by + oy), r)

    def _radial_sprite(self, r, color, peak):
        key = (r, color, peak)
        surf = self._sprite_cache.get(key)
        if surf is None:
            surf = pygame.Surface((2 * r, 2 * r), pygame.SRCALPHA)
            for rr in range(r, 0, -1):
                a = int(peak * (1 - rr / r) ** 2)
                pygame.draw.circle(surf, (*color, a), (r, r), rr)
            surf = surf.convert_alpha()               # match display format (avoids black-square artifacts)
            self._sprite_cache[key] = surf
        return surf

    def _blit_sprite(self, surf, pos, off=(0, 0)):
        p = self._p(pos)
        self.canvas.blit(surf, (p[0] - surf.get_width() // 2 + off[0], p[1] - surf.get_height() // 2 + off[1]))

    def _shadow(self, pos, r_px):
        self._blit_sprite(self._radial_sprite(int(r_px * 1.25), (0, 0, 0), 120), pos, off=(0, int(r_px * 0.35)))

    def _make_vignette(self, w, h):
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        cx, cy = w / 2, h / 2
        maxd = np.hypot(cx, cy)
        rings = 48
        for i in range(rings):
            t = i / rings
            a = int(120 * t ** 2.4)
            pygame.draw.ellipse(surf, (0, 0, 0, a), (t * cx, t * cy, w - 2 * t * cx, h - 2 * t * cy), max(2, int(maxd / rings) + 2))
        return surf.convert_alpha()

    # --- scene ---
    def _bg(self):
        self.canvas.fill(BG)
        step = int(10 * self._scale)
        for x in range(0, self.cw, step):
            pygame.draw.line(self.canvas, GRID, (x, 0), (x, self.ch))
        for y in range(0, self.ch, step):
            pygame.draw.line(self.canvas, GRID, (0, y), (self.cw, y))

    def _draw_obstacles(self, world):
        for pos, r, kind in zip(world.obstacle_pos, world.obstacle_r, world.obstacle_kind):
            rp = int(r * self._scale)
            self._shadow(pos, rp)
            if kind == 1:                                  # tree
                self._circle(TRUNK, pos, max(2, rp // 3))
                self._circle(LEAF, pos, rp)
                self._circle(LEAF2, pos + np.array([-r * 0.25, -r * 0.3]), int(rp * 0.66))
            else:                                          # rock
                self._circle(ROCK_SH, pos, rp)
                self._circle(ROCK, pos, int(rp * 0.9))
                self._circle(ROCK_HI, pos + np.array([-r * 0.3, -r * 0.32]), int(rp * 0.38))

    def _draw_chickens(self, world, positions, dirs):
        cr = world.cfg.chicken_radius
        rp = max(4, int(cr * self._scale))
        for cpos, cdir in zip(positions, dirs):
            self._shadow(cpos, rp)
            for dx in (-0.3, 0.0, 0.3):                # little red comb on top
                pygame.draw.circle(self.canvas, COMB, self._p(cpos + np.array([dx * cr, -cr * 0.7])), max(2, int(cr * 0.24 * self._scale)))
            self._circle(CHICK_SH, cpos + np.array([0, cr * 0.28]), rp)
            self._circle(CHICK, cpos, int(cr * 0.9 * self._scale))
            d = np.array([np.cos(cdir), np.sin(cdir)]); perp = np.array([-d[1], d[0]])
            pygame.draw.circle(self.canvas, BEAK, self._p(cpos + d * cr * 0.95), max(2, int(cr * 0.32 * self._scale)))
            pygame.draw.circle(self.canvas, EYE, self._p(cpos + d * cr * 0.3 + perp * cr * 0.35), max(1, int(cr * 0.2 * self._scale)))

    def add_flash(self, pos):
        """Register a catch effect (an expanding ring) at a wrapped world position."""
        self._flashes.append([np.asarray(pos, float).copy(), 0])

    def _draw_flashes(self):
        alive = []
        for pos, age in self._flashes:
            t = age / 15.0
            col = _lerp((255, 244, 180), BG, t)
            r = int((1.4 + age * 0.6) * self._scale)
            p = self._p(pos)
            pygame.draw.circle(self.canvas, col, p, r, max(1, int(0.16 * self._scale)))
            if age < 15:
                alive.append([pos, age + 1])
        self._flashes = alive

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
        flick = snake.dashed or (self._t % 48 < 6)         # tongue out on a dash, and a periodic idle flick
        if flick:                                          # forked tongue tasting the air
            reach = 1.35 if snake.dashed else 1.15
            tip = head + d * hr * reach; base = head + d * hr * 0.9
            pygame.draw.line(self.canvas, TONGUE, self._p(base), self._p(tip), max(2, int(0.14 * self._scale)))
            for s2 in (1, -1):
                pygame.draw.line(self.canvas, TONGUE, self._p(tip), self._p(tip + (d * 0.45 + perp * 0.45 * s2) * hr), max(2, int(0.12 * self._scale)))
        for s2 in (1, -1):                                 # eyes
            e = head + d * hr * 0.32 + perp * hr * 0.46 * s2
            pygame.draw.circle(self.canvas, SNAKE_EYE, self._p(e), max(2, int(hr * 0.3 * self._scale)))
            pygame.draw.circle(self.canvas, PUPIL, self._p(e + d * hr * 0.13), max(1, int(hr * 0.15 * self._scale)))

    def _ring_hud(self, world, snake, head_pos, big=False):
        """Per-snake 3-ring concentric badge floating above the head: outer=energy(green),
        middle=stamina(cyan), inner=length->length_cap(amber) -- each ring's OWN fixed color
        encodes which stat it is (not the snake's identity color, or all 3 rings look the same);
        a thin outline in the snake's hue groups the badge to its owner. The followed/ego snake
        gets a larger version -- replaces the old single-snake bar HUD (doesn't scale to 6).
        `head_pos` is the (possibly interpolated) wrapped head position the body was just drawn
        at -- using it here, not `snake.head` (the raw un-interpolated stepped position), is what
        keeps the badge glued to the head instead of jittering a step behind it."""
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

    def _draw_eggs(self, world):
        e = world.eggs
        if not len(e["pos"]):
            return
        c = world.cfg
        pulse = 0.85 + 0.15 * np.sin(self._t * 0.15)
        for pos, timer in zip(e["pos"], e["timer"]):
            p = wrap(pos, world.size)
            r = c.egg_radius * pulse
            rp = max(2, int(r * self._scale))
            self._shadow(p, rp)
            self._circle(EGG_SHELL, p, rp)
            self._circle(EGG_SHADE, p + np.array([-r * 0.28, -r * 0.28]), max(1, int(rp * 0.45)))
            label = self.font.render(str(max(0, int(timer))), True, EGG_TEXT)
            lp = self._p(p)
            self.canvas.blit(label, (lp[0] - label.get_width() // 2, lp[1] - label.get_height() // 2))

    def _draw_corpses(self, world):
        corpses = world.corpses
        if not len(corpses["pos"]):
            return
        cfg = world.cfg
        for pos, food in zip(corpses["pos"], corpses["food"]):
            p = wrap(pos, world.size)
            n_piles = max(1, min(4, int(food / max(1.0, cfg.corpse_food_per_length * 2)) + 1))
            r = max(0.6, min(2.5, food / (cfg.corpse_food_per_length * 6)))
            rp = max(2, int(r * self._scale))
            self._shadow(p, rp)
            offs = [(0, 0), (-0.5, 0.35), (0.5, 0.3), (-0.15, -0.4)][:n_piles]
            for ox, oy in offs:
                self._circle(CORPSE, p + np.array([ox, oy]) * r, max(1, int(rp * 0.6)))

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
        """Draw every live snake in `world.snakes` (golden-angle colors + ring HUD), eggs, corpses.
        `bodies`: optional {snake_id: unwrapped body polyline} (e.g. interpolated) -- falls back to
        the world's current path for any snake not in the dict. `follow_id`: which snake gets the
        larger ring HUD and the sensor overlay (defaults to slot-0; None picks slot-0 too -- an
        explicit "no one alive" overview is the caller's job, drawing just skips dead snakes)."""
        self._t += 1
        self._ensure(world)
        if chick_pos is None:
            chick_pos, chick_dir = world.chicken_pos, world.chicken_dir
        self._bg()
        self._draw_obstacles(world)
        self._draw_chickens(world, chick_pos, chick_dir)
        self._draw_eggs(world)
        self._draw_corpses(world)
        self._draw_flashes()
        follow = follow_id if follow_id is not None else world.snakes[0].id
        sensor_snake = None
        for s in world.snakes:
            if not s.alive:
                continue
            big = (s.id == follow)
            b = (bodies or {}).get(s.id)
            if b is None:
                b = world._body_render_path_uw(s)
            self._draw_snake(world, s, b, big=big)
            self._ring_hud(world, s, wrap(b[0], world.size), big=big)
            if big:
                sensor_snake = (s, b)
        if self.show_sensors and sensor_snake is not None:
            s, b = sensor_snake
            d = b[0] - b[1] if len(b) > 1 else s.heading_vec()
            nrm = np.linalg.norm(d); d = d / nrm if nrm > 1e-6 else s.heading_vec()
            self._draw_sensors(world, b[0], float(np.arctan2(d[1], d[0])), s)
        # Blit into the window via a temp surface (writing straight into the window surface
        # renders black on some backends). At SS=1 the canvas is already display-sized.
        if self._ss == 1:
            self.display.blit(self.canvas, (0, 0))
        else:
            self.display.blit(pygame.transform.smoothscale(self.canvas, (self.dw, self.dh)), (0, 0))
        self.display.blit(self._vignette, (0, 0))
        pygame.display.flip()

    def toggle_sensors(self):
        self.show_sensors = not self.show_sensors

    def close(self):
        pygame.quit()
