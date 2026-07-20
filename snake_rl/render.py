"""pygame renderer: supersampled AA, torus-aware, shadows/vignette, continuous tapering snake."""
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
SNAKE_HEAD = (128, 224, 158); SNAKE_TAIL = (30, 100, 86); SNAKE_RIM = (16, 40, 34)
SNAKE_GLOSS = (196, 252, 214); SNAKE_EYE = (250, 250, 250); PUPIL = (16, 20, 18)
TONGUE = (236, 74, 96)
RAY_NONE = (70, 92, 120); RAY_OBST = (226, 96, 84); RAY_CHICK = (240, 208, 96); RAY_SELF = (168, 128, 224)
RAY_KIND = {-1: RAY_NONE, 0: RAY_OBST, 1: RAY_CHICK, 2: RAY_SELF}


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

    def _draw_snake(self, world, body_uw):
        n = len(body_uw)
        hr = world.cfg.head_radius
        wpts = [wrap(body_uw[k], world.size) for k in range(n)]
        radii = [hr * (1 - k / max(1, n - 1)) + world.cfg.body_radius * 0.7 * (k / max(1, n - 1)) for k in range(n)]
        for k in range(n - 1, -1, -1):                     # rim pass
            self._circle(SNAKE_RIM, wpts[k], max(3, int((radii[k] + 0.28) * self._scale)))
        for k in range(n - 1, -1, -1):                     # body pass
            self._circle(_lerp(SNAKE_HEAD, SNAKE_TAIL, k / max(1, n - 1)), wpts[k], max(2, int(radii[k] * self._scale)))
        for k in range(min(n, n * 3 // 4) - 1, -1, -1):    # gloss highlight (front ~3/4)
            self._circle(SNAKE_GLOSS, wpts[k] - np.array([hr * 0.18, hr * 0.22]), max(1, int(radii[k] * 0.34 * self._scale)))
        head = wpts[0]
        d = body_uw[0] - body_uw[1] if n > 1 else world.heading_vec()
        nrm = np.linalg.norm(d); d = d / nrm if nrm > 1e-6 else world.heading_vec()
        perp = np.array([-d[1], d[0]])
        if world.dashed:                                   # forked tongue on a dash
            tip = head + d * hr * 1.35; base = head + d * hr * 0.9
            pygame.draw.line(self.canvas, TONGUE, self._p(base), self._p(tip), max(2, int(0.14 * self._scale)))
            for s in (1, -1):
                pygame.draw.line(self.canvas, TONGUE, self._p(tip), self._p(tip + (d * 0.45 + perp * 0.45 * s) * hr), max(2, int(0.12 * self._scale)))
        for s in (1, -1):                                  # eyes
            e = head + d * hr * 0.32 + perp * hr * 0.46 * s
            pygame.draw.circle(self.canvas, SNAKE_EYE, self._p(e), max(2, int(hr * 0.3 * self._scale)))
            pygame.draw.circle(self.canvas, PUPIL, self._p(e + d * hr * 0.13), max(1, int(hr * 0.15 * self._scale)))

    def _draw_sensors(self, world, head_uw, heading):
        head = wrap(head_uw, world.size)
        dirs, dist, kinds = vision_distances(world, head, heading)
        for u, dd, kd in zip(dirs, dist, kinds):
            nseg = max(2, int(dd))
            pts = [np.asarray(self._p(wrap(head + u * (dd * k / nseg), world.size))) for k in range(nseg + 1)]
            col = RAY_KIND[int(kd)]
            for a, b in zip(pts, pts[1:]):
                if abs(a[0] - b[0]) < self.cw / 2 and abs(a[1] - b[1]) < self.ch / 2:
                    pygame.draw.line(self.canvas, col, a, b, max(1, SS))
            if kd != -1:
                pygame.draw.circle(self.canvas, col, tuple(pts[-1]), max(2, SS + 1))

    def _hud(self, world):
        txt = f"length {world.target_length:.0f}    stamina {world.stamina:.0f}/{world.cfg.s_max:.0f}    energy {world.energy:.0f}"
        label = self.font.render(txt, True, (228, 231, 238))
        pad = 8
        panel = pygame.Surface((label.get_width() + 2 * pad, label.get_height() + 2 * pad), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 110))
        self.display.blit(panel.convert_alpha(), (8, 8)); self.display.blit(label, (8 + pad, 8 + pad))

    def draw(self, world, body_uw=None, chick_pos=None, chick_dir=None):
        self._ensure(world)
        if body_uw is None:
            body_uw = world.body_render_path_uw()
        if chick_pos is None:
            chick_pos, chick_dir = world.chicken_pos, world.chicken_dir
        self._bg()
        self._draw_obstacles(world)
        self._draw_chickens(world, chick_pos, chick_dir)
        self._draw_snake(world, body_uw)
        if self.show_sensors:
            d = body_uw[0] - body_uw[1] if len(body_uw) > 1 else world.heading_vec()
            self._draw_sensors(world, body_uw[0], float(np.arctan2(d[1], d[0])))
        # Blit into the window via a temp surface (writing straight into the window surface
        # renders black on some backends). At SS=1 the canvas is already display-sized.
        if self._ss == 1:
            self.display.blit(self.canvas, (0, 0))
        else:
            self.display.blit(pygame.transform.smoothscale(self.canvas, (self.dw, self.dh)), (0, 0))
        self.display.blit(self._vignette, (0, 0))
        self._hud(world)
        pygame.display.flip()

    def toggle_sensors(self):
        self.show_sensors = not self.show_sensors

    def close(self):
        pygame.quit()
