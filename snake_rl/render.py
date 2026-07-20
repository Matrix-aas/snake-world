"""pygame renderer: torus-aware, prettier world + continuous tapering snake + optional sensors."""
import numpy as np
import pygame
from .sensors import ray_dirs, sense_vision
from .world import wrap

BG = (16, 18, 26); GRID = (26, 30, 42)
ROCK = (108, 110, 120); ROCK_HI = (150, 152, 162); ROCK_SH = (64, 66, 76)
TRUNK = (96, 66, 42); LEAF = (56, 136, 74); LEAF2 = (80, 168, 98)
CHICK = (246, 242, 230); CHICK_SH = (208, 202, 186); BEAK = (240, 166, 60); EYE = (24, 24,28)
SNAKE_HEAD = (122, 214, 150); SNAKE_TAIL = (36, 108, 92); SNAKE_EYE = (250, 250, 250); PUPIL = (18, 22, 20)
TONGUE = (232, 72, 92); RAY = (90, 120, 150); GLOW = (250, 214, 120)
TARGET_PX = 840          # window fits ~this many pixels on its long side


def _lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


class Renderer:
    def __init__(self, scale=None, show_sensors=True):
        pygame.init()
        self.scale = scale            # None -> adaptive per world size
        self.show_sensors = show_sensors
        self.screen = None
        self.wpx = self.hpx = 0
        self.font = pygame.font.SysFont("menlo,consolas,monospace", 14)
        self._glow_cache = {}

    def _ensure(self, world):
        scale = self.scale or max(4, int(TARGET_PX / max(world.size)))
        self._scale = scale
        w = int(world.size[0] * scale); h = int(world.size[1] * scale)
        if self.screen is None or (self.wpx, self.hpx) != (w, h):
            self.wpx, self.hpx = w, h
            self.screen = pygame.display.set_mode((w, h))

    def _p(self, xy):
        return (int(xy[0] * self._scale), int(xy[1] * self._scale))

    def _circle(self, color, pos, radius_px):
        """Draw a circle at its wrapped position AND the 8 neighbor images (shows across the seam)."""
        bx = int(pos[0] * self._scale); by = int(pos[1] * self._scale)
        for ox in (0, -self.wpx, self.wpx):
            for oy in (0, -self.hpx, self.hpx):
                if -radius_px <= bx + ox <= self.wpx + radius_px and -radius_px <= by + oy <= self.hpx + radius_px:
                    pygame.draw.circle(self.screen, color, (bx + ox, by + oy), radius_px)

    def _glow(self, pos, radius_px, color):
        key = (radius_px, color)
        surf = self._glow_cache.get(key)
        if surf is None:
            surf = pygame.Surface((2 * radius_px, 2 * radius_px), pygame.SRCALPHA)
            for rr in range(radius_px, 0, -1):
                a = int(55 * (1 - rr / radius_px) ** 2)
                pygame.draw.circle(surf, (*color, a), (radius_px, radius_px), rr)
            self._glow_cache[key] = surf
        p = self._p(pos)
        self.screen.blit(surf, (p[0] - radius_px, p[1] - radius_px), special_flags=pygame.BLEND_RGBA_ADD)

    # --- scene ---
    def _bg(self):
        self.screen.fill(BG)
        step = int(10 * self._scale)
        for x in range(0, self.wpx, step):
            pygame.draw.line(self.screen, GRID, (x, 0), (x, self.hpx))
        for y in range(0, self.hpx, step):
            pygame.draw.line(self.screen, GRID, (0, y), (self.wpx, y))

    def _draw_obstacles(self, world):
        for pos, r, kind in zip(world.obstacle_pos, world.obstacle_r, world.obstacle_kind):
            rp = int(r * self._scale)
            if kind == 1:                                 # tree: trunk + layered foliage
                self._circle(TRUNK, pos, max(2, rp // 3))
                self._circle(LEAF, pos, rp)
                self._circle(LEAF2, pos + np.array([-r * 0.25, -r * 0.25]), int(rp * 0.7))
            else:                                          # rock: shaded boulder
                self._circle(ROCK_SH, pos, rp)
                self._circle(ROCK, pos, int(rp * 0.92))
                self._circle(ROCK_HI, pos + np.array([-r * 0.3, -r * 0.3]), int(rp * 0.4))

    def _draw_chickens(self, world):
        cr = world.cfg.chicken_radius
        for cpos, cdir in zip(world.chicken_pos, world.chicken_dir):
            self._glow(cpos, max(8, int(cr * 3.0 * self._scale)), GLOW)
            self._circle(CHICK_SH, cpos + np.array([0, cr * 0.25]), max(4, int(cr * self._scale)))
            self._circle(CHICK, cpos, max(3, int(cr * 0.9 * self._scale)))
            d = np.array([np.cos(cdir), np.sin(cdir)])
            pygame.draw.circle(self.screen, BEAK, self._p(cpos + d * cr * 0.9), max(2, int(cr * 0.3 * self._scale)))
            perp = np.array([-d[1], d[0]])
            pygame.draw.circle(self.screen, EYE, self._p(cpos + d * cr * 0.3 + perp * cr * 0.35), max(1, int(cr * 0.18 * self._scale)))

    def _draw_snake(self, world, body_uw):
        n = len(body_uw)
        hr = world.cfg.head_radius
        for k in range(n - 1, -1, -1):                    # tail first so the head lands on top
            t = k / max(1, n - 1)                          # 0 at head, 1 at tail
            r = hr * (1 - t) + world.cfg.body_radius * 0.7 * t
            self._circle(_lerp(SNAKE_HEAD, SNAKE_TAIL, t), wrap(body_uw[k], world.size), max(2, int(r * self._scale)))
        head = wrap(body_uw[0], world.size)
        d = body_uw[0] - body_uw[1] if n > 1 else world.heading_vec()
        norm = np.linalg.norm(d)
        d = d / norm if norm > 1e-6 else world.heading_vec()
        perp = np.array([-d[1], d[0]])
        hrp = hr * self._scale
        if world.dashed:                                   # forked tongue on a dash
            tip = head + d * hr * 1.2
            base = head + d * hr * 0.9
            pygame.draw.line(self.screen, TONGUE, self._p(base), self._p(tip), 2)
            for s in (1, -1):
                pygame.draw.line(self.screen, TONGUE, self._p(tip), self._p(tip + (d * 0.4 + perp * 0.4 * s) * hr), 2)
        for s in (1, -1):                                  # two eyes
            e = head + d * hr * 0.35 + perp * hr * 0.45 * s
            pygame.draw.circle(self.screen, SNAKE_EYE, self._p(e), max(2, int(hrp * 0.28)))
            pygame.draw.circle(self.screen, PUPIL, self._p(e + d * hr * 0.12), max(1, int(hrp * 0.14)))

    def _draw_sensors(self, world):
        v = sense_vision(world)
        for u, row in zip(ray_dirs(world.cfg, world.heading), v):
            dist = row[0] * world.cfg.ray_range
            nseg = max(2, int(dist))
            pts = [np.asarray(self._p(wrap(world.head + u * (dist * k / nseg), world.size))) for k in range(nseg + 1)]
            for a, b in zip(pts, pts[1:]):                 # skip the segment that jumps across the seam
                if abs(a[0] - b[0]) < self.wpx / 2 and abs(a[1] - b[1]) < self.hpx / 2:
                    pygame.draw.aaline(self.screen, RAY, a, b)

    def _hud(self, world):
        txt = f"len {world.target_length:.0f}   stamina {world.stamina:.0f}/{world.cfg.s_max:.0f}   energy {world.energy:.0f}"
        panel = self.font.render(txt, True, (225, 228, 235))
        self.screen.blit(panel, (10, 8))

    def draw(self, world, body_uw=None):
        self._ensure(world)
        if body_uw is None:
            body_uw = world.body_render_path_uw()
        self._bg()
        self._draw_obstacles(world)
        self._draw_chickens(world)
        self._draw_snake(world, body_uw)
        if self.show_sensors:
            self._draw_sensors(world)
        self._hud(world)
        pygame.display.flip()

    def toggle_sensors(self):
        self.show_sensors = not self.show_sensors

    def close(self):
        pygame.quit()
