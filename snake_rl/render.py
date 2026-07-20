"""pygame renderer: torus-aware world + optional sensor overlay + HUD."""
import numpy as np
import pygame
from .sensors import ray_dirs, sense_vision, smell
from .world import wrap

BG = (18, 20, 24); ROCK = (120, 120, 130); TREE = (60, 160, 80)
CHICK = (240, 220, 90); SNAKE = (90, 200, 240); HEAD = (240, 120, 120); RAY = (80, 90, 110)


class Renderer:
    def __init__(self, scale=8, show_sensors=True):
        pygame.init()
        self.scale = scale
        self.show_sensors = show_sensors
        self.screen = None
        self.wpx = self.hpx = 0
        self.font = pygame.font.SysFont("menlo,consolas,monospace", 14)

    def _ensure(self, world):
        w = int(world.size[0] * self.scale); h = int(world.size[1] * self.scale)
        if self.screen is None or (self.wpx, self.hpx) != (w, h):
            self.wpx, self.hpx = w, h
            self.screen = pygame.display.set_mode((w, h))

    def _circle(self, color, pos, radius_px):
        """Draw a circle at its wrapped position AND the 8 neighbor images (so it shows across the seam)."""
        bx = int(pos[0] * self.scale); by = int(pos[1] * self.scale)
        for ox in (0, -self.wpx, self.wpx):
            for oy in (0, -self.hpx, self.hpx):
                pygame.draw.circle(self.screen, color, (bx + ox, by + oy), radius_px)

    def draw(self, world):
        self._ensure(world)
        self.screen.fill(BG)
        for pos, r, kind in zip(world.obstacle_pos, world.obstacle_r, world.obstacle_kind):
            self._circle(TREE if kind == 1 else ROCK, pos, int(r * self.scale))
        for cpos in world.chicken_pos:
            self._circle(CHICK, cpos, max(3, int(world.cfg.chicken_radius * self.scale)))
        for bp in world.body_points():
            self._circle(SNAKE, bp, max(2, int(world.cfg.body_radius * self.scale)))
        self._circle(HEAD, world.head, max(3, int(world.cfg.head_radius * self.scale)))
        if self.show_sensors:
            self._draw_sensors(world)
        self._hud(world)
        pygame.display.flip()

    def _draw_sensors(self, world):
        v = sense_vision(world)
        for u, row in zip(ray_dirs(world.cfg, world.heading), v):
            dist = row[0] * world.cfg.ray_range
            n = max(2, int(dist))
            pts = [wrap(world.head + u * (dist * k / n), world.size) * self.scale for k in range(n + 1)]
            for a, b in zip(pts, pts[1:]):           # skip the segment that jumps across the seam
                if abs(a[0] - b[0]) < self.wpx / 2 and abs(a[1] - b[1]) < self.hpx / 2:
                    pygame.draw.line(self.screen, RAY, a, b, 1)

    def _hud(self, world):
        sm = smell(world)
        txt = f"len={world.target_length:.1f} sta={world.stamina:.0f} en={world.energy:.0f} smell={sm[0]:.2f}"
        self.screen.blit(self.font.render(txt, True, (220, 220, 220)), (6, 6))

    def toggle_sensors(self):
        self.show_sensors = not self.show_sensors

    def close(self):
        pygame.quit()
