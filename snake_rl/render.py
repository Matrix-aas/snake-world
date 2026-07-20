"""pygame renderer: torus world + optional sensor overlay + HUD."""
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
        self.font = pygame.font.SysFont("menlo", 14)

    def _ensure(self, world):
        w = int(world.size[0] * self.scale); h = int(world.size[1] * self.scale)
        if self.screen is None or self.screen.get_size() != (w, h):
            self.screen = pygame.display.set_mode((w, h))

    def _p(self, xy):
        return (int(xy[0] * self.scale), int(xy[1] * self.scale))

    def draw(self, world):
        self._ensure(world)
        s = self.screen; s.fill(BG)
        for pos, r, kind in zip(world.obstacle_pos, world.obstacle_r, world.obstacle_kind):
            pygame.draw.circle(s, TREE if kind == 1 else ROCK, self._p(pos), int(r * self.scale))
        for cpos in world.chicken_pos:
            pygame.draw.circle(s, CHICK, self._p(cpos), max(3, int(world.cfg.chicken_radius * self.scale)))
        for bp in world.body_points():
            pygame.draw.circle(s, SNAKE, self._p(bp), max(2, int(world.cfg.body_radius * self.scale)))
        pygame.draw.circle(s, HEAD, self._p(world.head), max(3, int(world.cfg.head_radius * self.scale)))
        if self.show_sensors:
            self._draw_sensors(world)
        self._hud(world)
        pygame.display.flip()

    def _draw_sensors(self, world):
        v = sense_vision(world)
        for u, row in zip(ray_dirs(world.cfg, world.heading), v):
            end = world.head + u * row[0] * world.cfg.ray_range
            pygame.draw.line(self.screen, RAY, self._p(world.head), self._p(wrap(end, world.size)), 1)

    def _hud(self, world):
        sm = smell(world)
        txt = f"len={world.target_length:.1f} sta={world.stamina:.0f} en={world.energy:.0f} smell={sm[0]:.2f}"
        self.screen.blit(self.font.render(txt, True, (220, 220, 220)), (6, 6))

    def toggle_sensors(self):
        self.show_sensors = not self.show_sensors

    def close(self):
        pygame.quit()
