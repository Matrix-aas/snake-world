import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from snake_rl.config import CFG
from snake_rl.worldgen import generate_world
from snake_rl.render import Renderer


def test_render_draws_without_error():
    w = generate_world(CFG, seed=0)
    r = Renderer(scale=6)
    r.draw(w)             # must not raise
    r.toggle_sensors()
    r.draw(w)
    r.close()
