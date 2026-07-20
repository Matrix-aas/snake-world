import numpy as np
from snake_rl.config import CFG
from snake_rl.world import World, wrap


def test_head_into_obstacle_kills():
    w = World(CFG, seed=0, size=(60, 60))
    w.head = np.array([30.0, 30.0]); w.head_uw = w.head.copy(); w.heading = 0.0
    w.obstacle_pos = np.array([[31.0, 30.0]]); w.obstacle_r = np.array([1.0]); w.obstacle_kind = np.array([0])
    out = w.step(steering=1, dash=0)   # moves +x into obstacle
    assert out["died"] and not w.alive


def test_dash_tunneling_still_kills():
    w = World(CFG, seed=0, size=(60, 60))
    w.head = np.array([30.0, 30.0]); w.head_uw = w.head.copy(); w.heading = 0.0
    w.stamina = CFG.s_max
    w.obstacle_pos = np.array([[31.2, 30.0]]); w.obstacle_r = np.array([0.3]); w.obstacle_kind = np.array([0])
    out = w.step(steering=1, dash=1)   # v_dash step would leap past a thin obstacle
    assert out["died"]


def test_no_death_on_empty_world():
    w = World(CFG, seed=0, size=(60, 60))
    out = w.step(steering=1, dash=0)
    assert not out["died"] and w.alive


def test_step_reports_eat():
    w = World(CFG, seed=0, size=(60, 60)); w.heading = 0.0
    w.head = np.array([30.0, 30.0]); w.head_uw = w.head.copy()
    w.obstacle_pos = np.zeros((0, 2)); w.obstacle_r = np.zeros((0,)); w.obstacle_kind = np.zeros((0,), int)
    w.set_chickens([[31.0, 30.0]])       # sits on the post-move head cell -> dist 0 -> wanders (no flee) -> eaten
    out = w.step(steering=1, dash=0)
    assert out["ate"] == 1 and not out["died"]


def test_self_collision_when_curled():
    # craft a full circular curl (radius = min turn radius); head returns onto the tail
    w = World(CFG, seed=0, size=(80, 80))
    w.target_length = CFG.length_cap
    R = CFG.v_snake / np.radians(CFG.turn_deg)
    ang = np.linspace(0.0, 2 * np.pi * 1.1, 500)      # slightly past a full circle
    center = np.array([40.0, 40.0])
    w.path_uw = [center + R * np.array([np.cos(a), np.sin(a)]) for a in ang]
    w.head_uw = w.path_uw[-1].copy(); w.head = wrap(w.head_uw, w.size)
    w._prev_head_uw = w.path_uw[-2].copy()
    assert w.check_death()               # head overlaps a body point ~one circumference back
