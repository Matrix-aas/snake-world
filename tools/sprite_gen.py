#!/usr/bin/env python3
"""Generate game sprites (and animation sheets) with REAL transparency via Codex.

Codex's `image_gen` cannot reliably emit a real alpha channel — it bakes a fake background (a
grey/white checkerboard, or a solid fill). So the reliable recipe is: ask Codex for the art on a
FLAT SOLID MAGENTA backdrop, then chroma-key that backdrop to alpha=0 here. Animation is one
sprite-SHEET image (a grid of frames) that this tool slices, keys, and centers into frame_*.png.

Requires the codex-vision wrapper (override with $CODEX_VISION):
  ~/.claude/skills/codex-vision/scripts/codex-vision.sh

Usage:
  python tools/sprite_gen.py "a mossy rock, top-down" snake_rl/assets/rock3.png
  python tools/sprite_gen.py "2x2 sheet: hen peck cycle, ..." out.png --sheet 2x2   # -> out_0..3.png
  python tools/sprite_gen.py --key snake_rl/assets/chicken.png                       # key an existing PNG
  python tools/sprite_gen.py --selfcheck                                             # offline asserts

The keyer auto-detects magenta vs checkerboard, so it also cleans sprites Codex returned with the
checkerboard background.
"""
import os
import subprocess
import sys

import numpy as np
import pygame

MAGENTA_SUFFIX = (", the background a FLAT SOLID PURE MAGENTA (#FF00FF) fill and nothing else, "
                  "no ground, no cast shadow")
CODEX = os.environ.get("CODEX_VISION",
                       os.path.expanduser("~/.claude/skills/codex-vision/scripts/codex-vision.sh"))


def _init():
    if not pygame.get_init():
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init(); pygame.display.set_mode((64, 64))


def _flood(mask, seed):
    cur = seed & mask
    while True:
        g = cur.copy()
        g[1:, :] |= cur[:-1, :]; g[:-1, :] |= cur[1:, :]
        g[:, 1:] |= cur[:, :-1]; g[:, :-1] |= cur[:, 1:]
        g &= mask
        if g.sum() == cur.sum():
            return cur
        cur = g


def _dilate1(m):
    g = m.copy()
    g[1:, :] |= m[:-1, :]; g[:-1, :] |= m[1:, :]
    g[:, 1:] |= m[:, :-1]; g[:, :-1] |= m[:, 1:]
    return g


def key_alpha(surf):
    """Return a copy of `surf` with the fake background (magenta OR checkerboard) keyed to alpha=0."""
    surf = surf.convert_alpha()
    rgb = pygame.surfarray.array3d(surf).astype(int)     # (W,H,3), pygame x,y indexing
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    val = (R + G + B) / 3.0
    corner = rgb[:8, :8].reshape(-1, 3).mean(0)
    if corner[0] > 150 and corner[2] > 150 and corner[1] < 120:          # solid magenta
        bg = (R > 120) & (B > 120) & (G < R - 30) & (G < B - 30)
        bg = _dilate1(bg) & (R > G) & (B > G) & (val > 90)               # 1px pink AA fringe
    else:                                                                 # grey/white checkerboard
        neutral = (np.abs(R - G) <= 7) & (np.abs(G - B) <= 7) & (np.abs(R - B) <= 9)
        checker = neutral & (val >= 224)
        seed = np.zeros_like(checker)
        seed[0, :] = seed[-1, :] = seed[:, 0] = seed[:, -1] = True
        bg = _flood(checker, seed & checker)
        bg = bg | (_dilate1(bg) & (np.abs(R - B) <= 20) & (val >= 208))   # 1px de-halo
    alpha = pygame.surfarray.array_alpha(surf).copy()
    alpha[bg] = 0
    out = surf.copy()
    av = pygame.surfarray.pixels_alpha(out); av[:] = alpha; del av
    return out


def _bbox(surf):
    a = pygame.surfarray.array_alpha(surf)
    xs, ys = np.where(a > 24)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _center_square(surf, side):
    """Crop `surf` to its subject and paste it centered into a transparent `side`x`side` canvas."""
    b = _bbox(surf)
    out = pygame.Surface((side, side), pygame.SRCALPHA)
    if b is None:
        return out
    x0, y0, x1, y1 = b
    sub = surf.subsurface(pygame.Rect(x0, y0, x1 - x0, y1 - y0)).copy()
    out.blit(sub, ((side - (x1 - x0)) // 2, (side - (y1 - y0)) // 2))
    return out


def slice_sheet(surf, rows, cols):
    """Key + slice a grid sprite sheet into per-frame surfaces, each auto-centered in a common
    square (so frames don't jump); row-major (left->right, top->bottom)."""
    surf = key_alpha(surf)
    w, h = surf.get_size()
    cw, ch = w // cols, h // rows
    frames = [surf.subsurface(pygame.Rect(c * cw, r * ch, cw, ch)).copy()
              for r in range(rows) for c in range(cols)]
    side = max((max((_bbox(f) or (0, 0, 1, 1))[2] - (_bbox(f) or (0, 0, 0, 0))[0],
                     (_bbox(f) or (0, 0, 0, 1))[3] - (_bbox(f) or (0, 0, 0, 0))[1]) for f in frames),
               default=cw)
    side = int(side * 1.12)
    return [_center_square(f, side) for f in frames]


def _run_codex(prompt, out):
    if not os.path.exists(CODEX):
        sys.exit(f"codex-vision wrapper not found at {CODEX} (set $CODEX_VISION).")
    subprocess.run([CODEX, "generate", prompt + MAGENTA_SUFFIX, "--out", out, "--fresh"], check=True)


def generate(prompt, out, sheet=None, size=512):
    """Generate a sprite on a magenta backdrop via Codex, key it to alpha, save to `out`.
    `sheet=(rows,cols)` slices an animation grid into <stem>_<i>.png frames instead."""
    _init()
    _run_codex(prompt, out)
    surf = pygame.image.load(out).convert_alpha()
    if sheet:
        stem, ext = os.path.splitext(out)
        os.remove(out)
        paths = []
        for i, f in enumerate(slice_sheet(surf, *sheet)):
            p = f"{stem}_{i}{ext}"
            pygame.image.save(pygame.transform.smoothscale(f, (size, size)), p)
            paths.append(p)
        return paths
    keyed = pygame.transform.smoothscale(key_alpha(surf), (size, size))
    pygame.image.save(keyed, out)
    return [out]


def key_file(path):
    _init()
    pygame.image.save(key_alpha(pygame.image.load(path)), path)


def _selfcheck():
    _init()
    for name, fill, subj in [("magenta", (255, 0, 255), (20, 200, 40)),
                             ("checker", None, (200, 30, 30))]:
        s = pygame.Surface((64, 64), pygame.SRCALPHA)
        if fill:
            s.fill(fill)
        else:                                            # paint a grey/white checkerboard
            for y in range(64):
                for x in range(64):
                    s.set_at((x, y), (252, 252, 252) if (x // 8 + y // 8) % 2 else (233, 233, 233))
        pygame.draw.circle(s, subj, (32, 32), 12)        # opaque subject in the middle
        k = key_alpha(s)
        a = pygame.surfarray.array_alpha(k)
        assert a[32, 32] > 200, f"{name}: subject center must stay opaque"
        assert a[2, 2] == 0, f"{name}: background corner must be keyed transparent"
    # sheet: a 1x2 magenta strip with two dots -> 2 centered frames, both non-empty
    s = pygame.Surface((128, 64), pygame.SRCALPHA); s.fill((255, 0, 255))
    pygame.draw.circle(s, (0, 0, 0), (20, 32), 8); pygame.draw.circle(s, (0, 0, 0), (100, 40), 8)
    frames = slice_sheet(s, 1, 2)
    assert len(frames) == 2 and all(_bbox(f) is not None for f in frames), "sheet slice failed"
    print("sprite_gen selfcheck: OK")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__); sys.exit(0)
    if args[0] == "--selfcheck":
        _selfcheck(); sys.exit(0)
    if args[0] == "--key":
        for p in args[1:]:
            key_file(p); print("keyed", p)
        sys.exit(0)
    prompt, out = args[0], args[1]
    sheet = None
    if "--sheet" in args:
        r, c = args[args.index("--sheet") + 1].lower().split("x")
        sheet = (int(r), int(c))
    print("wrote:", generate(prompt, out, sheet=sheet))
