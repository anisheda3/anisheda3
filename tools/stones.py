"""
Carved stepping stones: one per project, each rendered as its own transparent PNG
so the README can wrap every stone in its own link.
"""
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from build_koi_realistic import STONE_COLS, fbm, smoothstep

FONT = Path(__file__).parent / "fonts" / "CormorantGaramond.ttf"
TW, TH = 300, 150          # tile size in px (shown at half size, so it stays crisp on retina)


def _font(size, weight=600):
    f = ImageFont.truetype(str(FONT), size)
    try:
        f.set_variation_by_axes([weight])
    except Exception:
        pass
    return f


def _shift(m, dx, dy):
    return ImageChops.offset(m, dx, dy)


def render_stone(label, seed, out):
    r = np.random.default_rng(seed)
    pad = 14
    w, h = TW - 2 * pad, TH - 2 * pad - 8
    yy, xx = np.mgrid[0:TH, 0:TW].astype(float)
    X, Y = (xx - TW / 2) / (w / 2), (yy - (TH / 2 - 4)) / (h / 2)
    th = np.arctan2(Y, X); rad = np.hypot(X, Y)
    wob = 0.93 + sum(r.uniform(.015, .04) * np.sin(th * k + r.uniform(0, 6.28)) for k in (2, 3, 5, 7))
    d = rad / wob
    inside = d < 1
    alpha = Image.fromarray((inside * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.0))

    n = fbm(TH, TW, r, ((24, .5), (8, .3), (3, .2)))
    height = np.clip(1 - d ** 2, 0, 1) ** 0.22 + (n - .5) * 0.10       # flat-topped stepping stone
    gy, gx = np.gradient(height * 26)
    nrm = np.dstack([-gx, -gy, np.ones_like(gx)]); nrm /= np.linalg.norm(nrm, axis=2, keepdims=True)
    L = np.array([-.45, -.55, .70]); L /= np.linalg.norm(L)
    diff = np.clip((nrm * L).sum(2), 0, 1)

    base = STONE_COLS[[0, 5, 1, 6][seed % 4]] * 1.12      # light granite / sandstone only: carving stays legible
    grain = fbm(TH, TW, r, ((2, .5), (1, .5)))
    alb = base * (0.82 + 0.3 * grain)[..., None] * (0.9 + 0.2 * n)[..., None]
    moss = smoothstep(0.66, 0.84, n + smoothstep(0.8, 1.0, d) * 0.3 - 0.18) * inside
    alb = alb * (1 - moss[..., None] * .7) + np.array([.20, .27, .13]) * (moss[..., None] * .7)
    rgb = alb * (0.38 + 0.8 * diff)[..., None]

    # incised lettering: dark groove, shadowed upper-left wall, lit lower-right wall
    size = 46
    while size > 18:
        f = _font(size)
        bb = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), label, font=f)
        if bb[2] - bb[0] <= w * 0.74:
            break
        size -= 2
    m = Image.new("L", (TW, TH), 0)
    tx = TW / 2 - (bb[2] - bb[0]) / 2 - bb[0]
    ty = TH / 2 - 4 - (bb[3] - bb[1]) / 2 - bb[1]
    ImageDraw.Draw(m).text((tx, ty), label, font=f, fill=255)
    m = m.filter(ImageFilter.GaussianBlur(0.5))
    M = np.asarray(m, float) / 255
    shade = np.clip(M - np.asarray(_shift(m, 2, 2), float) / 255, 0, 1)
    lit = np.clip(M - np.asarray(_shift(m, -2, -2), float) / 255, 0, 1)
    rgb = rgb * (1 - 0.46 * M)[..., None] * (1 - 0.40 * shade)[..., None] + (0.20 * lit)[..., None]

    rgba = np.dstack([np.clip(rgb, 0, 1), np.asarray(alpha, float) / 255])
    stone = Image.fromarray((rgba * 255).astype(np.uint8), "RGBA")

    sh_a = _shift(alpha, 3, 6).filter(ImageFilter.GaussianBlur(5)).point(lambda v: int(v * .5))
    shadow = Image.new("RGBA", (TW, TH), (0, 0, 0, 0)); shadow.putalpha(sh_a)
    tile = Image.new("RGBA", (TW, TH), (0, 0, 0, 0))
    tile.alpha_composite(shadow); tile.alpha_composite(stone)
    out.parent.mkdir(parents=True, exist_ok=True)
    tile.save(out, optimize=True)
    return out
