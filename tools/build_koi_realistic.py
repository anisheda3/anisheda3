"""
Generates assets/koi-pond-realistic.svg: a naturalistic koi pond.

Koi skin, fins, lily pads and the pond bed are rendered procedurally with numpy
(3D-shaded bodies, overlapping scales, soft-edged pigment, translucent rayed fins)
and embedded as WebP data URIs. Motion is SMIL (jointed swimming, drifting
dappled sunlight, caustics, ripples), so it animates inside GitHub's <img> sandbox.

Run:  python tools/build_koi_realistic.py
"""
import base64, io, math, random, re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.interpolate import CubicSpline

import build_koi as bk  # reuse geometry helpers + caustic texture

OUT = Path(__file__).resolve().parents[1] / "assets" / "koi-pond-realistic.svg"
W, H = 1200, 420
PX = 6                      # raster pixels per koi unit
SLIM = 0.9
rng = random.Random(23)
nrng = np.random.default_rng(23)
f = bk.f


def webp(rgba, q=88):
    buf = io.BytesIO()
    Image.fromarray(np.clip(np.nan_to_num(rgba) * 255, 0, 255).astype(np.uint8), "RGBA").save(buf, "WEBP", quality=q, method=6)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def wave_noise(X, Y, n=7, fx=(0.05, 0.16), fy=(0.08, 0.25), seed=0):
    r = np.random.default_rng(seed)
    out = np.zeros_like(X)
    for _ in range(n):
        kx, ky = r.uniform(*fx) * r.choice([-1, 1]), r.uniform(*fy) * r.choice([-1, 1])
        out += r.uniform(.5, 1) * np.sin(kx * X + ky * Y + r.uniform(0, 6.28))
    return out / n * 2.2


# --------------------------------------------------------------------------
# Koi body
# --------------------------------------------------------------------------
PROF_X = [-50, -47, -40, -30, -18, -5, 8, 20, 30, 38, 44, 48, 50.5, 52]
PROF_W = [2.6, 2.8, 4.0, 7.5, 11.5, 14.5, 15.8, 16.0, 15.4, 13.6, 10.8, 7.4, 4.2, 0.0]
width_at = CubicSpline(PROF_X, PROF_W, bc_type="natural")

PIG = dict(orenji=np.array([.95, .45, .12]), yamabuki=np.array([.97, .84, .30]), white=np.array([.95, .94, .90]), red=np.array([.80, .17, .08]), black=np.array([.06, .06, .07]),
           gold=np.array([.96, .70, .26]), plat=np.array([.92, .92, .90]), asagi=np.array([.40, .52, .60]),
           orange=np.array([.90, .40, .14]))


def body_albedo(kind, X, Y, u, seed):
    shape = X.shape + (3,)
    back = 1 - np.abs(u)
    if kind in ("kohaku", "sanke", "tancho"):
        col = np.broadcast_to(PIG["white"], shape).copy()
        if kind == "tancho":
            m = 1 - smoothstep(5.2, 6.4, np.hypot(X - 37.5, Y / 1.05))
        else:
            n = wave_noise(X, Y, seed=seed) + 0.9 * back - 0.25
            m = smoothstep(0.08, 0.18, n) * (X < 47) * smoothstep(-44, -36, X)
        col = col * (1 - m[..., None]) + PIG["red"] * m[..., None]
        if kind == "sanke":
            s = wave_noise(X, Y, n=9, fx=(.2, .45), fy=(.25, .55), seed=seed + 7) + 0.4 * back
            k = smoothstep(0.78, 0.86, s) * (X < 36) * (X > -38)
            col = col * (1 - k[..., None]) + PIG["black"] * k[..., None]
    elif kind == "showa":
        col = np.broadcast_to(PIG["black"], shape).copy()
        n = wave_noise(X, Y, seed=seed) + 0.8 * back - 0.2
        rm = smoothstep(0.30, 0.40, n) * (X < 48)
        w_ = wave_noise(X, Y, seed=seed + 3) - 0.5 * back
        wm = smoothstep(0.35, 0.45, w_) * (1 - rm)
        col = col * (1 - rm[..., None]) + PIG["red"] * rm[..., None]
        col = col * (1 - wm[..., None]) + PIG["white"] * wm[..., None]
    elif kind in ("ogon", "orenji", "yamabuki"):
        col = np.broadcast_to(PIG["gold" if kind == "ogon" else kind], shape).copy() * (0.9 + 0.1 * back[..., None])
    elif kind == "platinum":
        col = np.broadcast_to(PIG["plat"], shape).copy()
    elif kind == "asagi":
        col = np.broadcast_to(PIG["asagi"], shape).copy()
        side = smoothstep(0.62, 0.8, np.abs(u)) * (X < 40)
        cheek = (1 - smoothstep(4, 7, np.hypot(X - 40, (np.abs(Y) - 8) * 1.3)))
        o = np.clip(side + cheek * 0.8, 0, 1)
        col = col * (1 - o[..., None]) + PIG["orange"] * o[..., None]
        head = smoothstep(36, 42, X)
        col = col * (1 - head[..., None] * .7) + PIG["white"] * (head[..., None] * .7)
    return col


def scale_field(X, A, s=2.35):
    """Overlapping scales laid out in (X, arc) space so rows wrap around the body.
    Returns edge (0..1), per-scale random (0..1), in-scale distance, and scale centres."""
    rowh = s * 0.6
    row = np.floor(A / rowh)
    col = np.floor((X + (row % 2) * s / 2) / s)
    best = np.full(X.shape, 1e9); bid = np.zeros(X.shape); bcx = np.zeros(X.shape); bca = np.zeros(X.shape)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r_, c_ = row + dr, col + dc
            cx = (c_ + 0.5) * s - (r_ % 2) * s / 2 + s * 0.3     # scale roots point toward the head
            ca = (r_ + 0.5) * rowh
            d = np.hypot((X - cx) / s, (A - ca) / (rowh * 1.25))
            take = d < best
            best = np.where(take, d, best); bid = np.where(take, r_ * 131.7 + c_ * 71.3, bid)
            bcx = np.where(take, cx, bcx); bca = np.where(take, ca, bca)
    edge = smoothstep(0.40, 0.56, best)
    rnd = (np.sin(bid * 12.9898) * 43758.5453) % 1
    return edge, rnd, best, bcx, bca


def render_body(kind, seed):
    x0, x1, y0, y1 = -50.0, 53.0, -16.0, 16.0
    nx, ny = int((x1 - x0) * PX), int((y1 - y0) * PX)
    X, Y = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
    w = np.clip(width_at(X) * SLIM, 0.01, None)
    u = np.clip(Y / w, -1.2, 1.2)
    alpha = np.clip((w - np.abs(Y)) * PX * 0.8, 0, 1) * (X < 52)
    z = np.sqrt(np.clip(1 - u * u, 0, 1)) * np.clip(w, 0, None) * 0.55
    gy, gx = np.gradient(z, (y1 - y0) / ny, (x1 - x0) / nx)
    n = np.dstack([-gx, -gy, np.ones_like(z)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    nz = n[..., 2]
    metallic = kind in ("ogon", "orenji", "yamabuki", "platinum")

    # arc-length coordinate across the back: scale rows follow the body's curvature
    A = np.arcsin(np.clip(u, -1, 1)) * 9.0
    edge, rnd, dsc, CX, CA = scale_field(X, A)
    scale_zone = smoothstep(34.5, 31.5, X) * smoothstep(-47, -41, X) * smoothstep(1.0, 0.8, np.abs(u))

    # pigment is set per scale (real koi pattern edges follow scale rows), then lightly mixed with per-pixel
    alb_px = body_albedo(kind, X, Y, u, seed)
    uc = np.sin(np.clip(CA / 9.0, -1.5, 1.5))
    Yc = uc * np.clip(width_at(CX) * SLIM, 0.01, None)
    alb_sc = body_albedo(kind, CX, Yc, uc, seed)
    k = (0.75 * scale_zone)[..., None]
    alb = alb_px * (1 - k) + alb_sc * k

    # natural pigment depth: richer along the spine, lighter toward the flanks
    alb = alb * (0.9 + 0.14 * np.clip(1 - np.abs(u), 0, 1) ** 0.5)[..., None]

    # scale relief: darker overlapping rims, slightly domed centres
    edge_dark = (0.22 if kind != "showa" else 0.14) if kind != "asagi" else -0.32
    alb = alb * (1 - (edge * edge_dark * scale_zone)[..., None])
    alb = alb * (1 + ((0.5 - dsc) * 0.10 * scale_zone)[..., None])
    alb = alb * (1 + ((rnd - 0.5) * (0.14 if metallic else 0.06) * scale_zone)[..., None])

    # gill plate (operculum): fine dark crease with a lit lip behind it
    gill_x = 34.2 - 3.2 * u ** 2
    crease = np.exp(-((X - gill_x) / 0.32) ** 2) * smoothstep(0.98, 0.55, np.abs(u))
    lip = np.exp(-((X - gill_x + 0.8) / 0.5) ** 2) * smoothstep(0.98, 0.55, np.abs(u))
    alb = alb * (1 - 0.32 * crease)[..., None] + 0.05 * lip[..., None]

    # dorsal fin seen from above: narrow translucent ridge with raked rays
    dz = smoothstep(-31, -26, X) * smoothstep(17, 12, X)
    dw = 1.25 * dz + 1e-3
    dm = smoothstep(dw, dw * 0.55, np.abs(Y)) * (dz > 0.01)
    rays = (0.5 + 0.5 * np.cos((X + np.abs(Y) * 2.2) * 2.4)) ** 6
    fin_col = np.array(FIN_TINT[kind]) * (0.55 if kind == "showa" else 0.9)
    alb = alb * (1 - (0.45 * dm)[..., None]) + fin_col * (0.45 * dm)[..., None] * (0.85 + 0.3 * rays[..., None])

    # lighting: overhead sun + sky reflection (rotation invariant, so it stays right as the fish turns)
    diffuse = 0.30 + 0.80 * nz ** 1.25
    spec = nz ** (130 if metallic else 100) * 0.30
    spec = spec * (0.55 + 0.9 * rnd * scale_zone + (1 - scale_zone) * 0.45)      # each scale catches light differently
    sheen = np.exp(-((u / 0.2) ** 2)) * smoothstep(-42, 0, X) * smoothstep(46, 30, X) * (0.10 if not metallic else 0.15)
    head = smoothstep(33, 37, X)
    head_gloss = np.exp(-(((X - 41) / 5) ** 2 + (u / 0.45) ** 2)) * 0.12
    glitter = smoothstep(0.82, 1.0, rnd) * (1 - edge) * scale_zone * (0.16 if metallic else 0.05) * nz
    ao = 0.5 + 0.5 * smoothstep(0, 0.6, z / (z.max() + 1e-6) * 3)
    belly = smoothstep(0.82, 1.0, np.abs(u)) * 0.18                                # flank turning under, toward the belly
    rgb = (alb * (diffuse * ao * (1 - belly))[..., None]
           + (spec + sheen + glitter + head_gloss * head)[..., None] * np.array([1, .99, .96]))

    # eyes (gold iris ring, dark pupil, catch-light) and nostrils
    for sgn in (-1, 1):
        ex, ey = 40.5, sgn * width_at(40.5) * SLIM * 0.84
        d = np.hypot(X - ex, Y - ey)
        rgb = np.where((d < 1.7)[..., None], np.array([.58, .45, .22]) * (0.7 + 0.3 * nz[..., None]), rgb)
        rgb = np.where((d < 1.05)[..., None], np.array([.025, .025, .03]), rgb)
        rgb = rgb * (1 - 0.35 * np.exp(-((d - 1.8) / 0.25) ** 2))[..., None]
        rgb += (np.exp(-((np.hypot(X - ex + .35, Y - ey + .35 * sgn)) / .28) ** 2) * 0.95)[..., None]
        dn = np.hypot(X - 47.4, Y - sgn * 2.7)
        rgb *= (1 - 0.55 * np.exp(-(dn / .5) ** 2))[..., None]
    # mouth line at the snout
    rgb *= (1 - 0.35 * np.exp(-((X - (51.3 - 0.9 * (Y / 3) ** 2)) / 0.3) ** 2) * (np.abs(Y) < 3))[..., None]
    rgba = np.dstack([np.clip(rgb, 0, 1), alpha])

    def seg(xa, xb, fade_from=None, fade_to=None):
        m = ((X >= xa) & (X <= xb)).astype(float)
        if fade_from is not None:
            m *= smoothstep(fade_from, fade_to, X)
        im = rgba.copy(); im[..., 3] *= m
        cols = np.where(im[..., 3].max(0) > 0.004)[0]; rows = np.where(im[..., 3].max(1) > 0.004)[0]
        c0, c1, r0, r1 = cols[0], cols[-1] + 1, rows[0], rows[-1] + 1
        crop = im[r0:r1, c0:c1]
        return dict(href=webp(crop), x=x0 + c0 / PX, y=y0 + r0 / PX, w=(c1 - c0) / PX, h=(r1 - r0) / PX)

    return dict(front=seg(0, 53, 0, 6), mid=seg(-31, 8, -30, -23), ped=seg(-50, -20))


# --------------------------------------------------------------------------
# Fins (translucent, rayed)
# --------------------------------------------------------------------------
def sample_path(d, steps=28):
    toks = re.findall(r"[MCLZ]|-?\d+\.?\d*", d)
    pts, cur, i = [], None, 0
    while i < len(toks):
        t = toks[i]
        if t == "M":
            cur = (float(toks[i + 1]), float(toks[i + 2])); pts.append(cur); i += 3
        elif t == "L":
            cur = (float(toks[i + 1]), float(toks[i + 2])); pts.append(cur); i += 3
        elif t == "C":
            c = [float(v) for v in toks[i + 1:i + 7]]
            p0, p1, p2, p3 = cur, (c[0], c[1]), (c[2], c[3]), (c[4], c[5])
            for k in range(1, steps + 1):
                s = k / steps; m = 1 - s
                pts.append((m ** 3 * p0[0] + 3 * m * m * s * p1[0] + 3 * m * s * s * p2[0] + s ** 3 * p3[0],
                            m ** 3 * p0[1] + 3 * m * m * s * p1[1] + 3 * m * s * s * p2[1] + s ** 3 * p3[1]))
            cur = p3; i += 7
        else:
            i += 1
    return pts


def render_fin(d, base, tint, a0=.85, a1=.22, rays=11, dark_base=False, px=PX):
    pts = sample_path(d)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    bx0, by0, bx1, by1 = min(xs) - 1, min(ys) - 1, max(xs) + 1, max(ys) + 1
    ss = 4
    wpx, hpx = int((bx1 - bx0) * px), int((by1 - by0) * px)
    img = Image.new("L", (wpx * ss, hpx * ss), 0)
    ImageDraw.Draw(img).polygon([((x - bx0) * px * ss, (y - by0) * px * ss) for x, y in pts], fill=255)
    mask = np.asarray(img.resize((wpx, hpx), Image.LANCZOS), dtype=float) / 255
    X, Y = np.meshgrid(np.linspace(bx0, bx1, wpx), np.linspace(by0, by1, hpx))
    dx, dy = X - base[0], Y - base[1]
    dist = np.hypot(dx, dy); t = dist / (dist[mask > .5].max() + 1e-6)
    ang = np.arctan2(dy, dx)
    ray = (0.5 + 0.5 * np.cos(ang * rays * 2)) ** 6
    a = mask * (a0 * (1 - t) + a1 * t) * (0.78 + 0.22 * ray)
    col = np.broadcast_to(np.array(tint), X.shape + (3,)) * (0.86 + 0.2 * ray[..., None])
    if dark_base:
        k = smoothstep(0.45, 0.15, t)[..., None]
        col = col * (1 - k) + np.array([.05, .05, .06]) * k
        a = np.maximum(a, mask * 0.8 * k[..., 0])
    edge = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.FIND_EDGES), dtype=float) / 255
    col = col + edge[..., None] * 0.15
    return dict(href=webp(np.dstack([np.clip(col, 0, 1), a])), x=bx0, y=by0, w=bx1 - bx0, h=by1 - by0)


SL = lambda d: re.sub(r"(-?\d+\.?\d*),(-?\d+\.?\d*)", lambda m: f"{m.group(1)},{f(float(m.group(2)) * SLIM)}", d)
CAUDAL = SL("M-45,0 C-51,-5 -60,-12 -80,-22 C-79,-12 -72,-5 -66,0 C-72,5 -79,12 -80,22 C-60,12 -51,5 -45,0Z")
PECT_L = SL("M33,-13 C32,-22 26,-31 14,-35 C8,-36.5 4.5,-32.5 8,-27 C12,-21 19,-16 25.5,-12Z")
PELV_L = SL("M-5,-12.5 C-7,-18 -12,-22.5 -19,-23.5 C-22,-23.5 -22,-20 -18.5,-17.5 C-15,-15 -12,-13 -10,-11.5Z")
FLIP = lambda d: re.sub(r"(-?\d+\.?\d*),(-?\d+\.?\d*)", lambda m: f"{m.group(1)},{f(-float(m.group(2)))}", d)

FIN_TINT = dict(orenji=(.98, .62, .35), yamabuki=(.98, .88, .55), kohaku=(.96, .95, .92), tancho=(.96, .95, .92), sanke=(.96, .95, .92), showa=(.93, .92, .9),
                ogon=(.98, .80, .45), platinum=(.95, .95, .93), asagi=(.92, .55, .32))


def img_tag(r):
    return f'<image href="{r["href"]}" x="{f(r["x"])}" y="{f(r["y"])}" width="{f(r["w"])}" height="{f(r["h"])}"/>'


def koi(uid, kind, path_d, dur, scale, wag, delay, seed):
    b = render_body(kind, seed)
    tint = FIN_TINT[kind]; dark = kind == "showa"
    caudal = render_fin(CAUDAL, (-45, 0), tint, .8, .2, rays=9, dark_base=dark)
    pl, pr = (render_fin(PECT_L, (28, -12.5 * SLIM), tint, .7, .25, rays=7, dark_base=dark),
              render_fin(FLIP(PECT_L), (28, 12.5 * SLIM), tint, .7, .25, rays=7, dark_base=dark))
    vl, vr = (render_fin(PELV_L, (-7, -12 * SLIM), tint, .65, .25, rays=5),
              render_fin(FLIP(PELV_L), (-7, 12 * SLIM), tint, .65, .25, rays=5))
    ph = lambda frac: f(-wag * frac - delay)

    def rot(amp, frac, cx, cy=0, k=1.0):
        return (f'<animateTransform attributeName="transform" type="rotate" dur="{f(wag * k)}s" begin="{ph(frac)}s" '
                f'repeatCount="indefinite" calcMode="spline" keyTimes="0;.5;1" keySplines=".45 0 .55 1;.45 0 .55 1" '
                f'values="{-amp} {cx} {cy};{amp} {cx} {cy};{-amp} {cx} {cy}"/>')

    tail = f'<g>{rot(11, .5, -27)}<g>{rot(10, .85, -46)}{img_tag(caudal)}</g>{img_tag(b["ped"])}</g>'
    mid = (f'<g>{rot(5, .25, 4)}'
           f'<g>{rot(9, .3, -8, -11, 1.3)}{img_tag(vl)}</g><g>{rot(9, .8, -8, 11, 1.3)}{img_tag(vr)}</g>'
           f'{tail}{img_tag(b["mid"])}</g>')
    head = (f'<g>{rot(13, 0, 27, -12, 1.3)}{img_tag(pl)}</g><g>{rot(13, .5, 27, 12, 1.3)}{img_tag(pr)}</g>'
            f'{mid}{img_tag(b["front"])}'
            f'<path d="M49.5,-2.6 q2.5,-1 4,-3.6 M49.5,2.6 q2.5,1 4,3.6" stroke="rgba(235,225,210,.55)" stroke-width=".45" fill="none"/>')
    return (f'<g id="{uid}"><animateMotion dur="{f(dur)}s" begin="{f(-delay * 7.3)}s" repeatCount="indefinite" rotate="auto" path="{path_d}"/>'
            f'<g transform="scale({f(scale)})">{head}</g></g>')


# --------------------------------------------------------------------------
# Environment textures
# --------------------------------------------------------------------------
def fbm(h, w, r, octaves=((40, .5), (16, .3), (6, .2))):
    out = np.zeros((h, w))
    for sc, amp in octaves:
        g = r.random((h // sc + 3, w // sc + 3))
        out += amp * np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC), float) / 255
    return out


STONE_COLS = [np.array(c) for c in ([.55, .52, .47], [.68, .58, .44], [.40, .38, .36], [.62, .42, .30],
                                    [.30, .29, .29], [.72, .68, .60], [.52, .47, .36], [.58, .36, .26])]


def pond_bed(w=W, h=H, seed=8):
    """Rounded river pebbles resting on fine gravel, seen through green water."""
    from scipy.spatial import cKDTree
    r = np.random.default_rng(seed)
    seeds = r.random((640, 2)) * [w, h]
    yy, xx = np.mgrid[0:h, 0:w]
    pts = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
    for _ in range(3):                                          # Lloyd relaxation: pebbles settle evenly
        _, idx = cKDTree(seeds).query(pts, k=1)
        cnt = np.bincount(idx, minlength=len(seeds)) + 1e-9
        seeds = np.column_stack([np.bincount(idx, pts[:, 0], len(seeds)) / cnt, np.bincount(idx, pts[:, 1], len(seeds)) / cnt])
    seeds += r.normal(0, 2.5, seeds.shape)
    tree = cKDTree(seeds)
    nn = tree.query(seeds, k=2)[0][:, 1]
    _, idx = tree.query(pts, k=1)
    cell = idx.reshape(h, w)
    ang = r.uniform(0, math.pi, len(seeds)); elong = r.uniform(.7, .95, len(seeds))
    rin = nn * 0.5 * r.uniform(.92, 1.08, len(seeds))
    dx, dy = xx - seeds[cell, 0], yy - seeds[cell, 1]
    ca, sa = np.cos(ang[cell]), np.sin(ang[cell])
    a_, b_ = rin[cell], rin[cell] * elong[cell]
    e = ((dx * ca + dy * sa) / a_) ** 2 + ((-dx * sa + dy * ca) / b_) ** 2
    height = np.sqrt(np.clip(1 - e, 0, 1))
    stone = (height > 0).astype(float)
    cols = np.array(STONE_COLS)[r.integers(len(STONE_COLS), size=len(seeds))] * r.uniform(.8, 1.12, (len(seeds), 1))
    gravel = fbm(h, w, r, ((2, .5), (1, .5)))
    alb = np.where(stone[..., None] > 0, cols[cell], np.array([.30, .27, .21]) * (0.6 + 0.6 * gravel[..., None]))
    gy, gx = np.gradient(height * 5)
    lit = np.clip(0.85 + (-gx * 0.6 - gy * 0.8), 0.35, 1.3)
    contact = 0.45 + 0.55 * smoothstep(0, 0.35, height + (1 - stone) * 0.5)   # contact shadow at pebble rims
    rgb = alb * (lit * contact)[..., None] * (0.9 + 0.2 * fbm(h, w, r, ((3, .6), (1, .4)))[..., None])
    rgb += (np.clip(height - 0.8, 0, 1) * 0.3)[..., None]                       # wet crowns
    rgb *= (0.75 + 0.35 * fbm(h, w, r))[..., None]                              # silt / algae mottling
    water = np.array([.07, .20, .16])
    rgb = rgb * np.array([.62, .84, .76]) * 0.86 + water * 0.26                 # absorbed and tinted by depth
    im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))
    buf = io.BytesIO(); im.save(buf, "WEBP", quality=82, method=6)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


def boulder(seed, wpx, hpx, moss=0.5, tone=0):
    """Mossy pond-edge boulder, lit from the top-left, with a dark wet band at the waterline."""
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:hpx, 0:wpx].astype(float)
    X, Y = (xx - wpx / 2) / (wpx / 2), (yy - hpx / 2) / (hpx / 2)
    th = np.arctan2(Y, X); rad = np.hypot(X, Y)
    wob = 0.86 + sum(r.uniform(.02, .06) * np.sin(th * k + r.uniform(0, 6.28)) for k in (2, 3, 5, 8))
    n = fbm(hpx, wpx, r, ((max(wpx // 6, 2), .5), (max(wpx // 18, 2), .3), (3, .2)))
    d = rad / wob
    height = np.clip(1 - d ** 2, 0, 1) ** 0.55 + (n - .5) * 0.22 * (d < 1)
    alpha = np.asarray(Image.fromarray(((d < 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2)), float) / 255
    gy, gx = np.gradient(height * min(wpx, hpx) * 0.35)
    nrm = np.dstack([-gx, -gy, np.ones_like(gx)]); nrm /= np.linalg.norm(nrm, axis=2, keepdims=True)
    L = np.array([-.45, -.55, .70]); L /= np.linalg.norm(L)
    diff = np.clip((nrm * L).sum(2), 0, 1)
    spec = np.clip((nrm * (L + [0, 0, 1]) / np.linalg.norm(L + [0, 0, 1])).sum(2), 0, 1) ** 40
    stone = STONE_COLS[[0, 2, 5][tone]] * (0.85 + 0.3 * fbm(hpx, wpx, r, ((2, .5), (1, .5))))[..., None]
    mossm = smoothstep(0.55, 0.75, n + nrm[..., 2] * 0.3 - (1 - moss) * 0.6) * (d < 0.97)
    mossc = np.array([.24, .34, .13]) * (0.75 + 0.5 * fbm(hpx, wpx, r, ((3, .5), (1, .5))))[..., None]
    alb = stone * (1 - mossm[..., None]) + mossc * mossm[..., None]
    wet = smoothstep(0.84, 0.99, d)                     # waterline: darker, glossier
    alb = alb * (1 - 0.45 * wet[..., None])
    rgb = alb * 1.25 * (0.32 + 0.95 * diff)[..., None] + (spec * (0.1 + 0.5 * wet) * (1 - mossm))[..., None]
    rgb *= (0.7 + 0.3 * smoothstep(0, 0.35, height))[..., None]
    return webp(np.dstack([np.clip(rgb, 0, 1), alpha]), q=86)


def canopy_shadow(n=512, seed=4):
    """Tileable dappled tree shade: dark alpha with soft sunlit holes."""
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n] / n
    acc = np.zeros((n, n))
    for k, amp in ((2, 1), (3, .6), (5, .35), (9, .18)):
        for _ in range(4):
            fx, fy = r.integers(-k, k + 1), r.integers(-k, k + 1)
            acc += amp * np.sin(2 * np.pi * (fx * xx + fy * yy) + r.uniform(0, 6.28))
    acc = (acc - acc.min()) / (acc.max() - acc.min())
    a = smoothstep(0.42, 0.62, acc) * 0.85
    rgba = np.dstack([np.full((n, n), .0), np.full((n, n), .03), np.full((n, n), .02), a])
    im = Image.fromarray((rgba * 255).astype(np.uint8), "RGBA").filter(ImageFilter.GaussianBlur(6))
    buf = io.BytesIO(); im.save(buf, "WEBP", quality=80, method=6)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


def lily_pad_raster(seed, red=False, px=220):
    r = np.random.default_rng(seed)
    yy, xx = (np.mgrid[0:px, 0:px] - px / 2) / (px / 2)
    rad, th = np.hypot(xx, yy), np.arctan2(yy, xx)
    notch = np.abs(((th + math.pi) % (2 * math.pi)) - math.pi) < 0.17 + 0.25 * (1 - rad)
    wob = 0.965 + 0.02 * np.sin(th * 7 + r.uniform(0, 6)) + 0.012 * np.sin(th * 13)
    inside = (rad < wob) & ~notch
    alpha = np.asarray(Image.fromarray((inside * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(.8)), float) / 255
    veins = (0.5 + 0.5 * np.cos(th * 26)) ** 18 * smoothstep(0.05, 0.25, rad)
    noise = np.asarray(Image.fromarray((r.random((22, 22)) * 255).astype(np.uint8)).resize((px, px), Image.BICUBIC), float) / 255
    g0, g1 = (np.array([.36, .26, .16]), np.array([.55, .36, .22])) if red else (np.array([.13, .30, .12]), np.array([.30, .50, .22]))
    t = np.clip(0.35 + 0.5 * noise - 0.2 * rad, 0, 1)[..., None]
    col = g0 + (g1 - g0) * t
    col = col * (1 + 0.18 * veins[..., None])
    rim = smoothstep(0.86, 0.97, rad / wob)
    col = col * (1 - 0.35 * rim[..., None]) + np.array([.45, .28, .18]) * (0.35 * rim[..., None])
    gloss = np.exp(-(((xx + .3) ** 2 + (yy + .35) ** 2) / 0.12)) * 0.28 + 0.12 * smoothstep(0.9, 0.97, rad / wob) * (yy < 0)
    col = np.clip(col + gloss[..., None], 0, 1)
    return webp(np.dstack([col, alpha]), q=85)


def leaf_raster(seed, px=90):
    r = np.random.default_rng(seed)
    yy, xx = (np.mgrid[0:px // 2, 0:px] - np.array([px / 4, px / 2])[:, None, None]) / np.array([px / 4, px / 2])[:, None, None]
    shape = (np.abs(yy) < (1 - xx ** 2) ** 1.2 * 0.95)
    alpha = np.asarray(Image.fromarray((shape * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(.6)), float) / 255
    hue = r.choice([np.array([.62, .38, .12]), np.array([.55, .22, .10]), np.array([.66, .52, .18])])
    vein = np.exp(-(yy / 0.05) ** 2) * 0.25 + (0.5 + 0.5 * np.cos((xx * 9 - np.abs(yy) * 6) * math.pi)) ** 14 * 0.12
    col = np.clip(hue * (0.8 + 0.25 * (1 - np.abs(yy))[..., None]) + vein[..., None], 0, 1)
    return webp(np.dstack([col, alpha]), q=85)


# --------------------------------------------------------------------------
# Assemble
# --------------------------------------------------------------------------
DEFAULT_ROSTER = [
    dict(id="rk0", kind="kohaku", cx=330, cy=215, rx=360, ry=165, scale=1.5, wag=2.6, layer="top", seed=100),
    dict(id="rk1", kind="ogon", cx=860, cy=205, rx=360, ry=150, scale=1.4, wag=2.4, layer="top", seed=101),
    dict(id="rk2", kind="showa", cx=620, cy=240, rx=560, ry=175, scale=1.55, wag=2.8, layer="top", seed=102),
    dict(id="rk3", kind="tancho", cx=930, cy=250, rx=300, ry=150, scale=1.3, wag=2.3, layer="top", seed=103),
    dict(id="rk4", kind="sanke", cx=470, cy=180, rx=430, ry=140, scale=1.05, wag=2.4, layer="deep", seed=104),
    dict(id="rk5", kind="asagi", cx=760, cy=270, rx=470, ry=130, scale=1.0, wag=2.6, layer="deep", seed=105),
    dict(id="rk6", kind="platinum", cx=250, cy=290, rx=260, ry=110, scale=.9, wag=2.2, layer="deep", seed=106),
]


def build(roster=None, out=None, open_bottom=False):
    """roster: list of dicts(id, kind, cx, cy, rx, ry, scale, wag, layer, seed). Each fish's path,
    speed, pattern and timing derive only from its own seed, so adding/removing a fish never
    reshuffles the others."""
    roster = roster or DEFAULT_ROSTER
    out = Path(out) if out else OUT
    caustic = bk.tileable_caustics(seed=9)
    bed, canopy = pond_bed(), canopy_shadow()
    pads = [lily_pad_raster(s, red=(s % 3 == 2)) for s in range(3)]
    leaves = [leaf_raster(s) for s in range(3)]
    rocks = [  # x, y, w, h, seed, moss, tone
        (480, -6, 230, 120, 1, .65, 0), (618, 4, 120, 72, 2, .4, 2), (370, 6, 92, 58, 3, .5, 1),
        (770, 418, 250, 120, 4, .55, 1), (912, 432, 112, 68, 5, .3, 0),
        (1150, 18, 150, 115, 6, .6, 2), (-12, 240, 120, 150, 7, .5, 0),
    ]
    if open_bottom:                                   # keep the lower edge clear: koi linger there above their stones
        rocks = [rk for rk in rocks if rk[1] < 300]
    rock_imgs = [(x, y, w_, h_, boulder(sd, int(w_ * 1.6), int(h_ * 1.6), m, t)) for x, y, w_, h_, sd, m, t in rocks]

    koi_defs, layers = [], {"top": [], "deep": []}
    for fish in roster:
        fr = random.Random(fish["seed"])
        bk.rng = random.Random(fish["seed"] * 7 + 1)          # path shape + body patches: per-fish only
        rx, ry, s = fish["rx"], fish["ry"], fish["scale"]
        d = fish.get("path") or bk.loop_path(fish["cx"], fish["cy"], rx, ry, k=6, ccw=fish["seed"] % 2 == 1)
        circ = math.pi * (3 * (rx + ry) - math.sqrt((3 * rx + ry) * (rx + 3 * ry)))
        dur = circ / (fr.uniform(22, 28) * s ** .5)
        g = koi(fish["id"], fish["kind"], d, dur, s, fish["wag"], fr.uniform(0, 9), seed=fish["seed"])
        if fish.get("linger"):                        # slow, unhurried pass along the path's midpoint
            g = g.replace('rotate="auto"', 'rotate="auto" calcMode="spline" keyPoints="0;0.40;0.60;1" '
                          'keyTimes="0;0.28;0.78;1" keySplines=".3 0 .7 1;.3 0 .7 1;.3 0 .7 1"', 1)
        koi_defs.append(g)
        layers[fish["layer"]].append(fish["id"])
    use = lambda ids: "".join(f'<use href="#{u}"/>' for u in ids)

    def pad(i, cx, cy, size, rot, sway, dur, begin):
        return (f'<g transform="translate({cx},{cy})">'
                f'<ellipse cx="14" cy="18" rx="{f(size * .5)}" ry="{f(size * .48)}" fill="url(#padShadow)"/>'
                f'<g><animateTransform attributeName="transform" type="rotate" dur="{dur}s" begin="{begin}s" repeatCount="indefinite" '
                f'calcMode="spline" keyTimes="0;.5;1" keySplines=".45 0 .55 1;.45 0 .55 1" values="{rot - sway};{rot + sway};{rot - sway}"/>'
                f'<image href="{pads[i]}" x="{f(-size / 2)}" y="{f(-size / 2)}" width="{size}" height="{size}"/></g></g>')

    def leaf(i, path_d, dur, begin, spin, size=34):
        return (f'<g><animateMotion dur="{dur}s" begin="{begin}s" repeatCount="indefinite" path="{path_d}"/>'
                f'<g><animateTransform attributeName="transform" type="rotate" from="0" to="{spin}" dur="{dur}s" begin="{begin}s" repeatCount="indefinite"/>'
                f'<image href="{leaves[i]}" x="{f(-size / 2 + 3)}" y="{f(-size / 4 + 4)}" width="{size}" height="{f(size / 2)}" opacity=".35" filter="url(#dark)"/>'
                f'<image href="{leaves[i]}" x="{f(-size / 2)}" y="{f(-size / 4)}" width="{size}" height="{f(size / 2)}"/></g></g>')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Animated koi pond">
<title>Koi pond</title>
<defs>
<filter id="shadowify" x="-10%" y="-10%" width="120%" height="120%" color-interpolation-filters="sRGB">
  <feColorMatrix type="matrix" values="0 0 0 0 0  0 0 0 0 .02  0 0 0 0 .02  0 0 0 .6 0"/><feGaussianBlur stdDeviation="6"/>
</filter>
<filter id="rockShadow" x="-20%" y="-20%" width="140%" height="140%"><feColorMatrix type="matrix" values="0 0 0 0 0  0 0 0 0 .02  0 0 0 0 .01  0 0 0 .7 0"/><feGaussianBlur stdDeviation="9"/></filter>
<filter id="meniscus" x="-20%" y="-20%" width="140%" height="140%"><feColorMatrix type="matrix" values="0 0 0 0 .85  0 0 0 0 .95  0 0 0 0 .92  0 0 0 .16 0"/><feGaussianBlur stdDeviation="2"/></filter>
<filter id="depthBlur" x="-10%" y="-10%" width="120%" height="120%"><feGaussianBlur stdDeviation=".9"/></filter>
<filter id="dark"><feColorMatrix type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 1 0"/><feGaussianBlur stdDeviation="1.5"/></filter>
<radialGradient id="padShadow"><stop offset=".6" stop-color="#000" stop-opacity=".45"/><stop offset="1" stop-color="#000" stop-opacity="0"/></radialGradient>
<radialGradient id="vignette" cx=".5" cy=".5" r=".75"><stop offset=".5" stop-color="#000" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity=".55"/></radialGradient>
<radialGradient id="sun" cx=".35" cy=".2" r=".85"><stop offset="0" stop-color="#7fc9a8" stop-opacity=".22"/><stop offset="1" stop-color="#7fc9a8" stop-opacity="0"/></radialGradient>
<linearGradient id="sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#dff3ea" stop-opacity=".10"/><stop offset=".35" stop-color="#dff3ea" stop-opacity="0"/></linearGradient>
<linearGradient id="murk" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#1d4a3a" stop-opacity=".05"/><stop offset="1" stop-color="#021008" stop-opacity=".38"/></linearGradient>
<pattern id="caustA" width="576" height="576" patternUnits="userSpaceOnUse">
  <image href="data:image/png;base64,{caustic}" width="576" height="576"/>
  <animateTransform attributeName="patternTransform" type="translate" from="0 0" to="576 576" dur="80s" repeatCount="indefinite"/></pattern>
<pattern id="caustB" width="800" height="800" patternUnits="userSpaceOnUse">
  <image href="data:image/png;base64,{caustic}" width="800" height="800"/>
  <animateTransform attributeName="patternTransform" type="translate" from="0 0" to="-800 800" dur="110s" repeatCount="indefinite"/></pattern>
<pattern id="canopy" width="900" height="900" patternUnits="userSpaceOnUse">
  <image href="{canopy}" width="900" height="900"/>
  <animateTransform attributeName="patternTransform" type="translate" from="0 0" to="900 0" dur="240s" repeatCount="indefinite"/></pattern>
<pattern id="canopy2" width="1300" height="1300" patternUnits="userSpaceOnUse">
  <image href="{canopy}" width="1300" height="1300"/>
  <animateTransform attributeName="patternTransform" type="translate" from="0 0" to="-1300 1300" dur="300s" repeatCount="indefinite"/></pattern>
{''.join(koi_defs)}
<clipPath id="frame"><rect width="{W}" height="{H}" rx="18"/></clipPath>
</defs>
<g clip-path="url(#frame)">
  <rect width="{W}" height="{H}" fill="#031009"/>
  <image href="{bed}" width="{W}" height="{H}"/>
  <rect width="{W}" height="{H}" fill="url(#sun)"/>
  <rect width="{W}" height="{H}" fill="url(#caustB)" opacity=".07" style="mix-blend-mode:screen"/>
  <g transform="translate(18,24)" filter="url(#shadowify)">{use(layers["deep"] + layers["top"])}</g>
  <g opacity=".82" filter="url(#depthBlur)">{use(layers["deep"])}</g>
  <rect width="{W}" height="{H}" fill="#123a2c" opacity=".2"/>
  {use(layers["top"])}
  <g style="mix-blend-mode:screen">
    <rect width="{W}" height="{H}" fill="url(#caustA)" opacity=".13"/>
    <rect width="{W}" height="{H}" fill="url(#caustB)" opacity=".10"/>
  </g>
  <rect width="{W}" height="{H}" fill="url(#murk)"/>
  {bk.ripple(520, 120, 0)}{bk.ripple(880, 330, 4.2)}{bk.ripple(300, 340, 7.1, rmax=38)}{bk.ripple(1000, 110, 2.6, rmax=34)}
  {pad(0, 80, 70, 150, 30, 2, 13, 0)}{pad(2, 175, 18, 86, 200, 3, 10, -3)}
  {pad(1, 1105, 330, 170, 250, 1.5, 14, -5)}{pad(0, 1010, 400, 96, 120, 2.5, 9, -2)}
  {pad(2, 1165, 225, 78, 160, 3, 8, -6)}{pad(1, 45, 385, 110, 300, 2, 12, -4)}
  {leaf(0, bk.loop_path(600, 200, 560, 190, k=5), 150, -10, 400)}
  {leaf(1, bk.loop_path(500, 260, 420, 150, k=5, ccw=True), 120, -40, -300, 28)}
  {leaf(2, bk.loop_path(800, 150, 380, 120, k=5), 100, -70, 260, 30)}
  {''.join(f'<image href="{im}" x="{f(x - w_ / 2 + 16)}" y="{f(y - h_ / 2 + 20)}" width="{w_}" height="{h_}" filter="url(#rockShadow)"/>' for x, y, w_, h_, im in rock_imgs)}
  {''.join(f'<image href="{im}" x="{f(x - w_ * .53)}" y="{f(y - h_ * .53)}" width="{f(w_ * 1.06)}" height="{f(h_ * 1.06)}" filter="url(#meniscus)"/>' for x, y, w_, h_, im in rock_imgs)}
  {''.join(f'<image href="{im}" x="{f(x - w_ / 2)}" y="{f(y - h_ / 2)}" width="{w_}" height="{h_}"/>' for x, y, w_, h_, im in rock_imgs)}
  <rect width="{W}" height="{H}" fill="url(#canopy)" opacity=".32"/>
  <rect width="{W}" height="{H}" fill="url(#canopy2)" opacity=".2"/>
  <rect width="{W}" height="{H}" fill="url(#sky)"/>
  <rect width="{W}" height="{H}" fill="url(#vignette)"/>
</g>
</svg>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg, encoding="utf-8", newline="\n")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    build()
