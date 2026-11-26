"""
Generates assets/koi-pond.svg: an original, self-contained animated koi pond banner
for a GitHub profile README. Pure SVG + SMIL (no JS, no external resources), so
it animates inside GitHub's <img> sandbox.

Run:  python tools/build_koi.py
"""
import base64, io, math, random
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parents[1] / "assets" / "koi-pond.svg"
W, H = 1200, 420
rng = random.Random(11)
f = lambda v: f"{v:.2f}".rstrip("0").rstrip(".")


# --------------------------------------------------------------------------
# Raster textures (embedded as data URIs)
# --------------------------------------------------------------------------
def tileable_caustics(n=384, pts=34, seed=3):
    """Tileable Worley (F2-F1) caustic network -> white RGBA with alpha = light."""
    r = np.random.default_rng(seed)
    p = r.random((pts, 2))
    yy, xx = np.mgrid[0:n, 0:n] / n
    # gentle domain warp (periodic so tiling stays seamless)
    wx = (xx + 0.055 * np.sin(2 * np.pi * (yy * 2 + xx)) + 0.025 * np.sin(2 * np.pi * (yy * 5 - xx * 3)))
    wy = (yy + 0.055 * np.sin(2 * np.pi * (xx * 2 - yy)) + 0.025 * np.sin(2 * np.pi * (xx * 4 + yy * 3)))
    d = []
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            dx = wx[..., None] - (p[:, 0] + ox)
            dy = wy[..., None] - (p[:, 1] + oy)
            d.append(np.sqrt(dx * dx + dy * dy))
    d = np.sort(np.concatenate(d, axis=-1), axis=-1)
    edge = d[..., 1] - d[..., 0]
    light = np.exp(-edge * 85.0) * 0.9 + np.exp(-edge * 22.0) * 0.18
    a8 = Image.fromarray((np.clip(light, 0, 1) * 255).astype(np.uint8))
    from PIL import ImageFilter
    a = np.asarray(a8.filter(ImageFilter.GaussianBlur(1.1)))
    img = np.dstack([np.full_like(a, 235), np.full_like(a, 252), np.full_like(a, 255), a])
    buf = io.BytesIO()
    Image.fromarray(img, "RGBA").save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def pond_floor(w=600, h=210, seed=5):
    """Soft, out-of-focus pond bed (stones/silt), returned as JPEG."""
    r = np.random.default_rng(seed)
    def smooth_noise(scale):
        g = r.random((h // scale + 2, w // scale + 2))
        im = Image.fromarray((g * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
        return np.asarray(im, dtype=float) / 255
    n = 0.55 * smooth_noise(40) + 0.3 * smooth_noise(14) + 0.15 * smooth_noise(5)
    # pebbles: soft blobs
    yy, xx = np.mgrid[0:h, 0:w]
    peb = np.zeros((h, w))
    for _ in range(160):
        cx, cy, rad = r.random() * w, r.random() * h, 3 + r.random() * 9
        peb += np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * rad ** 2))) * (0.25 + r.random() * 0.5)
    n = np.clip(0.75 * n + 0.35 * np.clip(peb, 0, 1), 0, 1)
    deep = np.array([4, 18, 24]); mid = np.array([10, 44, 52]); hi = np.array([30, 78, 80])
    t = n[..., None]
    col = np.where(t < 0.5, deep + (mid - deep) * (t / 0.5), mid + (hi - mid) * ((t - 0.5) / 0.5))
    im = Image.fromarray(col.astype(np.uint8), "RGB")
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def closed_spline(points, tension=1.0):
    """Closed Catmull-Rom spline -> SVG cubic path."""
    n = len(points)
    d = f"M{f(points[0][0])},{f(points[0][1])}"
    for i in range(n):
        p0, p1, p2, p3 = (points[(i - 1) % n], points[i], points[(i + 1) % n], points[(i + 2) % n])
        c1 = (p1[0] + (p2[0] - p0[0]) / 6 * tension, p1[1] + (p2[1] - p0[1]) / 6 * tension)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6 * tension, p2[1] - (p3[1] - p1[1]) / 6 * tension)
        d += f" C{f(c1[0])},{f(c1[1])} {f(c2[0])},{f(c2[1])} {f(p2[0])},{f(p2[1])}"
    return d + "Z"


def blob(cx, cy, rx, ry, k=9, jitter=0.28, rot=0.0):
    pts = []
    for i in range(k):
        a = 2 * math.pi * i / k
        m = 1 + rng.uniform(-jitter, jitter)
        x, y = math.cos(a) * rx * m, math.sin(a) * ry * m
        pts.append((cx + x * math.cos(rot) - y * math.sin(rot), cy + x * math.sin(rot) + y * math.cos(rot)))
    return closed_spline(pts)


def loop_path(cx, cy, rx, ry, k=6, jit=0.22, ccw=False):
    pts = []
    for i in range(k):
        a = 2 * math.pi * i / k * (-1 if ccw else 1) + rng.uniform(-0.15, 0.15)
        m = 1 + rng.uniform(-jit, jit)
        pts.append((cx + math.cos(a) * rx * m, cy + math.sin(a) * ry * m))
    return closed_spline(pts)


def path_length(d_points):
    return sum(math.dist(d_points[i], d_points[(i + 1) % len(d_points)]) for i in range(len(d_points)))


# --------------------------------------------------------------------------
# Koi anatomy (local coords: nose at +x, length ~120)
# --------------------------------------------------------------------------
FRONT = "M50,0 C50,-9.5 41,-16.5 26,-17.5 C15,-18 6,-16.6 0,-15.6 L0,15.6 C6,16.6 15,18 26,17.5 C41,16.5 50,9.5 50,0Z"
MID = "M7,-15.8 C-6,-15.2 -18,-12.2 -30,-7.8 L-30,7.8 C-18,12.2 -6,15.2 7,15.8Z"
PED = "M-20,-10.6 C-28,-9 -35,-6 -41,-3.8 C-43,-3.2 -45,-2.9 -47,-2.8 L-47,2.8 C-45,2.9 -43,3.2 -41,3.8 C-35,6 -28,9 -20,10.6Z"
CAUDAL = ("M-44,0 C-49,-4 -55,-12 -71,-22 C-73,-13 -67,-6 -62,0 "
          "C-67,6 -73,13 -71,22 C-55,12 -49,4 -44,0Z")
PECT_L = "M32,-13 C31,-21 25,-30 14,-34 C8,-35.5 5,-31.5 8.5,-26.5 C12.5,-20.5 19,-15.5 25,-12Z"
PELV_L = "M-5,-12.5 C-7,-18 -12,-22 -18,-23 C-21,-23 -21,-20 -18,-17.5 C-15,-15 -12,-13 -10,-11.5Z"
import re as _re
SLIM = 0.88
ys = lambda d: _re.sub(r"(-?\d+\.?\d*),(-?\d+\.?\d*)", lambda m: f"{m.group(1)},{f(float(m.group(2)) * SLIM)}", d)
mirror = lambda d: __import__("re").sub(r"(-?\d+\.?\d*),(-?\d+\.?\d*)",
                                        lambda m: f"{m.group(1)},{f(-float(m.group(2)))}", d)
PECT_R, PELV_R = mirror(PECT_L), mirror(PELV_L)
FIN_RAYS = "M-46,0 L-70,-19 M-46,0 L-67,-10 M-46,0 L-67,10 M-46,0 L-70,19"
FRONT, MID, PED, CAUDAL, PECT_L, PELV_L, PECT_R, PELV_R = map(ys, (FRONT, MID, PED, CAUDAL, PECT_L, PELV_L, PECT_R, PELV_R))
CAUDAL = ("M-44,0 C-50,-4 -58,-10 -78,-19 C-76,-10 -70,-5 -64,0 "
          "C-70,5 -76,10 -78,19 C-58,10 -50,4 -44,0Z")

VARIETIES = {
    # base, fin fill, patches: list of (kind, color)
    "kohaku": dict(base="#f3efe6", fin="rgba(248,243,233,.55)"),
    "tancho": dict(base="#f4f0e8", fin="rgba(248,243,233,.55)"),
    "showa":  dict(base="#16161a", fin="rgba(40,40,46,.75)"),
    "sanke":  dict(base="#f2eee5", fin="rgba(248,243,233,.55)"),
    "ogon":   dict(base="url(#gold)", fin="rgba(246,205,120,.5)"),
    "asagi":  dict(base="#7d97a6", fin="rgba(214,224,230,.5)"),
}
RED, RED2, INK, WHITE = "#d63b22", "#c5321d", "#141417", "#f1ede4"


def patches_for(kind):
    out = []
    if kind == "kohaku":
        for x in (34, 12, -12):
            out.append((blob(x + rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(9, 13), rng.uniform(13, 19)), RED))
        out.append((blob(-30, 0, 6, 7), RED2))
    elif kind == "tancho":
        out.append((blob(37, 0, 6.5, 6.5, jitter=0.06), RED))
    elif kind == "sanke":
        for x in (30, 4, -22):
            out.append((blob(x, rng.uniform(-4, 4), rng.uniform(9, 12), rng.uniform(11, 17)), RED))
        for _ in range(7):
            out.append((blob(rng.uniform(-30, 26), rng.uniform(-12, 12), rng.uniform(2, 4.5), rng.uniform(2, 4)), INK))
    elif kind == "showa":
        for x in (36, 8, -18):
            out.append((blob(x, rng.uniform(-3, 3), rng.uniform(9, 13), rng.uniform(12, 18)), RED))
        for x in (22, -4, -30):
            out.append((blob(x, rng.uniform(-6, 6), rng.uniform(5, 8), rng.uniform(7, 11)), WHITE))
    elif kind == "asagi":
        for s in (-1, 1):
            out.append((blob(10, s * 15.5, 30, 4.5, jitter=0.18), "#e0602f"))
            out.append((blob(-24, s * 8.5, 10, 3, jitter=0.2), "#e0602f"))
        out.append((blob(43, 0, 7, 9, jitter=0.1), "#e8e4da"))
    return out


def koi(uid, kind, path_d, dur, scale, wag, delay):
    v = VARIETIES[kind]
    pats = [(ys(d), c) for d, c in patches_for(kind)]
    ph = lambda frac: f(-wag * frac - delay)  # begin offsets give a travelling wave

    def seg_patches(clip):
        if not pats:
            return ""
        inner = "".join(f'<path d="{d}" fill="{c}"/>' for d, c in pats)
        return f'<g clip-path="url(#{clip})">{inner}</g>'

    def rot(amp, frac, cx):
        return (f'<animateTransform attributeName="transform" type="rotate" dur="{f(wag)}s" begin="{ph(frac)}s" '
                f'repeatCount="indefinite" calcMode="spline" keyTimes="0;.5;1" '
                f'keySplines=".45 0 .55 1;.45 0 .55 1" values="{-amp} {cx} 0;{amp} {cx} 0;{-amp} {cx} 0"/>')

    def fin_flap(amp, frac, cx, cy):
        return (f'<animateTransform attributeName="transform" type="rotate" dur="{f(wag * 1.3)}s" begin="{ph(frac)}s" '
                f'repeatCount="indefinite" calcMode="spline" keyTimes="0;.5;1" '
                f'keySplines=".45 0 .55 1;.45 0 .55 1" values="{-amp} {cx} {cy};{amp} {cx} {cy};{-amp} {cx} {cy}"/>')

    scales = '<rect x="-30" y="-18" width="64" height="36" fill="url(#scales)"/>' if kind != "ogon" else \
             '<rect x="-30" y="-18" width="64" height="36" fill="url(#scalesGold)"/>'
    if kind == "asagi":
        scales = '<rect x="-30" y="-18" width="64" height="36" fill="url(#net)"/>'
    spine = '<path d="M28,0 C10,0 -10,0 -26,0" stroke="rgba(0,0,0,.14)" stroke-width="1.6" fill="none" stroke-linecap="round"/>'
    gloss = '<path d="M44,-3 C34,-7 18,-8 4,-6" stroke="rgba(255,255,255,.28)" stroke-width="2" fill="none" stroke-linecap="round"/>'

    tail = (f'<g>{rot(11, 0.5, -27)}'
            f'<g>{rot(9, 0.85, -45)}<path d="{CAUDAL}" fill="url(#fin-{kind})" stroke="rgba(255,255,255,.22)" stroke-width=".5"/>'
            f'<path d="{FIN_RAYS}" stroke="rgba(255,255,255,.16)" stroke-width=".45" fill="none"/></g>'
            f'<path d="{PED}" fill="{v["base"]}"/>{seg_patches("kc-ped")}'
            f'<path d="{PED}" fill="url(#shade)"/></g>')
    mid = (f'<g>{rot(5, 0.25, 4)}'
           f'<g>{fin_flap(10, 0.3, -8, -12)}<path d="{PELV_L}" fill="{v["fin"]}"/></g>'
           f'<g>{fin_flap(10, 0.8, -8, 12)}<path d="{PELV_R}" fill="{v["fin"]}"/></g>'
           f'{tail}'
           f'<g mask="url(#kf-mid)"><path d="{MID}" fill="{v["base"]}"/>{seg_patches("kc-mid")}'
           f'<g clip-path="url(#kc-mid)">{scales}</g>'
           f'<path d="{MID}" fill="url(#shade)"/></g></g>')
    head = (f'<g>{fin_flap(14, 0.0, 27, -14)}<path d="{PECT_L}" fill="{v["fin"]}"/></g>'
            f'<g>{fin_flap(14, 0.5, 27, 14)}<path d="{PECT_R}" fill="{v["fin"]}"/></g>'
            f'{mid}'
            f'<g mask="url(#kf-front)"><path d="{FRONT}" fill="{v["base"]}"/>{seg_patches("kc-front")}'
            f'<g clip-path="url(#kc-front)"><g clip-path="url(#kc-scaleZone)">{scales}</g></g>'
            f'<path d="{FRONT}" fill="url(#shade)"/>'
            f'<path d="{FRONT}" fill="url(#headShade)"/></g>'
            f'{spine}{gloss}'
            f'<circle cx="41.5" cy="-7" r="1.5" fill="#0c0c0e"/><circle cx="41.5" cy="7" r="1.5" fill="#0c0c0e"/>'
            f'<path d="M49,-3 l3.5,-3 M49,3 l3.5,3" stroke="rgba(255,255,255,.35)" stroke-width=".6"/>')
    return (f'<g id="{uid}">'
            f'<animateMotion dur="{f(dur)}s" begin="{f(-delay * 7.3)}s" repeatCount="indefinite" rotate="auto" path="{path_d}"/>'
            f'<g transform="scale({f(scale)})">{head}</g></g>')


# --------------------------------------------------------------------------
# Scene pieces
# --------------------------------------------------------------------------
def lily_pad(cx, cy, r, rot, notch=24, sway=2.0, dur=11, begin=0.0, flower=False):
    a0, a1 = math.radians(rot + notch / 2), math.radians(rot - notch / 2 + 360)
    x0, y0 = cx + r * math.cos(a0), cy + r * math.sin(a0)
    x1, y1 = cx + r * math.cos(a1), cy + r * math.sin(a1)
    d = f"M{f(cx)},{f(cy)} L{f(x0)},{f(y0)} A{f(r)},{f(r)} 0 1 1 {f(x1)},{f(y1)}Z"
    veins = ""
    for i in range(13):
        a = math.radians(rot + notch / 2 + 8 + i * (360 - notch - 16) / 12)
        veins += f"M{f(cx)},{f(cy)} L{f(cx + r * .93 * math.cos(a))},{f(cy + r * .93 * math.sin(a))} "
    fl = lotus(cx - r * .15, cy - r * .1, r * .55) if flower else ""
    return (f'<ellipse cx="{f(cx + 12)}" cy="{f(cy + 16)}" rx="{f(r * 1.05)}" ry="{f(r * 1.0)}" fill="url(#padShadow)"/>'
            f'<g><animateTransform attributeName="transform" type="rotate" dur="{dur}s" begin="{begin}s" '
            f'repeatCount="indefinite" calcMode="spline" keyTimes="0;.5;1" keySplines=".45 0 .55 1;.45 0 .55 1" '
            f'values="{-sway} {f(cx)} {f(cy)};{sway} {f(cx)} {f(cy)};{-sway} {f(cx)} {f(cy)}"/>'
            f'<path d="{d}" fill="url(#pad)"/>'
            f'<path d="{veins}" stroke="rgba(190,230,170,.14)" stroke-width="1" fill="none"/>'
            f'<path d="{d}" fill="none" stroke="rgba(110,60,45,.55)" stroke-width="2.2"/>'
            f'<path d="{d}" fill="none" stroke="rgba(200,235,170,.22)" stroke-width=".8"/>'
            f'<path d="{d}" fill="url(#padSheen)"/>{fl}</g>')


def lotus(cx, cy, s):
    out = ""
    for layer, (n, length, width, col) in enumerate([(10, 1.0, .34, "url(#petalOuter)"),
                                                       (8, .78, .3, "url(#petalMid)"),
                                                       (6, .52, .26, "url(#petalInner)")]):
        for i in range(n):
            a = 360 * i / n + layer * 17
            L, Wd = s * length, s * width
            p = f"M0,0 C{f(L * .35)},{f(-Wd)} {f(L * .85)},{f(-Wd * .7)} {f(L)},0 C{f(L * .85)},{f(Wd * .7)} {f(L * .35)},{f(Wd)} 0,0Z"
            out += f'<path d="{p}" fill="{col}" transform="translate({f(cx)},{f(cy)}) rotate({f(a)})" stroke="rgba(120,30,60,.18)" stroke-width=".6"/>'
    out += f'<circle cx="{f(cx)}" cy="{f(cy)}" r="{f(s * .16)}" fill="#f2c94c"/>'
    for i in range(14):
        a = math.radians(i * 360 / 14)
        out += f'<circle cx="{f(cx + math.cos(a) * s * .22)}" cy="{f(cy + math.sin(a) * s * .22)}" r="{f(s * .03)}" fill="#f7dc7b"/>'
    return out


def ripple(cx, cy, begin, dur=10, rmax=46):
    rings = ""
    for k, lag in enumerate((0, .35, .7)):
        rings += (f'<circle cx="{cx}" cy="{cy}" r="1" fill="none" stroke="rgba(220,245,250,.55)" stroke-width="1.4" opacity="0">'
                  f'<animate attributeName="r" values="1;{rmax - k * 9};{rmax - k * 9}" keyTimes="0;.32;1" dur="{dur}s" begin="{f(begin + lag)}s" repeatCount="indefinite"/>'
                  f'<animate attributeName="opacity" values="0;.85;0;0" keyTimes="0;.02;.32;1" dur="{dur}s" begin="{f(begin + lag)}s" repeatCount="indefinite"/>'
                  f'<animate attributeName="stroke-width" values="1.6;.3;.3" keyTimes="0;.32;1" dur="{dur}s" begin="{f(begin + lag)}s" repeatCount="indefinite"/>'
                  f'</circle>')
    return rings


def petal(path_d, dur, begin, spin, s=1.0):
    p = f"M0,0 C3,-4 9,-4 12,0 C9,4 3,4 0,0Z"
    return (f'<g><animateMotion dur="{dur}s" begin="{begin}s" repeatCount="indefinite" path="{path_d}"/>'
            f'<g transform="scale({f(s)})"><g><animateTransform attributeName="transform" type="rotate" from="0" to="{spin}" dur="{dur}s" begin="{begin}s" repeatCount="indefinite"/>'
            f'<path d="{p}" transform="translate(2,3)" fill="rgba(0,0,0,.25)"/>'
            f'<path d="{p}" fill="url(#petalMid)"/></g></g></g>')


# --------------------------------------------------------------------------
# Assemble
# --------------------------------------------------------------------------
def build():
    caustic = tileable_caustics()
    floor = pond_floor()

    fin_tint = dict(kohaku="248,243,233", tancho="248,243,233", sanke="248,243,233",
                    showa="70,70,78", ogon="250,214,135", asagi="220,230,236")
    fin_grads = "".join(
        f'<linearGradient id="fin-{k}" gradientUnits="userSpaceOnUse" x1="-44" y1="0" x2="-78" y2="0">'
        f'<stop offset="0" stop-color="rgb({c})" stop-opacity=".85"/><stop offset="1" stop-color="rgb({c})" stop-opacity=".18"/></linearGradient>'
        for k, c in fin_tint.items())
    defs = f"""
<linearGradient id="shade" gradientUnits="userSpaceOnUse" x1="0" y1="-14.5" x2="0" y2="14.5">
  <stop offset="0" stop-color="#000" stop-opacity=".42"/><stop offset=".26" stop-color="#000" stop-opacity="0"/>
  <stop offset=".46" stop-color="#fff" stop-opacity=".16"/><stop offset=".56" stop-color="#fff" stop-opacity=".05"/>
  <stop offset=".74" stop-color="#000" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity=".42"/>
</linearGradient>
<radialGradient id="headShade" gradientUnits="userSpaceOnUse" cx="40" cy="0" r="16">
  <stop offset="0" stop-color="#fff" stop-opacity=".10"/><stop offset="1" stop-color="#fff" stop-opacity="0"/>
</radialGradient>
<linearGradient id="gold" gradientUnits="userSpaceOnUse" x1="0" y1="-18" x2="0" y2="18">
  <stop offset="0" stop-color="#b9822a"/><stop offset=".45" stop-color="#ffe29a"/><stop offset=".6" stop-color="#f4c766"/><stop offset="1" stop-color="#b37a22"/>
</linearGradient>
<pattern id="scales" width="5" height="4.2" patternUnits="userSpaceOnUse">
  <path d="M0,4.2 A2.5,2.5 0 0 1 5,4.2 M-2.5,2.1 A2.5,2.5 0 0 1 2.5,2.1 M2.5,2.1 A2.5,2.5 0 0 1 7.5,2.1" fill="none" stroke="#000" stroke-opacity=".055" stroke-width=".4"/>
</pattern>
<pattern id="scalesGold" width="5" height="4.2" patternUnits="userSpaceOnUse">
  <path d="M0,4.2 A2.5,2.5 0 0 1 5,4.2 M-2.5,2.1 A2.5,2.5 0 0 1 2.5,2.1 M2.5,2.1 A2.5,2.5 0 0 1 7.5,2.1" fill="none" stroke="#fff6d8" stroke-opacity=".35" stroke-width=".5"/>
</pattern>
<pattern id="net" width="5" height="4.2" patternUnits="userSpaceOnUse">
  <path d="M0,4.2 A2.5,2.5 0 0 1 5,4.2 M-2.5,2.1 A2.5,2.5 0 0 1 2.5,2.1 M2.5,2.1 A2.5,2.5 0 0 1 7.5,2.1" fill="none" stroke="#dfe9ee" stroke-opacity=".55" stroke-width=".55"/>
</pattern>
<linearGradient id="kfg-front" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="6" y2="0"><stop offset="0" stop-color="#fff" stop-opacity="0"/><stop offset="1" stop-color="#fff"/></linearGradient>
<linearGradient id="kfg-mid" gradientUnits="userSpaceOnUse" x1="-30" y1="0" x2="-23" y2="0"><stop offset="0" stop-color="#fff" stop-opacity="0"/><stop offset="1" stop-color="#fff"/></linearGradient>
<mask id="kf-front" maskUnits="userSpaceOnUse" x="-5" y="-25" width="65" height="50"><rect x="-5" y="-25" width="65" height="50" fill="url(#kfg-front)"/></mask>
<mask id="kf-mid" maskUnits="userSpaceOnUse" x="-35" y="-25" width="50" height="50"><rect x="-35" y="-25" width="50" height="50" fill="url(#kfg-mid)"/></mask>
<clipPath id="kc-front"><path d="{FRONT}"/></clipPath>
<clipPath id="kc-mid"><path d="{MID}"/></clipPath>
<clipPath id="kc-ped"><path d="{PED}"/></clipPath>
<clipPath id="kc-scaleZone"><path d="M33,-18 L0,-18 L0,18 L33,18Z M0,0"/></clipPath>
{fin_grads}
<radialGradient id="pad" cx=".42" cy=".4" r=".7">
  <stop offset="0" stop-color="#5f9a4f"/><stop offset=".55" stop-color="#3a6f3a"/><stop offset=".9" stop-color="#244a2a"/><stop offset="1" stop-color="#5a3a2e"/>
</radialGradient>
<radialGradient id="padSheen" cx=".3" cy=".25" r=".6">
  <stop offset="0" stop-color="#fff" stop-opacity=".16"/><stop offset="1" stop-color="#fff" stop-opacity="0"/>
</radialGradient>
<radialGradient id="padShadow"><stop offset=".55" stop-color="#000" stop-opacity=".38"/><stop offset="1" stop-color="#000" stop-opacity="0"/></radialGradient>
<linearGradient id="petalOuter" x1="0" x2="1"><stop offset="0" stop-color="#f7d6e2"/><stop offset="1" stop-color="#e46f98"/></linearGradient>
<linearGradient id="petalMid" x1="0" x2="1"><stop offset="0" stop-color="#fbe4ec"/><stop offset="1" stop-color="#ee8db0"/></linearGradient>
<linearGradient id="petalInner" x1="0" x2="1"><stop offset="0" stop-color="#fff3f7"/><stop offset="1" stop-color="#f4a9c4"/></linearGradient>
<radialGradient id="light" cx=".3" cy=".15" r=".8">
  <stop offset="0" stop-color="#3fa3a6" stop-opacity=".22"/><stop offset="1" stop-color="#3fa3a6" stop-opacity="0"/>
</radialGradient>
<radialGradient id="lightMaskG" cx=".32" cy=".2" r=".85">
  <stop offset="0" stop-color="#fff"/><stop offset="1" stop-color="#fff" stop-opacity=".25"/>
</radialGradient>
<mask id="lightMask" maskUnits="userSpaceOnUse" x="0" y="0" width="{W}" height="{H}"><rect width="{W}" height="{H}" fill="url(#lightMaskG)"/></mask>
<radialGradient id="vignette" cx=".5" cy=".5" r=".75">
  <stop offset=".55" stop-color="#000" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity=".62"/>
</radialGradient>
<linearGradient id="depth" x1="0" y1="0" x2="1" y2="1">
  <stop offset="0" stop-color="#0d4a52" stop-opacity=".0"/><stop offset="1" stop-color="#021116" stop-opacity=".45"/>
</linearGradient>
<linearGradient id="ray" x1="0" y1="0" x2="1" y2="0">
  <stop offset="0" stop-color="#cff6ff" stop-opacity="0"/><stop offset=".5" stop-color="#cff6ff" stop-opacity=".06"/><stop offset="1" stop-color="#cff6ff" stop-opacity="0"/>
</linearGradient>
<filter id="shadowify" x="-10%" y="-10%" width="120%" height="120%" color-interpolation-filters="sRGB">
  <feColorMatrix type="matrix" values="0 0 0 0 0  0 0 0 0 .03  0 0 0 0 .05  0 0 0 .55 0"/>
  <feGaussianBlur stdDeviation="5"/>
</filter>
<pattern id="caustA" width="576" height="576" patternUnits="userSpaceOnUse">
  <image href="data:image/png;base64,{caustic}" width="576" height="576"/>
  <animateTransform attributeName="patternTransform" type="translate" from="0 0" to="576 576" dur="90s" repeatCount="indefinite"/>
</pattern>
<pattern id="caustB" width="768" height="768" patternUnits="userSpaceOnUse">
  <image href="data:image/png;base64,{caustic}" width="768" height="768"/>
  <animateTransform attributeName="patternTransform" type="translate" from="0 0" to="-768 768" dur="120s" repeatCount="indefinite"/>
</pattern>
"""

    # koi: (variety, loop centre, radii, scale, wag period, depth layer)
    roster = [
        ("kohaku", (330, 215), (360, 165), 1.55, 2.6, "top"),
        ("ogon",   (860, 205), (360, 150), 1.4, 2.4, "top"),
        ("showa",  (620, 240), (560, 175), 1.6, 2.8, "top"),
        ("tancho", (930, 250), (300, 150), 1.3, 2.3, "top"),
        ("sanke",  (470, 180), (430, 140), 1.05, 2.4, "deep"),
        ("asagi",  (760, 270), (470, 130), 1.0, 2.6, "deep"),
    ]
    koi_defs, layers = [], {"top": [], "deep": []}
    for i, (kind, (cx, cy), (rx, ry), s, wag, layer) in enumerate(roster):
        pts_d = loop_path(cx, cy, rx, ry, k=6, ccw=(i % 2 == 1))
        circumference = math.pi * (3 * (rx + ry) - math.sqrt((3 * rx + ry) * (rx + 3 * ry)))
        speed = rng.uniform(24, 30) * (s ** .5)
        dur = circumference / speed
        uid = f"koi{i}"
        koi_defs.append(koi(uid, kind, pts_d, dur, s, wag, delay=rng.uniform(0, 9)))
        layers[layer].append(uid)

    defs += "".join(koi_defs)

    use = lambda ids: "".join(f'<use href="#{u}"/>' for u in ids)

    rays = ""
    for i, (x, w_, op_dur) in enumerate([(140, 120, 13), (420, 80, 17), (760, 140, 15), (1010, 90, 19)]):
        rays += (f'<path d="M{x},-20 l{w_},0 l{w_ * 1.8},460 l-{w_},0Z" fill="url(#ray)" opacity=".6">'
                 f'<animate attributeName="opacity" values=".15;.75;.15" dur="{op_dur}s" begin="-{i * 3}s" repeatCount="indefinite"/></path>')

    pads = (lily_pad(70, 60, 58, 35, sway=1.6, dur=13)
            + lily_pad(150, 20, 34, 200, sway=2.5, dur=10, begin=-3)
            + lily_pad(1110, 340, 66, 250, sway=1.4, dur=14, begin=-5, flower=True)
            + lily_pad(1020, 395, 38, 120, sway=2.2, dur=9, begin=-2)
            + lily_pad(1165, 230, 30, 160, sway=2.8, dur=8, begin=-6)
            + lily_pad(40, 385, 44, 300, sway=2.0, dur=12, begin=-4))

    petals = (petal(loop_path(600, 200, 560, 190, k=5), 140, -10, 540, 1.1)
              + petal(loop_path(500, 260, 420, 150, k=5, ccw=True), 110, -40, -400, .9)
              + petal(loop_path(800, 150, 380, 120, k=5), 95, -70, 300, .8))

    ripples = ripple(520, 120, 0) + ripple(880, 330, 4.2) + ripple(300, 340, 7.1, rmax=38) + ripple(1000, 110, 2.6, rmax=34)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Animated koi pond">
<title>Koi pond</title>
<defs>{defs}<clipPath id="frame"><rect width="{W}" height="{H}" rx="18"/></clipPath></defs>
<g clip-path="url(#frame)">
  <rect width="{W}" height="{H}" fill="#04161c"/>
  <image href="data:image/jpeg;base64,{floor}" width="{W}" height="{H}" preserveAspectRatio="xMidYMid slice" opacity=".75"/>
  <rect width="{W}" height="{H}" fill="url(#light)"/>
  <rect width="{W}" height="{H}" fill="url(#caustB)" opacity=".10"/>
  <g transform="translate(16,22)" filter="url(#shadowify)">{use(layers["deep"] + layers["top"])}</g>
  <g opacity=".86">{use(layers["deep"])}</g>
  <rect width="{W}" height="{H}" fill="#0a3640" opacity=".28"/>
  {use(layers["top"])}
  <g style="mix-blend-mode:screen">
    <rect width="{W}" height="{H}" fill="url(#caustA)" opacity=".16" mask="url(#lightMask)"/>
    <rect width="{W}" height="{H}" fill="url(#caustB)" opacity=".11" mask="url(#lightMask)"/>
  </g>
  {rays}
  <rect width="{W}" height="{H}" fill="url(#depth)"/>
  {ripples}
  {pads}
  {petals}
  <rect width="{W}" height="{H}" fill="url(#vignette)"/>
</g>
</svg>"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(svg, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    build()
