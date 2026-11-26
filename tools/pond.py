"""
The Living Pond: every koi is one of my GitHub projects.

  * variety  = project domain (pond.json)
  * size     = codebase volume (75%) + commits (25%), log-scaled, so a koi grows as its project grows
  * liveliness = faster tail beat if the project was pushed to in the last 14 days
  * new public repos not listed in pond.json hatch automatically as young koi
  * every koi circles above its own carved stepping stone (clickable, below the pond)

Run locally:   python tools/pond.py            (optional: GITHUB_TOKEN for higher rate limits)
Run nightly:   .github/workflows/pond.yml
"""
import json, math, os, re, sys, urllib.request, zlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
import build_koi_realistic as pond_render  # noqa: E402
import stones  # noqa: E402

W, H = 1200, 420
CONFIG = json.loads((ROOT / "pond.json").read_text(encoding="utf-8"))
USER = CONFIG["user"]
NOW = datetime.now(timezone.utc)


def api(url):
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "living-pond"})
    if os.environ.get("GITHUB_TOKEN"):
        req.add_header("Authorization", f"Bearer {os.environ['GITHUB_TOKEN']}")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read() or b"null"), r.headers
    except urllib.error.HTTPError as e:
        if e.code == 409:  # empty repository
            return [], {}
        raise


def commit_count(repo):
    data, headers = api(f"https://api.github.com/repos/{USER}/{repo}/commits?per_page=1")
    m = re.search(r'[?&]page=(\d+)>; rel="last"', headers.get("Link", "") if headers else "")
    return int(m.group(1)) if m else len(data or [])


def code_bytes(repo):
    langs, _ = api(f"https://api.github.com/repos/{USER}/{repo}/languages")
    # notebooks carry saved outputs, so they count at a quarter weight
    return sum(v * (0.25 if k == "Jupyter Notebook" else 1.0) for k, v in (langs or {}).items())


def days_since(ts):
    return (NOW - datetime.fromisoformat(ts.replace("Z", "+00:00"))).days


CACHE = ROOT / "tools" / ".pond-cache.json"   # local only (gitignored): lets previews run when rate-limited


def gather():
    try:
        fish = _gather_live()
        CACHE.write_text(json.dumps(fish, indent=1), encoding="utf-8")
        return fish
    except urllib.error.HTTPError as e:
        if e.code in (403, 429) and CACHE.exists() and not os.environ.get("CI"):
            print("GitHub API rate-limited; using cached project stats")
            cached = {f["name"]: f for f in json.loads(CACHE.read_text(encoding="utf-8"))}
            return [{**cached[f["name"]], **f} for f in CONFIG["fish"] if f["name"] in cached]
        raise


def _gather_live():
    repos, _ = api(f"https://api.github.com/users/{USER}/repos?per_page=100&type=owner")
    repos = {r["name"]: r for r in repos if not r["fork"] and not r["private"]}
    listed = {name for fish in CONFIG["fish"] for name in fish["repos"]}
    fish = [dict(f) for f in CONFIG["fish"] if any(r in repos for r in f["repos"])]
    for name in sorted(repos):                                   # new repos hatch automatically
        if name not in listed and name not in CONFIG["ignore"]:
            fish.append({"name": name, "domain": "new", "repos": [name], "about": "just hatched"})
    for f in fish:
        rs = [repos[r] for r in f["repos"] if r in repos]
        f["commits"] = sum(commit_count(r["name"]) for r in rs)
        f["code"] = int(sum(code_bytes(r["name"]) for r in rs))
        f["age_days"] = min(days_since(r["created_at"]) for r in rs)
        f["idle_days"] = min(days_since(r["pushed_at"]) for r in rs)
        f["url"] = rs[0]["html_url"]
        f.setdefault("about", rs[0].get("description") or "")
    return fish


def swim_path(cx, seed, ccw, cy=205, rx=200, ry=167):
    """Closed loop that starts at the top and reaches its lowest point exactly halfway round,
    right above the koi's stone, so the lingering stretch of the swim happens there."""
    import random
    jr = random.Random(seed)
    pts = []
    for j in range(8):
        a = math.radians(-90 + 45 * j) * (-1 if ccw else 1)
        m = 1 if j in (0, 4) else 1 + jr.uniform(-0.18, 0.18)
        pts.append((cx + math.cos(a) * rx * m, cy + math.sin(a) * ry * m))
    return pond_render.bk.closed_spline(pts)


def roster_from(fish):
    per_domain = {}
    for f in fish:                                             # variety: cycle through the domain's koi family
        kinds = CONFIG["domains"][f["domain"]]["kinds"]
        f["kind"] = kinds[per_domain.get(f["domain"], 0) % len(kinds)]
        per_domain[f["domain"]] = per_domain.get(f["domain"], 0) + 1
        code_g = min(max(math.log(max(f["code"], 1) / 50_000) / math.log(3_000_000 / 50_000), 0), 1)
        commit_g = min(math.log(f["commits"] + 1) / math.log(150), 1.0)
        s = 0.65 + 0.75 * (0.75 * code_g + 0.25 * commit_g)   # 0.65 (fry) .. 1.40 (mature)
        if f["domain"] == "new" and f["age_days"] < 30:        # auto-hatched repos grow in over a month
            s = min(s, 0.55 + 0.45 * f["age_days"] / 30)
        f["scale"] = round(s, 3)
        f["wag"] = 2.0 if f["idle_days"] <= 14 else 2.7
        f["seed"] = 100 + zlib.crc32(f["name"].encode()) % 9000

    n = len(fish)
    rx = max(150, min(380, W / n * 1.7))
    for i, f in enumerate(fish):                               # each koi circles above its own stone
        f["stone_x"] = W * (i + 0.5) / n
        f["path"] = swim_path(f["stone_x"], f["seed"], ccw=i % 2 == 1, rx=rx)
    by_size = sorted(fish, key=lambda f: -f["scale"])
    for rank, f in enumerate(by_size):
        f["layer"] = "top" if rank < max(4, round(n * 0.6)) else "deep"
    return [dict(id=f"koi{i}", kind=f["kind"], cx=f["stone_x"], cy=205, rx=rx, ry=167, path=f["path"], linger=True,
                 scale=f["scale"], wag=f["wag"], layer=f["layer"], seed=f["seed"]) for i, f in enumerate(fish)]


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def write_stones_and_readme(fish):
    n = len(fish)
    pct = math.floor(990 / n) / 10
    links = []
    for f in fish:
        label = f.get("label") or f["name"]
        path = stones.render_stone(label, f["seed"], ROOT / "assets" / "stones" / f"{slug(f['name'])}.png")
        tip = f'{f["name"]}: {f.get("about", "")}'.rstrip(": ").replace('"', "'")
        links.append(f'<a href="{f["url"]}" title="{tip}"><img src="assets/stones/{path.name}" width="{pct}%" alt="{label}"></a>')
    readme = f"""<p align="center">
  <img src="assets/koi-pond.svg" width="100%" alt="Animated koi pond. Every koi is one of my projects.">
</p>

<p align="center">{"".join(links)}</p>

<p align="center"><sub>Every koi is one of my projects and never strays far from its stone. Tap a stone to visit it.</sub></p>
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    fish = gather()
    for f in fish:
        print(f'{f["name"]:<34} {f["domain"]:<8} code={f["code"]:<8} commits={f["commits"]:<5} idle={f["idle_days"]}')
    pond_render.build(roster_from(fish), out=ROOT / "assets" / "koi-pond.svg", open_bottom=True)
    write_stones_and_readme(fish)
    (ROOT / "assets" / "pond-state.json").write_text(json.dumps(
        [{k: f[k] for k in ("name", "domain", "kind", "code", "commits", "scale", "wag")} for f in fish], indent=2) + "\n",
        encoding="utf-8", newline="\n")
