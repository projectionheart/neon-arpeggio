"""Neon Arpeggio: an 8-bit point-and-click music video, rendered entirely from code.

    python render.py                       # 1080p music video -> out/neon_arpeggio.mp4
    python render.py --clean --scale 2     # no baked-in cursor/verb bar (used by the web player)
    python render.py --stills 5 22 43 60   # PNG stills at those timestamps

Every frame is drawn at 320x180 with a fixed 32-colour palette, then
upscaled with nearest-neighbour so the pixels stay crisp.

Scene timeline (times in seconds, locked to the song's sections):
  0.0  - 14.6  painted rainy skyline, camera pans to the BAR sign
  14.6 - 30.0  storefront of Caffè Neon, punks out front
  30.0 - 53.3  inside the bar; Volpe the hacker-barista pours espresso,
               the trip starts creeping in at 42.2
  53.3 - 56.4  breakdown: frame freezes into a single cup
  56.4 - 67.2  the trip: plasma sky, teacup moon, title card
"""
import argparse, math, os, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import analyze

HERE = os.path.dirname(os.path.abspath(__file__))
W, H, FPS = 320, 180, 30

# Song-specific timing. analyze.py estimates BPM/phase for a new song;
# the scene cut times below are hand-placed on this song's sections.
SONG = dict(path=os.path.join(HERE, 'assets', 'song.m4a'), bpm=140.82, phase=0.135, duration=67.24)
CUTS = [14.6, 30.0, 53.3, 56.4]

BPM, PH = SONG['bpm'], SONG['phase']
BP = 60 / BPM
DUR = SONG['duration']
NF = int(DUR * FPS)
rng = np.random.default_rng(7)

# ---------------------------------------------------------------- palette
C = dict(
    BLK=(8, 6, 18), NAVY=(16, 12, 38), DPUR=(34, 20, 66), PUR=(62, 30, 96), PLUM=(108, 32, 104),
    MAG=(255, 46, 136), PINK=(255, 120, 200), LPINK=(255, 190, 230), CYAN=(45, 226, 230),
    TEAL=(16, 110, 128), DTEAL=(10, 50, 64), BLUE=(40, 70, 170), SKY=(90, 140, 255),
    ORNG=(255, 138, 61), AMBR=(255, 200, 87), RED=(220, 40, 60), DRED=(110, 20, 40),
    GRN=(140, 255, 107), DGRN=(30, 120, 70), WHT=(240, 236, 255), GRY=(120, 120, 150),
    DGRY=(60, 58, 84), SKIN=(214, 150, 120), SKIN2=(150, 96, 80), BRN=(90, 50, 40),
    YEL=(255, 240, 140), LCYN=(170, 250, 255), VIO=(150, 80, 255), BLD=(24, 16, 48), BLD2=(14, 10, 30))
palimg = Image.new('P', (1, 1))
_flat = [c for v in C.values() for c in v] + list(C['BLK']) * (256 - len(C))
palimg.putpalette(_flat)

# 4x4 Bayer matrix: the ordered-dither threshold used for every gradient, fog and fade
BAY = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) / 16
YY, XX = np.mgrid[0:H, 0:W]
BAYF = BAY[YY % 4, XX % 4]

FONT = ImageFont.truetype(os.path.join(HERE, 'assets', 'unifont-latin.otf'), 16)


def ease(x):
    x = min(max(x, 0), 1)
    return x * x * (3 - 2 * x)


def lerp(a, b, x):
    return a + (b - a) * x


def dgrad(h, w, cols, y0=0):
    """Vertical gradient through a list of palette colours, Bayer-dithered."""
    yy, xx = np.mgrid[0:h, 0:w]
    v = (yy / max(h - 1, 1)) * (len(cols) - 1)
    b = BAY[(yy + y0) % 4, xx % 4]
    idx = np.clip(np.floor(v + b), 0, len(cols) - 1).astype(int)
    return np.array(cols, np.uint8)[idx]


# ---------------------------------------------------------------- audio features
FEAT = analyze.features(SONG['path'])
tt = np.arange(NF) / FPS


def _env(x, q=95):
    return np.clip(x / np.percentile(x, q), 0, 1.5)


BASS = np.interp(tt, FEAT['t'], _env(FEAT['bass']))
HI = np.interp(tt, FEAT['t'], _env(FEAT['hi']))
ONS = np.interp(tt, FEAT['t'], _env(FEAT['onset'], 98))


def beat(t):
    b = (t - PH) / BP
    return int(math.floor(b)), b % 1


# ---------------------------------------------------------------- tileable value noise
def tilenoise(S, octaves, seed):
    r = np.random.default_rng(seed)
    acc = np.zeros((S, S), np.float32); amp = 1; tot = 0
    for g in octaves:
        big = np.tile(r.random((g, g)).astype(np.float32), (3, 3))
        im = Image.fromarray((big * 255).astype(np.uint8)).resize((S * 3, S * 3), Image.BICUBIC)
        acc += np.asarray(im, np.float32)[S:2 * S, S:2 * S] / 255 * amp
        tot += amp; amp *= 0.55
    acc /= tot
    return (acc - acc.min()) / (acc.max() - acc.min())


S = 512
N1 = tilenoise(S, [4, 8, 16, 32], 1)
N2 = tilenoise(S, [3, 6, 12], 2)
N3 = tilenoise(S, [8, 16, 48], 3)


def samp(N, x, y):
    return N[y.astype(np.int32) % S, x.astype(np.int32) % S]


# ---------------------------------------------------------------- text + point-and-click UI
CLEAN = False  # --clean: no cursor / verb bar (the web player draws its own)


def txt(dr, x, y, s, col=C['WHT'], sh=C['BLK']):
    if sh: dr.text((x + 1, y + 1), s, font=FONT, fill=sh)
    dr.text((x, y), s, font=FONT, fill=col)


def tlen(s):
    return int(FONT.getlength(s))


def dialog(dr, t, t0, t1, who, lines, col, ax=None):
    """Typewriter dialog box; lines reveal at 26 characters per second."""
    if not (t0 <= t < t1): return
    if isinstance(lines, str): lines = [lines]
    n = max(0, int((t - t0) * 26))
    w = max(tlen(l) for l in lines + [who]) + 8
    h = 16 * (len(lines) + 1) + 3
    x = 160 - w // 2 if ax is None else int(min(max(ax - w // 2, 2), W - w - 2))
    y = 3
    dr.rectangle([x, y, x + w, y + h], fill=C['BLK'], outline=col)
    dr.rectangle([x + 1, y + 1, x + w - 1, y + h - 1], outline=C['DPUR'])
    txt(dr, x + 4, y + 1, who, col, None)
    for k, l in enumerate(lines):
        txt(dr, x + 4, y + 17 + 16 * k, l[:max(0, n)], C['WHT'], None)
        n -= len(l)
    if ax is not None:
        dr.polygon([(ax - 3, y + h + 1), (ax + 3, y + h + 1), (ax, y + h + 6)], fill=col)


VERBS = ['CAMMINA', 'GUARDA', 'PARLA', 'BEVI']


def verbbar(dr, t, active, action, slide=1.0, scramble=0):
    if CLEAN or slide <= 0: return
    y0 = int(H - 18 * slide)
    dr.rectangle([0, y0, W, H], fill=C['BLK'])
    dr.line([0, y0, W, y0], fill=C['PLUM'])
    x = 3
    for v in VERBS:
        w = tlen(v) + 6
        on = v == active
        dr.rectangle([x, y0 + 2, x + w, y0 + 17], fill=C['MAG'] if on else C['NAVY'], outline=C['PINK'] if on else C['DPUR'])
        txt(dr, x + 3, y0 + 1, v, C['WHT'] if on else C['GRY'], None)
        x += w + 3
    for i in range(4):  # inventory
        sx = 232 + i * 22
        dr.rectangle([sx, y0 + 2, sx + 18, y0 + 17], fill=C['NAVY'], outline=C['TEAL'])
    ix, iy = 232, y0 + 4
    dr.rectangle([ix + 5, iy + 4, ix + 12, iy + 11], fill=C['WHT']); dr.rectangle([ix + 6, iy + 5, ix + 11, iy + 6], fill=C['BRN'])
    dr.rectangle([ix + 13, iy + 6, ix + 14, iy + 9], outline=C['WHT'])
    ix += 22; dr.rectangle([ix + 7, iy + 3, ix + 11, iy + 12], fill=C['CYAN']); dr.rectangle([ix + 8, iy + 1, ix + 10, iy + 2], fill=C['WHT'])
    ix += 22; dr.rectangle([ix + 5, iy + 4, ix + 13, iy + 10], fill=C['ORNG'])
    for k in range(5):
        dr.point((ix + 5 + 2 * k, iy + 3), C['AMBR']); dr.point((ix + 5 + 2 * k, iy + 11), C['AMBR'])
    ix += 22; txt(dr, ix + 5, y0 + 1, '€', C['AMBR'], None)
    if action:
        s = action if scramble < 0.5 else ''.join(ch if rng.random() > scramble * 0.3 else rng.choice(list('#%@01€')) for ch in action)
        txt(dr, 160 - tlen(s) // 2, y0 - 17, s, C['LCYN'])


CUR = ["X..........", "XX.........", "XWX........", "XWWX.......", "XWWWX......", "XWWWWX.....",
       "XWWWWWX....", "XWWWWWWX...", "XWWWWXXXX..", "XWXWWX.....", "XX.XWWX....", "X...XWX....", "....XX....."]


def cursor(dr, x, y, t, click_t=None):
    if CLEAN: return
    x, y = int(x), int(y)
    if click_t is not None and 0 <= t - click_t < 0.45:
        r = int(2 + (t - click_t) * 30)
        dr.ellipse([x - r, y - r // 2, x + r, y + r // 2], outline=C['CYAN'] if int((t - click_t) * 20) % 2 else C['MAG'])
    for j, row in enumerate(CUR):
        for i, ch in enumerate(row):
            if ch == 'X': dr.point((x + i, y + j), C['BLK'])
            elif ch == 'W': dr.point((x + i, y + j), C['WHT'])


def path(t, keys):
    """Eased interpolation through (time, x, y) keyframes: the scripted mouse."""
    if t <= keys[0][0]: return keys[0][1:]
    for a, b in zip(keys, keys[1:]):
        if t <= b[0]:
            u = ease((t - a[0]) / (b[0] - a[0]))
            return (lerp(a[1], b[1], u), lerp(a[2], b[2], u))
    return keys[-1][1:]


# ---------------------------------------------------------------- sprites (16x16, Game Boy overworld style)
# O outline  H hair  S skin  J jacket  P pants  A white trim  V visor  W eye glint  B blush  M headphones
SP_FRONT = """
.......OO.......
......OHHO......
.....OHHHHO.....
...OOOHHHHOOO...
..OSSSOHHOSSSO..
..OSSSSSSSSSSO..
..OSOVVVVVVOSO..
..OSOVWVVWVOSO..
..OSSSSSSSSSSO..
...OSSSOOSSSO...
...OOJJJJJJOO...
..OJJAJJJJAJJO..
..OSOJJJJJJOSO..
...OOJJJJJJOO...
....OPPOOPPO....
....OOO..OOO....
"""
SP_BACK = """
.......OO.......
......OHHO......
.....OHHHHO.....
...OOOHHHHOOO...
..OSSSOHHOSSSO..
..OSSSOHHOSSSO..
..OSSSOHHOSSSO..
..OSSSSOOSSSSO..
..OSSSSSSSSSSO..
...OSSSSSSSSO...
...OOJJJJJJOO...
..OJJJAAAAJJJO..
..OSOJAVVAJOSO..
...OOJJJJJJOO...
....OPPOOPPO....
....OOO..OOO....
"""
SP_SIDE = """
....OO.OO.......
...OHHOHHO......
..OHHHHHHHO.....
.OOOOOOOOOOO....
.OSSSSSSSSSO....
.OSSSSSSSSSO....
.OSSSSSOVVVVO...
.OSSSSSOVWVVO...
.OSSSSSSSSSSO...
..OSSSSSSSOO....
...OJJJJJJO.....
..OJJJJJAJJO....
..OJJJJJOSO.....
...OJJJJJO......
...OPPOPPO......
...OOO.OOO......
"""
SP_SIDE2 = SP_SIDE.strip().split()[:14] + ["..OPPO..OPPO....", "..OOO....OOO...."]

# Volpe, the tsundere hacker-barista
HK_FRONT = """
....OOOOOOOO....
...OHHHHHHHHO...
..OHHHHHHHHHHO..
.OMOHHHHHHHHOMO.
.OMOSHHHHHHSOMO.
.OMOSSSSSSSSOMO.
.OOOVVVOOVVVOOO.
..OSVWVOOVWVSO..
..OSSSSSSSSSSO..
..OHSSSSOSSSHO..
...OOSSSSSSOO...
..OJJJOOOOJJJO..
.OJJJJJAAJJJJJO.
.OJJJJJAAJJJJJO.
.OSOJJJJJJJJOSO.
..OOOOOOOOOOOO..
"""
HK_BACK = """
....OOOOOOOO....
...OHHHHHHHHO...
..OHHHHHHHHHHO..
.OMOHHHHHHHHOMO.
.OMOHHHHHHHHOMO.
.OMOHHHHHHHHOMO.
.OOOHHHHHHHHOOO.
..OHHHHHHHHHHO..
..OHHHHHHHHHHO..
..OHHHHHHHHHHO..
...OOHHHHHHOO...
..OJJJJJJJJJJO..
.OJJJJAAAAJJJJO.
.OJJJJJJJJJJJJO.
.OSOJJJJJJJJOSO.
..OOOOOOOOOOOO..
"""


def _hk_variant(src, blush=False, away=False, pout=False):
    rows = [list(r) for r in src.strip().split()]
    if away: rows[7] = list("..OSVVWOOVVWSO..")  # eyes slide sideways: "hmph"
    if blush: rows[8][3] = rows[8][4] = rows[8][11] = rows[8][12] = 'B'
    if pout: rows[9][7] = rows[9][8] = 'O'
    return [''.join(r) for r in rows]


def _grid(src):
    rows = src if isinstance(src, list) else src.strip().split()
    return [r.ljust(16, '.')[:16] for r in rows]


SPRITES = {'front': _grid(SP_FRONT), 'back': _grid(SP_BACK), 'side': _grid(SP_SIDE), 'side2': _grid(SP_SIDE2),
           'hk': _grid(HK_FRONT), 'hk_back': _grid(HK_BACK),
           'hk_away': _grid(_hk_variant(HK_FRONT, away=True)),
           'hk_blush': _grid(_hk_variant(HK_FRONT, blush=True, away=True, pout=True)),
           'hk_mad': _grid(_hk_variant(HK_FRONT, pout=True))}
_SC = {}


def sprite_img(view, hair, jacket, visor=None, skin=None, flip=False, rows=16):
    key = (view, hair, jacket, visor, skin, flip, rows)
    if key in _SC: return _SC[key]
    cmap = {'O': C['BLK'], 'H': hair, 'S': skin or C['SKIN'], 'J': jacket, 'P': C['DGRY'],
            'A': C['WHT'], 'V': visor or C['CYAN'], 'W': C['WHT'], 'B': C['PINK'], 'M': C['MAG']}
    a = np.zeros((rows, 16, 4), np.uint8)
    for y, row in enumerate(SPRITES[view][:rows]):
        for x, ch in enumerate(row):
            if ch in cmap: a[y, x, :3] = cmap[ch]; a[y, x, 3] = 255
    if flip: a = a[:, ::-1]
    _SC[key] = Image.fromarray(a)
    return _SC[key]


def guy(im, x, y, hair, jacket, view='front', scale=2, bob=0, flip=False, walk=None, visor=None, skin=None, rows=16, cup=False):
    """Draw a chibi with feet at (x, y)."""
    if walk is not None and view == 'side': view = 'side2' if walk % 2 else 'side'
    sp = sprite_img(view, hair, jacket, visor, skin, flip, rows).resize((16 * scale, rows * scale), Image.NEAREST)
    top = int(y - rows * scale - bob * scale)
    im.alpha_composite(sp, (int(x - 8 * scale), top))
    if cup:  # espresso raised to the shoulder
        dr = ImageDraw.Draw(im)
        cx, cy = int(x + 6 * scale), top + 7 * scale
        dr.rectangle([cx - 4, cy - 4, cx + 4, cy + 4], fill=C['BLK'])
        dr.rectangle([cx - 3, cy - 3, cx + 3, cy + 3], fill=C['WHT'])
        dr.line([cx - 3, cy - 3, cx + 3, cy - 3], fill=C['BRN'])


HAIR_HK, HOOD_HK = C['VIO'], C['DTEAL']


def hacker(im, x, t, mood='hk', vein=False, typing=False, trip=0.0):
    x = int(x)
    visor = C['CYAN'] if trip < 0.3 else [C['MAG'], C['GRN'], C['AMBR'], C['CYAN']][int(t * 6) % 4]
    guy(im, x, 118, HAIR_HK, HOOD_HK, view=mood, scale=3, rows=13, visor=visor)
    dr = ImageDraw.Draw(im); dr.fontmode = '1'
    top = 118 - 39
    if vein and int(t * 5) % 2 == 0:  # anime anger mark
        vx, vy = x + 15, top + 2
        for ox, oy in [(-2, -2), (2, -2), (-2, 2), (2, 2)]:
            sx_, sy_ = (1 if ox > 0 else -1), (1 if oy > 0 else -1)
            dr.line([vx + ox, vy + oy, vx + ox + sx_, vy + oy], fill=C['RED'])
            dr.line([vx + ox, vy + oy, vx + ox, vy + oy + sy_], fill=C['RED'])
    if typing:  # holo keyboard + floating code
        dr.rectangle([x - 16, 111, x + 16, 116], outline=C['CYAN'])
        for k in range(-14, 15, 3):
            if (k + int(t * 24)) % 5: dr.point((x + k, 113 + (k // 3) % 2), C['LCYN'])
        for k in range(6):
            ph = (t * 0.7 + k / 6) % 1
            cx, cy = x - 20 + (k * 37) % 40, int(108 - ph * 40)
            dr.point((cx, cy), C['GRN'] if k % 2 else C['CYAN'])
            if ph < 0.5: dr.point((cx, cy + 2), C['DGRN'])
    return (x + 22, top + 26)


def cup(dr, x, y, full=True):
    x, y = int(x), int(y)
    dr.rectangle([x - 3, y - 6, x + 3, y], fill=C['BLK'])
    dr.rectangle([x - 2, y - 5, x + 2, y], fill=C['WHT'])
    dr.line([x + 3, y - 4, x + 4, y - 4], fill=C['WHT']); dr.line([x + 4, y - 4, x + 4, y - 2], fill=C['WHT'])
    if full: dr.line([x - 2, y - 5, x + 2, y - 5], fill=C['BRN'])


def steam(dr, x, y, t, n=6, col=C['LPINK'], hgt=18):
    for k in range(n):
        ph = (t * 0.8 + k / n) % 1
        sy = y - ph * hgt
        sx = x + math.sin(t * 3 + k * 1.7 + ph * 6) * (1 + ph * 3)
        if (k + int(t * 10)) % 3 != 0 or ph < 0.5:
            dr.point((int(sx), int(sy)), col if ph < 0.6 else C['GRY'])


# ---------------------------------------------------------------- painted parallax layers
def skyline(w, h, base, hmin, hmax, col, wins, pwin, seed, wmin=10, wmax=34, antenna=0.3):
    r = np.random.default_rng(seed)
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0)); dr = ImageDraw.Draw(im)
    x = -5
    while x < w:
        bw = int(r.integers(wmin, wmax)); bh = int(r.integers(hmin, hmax)); top = base - bh
        dr.rectangle([x, top, x + bw, h], fill=col + (255,))
        if r.random() < antenna:
            ax = x + int(r.integers(2, max(3, bw - 2)))
            dr.line([ax, top - int(r.integers(4, 16)), ax, top], fill=col + (255,))
            dr.point((ax, top - 1 - int(r.integers(4, 16))), C['RED'] + (255,))
        if r.random() < 0.25:
            dr.rectangle([x + 2, top - 4, x + bw - 3, top], fill=col + (255,))
        for wy in range(top + 3, h - 2, 4):
            for wx in range(x + 2, x + bw - 1, 3):
                if r.random() < pwin:
                    dr.point((wx, wy), wins[int(r.integers(len(wins)))] + (255,))
        if r.random() < 0.18:  # vertical neon sign
            sx, sy = x + bw - 3, top + 6
            nc = [C['MAG'], C['CYAN'], C['AMBR'], C['GRN']][int(r.integers(4))]
            dr.rectangle([sx, sy, sx + 2, sy + int(r.integers(10, 24))], fill=nc + (255,))
        x += bw + int(r.integers(-3, 3))
    return im


def fog(w, h, col, y0, y1, dens):
    a = np.zeros((h, w, 4), np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    v = np.clip((yy - y0) / max(y1 - y0, 1), 0, 1) * dens
    a[BAY[yy % 4, xx % 4] < v] = col + (255,)
    return Image.fromarray(a)


def disc(cx, cy, r, cols):
    a = np.zeros((H, W, 4), np.uint8)
    dd = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)
    m = dd < r
    v = np.clip((YY - (cy - r)) / (2 * r), 0, 1) * (len(cols) - 1)
    idx = np.clip(np.floor(v + BAYF), 0, len(cols) - 1).astype(int)
    a[m, :3] = np.array(cols, np.uint8)[idx][m]; a[m, 3] = 255
    hm = (dd >= r) & (dd < r + 10) & (BAYF < (1 - (dd - r) / 10) * 0.35)
    a[hm, :3] = C['PLUM']; a[hm, 3] = 255
    return Image.fromarray(a)


def duomo(d_, cx, base, col, r=20):
    d_.rectangle([cx - r - 8, base - 22, cx + r + 8, H], fill=col + (255,))
    d_.rectangle([cx - r + 2, base - 34, cx + r - 2, base - 22], fill=col + (255,))
    d_.pieslice([cx - r, base - 34 - r, cx + r, base - 34 + r], 180, 360, fill=col + (255,))
    d_.rectangle([cx - 3, base - 34 - r - 8, cx + 3, base - 34 - r], fill=col + (255,))
    d_.line([cx, base - 34 - r - 14, cx, base - 34 - r - 8], fill=col + (255,))
    d_.point((cx, base - 34 - r - 15), C['AMBR'] + (255,))
    for k in range(-r + 4, r - 3, 5):
        d_.line([cx + k, base - 31, cx + k, base - 26], fill=C['AMBR'] + (255,))
    for k in range(-2, 3):
        d_.arc([cx - r + 2, base - 34 - r + 2, cx + r - 2, base - 34 + r - 2], 200 + k * 30, 205 + k * 30, fill=C['MAG'] + (255,))


def campanile(d_, cx, base, col, h=80, lean=0):
    for yy in range(base - h, H):
        off = int(lean * (base - yy) / h)
        d_.line([cx - 5 + off, yy, cx + 5 + off, yy], fill=col + (255,))
    for yy in range(base - h + 6, base, 12):
        o = int(lean * (base - yy) / h)
        d_.point((cx - 2 + o, yy), C['CYAN'] + (255,)); d_.point((cx + 2 + o, yy), C['CYAN'] + (255,))
    d_.rectangle([cx - 6 + lean, base - h - 4, cx + 6 + lean, base - h], fill=col + (255,))
    d_.line([cx + lean, base - h - 10, cx + lean, base - h - 4], fill=col + (255,))


LW = 720
SKY1 = Image.fromarray(dgrad(H, W, [C['BLK'], C['NAVY'], C['DPUR'], C['PUR'], C['PLUM'], C['MAG']])).convert('RGBA')
MOON = disc(215, 62, 30, [C['LPINK'], C['PINK'], C['ORNG'], C['MAG']])
FAR = skyline(LW, H, 140, 30, 95, C['BLD'], [C['SKY'], C['PINK'], C['PUR']], 0.10, 11)
FOGA = fog(LW, H, C['PLUM'], 70, 150, 0.7)
MID = skyline(LW, H, 160, 20, 80, C['BLD2'], [C['AMBR'], C['CYAN'], C['ORNG'], C['PINK']], 0.18, 12, wmin=14, wmax=40)
FOGB = fog(LW, H, C['DPUR'], 120, 175, 0.55)
NEAR = Image.new('RGBA', (LW, H), (0, 0, 0, 0))
_nd = ImageDraw.Draw(NEAR)
_r = np.random.default_rng(5)
_x = 0
while _x < LW:  # foreground rooftops with fire escapes
    bw = int(_r.integers(40, 90)); top = int(_r.integers(128, 152))
    _nd.rectangle([_x, top, _x + bw, H], fill=C['BLK'] + (255,))
    _nd.line([_x, top, _x + bw, top], fill=C['DPUR'] + (255,))
    for fy in range(top + 14, H - 10, 14):
        if _r.random() < 0.5:
            _nd.line([_x + 4, fy, _x + bw - 4, fy], fill=C['NAVY'] + (255,))
            for k in range(_x + 4, _x + bw - 4, 3): _nd.point((k, fy - 2), C['NAVY'] + (255,))
    for wy in range(top + 6, H, 10):
        for wx in range(_x + 5, _x + bw - 6, 9):
            if _r.random() < 0.12: _nd.rectangle([wx, wy, wx + 3, wy + 4], fill=C['AMBR'] + (255,))
    _x += bw
TX = 520
_nd.rectangle([TX - 30, 118, TX + 60, H], fill=C['BLK'] + (255,))
NEAR_SIGN = (TX, 92)
_nd.line([TX + 17, 128, TX + 17, 118], fill=C['DGRY'] + (255,))
_md = ImageDraw.Draw(MID)  # mega tower, duomo and leaning campanile
_md.rectangle([380, 15, 396, 160], fill=C['BLD2'] + (255,)); _md.rectangle([384, 2, 392, 15], fill=C['BLD2'] + (255,))
_md.line([388, -5, 388, 2], fill=C['BLD2'] + (255,))
for wy in range(20, 160, 5):
    _md.point((383, wy), C['CYAN'] + (255,)); _md.point((393, wy), C['MAG'] + (255,))
duomo(_md, 250, 160, C['BLD2']); campanile(_md, 290, 160, C['BLD2'], 95, lean=6)
duomo(_md, 480, 150, C['BLD2'], r=24); campanile(_md, 520, 150, C['BLD2'], 80)


def glyph_big(s, col, scale=2):
    w = tlen(s) + 2
    im = Image.new('RGBA', (w, 17), (0, 0, 0, 0)); dr = ImageDraw.Draw(im); dr.fontmode = '1'
    dr.text((1, 0), s, font=FONT, fill=col + (255,))
    return im.resize((w * scale, 17 * scale), Image.NEAREST)


BAR_GLYPH = glyph_big('BAR', C['PINK'])
BAR_GLYPH_DIM = glyph_big('BAR', C['DRED'])


def rain(dr, t, n=110, sx=-30, sy=240, col=C['TEAL'], col2=C['SKY'], ymax=H, seed=3):
    r = np.random.default_rng(seed)
    X0 = r.random(n) * (W + 60); Y0 = r.random(n) * (ymax + 20); sp = 0.7 + r.random(n) * 0.6
    for i in range(n):
        x = (X0[i] + t * sx * sp[i]) % (W + 60) - 30
        y = (Y0[i] + t * sy * sp[i]) % (ymax + 20) - 10
        dr.line([x, y, x - 1, y + 3], fill=col if i % 3 else col2)


def cars(dr, t, cam):
    for i, (y, sp, col) in enumerate([(40, 35, C['AMBR']), (55, -50, C['CYAN']), (28, 22, C['MAG']), (70, -28, C['WHT'])]):
        x = ((i * 97 + t * sp - cam * 0.4) % (W + 40)) - 20
        dr.rectangle([x, y, x + 3, y + 1], fill=col)
        dr.point((x - (2 if sp > 0 else -5), y), C['PLUM'])


# ================================================================ SCENE 1: the city
def scene1(t):
    cam = ease(t / 14.6) * 300
    im = SKY1.copy()
    im.alpha_composite(MOON, (0, 0))
    fl = max(0, 1 - abs(t - 1.05) / 0.35) + max(0, 1 - abs(t - 7.45) / 0.35)  # lightning on the two bass hits
    if fl > 0:
        a = np.array(im); a[(BAYF < fl * 0.5) & (YY < 120), :3] = C['PINK']; im = Image.fromarray(a)
    for L, sp in [(FAR, 0.3), (FOGA, 0.3), (MID, 0.6), (FOGB, 0.6), (NEAR, 1.0)]:
        im.alpha_composite(L, (-int(cam * sp), 0))
    dr = ImageDraw.Draw(im); dr.fontmode = '1'
    cars(dr, t, cam)
    sx, sy = NEAR_SIGN[0] - int(cam), NEAR_SIGN[1]
    on = t > 1.0 and not (2.0 < t < 2.2 or 7.4 < t < 7.55 or (int(t * 13) % 17 == 0))
    dr.rectangle([sx - 3, sy - 3, sx + 54, sy + 36], fill=C['BLK'], outline=C['MAG'] if on else C['DRED'])
    im.alpha_composite(BAR_GLYPH if on else BAR_GLYPH_DIM, (sx + 1, sy + 1))
    for k, ch in enumerate('CAFFÈ'):
        txt(dr, sx + 60, sy - 2 + k * 14, ch, C['CYAN'] if on and (k != int(t * 4) % 6) else C['DTEAL'], None)
    if on:
        a = np.array(im)
        gm = (abs(XX - (sx + 26)) < 40) & (abs(YY - (sy + 17)) < 28) & (BAYF < 0.12) & (a[..., 0] < 40)
        a[gm, :3] = C['PLUM']; im = Image.fromarray(a); dr = ImageDraw.Draw(im); dr.fontmode = '1'
    rain(dr, t)
    dr.rectangle([0, 172, W, H], fill=C['BLK'])
    dialog(dr, t, 1.8, 6.8, 'TU', ['PIOVE NEON SULLA CITTÀ STANCA,', 'IL TURNO È FINITO, LA NOTTE NO.'], C['GRN'])
    dialog(dr, t, 7.6, 13.4, 'TU', ["C'È UN BAR DOVE IL TEMPO SI FERMA,", 'DOVE IL CAFFÈ SA DI STELLE.'], C['GRN'])
    tgt = (sx + 26, sy + 17)
    cx, cy = path(t, [(3.0, 60, 150), (6.0, 120, 90), (9.5, tgt[0] - 10, tgt[1] + 20), (11.5, tgt[0], tgt[1])])
    act = 'GUARDA IL BAR' if t > 11.0 else ('GUARDA LA CITTÀ' if t > 4 else '')
    verbbar(dr, t, 'GUARDA', act, ease((t - 3) / 0.6))
    if t > 3: cursor(dr, cx, cy, t, 13.8)
    return im


# ================================================================ SCENE 2: outside Caffè Neon
def build_s2():
    im = Image.fromarray(dgrad(H, W, [C['NAVY'], C['DPUR'], C['PUR'], C['PLUM']])).convert('RGBA')
    im.alpha_composite(MID, (-200, -30))
    im.alpha_composite(fog(W, H, C['DPUR'], 40, 130, 0.5))
    dr = ImageDraw.Draw(im)
    dr.rectangle([90, 30, 300, 150], fill=C['BLD'])
    br = np.array(im)
    m = (XX >= 90) & (XX <= 300) & (YY >= 30) & (YY <= 150)
    brick = ((YY // 4) % 2 == 0) & (((XX + (YY // 4 % 2) * 4) % 8) == 0) | (YY % 4 == 0)
    br[m & brick & (BAYF < 0.6), :3] = C['NAVY']
    im = Image.fromarray(br); dr = ImageDraw.Draw(im)
    dr.rectangle([0, 50, 88, 150], fill=C['BLD2'])
    for wy in range(60, 140, 16):
        for wx in range(8, 80, 18):
            dr.rectangle([wx, wy, wx + 8, wy + 9], fill=C['NAVY'], outline=C['BLK'])
    dr.rectangle([20, 60, 28, 69], fill=C['AMBR'])
    im.paste(Image.fromarray(dgrad(54, 74, [C['AMBR'], C['ORNG'], C['PLUM'], C['DPUR']])), (118, 88))
    dr = ImageDraw.Draw(im)
    for lx in (132, 156, 178):  # pendant lamps
        dr.line([lx, 88, lx, 95], fill=C['BLK'])
        dr.polygon([(lx - 5, 101), (lx + 5, 101), (lx + 2, 95), (lx - 2, 95)], fill=C['BLK'])
        dr.rectangle([lx - 1, 102, lx + 1, 103], fill=C['YEL'])
    dr.rectangle([118, 128, 192, 142], fill=C['BRN'])
    for sx_ in (128, 150, 170, 184):
        dr.rectangle([sx_ - 3, 112, sx_ + 3, 128], fill=C['BLK']); dr.rectangle([sx_ - 2, 107, sx_ + 2, 112], fill=C['BLK'])
    dr.rectangle([116, 86, 193, 143], outline=C['BLK'])
    dr.line([155, 86, 155, 143], fill=C['BLK'])
    tri = [C['DGRN'], C['WHT'], C['RED']]  # tricolore awning
    for j, k in enumerate(range(0, 84, 4)):
        dr.rectangle([113 + k, 76, 116 + k, 85], fill=tri[j % 3])
    for k in range(113, 197, 4): dr.point((k + 1, 86), tri[(k // 4) % 3])
    dr.rectangle([204, 92, 228, 146], fill=C['BLK'], outline=C['CYAN'])  # door
    dr.rectangle([206, 94, 226, 146], fill=C['NAVY'])
    dr.point((223, 120), C['AMBR'])
    dr.rectangle([244, 40, 262, 52], fill=C['DGRY']); dr.line([246, 46, 260, 46], fill=C['BLK'])
    dr.line([296, 30, 296, 150], fill=C['DGRY']); dr.line([298, 30, 298, 150], fill=C['BLK'])
    dr.rectangle([240, 96, 262, 126], fill=C['TEAL']); dr.rectangle([243, 100, 259, 108], fill=C['MAG'])
    dr.rectangle([266, 100, 284, 118], fill=C['AMBR']); dr.rectangle([269, 103, 281, 106], fill=C['BLK'])
    im.paste(Image.fromarray(dgrad(16, W, [C['DPUR'], C['NAVY'], C['BLK']])), (0, 146))
    return im


S2BG = build_s2()


def reflect(a, y0, y1, t):
    """Rain-puddle reflection: mirror rows above y0, wobble, darken, dither."""
    out = a.copy()
    for y in range(y0, y1):
        src = y0 - (y - y0) - 1
        if src < 0: break
        sh = int(round(math.sin(y * 0.9 + t * 6) * (1 + (y - y0) * 0.15)))
        row = np.roll(a[src], sh, axis=0)
        mix = BAY[y % 4, np.arange(W) % 4] < 0.55
        out[y, mix] = (row[mix] * 0.6).astype(np.uint8)
    return out


def scene2(t):
    ts = t - 14.6
    b, ph = beat(t)
    bob = 1 if ph < 0.25 else 0
    im = S2BG.copy()
    dr = ImageDraw.Draw(im); dr.fontmode = '1'
    dr.rectangle([112, 44, 200, 66], fill=C['BLK'], outline=C['MAG'])
    for k, ch in enumerate('CAFFÈ NEON'):  # letters chase the beat
        lit = (k + b) % 4 != 0
        txt(dr, 116 + k * 8, 47, ch, (C['PINK'] if k < 5 else C['CYAN']) if lit else C['DTEAL'], None)
    tagp = ease((ts - 1) / 12)  # graffiti grows out of noise contours
    if tagp > 0:
        a = np.array(im)
        tx0, ty0 = 236, 58
        sub = (XX > tx0) & (XX < tx0 + 60) & (YY > ty0) & (YY < ty0 + 32)
        n = samp(N3, XX * 3 + 40, YY * 3 + 90)
        dist = np.sqrt(((XX - (tx0 + 30)) / 30) ** 2 + ((YY - (ty0 + 16)) / 16) ** 2)
        a[sub & (n > 0.5) & (dist < tagp * 1.2) & (np.abs((n * 8) % 1 - 0.5) < 0.25), :3] = C['GRN']
        a[sub & (n > 0.62) & (dist < tagp * 1.2), :3] = C['MAG']
        im = Image.fromarray(a); dr = ImageDraw.Draw(im); dr.fontmode = '1'
    if ts > 14.4:
        dr.rectangle([206, 94, 206 + int(20 * ease((ts - 14.4) / 0.6)), 146], fill=C['AMBR'])
    # hover-Vespa
    hb = int(math.sin(t * 4) * 1.5)
    bx, by = 52, 138 + hb
    dr.ellipse([bx - 24, by - 12, bx + 2, by + 4], fill=C['BLK'])
    dr.ellipse([bx - 23, by - 11, bx + 1, by + 3], fill=C['LCYN'])
    dr.arc([bx - 21, by - 9, bx - 1, by + 1], 200, 300, fill=C['WHT'])
    dr.rectangle([bx - 2, by - 1, bx + 14, by + 3], fill=C['TEAL'], outline=C['BLK'])
    dr.polygon([(bx + 12, by + 3), (bx + 14, by - 15), (bx + 19, by - 17), (bx + 19, by + 3)], fill=C['LCYN'], outline=C['BLK'])
    dr.line([bx + 13, by - 21, bx + 22, by - 21], fill=C['GRY']); dr.line([bx + 17, by - 21, bx + 17, by - 16], fill=C['GRY'])
    dr.ellipse([bx + 18, by - 20, bx + 23, by - 15], fill=C['YEL'], outline=C['BLK'])
    dr.rectangle([bx - 19, by - 13, bx - 4, by - 11], fill=C['BRN'], outline=C['BLK'])
    for k in range(-20, 22, 2):
        if (k + int(t * 20)) % 4: dr.point((bx + k, by + 7 + ((k // 2) % 2)), C['CYAN'])
    guy(im, bx - 10, by - 11, C['CYAN'], C['PUR'], view='side', rows=12, bob=bob, visor=C['MAG'])
    guy(im, 100, 147, C['MAG'], C['DRED'], view='front', bob=bob)
    cxp = 274
    guy(im, cxp, 147, C['AMBR'], C['TEAL'], view='side', flip=True, bob=bob, visor=C['PINK'])
    if 0 < tagp < 1:
        for k in range(5):
            dr.point((cxp - 12 - int(rng.random() * 10), 118 - int(rng.random() * 26)), C['GRN'])
    # the player walks where the cursor clicks
    keys = [(0.0, -12), (0.9, -12), (4.4, 140), (12.6, 140), (14.4, 214)]
    walking, px = False, None
    if ts < keys[1][0]: px = -12
    else:
        for a_, b_ in zip(keys, keys[1:]):
            if ts <= b_[0]:
                px = lerp(a_[1], b_[1], (ts - a_[0]) / (b_[0] - a_[0])); walking = a_[1] != b_[1]; break
        if px is None: px = 214
    if ts < 14.8:
        if walking: guy(im, px, 147, C['GRN'], C['BLUE'], view='side', walk=int(t * 6))
        elif 7 < ts < 9.5: guy(im, px, 147, C['GRN'], C['BLUE'], view='side', flip=True, bob=bob)
        else: guy(im, px, 147, C['GRN'], C['BLUE'], view='front', bob=bob)
    im = Image.fromarray(reflect(np.array(im.convert('RGB')), 147, 162, t)).convert('RGBA')
    dr = ImageDraw.Draw(im); dr.fontmode = '1'
    rain(dr, t, n=70, ymax=150, seed=4)
    for k in range(6):
        if (int(t * 10) + k) % 3 == 0: dr.point(((k * 53 + int(t * 37) * 7) % W, 150 + k % 3 * 3), C['LCYN'])
    dialog(dr, t, 14.6 + 4.8, 14.6 + 9.4, 'PUNK', ['OI, FRATELLO DELLA PIOGGIA,', "C'È ANCORA L'ESPRESSO DI LUNA?"], C['MAG'], ax=100)
    dialog(dr, t, 14.6 + 9.6, 14.6 + 13.6, 'TU', ['SOLO CHI ENTRA LO SAPRÀ,', 'E CHI BEVE NON TORNA UGUALE.'], C['GRN'], ax=140)
    cpos = path(ts, [(0, 190, 120), (0.8, 140, 150), (3.0, 110, 120), (4.6, 102, 118), (11.4, 120, 125), (12.5, 216, 118)])
    act = 'CAMMINA PER STRADA' if ts < 4 else ('PARLA COL PUNK' if ts < 11.6 else 'ENTRA NEL BAR')
    verbbar(dr, t, 'CAMMINA' if (ts < 4 or ts > 11.6) else 'PARLA', act)
    cursor(dr, *cpos, t, max([c for c in [15.4, 19.2, 27.1] if c <= t], default=None))
    return im


# ================================================================ SCENE 3: inside the bar
WIN_CITY = Image.new('RGBA', (W, H)); WIN_CITY.alpha_composite(SKY1); WIN_CITY.alpha_composite(MOON, (-60, 10))
WIN_CITY.alpha_composite(FAR, (-100, 0)); WIN_CITY.alpha_composite(MID, (-150, 10))
WIN_CITY = WIN_CITY.resize((120, 68), Image.NEAREST)


def build_s3():
    im = Image.fromarray(dgrad(H, W, [C['DPUR'], C['NAVY'], C['DTEAL'], C['NAVY']])).convert('RGBA')
    dr = ImageDraw.Draw(im)
    for x in range(0, W, 12): dr.line([x, 0, x, 118], fill=C['BLD'])
    mask = Image.new('L', (60, 60)); ImageDraw.Draw(mask).ellipse([0, 0, 59, 59], fill=255)
    im.paste(WIN_CITY.crop((20, 0, 80, 60)), (16, 22), mask)  # porthole onto the city
    dr.ellipse([15, 21, 76, 82], outline=C['BRN']); dr.ellipse([14, 20, 77, 83], outline=C['BLK'])
    dr.line([46, 22, 46, 81], fill=C['BRN']); dr.line([16, 52, 76, 52], fill=C['BRN'])
    r = np.random.default_rng(9)
    for sy in (36, 60, 84):  # bottle shelves
        dr.rectangle([222, sy, 314, sy + 2], fill=C['BRN'])
        x = 225
        while x < 308:
            w = int(r.integers(4, 7)); h = int(r.integers(9, 15))
            col = [C['RED'], C['YEL'], C['DGRN'], C['AMBR'], C['BRN'], C['PINK']][int(r.integers(6))]
            dr.rectangle([x, sy - h + 4, x + w, sy - 1], fill=col, outline=C['BLK'])
            dr.rectangle([x + w // 2 - 1, sy - h, x + w // 2 + 1, sy - h + 4], fill=C['BLK'])
            dr.rectangle([x + 1, sy - h + 7, x + w - 1, sy - h + 9], fill=C['WHT'])
            x += w + 3
    dr.rectangle([0, 118, W, 122], fill=C['PLUM']); dr.line([0, 118, W, 118], fill=C['PINK'])
    im.paste(Image.fromarray(dgrad(40, W, [C['PUR'], C['DPUR'], C['NAVY'], C['BLK']])), (0, 124))
    dr = ImageDraw.Draw(im)
    for x in range(10, W, 40): dr.rectangle([x, 128, x + 30, 156], outline=C['BLD'])
    return im


S3BG = build_s3()
MENU = [('ESPRESSO', '€1'), ('RISTRETTO', '€1'), ('MACCHIATO', '€2'), ('AMARO', '€4')]
SEATS = [(62, C['GRN'], C['BLUE']), (112, C['MAG'], C['DRED']), (208, C['CYAN'], C['PUR']), (258, C['AMBR'], C['TEAL'])]


def scene3(t):
    ts = t - 30.0
    b, ph = beat(t)
    trip = ease((t - 42.2) / 10.5)
    bob = 1 if ph < 0.3 else 0
    im = S3BG.copy()
    if trip > 0:  # the walls start to breathe
        a = np.array(im)
        n = samp(N1, XX + samp(N2, XX * 2, YY * 2 + t * 30) * 60, YY + t * 8)
        f = n * 10 + t * 0.6
        vein = (np.abs(f % 1 - 0.5) < 0.08 + 0.1 * trip) & (YY < 118) & (BAYF < trip * 1.1)
        cols = np.array([C['MAG'], C['TEAL'], C['GRN'], C['VIO']], np.uint8)
        a[vein, :3] = cols[np.floor(f).astype(int) % 4][vein]
        im = Image.fromarray(a)
    dr = ImageDraw.Draw(im); dr.fontmode = '1'
    for i, lx in enumerate([100, 172, 300]):  # swinging lamps
        cx_ = lx + math.sin(t * 2 + i) * 3
        col = [C['DGRN'], C['WHT'], C['RED']][i] if trip < 0.3 else [C['RED'], C['MAG'], C['VIO'], C['GRN']][(b + i) % 4]
        dr.line([lx, 0, cx_, 12], fill=C['BLK'])
        rr = 9 + (2 if ph < 0.2 else 0)
        for k in range(18):
            ang = k / 18 * 6.283
            dr.point((int(cx_ + math.cos(ang) * (rr + 3)), int(19 + math.sin(ang) * (rr + 1))), C['PLUM'])
        dr.polygon([(cx_ - 9, 24), (cx_ + 9, 24), (cx_ + 3, 13), (cx_ - 3, 13)], fill=col, outline=C['BLK'])
        dr.line([cx_ - 8, 23, cx_ + 8, 23], fill=C['AMBR'])
        dr.rectangle([cx_ - 2, 25, cx_ + 2, 27], fill=C['YEL'])
    mx, my = 118, 26  # holo menu
    if int(t * 30) % 23 != 0:
        dr.rectangle([mx, my, mx + 100, my + 70], outline=C['CYAN'])
        dr.line([mx + 1, my + 1, mx + 99, my + 1], fill=C['TEAL'])
        for k, (row, p) in enumerate(MENU):
            if trip > 0.2:
                row = ''.join(ch if rng.random() > trip * 0.4 else rng.choice(list('€#%01▓░')) for ch in row)
            hl = (b // 2) % 4 == k and ts > 3.5
            txt(dr, mx + 3, my + 3 + k * 16, row[:9], C['WHT'] if hl else C['CYAN'], None)
            txt(dr, mx + 84, my + 3 + k * 16, p, C['AMBR'], None)
        for yy in range(my + 2, my + 70, 2):
            if (yy + int(t * 20)) % 9 == 0: dr.line([mx + 1, yy, mx + 99, yy], fill=C['TEAL'])
    # Volpe walks the counter pouring for each seat
    pours = [(6.0 + i * 1.3, SEATS[i][0]) for i in range(4)]
    rx, pouring = 160, None
    for i, (pt, sx) in enumerate(pours):
        if ts >= pt - 0.6: rx = lerp(rx, sx + 8, ease((ts - (pt - 0.6)) / 0.5))
        if pt <= ts < pt + 0.9: pouring = i
    if ts > pours[-1][0] + 1.2: rx = lerp(rx, 160, ease((ts - pours[-1][0] - 1.2) / 0.6))
    talking_mad = 30.3 <= t < 33.8
    flustered = (39.0 <= t < 42.0) or (48.4 <= t < 52.8)
    mood = 'hk_blush' if flustered else 'hk_mad' if talking_mad else 'hk_away' if pouring is not None else 'hk'
    hx, hy = hacker(im, rx, t, mood, vein=talking_mad or (48.4 <= t < 52.8),
                    typing=(pouring is None and not flustered), trip=trip)
    dr = ImageDraw.Draw(im); dr.fontmode = '1'
    if pouring is not None or 5.4 < ts < pours[-1][0] + 1:  # moka pot
        dr.polygon([(hx, hy), (hx + 8, hy), (hx + 6, hy - 4), (hx + 8, hy - 9), (hx, hy - 9), (hx + 2, hy - 4)], fill=C['GRY'], outline=C['BLK'])
        dr.rectangle([hx + 2, hy - 11, hx + 6, hy - 9], fill=C['BLK'])
        dr.line([hx - 2, hy - 7, hx, hy - 7], fill=C['GRY'])
        if pouring is not None:
            tx_ = SEATS[pouring][0] + 18
            for k in range(0, 30, 2):
                yy = hy - 2 + k * (116 - hy) / 30
                xx = hx - 2 + (tx_ - (hx - 2)) * (k / 30) ** 0.6
                if (k + int(t * 30)) % 4: dr.point((int(xx), int(yy)), C['BRN'] if k % 3 else C['AMBR'])
    for i, (sx, hair, jak) in enumerate(SEATS):
        if ts > pours[i][0] + 0.2 and not (12.0 <= ts < 13.4):
            cup(dr, sx + 18, 117, full=ts < 13.4)
            if ts < 13.4 or trip > 0:
                steam(dr, sx + 18, 110, t + i, col=C['LPINK'] if trip < 0.3 else [C['MAG'], C['CYAN'], C['GRN']][i % 3])
    for i, (sx, hair, jak) in enumerate(SEATS):
        bb = bob if (i + b) % 2 == 0 or trip > 0.3 else 0
        if trip > 0.4: bb = int(round(math.sin(t * 8 + i) * 2))
        guy(im, sx, 170, hair, jak, view='back', scale=3, bob=bb, cup=(12.0 <= ts < 13.4))
    dialog(dr, t, 30.3, 33.8, 'VOLPE', ['CHE VOLETE? NON HO TEMPO PER VOI,', '...MA LA NOTTE È LUNGA. SEDETEVI.'], C['CYAN'], ax=int(rx))
    dialog(dr, t, 34.0, 38.6, 'TU', ['QUATTRO ESPRESSI, NERI COME IL MARE,', "AMARI COME L'ALBA."], C['GRN'], ax=62)
    dialog(dr, t, 39.0, 42.0, 'VOLPE', ["NON È CHE LI HO FATTI PER TE!", 'È SOLO CHE IL CODICE ERA CALDO.'], C['CYAN'], ax=int(rx))
    dialog(dr, t, 43.8, 47.8, 'PUNK', ['OH... I MURI RESPIRANO,', 'E IL NEON CANTA IN MINORE.'], C['MAG'], ax=112)
    dialog(dr, t, 48.4, 52.8, 'VOLPE', ['OGNI SORSO È UN SOGNO, SCEMO...', 'E NON RINGRAZIARMI. MAI.'], C['CYAN'], ax=int(rx))
    if ts < 3.6: verb, act, cp = 'GUARDA', "GUARDA L'HACKER", path(ts, [(0, 160, 120), (1.5, rx, 80)])
    elif ts < 11.6: verb, act, cp = 'PARLA', 'ORDINA ESPRESSO', path(ts, [(3.6, rx, 80), (4.6, 160, 70), (11, 170, 65)])
    else: verb, act, cp = 'BEVI', 'BEVI ESPRESSO', path(ts, [(11.6, 170, 65), (11.9, 68, 112), (14, 70, 112), (23, 160, 90)])
    verbbar(dr, t, verb, act, scramble=trip)
    cursor(dr, *cp, t, max([c for c in [31.4, 34.6, 41.9] if c <= t], default=None))
    return im


# ================================================================ SCENE 4: breakdown
_FREEZE = None


def scene4(t):
    global _FREEZE
    if _FREEZE is None:
        _FREEZE = np.array(scene3(53.25).convert('RGB')).astype(np.float32).mean(2) / 255
    ts = t - 53.3
    g = _FREEZE * (1 - ease(ts / 2.6) * 0.8)
    a = np.zeros((H, W, 3), np.uint8); a[:] = C['BLK']
    a[g > BAYF * 0.6] = C['DPUR']
    a[g > 0.3 + BAYF * 0.5] = C['PUR']
    n = samp(N3, XX + ts * 20, YY - ts * 14)  # contours eat the frame
    a[np.abs((n * 7 + ts * 0.5) % 1 - 0.5) < 0.05 + ts * 0.04] = C['PLUM']
    im = Image.fromarray(a).convert('RGBA')
    dr = ImageDraw.Draw(im); dr.fontmode = '1'
    big = Image.new('RGBA', (24, 20), (0, 0, 0, 0)); bd = ImageDraw.Draw(big)
    bd.rectangle([4, 6, 16, 18], fill=C['BLK']); bd.rectangle([5, 7, 15, 18], fill=C['WHT'])
    bd.rectangle([5, 7, 15, 8], fill=C['BRN']); bd.rectangle([16, 9, 19, 14], outline=C['WHT'])
    im.alpha_composite(big.resize((72, 60), Image.NEAREST), (124, 80))
    for k in range(3): steam(dr, 145 + k * 10, 98, t * 0.6 + k, n=10, col=C['LPINK'], hgt=40)
    if int(ts * 2) % 2 == 0 and ts < 2.4: txt(dr, 148, 150, '...', C['GRY'])
    if ts > 2.1:
        s = 'CARICAMENTO SOGNO'
        txt(dr, 160 - tlen(s) // 2, 150, s[:int((ts - 2.1) * 30)], C['GRN'])
    return im


# ================================================================ SCENE 5: the trip
TITLE = glyph_big('NEON ARPEGGIO', C['WHT'])
TITLE_SH = glyph_big('NEON ARPEGGIO', C['MAG'])
ROOF = Image.new('RGBA', (W, H), (0, 0, 0, 0))
_rd = ImageDraw.Draw(ROOF)
_rd.rectangle([0, 148, W, H], fill=C['BLK'] + (255,)); _rd.line([0, 148, W, 148], fill=C['PLUM'] + (255,))
_rd.rectangle([20, 128, 40, 148], fill=C['BLK'] + (255,)); _rd.rectangle([280, 120, 286, 148], fill=C['BLK'] + (255,))
SKYCOLS = np.array([C['NAVY'], C['DPUR'], C['PUR'], C['PLUM'], C['MAG'], C['PINK'], C['ORNG'], C['AMBR'],
                    C['PINK'], C['PLUM'], C['PUR'], C['DPUR']], np.uint8)


def scene5(t):
    ts = t - 56.4
    b, ph = beat(t)
    # plasma sky: domain-warped noise, colour-cycled through the palette, Bayer-dithered
    w1 = samp(N2, XX * 1.5 + ts * 15, YY * 1.5)
    n = samp(N1, XX + w1 * 90 + ts * 10, YY + w1 * 90 - ts * 6)
    v = (n * 14 + ts * 0.8) % len(SKYCOLS)
    im = Image.fromarray(SKYCOLS[(np.floor(v + BAYF) % len(SKYCOLS)).astype(int)]).convert('RGBA')
    dr = ImageDraw.Draw(im); dr.fontmode = '1'
    k = 1 + (0.08 if ph < 0.2 else 0)  # the espresso moon pulses on the beat
    cx, cy = 160, 58
    for rr in range(3):
        r_ = (ph + rr) * 22 + 20
        if r_ < 90: dr.ellipse([cx - r_, cy - r_ * 0.6, cx + r_, cy + r_ * 0.6], outline=C['LPINK'] if rr == 0 else C['MAG'])
    w, h = int(22 * k), int(18 * k)
    dr.rectangle([cx - w - 1, cy - h // 2 - 1, cx + w + 1, cy + h + 1], fill=C['BLK'])
    dr.rectangle([cx - w, cy - h // 2, cx + w, cy + h], fill=C['WHT'])
    dr.rectangle([cx - w, cy - h // 2, cx + w, cy - h // 2 + 3], fill=C['BRN'])
    dr.line([cx - w, cy - h // 2, cx + w, cy - h // 2], fill=C['AMBR'])
    dr.ellipse([cx + w - 2, cy - 4, cx + w + 12, cy + 12], outline=C['WHT'], width=3)
    dr.line([cx - w - 8, cy + h + 2, cx + w + 8, cy + h + 2], fill=C['WHT'])
    for j in range(5): steam(dr, cx - 14 + j * 7, cy - h // 2 - 2, t + j * 0.3, n=12, col=C['LCYN'], hgt=45)
    im.alpha_composite(FAR, (-260, 15))
    im.alpha_composite(MID, (-320, 18))
    a = np.array(im)
    hi = HI[min(int(t * FPS), NF - 1)]  # windows flash with the arpeggio
    a[(samp(N3, XX * 7 + b * 31, YY * 7) > 1 - hi * 0.25) & (a[..., 0] < 30) & (YY > 90) & (BAYF < 0.3), :3] = C['CYAN']
    im = Image.fromarray(a)
    im.alpha_composite(ROOF)
    for i, (sx, hair, jak) in enumerate(SEATS):
        guy(im, sx + 10, 152, hair, jak, view='back', scale=2, bob=int(round(math.sin(t * 7 + i))))
    guy(im, 170, 152, HAIR_HK, HOOD_HK, view='hk_back', scale=2, bob=int(round(math.sin(t * 5))))
    dr = ImageDraw.Draw(im); dr.fontmode = '1'
    if t > 62.7:
        u = ease((t - 62.7) / 0.6)
        bw = int(240 * u)
        dr.rectangle([160 - bw // 2, 92, 160 + bw // 2, 144], fill=C['BLK'], outline=C['MAG'])
        if u > 0.9:
            tw = TITLE.size[0]
            im.alpha_composite(TITLE_SH, (160 - tw // 2 + 2 + (1 if ph < 0.15 else 0), 94))
            im.alpha_composite(TITLE, (160 - tw // 2, 93))
            s = 'TEXTURE ORGANICHE'
            txt(dr, 160 - tlen(s) // 2, 126, s[:int((t - 63.3) * 40)], C['CYAN'], None)
    if t > 64.2:
        s = 'AV MEDICINE SHOW'
        txt(dr, 160 - tlen(s) // 2, 152, s[:int((t - 64.2) * 30)], C['AMBR'])
    return im


# ================================================================ organic texture overlay + post
_prev = None


def overlay(a, t, amt, i):
    """Drifting contour veins, channel-rotated blobs and spores over every frame."""
    if amt <= 0: return a
    ons = ONS[i]
    w = samp(N2, XX * 0.7 + t * 9, YY * 0.7 - t * 4)
    n = samp(N1, XX * 0.8 + w * 70 + t * 13, YY * 0.8 + w * 70 + t * 5)
    f = n * 9 + t * 0.35
    dv = np.abs(f % 1 - 0.5)
    reg = samp(N3, XX * 0.5 - t * 6, YY * 0.5 + t * 3)
    vm = (dv < 0.025 + 0.05 * amt) & (BAYF < 0.25 + amt * 0.6) & (reg > 0.75 - amt * 0.7)
    cols = np.array([C['MAG'], C['CYAN'], C['GRN'], C['ORNG']], np.uint8)
    a[vm] = cols[np.floor(f).astype(int) % 4][vm]
    bm = n > 1 - (0.12 + 0.35 * amt) * (0.5 + 0.7 * ons)
    a[bm] = a[bm][:, [2, 0, 1]]
    gm = (samp(N3, XX * 9 + t * 300, YY * 9 - t * 200) > 0.93 - 0.05 * amt) & (BAYF < 0.2)
    a[gm] = C['LPINK']
    return a


def post(a, t, amt, i, trans):
    """Row tears, datamosh blocks from the previous frame, chroma split on kicks."""
    global _prev
    r = np.random.default_rng(i * 13 + 1)
    kk = amt * 0.6 + ONS[i] * 0.6 + trans * 2
    for _ in range(int(r.poisson(kk * 2))):
        y = int(r.integers(0, H - 4)); h = int(r.integers(1, 10))
        a[y:y + h] = np.roll(a[y:y + h], int(r.integers(-18, 18) * (0.4 + kk)), axis=1)
    if _prev is not None:
        for _ in range(int(r.poisson(kk * 6))):
            x = int(r.integers(0, W - 8)) // 8 * 8; y = int(r.integers(0, H - 8)) // 8 * 8
            dx, dy = int(r.integers(-3, 4)), int(r.integers(-2, 3))
            sx, sy = min(max(x + dx, 0), W - 8), min(max(y + dy, 0), H - 8)
            a[y:y + 8, x:x + 8] = _prev[sy:sy + 8, sx:sx + 8]
    if BASS[i] > 0.9 and amt > 0.15:
        s = 1 + int(BASS[i] > 1.1)
        a[..., 0] = np.roll(a[..., 0], s, axis=1); a[..., 2] = np.roll(a[..., 2], -s, axis=1)
    _prev = a.copy()
    return a


def quant(a):
    return np.array(Image.fromarray(a).quantize(palette=palimg, dither=Image.Dither.NONE).convert('RGB'))


def frame(i):
    t = i / FPS
    if t < CUTS[0]: im, amt = scene1(t), 0.12 + 0.1 * ease(t / 14.6)
    elif t < CUTS[1]: im, amt = scene2(t), 0.22
    elif t < CUTS[2]: im, amt = scene3(t), 0.25 + 0.75 * ease((t - 42.2) / 10.5)
    elif t < CUTS[3]: im, amt = scene4(t), 0.5 + 0.5 * ease((t - 53.3) / 3)
    else: im, amt = scene5(t), 0.55
    a = quant(overlay(np.array(im.convert('RGB')), t, amt, i))
    trans = max(max(0, 1 - abs(t - c) / 0.25) for c in CUTS)
    a = post(a, t, amt, i, trans)
    if t > 66.2: a[BAYF < (t - 66.2)] = C['BLK']  # Bayer dissolve to black
    if t < 0.4: a[BAYF > t / 0.4] = C['BLK']
    return a


def main():
    global CLEAN
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out', nargs='?', default='out/neon_arpeggio.mp4')
    ap.add_argument('--scale', type=int, default=6, help='integer upscale (6 = 1920x1080, 2 = 640x360)')
    ap.add_argument('--clean', action='store_true', help='omit the scripted cursor and verb bar')
    ap.add_argument('--crf', type=int, default=20)
    ap.add_argument('--preset', default='veryfast', help='x264 preset (slower = smaller file)')
    ap.add_argument('--stills', type=float, nargs='*', help='write PNGs at these timestamps instead')
    args = ap.parse_args()
    CLEAN = args.clean
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    if args.stills is not None:
        base = os.path.splitext(args.out)[0]
        targets = {int(s * FPS) for s in args.stills}
        for i in range(max(targets) + 1):  # run in order so the datamosh history is real
            a = frame(i)
            if i in targets:
                p = f'{base}_{i / FPS:05.1f}.png'
                Image.fromarray(a).resize((W * args.scale, H * args.scale), Image.NEAREST).save(p); print(p)
        return
    p = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
                          '-i', '-', '-i', SONG['path'], '-map', '0:v', '-map', '1:a',
                          '-vf', f'scale={W * args.scale}:{H * args.scale}:flags=neighbor',
                          '-c:v', 'libx264', '-preset', args.preset, '-tune', 'animation', '-crf', str(args.crf), '-pix_fmt', 'yuv420p',
                          '-c:a', 'copy', '-shortest', '-movflags', '+faststart', args.out], stdin=subprocess.PIPE)
    for i in range(NF):
        p.stdin.write(frame(i).tobytes())
        if i % 300 == 0: print(f'frame {i}/{NF}', flush=True)
    p.stdin.close(); p.wait()
    print('wrote', args.out)


if __name__ == '__main__':
    main()
