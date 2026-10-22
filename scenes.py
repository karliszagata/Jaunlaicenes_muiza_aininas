"""
scenes.py - sadzīves ainiņas, kas zīmētas TĀDĀ PAŠĀ stilā kā piktogramma marker.png
(melnas, vienāda biezuma līnijas uz balta fona, kvadrātveida rāmis).

Aina ir klase ar:
    marker           - marķiera attēla fails (piktogramma, ko atpazīt uz loga)
    render(t) -> BGR - attēls (S x S) laika momentā t sekundēs. Attēla ārmala sakrīt
                       ar piktogrammas rāmja ārmalu, tāpēc to var tieši uzlikt marķierim.
    caption(t)       - paskaidrojošs teksts latviski

Koordinātes zemāk ir oriģinālā faila Putra_grafisks_01.jpg pikseļos (rāmis: x 365..1042,
y 45..723); funkcija P() tās pārrēķina uz ainas attēlu.

Pārbaude datorā:
    python scenes.py                    - rāda animāciju logā (ESC - iziet)
    python scenes.py --sheet lapa.png   - saglabā kontaktlapu ar atslēgas kadriem
"""
import math
import sys

import cv2
import numpy as np

# ------------------------------------------------------------------ iestatījumi
COLOR = True          # True: ūdens/putra/liesmas ar maigu krāsu; False: tikai melnbalts
S = 600               # ainas attēla malas garums pikseļos
X0, Y0, W0 = 365.0, 45.0, 678.0   # piktogrammas rāmja ārmala oriģinālā attēlā
SC = S / W0
STROKE = max(2, int(round(10 * SC)))  # līnijas biezums (oriģinālā ~10 px)

BLACK = (25, 25, 25)
WHITE = (255, 255, 255)
if COLOR:
    WATER, PORRIDGE, GRAIN, FLAME = (240, 200, 150), (165, 220, 245), (45, 150, 215), (50, 150, 250)
else:
    WATER, PORRIDGE, GRAIN, FLAME = (225, 225, 225), (200, 200, 200), (60, 60, 60), (150, 150, 150)


# ------------------------------------------------------------------ palīgfunkcijas
def clamp01(x):
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def ease(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def lerp(a, b, k):
    return a + (b - a) * k


def mix(c1, c2, k):
    return tuple(int(lerp(a, b, k)) for a, b in zip(c1, c2))


def P(x, y):
    """Oriģinālā attēla koordinātes -> ainas pikseļi (float)."""
    return ((x - X0) * SC, (y - Y0) * SC)


def ipts(pts):
    return np.array([P(x, y) for x, y in pts], np.float32).round().astype(np.int32)


def ip(x, y):
    p = P(x, y)
    return (int(round(p[0])), int(round(p[1])))


def stroke(img, pts, closed=False, th=STROKE, color=BLACK):
    a = ipts(pts)
    cv2.polylines(img, [a], closed, color, th, cv2.LINE_AA)
    if not closed:
        for p in (a[0], a[-1]):
            cv2.circle(img, (int(p[0]), int(p[1])), max(1, th // 2), color, -1, cv2.LINE_AA)


def fill(img, pts, color):
    cv2.fillPoly(img, [ipts(pts)], color, cv2.LINE_AA)


def rot(pts, cx, cy, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return [(x * c - y * s + cx, x * s + y * c + cy) for x, y in pts]


def arc(cx, cy, rx, ry, a0, a1, n=14):
    return [(cx + rx * math.cos(math.radians(a)), cy + ry * math.sin(math.radians(a)))
            for a in np.linspace(a0, a1, n)]


# ------------------------------------------------------------------ piktogrammas ģeometrija
POT_L, POT_R, POT_TOP, POT_BOT, POT_RAD = 551.0, 856.5, 398.5, 611.5, 52.0
POT_OUTLINE = ([(POT_L, POT_TOP), (POT_L, POT_BOT - POT_RAD)]
               + arc(POT_L + POT_RAD, POT_BOT - POT_RAD, POT_RAD, POT_RAD, 180, 90)
               + arc(POT_R - POT_RAD, POT_BOT - POT_RAD, POT_RAD, POT_RAD, 90, 0)
               + [(POT_R, POT_TOP)])
FLOOR = 606.0          # katla iekšpuses apakša
STOVE_Y = 640.5

# tvaika līkne (no piktogrammas, augša -> apakša)
_STEAM_RAW = [(719, 180), (716, 194), (709, 205), (699, 215), (687, 223), (673, 231), (660, 238),
              (647, 249), (640, 262), (640, 276), (647, 289), (655, 300), (663, 307), (670, 314),
              (678, 320), (686, 331), (693, 343), (696, 357), (697, 360)]


def _resample(pts, n):
    p = np.array(pts, np.float32)
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    u = np.linspace(0, d[-1], n)
    return np.c_[np.interp(u, d, p[:, 0]), np.interp(u, d, p[:, 1])]


STEAM = _resample(_STEAM_RAW, 70)

# karote (koordinātes attiecībā pret karotes galvas centru, pozā kā piktogrammā)
SPOON_C0 = (779.5, 383.5)
HEAD_R = 53.0
_L0, _L1 = (28.5, -46.5), (168.5, -245.5)
_R0, _R1 = (42.5, -33.5), (207.5, -223.5)
_K, _KR = (187.5, -242.5), 23.0
_A_L = math.degrees(math.atan2(_L1[1] - _K[1], _L1[0] - _K[0]))
_A_R = math.degrees(math.atan2(_R1[1] - _K[1], _R1[0] - _K[0]))
_CAP = [(_K[0] + _KR * math.cos(math.radians(a)), _K[1] + _KR * math.sin(math.radians(a)))
        for a in np.linspace(_A_L, _A_R, 14)]
HANDLE_OPEN = [_L0, _L1] + _CAP + [_R1, _R0]
_ANG_R0 = math.degrees(math.atan2(_R0[1], _R0[0]))
_ANG_L0 = math.degrees(math.atan2(_L0[1], _L0[0]))
DIR = (math.cos(math.radians(-52.3)), math.sin(math.radians(-52.3)))  # karotes ass virziens

# šķīvis (parādās labajā apakšā)
BOWL_C, BOWL_RX, BOWL_RY = (940.0, 575.0), 78.0, 61.0
BOWL_PTS = arc(BOWL_C[0], BOWL_C[1], BOWL_RX, BOWL_RY, 0, 180, 30)


class Scene:
    duration = 30.0
    marker = None
    stages = []

    def caption(self, t):
        t = t % self.duration
        text = ""
        for start, label in self.stages:
            if t >= start:
                text = label
        return text

    def render(self, t):
        raise NotImplementedError


class PorridgeScene(Scene):
    duration = 40.0
    marker = "marker.png"
    stages = [
        (0.0, "Gatavojam putru"),
        (2.8, "Ielejam ūdeni katlā"),
        (8.0, "Ieslēdzam plīti - ūdens sasilst"),
        (12.0, "Iebērām putraimus"),
        (17.5, "Vārām un maisām"),
        (29.0, "Liekam putru šķīvī"),
        (36.0, "Putra gatava! Labu apetīti!"),
    ]
    FADE = 0.8   # cik sekundes ilgst pāreja uz cikla sākumu

    def __init__(self):
        self.potmask = np.zeros((S, S), np.uint8)
        cv2.fillPoly(self.potmask, [ipts(POT_OUTLINE)], 255)
        self.inner = np.zeros((S, S), np.uint8)
        cv2.rectangle(self.inner, (STROKE, STROKE), (S - 1 - STROKE, S - 1 - STROKE), 255, -1)

    # ---------------------------------------------------------- darbību laika grafiks
    @staticmethod
    def spoon_state(t):
        """(galvas centrs, pagrieziens grādos, piepildījums 0..1) vai None, ja karotes nav."""
        if t < 1.6:
            return SPOON_C0, 0.0, 0.0
        if t < 2.8:                               # karote tiek izcelta
            k = ease((t - 1.6) / 1.2) * 700
            return (SPOON_C0[0] + DIR[0] * k, SPOON_C0[1] + DIR[1] * k), 0.0, 0.0
        if t < 17.5:
            return None
        if t < 28.0:                              # maisīšana
            A = 1 - ease((t - 27.0) / 1.0)
            w = (t - 18.5) * 3.2
            e = (1 - ease((t - 17.5) / 1.0)) * 650
            c = (704 + 35 * A * math.cos(w) + DIR[0] * e, 515 + 15 * A * math.sin(w) + DIR[1] * e)
            return c, 6 * A * math.sin(w), 0.0
        kf = [(28.0, 704, 515, 0), (29.2, 650, 545, 0), (30.4, 660, 330, -5),
              (31.6, 900, 380, -22), (32.4, 940, 490, -30), (34.6, 940, 490, -30),
              (35.3, 900, 380, -22), (36.0, SPOON_C0[0], SPOON_C0[1], 0)]
        loaded = ease((t - 29.2) / 1.2) if t < 32.6 else 1 - ease((t - 32.6) / 1.8)
        if t >= kf[-1][0]:
            return SPOON_C0, 0.0, 0.0
        for (t0, x0, y0, d0), (t1, x1, y1, d1) in zip(kf, kf[1:]):
            if t0 <= t < t1:
                k = ease((t - t0) / (t1 - t0))
                return (lerp(x0, x1, k), lerp(y0, y1, k)), lerp(d0, d1, k), loaded
        return None

    @staticmethod
    def steam_amount(t):
        if t < 1.6:
            return 1.0
        if t < 2.8:
            return 1 - ease((t - 1.6) / 1.2)
        if t < 10:
            return 0.0
        return ease((t - 10) / 3)

    # ---------------------------------------------------------- zīmēšanas daļas
    def draw_steam(self, img, amount, t, amp, scale=1.0, anchor=None):
        if amount < 0.03:
            return
        n = len(STEAM)
        k0 = int((1 - amount) * (n - 1))
        bx, by = STEAM[-1]
        pts = []
        for i in range(k0, n):
            s = i / (n - 1)
            x = STEAM[i][0] + amp * (1 - s) * math.sin(t * 3.0 + s * 10)
            y = STEAM[i][1]
            if anchor is not None:
                x = anchor[0] + (x - bx) * scale
                y = anchor[1] + (y - by) * scale
            pts.append((x, y))
        stroke(img, pts)

    def draw_spoon(self, img, c, delta, loaded):
        cx, cy = c
        hc = ip(cx, cy)
        hr = int(round(HEAD_R * SC))
        fill(img, rot(HANDLE_OPEN, cx, cy, delta), WHITE)
        cv2.circle(img, hc, hr, WHITE, -1, cv2.LINE_AA)
        if loaded > 0.01:
            level = cy + HEAD_R - loaded * 2 * HEAD_R
            ys = max(0, int(P(0, level)[1]))
            layer = img.copy()
            cv2.circle(layer, hc, hr, PORRIDGE, -1, cv2.LINE_AA)
            img[ys:] = layer[ys:]
        stroke(img, rot(HANDLE_OPEN, cx, cy, delta))
        cv2.ellipse(img, hc, (hr, hr), 0, _ANG_R0 + delta, _ANG_L0 + 360 + delta, BLACK, STROKE, cv2.LINE_AA)
        for a in (_ANG_R0 + delta, _ANG_L0 + delta):
            q = (hc[0] + hr * math.cos(math.radians(a)), hc[1] + hr * math.sin(math.radians(a)))
            cv2.circle(img, (int(q[0]), int(q[1])), STROKE // 2, BLACK, -1, cv2.LINE_AA)

    def draw_jug(self, img, t, surf):
        vis = ease((t - 2.6) / 0.8) * (1 - ease((t - 7.4) / 0.8))
        if vis <= 0:
            return
        tilt = 55 * ease((t - 3.2) / 0.6) * (1 - ease((t - 6.9) / 0.5))
        cx, cy = 510, 280 - (1 - vis) * 230
        body = [(-45, -55), (30, -55), (58, -72), (50, -40), (45, 60), (-45, 60)]
        handle = [(-45 + 26 * math.cos(math.radians(a)), 34 * math.sin(math.radians(a)))
                  for a in np.linspace(90, 270, 12)]
        wb, wh = rot(body, cx, cy, tilt), rot(handle, cx, cy, tilt)
        tip = rot([(58, -72)], cx, cy, tilt)[0]
        if 3.6 < t < 7.1:                          # ūdens strūkla
            p0, p1 = ip(tip[0], tip[1]), ip(tip[0] + 4, surf)
            cv2.line(img, p0, p1, BLACK, 10, cv2.LINE_AA)
            cv2.line(img, p0, p1, WATER, 5, cv2.LINE_AA)
        fill(img, wb, WHITE)
        stroke(img, wb, closed=True)
        stroke(img, wh)

    def draw_jar(self, img, t, surf):
        vis = ease((t - 12.0) / 0.8) * (1 - ease((t - 16.7) / 0.8))
        if vis <= 0:
            return
        tilt = -50 * ease((t - 12.6) / 0.6) * (1 - ease((t - 16.2) / 0.5))
        cx, cy = 800, 255 - (1 - vis) * 230
        body = [(-40, -45), (40, -45), (40, 60), (-40, 60)]
        drain = ease((t - 13.2) / 3.0)
        ft = -15 + 70 * drain
        wb = rot(body, cx, cy, tilt)
        fill(img, wb, WHITE)
        if ft < 58:
            fill(img, rot([(-40, ft), (40, ft), (40, 60), (-40, 60)], cx, cy, tilt), GRAIN)
        stroke(img, wb, closed=True)
        stroke(img, rot([(-40, -33), (40, -33)], cx, cy, tilt), th=max(2, STROKE // 2))
        sx, sy = rot([(-40, -45)], cx, cy, tilt)[0]
        if 13.0 < t < 17.0:
            for i in range(60):
                life = (t - (13.0 + i * 0.055)) / 0.9
                if 0 <= life <= 1:
                    x = sx - 12 * life + 7 * math.sin(i * 7.3)
                    y = sy + life * (surf - sy)
                    cv2.circle(img, ip(x, y), 3, GRAIN if COLOR else BLACK, -1, cv2.LINE_AA)

    def draw_bowl(self, img, t):
        if t < 27.0:
            return
        dx = 220 * (1 - ease((t - 27.0) / 1.0))
        pts = [(x + dx, y) for x, y in BOWL_PTS]
        fill(img, pts, WHITE)
        fb = 0.85 * ease((t - 32.8) / 1.8)
        if fb > 0.01:
            ys = BOWL_C[1] + BOWL_RY - fb * 62
            m = np.zeros((S, S), np.uint8)
            cv2.fillPoly(m, [ipts(pts)], 255)
            m[:max(0, int(P(0, ys)[1]))] = 0
            img[m > 0] = PORRIDGE
            hw = BOWL_RX * math.sqrt(max(0.0, 1 - ((ys - BOWL_C[1]) / BOWL_RY) ** 2))
            stroke(img, [(BOWL_C[0] + dx - hw, ys), (BOWL_C[0] + dx + hw, ys)], th=max(2, STROKE // 2))
        stroke(img, pts, closed=True)
        stroke(img, [(862 + dx, STOVE_Y), (1018 + dx, STOVE_Y)])
        if t >= 36.0:
            self.draw_steam(img, ease((t - 36.0) / 1.5), t, 4.0, 0.5, (BOWL_C[0], 560))

    # ---------------------------------------------------------- galvenā zīmēšana
    def _render(self, t):
        img = np.full((S, S, 3), 255, np.uint8)

        stroke(img, [(552, STOVE_Y), (856, STOVE_Y)])                    # plīts līnija

        heat = ease((t - 8.0) / 1.0) * (1 - ease((t - 30.0) / 1.5))      # liesmas
        if heat > 0.02:
            for i in range(5):
                x = 620 + i * 42
                h = (10 + 6 * math.sin(t * 13 + i * 1.9)) * heat
                fl = [(x - 9, 634), (x - 4, 634 - h * 0.55), (x + 1.5 * math.sin(t * 9 + i), 634 - h),
                      (x + 4, 634 - h * 0.55), (x + 9, 634)]
                fill(img, fl, FLAME)
                stroke(img, fl, closed=True, th=max(2, STROKE // 3))

        self.draw_bowl(img, t)

        # šķidruma līmenis katlā
        hw = 117.0 * ease((t - 3.6) / 3.6)
        g = ease((t - 13.5) / 4.0)
        drop = 30.0 * ease((t - 29.2) / 1.2)
        h = hw + 8 * g - drop
        surf = FLOOR - h
        boil = ease((t - 9.5) / 2.5) * (1 - ease((t - 28.0) / 1.5))
        if h > 1.5:
            amp = 1.0 + 3.5 * boil
            xs = np.arange(POT_L, POT_R + 1, 12.0)
            wave = [(x, surf + amp * math.sin(x * 0.05 + t * 4)) for x in xs]
            poly = wave + [(POT_R, POT_BOT), (POT_L, POT_BOT)]
            lm = np.zeros((S, S), np.uint8)
            cv2.fillPoly(lm, [ipts(poly)], 255)
            lm = cv2.bitwise_and(lm, self.potmask)
            img[lm > 0] = mix(WATER, PORRIDGE, g)
            stroke(img, wave, th=max(2, STROKE // 2))
            for i in range(int(12 * boil)):
                ph = (t * 0.6 + i * 0.37) % 1.0
                bx = 580 + (i * 61) % 250
                by = FLOOR - ph * max(h - 6, 1)
                if h > 25 and by > surf + 6:
                    cv2.circle(img, ip(bx, by), int((4 + (i % 3) * 2) * SC), BLACK,
                               max(2, STROKE // 3), cv2.LINE_AA)

        stroke(img, POT_OUTLINE, closed=True)                             # katls
        stroke(img, [(490, 434.5), (POT_L, 434.5)])
        stroke(img, [(POT_R, 434.5), (917, 434.5)])

        self.draw_jug(img, t, surf)
        self.draw_jar(img, t, surf)

        st = self.spoon_state(t)
        if st is not None:
            c, delta, loaded = st
            if 32.6 <= t < 34.4:                                          # putras strūkla šķīvī
                fb = 0.85 * ease((t - 32.8) / 1.8)
                ys = BOWL_C[1] + BOWL_RY - fb * 62
                p0, p1 = ip(c[0], c[1] + HEAD_R), ip(c[0], ys)
                cv2.line(img, p0, p1, BLACK, 10, cv2.LINE_AA)
                cv2.line(img, p0, p1, PORRIDGE, 5, cv2.LINE_AA)
            self.draw_spoon(img, c, delta, loaded)

        amp = 6.0 * ease((t - 1.0) / 1.0)
        self.draw_steam(img, self.steam_amount(t), t, amp)

        # apgriež visu, kas iziet ārpus rāmja, un uzzīmē rāmi
        img[self.inner == 0] = 255
        th = STROKE
        cv2.rectangle(img, (th // 2, th // 2), (S - 1 - th // 2, S - 1 - th // 2), BLACK, th, cv2.LINE_AA)
        return img

    def render(self, t):
        t = t % self.duration
        if t > self.duration - self.FADE:
            a = (t - (self.duration - self.FADE)) / self.FADE
            return cv2.addWeighted(self._render(t), 1 - a, self._render(0.0), a, 0)
        return self._render(t)


# Reģistrs: nosaukums -> ainas klase (katrai ainai savs marķiera fails)
SCENES = {
    "PUTRA": PorridgeScene,
}


if __name__ == "__main__":
    scene = PorridgeScene()
    if "--sheet" in sys.argv:
        out = sys.argv[sys.argv.index("--sheet") + 1]
        times = [0.5, 2.2, 5.5, 9.5, 14.0, 20.0, 30.0, 31.8, 33.5, 38.0]
        tiles = []
        for tt in times:
            im = cv2.resize(scene.render(tt), (300, 300), interpolation=cv2.INTER_AREA)
            cv2.putText(im, f"{tt}s", (8, 292), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
            tiles.append(im)
        rows = [np.hstack(tiles[i:i + 5]) for i in range(0, 10, 5)]
        cv2.imwrite(out, np.vstack(rows))
        print("saglabāts", out)
    else:
        import time
        t0, last = time.time(), None
        while True:
            t = time.time() - t0
            cv2.imshow("Ainina", scene.render(t))
            cap = scene.caption(t)
            if cap != last:
                print(f"[{t % scene.duration:5.1f} s] {cap}")
                last = cap
            if cv2.waitKey(33) & 0xFF == 27:
                break
        cv2.destroyAllWindows()
