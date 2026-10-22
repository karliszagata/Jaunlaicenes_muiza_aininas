"""
ar_core.py - AR loģika bez Kivy atkarības: piktogrammas atrašana kadrā, izsekošana,
animācijas perspektīva uzlikšana. To izmanto gan Android lietotne (main.py),
gan testa skripts datorā (test_image.py).
"""
import os
import time

import cv2
import numpy as np

from marker import MarkerSet
from scenes import SCENES

HERE = os.path.dirname(os.path.abspath(__file__))

WINDOW_SCALE = 1.0     # animācijas izmērs attiecībā pret atrasto piktogrammu (1.0 = tieši uz tās)
OPACITY = 1.0          # 1.0 = animācija pilnībā nosedz zīmējumu uz loga
DETECT_EVERY = 2       # piktogrammas meklēšana katrā n-tajā kadrā
MAX_LOST = 12          # cik kadrus turēt animāciju pēc piktogrammas pazušanas
SMOOTH = 0.6           # stūru nogludināšana (0..1, lielāks = ātrāk seko kustībai)


def overlay(frame, quad, scene_img, opacity=OPACITY):
    """Perspektīvi uzliek scene_img uz kadra četrstūrī quad (TL, TR, BR, BL)."""
    h, w = frame.shape[:2]
    s = scene_img.shape[0]
    src = np.float32([[0, 0], [s, 0], [s, s], [0, s]])
    m = cv2.getPerspectiveTransform(src, np.asarray(quad, np.float32))
    warped = cv2.warpPerspective(scene_img, m, (w, h), flags=cv2.INTER_AREA if False else cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((s, s), 255, np.uint8), m, (w, h))
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    alpha = (mask.astype(np.float32) / 255.0 * opacity)[..., None]
    out = frame.astype(np.float32) * (1 - alpha) + warped.astype(np.float32) * alpha
    return out.astype(np.uint8)


class Tracker:
    def __init__(self):
        files = {name: os.path.join(HERE, cls.marker) for name, cls in SCENES.items()}
        self.markers = MarkerSet(files)
        self.n = 0
        self.quad = None
        self.lost = 0
        self.scene = None
        self.scene_name = None
        self.t0 = 0.0
        self.score = 0.0

    def _reset(self):
        self.quad = None
        self.scene = None
        self.scene_name = None

    def process(self, frame, t=None, detect_every=DETECT_EVERY):
        """Atgriež (kadrs_ar_animāciju, paraksts). t - ainas laiks (None = reālais laiks)."""
        self.n += 1
        if self.quad is None or self.n % detect_every == 0:
            name, quad, score = self.markers.find(frame)
            if name is None:
                self.lost += 1
                if self.lost > MAX_LOST:
                    self._reset()
            else:
                self.lost = 0
                self.score = score
                if self.quad is not None and name == self.scene_name:
                    size = np.linalg.norm(quad[0] - quad[2]) + 1e-6
                    jump = np.linalg.norm(quad - self.quad, axis=1).mean() / size
                    self.quad = quad if jump > 0.3 else SMOOTH * quad + (1 - SMOOTH) * self.quad
                else:
                    self.quad = quad
                if name != self.scene_name:
                    self.scene = SCENES[name]()
                    self.scene_name = name
                    self.t0 = time.time()

        if self.quad is None or self.scene is None:
            return frame, ""
        if t is None:
            t = time.time() - self.t0
        c = self.quad.mean(axis=0)
        big = c + (self.quad - c) * WINDOW_SCALE
        return overlay(frame, big, self.scene.render(t)), self.scene.caption(t)
