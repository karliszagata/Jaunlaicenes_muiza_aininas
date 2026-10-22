"""
marker.py - piktogrammas (attēla) atpazīšana kameras kadrā.

Princips (līdzīgi kā ArUco marķieriem, bet ar tavu zīmējumu):
  1. kadrā tiek meklēti četrstūri (tavai piktogrammai ir biezs kvadrātveida rāmis);
  2. katrs četrstūris tiek "iztaisnots" līdz kvadrātam;
  3. iztaisnotais attēls tiek salīdzināts ar marker.png (arī pagriezts par 90/180/270°);
  4. ja līdzība pietiekami augsta - marķieris atrasts, un ir zināmi tā 4 stūri.

Līniju zīmējumam tas ir stabilāks nekā ORB/SIFT pazīmju meklēšana, jo piktogrammā
gandrīz nav tekstūras.
"""
import cv2
import numpy as np

REF_N = 96            # salīdzināšanas izmērs pikseļos
MIN_SCORE = 0.75      # minimālā līdzība (0..1). Mazināt, ja marķieris netiek atpazīts; palielināt, ja parādās viltus atpazīšana
_DST = np.float32([[0, 0], [REF_N - 1, 0], [REF_N - 1, REF_N - 1], [0, REF_N - 1]])


def order_quad(pts):
    """Sakārto 4 punktus: TL, TR, BR, BL (pulksteņrādītāja virzienā)."""
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]
    return np.roll(pts, -int(np.argmin(pts.sum(axis=1))), axis=0)


class MarkerSet:
    def __init__(self, files):
        """files: {nosaukums: ceļš_līdz_attēlam}"""
        self.refs = {}
        for name, path in files.items():
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(path)
            img = cv2.resize(img, (REF_N, REF_N), interpolation=cv2.INTER_AREA)
            img = cv2.GaussianBlur(img, (5, 5), 0)
            self.refs[name] = [np.ascontiguousarray(np.rot90(img, k)) for k in range(4)]

    def find(self, bgr, max_w=640, min_score=MIN_SCORE):
        """Atgriež (nosaukums, četrstūris[4x2] kadra koordinātēs, līdzība) vai (None, None, 0)."""
        h, w = bgr.shape[:2]
        sc = 1.0
        small = bgr
        if w > max_w:
            sc = max_w / float(w)
            small = cv2.resize(bgr, (max_w, int(h * sc)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 8)
        cnts = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

        sh, sw = gray.shape
        cands = []
        for c in cnts:
            a = cv2.contourArea(c)
            if a < 0.004 * sw * sh or a > 0.95 * sw * sh:
                continue
            ap = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
            if len(ap) != 4 or not cv2.isContourConvex(ap):
                continue
            cands.append((a, ap.reshape(4, 2)))
        cands.sort(key=lambda x: -x[0])

        best = (None, None, 0.0)
        best_s = min_score
        for _, q in cands[:25]:
            quad = order_quad(q)
            sides = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)
            if sides.min() < 12 or sides.max() / sides.min() > 3.0:
                continue
            m = cv2.getPerspectiveTransform(quad, _DST)
            rect = cv2.GaussianBlur(cv2.warpPerspective(blur, m, (REF_N, REF_N)), (5, 5), 0)
            for name, rots in self.refs.items():
                for k, ref in enumerate(rots):
                    s = float(cv2.matchTemplate(rect, ref, cv2.TM_CCOEFF_NORMED)[0, 0])
                    if s > best_s:
                        best_s = s
                        best = (name, np.roll(quad, k, axis=0) / sc, s)
        return best
