"""
Pārbaude datorā, bez Android un bez kameras:

    python test_image.py foto.jpg                 - atrod piktogrammu fotogrāfijā un uzliek animāciju
    python test_image.py foto.jpg --t 20          - animācijas laiks sekundēs (noklusēti 20)
    python test_image.py foto.jpg --out rez.png   - kur saglabāt rezultātu

Noderīgi, lai pārbaudītu, vai tava maketa loga foto tiek atpazīts (skaties izvadīto 'līdzību').
"""
import sys

import cv2

from ar_core import Tracker
from marker import MIN_SCORE

path = sys.argv[1]
t = float(sys.argv[sys.argv.index("--t") + 1]) if "--t" in sys.argv else 20.0
out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "rezultats.png"

img = cv2.imread(path)
if img is None:
    sys.exit("Nevar nolasīt attēlu: " + path)
tr = Tracker()
name, quad, score = tr.markers.find(img, min_score=0.0)
print(f"labākā līdzība: {score:.2f} (slieksnis {MIN_SCORE})")
if score < MIN_SCORE:
    print("Piktogramma NAV atpazīta. Pamēģini labāku apgaismojumu / lielāku zīmējumu, "
          "vai samazini MIN_SCORE failā marker.py.")
res, cap = tr.process(img, t=t, detect_every=1)
print("paraksts:", cap)
cv2.imwrite(out, res)
print("saglabāts:", out)
