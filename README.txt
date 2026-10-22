Loga AR - piktogramma -> atdzīvināta ainiņa

Faili
  main.py        Android lietotne (Kivy)
  ar_core.py     AR loģika (atpazīšana, izsekošana, uzlikšana)
  marker.py      piktogrammas atpazīšana
  scenes.py      ainiņas (putras vārīšana), zīmētas piktogrammas stilā
  marker.png     tava piktogramma (apgriezta līdz rāmim)
  test_image.py  pārbaude datorā ar fotogrāfiju
  buildozer.spec, .github/workflows/build-apk.yml   APK būvēšana

Pārbaude datorā
  pip install opencv-python numpy
  python scenes.py                      (animācija logā)
  python test_image.py foto.jpg         (piktogrammas atrašana fotogrāfijā)
