"""
main.py - Android lietotne (Kivy). Kamera redz maketa logu ar uzzīmēto piktogrammu;
lietotne to atpazīst (marker.py) un uz loga uzliek atdzīvinātu ainiņu (scenes.py).
Visa AR loģika ir ar_core.py, šeit ir tikai kamera un ekrāns.
"""
import cv2
import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.camera import Camera
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.utils import platform

from ar_core import Tracker

FPS = 24
CAMERA_RESOLUTION = (640, 480)   # ja piktogramma ir maza/tālu, pamēģini (1280, 720)

# Kameras kadra orientācija. Ja uz telefona attēls ir sagriezts/otrādi - pamaini šos divus.
FLIP_VERTICAL = True
FRAME_ROTATE = cv2.ROTATE_90_CLOCKWISE if platform == "android" else None
# citas vērtības: None, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180

HINT = "Pavērsiet kameru uz loga piktogrammu"


class ARWindowView(FloatLayout):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.cam = None
        self.tex = None
        self.tracker = Tracker()

        self.view = Image(fit_mode="contain")
        self.add_widget(self.view)
        self.caption = Label(
            text="", font_size="22sp", size_hint=(1, None), height="64dp",
            pos_hint={"x": 0, "y": 0}, color=(1, 1, 1, 1),
            outline_width=2, outline_color=(0, 0, 0, 1))
        self.add_widget(self.caption)
        self.hint = Label(
            text=HINT, font_size="18sp", size_hint=(1, None), height="40dp",
            pos_hint={"x": 0, "top": 1}, color=(1, 1, 1, 1),
            outline_width=2, outline_color=(0, 0, 0, 1))
        self.add_widget(self.hint)

    def start_camera(self):
        if self.cam is not None:
            return
        # Kamera nav redzama (opacity=0): attēlu zīmējam paši, ar animāciju.
        self.cam = Camera(index=0, resolution=CAMERA_RESOLUTION, play=True,
                          size_hint=(None, None), size=(2, 2), opacity=0)
        self.add_widget(self.cam)
        Clock.schedule_interval(self.update, 1.0 / FPS)

    def grab_frame(self):
        tex = self.cam.texture
        if tex is None:
            return None
        w, h = tex.size
        buf = np.frombuffer(tex.pixels, np.uint8)
        if buf.size != w * h * 4:
            return None
        bgr = cv2.cvtColor(buf.reshape(h, w, 4), cv2.COLOR_RGBA2BGR)
        if FLIP_VERTICAL:
            bgr = cv2.flip(bgr, 0)
        if FRAME_ROTATE is not None:
            bgr = cv2.rotate(bgr, FRAME_ROTATE)
        return np.ascontiguousarray(bgr)

    def update(self, dt):
        frame = self.grab_frame()
        if frame is None:
            return
        out, text = self.tracker.process(frame)
        self.caption.text = text
        self.hint.text = "" if text else HINT
        self.show(out)

    def show(self, bgr):
        h, w = bgr.shape[:2]
        if self.tex is None or tuple(self.tex.size) != (w, h):
            self.tex = Texture.create(size=(w, h), colorfmt="rgb")
            self.tex.flip_vertical()
            self.view.texture = self.tex
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.tex.blit_buffer(rgb.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
        self.view.canvas.ask_update()


class WindowARApp(App):
    title = "Loga AR"

    def build(self):
        self.ar = ARWindowView()
        return self.ar

    def on_start(self):
        if platform == "android":
            from android.permissions import Permission, request_permissions

            def callback(perms, grants):
                if grants and all(grants):
                    Clock.schedule_once(lambda dt: self.ar.start_camera(), 0)
                else:
                    self.ar.hint.text = "Nepieciešama kameras atļauja"

            request_permissions([Permission.CAMERA], callback)
        else:
            self.ar.start_camera()

    def on_pause(self):
        if self.ar.cam:
            self.ar.cam.play = False
        return True

    def on_resume(self):
        if self.ar.cam:
            self.ar.cam.play = True


if __name__ == "__main__":
    WindowARApp().run()
