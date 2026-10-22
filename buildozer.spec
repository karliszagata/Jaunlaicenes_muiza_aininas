[app]
title = Loga AR
package.name = logaar
package.domain = lv.example
source.dir = .
source.include_exts = py,png,jpg
source.exclude_dirs = .github,bin,.buildozer
version = 0.2

requirements = python3,kivy==2.3.0,numpy,opencv,android

orientation = portrait
fullscreen = 0

android.permissions = CAMERA
android.api = 33
android.minapi = 24
android.ndk = 25b
android.ndk_api = 24
android.archs = arm64-v8a
android.accept_sdk_license = True

p4a.branch = v2024.01.21

[buildozer]
log_level = 2
warn_on_root = 1
