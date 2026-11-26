"""Synthetic side-view car drawing - used for smoke tests and for trying the UI without a real photo.

It is NOT a photo; Stable Diffusion results on it are only a pipeline check. Use real car photos
(see README) for meaningful variants.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def make_synthetic_car(width: int = 768, height: int = 432) -> Image.Image:
    img = np.full((height, width, 3), (205, 214, 224), dtype=np.uint8)
    cv2.rectangle(img, (0, int(height * 0.80)), (width, height), (95, 98, 104), -1)  # road
    body = np.array([[0.10, 0.70], [0.12, 0.54], [0.30, 0.50], [0.38, 0.30], [0.66, 0.30],
                     [0.78, 0.50], [0.92, 0.55], [0.93, 0.70]], dtype=np.float32)
    pts = (body * [width, height]).astype(np.int32)
    cv2.fillPoly(img, [pts], (170, 30, 40))
    cv2.polylines(img, [pts], True, (60, 10, 15), 3)
    win = np.array([[0.40, 0.33], [0.64, 0.33], [0.73, 0.49], [0.33, 0.49]], dtype=np.float32)
    cv2.fillPoly(img, [(win * [width, height]).astype(np.int32)], (150, 175, 195))
    for cx in (0.27, 0.76):
        c = (int(cx * width), int(0.72 * height))
        r = int(0.085 * width)
        cv2.circle(img, c, r, (25, 25, 28), -1)
        cv2.circle(img, c, int(r * 0.68), (185, 188, 192), -1)
        cv2.circle(img, c, int(r * 0.18), (60, 60, 64), -1)
        for k in range(5):
            ang = k * 2 * np.pi / 5
            cv2.line(img, c, (int(c[0] + r * 0.66 * np.cos(ang)), int(c[1] + r * 0.66 * np.sin(ang))),
                     (90, 92, 96), 4)
    return Image.fromarray(img)
