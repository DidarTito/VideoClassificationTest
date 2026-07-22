"""Generates a short synthetic test clip at videos/test.mp4 (moving pattern,
64 frames @ 320x240) so the benchmark can run without Kinetics data."""
import numpy as np
import cv2
from pathlib import Path

Path("videos").mkdir(exist_ok=True)
out = cv2.VideoWriter("videos/test.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
for t in range(64):
    frame = np.zeros((240, 320, 3), np.uint8)
    cv2.circle(frame, (40 + t * 4, 120), 30, (0, 200, 255), -1)
    cv2.rectangle(frame, (300 - t * 4, 60), (340 - t * 4, 100), (255, 80, 0), -1)
    frame += np.random.randint(0, 20, frame.shape, np.uint8)
    out.write(frame)
out.release()
print("Wrote videos/test.mp4")
