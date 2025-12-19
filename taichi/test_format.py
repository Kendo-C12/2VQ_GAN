import time

import os
import re
import cv2
import serial
from PIL import Image

import io

latest_jpg = "Cute_dog.jpg"

webp_path = "Hi.webp"
img = cv2.imread(latest_jpg)

if img is None:
    print("Failed to load JPG")

small = cv2.resize(img, (240, 240), interpolation=cv2.INTER_AREA)
pil_img = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

pil_img.save(webp_path, "WEBP", quality=25)

t = time.localtime()
c = time.strftime("%H:%M:%S", t)

with open(webp_path, 'rb') as f:
    data = f.read()

