from picamera2 import Picamera2
from libcamera import controls
from PIL import Image
import time
import os
import cv2

def stop_cam():
    picam = Picamera2(0)

    config = picam.create_preview_configuration(
        main={"size": (240, 240), "format": "BGR888"}
    )

    picam.configure(config)
    picam.stop()

    print("Camera stopped.")

if __name__ == "__main__":
    stop_cam()