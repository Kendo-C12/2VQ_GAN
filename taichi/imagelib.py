from picamera2 import Picamera2
from libcamera import controls
from PIL import Image
import time
import os
import cv2

# INIT
print("Initializing camera...")
picam = Picamera2(0)
config = picam.create_preview_configuration(
    main={"size": (240, 240), "format": "BGR888"}
)
picam.configure(config)

def capture():
    print("Capturing image...")

    while 1:
        try:
            picam.start()
            print("Camera started")
            break
        except Exception as e:
            print(f"Camera start error retrying...")
            time.sleep(1)
    picam.set_controls({'AfMode': controls.AfModeEnum.Continuous})
    time.sleep(3)
    try:
        filename = "temp.webp"
        picam.capture_file(filename)
        with Image.open(filename) as img:
            img = img.resize((240, 240))
            img.save("image.webp", "WEBP", quality=25)
        with open("image.webp", "rb") as f:
            data = f.read()
        picam.stop()

        print("Image captured data type:", type(data))
        return data
    except Exception as e:
        print(f"An error occurred: {e}")
    
def close_camera():
    if picam.is_running():
        picam.stop()
    picam.close()
    print("Camera stopped")

if __name__ == "__main__":
    frame = capture()
    print(f"Captured image size: {len(frame)} bytes")
    print(f"Data type: {type(frame)}")