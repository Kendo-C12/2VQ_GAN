from picamera2 import Picamera2
from libcamera import controls
import time

import os
import re
import cv2
import serial
from PIL import Image

import io

# INIT

lat = 1
lon = 1
alt = 1

print("Initializing camera...")
picam = Picamera2(0)
config = picam.create_preview_configuration(
    main={"size": (240, 240), "format": "BGR888"}
)
picam.configure(config)

save_directory = f'pictures'

def binary_search_resize(img, format, target_size_b = 4500):
    low, high = 1, 100
    best_quality = 1
    best_size = float('inf')

    while low <= high:
        mid = (low + high) // 2
        buffer = io.BytesIO()
        img.save(buffer, format=format, quality=mid)
        size = len(buffer.getvalue())
        if size <= target_size_b:
            best_quality = mid
            best_size = size
            low = mid + 1
        else:
            high = mid - 1

    return best_quality

def get_next_filename(directory, prefix="image", ext=".jpg"):
    os.makedirs(directory, exist_ok=True)

    files = os.listdir(directory)
    numbers = []

    for f in files:
        match = re.match(rf"{prefix}(\d+){ext}", f)
        if match:
            numbers.append(int(match.group(1)))

    next_num = max(numbers) + 1 if numbers else 1
    return os.path.join(directory, f"{prefix}{next_num}{ext}")

def get_latest_file(directory, ext=".jpg"):
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(ext)
    ]
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def get_webp_image():
    webp_path = get_next_filename(save_directory, "image", ".webp")
    latest_jpg = get_latest_file(save_directory, ".jpg")
    if latest_jpg is None:
        print("No JPG available yet")
        return None
    
    webp_path = get_next_filename(save_directory, "image", ".webp")
    img = cv2.imread(latest_jpg)
    
    if img is None:
        print("Failed to load JPG")
        return None
    
    small = cv2.resize(img, (240, 240), interpolation=cv2.INTER_AREA)
    pil_img = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

    best_quality = binary_search_resize(pil_img, "WEBP")
    print(f"Best WEBP quality found: {best_quality}")
    pil_img.save(webp_path, "WEBP", quality=best_quality)

    t = time.localtime()
    c = time.strftime("%H:%M:%S", t)
    
    with open(webp_path, 'rb') as f:
        data = f.read()

    with open('image_log.txt', 'a') as l:
        l.write(f"{webp_path} bytes sent: {len(data)}\n")
    return data
    

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
            best_quality = binary_search_resize(img, "WEBP")
            print(f"Best WEBP quality found: {best_quality}")
            img.save("image.webp", "WEBP", quality=best_quality)
        with open("image.webp", "rb") as f:
            data = f.read()
        picam.stop()

        print("Image captured data type:", type(data))
        return data
    except Exception as e:
        print(f"An error occurred: {e}")
    
def capture_interval(ser):
    global lat, lon, alt,i
    jpg_path = get_next_filename(save_directory, "image", ".jpg")
    picam.start()
    picam.set_controls({'AfMode': controls.AfModeEnum.Continuous})
    time.sleep(3)
    ser.write("GG".encode('ascii'))
    time.sleep(1)
    lat = None
    lon = None
    alt = None
    last_GG = time.time()
    while (lat == None) and (lon == None) and (alt == None):
        if time.time() - last_GG > 10:
            ser.write("GG".encode('ascii'))
            last_GG = time.time()
        while(ser.in_waiting > 0):
            line_gps = ser.readline()
            print("GET LINE" + str(line_gps))
            try:
                decoded_line = line_gps.decode('ascii').strip()
                gps_line = decoded_line.strip().split(',')
                lat = gps_line[0]
                lon = gps_line[1]
                alt = gps_line[2]
            except:
                continue
    print("image from path:", jpg_path)
    picam.capture_file(jpg_path)
    t = time.localtime()
    c = time.strftime("%H:%M:%S", t)
    with open('image_log.txt', 'a') as l:
        l.write(f"{jpg_path} saved at : {c}\nGPS position at : lat [{lat}], lon [{lon}], alt [{alt}]")
    picam.stop()

def close_camera():
    if picam.is_running():
        picam.stop()
    picam.close()
    print("Camera stopped")

if __name__ == "__main__":
    frame = capture()
    print(f"Captured image size: {len(frame)} bytes")
    print(f"Data type: {type(frame)}")