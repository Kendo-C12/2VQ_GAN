import serial
import time
import av
import cv2
import traceback
import numpy as np

image = {}
frame_count = 0
packet = ""

apogee = False

def save_img():
    global image
    global frame_count

    filename = f"frame_{frame_count}.webp"

    byte = b"".join(image.values())

    with open(filename, "wb") as f:
        f.write(byte)

def show_rssi(rssi):
    pass

if __name__ == "__main__":
    # Open COM port
    ser = serial.Serial(
        port='COM3',        
        baudrate=115200,
        timeout=1           # seconds
    )

    if ser.in_waiting > 0:
        line = ser.readline().decode('ascii').strip()

        comma_index = line.find(',')

        header = line[:comma_index].decode()  # header
        data = line[comma_index + 1:].decode()  # packet
        
        if header == "FC": # Frame Count
            if frame_count != int(data):
                frame_count = int(data)
                save_img()
        elif header == "IX": # NORMAL IMAGE
            packet = data
        elif header == "AP": # APOGEE IMAGE
            packet = data
            apogee = True
            save_img()
        elif header == "PL": # PACKET LEFT
            image[int(data)] = packet
        elif header == "RS": # RSSI
            show_rssi(data)
        else:
            print(f"Unknown header received: {header}")