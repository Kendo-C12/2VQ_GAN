import serial
import time
import av
import cv2
import traceback
import numpy as np

image = {}
frame_count = 0
packet = ""

pack_size = 0

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
    while 1:
        try:
            ser = serial.Serial(
                port='COM4',        
                baudrate=115200,
                timeout=1           # seconds
            )
            ser.reset_input_buffer()
            break
        except:
            print("Failed to open COM port. Retrying...")
            time.sleep(1)

    try:
        while 1:
            if ser.in_waiting > 0:
                if pack_size > 0:
                    line = ser.read(pack_size + 5)  # read full packet(3) + header + '\r'(1) + '\n'(1)
                else:
                    line = ser.readline()

                try:
                    header = line[:2].decode("ascii")  # header
                except:
                    header = "XX"

                if pack_size > 0:
                    line = line[2 + 1:-2]  # remove \r\n
                    data = line            # binary packet
                    print(header)
                    print(data)      

                    print(f"Received packet size: {len(data)} bytes | packet size from header: {pack_size} | {(pack_size == len(data))}")

                    pack_size = 0
               
                else:
                    line = line[2 + 1:-1]  # remove \n
                    try:
                        data = line.decode("ascii")  # str
                        print(f"{header},{line.decode('ascii').strip()}")
                    except:
                        header = "XX"
                        print(line)

                if header == "FC": # Frame Count
                    if frame_count != int(data):
                        frame_count = int(data)
                        save_img()
                elif header == "PS":
                    pack_size = int(data)
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
                    pass
                    # print(line.decode("ascii").strip())
                    # print(f"Unknown header received: {header}")
    except Exception as e:
        print(f"Error: {e}\n")
        traceback.print_exc()
    finally:
        ser.close()