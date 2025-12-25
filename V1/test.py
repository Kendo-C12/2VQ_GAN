import serial
import time
import RPi.GPIO as GPIO
import sys
from imagelib import *
import traceback

import signal

import atexit

# --- UART setup ---
SERIAL_PORT = '/dev/serial0'  # Pi UART TX/RX
BAUD_RATE = 38400

lat  = None
lon = None
alt = None

capture_image = True

capture_timer = time.time() - 60

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    ser.reset_input_buffer()
except serial.SerialException:
    print(f"Cannot open {SERIAL_PORT}")
    sys.exit(1)

# --- Handle cleanup on Ctrl+C ---
def cleanup():
    print("\nInterrupt received! Cleaning up...")
    GPIO.cleanup()
    if ser.is_open:
        ser.close()
    sys.exit(0)
    close_camera()

atexit.register(cleanup)

def signal_handler(sig, frame):
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # kill <pid>

if __name__ == "__main__":
    # --- Main loop ---
    print(f"Transmitting 'Hello World!' to STM32 every 3 seconds via {SERIAL_PORT}")
    try:
        while True:
            if time.time() - capture_timer > 60:
                print("LAST CAPTURE TIME:", time.time() - capture_timer)
                capture_timer = time.time()
                print("Capturing image interval...")
                capture_interval(ser)
                print("USED TIME FOR INTERVAL CAPTURE:", time.time() - capture_timer)
                
        
            if ser.in_waiting > 0:  # If there is data in the buffer
                data = ""
                try:
                    data = ser.readline()  # Read one line from the serial buffer
                    data = data.decode('ascii').strip()
                    ser.reset_input_buffer()
                except Exception as e:
                    print(f"Decoding error: {e}")
                    data = "UNKNOW_DATA"
                print(data)

                bytes_left = ser.in_waiting
                print(f"Bytes left in buffer: {bytes_left}")

                if data == "PACK":
                    # buffer = get_webp_image()
                    buffer = capture()

                    if buffer is None:
                        print("No image to send.")
                        continue

                    header = "IX".encode('ascii')
                    ender = "END".encode('ascii')

                    # print(f"Captured image size: {len(buffer)} bytes")
                    # print(f"Data type: {type(buffer)}")
                    # print(f"Header type: {type(header)}")
                    # print(f"Ender type: {type(ender)}")

                    message = header + buffer + ender
                    
                    ser.write(message)  # send message
            
                    # print("Header: " + message[:2].decode('ascii')
                    #     + " | Ender: " + message[-3:].decode('ascii')
                    #     + " | Total Length: " + str(len(message)) + " bytes")

                    time.sleep(0.1)  # Small delay to avoid busy waiting
                elif data[:3] == "APO":
                    # buffer = get_webp_image()
                    buffer = get_apogee_img()

                    if buffer is None:
                        print("No image to send.")
                        continue

                    header = "AP".encode('ascii')
                    ender = "END".encode('ascii')

                    # print(f"Captured image size: {len(buffer)} bytes")
                    # print(f"Data type: {type(buffer)}")
                    # print(f"Header type: {type(header)}")
                    # print(f"Ender type: {type(ender)}")

                    message = header + buffer + ender
                    
                    ser.write(message)  # send message
            
                    # print("Header: " + message[:2].decode('ascii')
                    #     + " | Ender: " + message[-3:].decode('ascii')
                    #     + " | Total Length: " + str(len(message)) + " bytes")

                    time.sleep(0.1)  # Small delay to avoid busy waiting

            if capture_image:
                ser.write("GG".encode('ascii'))
                capture_image = False
    except Exception as e:
        print(f"Error: {e}\n")
        traceback.print_exc()
    finally:
        cleanup()