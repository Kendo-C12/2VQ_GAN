import serial
import time
import RPi.GPIO as GPIO
import sys
from imagelib import *
import traceback


# --- UART setup ---
SERIAL_PORT = '/dev/serial0'  # Pi UART TX/RX
BAUD_RATE = 115200

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