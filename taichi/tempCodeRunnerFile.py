import serial
import time
import RPi.GPIO as GPIO
import sys
from imagelib import *
import traceback


# --- UART setup ---
SERIAL_PORT = '/dev/serial0'  # Pi UART TX/RX
BAUD_RATE = 115200

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

if __name__ == "__main__":
    # --- Main loop ---
    print(f"Transmitting 'Hello World!' to STM32 every 3 seconds via {SERIAL_PORT}")
    try:
        while True:
            if time.time() - capture_timer > 60:
                capture_timer = time.time()
                print("Capturing image interval...")
                capture_interval(ser)
                
        
            if ser.in_waiting > 0:  # If there is data in the buffer
                data = ser.readline().decode('ascii').strip()  # Read one line from the serial buffer
                print(f"Received: {data}")

                bytes_left = ser.in_waiting
                print(f"Bytes left in buffer: {bytes_left}")

                if data == "PACKET_PLEASE":
                    buffer = get_webp_image()

                    if buffer is None:
                        print("No image to send.")
                        continue

                    header = "IX".encode('ascii')
                    ender = "END".encode('ascii')

                    print(f"Captured image size: {len(buffer)} bytes")
                    print(f"Data type: {type(buffer)}")
                    print(f"Header type: {type(header)}")
                    print(f"Ender type: {type(ender)}")

                    message = header + buffer + ender
                    
                    ser.write(message)  # send message
            
                    print("Header: " + message[:2].decode('ascii')
                        + " | Ender: " + message[-3:].decode('ascii')
                        + " | Total Length: " + str(len(message)) + " bytes")

                    time.sleep(0.1)  # Small delay to avoid busy waiting
                elif data[:2] == "GG":
                    lat,lon,alt = data[2:].split(',')
                    print(f"Received GPS Data - Latitude: {lat}, Longitude: {lon}, Altitude: {alt}")

            if capture_image:
                ser.write("GG".encode('ascii'))
                capture_image = False
    except Exception as e:
        print(f"Error: {e}\n")
        traceback.print_exc()
    finally:
        cleanup()