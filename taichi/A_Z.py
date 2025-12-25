import serial
import time
import RPi.GPIO as GPIO
import signal
import sys
import atexit

# --- UART setup ---
SERIAL_PORT = '/dev/serial0'  # Pi UART TX/RX
BAUD_RATE = 38400

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
except serial.SerialException:
    print(f"Cannot open {SERIAL_PORT}")
    sys.exit(1)

# --- Handle cleanup on Ctrl+C ---
def cleanup(signal_received, frame):
    print("\nInterrupt received! Cleaning up...")
    GPIO.cleanup()
    if ser.is_open:
        ser.close()
    sys.exit(0)

# Catch normal exit
atexit.register(cleanup)

# Catch signals
signal.signal(signal.SIGINT, lambda s,f: cleanup())
signal.signal(signal.SIGTERM, lambda s,f: cleanup())
signal.signal(signal.SIGHUP, lambda s,f: cleanup())

if __name__ == "__main__":
    # --- Main loop ---
    a = time.time()
    print(f"Transmitting 'Hello World!' to STM32 every 3 seconds via {SERIAL_PORT}")
    try:
        while True:
            if a > time.time():
                a = time.time() + 3
                ser.write(b'Hello World!\n')
                print("Sent: Hello World!")
            if ser.in_waiting > 0:  # If there is data in the buffer
                data = ser.readline().decode('ascii').strip()  # Read one line from the serial buffer
                print(f"Received: {data}")

                bytes_left = ser.in_waiting
                print(f"Bytes left in buffer: {bytes_left}")

                if data == "PACKET_PLEASE":
                    buffer = ""
                    for i in range(5000):
                        buffer += chr(ord('A') + (i%26))
                    buffer = buffer.encode('ascii')
                    header = "IX".encode('ascii')
                    ender = "END".encode('ascii')

                    message = header + buffer + ender

                    ser.write(message)  # send message1
                    print(f"Header: {message[0:2].decode('ascii')}")
                    print(f"Ender: {message[-3:].decode('ascii')}")
                    time.sleep(0.1)  # Small delay to avoid busy waiting
    except:
        cleanup(None,None)