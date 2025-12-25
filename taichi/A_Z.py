import serial
import time
import RPi.GPIO as GPIO
import signal
import sys
import atexit

SERIAL_PORT = '/dev/serial0'
BAUD_RATE = 38400

exit_reason = "UNKNOWN"

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
except serial.SerialException:
    print(f"Cannot open {SERIAL_PORT}")
    sys.exit(1)

# -------- Signal handler (WITH args) --------
def signal_cleanup(sig, frame):
    global exit_reason

    if sig == signal.SIGINT:
        exit_reason = "SIGINT (Ctrl+C)"
    elif sig == signal.SIGTERM:
        exit_reason = "SIGTERM (kill)"
    elif sig == signal.SIGHUP:
        exit_reason = "SIGHUP (UART / TTY / electrical glitch)"
    else:
        exit_reason = f"Signal {sig}"

    print(f"\n[Signal Exit] Reason: {exit_reason}")
    cleanup_common()
    sys.exit(0)

# -------- atexit handler (NO args) --------
def atexit_cleanup():
    print(f"\n[atexit Exit] Program ended normally")
    cleanup_common()

# -------- shared cleanup logic --------
def cleanup_common():
    print("[Cleanup] Closing resources...")
    try:
        GPIO.cleanup()
    except Exception as e:
        print(f"[Cleanup] GPIO cleanup skipped: {e}")

    if ser.is_open:
        ser.close()
        print("[Cleanup] UART closed")

# Register handlers
atexit.register(atexit_cleanup)

signal.signal(signal.SIGINT, signal_cleanup)
signal.signal(signal.SIGTERM, signal_cleanup)
signal.signal(signal.SIGHUP, signal_cleanup)

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
        signal_cleanup("Crtl C",None)