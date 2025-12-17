import serial
import time
import RPi.GPIO as GPIO
import signal
import sys
import atexit

import traceback
import numpy as np
import av
import cv2

codecEN = av.codec.CodecContext.create("h264", "w")

codecEN.pix_fmt = "yuv420p"
codecEN.width = 640
codecEN.height = 480

codecEN.bit_rate = 100 * 1000
codecEN.framerate = 12
codecEN.gop_size = 12
codecEN.thread_count = 4
codecEN.profile = "baseline"

codecEN.options = {
    "preset": "ultrafast",
    "tune": "zerolatency",
    "keyint": "12",        # max I-frame interval
    "min-keyint": "12",    # force exactly 12
    "scenecut": "0",       # prevent auto keyframes
    "bframes": "0"
}

codecEN.open()

def encode_h264(frame):
    packets = codecEN.encode(frame)
    print("Packet len: ", len(packets))
    for p in packets:
        print("PACKET OBJECT:", p)
        print("  is_keyframe:", p.is_keyframe)
        print("  pts:", p.pts, "  dts:", p.dts)
        print("  size:", p.size)
    
    return b"".join(p for p in packets)

# --- UART setup ---
SERIAL_PORT = '/dev/serial0'  # Pi UART TX/RX
BAUD_RATE = 115200

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

# Catch signals
signal.signal(signal.SIGINT, lambda s,f: cleanup(s,f))
signal.signal(signal.SIGTERM, lambda s,f: cleanup(s,f))
signal.signal(signal.SIGHUP, lambda s,f: cleanup(s,f))

# --- Main loop ---
if __name__ == "__main__":
    cap = cv2.VideoCapture("moving_train.mp4")

    if not cap.isOpened():
        print("Fail")       

    target_fps = 12
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0

    skip_frames = max(1, int(video_fps / target_fps))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
                
            # Skip frames to match target FPS
            if frame_idx % skip_frames != 0:
                continue

            print("Frame idx: ",frame_idx/skip_frames)

            frame = cv2.resize(frame, (640, 480))
            
            frame = av.VideoFrame.from_ndarray(np.array(frame), format="rgb24")
            packet = encode_h264(frame)

            ser.write(packet)
            time.sleep(1/target_fps)

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

        cv2.destroyAllWindows()
        cleanup(None,None)
        
    cv2.destroyAllWindows()
    cleanup(None,None)

