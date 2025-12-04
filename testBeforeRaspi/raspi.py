import av
import io
import base64
import zipfile
from PIL import Image
import numpy as np
import os
import cv2

import serial
import time
import RPi.GPIO as GPIO
import signal
import sys

class H264PipelineLastFrame:
    def __init__(self,kbps = 100):
        self.prev_frame = None
        self.bps = kbps * 1000
        self.maxByte = 128
        self.frameCount = 0
        
    def transmitImage(self, packet):
        print("Frame number: " + str(self.frameCount))
        self.frameCount += 1
        i = 0
        while(i < len(packet)):
            j = min(i+self.maxByte,len(packet))
            print("Sending packet: " + str(i) + " to " + str(j) + " maximum packet: " + str(len(packet)))
            ser.write(packet[i:j])
            ser.flush()
            time.sleep(0.1)
            i = j

    def image_to_frame(self, img):
        if isinstance(img, Image.Image):
            arr = np.array(img)
        else:
            arr = img
        return av.VideoFrame.from_ndarray(arr, format='rgb24')

    def encode_last_frame(self, frame_prev, frame_new):
        buffer = io.BytesIO()

        with av.open(buffer, mode='w', format='h264') as container:
            stream = container.add_stream('libx264', rate=30)
            stream.width = frame_new.width
            stream.height = frame_new.height
            stream.pix_fmt = 'yuv420p'

            packets = []

            # Encode previous → generates I-frame
            p1 = stream.encode(frame_prev)
            if p1:
                packets.append(p1)

            # Encode new frame → generates P-frame (delta)
            p2 = stream.encode(frame_new)
            if p2:
                packets.append(p2)

            # Flush
            p_flush = stream.encode(None)
            if p_flush:
                packets.append(p_flush)

            # Return ONLY last packet (last frame)
            last_packet_data = packets[-1].to_bytes()

        return last_packet_data

    def compress_zip(self, data):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("last_frame.h264", data)
        return buf.getvalue()

    def encode_base64(self, b):
        return base64.b64encode(b)

    def process(self, new_image):
        frame_new = self.image_to_frame(new_image)

        if self.prev_frame is None:
            # no previous frame → output full frame (I-frame)
            result = self.encode_base64(
                self.compress_zip(
                    frame_new.to_ndarray().tobytes()
                )
            )
            self.prev_frame = frame_new
            return result

        # Encode only previous + new
        last_frame_bytes = self.encode_last_frame(self.prev_frame, frame_new)

        zipped = self.compress_zip(last_frame_bytes)
        b64 = self.encode_base64(zipped)

        # Update stored previous frame
        self.prev_frame = frame_new

        return b64


    def process_video(self):
        input_path = "moving_train.mp4"
        fps = 6

        print(f"Processing video: {input_path} , fps: {fps}")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print("❌ Error: Cannot open video file.")
            return

        fps_video = int(cap.get(cv2.CAP_PROP_FPS)) + 1
        bpf = self.bps / fps

        skip_frames = max(0, int(fps_video / fps))

        frame_count = 0
        while True:
            frame_count += 1

            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % (skip_frames + 1) != 0:
                continue

            time.sleep(5)
            frame = cv2.resize(frame, (320,240), interpolation=cv2.INTER_LINEAR)
            processed_frame = self.process(frame)
            self.transmitImage(processed_frame)

            if frame_count // (skip_frames + 1) % 10 == 0:
                print(f"Processed {frame_count // (skip_frames + 1)} frames...")

        cap.release()




if __name__ == "__main__":
    # --- UART setup ---
    SERIAL_PORT = '/dev/serial0'  # Pi UART TX/RX
    BAUD_RATE = 460800

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

    signal.signal(signal.SIGINT, cleanup)

    try:
        h264 = H264PipelineLastFrame()
        h264.process_video()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        cleanup(None,None)
