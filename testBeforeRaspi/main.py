import av
import io
import base64
import zipfile
from PIL import Image
import numpy as np
import os
import cv2

class encoder:
    def transmitImage(self, packet):
        print(len(packet))

    def __init__(self, width=320, height=240, fps=6, kbps=100):
        self.width = width
        self.height = height
        self.fps = fps
        self.bps = kbps * 1000
        self.maxByte = 128
        self.frameCount = 0
        
        # Persistent encoder components
        self.buffer = io.BytesIO()
        self.container = None
        self.stream = None
        self.encoder_initialized = False
        
        self._init_encoder()
        
    def _init_encoder(self):
        """Initialize persistent H.264 encoder"""
        self.buffer = io.BytesIO()
        self.container = av.open(self.buffer, mode='w', format='h264')
        self.stream = self.container.add_stream('libx264', rate=self.fps)
        
        self.stream.width = self.width
        self.stream.height = self.height
        self.stream.pix_fmt = 'yuv420p'
        self.stream.bit_rate = self.bps
        
        # Encoder options optimized for Raspberry Pi and low latency
        self.stream.options = {
            'preset': 'ultrafast',      # Fastest encoding (important for Pi)
            'tune': 'zerolatency',      # Minimize latency
            'crf': '28',                # Quality (23=high, 28=medium, 32=low)
            'g': str(self.fps * 2),     # GOP size (keyframe every 2 seconds)
            'bf': '0',                  # No B-frames (reduces latency)
            'profile': 'baseline',      # Compatible profile
            'level': '3.0',             # H.264 level
        }
        
        self.encoder_initialized = True
        print(f"✓ Encoder initialized: {self.width}x{self.height} @ {self.fps}fps, {self.bps/1000}kbps")
        
    def image_to_frame(self, img):
        """Convert image to AVFrame"""
        if isinstance(img, np.ndarray):
            # OpenCV uses BGR, convert to RGB
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return av.VideoFrame.from_ndarray(img, format='rgb24')
        return img
    
    def encode_frame(self, image):
        """
        Encode a single frame and return ONLY the bytes for this frame.
        Returns bytes object containing H.264 NAL units for this frame.
        """
        if not self.encoder_initialized:
            raise RuntimeError("Encoder not initialized")
        
        frame = self.image_to_frame(image)
        
        # Encode the frame
        packets = self.stream.encode(frame)
        
        # Collect all packet data for this frame
        frame_data = b''
        for packet in packets:
            frame_data += bytes(packet)
        
        return frame_data
    
    def process_frame(self, image):
        """
        Process a single frame and return encoded bytes.
        This is your main function to call.
        """
        encoded_bytes = self.encode_frame(image)
        return encoded_bytes
    
    def process_video(self, input_path="testBeforeRaspi\\moving_train.mp4", target_fps=None):
        """Process video file frame by frame"""
        if target_fps is None:
            target_fps = self.fps
            
        print(f"Processing video: {input_path} @ {target_fps}fps target")
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            print("❌ Error: Cannot open video file.")
            return
        
        video_fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame skip to achieve target FPS
        skip_frames = max(1, int(video_fps / target_fps))
        
        print(f"Video: {video_fps}fps, {total_frames} frames")
        print(f"Processing every {skip_frames} frame(s) to achieve ~{target_fps}fps")
        print(f"Target size per frame: {self.bps/target_fps/8:.0f} bytes")
        print("-" * 60)
        
        frame_idx = 0
        processed_count = 0
        total_bytes = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # Skip frames to match target FPS
            if frame_idx % skip_frames != 0:
                continue
            
            # Process and transmit frame
            encoded_bytes = self.process_frame(frame)
            total_bytes += len(encoded_bytes)
            
            if len(encoded_bytes) > 0:
                self.transmitImage(encoded_bytes)
            
            processed_count += 1
            
            # Progress update every 10 frames
            if processed_count % 10 == 0:
                avg_bytes = total_bytes / processed_count
                avg_kbps = (avg_bytes * 8 * target_fps) / 1000
                print(f"Stats: {processed_count} frames | Avg: {avg_bytes:.0f} bytes/frame ({avg_kbps:.1f} kbps)")
        
        # Flush encoder
        print("\nFlushing encoder...")
        flush_packets = self.stream.encode(None)
        for packet in flush_packets:
            flush_data = bytes(packet)
            if len(flush_data) > 0:
                self.transmitImage(flush_data)
        
        cap.release()
        
        # Final statistics
        print("-" * 60)
        print(f"✓ Complete: {processed_count} frames processed")
        print(f"  Total data: {total_bytes:,} bytes ({total_bytes/1024:.1f} KB)")
        print(f"  Average: {total_bytes/processed_count:.0f} bytes/frame")
        print(f"  Effective bitrate: {(total_bytes*8*target_fps/processed_count)/1000:.1f} kbps")


import av
import io
import numpy as np
import cv2
import serial
import time
import signal
import sys
from collections import deque

class decoder:
    def __init__(self, width=320, height=240, fps=6):
        self.width = width
        self.height = height
        self.fps = fps
        self.maxByte = 128
        self.frameCount = 0
        
        # Decoder components
        self.decoder = None
        self.codec_context = None
        self.packet_buffer = bytearray()
        self.frame_buffer = deque(maxlen=100)  # Store recent frames
        
        # Reception buffer
        self.receive_buffer = bytearray()
        self.expected_frame_size = None
        
        self._init_decoder()
        
    def _init_decoder(self):
        """Initialize H.264 decoder"""
        try:
            self.codec = av.codec.Codec('h264', 'r')
            self.codec_context = self.codec.create()
            
            # Set decoder parameters
            self.codec_context.width = self.width
            self.codec_context.height = self.height
            
            print(f"✓ Decoder initialized: {self.width}x{self.height}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize decoder: {e}")
            return False
    
    def receive_packet_chunk(self, ser, timeout=5):
        """
        Receive a single chunk from UART.
        Returns bytes received or None if timeout.
        """
        start_time = time.time()
        chunk = bytearray()
        
        while time.time() - start_time < timeout:
            if ser.in_waiting > 0:
                # Read available bytes (up to maxByte)
                data = ser.read(min(ser.in_waiting, self.maxByte))
                chunk.extend(data)
                
                # Check if we received a full chunk
                if len(chunk) >= self.maxByte:
                    return bytes(chunk)
                    
            time.sleep(0.001)  # Small delay to prevent CPU spinning
        
        # Return whatever we got (may be partial or empty)
        return bytes(chunk) if len(chunk) > 0 else None
    
    def receive_full_frame(self, ser, frame_size_bytes=None, timeout=10):
        """
        Receive a complete frame from UART.
        If frame_size_bytes is None, reads until timeout with no new data.
        Returns complete frame data as bytes.
        """
        frame_data = bytearray()
        last_receive_time = time.time()
        no_data_timeout = 0.5  # If no data for 0.5s, assume frame complete
        
        print(f"Receiving frame {self.frameCount}...")
        
        while True:
            chunk = self.receive_packet_chunk(ser, timeout=1)
            
            if chunk:
                frame_data.extend(chunk)
                last_receive_time = time.time()
                print(f"  Received chunk: {len(chunk)} bytes (total: {len(frame_data)})")
                
                # If we know the frame size, check if complete
                if frame_size_bytes and len(frame_data) >= frame_size_bytes:
                    break
            else:
                # No data received
                if time.time() - last_receive_time > no_data_timeout:
                    # No data for timeout period, assume frame complete
                    if len(frame_data) > 0:
                        break
                    else:
                        print("  Timeout waiting for frame data")
                        return None
                
                # Check overall timeout
                if time.time() - last_receive_time > timeout:
                    print("  Overall timeout exceeded")
                    return None
        
        print(f"✓ Frame received: {len(frame_data)} bytes")
        return bytes(frame_data)
    
    def decode_frame(self, frame_data):
        """
        Decode H.264 frame data and return as numpy array (BGR format).
        Returns None if decoding fails.
        """
        if not frame_data or len(frame_data) == 0:
            print("❌ No frame data to decode")
            return None
        
        try:
            # Create packet from frame data
            packet = av.Packet(frame_data)
            
            # Decode packet
            frames = self.codec_context.decode(packet)
            
            if frames:
                # Get the first frame
                frame = frames[0]
                
                # Convert to numpy array (RGB format)
                img = frame.to_ndarray(format='rgb24')
                
                # Convert RGB to BGR for OpenCV
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                return img_bgr
            else:
                print("⚠ No frames decoded from packet")
                return None
                
        except Exception as e:
            print(f"❌ Decode error: {e}")
            return None
    
    def process_received_frame(self, frame_data):
        """
        Process received frame data: decode and optionally display/save.
        Returns decoded frame as numpy array.
        """
        self.frameCount += 1
        
        # Decode the frame
        decoded_frame = self.decode_frame(frame_data)
        
        if decoded_frame is not None:
            print(f"✓ Frame {self.frameCount} decoded: {decoded_frame.shape}")
            
            # Store in buffer
            self.frame_buffer.append(decoded_frame)
            
            return decoded_frame
        else:
            print(f"❌ Frame {self.frameCount} decode failed")
            return None
    
    def display_frame(self, frame, window_name="Received Frame", wait_time=1):
        """Display frame using OpenCV"""
        if frame is not None:
            cv2.imshow(window_name, frame)
            cv2.waitKey(wait_time)
    
    def save_frame(self, frame, filepath):
        """Save frame to file"""
        if frame is not None:
            cv2.imwrite(filepath, frame)
            print(f"✓ Frame saved: {filepath}")
    
    def receive_and_decode_continuous(self, ser, display=True, save_path=None):
        """
        Continuously receive and decode frames from UART.
        
        Args:
            ser: Serial port object
            display: Show frames in window
            save_path: Directory to save frames (None = don't save)
        """
        print("=" * 60)
        print("Starting continuous reception...")
        print("Press Ctrl+C to stop")
        print("=" * 60)
        
        if display:
            cv2.namedWindow("Received Frame", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Received Frame", 640, 480)
        
        try:
            while True:
                # Receive complete frame
                frame_data = self.receive_full_frame(ser)
                
                if frame_data:
                    # Decode frame
                    decoded_frame = self.process_received_frame(frame_data)
                    
                    if decoded_frame is not None:
                        # Display frame
                        if display:
                            self.display_frame(decoded_frame, wait_time=1)
                        
                        # Save frame
                        if save_path:
                            filename = f"{save_path}/frame_{self.frameCount:04d}.png"
                            self.save_frame(decoded_frame, filename)
                    
                    # Stats every 10 frames
                    if self.frameCount % 10 == 0:
                        print(f"\n--- Statistics ---")
                        print(f"Frames decoded: {self.frameCount}")
                        print(f"Buffer size: {len(self.frame_buffer)}")
                        print("-" * 60 + "\n")
                # else:
                #     print("⚠ No frame received, waiting...")
                #     time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\n⚠ Reception stopped by user")
        finally:
            if display:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    # --- UART setup ---
    SERIAL_PORT = '/dev/serial0'  # Pi UART TX/RX
    BAUD_RATE = 115200

    try:
        h264 = encoder()
        h264.process_video()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
