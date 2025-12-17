from pyldpc import make_ldpc, encode, decode, get_message
import numpy as np
import av, io
from PIL import Image
import cv2

# Create LDPC matrices
n = 512     # codeword length
d_v = 2     # variable node degree
d_c = 4     # check node degree
H, G = make_ldpc(n, d_v, d_c, systematic=True)

def bytes_to_bits(b):
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8))

def bits_to_bytes(bits):
    arr = np.packbits(bits)
    return arr.tobytes()

def encode_h264(frame):
    buffer = io.BytesIO()
    with av.open(buffer, "w", "h264") as container:
        stream = container.add_stream("libx264", rate=12)
        stream.width = frame.width
        stream.height = frame.height
        stream.pix_fmt = "yuv420p"
        packets = stream.encode(frame) + stream.encode(None)
    return b"".join(p.to_bytes() for p in packets)

def ldpc_encode_bits(bits, G):
    k = G.shape[1]       # number of data bits
    blocks = []

    # pad bits so divisible by k
    if len(bits) % k != 0:
        pad = k - (len(bits) % k)
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])

    for i in range(0, len(bits), k):
        data_block = bits[i:i+k]
        codeword = encode(G, data_block)
        blocks.append(codeword)

    return np.concatenate(blocks)

def add_noise(bits, ber=0.01):
    noise = np.random.rand(len(bits)) < ber
    return bits ^ noise.astype(np.uint8)

def ldpc_decode_bits(noisy_bits, H, G):
    n = H.shape[1]
    blocks = []

    for i in range(0, len(noisy_bits), n):
        block = noisy_bits[i:i+n]
        decoded = decode(H, block, maxiter=50)
        msg = get_message(G, decoded)
        blocks.append(msg)

    return np.concatenate(blocks)

def decode_h264(data):
    buffer = io.BytesIO(data)
    with av.open(buffer, 'r', 'h264') as container:
        for frame in container.decode(video=0):
            return frame.to_ndarray(format="rgb24")

def simulate_ldpc_video_pipeline(image):
    # Convert PIL → PyAV frame
    frame = av.VideoFrame.from_ndarray(np.array(image), format="rgb24")

    # Step 1: H.264 encode
    h264_bytes = encode_h264(frame)
    print("H264 size:", len(h264_bytes))

    # Step 2: bytes → bits
    bits = bytes_to_bits(h264_bytes)

    # Step 3: LDPC encode
    ldpc_bits = ldpc_encode_bits(bits, G)

    # Step 5: LDPC decode
    corrected_bits = ldpc_decode_bits(ldpc_bits, H, G)

    # Step 6: bits → bytes
    corrected_bytes = bits_to_bytes(corrected_bits[:len(bits)])  # remove padding

    # Step 7: H.264 decode → image
    restored = decode_h264(corrected_bytes)

    return restored

if __name__ == "__main__" :
    
    cap = cv2.VideoCapture("moving_trian.mp4")

    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame skip to achieve target FPS
    skip_frames = max(1, int(video_fps / 12))
        

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # Skip frames to match target FPS
        if frame_idx % skip_frames != 0:
            continue
        
        # Process and transmit frame
        encoded_bytes = simulate_ldpc_video_pipeline(frame)
        total_bytes += len(encoded_bytes)
        
        print(len(encoded_bytes))
        
        processed_count += 1
        
        # Progress update every 10 frames
        if processed_count % 10 == 0:
            avg_bytes = total_bytes / processed_count
            avg_kbps = (avg_bytes * 8 * 12) / 1000
            print(f"Stats: {processed_count} frames | Avg: {avg_bytes:.0f} bytes/frame ({avg_kbps:.1f} kbps)")
    
        simulate_ldpc_video_pipeline()
